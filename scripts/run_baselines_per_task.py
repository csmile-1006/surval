"""
Per-task baseline driver: extract TB scalars + aggregate three proxy metrics
(Valid/Loss, Valid/Off_Manifold_Norm, SequentialValid/ActionL2_mean = valid_mse)
for every dexmimicgen two-arm task under a local_threshold_seqval run.

Output layout under --output_root:

    <output_root>/
      <task>/
        cache/                         # JSON tb-scalar cache (extract output)
        aggregate.json                 # aggregate JSON for the three proxies
        aggregate.csv                  # long-format CSV (one row per metric)
        aggregate_print.csv            # wide-format CSV (one row per hp_key)

The three proxies are all lower-is-better, so they're flipped via --negate_tag
before computing rank metrics. Valid/Off_Manifold_Norm is read from the train
cache (it isn't logged into the seqval TB) — the aggregator transparently
falls back to the train cache for any proxy tag missing from the seqval cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

# Bootstrap surval imports + sibling scripts when invoked from the repo
# without an install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
for p in (_SRC, _HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from surval.tb_aggregate.aggregate import (  # noqa: E402
    finalize_aggregate,
    run_seed_loop,
)
from surval.tb_aggregate.layouts import (  # noqa: E402
    collect_seed_dirs_by_seed,
    iter_seqval_eval_dirs,
    parse_hparam,
)
from surval.tb_aggregate.tb_io import load_cache_file  # noqa: E402


# ---------------------------------------------------------------------------
# Task table
# ---------------------------------------------------------------------------


TASKS = [
    # (short_name, seed_glob_suffix). The TB task name is auto-detected from
    # the train TB's Rollout/Success_Rate/*-mean tag — naming isn't fully
    # systematic (e.g. coffee_humanoid logs as TwoArmCoffee).
    ("drawer_cleanup",       "two_arm_drawer_cleanup"),
    ("three_piece_assembly", "two_arm_three_piece_assembly"),
    ("coffee_humanoid",      "two_arm_coffee_humanoid"),
    ("can_sort_humanoid",    "two_arm_can_sort_humanoid"),
    ("box_cleanup",          "two_arm_box_cleanup"),
    ("threading",            "two_arm_threading"),
    ("pouring_humanoid",     "two_arm_pouring_humanoid"),
    ("lift_tray",            "two_arm_lift_tray"),
]

PROXY_TAGS = [
    # All proxies come from the per-(seed, dataset) HDF5 seqcache files (see
    # surval.tb_aggregate.seqcache_metrics). They are HP-invariant — the same
    # value is reported for every HP under the same (seed, dataset).
    "Cache/Valid/Loss",
    "Cache/Valid/Off_Manifold_Norm",
    "Cache/Valid/MSE",               # E_{s,n,t,a}[(pred - gt)^2]  (variance-included)
    "Cache/Valid/MSE_mean_pred",     # E_{n,t,a}[(mean_s(pred) - gt)^2]
    "Cache/Valid/MSE_t0_only",       # one-step-ahead accuracy of mean prediction
    "Cache/Valid/MSE_tlast",         # last-horizon-step accuracy of mean prediction
    "Cache/Valid/MSE_per_dim_norm",  # ((mean_s(pred) - gt) / std_a)^2 per dim
]

# Force these tags to load from the seqcache_metrics cache (skip the seqval
# TB and train TB caches that may have differently-scoped same-named tags).
FORCE_SEQCACHE_FOR_TAG = list(PROXY_TAGS)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_root_dir", type=str, required=True)
    p.add_argument("--eval_root_dir",  type=str, required=True)
    p.add_argument(
        "--seqcache_root", type=str, required=True,
        help="Root dir for the HDF5 seqcache files "
             "(<seed>/<cache_group_dir>/seqcache_epoch_*.hdf5). Used to compute "
             "the per-(seed, dataset) Cache/Valid/{Loss,Off_Manifold_Norm,MSE} tags.",
    )
    p.add_argument("--output_root",    type=str, required=True)
    p.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help=f"Subset of task short-names (default: all). Choices: "
             f"{[t[0] for t in TASKS]}",
    )
    p.add_argument("--extract_workers",  type=int, default=8)
    p.add_argument("--aggregate_workers", type=int, default=8)
    p.add_argument(
        "--skip_existing", action="store_true",
        help="Skip extraction if the cache JSON already exists. "
             "Useful for incremental re-runs.",
    )
    p.add_argument("--num_boot", type=int, default=2000)
    p.add_argument(
        "--report_loo", action="store_true",
        help="Include leave-one-seed-out diagnostics in aggregate.json.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the planned commands without running.",
    )
    return p


# ---------------------------------------------------------------------------
# Extract / aggregate as in-process calls (no subprocess)
# ---------------------------------------------------------------------------


def _run_extract(train_root, eval_root, seed_prefix, cache_dir, num_workers, skip_existing):
    from extract_tb_scalars import main as extract_main
    argv = [
        "--train_root_dir", train_root,
        "--eval_root_dir",  eval_root,
        "--seed_glob_prefix", seed_prefix,
        "--cache_dir", cache_dir,
        "--num-workers", str(num_workers),
    ]
    if skip_existing:
        argv.append("--skip_existing")
    extract_main(argv)


def _run_extract_seqcache(seqcache_root, seed_prefix, cache_dir, num_workers, skip_existing):
    from extract_seqcache_metrics import main as extract_seqcache_main
    argv = [
        "--seqcache_root", seqcache_root,
        "--seed_glob_prefix", seed_prefix,
        "--cache_dir", cache_dir,
        "--num-workers", str(num_workers),
    ]
    if skip_existing:
        argv.append("--skip_existing")
    extract_seqcache_main(argv)


def _detect_train_success_tag(cache_dir):
    """Find a Rollout/Success_Rate/<task>-mean tag from any seed's train cache."""
    train_dir = os.path.join(cache_dir, "train")
    if not os.path.isdir(train_dir):
        return None
    for name in sorted(os.listdir(train_dir)):
        if not name.endswith(".json"):
            continue
        payload = load_cache_file(os.path.join(train_dir, name))
        candidates = [
            t for t in payload.get("tags", {})
            if t.startswith("Rollout/Success_Rate/") and t.endswith("-mean")
        ]
        if candidates:
            # Prefer a single unique tag; warn if multiple.
            if len(candidates) > 1:
                print(f"  [driver] multiple success tags in {name}: {candidates} "
                      f"-- using {candidates[0]!r}")
            return candidates[0]
    return None


def _run_aggregate(
    train_root, eval_root, seed_prefix, cache_dir,
    train_tag, seqval_tags, negate_tags,
    force_train_tags, force_seqcache_tags,
    out_json, out_csv, out_print_csv,
    num_workers, num_boot, report_loo,
):
    args = SimpleNamespace(
        train_root_dir=train_root,
        eval_root_dir=eval_root,
        cache_dir=os.path.abspath(cache_dir),
        seed_glob_prefix=seed_prefix,
        train_tag=train_tag,
        seqval_tag=list(seqval_tags),
        train_success_count_tag=None,
        train_tag_is_rate=True,
        k_list=[1, 2, 3, 4, 5],
        tie_tol=0.0,
        eps=1e-8,
        nan_policy="omit",
        min_steps=2,
        negate_seqval_tag=False,
        negate_tag=list(negate_tags),
        force_train_for_tag=list(force_train_tags or []),
        force_seqcache_for_tag=list(force_seqcache_tags or []),
        out_json=out_json,
        out_csv=out_csv,
        out_print_csv=out_print_csv,
        num_boot=int(num_boot),
        ci_level=0.95,
        bootstrap_seed=0,
        use_posterior_success=False,
        num_mc=4000,
        n_rollouts=200,
        beta_prior_a=1.0,
        beta_prior_b=1.0,
        disable_kendall=False,
        enable_delta_kendall=False,
        report_loo=bool(report_loo),
        print_aggregate=False,
        print_all_metrics=False,
        print_sort_by="mmrv",
        num_workers=int(num_workers),
    )

    train_seed_dirs = collect_seed_dirs_by_seed(train_root, seed_prefix)
    eval_seed_dirs  = collect_seed_dirs_by_seed(eval_root,  seed_prefix)
    common_seeds = sorted(set(train_seed_dirs.keys()) & set(eval_seed_dirs.keys()))

    if not common_seeds:
        print(f"  [aggregate] no common seeds for prefix {seed_prefix!r} — skipping.")
        return

    per_hp_per_seed, skipped = run_seed_loop(
        args, common_seeds, train_seed_dirs, eval_seed_dirs,
        cache_dir=args.cache_dir, parse_fn=parse_hparam, iter_fn=iter_seqval_eval_dirs,
    )
    finalize_aggregate(
        args, per_hp_per_seed, skipped,
        train_seed_dirs, eval_seed_dirs, common_seeds,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    if args.tasks:
        selected = [t for t in TASKS if t[0] in set(args.tasks)]
        missing = set(args.tasks) - {t[0] for t in TASKS}
        if missing:
            raise SystemExit(f"unknown tasks: {sorted(missing)}")
    else:
        selected = list(TASKS)

    print(f"[driver] {len(selected)} task(s): {[t[0] for t in selected]}")
    print(f"[driver] output_root = {output_root}")
    print(f"[driver] proxy tags  = {PROXY_TAGS}")

    summary_rows = []
    for short, seed_suffix in selected:
        task_dir = os.path.join(output_root, short)
        cache_dir = os.path.join(task_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        seed_prefix = f"flow_policy_image_ds_{seed_suffix}_D0_n_demo_500"

        print("\n" + "=" * 80)
        print(f"[driver] task = {short}")
        print(f"        seed_glob = {seed_prefix}")
        print(f"        cache_dir = {cache_dir}")

        if args.dry_run:
            continue

        _run_extract(
            train_root=args.train_root_dir,
            eval_root=args.eval_root_dir,
            seed_prefix=seed_prefix,
            cache_dir=cache_dir,
            num_workers=args.extract_workers,
            skip_existing=args.skip_existing,
        )

        _run_extract_seqcache(
            seqcache_root=args.seqcache_root,
            seed_prefix=seed_prefix,
            cache_dir=cache_dir,
            num_workers=args.extract_workers,
            skip_existing=args.skip_existing,
        )

        train_tag = _detect_train_success_tag(cache_dir)
        if train_tag is None:
            print(f"  [driver] no Rollout/Success_Rate/*-mean tag found for {short} -- skipping aggregate.")
            continue
        print(f"  [driver] auto-detected train_tag = {train_tag}")

        out_json = os.path.join(task_dir, "aggregate.json")
        out_csv  = os.path.join(task_dir, "aggregate.csv")
        out_print_csv = os.path.join(task_dir, "aggregate_print.csv")

        # All three proxies are lower-is-better; negate all of them so
        # rank-based metrics treat the success-rate side as the ground truth.
        _run_aggregate(
            train_root=args.train_root_dir,
            eval_root=args.eval_root_dir,
            seed_prefix=seed_prefix,
            cache_dir=cache_dir,
            train_tag=train_tag,
            seqval_tags=PROXY_TAGS,
            negate_tags=PROXY_TAGS,
            force_train_tags=[],
            force_seqcache_tags=FORCE_SEQCACHE_FOR_TAG,
            out_json=out_json,
            out_csv=out_csv,
            out_print_csv=out_print_csv,
            num_workers=args.aggregate_workers,
            num_boot=args.num_boot,
            report_loo=args.report_loo,
        )

        summary_rows.append({
            "task": short,
            "train_tag": train_tag,
            "out_json": out_json,
            "out_csv": out_csv,
            "out_print_csv": out_print_csv,
        })

    print("\n" + "=" * 80)
    print(f"[driver] done. {len(summary_rows)} task(s) processed.")
    for r in summary_rows:
        print(f"  - {r['task']:24s} -> {r['out_print_csv']}")


if __name__ == "__main__":
    main()
