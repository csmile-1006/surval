"""
Per-task aggregation for our method's seqval proxy
(SequentialValid/PrefixSurvival_Score).

Unlike the baselines in `run_baselines_per_task.py`, this proxy IS
HP-dependent: each `tq/lsetau/ctf/share/skip/dist` combo gives a different
ranking signal. So the aggregation keeps the HP dimension — one row per
(HP × cache_group_dir × seqval_tag).

Reuses the JSON cache produced by extract_tb_scalars.py (writes nothing new
to disk on the cache side). Output files per task:

    <output_root>/<task>/our_method.json
    <output_root>/<task>/our_method.csv
    <output_root>/<task>/our_method_print.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

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


TASKS = [
    ("drawer_cleanup",       "two_arm_drawer_cleanup"),
    ("three_piece_assembly", "two_arm_three_piece_assembly"),
    ("coffee_humanoid",      "two_arm_coffee_humanoid"),
    ("can_sort_humanoid",    "two_arm_can_sort_humanoid"),
    ("box_cleanup",          "two_arm_box_cleanup"),
    ("threading",            "two_arm_threading"),
    ("pouring_humanoid",     "two_arm_pouring_humanoid"),
    ("lift_tray",            "two_arm_lift_tray"),
]

# Our-method proxy. Higher is better — do NOT negate.
PROXY_TAGS = ["SequentialValid/PrefixSurvival_Score"]


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_root_dir", type=str, required=True)
    p.add_argument("--eval_root_dir",  type=str, required=True)
    p.add_argument(
        "--output_root", type=str, required=True,
        help="Same directory as run_baselines_per_task.py output_root — "
             "this script writes our_method.* files alongside aggregate.*.",
    )
    p.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help=f"Subset of task short-names (default: all). Choices: "
             f"{[t[0] for t in TASKS]}",
    )
    p.add_argument(
        "--proxy_tags", type=str, nargs="+", default=PROXY_TAGS,
        help="Override the proxy tags (default: SequentialValid/PrefixSurvival_Score).",
    )
    p.add_argument("--aggregate_workers", type=int, default=8)
    p.add_argument("--num_boot", type=int, default=2000)
    p.add_argument("--report_loo", action="store_true")
    return p


def _detect_train_success_tag(cache_dir):
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
            return candidates[0]
    return None


def _run_aggregate(
    train_root, eval_root, seed_prefix, cache_dir,
    train_tag, seqval_tags,
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
        negate_tag=[],
        force_train_for_tag=[],
        force_seqcache_for_tag=[],
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
        print(f"  [our_method] no common seeds for prefix {seed_prefix!r} — skip.")
        return

    per_hp_per_seed, skipped = run_seed_loop(
        args, common_seeds, train_seed_dirs, eval_seed_dirs,
        cache_dir=args.cache_dir, parse_fn=parse_hparam, iter_fn=iter_seqval_eval_dirs,
    )
    finalize_aggregate(
        args, per_hp_per_seed, skipped,
        train_seed_dirs, eval_seed_dirs, common_seeds,
    )


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    output_root = os.path.abspath(args.output_root)

    if args.tasks:
        selected = [t for t in TASKS if t[0] in set(args.tasks)]
    else:
        selected = list(TASKS)

    print(f"[our_method] {len(selected)} task(s): {[t[0] for t in selected]}")
    print(f"[our_method] proxy tags = {args.proxy_tags}")

    for short, seed_suffix in selected:
        task_dir = os.path.join(output_root, short)
        cache_dir = os.path.join(task_dir, "cache")
        if not os.path.isdir(cache_dir):
            print(f"\n[our_method] {short}: cache dir missing ({cache_dir}). "
                  f"Run run_baselines_per_task.py first to populate the JSON cache.")
            continue
        seed_prefix = f"flow_policy_image_ds_{seed_suffix}_D0_n_demo_500"

        train_tag = _detect_train_success_tag(cache_dir)
        if train_tag is None:
            print(f"\n[our_method] {short}: no success_rate tag found in train cache "
                  f"— skip aggregate.")
            continue

        out_json = os.path.join(task_dir, "our_method.json")
        out_csv  = os.path.join(task_dir, "our_method.csv")
        out_print_csv = os.path.join(task_dir, "our_method_print.csv")

        print("\n" + "=" * 80)
        print(f"[our_method] task = {short}")
        print(f"             train_tag = {train_tag}")
        print(f"             out_json  = {out_json}")

        _run_aggregate(
            train_root=args.train_root_dir,
            eval_root=args.eval_root_dir,
            seed_prefix=seed_prefix,
            cache_dir=cache_dir,
            train_tag=train_tag,
            seqval_tags=args.proxy_tags,
            out_json=out_json,
            out_csv=out_csv,
            out_print_csv=out_print_csv,
            num_workers=args.aggregate_workers,
            num_boot=args.num_boot,
            report_loo=args.report_loo,
        )


if __name__ == "__main__":
    main()
