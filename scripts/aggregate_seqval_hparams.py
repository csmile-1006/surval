"""
Aggregate proxy metrics across seeds for local_threshold seqval runs, reading
the JSON scalar cache produced by `scripts/extract_tb_scalars.py`.

Metric set per (HP, seqval_tag) pair (computed against train_tag as A):
  - spearman, delta_spearman
  - kendall_tau (toggle with --disable_kendall)
  - hit@k for each k in --k_list
  - nregret, rank_pct
  - mmrv  (Mean Maximum Rank Violation, Li et al. 2024 — lower is better)

Multiple --seqval_tag values can be passed; the per-tag metrics share train_tag
and are emitted side-by-side in the output (one hp_key per tag).

Tag tips for the local_threshold layout:
  --train_tag                 Rollout/Success_Rate/<TaskName>-mean
                              (or Valid/Loss / Valid/Off_Manifold_Norm if not
                              using the rate; pass --no-train_tag_is_rate then)
  --seqval_tag                SequentialValid/PrefixSurvival_Score
                              SequentialValid/ActionL2_mean       (= valid MSE)
                              SequentialValid/valid_loss
                              Valid/Loss
                              Valid/Off_Manifold_Norm
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python scripts/aggregate_seqval_hparams.py` from the repo root without install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.tb_aggregate.aggregate import (  # noqa: E402
    finalize_aggregate,
    run_seed_loop,
)
from surval.tb_aggregate.layouts import (  # noqa: E402
    collect_seed_dirs_by_seed,
    iter_seqval_eval_dirs,
    parse_hparam,
)


def _bool_arg(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y", "on"}


def build_argparser():
    p = argparse.ArgumentParser(
        description=(
            "Aggregate cached TB proxy metrics across seeds for local_threshold "
            "seqval HP directories. Reads JSON cache produced by extract_tb_scalars.py."
        )
    )
    p.add_argument("--train_root_dir", type=str, required=True)
    p.add_argument("--eval_root_dir", type=str, required=True)
    p.add_argument(
        "--cache_dir", type=str, required=True,
        help="Directory containing the JSON cache (output of extract_tb_scalars.py).",
    )
    p.add_argument("--seed_glob_prefix", type=str, default="")
    p.add_argument(
        "--train_tag", type=str, required=True,
        help="Scalar tag from train TB used as the ground-truth signal A (e.g. "
             "Rollout/Success_Rate/<Task>-mean, or Valid/Loss).",
    )
    p.add_argument(
        "--seqval_tag", type=str, nargs="+", required=True,
        help="One or more scalar tags from seqval TB used as proxy signal B. "
             "Each tag is aggregated independently (added into hp_key as tag=...).",
    )
    p.add_argument(
        "--train_success_count_tag", type=str, default=None,
        help="Optional tag with success counts k (0..n_rollouts). Used by "
             "--use_posterior_success.",
    )
    p.add_argument(
        "--train_tag_is_rate", type=_bool_arg, default=True,
        help="If True (default) train_tag is treated as a success rate and rows "
             "with values outside [0, 1] are skipped. Set False when using Loss / OMN.",
    )
    p.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--tie_tol", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--nan_policy", type=str, choices=["raise", "omit"], default="omit")
    p.add_argument("--min_steps", type=int, default=2)
    p.add_argument(
        "--negate_seqval_tag", action="store_true",
        help="Negate ALL seqval_tag values before computing metrics. Use "
             "--negate_tag to negate only a subset (useful when mixing higher-"
             "is-better proxies like PrefixSurvival_Score with lower-is-better "
             "proxies like Loss / ActionL2_mean / Off_Manifold_Norm).",
    )
    p.add_argument(
        "--negate_tag", type=str, nargs="*", default=None,
        help="Specific seqval tags to negate (e.g. SequentialValid/ActionL2_mean "
             "Valid/Loss Valid/Off_Manifold_Norm). Applied in addition to "
             "--negate_seqval_tag.",
    )
    p.add_argument(
        "--force_train_for_tag", type=str, nargs="*", default=None,
        help="Read these tags from the train cache instead of the seqval cache. "
             "Use this for tags whose seqval-side meaning differs from the "
             "training-side meaning — e.g. Valid/Loss in the seqval TB is the "
             "loss on the cache's own val data and varies by cache_group_dir, "
             "while Valid/Loss in the train TB is the standard validation loss "
             "trajectory.",
    )
    p.add_argument(
        "--force_seqcache_for_tag", type=str, nargs="*", default=None,
        help="Read these tags from the seqcache_metrics cache (HDF5-derived "
             "per-(seed, dataset) scalars). Useful for the Cache/Valid/* "
             "baselines produced by scripts/extract_seqcache_metrics.py.",
    )
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_print_csv", type=str, default=None)
    p.add_argument("--num_boot", type=int, default=5000)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--use_posterior_success", action="store_true")
    p.add_argument("--num_mc", type=int, default=4000)
    p.add_argument("--n_rollouts", type=int, default=200)
    p.add_argument("--beta_prior_a", type=float, default=1.0)
    p.add_argument("--beta_prior_b", type=float, default=1.0)
    p.add_argument("--disable_kendall", action="store_true")
    p.add_argument("--enable_delta_kendall", action="store_true")
    p.add_argument("--report_loo", action="store_true")
    p.add_argument("--print_aggregate", action="store_true")
    p.add_argument("--print_all_metrics", action="store_true")
    p.add_argument(
        "--print_sort_by", type=str, default="spearman",
        help="Metric used to sort the printed table. For lower-is-better "
             "metrics (mmrv, nregret, rank_pct) sort flips to ascending.",
    )
    p.add_argument(
        "--num-workers", type=int, default=1,
        help="Process one seed per worker in parallel (default: 1, sequential).",
    )
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    args.cache_dir = os.path.abspath(args.cache_dir)
    if not os.path.isdir(args.cache_dir):
        raise SystemExit(f"--cache_dir not found: {args.cache_dir}. "
                         f"Run extract_tb_scalars.py first.")

    train_seed_dirs = collect_seed_dirs_by_seed(args.train_root_dir, args.seed_glob_prefix)
    eval_seed_dirs = collect_seed_dirs_by_seed(args.eval_root_dir, args.seed_glob_prefix)
    common_seeds = sorted(set(train_seed_dirs.keys()) & set(eval_seed_dirs.keys()))

    if not common_seeds:
        raise RuntimeError(
            f"No common seed dirs between {args.train_root_dir} and {args.eval_root_dir}."
        )

    per_hp_per_seed, skipped = run_seed_loop(
        args,
        common_seeds,
        train_seed_dirs,
        eval_seed_dirs,
        cache_dir=args.cache_dir,
        parse_fn=parse_hparam,
        iter_fn=iter_seqval_eval_dirs,
    )
    finalize_aggregate(
        args,
        per_hp_per_seed,
        skipped,
        train_seed_dirs,
        eval_seed_dirs,
        common_seeds,
    )


if __name__ == "__main__":
    main()
