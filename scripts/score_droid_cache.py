"""Score cached policies with policy-local or shared-DINO thresholds and OMN."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surval.droid import evaluate_policy_caches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-db-root", help="Shared DB root from build_droid_dino_db.py; default: policy encoder")
    parser.add_argument("--dino-omn-k", type=int, default=5, help="Full-split DINO OMN neighbors per observation")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--k-neighbors", type=int, default=50)
    parser.add_argument("--exact-neighbors", action="store_true", help="DINO: cap eligible neighbors at exactly k")
    parser.add_argument("--scale-multiplier", type=float, default=1.0, help="DINO: multiply both local group scales")
    parser.add_argument("--min-neighbors", type=int, default=10)
    parser.add_argument("--temporal-radius", type=int, default=5)
    parser.add_argument("--ta", type=int)
    parser.add_argument("--num-samples", type=int, help="Default: use every cached prediction sample")
    parser.add_argument("--chunk-top-frac", type=float, default=0.5)
    parser.add_argument("--lse-tau", type=float, default=1.0)
    parser.add_argument("--every-step", action="store_true", help="Score every row instead of stride=Ta")
    parser.add_argument(
        "--mode", choices=["epoch", "step"], default="epoch",
        help="Checkpoint index used by the cache filenames. openpi caches are step-indexed.",
    )
    args = parser.parse_args(argv)
    evaluate_policy_caches(
        args.cache_dir, args.output_dir, quantile=args.threshold_quantile,
        k_neighbors=args.k_neighbors, min_neighbors=args.min_neighbors,
        temporal_radius=args.temporal_radius, ta=args.ta, num_samples=args.num_samples,
        chunk_top_frac=args.chunk_top_frac, lse_tau=args.lse_tau, every_step=args.every_step,
        mode=args.mode, dino_db_root=args.dino_db_root, dino_omn_k=args.dino_omn_k,
        exact_neighbors=args.exact_neighbors, scale_multiplier=args.scale_multiplier,
    )


if __name__ == "__main__":
    main()
