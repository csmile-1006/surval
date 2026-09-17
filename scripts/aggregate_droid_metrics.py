"""Aggregate SURVAL/baseline correlation and selection against custom outcomes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surval.tb_aggregate.droid import aggregate_droid_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--outcomes-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args(argv)
    result = aggregate_droid_metrics(args.metrics_csv, args.outcomes_csv, args.output_json, k_list=args.top_k)
    print(f"Evaluated {len(result['per_group'])} run/proxy pairs; skipped {len(result['skipped'])}. "
          f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
