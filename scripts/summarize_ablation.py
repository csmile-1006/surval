"""
Leave-one-out ablation summary.

Reads the per-task `our_method.csv` files produced by `run_our_method_per_task.py`
and pivots the rows by the `ab` ablation tag (baseline, nogrp, fixthr,
fixthr_intra, tmean, globaldb, q05, q99).

For each (task, metric) it emits the baseline value, each ablation's value, and
the marginal Δ columns. Sign convention: positive Δ ⇒ baseline is BIGGER than
the variant on this metric. For higher-is-better metrics (spearman/hit@k),
positive Δ ⇒ baseline wins; for lower-is-better metrics (mmrv/nregret),
negative Δ ⇒ baseline wins.

    LOO comparisons (baseline minus single-component ablation):
        Δ_grouping             = baseline − nogrp
        Δ_state_cond_vs_inter  = baseline − fixthr
        Δ_state_cond_vs_intra  = baseline − fixthr_intra
        Δ_product              = baseline − tmean

    Inter vs intra direct comparison (independent of baseline):
        Δ_inter_vs_intra       = fixthr  − fixthr_intra   (positive ⇒ inter higher)

    Sanity baselines (baseline minus "broken" variant):
        Δ_vs_globaldb          = baseline − globaldb      (value of per-state info)
        Δ_vs_q05               = baseline − q05           (cost of too-lenient quantile)
        Δ_vs_q99               = baseline − q99           (cost of too-strict  quantile)

Output: `<output_root>/ablation_summary.csv` and `<output_root>/ablation_summary.md`.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


VARIANTS = (
    "baseline", "nogrp", "fixthr", "fixthr_intra", "tmean",
    "globaldb", "q05", "q99",
)
DELTAS = (
    ("delta_grouping", "baseline", "nogrp"),
    ("delta_state_cond_vs_inter", "baseline", "fixthr"),
    ("delta_state_cond_vs_intra", "baseline", "fixthr_intra"),
    ("delta_product", "baseline", "tmean"),
    ("delta_inter_vs_intra", "fixthr", "fixthr_intra"),
    ("delta_vs_globaldb", "baseline", "globaldb"),
    ("delta_vs_q05", "baseline", "q05"),
    ("delta_vs_q99", "baseline", "q99"),
)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _coerce_float(x):
    if x in (None, "", "nan", "NaN"):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _collect(task_csv: str, metric: str) -> dict[str, float]:
    """Return {variant: mean_value} for the requested metric.

    If a task has multiple HP-tuples per variant (e.g. legacy sweep entries
    landed alongside the ablation runs), the median of the means is used so an
    outlier HP doesn't dominate. This is defensive — the ablation script fixes
    HPs, so in practice there is one row per variant.
    """
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in _read_csv(task_csv):
        ab = (row.get("ab") or "").strip()
        if ab not in VARIANTS:
            continue
        if row.get("metric") != metric:
            continue
        by_variant[ab].append(_coerce_float(row.get("mean")))

    out: dict[str, float] = {}
    for v in VARIANTS:
        vals = [x for x in by_variant.get(v, []) if x == x]  # drop NaN
        if not vals:
            out[v] = float("nan")
        elif len(vals) == 1:
            out[v] = vals[0]
        else:
            # Median to be robust to stray rows.
            vals = sorted(vals)
            mid = len(vals) // 2
            out[v] = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    return out


def _fmt(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.4f}"


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_root", type=str, required=True,
        help="Directory containing per-task subdirs with our_method.csv "
             "(same as run_our_method_per_task.py --output_root).",
    )
    p.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Subset of task short-names to include (default: all subdirs that "
             "have our_method.csv).",
    )
    p.add_argument(
        "--metric", type=str, default="mmrv",
        help="Which metric column to compare (default: mmrv).",
    )
    p.add_argument(
        "--out_csv", type=str, default=None,
        help="Output CSV path (default: <output_root>/ablation_summary.csv).",
    )
    p.add_argument(
        "--out_md", type=str, default=None,
        help="Output Markdown path (default: <output_root>/ablation_summary.md).",
    )
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    out_csv = args.out_csv or os.path.join(output_root, "ablation_summary.csv")
    out_md = args.out_md or os.path.join(output_root, "ablation_summary.md")

    if args.tasks:
        candidates = sorted(set(args.tasks))
    else:
        candidates = sorted(
            name for name in os.listdir(output_root)
            if os.path.isdir(os.path.join(output_root, name))
        )

    rows = []
    for task in candidates:
        task_csv = os.path.join(output_root, task, "our_method.csv")
        if not os.path.isfile(task_csv):
            print(f"[summarize_ablation] skip {task}: no {task_csv}", file=sys.stderr)
            continue
        variant_vals = _collect(task_csv, args.metric)
        if all(v != v for v in variant_vals.values()):  # all NaN
            print(f"[summarize_ablation] skip {task}: no ablation rows for metric={args.metric!r}",
                  file=sys.stderr)
            continue
        row = {"task": task}
        for v in VARIANTS:
            row[v] = variant_vals[v]
        for col, ref, abl in DELTAS:
            row[col] = variant_vals[ref] - variant_vals[abl]
        rows.append(row)

    if not rows:
        print("[summarize_ablation] no rows to write.", file=sys.stderr)
        return 1

    fieldnames = ["task", *VARIANTS, *[c for c, _, _ in DELTAS]]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (_fmt(v) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[summarize_ablation] wrote {out_csv}")

    lines = [
        f"# Leave-one-out ablation (metric={args.metric})",
        "",
        "Columns: per-variant metric value. Δ = baseline − variant; positive Δ ⇒ that stage helped.",
        "",
        "| task | " + " | ".join(VARIANTS) + " | " + " | ".join(c for c, _, _ in DELTAS) + " |",
        "|" + "---|" * (1 + len(VARIANTS) + len(DELTAS)),
    ]
    for r in rows:
        cells = [r["task"]] + [_fmt(r[v]) for v in VARIANTS] + [_fmt(r[c]) for c, _, _ in DELTAS]
        lines.append("| " + " | ".join(cells) + " |")

    md_body = "\n".join(lines) + "\n"
    with open(out_md, "w") as f:
        f.write(md_body)
    print(f"[summarize_ablation] wrote {out_md}")
    print()
    print(md_body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
