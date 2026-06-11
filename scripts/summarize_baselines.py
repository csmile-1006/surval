"""
Collapse per-task aggregate.json files into a per-(task, dataset, proxy)
baseline summary. Reads `aggregate.json` produced by run_baselines_per_task.py
and writes:

    <output_root>/baselines_summary.json
    <output_root>/baselines_summary.csv         (long format)
    <output_root>/baselines_summary_wide.csv    (wide format, one row per task+dataset)

For the three baseline proxies (Valid/Loss, Valid/Off_Manifold_Norm,
SequentialValid/ActionL2_mean) the per-checkpoint series is the same for all
HPs within a (task, dataset, seed), so every (task, dataset, proxy) cell is
represented once — using the first matching hp_key.

The bootstrap 95% CI and hit@1..hit@5 already live in `aggregate.json`; this
script just pivots them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


BASELINE_TAGS = [
    "Cache/Valid/Loss",
    "Cache/Valid/Off_Manifold_Norm",
    "Cache/Valid/MSE",
]

PRIMARY_METRICS = [
    "spearman", "kendall_tau", "delta_spearman",
    "hit@1", "hit@2", "hit@3", "hit@4", "hit@5",
    "nregret", "rank_pct", "mmrv",
]


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Subset of task short-names. Default: every subdir of output_root "
             "containing aggregate.json.",
    )
    return p


def _stat(metric_entry):
    """Return a dict of the fields we care about for a metric stat."""
    return {
        "mean":      metric_entry.get("mean"),
        "median":    metric_entry.get("median"),
        "std":       metric_entry.get("std"),
        "ci95_low":  metric_entry.get("ci95_low"),
        "ci95_high": metric_entry.get("ci95_high"),
        "n":         metric_entry.get("n"),
    }


def collapse_task(aggregate_path, task_name):
    """
    Return list of (task, dataset, proxy_tag, proxy_source, metrics_dict).
    """
    with open(aggregate_path) as f:
        agg = json.load(f)["aggregated"]

    # Bucket entries by (dataset, tag). For baseline proxies they're all
    # identical across HPs, so we just grab the first.
    seen = {}
    for hp_key, info in agg.items():
        hparams = info["hparams"]
        tag = hparams.get("seqval_tag")
        if tag not in BASELINE_TAGS:
            continue
        dataset = hparams.get("dataset", "?")
        key = (dataset, tag)
        if key in seen:
            continue
        seen[key] = (info, hp_key)

    rows = []
    for (dataset, tag), (info, hp_key) in sorted(seen.items()):
        m = info["metrics"]
        rows.append({
            "task": task_name,
            "dataset": dataset,
            "proxy_tag": tag,
            "proxy_source": info["hparams"].get("proxy_source", "?"),
            "num_seeds": info.get("num_seeds"),
            "seeds": info.get("seeds"),
            "metrics": {name: _stat(m[name]) for name in m if name in PRIMARY_METRICS},
            "hp_representative": hp_key,
        })
    return rows


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    output_root = os.path.abspath(args.output_root)

    if args.tasks:
        tasks = list(args.tasks)
    else:
        tasks = [
            t for t in sorted(os.listdir(output_root))
            if os.path.isfile(os.path.join(output_root, t, "aggregate.json"))
        ]

    all_rows = []
    for t in tasks:
        path = os.path.join(output_root, t, "aggregate.json")
        if not os.path.isfile(path):
            print(f"[summary] skip {t}: aggregate.json missing")
            continue
        rows = collapse_task(path, t)
        all_rows.extend(rows)
        print(f"[summary] {t}: {len(rows)} (dataset, proxy) cells")

    # ------- baselines_summary.json -------
    json_out = os.path.join(output_root, "baselines_summary.json")
    with open(json_out, "w") as f:
        json.dump(
            {"tasks": tasks, "baseline_tags": BASELINE_TAGS, "rows": all_rows},
            f, indent=2,
        )
    print(f"[summary] wrote {json_out}")

    # ------- baselines_summary.csv (long: one row per metric) -------
    long_path = os.path.join(output_root, "baselines_summary.csv")
    with open(long_path, "w", newline="") as f:
        fieldnames = [
            "task", "dataset", "proxy_tag", "proxy_source", "num_seeds",
            "metric", "mean", "median", "std", "ci95_low", "ci95_high", "n",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            for metric, stat in r["metrics"].items():
                writer.writerow({
                    "task": r["task"], "dataset": r["dataset"],
                    "proxy_tag": r["proxy_tag"], "proxy_source": r["proxy_source"],
                    "num_seeds": r["num_seeds"], "metric": metric,
                    **stat,
                })
    print(f"[summary] wrote {long_path}")

    # ------- baselines_summary_wide.csv -------
    # One row per (task, dataset). Columns: each (proxy, metric) gets mean +
    # ci95_low + ci95_high. Easier to copy into a paper table.
    wide_path = os.path.join(output_root, "baselines_summary_wide.csv")
    by_td = defaultdict(dict)  # (task, dataset) -> {proxy: row}
    for r in all_rows:
        by_td[(r["task"], r["dataset"])][r["proxy_tag"]] = r

    wide_rows = []
    metric_order = [m for m in PRIMARY_METRICS]
    for (task, dataset), per_proxy in sorted(by_td.items()):
        row = {"task": task, "dataset": dataset}
        # num_seeds taken from any proxy (should all match).
        any_proxy = next(iter(per_proxy.values()))
        row["num_seeds"] = any_proxy["num_seeds"]
        for proxy in BASELINE_TAGS:
            r = per_proxy.get(proxy)
            short = (
                "valid_loss" if proxy == "Cache/Valid/Loss"
                else "valid_omn" if proxy == "Cache/Valid/Off_Manifold_Norm"
                else "valid_mse" if proxy == "Cache/Valid/MSE"
                else proxy.replace("/", "_")
            )
            for metric in metric_order:
                if r is None:
                    row[f"{short}__{metric}__mean"] = None
                    row[f"{short}__{metric}__ci_lo"] = None
                    row[f"{short}__{metric}__ci_hi"] = None
                else:
                    stat = r["metrics"].get(metric, {})
                    row[f"{short}__{metric}__mean"]  = stat.get("mean")
                    row[f"{short}__{metric}__ci_lo"] = stat.get("ci95_low")
                    row[f"{short}__{metric}__ci_hi"] = stat.get("ci95_high")
        wide_rows.append(row)

    if wide_rows:
        fieldnames = list(wide_rows[0].keys())
        with open(wide_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"[summary] wrote {wide_path}")
    else:
        print("[summary] no rows to write for wide CSV.")

    # ------- console table -------
    print("\n=== baseline spearman (mean [CI]) per (task, dataset) ===")
    print(f"{'task':<22} {'dataset':<55} {'valid_loss':>22} {'valid_omn':>22} {'valid_mse':>22}")
    print("-" * 145)
    def fmt(r, metric):
        stat = r["metrics"].get(metric, {})
        m = stat.get("mean")
        lo, hi = stat.get("ci95_low"), stat.get("ci95_high")
        if m is None: return "nan"
        return f"{m:+.3f} [{lo:+.2f},{hi:+.2f}]"
    for (task, dataset), per in sorted(by_td.items()):
        cols = []
        for proxy in BASELINE_TAGS:
            r = per.get(proxy)
            cols.append(fmt(r, "spearman") if r else "nan")
        print(f"{task:<22} {dataset[:55]:<55} {cols[0]:>22} {cols[1]:>22} {cols[2]:>22}")


if __name__ == "__main__":
    main()
