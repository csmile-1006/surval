"""
Per (task, dataset) reporting for the local_threshold_seqval baselines + our
method. Selects the best HP for our method (PrefixSurvival_Score) by
--select_by (default: hit@5; ties broken by mmrv asc) and reports a fixed
set of metrics:

    spearman, hit@1, hit@3, hit@5, nregret, mmrv

Outputs:
    <output_root>/report_task_dataset.csv      # 1 row per (task, dataset, method, metric)
    <output_root>/report_task_dataset_wide.csv # 1 row per (task, dataset),
                                                 columns = (method × metric)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


REPORT_METRICS = ["spearman", "hit@1", "hit@3", "hit@5", "nregret", "mmrv"]
LOWER_IS_BETTER = {"mmrv", "nregret", "rank_pct"}

BASELINE_TAGS = {
    "valid_loss": "Cache/Valid/Loss",
    "valid_omn":  "Cache/Valid/Off_Manifold_Norm",
    "valid_mse":  "Cache/Valid/MSE",
}
OUR_TAG = "SequentialValid/PrefixSurvival_Score"
OUR_NAME = "ours"

HP_FIELDS = ["k", "tq", "lsetau", "ctf", "ta", "nds", "share", "skip", "dist"]


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument(
        "--select_by", type=str, default="hit@5",
        help="Metric used to pick the best HP for our method per "
             "(task, dataset). Ties broken by mmrv ascending. Default: hit@5.",
    )
    p.add_argument("--tasks", type=str, nargs="*", default=None)
    p.add_argument(
        "--print", dest="print_table", action="store_true", default=True,
        help="Print a console table at the end (default: on).",
    )
    p.add_argument(
        "--no-print", dest="print_table", action="store_false",
        help="Skip the console table.",
    )
    return p


def _stat_pack(entry):
    return {
        "mean":      entry.get("mean"),
        "median":    entry.get("median"),
        "std":       entry.get("std"),
        "ci95_low":  entry.get("ci95_low"),
        "ci95_high": entry.get("ci95_high"),
        "n":         entry.get("n"),
    }


def _select_best_hp(entries, select_by):
    """Sort by (select_by [signed], mmrv asc) and return the head entry."""
    def _signed(name, info):
        v = info["metrics"].get(name, {}).get("mean")
        if v is None:
            return float("inf") if name in LOWER_IS_BETTER else float("-inf")
        return v if name in LOWER_IS_BETTER else -v
    return min(entries, key=lambda info_kv: (
        _signed(select_by, info_kv[0]),
        _signed("mmrv",   info_kv[0]),
    ))


def _load_our_method_best(output_root, task, select_by):
    """{dataset: (info, hp_key)} of best HP per dataset for our method."""
    path = os.path.join(output_root, task, "our_method.json")
    if not os.path.isfile(path):
        return {}
    agg = json.load(open(path))["aggregated"]
    by_ds = defaultdict(list)
    for hp_key, info in agg.items():
        if info["hparams"].get("seqval_tag") != OUR_TAG:
            continue
        by_ds[info["hparams"].get("dataset", "?")].append((info, hp_key))
    return {ds: _select_best_hp(entries, select_by) for ds, entries in by_ds.items()}


def _load_baseline(output_root, task):
    """
    {dataset: {short_name: info}} for the 3 Cache/Valid/* baselines. Picks the
    first hp_key per (dataset, tag) — all are identical because baselines are
    HP-invariant.
    """
    path = os.path.join(output_root, task, "aggregate.json")
    if not os.path.isfile(path):
        return {}
    agg = json.load(open(path))["aggregated"]
    out = defaultdict(dict)
    seen = set()
    for hp_key, info in agg.items():
        tag = info["hparams"].get("seqval_tag")
        ds  = info["hparams"].get("dataset", "?")
        short = next((s for s, t in BASELINE_TAGS.items() if t == tag), None)
        if short is None:
            continue
        k = (ds, short)
        if k in seen:
            continue
        seen.add(k)
        out[ds][short] = info
    return out


def _list_tasks(output_root):
    return [
        t for t in sorted(os.listdir(output_root))
        if os.path.isfile(os.path.join(output_root, t, "our_method.json"))
        or os.path.isfile(os.path.join(output_root, t, "aggregate.json"))
    ]


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    tasks = args.tasks or _list_tasks(output_root)

    # Build rows.
    long_rows = []  # {task, dataset, method, hp..., metric, mean, ci_lo, ci_hi, std, median, n}
    wide_rows = []  # {task, dataset, hp..., <method>__<metric>__{mean,ci_lo,ci_hi}}
    methods_in_order = [OUR_NAME, "valid_loss", "valid_omn", "valid_mse"]

    for task in tasks:
        our_best = _load_our_method_best(output_root, task, args.select_by)
        base = _load_baseline(output_root, task)
        datasets = sorted(set(our_best.keys()) | set(base.keys()))

        for ds in datasets:
            row = {"task": task, "dataset": ds, "select_by": args.select_by}
            # Our method
            best_pair = our_best.get(ds)
            hp_vals = {f"best_hp_{k}": None for k in HP_FIELDS}
            if best_pair is not None:
                info, hp_key = best_pair
                hp_vals = {f"best_hp_{k}": info["hparams"].get(k) for k in HP_FIELDS}
                row.update(hp_vals)
                row["best_hp_key"] = hp_key
                for metric in REPORT_METRICS:
                    stat = _stat_pack(info["metrics"].get(metric, {}))
                    row[f"{OUR_NAME}__{metric}__mean"]   = stat["mean"]
                    row[f"{OUR_NAME}__{metric}__ci_lo"]  = stat["ci95_low"]
                    row[f"{OUR_NAME}__{metric}__ci_hi"]  = stat["ci95_high"]
                    long_rows.append({
                        "task": task, "dataset": ds, "method": OUR_NAME,
                        "select_by": args.select_by, **hp_vals,
                        "metric": metric, **stat,
                    })
            else:
                row.update(hp_vals)
                for metric in REPORT_METRICS:
                    row[f"{OUR_NAME}__{metric}__mean"]   = None
                    row[f"{OUR_NAME}__{metric}__ci_lo"]  = None
                    row[f"{OUR_NAME}__{metric}__ci_hi"]  = None

            # Baselines
            base_ds = base.get(ds, {})
            for short in ("valid_loss", "valid_omn", "valid_mse"):
                info = base_ds.get(short)
                for metric in REPORT_METRICS:
                    stat = _stat_pack(info["metrics"].get(metric, {})) if info else _stat_pack({})
                    row[f"{short}__{metric}__mean"]   = stat["mean"]
                    row[f"{short}__{metric}__ci_lo"]  = stat["ci95_low"]
                    row[f"{short}__{metric}__ci_hi"]  = stat["ci95_high"]
                    if info is not None:
                        long_rows.append({
                            "task": task, "dataset": ds, "method": short,
                            "select_by": args.select_by,
                            **{f"best_hp_{k}": None for k in HP_FIELDS},
                            "metric": metric, **stat,
                        })

            wide_rows.append(row)

    # ---- write long CSV ----
    long_path = os.path.join(output_root, "report_task_dataset.csv")
    if long_rows:
        long_fields = [
            "task", "dataset", "method", "select_by",
            *[f"best_hp_{k}" for k in HP_FIELDS],
            "metric", "mean", "median", "std", "ci95_low", "ci95_high", "n",
        ]
        with open(long_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=long_fields)
            w.writeheader()
            w.writerows(long_rows)
        print(f"[report] wrote {long_path}  ({len(long_rows)} rows)")

    # ---- write wide CSV ----
    wide_path = os.path.join(output_root, "report_task_dataset_wide.csv")
    if wide_rows:
        wide_fields = ["task", "dataset", "select_by",
                       *[f"best_hp_{k}" for k in HP_FIELDS], "best_hp_key"]
        for method in methods_in_order:
            for metric in REPORT_METRICS:
                wide_fields += [
                    f"{method}__{metric}__mean",
                    f"{method}__{metric}__ci_lo",
                    f"{method}__{metric}__ci_hi",
                ]
        with open(wide_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(wide_rows)
        print(f"[report] wrote {wide_path}  ({len(wide_rows)} rows)")

    # ---- console report ----
    if not args.print_table:
        return

    def _cell(row, method, metric):
        m  = row.get(f"{method}__{metric}__mean")
        lo = row.get(f"{method}__{metric}__ci_lo")
        hi = row.get(f"{method}__{metric}__ci_hi")
        if m is None:
            return "    nan        "
        return f"{m:+.3f} [{lo:+.2f},{hi:+.2f}]"

    methods = methods_in_order
    metrics_to_show = REPORT_METRICS
    for task in tasks:
        task_rows = [r for r in wide_rows if r["task"] == task]
        if not task_rows:
            continue
        print(f"\n### {task}  (best-HP for ours selected by {args.select_by}, tiebreak mmrv asc) ###")
        for row in task_rows:
            print(f"  dataset = {row['dataset']}")
            hp = ",".join(f"{k}={row.get(f'best_hp_{k}')}" for k in HP_FIELDS
                          if row.get(f"best_hp_{k}") is not None)
            print(f"    ours best HP: {hp if hp else '(missing)'}")
            print(f"    {'metric':<10} | {'ours':>22} | {'valid_loss':>22} | "
                  f"{'valid_omn':>22} | {'valid_mse':>22}")
            print(f"    {'-'*10}-+-{'-'*22}-+-{'-'*22}-+-{'-'*22}-+-{'-'*22}")
            for metric in metrics_to_show:
                line = f"    {metric:<10} | "
                line += " | ".join(f"{_cell(row, m, metric):>22}" for m in methods)
                print(line)


if __name__ == "__main__":
    main()
