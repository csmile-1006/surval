"""
Collapse per-task our_method.json into a per-(task, dataset) summary,
selecting the best HP (by `--rank_by` metric) for each cell and reporting
its full metric set with bootstrap CI.

Writes alongside baselines_summary.* in the output_root:

    <output_root>/our_method_summary.json
    <output_root>/our_method_summary.csv         (long format)
    <output_root>/our_method_summary_wide.csv    (wide format, one row per task+dataset)
    <output_root>/our_method_topk.csv            (top-K HPs per (task, dataset))
    <output_root>/comparison_summary.csv         (best-our-method vs each baseline,
                                                  for the rank-by metric)

For PrefixSurvival_Score, higher-is-better metrics (spearman/kendall/hit@K)
are maximized; lower-is-better metrics (mmrv/nregret/rank_pct) are minimized.
Pass --rank_by to pick the selection criterion (default: mmrv).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


PRIMARY_METRICS = [
    "spearman", "kendall_tau", "delta_spearman",
    "hit@1", "hit@2", "hit@3", "hit@4", "hit@5",
    "nregret", "rank_pct", "mmrv",
]
LOWER_IS_BETTER = {"mmrv", "nregret", "rank_pct"}

HP_FIELDS = ["tq", "lsetau", "ctf", "ta", "nds", "share", "skip", "dist"]


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument(
        "--rank_by", type=str, default="mmrv",
        help="Metric used to pick the best HP per (task, dataset). Lower-is-"
             "better metrics auto-flip the direction. Default: mmrv.",
    )
    p.add_argument(
        "--topk", type=int, default=3,
        help="Number of top HPs per (task, dataset) to emit in our_method_topk.csv.",
    )
    p.add_argument("--tasks", type=str, nargs="*", default=None)
    return p


def _stat(entry):
    return {
        "mean":      entry.get("mean"),
        "median":    entry.get("median"),
        "std":       entry.get("std"),
        "ci95_low":  entry.get("ci95_low"),
        "ci95_high": entry.get("ci95_high"),
        "n":         entry.get("n"),
    }


def _sort_key(hp_info, rank_by):
    """
    Primary: --rank_by metric (auto-flip direction for lower-is-better).
    Tiebreaker: mmrv ascending (lower is better) so ties don't pick arbitrarily.
    """
    metrics = hp_info["metrics"]

    def _signed(name):
        stat = metrics.get(name, {})
        v = stat.get("mean")
        if v is None:
            return float("inf") if name in LOWER_IS_BETTER else float("-inf")
        return v if name in LOWER_IS_BETTER else -v

    return (_signed(rank_by), _signed("mmrv"))


def _summarize_task(aggregate_path, task_name, rank_by, topk):
    """
    Returns:
      best_rows: list of dicts, one per (task, dataset).
      topk_rows: list of dicts, top-K HPs per (task, dataset).
    """
    agg = json.load(open(aggregate_path))["aggregated"]

    # Group entries by (dataset, proxy_tag).
    by_cell = defaultdict(list)  # (dataset, proxy_tag) -> [(hp_info, hp_key)]
    for hp_key, info in agg.items():
        hp = info["hparams"]
        by_cell[(hp.get("dataset", "?"), hp.get("seqval_tag", "?"))].append(
            (info, hp_key)
        )

    best_rows = []
    topk_rows = []
    for (dataset, tag), entries in sorted(by_cell.items()):
        entries.sort(key=lambda iv: _sort_key(iv[0], rank_by))
        info_best, key_best = entries[0]
        best_rows.append({
            "task": task_name,
            "dataset": dataset,
            "proxy_tag": tag,
            "proxy_source": info_best["hparams"].get("proxy_source", "?"),
            "num_seeds": info_best.get("num_seeds"),
            "num_hps": len(entries),
            "best_hp_key": key_best,
            "best_hp": {k: info_best["hparams"].get(k) for k in HP_FIELDS},
            "metrics": {m: _stat(info_best["metrics"][m])
                        for m in PRIMARY_METRICS if m in info_best["metrics"]},
            "rank_by": rank_by,
        })
        for rank, (info, key) in enumerate(entries[:topk], 1):
            stat = info["metrics"].get(rank_by, {})
            row = {
                "task": task_name, "dataset": dataset, "rank": rank,
                "rank_by": rank_by,
                **{k: info["hparams"].get(k) for k in HP_FIELDS},
                f"{rank_by}_mean": stat.get("mean"),
                f"{rank_by}_ci95_low": stat.get("ci95_low"),
                f"{rank_by}_ci95_high": stat.get("ci95_high"),
                "spearman_mean": info["metrics"].get("spearman", {}).get("mean"),
                "hit@1_mean": info["metrics"].get("hit@1", {}).get("mean"),
                "hp_key": key,
            }
            topk_rows.append(row)
    return best_rows, topk_rows


def _load_baseline_summary(output_root):
    """Optional: load baselines_summary.json so we can emit a head-to-head CSV."""
    path = os.path.join(output_root, "baselines_summary.json")
    if not os.path.isfile(path):
        return None
    return json.load(open(path))


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    output_root = os.path.abspath(args.output_root)

    if args.tasks:
        tasks = list(args.tasks)
    else:
        tasks = [
            t for t in sorted(os.listdir(output_root))
            if os.path.isfile(os.path.join(output_root, t, "our_method.json"))
        ]

    all_best, all_topk = [], []
    for t in tasks:
        path = os.path.join(output_root, t, "our_method.json")
        if not os.path.isfile(path):
            print(f"[our_method_summary] skip {t}: our_method.json missing")
            continue
        best, topk = _summarize_task(path, t, args.rank_by, args.topk)
        all_best.extend(best)
        all_topk.extend(topk)
        print(f"[our_method_summary] {t}: {len(best)} (task, dataset) cells, "
              f"top-{args.topk}/cell")

    # ---- JSON ----
    js_path = os.path.join(output_root, "our_method_summary.json")
    with open(js_path, "w") as f:
        json.dump({"rank_by": args.rank_by, "rows": all_best}, f, indent=2)
    print(f"[our_method_summary] wrote {js_path}")

    # ---- long CSV (one row per (task, dataset, metric)) ----
    long_path = os.path.join(output_root, "our_method_summary.csv")
    with open(long_path, "w", newline="") as f:
        fieldnames = [
            "task", "dataset", "proxy_tag", "num_seeds", "num_hps", "rank_by",
            "best_hp_tq", "best_hp_lsetau", "best_hp_ctf", "best_hp_ta",
            "best_hp_nds", "best_hp_share", "best_hp_skip", "best_hp_dist",
            "metric", "mean", "median", "std", "ci95_low", "ci95_high", "n",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_best:
            hp = r["best_hp"]
            base = {
                "task": r["task"], "dataset": r["dataset"],
                "proxy_tag": r["proxy_tag"], "num_seeds": r["num_seeds"],
                "num_hps": r["num_hps"], "rank_by": r["rank_by"],
                **{f"best_hp_{k}": hp.get(k) for k in HP_FIELDS},
            }
            for metric, stat in r["metrics"].items():
                writer.writerow({**base, "metric": metric, **stat})
    print(f"[our_method_summary] wrote {long_path}")

    # ---- wide CSV (one row per (task, dataset), columns = metric × {mean, ci_lo, ci_hi}) ----
    wide_path = os.path.join(output_root, "our_method_summary_wide.csv")
    if all_best:
        wide_rows = []
        for r in all_best:
            hp = r["best_hp"]
            row = {
                "task": r["task"], "dataset": r["dataset"],
                "num_seeds": r["num_seeds"], "num_hps": r["num_hps"],
                "rank_by": r["rank_by"],
                **{f"best_hp_{k}": hp.get(k) for k in HP_FIELDS},
            }
            for metric in PRIMARY_METRICS:
                stat = r["metrics"].get(metric, {})
                row[f"{metric}__mean"]  = stat.get("mean")
                row[f"{metric}__ci_lo"] = stat.get("ci95_low")
                row[f"{metric}__ci_hi"] = stat.get("ci95_high")
            wide_rows.append(row)
        with open(wide_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"[our_method_summary] wrote {wide_path}")

    # ---- top-K CSV ----
    if all_topk:
        topk_path = os.path.join(output_root, "our_method_topk.csv")
        with open(topk_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_topk[0].keys()))
            writer.writeheader()
            writer.writerows(all_topk)
        print(f"[our_method_summary] wrote {topk_path}")

    # ---- head-to-head vs baselines ----
    baseline = _load_baseline_summary(output_root)
    if baseline is not None:
        base_lookup = {}  # (task, dataset, proxy_tag) -> stat[rank_by]
        for r in baseline.get("rows", []):
            base_lookup[(r["task"], r["dataset"], r["proxy_tag"])] = \
                r.get("metrics", {}).get(args.rank_by, {})

        cmp_path = os.path.join(output_root, "comparison_summary.csv")
        with open(cmp_path, "w", newline="") as f:
            fieldnames = [
                "task", "dataset", "rank_by",
                "our_method_mean", "our_method_ci_lo", "our_method_ci_hi",
                "valid_loss_mean", "valid_loss_ci_lo", "valid_loss_ci_hi",
                "valid_omn_mean",  "valid_omn_ci_lo",  "valid_omn_ci_hi",
                "valid_mse_mean",  "valid_mse_ci_lo",  "valid_mse_ci_hi",
                "best_hp_tq", "best_hp_lsetau", "best_hp_ctf", "best_hp_ta",
                "best_hp_nds", "best_hp_share", "best_hp_skip", "best_hp_dist",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_best:
                hp = r["best_hp"]
                our_stat = r["metrics"].get(args.rank_by, {})
                base_loss = base_lookup.get((r["task"], r["dataset"], "Cache/Valid/Loss"), {})
                base_omn  = base_lookup.get((r["task"], r["dataset"], "Cache/Valid/Off_Manifold_Norm"), {})
                base_mse  = base_lookup.get((r["task"], r["dataset"], "Cache/Valid/MSE"), {})
                writer.writerow({
                    "task": r["task"], "dataset": r["dataset"], "rank_by": args.rank_by,
                    "our_method_mean":   our_stat.get("mean"),
                    "our_method_ci_lo":  our_stat.get("ci95_low"),
                    "our_method_ci_hi":  our_stat.get("ci95_high"),
                    "valid_loss_mean":   base_loss.get("mean"),
                    "valid_loss_ci_lo":  base_loss.get("ci95_low"),
                    "valid_loss_ci_hi":  base_loss.get("ci95_high"),
                    "valid_omn_mean":    base_omn.get("mean"),
                    "valid_omn_ci_lo":   base_omn.get("ci95_low"),
                    "valid_omn_ci_hi":   base_omn.get("ci95_high"),
                    "valid_mse_mean":    base_mse.get("mean"),
                    "valid_mse_ci_lo":   base_mse.get("ci95_low"),
                    "valid_mse_ci_hi":   base_mse.get("ci95_high"),
                    **{f"best_hp_{k}": hp.get(k) for k in HP_FIELDS},
                })
        print(f"[our_method_summary] wrote {cmp_path}")

    # ---- console table ----
    print(f"\n=== best-HP {args.rank_by} per (task, dataset)  vs baselines ===")
    print(f"{'task':<22} {'dataset':<55} {'ours':>22} {'valid_loss':>22} "
          f"{'valid_omn':>22} {'valid_mse':>22}")
    print("-" * 168)
    base_lookup_full = {}
    if baseline is not None:
        for r in baseline.get("rows", []):
            base_lookup_full[(r["task"], r["dataset"], r["proxy_tag"])] = r["metrics"]

    def fmt(stat):
        if not stat: return "nan"
        m = stat.get("mean"); lo = stat.get("ci95_low"); hi = stat.get("ci95_high")
        if m is None: return "nan"
        return f"{m:+.3f} [{lo:+.2f},{hi:+.2f}]"

    for r in all_best:
        ours = fmt(r["metrics"].get(args.rank_by, {}))
        loss = fmt(base_lookup_full.get((r["task"], r["dataset"], "Cache/Valid/Loss"), {}).get(args.rank_by, {}))
        omn  = fmt(base_lookup_full.get((r["task"], r["dataset"], "Cache/Valid/Off_Manifold_Norm"), {}).get(args.rank_by, {}))
        mse  = fmt(base_lookup_full.get((r["task"], r["dataset"], "Cache/Valid/MSE"), {}).get(args.rank_by, {}))
        print(f"{r['task']:<22} {r['dataset'][:55]:<55} {ours:>22} {loss:>22} "
              f"{omn:>22} {mse:>22}")


if __name__ == "__main__":
    main()
