"""Select, report and canonically audit the completed DROID HP grid."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import csv
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np

from cache_droid_datasets import REPO, save_json
from tune_droid_hp import sha256

sys.path.insert(0, str(REPO / "src"))
from surval.droid import _json_value, load_policy_cache
from surval.droid_dino import align_dino_cache, dataset_db_path, load_shared_dino_db
from surval.droid_tuning import canonical_score, canonical_thresholds, hp_grid, primary_metrics

METRICS = ("nregret", "mmrv", "spearman", "balanced_loss", "regret_pp", "selected_success_rate")


def write_csv(path, rows):
    if not rows:
        return
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows):
    return {**{m: float(np.mean([r[m] for r in rows])) for m in METRICS},
            "eligible": all(r["eligible"] for r in rows)}


def best_index(rows, metric="balanced_loss"):
    eligible = [i for i, r in enumerate(rows) if r["eligible"]]
    if not eligible:
        raise ValueError("No nondegenerate HP covers all requested cells")
    sign = -1 if metric == "spearman" else 1
    return min(eligible, key=lambda i: (sign * rows[i][metric], i))


def pareto_indices(rows):
    indices = [i for i, r in enumerate(rows) if r["eligible"]]
    values = np.array([[rows[i]["nregret"], rows[i]["mmrv"], -rows[i]["spearman"]] for i in indices])
    return [i for i, value in zip(indices, values)
            if not np.any(np.all(values <= value, axis=1) & np.any(values < value, axis=1))]


def load_results(root, outcomes):
    manifests = sorted(root.glob("run_manifest*.json"))
    if not manifests:
        raise FileNotFoundError("No run manifest")
    run = json.loads(manifests[0].read_text())
    if not run["fixed"]["every_step"] or not run["fixed"]["exact_neighbors"]:
        raise ValueError("Expected every-step exact-neighbor sweep")
    grid = hp_grid(run["axes"])
    cells, score_tables = {}, {}
    for job in run["jobs"]:
        directory = Path(job["job_dir"]) / "eval"
        verification = json.loads((directory / "verification.json").read_text())
        if not verification["complete"] or verification["grid_size"] != len(grid):
            raise ValueError("Incomplete worker")
        for artifact in ("scores", "scales"):
            if sha256(directory / f"{artifact}.npz") != verification[f"{artifact}_sha256"]:
                raise ValueError(f"Changed {artifact} artifact")
        for cache, expected in verification["signature"]["cache_sha256"].items():
            if sha256(cache) != expected:
                raise ValueError("Input cache changed")
        saved = np.load(directory / "scores.npz")
        if saved["scores"].shape != (len(grid), 10) or not np.isfinite(saved["scores"]).all():
            raise ValueError("Incomplete or nonfinite checkpoint matrix")
        np.testing.assert_array_equal(saved["epochs"], outcomes["epoch_order"])
        actual = np.asarray(outcomes["successes"][job["task"]]) / outcomes["trials_per_checkpoint"]
        cells[job["dataset"]] = []
        for i, scores in enumerate(saved["scores"]):
            m = primary_metrics(actual, scores)
            cells[job["dataset"]].append({"hp_id": i, **grid[i], "task": job["task"], "val": job["val"],
                                           **m, "selected_epoch": int(saved["epochs"][m["selected_index"]])})
        score_tables[job["dataset"]] = {k: saved[k].copy() for k in saved.files}
    expected = {(task, size) for task in ("apple", "pan", "pet") for size in (5, 10, 20, 30)}
    if {(j["task"], j["val"]) for j in run["jobs"]} != expected or len(run["jobs"]) != 12:
        raise ValueError("Expected all three tasks and all four validation splits")
    return run, grid, cells, score_tables


def select(run, grid, cells):
    domains = {task: [] for task in ("apple", "pan", "pet")}
    jobs_by_task = {task: [j for j in run["jobs"] if j["task"] == task] for task in domains}
    for task, jobs in jobs_by_task.items():
        domains[task] = [{"hp_id": i, **hp, "task": task,
                          **mean_metrics([cells[j["dataset"]][i] for j in jobs])} for i, hp in enumerate(grid)]
    common = [{"hp_id": i, **hp, **mean_metrics([domains[t][i] for t in domains]),
               "worst_task_loss": max(domains[t][i]["balanced_loss"] for t in domains)}
              for i, hp in enumerate(grid)]
    selections = []
    for dataset, rows in cells.items():
        for metric in ("balanced_loss", "nregret", "mmrv", "spearman"):
            i = best_index(rows, metric)
            selections.append({"scope": "cell", "name": dataset, "criterion": metric, "hp_id": i})
    for task, rows in domains.items():
        for metric in ("balanced_loss", "nregret", "mmrv", "spearman"):
            i = best_index(rows, metric)
            selections.append({"scope": "domain", "name": task, "criterion": metric, "hp_id": i})
    eligible = [i for i, row in enumerate(common) if row["eligible"]]
    robust = min(eligible, key=lambda i: (common[i]["worst_task_loss"], common[i]["balanced_loss"], i))
    macro = best_index(common)
    selections += [{"scope": "common", "name": "all", "criterion": "robust", "hp_id": robust},
                   {"scope": "common", "name": "all", "criterion": "macro", "hp_id": macro}]
    pareto = pareto_indices(common)
    # Full frontier is saved; five diverse metric/robust representatives are audited.
    alternatives = set(sorted(pareto, key=lambda i: (common[i]["balanced_loss"], i))[:2])
    alternatives.update(best_index(common, m) for m in ("nregret", "mmrv", "spearman"))
    for rank, i in enumerate(sorted(alternatives), 1):
        selections.append({"scope": "common", "name": "all", "criterion": f"alternative_{rank}", "hp_id": i})
    for selected in selections:
        selected.update(grid[selected["hp_id"]])
        rows = cells[selected["name"]] if selected["scope"] == "cell" else (
               domains[selected["name"]] if selected["scope"] == "domain" else common)
        selected["metrics"] = {m: rows[selected["hp_id"]][m] for m in METRICS}
    return domains, common, selections, pareto


def selected_for_job(job, selections):
    return sorted({s["hp_id"] for s in selections if s["scope"] == "common" or
                   (s["scope"] == "cell" and s["name"] == job["dataset"]) or
                   (s["scope"] == "domain" and s["name"] == job["task"])})


def audit_selected(root, run, grid, selections, tables, outcomes, dino_root):
    audits, max_error, max_threshold_error = [], 0.0, 0.0
    for job in run["jobs"]:
        wanted = selected_for_job(job, selections)
        directory = Path(job["job_dir"]) / "eval"
        signature = {"hp_ids": wanted, "scores_sha256": sha256(directory / "scores.npz"),
                     "outcomes": outcomes}
        completed = directory / "selected_audit.json"
        if completed.exists():
            previous = json.loads(completed.read_text())
            if previous["signature"] == signature and previous["complete"]:
                audits.extend(previous["rows"])
                max_error = max(max_error, previous["max_abs_error"])
                max_threshold_error = max(max_threshold_error, previous["threshold_max_abs_error"])
                continue
        db, chunks, metadata = load_shared_dino_db(dataset_db_path(dino_root, job["dataset"]))
        stored_scales = np.load(directory / "scales.npz")
        thresholds = {}
        for k in sorted({grid[i]["k"] for i in wanted}):
            target = directory / "canonical_selected" / str(k)
            thresholds[k] = canonical_thresholds(db, run["axes"]["quantile"], k, target)
            error = float(np.max(abs(thresholds[k].thresholds - stored_scales[str(k)])))
            max_threshold_error = max(max_threshold_error, error)
            np.testing.assert_allclose(thresholds[k].thresholds, stored_scales[str(k)], rtol=1e-6, atol=1e-7)
        actual = np.array(outcomes["successes"][job["task"]]) / outcomes["trials_per_checkpoint"]
        canonical = np.empty((len(wanted), 10))
        cell_rows, cell_error = [], 0.0
        for col, row in enumerate(job["rows"]):
            cache = load_policy_cache(row["cache_file"])
            align_dino_cache(db, chunks, metadata, cache)
            for wi, i in enumerate(wanted):
                hp = grid[i]
                target = directory / "canonical_selected" / str(hp["k"])
                with redirect_stdout(io.StringIO()):
                    score = canonical_score(cache, target, thresholds[hp["k"]], hp)
                fast = tables[job["dataset"]]["scores"][i, col]
                error = abs(score - fast)
                cell_error = max(cell_error, error)
                np.testing.assert_allclose(score, fast, atol=2e-7, rtol=2e-6)
                canonical[wi, col] = score
                cell_rows.append({"dataset": job["dataset"], "hp_id": i, "epoch": row["epoch"],
                                  "fast_score": float(fast), "canonical_score": score, "abs_error": error})
        for wi, i in enumerate(wanted):
            fast_m = primary_metrics(actual, tables[job["dataset"]]["scores"][i])
            canonical_m = primary_metrics(actual, canonical[wi])
            for name in (*METRICS, "selected_index", "eligible"):
                if not np.isclose(fast_m[name], canonical_m[name], atol=1e-12, rtol=0, equal_nan=True):
                    raise AssertionError(f"Canonical metric changed: {job['dataset']} HP{i} {name}: "
                                         f"{fast_m[name]} != {canonical_m[name]}")
        save_json(completed, {"complete": True, "signature": signature, "rows": cell_rows,
                              "max_abs_error": cell_error, "threshold_max_abs_error": max_threshold_error,
                              "primary_metrics_and_selection_unchanged": True,
                              "global_thresholds": "Unused zero placeholders; no fallback rows"})
        audits.extend(cell_rows)
        max_error = max(max_error, cell_error)
        print(f"AUDIT {job['dataset']} settings={len(wanted)} checkpoints=10 max_error={cell_error:.3g}", flush=True)
    write_csv(root / "canonical_audit.csv", audits)
    return {"complete": True, "score_comparisons": len(audits), "max_abs_error": max_error,
            "threshold_max_abs_error": max_threshold_error, "primary_metrics_and_selection_unchanged": True}


def report(args):
    root = args.output_root.resolve()
    outcomes = json.loads(args.outcomes.read_text())
    run, grid, cells, tables = load_results(root, outcomes)
    domains, common, selections, pareto = select(run, grid, cells)
    save_json(root / "selections.json", _json_value(selections))
    write_csv(root / "all_cell_metrics.csv", [r for rows in cells.values() for r in rows])
    write_csv(root / "all_domain_metrics.csv", [r for rows in domains.values() for r in rows])
    write_csv(root / "all_common_metrics.csv", common)
    write_csv(root / "pareto_common.csv", [common[i] for i in pareto])
    selected_rows, controls = [], []
    checkpoint_rows = []
    default = {"ta": 8, "k": 50, "quantile": 0.95, "scale_multiplier": 1.0}
    default_i = grid.index(default)
    for job in run["jobs"]:
        dataset = job["dataset"]
        for s in selections:
            if s["hp_id"] in selected_for_job(job, [s]):
                selected_rows.append({"scope": s["scope"], "name": s["name"], "criterion": s["criterion"],
                                      **cells[dataset][s["hp_id"]]})
        actual = np.array(outcomes["successes"][job["task"]]) / outcomes["trials_per_checkpoint"]
        for name, scores in (("historical_buffered_stride8", tables[dataset]["old_stride"]),
                             ("buffered_every_step", tables[dataset]["old_every_step"]),
                             ("exact_k50_every_step", tables[dataset]["scores"][default_i])):
            m = primary_metrics(actual, scores)
            controls.append({"task": job["task"], "val": job["val"], "control": name, **m,
                             "selected_epoch": outcomes["epoch_order"][m["selected_index"]]})
        for i, hp in enumerate(grid):
            for col, epoch in enumerate(outcomes["epoch_order"]):
                checkpoint_rows.append({"task": job["task"], "val": job["val"], "hp_id": i, **hp,
                                        "epoch": epoch, "surval_score": tables[dataset]["scores"][i, col],
                                        "successes": outcomes["successes"][job["task"]][col],
                                        "trials": outcomes["trials_per_checkpoint"]})
    write_csv(root / "selected_cell_metrics.csv", selected_rows)
    write_csv(root / "controls.csv", controls)
    write_csv(root / "checkpoint_scores.csv.gz", checkpoint_rows)
    audit = audit_selected(root, run, grid, selections, tables, outcomes, args.dino_root) if args.audit else {
        "complete": False, "reason": "Run with --audit before calling these verified optima"}
    summary = {"axes": run["axes"], "fixed": run["fixed"], "grid_size": len(grid),
               "cells": len(cells), "checkpoint_scores": len(checkpoint_rows),
               "selections": selections, "canonical_audit": audit, "pareto_settings": len(pareto),
               "boundary_hits": [s for s in selections if s["criterion"] in ("balanced_loss", "robust", "macro")
                                 and (s["k"] == max(run["axes"]["k"]) or s["scale_multiplier"] in
                                      (min(run["axes"]["scale_multiplier"]), max(run["axes"]["scale_multiplier"])))],
               "caveat": "Post-hoc real-world tuning, one training run/task; not held-out validation."}
    save_json(root / "summary.json", _json_value(summary))
    lines = ["# DROID SurVAL HP tuning", "", "All scores: every_step=True, exact eligible-k, all8 samples.",
             "Exploratory post-hoc tuning on the supplied30-trial checkpoint outcomes; not held-out evidence.",
             "", f"Grid: {len(grid)} HPs x12 cells x10 checkpoints. Canonical audit complete: {audit['complete']}.",
             "", "Regret is normalized by the observed task SR range. MMRV is in0-1 SR units.",
             "Balanced loss = (nregret + MMRV/task SR range + (1-Spearman)/2)/3.",
             "Task rows average val5/10/20/30 equally; validation size is not a tuned axis.", "",
             "| Selection | Task | Ta | k | q | scale x | nregret | MMRV | Spearman |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in selections:
        if (s["scope"] == "domain" and s["criterion"] == "balanced_loss") or (
                s["scope"] == "common" and s["criterion"] in ("robust", "macro")):
            for task in ([s["name"]] if s["scope"] == "domain" else list(domains)):
                row = domains[task][s["hp_id"]]
                lines.append(f"| {s['scope']}/{s['criterion']} | {task} | {s['ta']} | {s['k']} | "
                             f"{s['quantile']} | {s['scale_multiplier']} | {row['nregret']:.4f} | "
                             f"{row['mmrv']:.4f} | {row['spearman']:.4f} |")
    lines += ["", "## Original-HP controls (same four-split means)", "",
              "| Control | Task | nregret | MMRV | Spearman |", "|---|---|---:|---:|---:|"]
    for name in sorted({r["control"] for r in controls}):
        for task in domains:
            m = mean_metrics([r for r in controls if r["control"] == name and r["task"] == task])
            lines.append(f"| {name} | {task} | {m['nregret']:.4f} | {m['mmrv']:.4f} | {m['spearman']:.4f} |")
    lines += ["", "## Files", "", "- selections.json: task/split, task and common metric-specific choices.",
              "- selected_cell_metrics.csv: each selected HP on its applicable validation splits, including selected epochs.",
              "- all_{cell,domain,common}_metrics.csv: exhaustive metric values, eligibility and balanced objective.",
              "- checkpoint_scores.csv.gz: every checkpoint score and real-world count for every HP.",
              "- pareto_common.csv: complete macro-metric nondominated set; not all points are uniquely optimal.",
              "- canonical_audit.csv and per-cell selected_audit.json: canonical checks of reported selections.",
              "", "No independent seeds/CI. Full-val reference changes with val split. Chunk>8 requires new cache inference.",
              "Baselines retain their prior definition/coverage; this sweep tunes SURVAL only.",
              "Scale multiplier is common to both group-specific quantiles, not an independently tuned group ratio."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"grid_size": len(grid), "selections": [s for s in selections if s["scope"] != "cell"
                     and s["criterion"] in ("balanced_loss", "robust", "macro")],
                      "boundary_hits": len(summary["boundary_hits"]), "canonical_audit": audit}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/droid_hp_every_step_20260913")
    parser.add_argument("--outcomes", type=Path, default=REPO / "outputs/droid_real_world_20260913/success_counts.json")
    parser.add_argument("--dino-root", type=Path, default=REPO / "outputs/shared_dino_val")
    parser.add_argument("--audit", action="store_true")
    report(parser.parse_args())
