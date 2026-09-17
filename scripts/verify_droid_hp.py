"""Independent full-matrix checks; run after report_droid_hp.py --audit."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

from cache_droid_datasets import REPO, save_json
from tune_droid_hp import sha256


def verify(root, outcomes_path, initial_root):
    manifest = json.loads((root / "run_manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    labels = json.loads(outcomes_path.read_text())
    status = json.loads((root / "run_status.json").read_text())
    assert status["state"] == "complete" and status["counts"] == {"complete": 12}
    assert manifest["fixed"] == {"every_step": True, "num_samples": 8, "temporal_radius": 5,
                                 "min_neighbors": 10, "chunk_top_frac": 0.5, "lse_tau": 1.0,
                                 "exact_neighbors": True, "seed": 0}
    assert summary["canonical_audit"]["complete"]
    with (root / "all_cell_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    grid_size = manifest["grid_size"]
    assert len(rows) == 12 * grid_size
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], int(row["val"])), []).append(row)
    assert set(grouped) == {(t, s) for t in ("apple", "pan", "pet") for s in (5, 10, 20, 30)}
    maximum_metric_error, canonical_checks, input_hashes = 0., 0, {}
    independent = {}
    initial = json.loads((initial_root / "run_manifest.json").read_text())
    initial_axes, axes = initial["axes"], manifest["axes"]
    from itertools import product
    hp_keys = list(axes)
    extended_grid = list(product(*(axes[k] for k in hp_keys)))
    initial_grid = list(product(*(initial_axes[k] for k in hp_keys)))
    initial_indices = [extended_grid.index(hp) for hp in initial_grid]
    initial_equal = 0
    for job in manifest["jobs"]:
        directory = Path(job["job_dir"]) / "eval"
        proof = json.loads((directory / "verification.json").read_text())
        assert proof["every_step"] and proof["exact_k"] and proof["fallback_rows"] == 0
        assert proof["signature"]["axes"] == axes and proof["signature"]["fixed"] == manifest["fixed"]
        for path, expected in proof["signature"]["code_sha256"].items():
            assert sha256(REPO / path) == expected
        for path, expected in proof["signature"]["cache_sha256"].items():
            assert sha256(path) == expected
            input_hashes[path] = expected
        for name in ("scores", "scales"):
            assert sha256(directory / f"{name}.npz") == proof[f"{name}_sha256"]
        saved = np.load(directory / "scores.npz")
        scores = saved["scores"]
        assert scores.shape == (grid_size, 10) and np.isfinite(scores).all()
        assert scores.min() >= 0 and scores.max() <= 1 + 1e-12
        np.testing.assert_array_equal(saved["epochs"], labels["epoch_order"])
        actual = np.asarray(labels["successes"][job["task"]]) / labels["trials_per_checkpoint"]
        selected = scores.argmax(axis=1)
        regret = (actual.max() - actual[selected]) / (np.ptp(actual) + 1e-12)
        # Independent vectorized MMRV definition, all pairs/all configurations.
        disagree = ((scores[:, :, None] < scores[:, None, :]) != (actual[:, None] < actual[None, :]))
        mmrv = (abs(actual[:, None] - actual[None, :]) * disagree).max(axis=2).mean(axis=1)
        rank_a, rank_b = rankdata(actual), rankdata(scores, axis=1)
        rank_a -= rank_a.mean()
        rank_b -= rank_b.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (rank_b @ rank_a) / (np.linalg.norm(rank_b, axis=1) * np.linalg.norm(rank_a))
        balanced = (regret + mmrv / np.ptp(actual) + (1 - rho) / 2) / 3
        eligible = np.isfinite(rho) & (np.ptp(scores, axis=1) > 1e-8)
        cell_rows = sorted(grouped[(job["task"], job["val"])], key=lambda r: int(r["hp_id"]))
        np.testing.assert_array_equal([int(r["hp_id"]) for r in cell_rows], np.arange(grid_size))
        for name, values in (("nregret", regret), ("mmrv", mmrv), ("spearman", rho), ("balanced_loss", balanced)):
            observed = np.array([float(r[name]) for r in cell_rows])
            np.testing.assert_allclose(observed, values, atol=1e-12, rtol=0, equal_nan=True)
            maximum_metric_error = max(maximum_metric_error, float(np.nanmax(abs(observed - values))))
        np.testing.assert_array_equal([r["eligible"] == "True" for r in cell_rows], eligible)
        np.testing.assert_array_equal([int(r["selected_index"]) for r in cell_rows], selected)
        independent[job["dataset"]] = balanced, eligible
        audit = json.loads((directory / "selected_audit.json").read_text())
        assert audit["complete"] and audit["primary_metrics_and_selection_unchanged"]
        assert audit["signature"]["scores_sha256"] == sha256(directory / "scores.npz")
        assert audit["signature"]["outcomes"] == labels
        canonical_checks += len(audit["rows"])
        earlier = next(j for j in initial["jobs"] if j["dataset"] == job["dataset"])
        old = np.load(Path(earlier["job_dir"]) / "eval/scores.npz")
        np.testing.assert_array_equal(scores[initial_indices], old["scores"])
        initial_equal += len(initial_indices) * 10
    assert len(input_hashes) == 120
    losses = []
    eligible_all = np.ones(grid_size, bool)
    for task in ("apple", "pan", "pet"):
        values = [independent[j["dataset"]] for j in manifest["jobs"] if j["task"] == task]
        mean = np.mean([v[0] for v in values], axis=0)
        valid = np.all([v[1] for v in values], axis=0)
        winner = next(s for s in summary["selections"] if s["scope"] == "domain" and s["name"] == task
                      and s["criterion"] == "balanced_loss")["hp_id"]
        assert valid[winner] and mean[winner] <= np.min(mean[valid]) + 1e-12
        losses.append(mean)
        eligible_all &= valid
    losses = np.stack(losses)
    for criterion, objective in (("robust", losses.max(axis=0)), ("macro", losses.mean(axis=0))):
        winner = next(s for s in summary["selections"] if s["scope"] == "common" and s["criterion"] == criterion)["hp_id"]
        assert eligible_all[winner] and objective[winner] <= np.min(objective[eligible_all]) + 1e-12
    with gzip.open(root / "checkpoint_scores.csv.gz", "rt") as stream:
        checkpoint_rows = sum(1 for _ in csv.DictReader(stream))
    assert checkpoint_rows == grid_size * 120
    assert canonical_checks == summary["canonical_audit"]["score_comparisons"]
    result = {"complete": True, "grid_size": grid_size, "cells": 12, "checkpoints": 120,
              "checkpoint_scores": checkpoint_rows, "independent_primary_metric_rows": len(rows),
              "max_independent_metric_abs_error": maximum_metric_error, "canonical_score_checks": canonical_checks,
              "initial_grid_scores_bitwise_unchanged": initial_equal, "source_caches_sha256_unchanged": len(input_hashes),
              "independent_domain_and_common_optimality_checks": True, "outcomes_sha256": sha256(outcomes_path),
              "report_code_sha256": sha256(REPO / "scripts/report_droid_hp.py")}
    save_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/droid_hp_every_step_extended_20260913")
    parser.add_argument("--initial-root", type=Path, default=REPO / "outputs/droid_hp_every_step_20260913")
    parser.add_argument("--outcomes", type=Path, default=REPO / "outputs/droid_real_world_20260913/success_counts.json")
    args = parser.parse_args()
    verify(args.output_root, args.outcomes, args.initial_root)
