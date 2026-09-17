"""Run: PYTHONPATH=src:tests .venv-droid/bin/python -m unittest droid_tuning_test -q."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import spearmanr

from surval.droid_tuning import (canonical_score, canonical_thresholds, expert_distance_matrices,
                                hp_grid, local_scale_table, primary_metrics, score_grid)
from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.database import StateDatabase, StateRecords
from surval.local_threshold.threshold import query_threshold_neighbors


def synthetic():
    rng = np.random.default_rng(42)
    n = 48
    actions = rng.normal(size=(n, 8, 10)).astype(np.float32) * 0.1
    angles = rng.normal(size=(n, 8)).astype(np.float32) * 0.5
    actions[..., 3:9] = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles),
                                  -np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=-1)
    ids, times = np.repeat(np.arange(6), 8), np.tile(np.arange(8), 6)
    features = rng.normal(size=(n, 16)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    cfg = LocalThresholdConfig.for_droid_action_space("pos_rot6d", use_proprio=False)
    db = StateDatabase(cfg)
    db.attach(features, StateRecords(actions[:, 0], ids, times, {i: str(i) for i in range(6)}))
    cache = {"actions": actions, "pred_actions": actions[None] +
             rng.normal(size=(8, n, 8, 10)).astype(np.float32) * 0.2,
             "demo_ids": ids.astype(str), "index_in_demo": times, "droid_action_space": "pos_rot6d",
             "valid_loss": 0.1, "valid_omn": 0.2}
    return db, cache


class TuningContract(unittest.TestCase):
    def test_exact_neighbors_scales_scores_and_order(self):
        db, cache = synthetic()
        cfg = replace(db.cfg, k_neighbors=10)
        exact = query_threshold_neighbors(db, cfg, exact_neighbors=True)
        legacy = query_threshold_neighbors(db, cfg)
        self.assertTrue(all(len(row) == 10 for row in exact))
        self.assertTrue(any(len(row) > 10 for row in legacy))
        for row, buffered in zip(exact, legacy):
            np.testing.assert_array_equal(row, buffered[:10])
        other = query_threshold_neighbors(db, replace(cfg, same_demo_allowed=False), exact_neighbors=True)
        self.assertTrue(all(len(row) == 10 for row in other))
        for i, row in enumerate(other):
            self.assertTrue(np.all(db.records.demo_id_int[row] != db.records.demo_id_int[i]))
        axes = {"ta": [1, 3, 8], "k": [10, 20], "quantile": [0.1, 0.95],
                "scale_multiplier": [0.25, 1.0, 4.0]}
        matrices = expert_distance_matrices(db)
        tables = {k: local_scale_table(db, matrices, k, axes["quantile"])[0] for k in axes["k"]}
        fast = score_grid(cache, tables, axes)
        with tempfile.TemporaryDirectory() as td:
            canonical = {k: canonical_thresholds(db, axes["quantile"], k, Path(td) / str(k))
                         for k in axes["k"]}
            for k in axes["k"]:
                np.testing.assert_allclose(tables[k], canonical[k].thresholds, rtol=1e-6, atol=1e-7)
            reference = [canonical_score(cache, Path(td) / str(h["k"]), canonical[h["k"]], h)
                         for h in hp_grid(axes)]
            np.testing.assert_allclose(fast, reference, rtol=2e-6, atol=2e-7)
        order = np.random.default_rng(11).permutation(len(cache["actions"]))
        shuffled = {**cache, **{k: cache[k][order] for k in ("actions", "demo_ids", "index_in_demo")},
                    "pred_actions": cache["pred_actions"][:, order]}
        np.testing.assert_allclose(score_grid(shuffled, {k: v[order] for k, v in tables.items()}, axes),
                                   fast, rtol=0, atol=1e-14)

    def test_primary_metrics_and_grid_guards(self):
        actual = np.array([0.2, 0.6, 0.6, 0.8])
        score = np.array([0.8, 0.3, 0.3, 0.1])
        m = primary_metrics(actual, score)
        self.assertAlmostEqual(m["nregret"], 1)
        self.assertAlmostEqual(m["spearman"], spearmanr(actual, score).statistic)
        expected = np.mean([max(abs(a - b) if ((a < b) != (x < y)) else 0
                                for b, y in zip(actual, score)) for a, x in zip(actual, score)])
        self.assertAlmostEqual(m["mmrv"], expected)
        self.assertFalse(primary_metrics(actual, np.ones(4))["eligible"])
        self.assertEqual(len(hp_grid()), 3584)
        with self.assertRaises(ValueError):
            hp_grid({"ta": [9], "k": [10], "quantile": [0.95], "scale_multiplier": [1]})


if __name__ == "__main__":
    unittest.main()
