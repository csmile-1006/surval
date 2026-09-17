"""CPU checks: PYTHONPATH=src:tests python -m unittest droid_dino_test -q."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from droid_pipeline_test import make_cache
from surval.cache_io import project_neighbor_actions, compute_off_manifold_errors
from surval.droid import evaluate_policy_caches
from surval.droid_dino import (DINO_OMN, align_dino_cache, compute_dino_omn, dataset_db_path,
                               dino_neighbors, encode_droid_rows, load_shared_dino_db,
                               save_shared_dino_db)
from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.database import StateDatabase, StateRecords
from surval.tb_aggregate.droid import aggregate_droid_metrics, ROW_KEY


def make_db(cache):
    cfg = LocalThresholdConfig.for_droid_action_space(
        "pos_rot6d", use_proprio=False, state_window_size=2, dataset_name="test/droid/val5")
    names, inverse = np.unique(cache["demo_ids"], return_inverse=True)
    features = np.random.default_rng(123).normal(size=(len(inverse), 8)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    db = StateDatabase(cfg)
    db.attach(features, StateRecords(cache["actions"][:, 0], inverse.astype(np.int32),
                                    cache["index_in_demo"], dict(enumerate(names))))
    meta = {"dataset_name": "test/droid/val5", "split": "val", "observation_horizon": 2,
            "action_start_offset": 1, "droid_action_space": "pos_rot6d"}
    return db, cache["actions"].copy(), meta


class SharedDinoContract(unittest.TestCase):
    def test_projection_and_legacy_retrieval_match(self):
        pred = np.array([[3., 4., 5.]])
        experts = np.array([[[1., 0., 0.], [0., 1., 0.]]])
        np.testing.assert_allclose(project_neighbor_actions(pred, experts), [5.])
        np.testing.assert_allclose(project_neighbor_actions(pred, experts[:, :1]), [np.sqrt(45)])
        features = np.array([[0., 0.], [1., 0.], [3., 0.]])
        actions = np.array([[1., 0.], [0., 1.], [1., 1.]])
        np.testing.assert_allclose(compute_off_manifold_errors(actions, features, actions, k=1),
                                   project_neighbor_actions(actions, actions[[1, 0, 1], None]))

    def test_alignment_rejects_wrong_rows_gt_or_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            cache, _ = make_cache(Path(td) / "seqcache_epoch_000005.hdf5")
            db, chunks, meta = make_db(cache)
            order = np.arange(12)[::-1]
            shuffled = copy.deepcopy(cache)
            for key in ("demo_ids", "index_in_demo", "actions", "obs_features"):
                shuffled[key] = cache[key][order]
            shuffled["pred_actions"] = cache["pred_actions"][:, order]
            np.testing.assert_array_equal(align_dino_cache(db, chunks, meta, shuffled), order)
            for mutate in (lambda c: c["index_in_demo"].__setitem__(0, 999),
                           lambda c: c["actions"].__setitem__((0, 0, 0), 99),
                           lambda c: c.__setitem__("action_start_offset", 0),
                           lambda c: c["provenance"].__setitem__("dataset_name", "other")):
                bad = copy.deepcopy(cache)
                mutate(bad)
                with self.assertRaises(ValueError):
                    align_dino_cache(db, chunks, meta, bad)
            with self.assertRaises(ValueError):
                dataset_db_path(td, "../elsewhere")

    def test_full_split_neighbors_offset_and_order_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            cache, _ = make_cache(Path(td) / "seqcache_epoch_000005.hdf5")
            db, chunks, _ = make_db(cache)
            neighbors = dino_neighbors(db, k=3, temporal_radius=5)
            for i, row in enumerate(neighbors):
                self.assertEqual(len(row), 3)
                self.assertTrue(np.all(db.records.demo_id_int[row] != db.records.demo_id_int[i]))
            # K=1 makes the expected same-offset residual unambiguous.
            neighbors = [row[:1] for row in neighbors]
            cache["pred_actions"][1, :, 1, 1] += 0.3
            value, info = compute_dino_omn(cache, chunks, np.arange(12), neighbors)
            normalized_pred = (cache["pred_actions"] - cache["action_offset"]) / cache["action_scale"]
            expert = (chunks[np.array([n[0] for n in neighbors])] - cache["action_offset"]) / cache["action_scale"]
            expected = np.linalg.norm(normalized_pred - expert, axis=-1).mean()
            self.assertAlmostEqual(value, expected, places=7)
            permuted = copy.deepcopy(cache)
            permuted["pred_actions"] = cache["pred_actions"][:, ::-1]
            other, _ = compute_dino_omn(permuted, chunks, np.arange(12)[::-1], neighbors)
            self.assertAlmostEqual(value, other)
            self.assertEqual(info["horizon"], 2)
            neighbors[0] = np.array([], dtype=np.int64)
            value, info = compute_dino_omn(cache, chunks, np.arange(12), neighbors)
            self.assertIsNone(value)
            self.assertEqual(info["rows_without_neighbors"], 1)

    def test_shared_thresholds_and_optional_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache, _ = make_cache(root / "seqcache_epoch_000005.hdf5", epoch=5)
            make_cache(root / "seqcache_epoch_000010.hdf5", epoch=10)
            db, chunks, meta = make_db(cache)
            target = dataset_db_path(root / "dino", meta["dataset_name"])
            save_shared_dino_db(target, db, chunks, meta)
            loaded, gt, manifest = load_shared_dino_db(target)
            align_dino_cache(loaded, gt, manifest, cache)
            result = evaluate_policy_caches(root, root / "scores", dino_db_root=root / "dino",
                                             min_neighbors=2, k_neighbors=3, temporal_radius=0)
            rows = result["results"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["state_db_dir"], rows[1]["state_db_dir"])
            self.assertAlmostEqual(rows[0][DINO_OMN], rows[1][DINO_OMN])
            self.assertEqual(rows[0]["Off_Manifold_Norm"], 0.3)
            reduced = evaluate_policy_caches(root, root / "reduced", dino_db_root=root / "dino",
                                             min_neighbors=2, k_neighbors=3, temporal_radius=0,
                                             ta=1, num_samples=1)["results"][0]
            self.assertAlmostEqual(reduced[DINO_OMN], rows[0][DINO_OMN])
            self.assertEqual((reduced["dino_omn"]["horizon"], reduced["dino_omn"]["num_samples"]), (2, 2))
            outcomes = root / "outcomes.csv"
            with outcomes.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[*ROW_KEY, "outcome"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({**{key: row[key] for key in ROW_KEY}, "outcome": row["epoch"] / 10})
            report = aggregate_droid_metrics(root / "scores/checkpoint_metrics.csv", outcomes, root / "report.json")
            dino = next(r for r in report["per_group"] if r["method"] == DINO_OMN)
            self.assertFalse(dino["higher_is_better"])
            self.assertEqual(dino["num_checkpoints"], 2)
            gt[0, 0, 0] += 1
            np.save(target / "expert_action_chunks.npy", gt)
            with self.assertRaises(ValueError):
                load_shared_dino_db(target)

    def test_encode_once_and_causal_history(self):
        with tempfile.TemporaryDirectory() as td:
            cache, _ = make_cache(Path(td) / "seqcache_epoch_000005.hdf5")
            cfg = LocalThresholdConfig.for_droid_action_space(
                "pos_rot6d", use_proprio=False, image_views=("left", "right"),
                state_window_size=2, encoder_batch_size=4)
            normalized = (cache["actions"] - cache["action_offset"]) / cache["action_scale"]
            rows = []
            for i in range(12):
                rows.append({"demo_id": cache["demo_ids"][i], "index_in_demo": cache["index_in_demo"][i],
                             "actions": np.concatenate([normalized[i, :1], normalized[i]]),
                             "obs": {view: np.full((2, 4, 4, 3), (i + 1) / 20, np.float32)
                                     for view in cfg.image_views}})

            class Encoder:
                num_images = 0

                def encode(self, images):
                    self.num_images += len(images)
                    return np.stack([images.mean(axis=(1, 2, 3)), np.ones(len(images))], axis=1)

            encoder = Encoder()
            db = encode_droid_rows(rows, cache, cfg, encoder)
            self.assertEqual(encoder.num_images, 24)  # not 48 from repeated history
            self.assertEqual(db.embeddings.shape, (12, 8))
            np.testing.assert_allclose(np.linalg.norm(db.embeddings, axis=1), 1, atol=1e-6)
            np.testing.assert_array_equal(db.embeddings[6, :2], db.embeddings[6, 2:4])
            rows[0]["actions"][1, 0] += 1
            with self.assertRaises(ValueError):
                encode_droid_rows(rows, cache, cfg, encoder)


if __name__ == "__main__":
    unittest.main()
