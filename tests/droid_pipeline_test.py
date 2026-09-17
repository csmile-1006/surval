"""CPU contracts: python -m unittest discover -s tests -p droid_pipeline_test.py."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from surval.cache_io import write_seqcache_hdf5
from surval.droid import (build_policy_thresholds, denormalize_actions, evaluate_policy_caches,
                          load_policy_cache, score_policy_cache)
from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.threshold import LocalThresholdMap
from surval.sequential_validate import _load_cache_hdf5
from surval.tb_aggregate.droid import aggregate_droid_metrics, PROXY_DIRECTIONS, ROW_KEY
from surval.rlds_cache import materialize_validation_rows, row_key, row_keys_digest, validate_row_coverage


def make_cache(path, epoch=5):
    n, horizon = 12, 2
    ids = np.repeat(["source/episode_a", "source/episode_b"], n // 2)
    times = np.tile(np.arange(n // 2), 2)
    physical = np.zeros((n, horizon, 10), np.float32)
    physical[..., 0] = (np.arange(n)[:, None] + np.arange(horizon)) * 0.01
    physical[..., 3] = physical[..., 7] = 1
    scale = np.array([2, 3, 4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.5], np.float32)
    offset = np.linspace(-0.1, 0.1, 10, dtype=np.float32)
    actions = (physical - offset) / scale
    features = np.random.default_rng(epoch).normal(size=(n, 8)).astype(np.float32)
    provenance = {"feature_source": "ema.policy.obs_encoder", "observation_horizon": 2,
                  "dataset_name": "test/droid/val5", "run": "test_run", "run_timestamp": "123"}
    write_seqcache_hdf5(str(path), demo_ids=ids, index_in_demo=times, actions=actions,
                       pred_actions_list=[actions.copy(), actions.copy()], checkpoint=f"policy_{epoch}.pth",
                       epoch=epoch, obs_features=features, val_loss=0.2, off_manifold_norms=0.3)
    with h5py.File(path, "a") as f:
        f.attrs.update({"complete": 1, "droid_cache_version": 1,
                        "action_space": "normalized_checkpoint", "action_start_offset": 1,
                        "feature_source": provenance["feature_source"], "provenance": json.dumps(provenance),
                        "action_scale": scale, "action_offset": offset})
    return load_policy_cache(path), physical


def make_thresholds(root, cache):
    root.mkdir()
    names, inverse = np.unique(cache["demo_ids"], return_inverse=True)
    np.save(root / "demo_id_int.npy", inverse)
    np.save(root / "t.npy", cache["index_in_demo"])
    (root / "demo_id_str.json").write_text(json.dumps({str(i): n for i, n in enumerate(names)}))
    n = len(inverse)
    scales = np.array([[0.02], [0.5]], np.float32)
    LocalThresholdMap(
        thresholds=np.broadcast_to(scales, (n, 2, 1)).copy(), fallback_used=np.zeros(n, bool),
        n_neighbors_used=np.full(n, 5), block_names=("pos", "rot_6d"), quantiles=(0.95,),
        global_thresholds=scales, config_hash="test", block_types={"rot_6d": "rot6d"},
    ).save(str(root / "thresholds"))


class DroidCacheContract(unittest.TestCase):
    def test_epoch_roundtrip_and_physical_actions(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seqcache_epoch_000005.hdf5"
            cache, physical = make_cache(path)
            self.assertEqual(cache["epoch"], 5)
            np.testing.assert_allclose(cache["actions"], physical, atol=1e-7)
            self.assertEqual(cache["obs_features"].shape, (12, 8))
            self.assertEqual(cache["valid_loss"], 0.2)
            self.assertEqual(cache["valid_omn"], 0.3)

    def test_writer_preserves_step_and_rejects_ambiguous_time(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "step.hdf5")
            actions = np.zeros((1, 2, 10), np.float32)
            kw = dict(demo_ids=["a"], index_in_demo=[0], actions=actions,
                      pred_actions_list=[actions], checkpoint="x")
            write_seqcache_hdf5(path, step=9, **kw)
            self.assertEqual(_load_cache_hdf5(path)[6], 9)
            with self.assertRaises(ValueError):
                write_seqcache_hdf5(path, step=9, epoch=9, **kw)
            with self.assertRaises(ValueError):
                denormalize_actions(actions, np.zeros(10), np.zeros(10))

    def test_continuous_score_matches_formula_and_perfect_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            cache, _ = make_cache(Path(td) / "seqcache_epoch_000005.hdf5")
            root = Path(td) / "thresholds"
            make_thresholds(root, cache)
            perfect, _ = score_policy_cache(cache, str(root))
            self.assertAlmostEqual(perfect["PrefixSurvival_Score"], 1.0)
            cache["pred_actions"][..., 0] += 0.04
            score, _ = score_policy_cache(cache, str(root))
            # L2/scale = 2 for pos, 0 for rot. Library gate -> logsumexp -> cumprod.
            probabilities = np.array([np.exp(-1), 1.0])
            survival = -np.log(np.exp(-probabilities).mean())
            expected = np.mean(survival ** np.arange(1, 4))
            self.assertAlmostEqual(score["PrefixSurvival_Score"], expected, places=5)

    def test_missing_threshold_row_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            cache, _ = make_cache(Path(td) / "seqcache_epoch_000005.hdf5")
            root = Path(td) / "thresholds"
            make_thresholds(root, cache)
            times = cache["index_in_demo"].copy()
            times[0] = 999
            np.save(root / "t.npy", times)
            with self.assertRaises(ValueError):
                score_policy_cache(cache, str(root))

    def test_score_is_invariant_to_cached_row_order(self):
        with tempfile.TemporaryDirectory() as td:
            cache, _ = make_cache(Path(td) / "seqcache_epoch_000005.hdf5")
            root = Path(td) / "thresholds"
            make_thresholds(root, cache)
            cache["pred_actions"][..., 0] += np.linspace(0, 0.1, 12)[None, :, None]
            expected, episodes = score_policy_cache(cache, str(root))
            order = np.random.default_rng(17).permutation(12)
            shuffled = {**cache, **{name: cache[name][order] for name in
                        ("demo_ids", "index_in_demo", "actions", "obs_features")},
                        "pred_actions": cache["pred_actions"][:, order]}
            actual, shuffled_episodes = score_policy_cache(shuffled, str(root))
            self.assertAlmostEqual(actual["PrefixSurvival_Score"], expected["PrefixSurvival_Score"])
            self.assertEqual(episodes, shuffled_episodes)

    def test_legacy_cache_requires_regeneration(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seqcache_epoch_000005.hdf5"
            make_cache(path)
            with h5py.File(path, "a") as f:
                del f.attrs["droid_cache_version"]
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                load_policy_cache(path)

    def test_reader_rejects_an_entire_missing_episode_from_v2_cache(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seqcache_epoch_000005.hdf5"
            cache, _ = make_cache(path)
            p = {**cache["provenance"], "producer": "robomimic_diffusion_policy", "producer_version": 2, "row_coverage": {
                "num_rows": 12, "num_demos": 2,
                "row_keys_sha256": row_keys_digest(list(zip(cache["demo_ids"].tolist(), cache["index_in_demo"].tolist())))}}
            with h5py.File(path, "a") as f:
                f.attrs["provenance"] = json.dumps(p)
            load_policy_cache(path)
            with h5py.File(path, "a") as f:
                f.attrs["action_start_offset"] = 0
            with self.assertRaisesRegex(ValueError, "Robomimic DROID"):
                load_policy_cache(path)
            with h5py.File(path, "a") as f:
                f.attrs["action_start_offset"] = 1
                del f["data"][list(f["data"])[-1]]
            with self.assertRaisesRegex(ValueError, "coverage manifest"):
                load_policy_cache(path)

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS runtime dependency unavailable")
    def test_real_threshold_build_reuse_and_checkpoint_results(self):
        with tempfile.TemporaryDirectory() as td:
            caches = Path(td) / "caches"
            caches.mkdir()
            p = caches / "seqcache_epoch_000005.hdf5"
            cache, _ = make_cache(p)
            cfg = LocalThresholdConfig.for_droid_action_space(
                "pos_rot6d", k_neighbors=50, min_neighbors_for_local=2,
                temporal_exclusion_radius=0, quantiles=(0.95,),
            )
            root = Path(td) / "thresholds"
            first = build_policy_thresholds(p, cache, root, cfg)
            self.assertEqual(first, build_policy_thresholds(p, cache, root, cfg))
            np.testing.assert_allclose(np.load(Path(first) / "embeddings.npy"),
                                       cache["obs_features"] / np.linalg.norm(cache["obs_features"], axis=1, keepdims=True))
            make_cache(caches / "seqcache_epoch_000010.hdf5", epoch=10)
            result = evaluate_policy_caches(caches, Path(td) / "results", k_neighbors=50,
                                             min_neighbors=2, temporal_radius=0)
            self.assertEqual([r["epoch"] for r in result["results"]], [5, 10])
            self.assertNotEqual(result["results"][0]["state_db_dir"], result["results"][1]["state_db_dir"])
            for row in result["results"]:
                self.assertAlmostEqual(row["MSE_mean_pred"], 0)
                self.assertAlmostEqual(row["surval_score"], 1)
                self.assertEqual(row["summary"]["LocalThreshold_MissingRows"], 0)


class DroidAggregationContract(unittest.TestCase):
    def test_baseline_direction_keyed_join_coverage_and_macro(self):
        with tempfile.TemporaryDirectory() as td:
            metrics, outcomes = [], []
            for timestamp in ("run_a", "run_b"):
                for epoch, outcome, score in ((15, 0.4, 0.3), (5, 0.2, 0.1), (10, 0.8, 0.9)):
                    key = dict(zip(ROW_KEY, ("test", timestamp, "task/val5", epoch)))
                    row = {**key, **{name: score if direction > 0 else 1-score
                                     for name, direction in PROXY_DIRECTIONS.items()}}
                    if timestamp == "run_a":
                        row["MSE_mean_pred"] = None
                    metrics.append(row)
                    outcomes.append({**key, "outcome": None if timestamp == "run_b" and epoch == 10 else outcome})
            for name, rows in (("metrics.csv", metrics), ("outcomes.csv", list(reversed(outcomes)))):
                with open(Path(td) / name, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            result = aggregate_droid_metrics(Path(td) / "metrics.csv", Path(td) / "outcomes.csv",
                                             Path(td) / "aggregate.json")
            # 4 proxies x 2 runs, minus the one run_a/MSE_mean_pred group that
            # has no values at all.
            self.assertEqual(len(result["per_group"]), 7)
            self.assertEqual(len(result["skipped"]), 1)
            self.assertEqual(result["num_unlabelled_checkpoints"], 1)
            for row in result["per_group"]:
                self.assertAlmostEqual(row["metrics"]["spearman"], 1)
                self.assertEqual(row["metrics"]["nregret"], 0)
                self.assertEqual(row["metrics"]["mmrv"], 0)
                self.assertEqual(row["metrics"]["hit@1"], 1)
                self.assertEqual(row["selected_epoch"], 10 if row["run_timestamp"] == "run_a" else 15)
                if row["run_timestamp"] == "run_b":
                    self.assertEqual(row["omitted_epochs"], [10])
                    self.assertIsNone(row["metrics"]["delta_spearman"])
            for row in result["aggregate"]:
                self.assertEqual(row["n_groups"], 1 if row["method"] == "MSE_mean_pred" else 2)
            self.assertEqual(json.loads((Path(td) / "aggregate.json").read_text())["per_group"], result["per_group"])

    def test_duplicate_unknown_nonfinite_keys_and_input_overwrite_fail(self):
        with tempfile.TemporaryDirectory() as td:
            metric_path, outcome_path = Path(td) / "metrics.csv", Path(td) / "outcomes.csv"
            key = dict(zip(ROW_KEY, ("run", "123", "task", 5)))
            row = {**key, **dict.fromkeys(PROXY_DIRECTIONS, 0.2)}
            with metric_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            for bad_rows, message in (([{**key, "outcome": 0.1}] * 2, "Duplicate"),
                                      ([{**key, "epoch": 6, "outcome": 0.1}], "absent"),
                                      ([{**key, "outcome": "nan"}], "Non-finite")):
                with outcome_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[*ROW_KEY, "outcome"])
                    writer.writeheader()
                    writer.writerows(bad_rows)
                with self.assertRaisesRegex(ValueError, message):
                    aggregate_droid_metrics(metric_path, outcome_path, Path(td) / "out.json")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                aggregate_droid_metrics(metric_path, outcome_path, outcome_path)


class DroidLoaderContract(unittest.TestCase):
    def test_coverage_rejects_missing_tail_episode_duplicates_and_order(self):
        expected = [(source, t) for source in ("a", "b") for t in range(3)]
        for actual in (expected[:-1], expected[:3], expected + [expected[0]],
                       expected[:-1] + [("c", 0)]):
            with self.assertRaises(ValueError):
                validate_row_coverage(expected, actual)
        validate_row_coverage(expected, expected[::-1])
        with self.assertRaises(ValueError):
            validate_row_coverage(expected, expected[::-1], require_order=True)

    def test_parallel_decode_preserves_windows_and_canonical_subsets(self):
        import tensorflow as tf
        from types import SimpleNamespace
        import dlimp as dl
        from octo.data.dataset import episode_row_metadata, apply_trajectory_transforms
        tf.config.set_visible_devices([], "GPU")
        image = tf.io.encode_jpeg(tf.zeros((4, 4, 3), tf.uint8))
        data = {"steps": {
            "action": tf.reshape(tf.range(120, dtype=tf.float32), (2, 6, 10)),
            "observation": {"image_primary": tf.fill((2, 6), image)},
            "task": {"language_instruction": tf.fill((2, 6), "move")},
            "absolute_action_mask": tf.ones((2, 6, 10), tf.bool)},
            "episode_metadata": {"file_path": tf.constant(["b", "a"])}}
        builder = SimpleNamespace(as_dataset=lambda **kwargs: tf.data.Dataset.from_tensor_slices(data))
        def dataset():
            ds = dl.DLataset.from_rlds(builder, shuffle=False, num_parallel_reads=1)
            ds = ds.traj_map(lambda t: {**t, **episode_row_metadata(t, "test", require_source_id=True)},
                             num_parallel_calls=1)
            return apply_trajectory_transforms(ds, train=False, window_size=2,
                                               future_action_window_size=3, num_parallel_calls=1)
        identity = lambda row: row
        kw = {"resize_size": {"primary": (4, 4)}}
        serial = materialize_validation_rows(dataset(), kw, identity, parallelism=1)
        parallel = materialize_validation_rows(dataset(), kw, identity, parallelism=4)
        self.assertEqual(serial["coverage"], parallel["coverage"])
        self.assertEqual(serial["row_keys"], [(s, t) for s in ("a", "b") for t in range(6)])
        for a, b in zip(serial["dataset"], parallel["dataset"]):
            for x, y in zip(tf.nest.flatten(a), tf.nest.flatten(b)):
                np.testing.assert_array_equal(x, y)
        for workers in (1, 4):
            subset = materialize_validation_rows(dataset(), kw, identity, parallelism=workers,
                                                 max_demos=1, max_rows=3)
            self.assertEqual(subset["row_keys"], [("a", t) for t in range(3)])
            np.testing.assert_array_equal(subset["dataset"][-1]["action"], serial["dataset"][2]["action"])
        corrupt = lambda row: {**row, "index_in_demo": row["index_in_demo"] + 1}
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            materialize_validation_rows(dataset(), kw, corrupt, parallelism=4)
        with self.assertRaisesRegex(ValueError, "file_path"):
            episode_row_metadata({"action": tf.zeros((3, 10)), "_traj_index": tf.zeros(3, tf.int64)},
                                 "test", require_source_id=True)


class DroidRuntimeContract(unittest.TestCase):
    """Requires the actual DROID/Octo dependency environment, but uses no GPU/model download."""

    def test_encoder_features_use_inference_ema_and_complete_history(self):
        import torch
        from types import SimpleNamespace
        from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

        class Encoder(torch.nn.Module):
            def __init__(self, shift):
                super().__init__()
                self.shift = shift

            def forward(self, obs):
                return obs["value"] + self.shift

        def networks(shift):
            return torch.nn.ModuleDict({"policy": torch.nn.ModuleDict({
                "obs_encoder": torch.nn.DataParallel(Encoder(shift), device_ids=[])
            })})

        model = DiffusionPolicyUNet.__new__(DiffusionPolicyUNet)
        model.nets = networks(1)
        model.ema = SimpleNamespace(averaged_model=networks(5))
        model.set_eval()
        obs = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        features = model.get_state_features({"value": obs})
        torch.testing.assert_close(features, (obs + 5).reshape(1, 6))
        self.assertFalse(model.nets.training)
        self.assertFalse(model.ema.averaged_model.training)

    def test_rlds_identity_and_trained_action_alignment(self):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        import torch
        from types import SimpleNamespace
        from octo.data.dataset import episode_row_metadata
        from octo.data.traj_transforms import chunk_act_obs
        from robomimic.utils.rlds_utils import robomimic_transform
        from robomimic.scripts.sequential_cache_checkpoints import _get_pred_and_gt_chunks

        length = 6
        traj = {
            "action": tf.repeat(tf.range(length, dtype=tf.float32)[:, None] / 10, 10, axis=1),
            "observation": {"image_primary": tf.zeros((length, 2, 2, 3)),
                            "image_secondary": tf.zeros((length, 2, 2, 3)),
                            "proprio": tf.zeros((length, 7))},
            "task": {"language_instruction": tf.repeat("move", length)},
            "traj_metadata": {"episode_metadata": {"file_path": tf.repeat("source/demo.hdf", length)}},
            "_frame_index": tf.range(length), "_traj_index": tf.repeat(13, length),
            "absolute_action_mask": tf.ones((length, 10), tf.bool),
        }
        traj.update(episode_row_metadata(traj, "test"))
        chunked = chunk_act_obs(traj, window_size=2, future_action_window_size=3)
        rows = [robomimic_transform(tf.nest.map_structure(lambda v: v[t], chunked)) for t in (2, 5)]
        actions = torch.from_numpy(np.stack([r["actions"].numpy() for r in rows]))
        calls = []
        model = SimpleNamespace(algo_config=SimpleNamespace(horizon=SimpleNamespace(observation_horizon=2)))
        def predict(**kwargs):
            calls.append(kwargs)
            return torch.tensor([[[0.3] * 10, [0.4] * 10], [[0.5] * 10, [0.5] * 10]])
        model.get_action = predict
        pred, gt = _get_pred_and_gt_chunks(model, {"obs": {}, "actions": actions}, Ta=2)
        np.testing.assert_allclose(gt, pred)
        self.assertFalse(calls[0]["eval_mode"])
        self.assertEqual(rows[0]["demo_id"].numpy(), b"source/demo.hdf")
        self.assertEqual(int(rows[0]["index_in_demo"]), 2)


if __name__ == "__main__":
    unittest.main()
