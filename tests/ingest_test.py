"""Unit tests for surval.ingest (chunking, builder, readers)."""

from __future__ import annotations

import numpy as np
import pytest

from surval.cache_io import write_seqcache_hdf5  # noqa: F401  (schema sibling)
from surval.ingest import (
    Episode,
    EpisodeReader,
    build_seqcache,
    chunk_actions,
)
from surval.ingest.builder import _resolve_features
from surval.sequential_validate import _load_cache_hdf5


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def test_chunk_actions_full_windows():
    actions = np.arange(10 * 2).reshape(10, 2).astype(np.float32)
    starts, chunks = chunk_actions(actions, horizon=3, pad_mode="drop")
    # frames 0..7 give full length-3 windows (8 -> end 10 ok); 8,9 dropped
    assert starts.tolist() == list(range(8))
    assert chunks.shape == (8, 3, 2)
    # first window is rows 0,1,2
    np.testing.assert_array_equal(chunks[0], actions[0:3])


def test_chunk_actions_repeat_last_pads_tail():
    actions = np.arange(5).reshape(5, 1).astype(np.float32)
    starts, chunks = chunk_actions(actions, horizon=3, pad_mode="repeat_last")
    assert starts.tolist() == list(range(5))  # every frame kept
    # last window starts at frame 4: [4, 4, 4] (repeat last)
    np.testing.assert_array_equal(chunks[-1][:, 0], [4, 4, 4])


def test_chunk_actions_stride():
    actions = np.zeros((10, 1), dtype=np.float32)
    starts, _ = chunk_actions(actions, horizon=2, stride=3, pad_mode="repeat_last")
    assert starts.tolist() == [0, 3, 6, 9]


def test_chunk_actions_validates_args():
    with pytest.raises(ValueError):
        chunk_actions(np.zeros((4, 2)), horizon=0)
    with pytest.raises(ValueError):
        chunk_actions(np.zeros((4,)), horizon=2)  # not 2D


# ---------------------------------------------------------------------------
# in-memory reader used to drive the builder
# ---------------------------------------------------------------------------


class _ListReader(EpisodeReader):
    def __init__(self, episodes):
        self._eps = episodes

    def __iter__(self):
        return iter(self._eps)

    def __len__(self):
        return len(self._eps)


def _make_episode(demo_id, t_ep, a_dim=4, f_dim=6, with_state=True):
    rng = np.random.default_rng(abs(hash(demo_id)) % (2**32))
    actions = rng.standard_normal((t_ep, a_dim)).astype(np.float32)
    obs = {"state": rng.standard_normal((t_ep, f_dim)).astype(np.float32)}
    state = obs["state"] if with_state else None
    return Episode(demo_id=demo_id, actions=actions, obs=obs, state=state)


# ---------------------------------------------------------------------------
# builder end-to-end -> canonical cache round-trip
# ---------------------------------------------------------------------------


def test_build_seqcache_roundtrip_with_feature_fn(tmp_path):
    eps = [_make_episode("demo_0", 12), _make_episode("demo_1", 9, with_state=False)]
    reader = _ListReader(eps)

    horizon, n_samples, f_dim = 3, 2, 5

    def predict_fn(obs_batch):
        b = obs_batch["state"].shape[0]
        # [S, B, T_h, A]
        return np.zeros((n_samples, b, horizon, 4), dtype=np.float32)

    def feature_fn(obs_batch):
        b = obs_batch["state"].shape[0]
        return np.ones((b, f_dim), dtype=np.float32)

    out = tmp_path / "seqcache_step_000042.hdf5"
    summary = build_seqcache(
        str(out),
        reader,
        predict_fn=predict_fn,
        feature_fn=feature_fn,
        horizon=horizon,
        num_samples=n_samples,
        step=42,
        pad_mode="drop",
        checkpoint="ckpt-42",
    )

    # demo_0: 12 frames -> 10 full windows; demo_1: 9 -> 7 windows
    assert summary["num_rows"] == 10 + 7
    assert summary["num_demos"] == 2
    assert summary["ac_dim"] == 4
    assert summary["num_samples"] == n_samples
    assert summary["obs_feat_dim"] == f_dim

    (
        demo_ids,
        index_in_demo,
        actions,
        pred_samples,
        obs_features,
        checkpoint,
        step,
        _valid_loss,
        _valid_off,
    ) = _load_cache_hdf5(str(out), load_obs_features=True)

    assert actions.shape == (17, horizon, 4)
    assert pred_samples.shape == (n_samples, 17, horizon, 4)
    assert obs_features.shape == (17, f_dim)
    assert step == 42
    assert checkpoint == "ckpt-42"
    assert set(np.unique(demo_ids)) == {"demo_0", "demo_1"}


def test_build_seqcache_state_fallback(tmp_path):
    eps = [_make_episode("demo_0", 8)]  # has state
    reader = _ListReader(eps)

    def predict_fn(obs_batch):
        b = obs_batch["state"].shape[0]
        return np.zeros((b, 2, 4), dtype=np.float32)  # 3D -> auto-wrap S=1

    out = tmp_path / "seqcache_step_000000.hdf5"
    summary = build_seqcache(
        str(out), reader, predict_fn=predict_fn, horizon=2, step=0, pad_mode="drop"
    )
    # no feature_fn -> falls back to Episode.state (f_dim=6)
    assert summary["obs_feat_dim"] == 6
    assert summary["num_samples"] == 1


def test_build_seqcache_requires_obs_features(tmp_path):
    eps = [_make_episode("demo_0", 8, with_state=False)]  # no state
    reader = _ListReader(eps)

    def predict_fn(obs_batch):
        b = obs_batch["state"].shape[0]
        return np.zeros((b, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="obs_features is required"):
        build_seqcache(
            str(tmp_path / "x.hdf5"), reader, predict_fn=predict_fn, horizon=2, step=0
        )


def test_build_seqcache_rejects_bad_predict_shape(tmp_path):
    reader = _ListReader([_make_episode("demo_0", 8)])

    def predict_fn(obs_batch):
        b = obs_batch["state"].shape[0]
        return np.zeros((b, 99, 4), dtype=np.float32)  # wrong horizon

    with pytest.raises(ValueError, match="inconsistent with batch"):
        build_seqcache(
            str(tmp_path / "x.hdf5"), reader, predict_fn=predict_fn, horizon=2, step=0,
            pad_mode="drop",
        )


def test_resolve_features_feature_fn_shape_guard():
    ep = _make_episode("demo_0", 4)
    starts = np.array([0, 1])
    with pytest.raises(ValueError, match="feature_fn must return"):
        _resolve_features(lambda ob: np.zeros((1, 3)), {"state": ep.obs["state"][starts]}, ep, starts)


# ---------------------------------------------------------------------------
# Episode validation
# ---------------------------------------------------------------------------


def test_episode_rejects_misaligned_obs():
    with pytest.raises(ValueError):
        Episode(demo_id="d", actions=np.zeros((5, 2)), obs={"x": np.zeros((4, 3))})
