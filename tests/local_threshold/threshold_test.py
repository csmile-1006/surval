"""Tests for compute_local_thresholds: synthetic case where pairwise distances
are known and the per-quantile output should match np.quantile."""

import importlib.util

import numpy as np
import pytest

from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.database import StateDatabase
from surval.local_threshold.database import StateRecords
from surval.local_threshold.threshold import _block_pair_distances
from surval.local_threshold.threshold import compute_global_intra_demo_thresholds_from_db
from surval.local_threshold.threshold import compute_global_thresholds_from_db
from surval.local_threshold.threshold import compute_local_intra_demo_thresholds
from surval.local_threshold.threshold import compute_local_thresholds

pytestmark = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")


def test_block_pair_distances_pairwise_matches_numpy():
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(20, 8)).astype(np.float32)
    sl = slice(0, 1)
    d = _block_pair_distances(actions, sl, pairwise=True)
    # 1-D block: ||a_i - a_j|| == |a_i - a_j|
    diff = np.abs(actions[:, 0:1] - actions[:, 0:1].T)
    expected = diff[np.triu_indices(20, k=1)]
    assert d.shape == expected.shape
    assert np.allclose(np.sort(d), np.sort(expected), atol=1e-5)


def test_block_pair_distances_query_centered():
    rng = np.random.default_rng(1)
    actions = rng.normal(size=(10, 8)).astype(np.float32)
    q = rng.normal(size=(8,)).astype(np.float32)
    d = _block_pair_distances(actions, slice(2, 5), pairwise=False, query_action=q)
    expected = np.linalg.norm(actions[:, 2:5] - q[None, 2:5], axis=-1)
    assert np.allclose(d, expected, atol=1e-5)


def _make_db_for_threshold(n=200, d_state=8, n_demos=10, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d_state)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    actions = rng.normal(scale=0.3, size=(n, 8)).astype(np.float32)
    demo_id_int = (np.arange(n) // (n // n_demos)).astype(np.int32)
    t = np.zeros(n, dtype=np.int32)
    for did in range(n_demos):
        mask = demo_id_int == did
        t[mask] = np.arange(mask.sum(), dtype=np.int32)
    rec = StateRecords(
        actions=actions,
        demo_id_int=demo_id_int,
        t=t,
        demo_id_str_by_int={i: f"demo_{i}" for i in range(n_demos)},
    )
    cfg = LocalThresholdConfig(
        k_neighbors=20,
        min_neighbors_for_local=5,
        temporal_exclusion_radius=0,  # keep most neighbors valid
        same_demo_allowed=True,
        quantiles=(0.5, 0.9, 0.95),
    )
    db = StateDatabase(cfg)
    db.attach(emb, rec)
    return db, cfg


def test_compute_local_thresholds_shape_and_quantile():
    db, cfg = _make_db_for_threshold()
    global_t = compute_global_thresholds_from_db(db, cfg)
    assert global_t.shape == (len(cfg.block_names), len(cfg.quantiles))
    assert (global_t >= 0).all()

    m = compute_local_thresholds(db, cfg, global_t)
    assert m.thresholds.shape == (db.n_states, len(cfg.block_names), len(cfg.quantiles))
    assert m.fallback_used.shape == (db.n_states,)
    # Quantiles should be monotonically non-decreasing along axis -1.
    diffs = np.diff(m.thresholds, axis=-1)
    assert (diffs >= -1e-6).all(), "per-quantile thresholds must be non-decreasing"


def test_intra_demo_sc_thresholds_match_known_chunk_motion():
    """For a deterministic linear-in-t demo with action[t] = t·v, every valid
    chunk-motion delta is exactly ``ta·v``. The pooled per-block threshold
    should equal ``|ta·v[block]|`` regardless of the quantile."""
    rng = np.random.default_rng(123)
    n_demos = 8
    per_demo = 25
    n = n_demos * per_demo
    d_state = 8
    emb = rng.normal(size=(n, d_state)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)

    v = np.array([0.1, 0.2, -0.3, 0.05, -0.15, 0.4, -0.25, 0.0], dtype=np.float32)
    actions = np.zeros((n, 8), dtype=np.float32)
    demo_id_int = np.zeros(n, dtype=np.int32)
    t_arr = np.zeros(n, dtype=np.int32)
    for d in range(n_demos):
        for s in range(per_demo):
            i = d * per_demo + s
            demo_id_int[i] = d
            t_arr[i] = s
            actions[i] = s * v

    rec = StateRecords(
        actions=actions,
        demo_id_int=demo_id_int,
        t=t_arr,
        demo_id_str_by_int={i: f"demo_{i}" for i in range(n_demos)},
    )
    cfg = LocalThresholdConfig(
        k_neighbors=40,
        min_neighbors_for_local=3,
        temporal_exclusion_radius=0,
        same_demo_allowed=True,
        quantiles=(0.5, 0.9, 0.99),
        scale_source="intra_demo_sc",
        ta=4,
    )
    db = StateDatabase(cfg)
    db.attach(emb, rec)

    g = compute_global_intra_demo_thresholds_from_db(db, cfg)
    assert g.shape == (len(cfg.block_names), len(cfg.quantiles))
    expected = np.array([abs(cfg.ta * float(v[i])) for i in range(7)], dtype=np.float32)
    # All quantiles equal because every chunk delta is identical.
    for qi in range(len(cfg.quantiles)):
        assert np.allclose(g[:, qi], expected, atol=1e-5), (g[:, qi], expected)

    m = compute_local_intra_demo_thresholds(db, cfg, g)
    assert m.thresholds.shape == (n, len(cfg.block_names), len(cfg.quantiles))
    # Local thresholds match expected wherever the local pass found enough
    # samples (i.e. did not fall back).
    for s in range(n):
        if not bool(m.fallback_used[s]):
            for qi in range(len(cfg.quantiles)):
                assert np.allclose(m.thresholds[s, :, qi], expected, atol=1e-5)


def test_intra_demo_sc_falls_back_when_demo_too_short():
    """If ta exceeds demo length, no chunk deltas exist and every state must
    fall back to the supplied global thresholds."""
    rng = np.random.default_rng(0)
    n = 30
    n_demos = 10
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    actions = rng.normal(size=(n, 8)).astype(np.float32)
    demo_id_int = (np.arange(n) // (n // n_demos)).astype(np.int32)
    t_arr = np.zeros(n, dtype=np.int32)
    for d in range(n_demos):
        mask = demo_id_int == d
        t_arr[mask] = np.arange(mask.sum(), dtype=np.int32)
    rec = StateRecords(
        actions=actions,
        demo_id_int=demo_id_int,
        t=t_arr,
        demo_id_str_by_int={i: f"demo_{i}" for i in range(n_demos)},
    )
    cfg = LocalThresholdConfig(
        k_neighbors=10,
        min_neighbors_for_local=1,
        temporal_exclusion_radius=0,
        same_demo_allowed=True,
        quantiles=(0.9,),
        scale_source="intra_demo_sc",
        ta=999,  # way past any demo length → zero valid samples
    )
    db = StateDatabase(cfg)
    db.attach(emb, rec)
    g = np.full((len(cfg.block_names), 1), 0.7, dtype=np.float32)
    m = compute_local_intra_demo_thresholds(db, cfg, g)
    assert m.fallback_used.all()
    assert np.allclose(m.thresholds, 0.7)


def test_fallback_when_too_few_neighbors():
    db, _ = _make_db_for_threshold(n=50, n_demos=2)
    # Force min_neighbors very high so almost everything falls back.
    cfg = LocalThresholdConfig(
        k_neighbors=5,
        min_neighbors_for_local=1000,
        temporal_exclusion_radius=0,
        same_demo_allowed=True,
        quantiles=(0.9,),
    )
    db.cfg = cfg  # keep records, swap cfg for query buffer
    global_t = np.full((len(cfg.block_names), len(cfg.quantiles)), 0.42, dtype=np.float32)
    m = compute_local_thresholds(db, cfg, global_t)
    assert m.fallback_used.all()
    assert np.allclose(m.thresholds, 0.42)
