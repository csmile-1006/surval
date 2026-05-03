"""Tests for StateDatabase: build, self-query, neighbor filtering."""

import importlib.util

import numpy as np
import pytest

from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.database import StateDatabase
from surval.local_threshold.database import StateRecords

pytestmark = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")


def _make_synth_db(n=50, d=8, n_demos=5, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    actions = rng.normal(size=(n, 8)).astype(np.float32)
    demo_id_int = (np.arange(n) // (n // n_demos)).astype(np.int32)
    t = np.zeros(n, dtype=np.int32)
    for d_id in range(n_demos):
        mask = demo_id_int == d_id
        t[mask] = np.arange(mask.sum(), dtype=np.int32)
    rec = StateRecords(
        actions=actions,
        demo_id_int=demo_id_int,
        t=t,
        demo_id_str_by_int={i: f"demo_{i}" for i in range(n_demos)},
    )
    cfg = LocalThresholdConfig()
    db = StateDatabase(cfg)
    db.attach(emb, rec)
    return db, emb, rec


def test_self_query_distance_zero():
    db, emb, _ = _make_synth_db()
    distances, indices = db.query(emb, k=1)
    # Top-1 of every state under FAISS Flat IP is itself.
    assert np.array_equal(indices[:, 0], np.arange(emb.shape[0]))
    assert np.allclose(distances[:, 0], 0.0, atol=1e-5)


def test_filter_excludes_self_and_temporal_window():
    db, emb, rec = _make_synth_db()
    cfg = LocalThresholdConfig(temporal_exclusion_radius=2, same_demo_allowed=True)
    distances, indices = db.query(emb, k=10)
    filtered = db.filter_neighbors(indices, rec.demo_id_int, rec.t, cfg)
    for q in range(emb.shape[0]):
        for nb in filtered[q]:
            same_demo = rec.demo_id_int[nb] == rec.demo_id_int[q]
            if same_demo:
                assert abs(int(rec.t[nb]) - int(rec.t[q])) > cfg.temporal_exclusion_radius
            # Self must always be excluded.
            assert not (same_demo and rec.t[nb] == rec.t[q])


def test_filter_excludes_same_demo_when_disallowed():
    db, emb, rec = _make_synth_db()
    cfg = LocalThresholdConfig(same_demo_allowed=False)
    _, indices = db.query(emb, k=20)
    filtered = db.filter_neighbors(indices, rec.demo_id_int, rec.t, cfg)
    for q in range(emb.shape[0]):
        for nb in filtered[q]:
            assert rec.demo_id_int[nb] != rec.demo_id_int[q]
