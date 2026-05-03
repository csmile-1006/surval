"""Tests for state vector composition + ProprioNormalizer."""

import os
import tempfile

import numpy as np

from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.state_embed import ProprioNormalizer
from surval.local_threshold.state_embed import build_window_indices
from surval.local_threshold.state_embed import compose_state_vectors_batch
from surval.local_threshold.state_embed import modality_block_norms


def test_proprio_normalizer_round_trip():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.5, size=(500, 14)).astype(np.float32)
    norm = ProprioNormalizer().fit(x)
    z = norm.transform(x)
    assert z.shape == x.shape
    assert z.dtype == np.float32
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-3)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "norm.pkl")
        norm.save(p)
        loaded = ProprioNormalizer.load(p)
    assert np.allclose(loaded.mean, norm.mean)
    assert np.allclose(loaded.std, norm.std)


def test_compose_state_unit_norm():
    cfg = LocalThresholdConfig(
        encoder_name="dinov2_vits14",
        image_views=("image", "wrist_image"),
        proprio_keys=("joint_position", "cartesian_position", "gripper_position"),
    )
    rng = np.random.default_rng(1)
    n, d_img, d_prop = 32, 384, 14
    img_feats = {
        "image": rng.normal(size=(n, d_img)).astype(np.float32),
        "wrist_image": rng.normal(size=(n, d_img)).astype(np.float32),
    }
    prop = rng.normal(size=(n, d_prop)).astype(np.float32)

    state = compose_state_vectors_batch(img_feats, prop, cfg)
    assert state.shape == (n, 2 * d_img + d_prop)
    assert state.dtype == np.float32
    norms = np.linalg.norm(state, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_modality_block_norms_balanced():
    cfg = LocalThresholdConfig(encoder_name="dinov2_vits14")
    rng = np.random.default_rng(2)
    n, d_img = 1000, 384
    img_feats = {
        "image": rng.normal(size=(n, d_img)).astype(np.float32) * 5.0,  # arbitrary scale
        "wrist_image": rng.normal(size=(n, d_img)).astype(np.float32) * 0.1,
    }
    prop = rng.normal(loc=10.0, size=(n, 14)).astype(np.float32) * 100.0
    norms = modality_block_norms(img_feats, prop, cfg)
    # Per-modality renormalization makes each block unit-norm regardless of
    # input scale.
    assert np.allclose(norms["image_block"], 1.0, atol=1e-5)
    assert np.allclose(norms["proprio_block"], 1.0, atol=1e-5)


def test_build_window_indices_clamps_at_demo_start():
    # Two demos: demo 0 has t=[0,1,2,3,4], demo 1 has t=[0,1,2]. State indices
    # are interleaved on purpose to exercise sorting.
    demo = np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=np.int32)
    t = np.array([0, 0, 1, 1, 2, 2, 3, 4], dtype=np.int32)
    win = build_window_indices(demo, t, K=3)
    # demo 0 in temporal order: state idx [0, 2, 4, 6, 7] at t=[0,1,2,3,4]
    # Expected windows (length 3, oldest first, clamped):
    #   t=0 → [0, 0, 0]
    #   t=1 → [0, 0, 2]
    #   t=2 → [0, 2, 4]
    #   t=3 → [2, 4, 6]
    #   t=4 → [4, 6, 7]
    assert win[0].tolist() == [0, 0, 0]
    assert win[2].tolist() == [0, 0, 2]
    assert win[4].tolist() == [0, 2, 4]
    assert win[6].tolist() == [2, 4, 6]
    assert win[7].tolist() == [4, 6, 7]
    # demo 1 in temporal order: [1, 3, 5] at t=[0,1,2]
    assert win[1].tolist() == [1, 1, 1]
    assert win[3].tolist() == [1, 1, 3]
    assert win[5].tolist() == [1, 3, 5]
    # last column is always self
    assert (win[:, -1] == np.arange(len(demo))).all()


def test_compose_state_window_concat_doubles_image_dim():
    cfg = LocalThresholdConfig(
        encoder_name="dinov2_vits14",
        state_window_size=2,
        state_window_aggregation="concat",
    )
    rng = np.random.default_rng(3)
    n, d_img, d_prop = 16, 384, 14
    img_feats = {
        "image": rng.normal(size=(n, d_img)).astype(np.float32),
        "wrist_image": rng.normal(size=(n, d_img)).astype(np.float32),
    }
    prop = rng.normal(size=(n, d_prop)).astype(np.float32)
    # Single-demo, sequential timesteps.
    demo = np.zeros(n, dtype=np.int32)
    t = np.arange(n, dtype=np.int32)
    win = build_window_indices(demo, t, K=2)
    state = compose_state_vectors_batch(img_feats, prop, cfg, window_indices=win)
    # concat doubles every per-frame block
    assert state.shape == (n, 2 * (2 * d_img) + 2 * d_prop)
    assert np.allclose(np.linalg.norm(state, axis=-1), 1.0, atol=1e-5)


def test_compose_state_window_mean_keeps_dim():
    cfg = LocalThresholdConfig(
        encoder_name="dinov2_vits14",
        state_window_size=3,
        state_window_aggregation="mean",
    )
    rng = np.random.default_rng(4)
    n, d_img, d_prop = 12, 384, 14
    img_feats = {
        "image": rng.normal(size=(n, d_img)).astype(np.float32),
        "wrist_image": rng.normal(size=(n, d_img)).astype(np.float32),
    }
    prop = rng.normal(size=(n, d_prop)).astype(np.float32)
    demo = np.zeros(n, dtype=np.int32)
    t = np.arange(n, dtype=np.int32)
    win = build_window_indices(demo, t, K=3)
    state = compose_state_vectors_batch(img_feats, prop, cfg, window_indices=win)
    # mean preserves per-frame dim
    assert state.shape == (n, 2 * d_img + d_prop)
    assert np.allclose(np.linalg.norm(state, axis=-1), 1.0, atol=1e-5)


def test_compose_state_window_k1_matches_no_window():
    """K=1 (whether passed as identity indices or None) must match the
    single-frame baseline bit-for-bit."""
    cfg = LocalThresholdConfig(encoder_name="dinov2_vits14", state_window_size=1)
    rng = np.random.default_rng(5)
    n, d_img = 24, 384
    img_feats = {
        "image": rng.normal(size=(n, d_img)).astype(np.float32),
        "wrist_image": rng.normal(size=(n, d_img)).astype(np.float32),
    }
    prop = rng.normal(size=(n, 14)).astype(np.float32)
    win = np.arange(n, dtype=np.int64)[:, None]  # (n, 1) identity
    s_none = compose_state_vectors_batch(img_feats, prop, cfg)
    s_win = compose_state_vectors_batch(img_feats, prop, cfg, window_indices=win)
    assert np.allclose(s_none, s_win, atol=1e-6)
