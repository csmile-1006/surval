"""Round-trip tests for surval.cache_io.write_seqcache_hdf5.

Each round-trip writes a small synthetic cache and reads it back via
``surval.sequential_validate._load_cache_hdf5`` to assert that the canonical
schema is preserved end-to-end.
"""

import math

import numpy as np
import pytest

from surval.cache_io import (
    ReservoirSampler,
    compute_off_manifold_errors,
    compute_off_manifold_norm,
    concat_trim_time,
    concat_trim_time_feature,
    group_rows_by_demo,
    sort_demo_keys,
    write_seqcache_hdf5,
)
from surval.sequential_validate import _load_cache_hdf5


def _make_dummy_cache_data(*, n=12, t=5, a=7, num_samples=2, num_demos=3, with_obs=False, f=4):
    """Return ``(demo_ids, index_in_demo, actions, pred_list, obs_features?)``."""
    rng = np.random.default_rng(0)
    base = np.repeat(np.arange(num_demos), n // num_demos)
    if base.shape[0] < n:
        base = np.concatenate([base, np.full(n - base.shape[0], num_demos - 1, dtype=base.dtype)])
    demo_ids = np.array([f"demo_{int(d)}" for d in base])
    # Per-demo, scatter index_in_demo non-monotonically so the writer's sort is exercised.
    index_in_demo = np.zeros(n, dtype=np.int64)
    for d in range(num_demos):
        rows = np.where(base == d)[0]
        index_in_demo[rows] = rng.permutation(len(rows))
    actions = rng.standard_normal((n, t, a)).astype(np.float32)
    pred_list = [rng.standard_normal((n, t, a)).astype(np.float32) for _ in range(num_samples)]
    obs_features = rng.standard_normal((n, f)).astype(np.float32) if with_obs else None
    return demo_ids, index_in_demo, actions, pred_list, obs_features


def test_round_trip_basic(tmp_path):
    demo_ids, idx, actions, pred_list, _ = _make_dummy_cache_data(num_samples=1)
    out = str(tmp_path / "seqcache_step_000042.hdf5")
    write_seqcache_hdf5(
        out,
        demo_ids=demo_ids,
        index_in_demo=idx,
        actions=actions,
        pred_actions_list=pred_list,
        checkpoint="dummy_ckpt",
        step=42,
        val_loss=0.123,
        off_manifold_norms={0: 0.456},
    )
    (
        demo_ids_arr,
        index_in_demo,
        gt_arr,
        pred_samples,
        obs_features,
        checkpoint,
        step,
        valid_loss,
        valid_off,
    ) = _load_cache_hdf5(out)
    assert step == 42
    assert checkpoint == "dummy_ckpt"
    assert valid_loss == pytest.approx(0.123)
    assert valid_off == pytest.approx(0.456)
    assert pred_samples.shape == (1, *actions.shape)
    assert gt_arr.shape == actions.shape
    assert obs_features is None
    by_demo: dict[str, list[int]] = {}
    for did, ii in zip(demo_ids_arr, index_in_demo, strict=True):
        by_demo.setdefault(str(did), []).append(int(ii))
    for ii_list in by_demo.values():
        assert ii_list == sorted(ii_list)


def test_round_trip_multi_sample_and_obs(tmp_path):
    demo_ids, idx, actions, pred_list, obs = _make_dummy_cache_data(num_samples=3, with_obs=True)
    out = str(tmp_path / "seqcache_step_000007.hdf5")
    write_seqcache_hdf5(
        out,
        demo_ids=demo_ids,
        index_in_demo=idx,
        actions=actions,
        pred_actions_list=pred_list,
        checkpoint="ckpt2",
        step=7,
        val_loss=0.5,
        off_manifold_norms={0: 0.1, 1: 0.2, 2: 0.3},
        obs_features=obs,
    )
    (_, _, _, pred_samples, obs_features_loaded, _, _, _, valid_off) = _load_cache_hdf5(
        out, load_obs_features=True
    )
    assert pred_samples.shape == (3, *actions.shape)
    # Reader exposes only sample_0; verify it matches the dict entry.
    assert valid_off == pytest.approx(0.1)
    assert obs_features_loaded is not None
    # 2D obs_features is read back as [N, F].
    assert obs_features_loaded.shape == (actions.shape[0], obs.shape[1])


def test_round_trip_no_omn(tmp_path):
    demo_ids, idx, actions, pred_list, _ = _make_dummy_cache_data()
    out = str(tmp_path / "seqcache_step_000000.hdf5")
    write_seqcache_hdf5(
        out,
        demo_ids=demo_ids,
        index_in_demo=idx,
        actions=actions,
        pred_actions_list=pred_list,
        checkpoint="x",
        step=0,
        val_loss=None,
    )
    (_, _, _, _, _, _, _, valid_loss, valid_off) = _load_cache_hdf5(out)
    assert math.isnan(valid_loss)
    assert valid_off is None


def test_off_manifold_scalar_shorthand(tmp_path):
    demo_ids, idx, actions, pred_list, _ = _make_dummy_cache_data(num_samples=1)
    out = str(tmp_path / "seqcache_step_000001.hdf5")
    write_seqcache_hdf5(
        out,
        demo_ids=demo_ids,
        index_in_demo=idx,
        actions=actions,
        pred_actions_list=pred_list,
        checkpoint="x",
        step=1,
        off_manifold_norms=0.99,
    )
    (*_, valid_off) = _load_cache_hdf5(out)
    assert valid_off == pytest.approx(0.99)


def test_round_trip_obs_features_3d(tmp_path):
    demo_ids, idx, actions, pred_list, _ = _make_dummy_cache_data(num_samples=1)
    rng = np.random.default_rng(1)
    obs_3d = rng.standard_normal((actions.shape[0], actions.shape[1], 8)).astype(np.float32)
    out = str(tmp_path / "seqcache_step_000003.hdf5")
    write_seqcache_hdf5(
        out,
        demo_ids=demo_ids,
        index_in_demo=idx,
        actions=actions,
        pred_actions_list=pred_list,
        checkpoint="x",
        step=3,
        obs_features=obs_3d,
    )
    (_, _, _, _, obs_features_loaded, *_) = _load_cache_hdf5(out, load_obs_features=True)
    # Reader collapses 3D obs_features by taking time index 0, so the F dim is preserved.
    assert obs_features_loaded.shape == (actions.shape[0], 8)


def test_writer_rejects_shape_mismatch(tmp_path):
    demo_ids, idx, actions, pred_list, _ = _make_dummy_cache_data(num_samples=1)
    bad_pred = pred_list[0][:, :, :-1]  # truncate ac_dim
    with pytest.raises(ValueError, match="does not match actions shape"):
        write_seqcache_hdf5(
            str(tmp_path / "x.hdf5"),
            demo_ids=demo_ids,
            index_in_demo=idx,
            actions=actions,
            pred_actions_list=[bad_pred],
            checkpoint="x",
            step=0,
        )


def test_sort_demo_keys():
    keys = ["demo_10", "demo_2", "demo_0", "extra", "row_5"]
    out = sort_demo_keys(keys)
    assert out == ["demo_0", "demo_2", "row_5", "demo_10", "extra"]


def test_concat_trim_time():
    a = np.zeros((3, 5, 4), dtype=np.float32)
    b = np.ones((2, 4, 6), dtype=np.float32)
    out = concat_trim_time([a, b])
    assert out.shape == (5, 4, 4)
    assert out[:3].sum() == 0.0
    assert (out[3:] == 1.0).all()


def test_concat_trim_time_empty():
    out = concat_trim_time([])
    assert out.shape == (0, 0, 0)


def test_concat_trim_time_feature():
    a = np.zeros((3, 5, 4), dtype=np.float32)
    b = np.ones((2, 4, 6), dtype=np.float32)
    out = concat_trim_time_feature([a, b])
    assert out.shape == (5, 4, 4)


def test_group_rows_by_demo():
    demo_ids = np.array(["a", "a", "b", "a", "b"])
    idx = np.array([2, 0, 0, 1, 1])
    by_demo = group_rows_by_demo(demo_ids, idx)
    assert set(by_demo.keys()) == {"a", "b"}
    # 'a' rows: original indices [0, 1, 3] with idx [2, 0, 1] → sorted by idx: [1, 3, 0]
    assert list(by_demo["a"]) == [1, 3, 0]
    assert list(by_demo["b"]) == [2, 4]


def test_reservoir_sampler_below_capacity():
    s = ReservoirSampler(capacity=10, slots=("x",), rng_seed=0)
    for i in range(5):
        s.observe(x=np.array(i))
    out = s.collect()
    assert len(out["x"]) == 5
    assert s.n_seen == 5


def test_reservoir_sampler_above_capacity():
    s = ReservoirSampler(capacity=3, slots=("x",), rng_seed=0)
    for i in range(100):
        s.observe(x=np.array(i))
    out = s.collect()
    assert len(out["x"]) == 3
    assert s.n_seen == 100


def test_reservoir_sampler_unbounded_keeps_everything():
    s = ReservoirSampler(capacity=None, slots=("x",), rng_seed=0)
    for i in range(50):
        s.observe(x=np.array(i))
    assert len(s) == 50


def test_reservoir_sampler_multi_slot_alignment():
    s = ReservoirSampler(capacity=4, slots=("a", "b"), rng_seed=0)
    for i in range(20):
        s.observe(a=np.array(i), b=np.array(-i))
    out = s.collect()
    assert len(out["a"]) == 4
    # Slots stay paired: every kept (a, b) satisfies b == -a.
    for a_v, b_v in zip(out["a"], out["b"], strict=True):
        assert int(b_v) == -int(a_v)


def test_reservoir_sampler_rejects_unknown_slot():
    s = ReservoirSampler(capacity=2, slots=("a", "b"), rng_seed=0)
    with pytest.raises(ValueError, match="expected slot kwargs"):
        s.observe(a=1, c=2)


def test_off_manifold_errors_zero_when_no_expert():
    pred = np.zeros((4, 5, 3), dtype=np.float32)
    feats = np.zeros((4, 5, 6), dtype=np.float32)
    out = compute_off_manifold_errors(pred, feats, expert_actions=None)
    assert out.shape == (4, 5)
    assert (out == 0).all()


def test_off_manifold_norm_pure_numpy():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    expert = rng.standard_normal((10, 4)).astype(np.float32)
    pred = expert.copy()
    state_feats = rng.standard_normal((10, 6)).astype(np.float32)
    err = compute_off_manifold_norm(pred, state_feats, expert, k=3)
    # Projecting expert actions onto neighbor expert actions yields a small residual.
    assert err < 5.0
    assert err >= 0.0
