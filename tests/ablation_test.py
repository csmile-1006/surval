"""Unit tests for the leave-one-out ablation knobs added to sequential_validate."""

from __future__ import annotations

import numpy as np
import pytest

from surval.sequential_validate import _flatten_scale_groups


# ---------------------------------------------------------------------------
# scale_groups flattening (component A: grouping)
# ---------------------------------------------------------------------------


def _make_action_space_config():
    return {
        "action_dim": 10,
        "block_names": ["pos", "rot_6d"],
        "block_slices": {"pos": slice(0, 3), "rot_6d": slice(3, 9)},
        "block_dims": {"pos": 3, "rot_6d": 6},
        "arm_pairs": [],
        "scale_groups": [
            {"blocks": ["pos", "rot_6d"], "summary_key": "s_pose"},
        ],
        "summary_scale_fields": [("ActionBlockScale_Pose", "s_pose")],
        "block_types": {"rot_6d": "rot6d"},
    }


def test_flatten_scale_groups_produces_one_block_per_group():
    cfg = _make_action_space_config()
    flat = _flatten_scale_groups(cfg)
    assert flat["scale_groups"] == [
        {"blocks": ["pos"], "summary_key": "pos"},
        {"blocks": ["rot_6d"], "summary_key": "rot_6d"},
    ]


def test_flatten_scale_groups_does_not_mutate_input():
    cfg = _make_action_space_config()
    original_groups = list(cfg["scale_groups"])
    _flatten_scale_groups(cfg)
    assert cfg["scale_groups"] == original_groups


def test_flatten_scale_groups_preserves_other_keys():
    cfg = _make_action_space_config()
    flat = _flatten_scale_groups(cfg)
    for k in ("action_dim", "block_names", "block_slices", "block_dims", "arm_pairs", "block_types"):
        assert flat[k] is cfg[k] or flat[k] == cfg[k]


# ---------------------------------------------------------------------------
# Cumulative product vs cumulative mean (component C: time reduction)
# ---------------------------------------------------------------------------


def _cum_reduce(e_t: np.ndarray, mode: str) -> np.ndarray:
    """Mirror of the branching done in sequential_validate._compute_from_cache."""
    t_len = e_t.shape[0]
    if mode == "mean":
        return np.cumsum(e_t) / np.arange(1, t_len + 1, dtype=np.float64)
    return np.cumprod(e_t)


def test_cum_mean_differs_from_cum_product_on_decaying_signal():
    e_t = np.array([0.9, 0.8, 0.5, 0.2], dtype=np.float64)
    prod = _cum_reduce(e_t, "product")
    mean = _cum_reduce(e_t, "mean")
    assert prod[0] == pytest.approx(mean[0])  # first step always equal
    assert prod[-1] == pytest.approx(0.9 * 0.8 * 0.5 * 0.2)
    assert mean[-1] == pytest.approx((0.9 + 0.8 + 0.5 + 0.2) / 4.0)
    assert mean[-1] > prod[-1]


def test_cum_mean_absorbs_zero_without_cliff():
    """A zero in e_t collapses cumprod to 0 forever; cum-mean keeps going."""
    e_t = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    prod = _cum_reduce(e_t, "product")
    mean = _cum_reduce(e_t, "mean")
    assert prod.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert mean.tolist() == [1.0, 0.5, 2.0 / 3.0, 0.75]


def test_cum_product_matches_default_branch():
    rng = np.random.default_rng(0)
    e_t = rng.uniform(0.0, 1.0, size=12)
    assert np.allclose(_cum_reduce(e_t, "product"), np.cumprod(e_t))
