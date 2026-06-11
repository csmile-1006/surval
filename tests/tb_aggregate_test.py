"""Unit tests for surval.tb_aggregate (layouts + metrics + cache round-trip)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from surval.tb_aggregate.layouts import (
    LOCAL_THRESHOLD_HP_DIR_PATTERN,
    parse_hparam,
)
from surval.tb_aggregate.metrics import (
    _compute_seed_metrics_from_arrays,
    bootstrap_ci_of_mean,
    mmrv,
)
from surval.tb_aggregate.tb_io import (
    CachedScalarLoader,
    cache_path_for_seqval,
    cache_path_for_train,
    load_cache_file,
    save_cache_file,
)


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------


def test_hp_dir_regex_matches_local_threshold():
    name = "tq0.75_lsetau1.0_ctf0.5_ta15_nds8_noshare_noskip_L2"
    assert LOCAL_THRESHOLD_HP_DIR_PATTERN.match(name) is not None


def test_parse_hparam_returns_key_and_dict():
    hp_dir = "tq0.95_lsetau25.0_ctf1.0_ta15_nds8_share_skip_L2"
    group_dir = (
        "seqcache_dsnoise_0.05_0.2_trajectory_epoch_600_success_n100.hdf5"
        "_modefull_splitvalid_vkeynone_efd10"
    )
    result = parse_hparam(hp_dir, group_dir)
    assert result is not None
    hp_key, hp = result
    assert hp["tq"] == "0.95"
    assert hp["lsetau"] == "25.0"
    assert hp["ctf"] == "1.0"
    assert hp["ta"] == "15"
    assert hp["nds"] == "8"
    assert hp["share"] == "share"
    assert hp["skip"] == "skip"
    assert hp["dist"] == "L2"
    assert hp["cache_mode"] == "full"
    assert hp["cache_split"] == "valid"
    assert hp["eval_first_n_demos"] == "10"
    assert "tq=0.95" in hp_key


def test_parse_hparam_rejects_bad_hp_dir():
    assert parse_hparam("not_a_match", "seqcache_dsx_modea_splitb_vkeyc_efd1") is None


def test_parse_hparam_baseline_hp_has_none_ab():
    hp_dir = "tq0.75_lsetau1.0_ctf0.5_ta15_nds8_noshare_noskip_L2"
    group_dir = "seqcache_dsx_modefull_splitvalid_vkeynone_efd10"
    _, hp = parse_hparam(hp_dir, group_dir)
    assert hp["ab"] is None


@pytest.mark.parametrize(
    "variant",
    ["baseline", "nogrp", "fixthr", "fixthr_intra", "tmean", "globaldb", "q05", "q99"],
)
def test_parse_hparam_captures_ab_tag(variant):
    hp_dir = f"tq0.75_lsetau1.0_ctf0.5_ta15_nds8_noshare_noskip_L2_ab{variant}"
    group_dir = "seqcache_dsx_modefull_splitvalid_vkeynone_efd10"
    hp_key, hp = parse_hparam(hp_dir, group_dir)
    assert hp["ab"] == variant
    assert f"ab={variant}" in hp_key


def test_parse_hparam_rejects_unknown_ab_variant():
    hp_dir = "tq0.75_lsetau1.0_ctf0.5_ta15_nds8_noshare_noskip_L2_abother"
    group_dir = "seqcache_dsx_modefull_splitvalid_vkeynone_efd10"
    assert parse_hparam(hp_dir, group_dir) is None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_mmrv_perfect_ranking_is_zero():
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    B = A.copy()
    assert mmrv(A, B) == 0.0


def test_mmrv_fully_reversed_matches_closed_form():
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    B = -A
    # Every off-diagonal pair triggers, so each row's max is max |A_i - A_j|.
    expected = float(np.mean([np.max(np.abs(A - a)) for a in A]))
    assert mmrv(A, B) == pytest.approx(expected)


def test_mmrv_one_swap():
    A = np.array([0.0, 1.0, 2.0])
    # Swap top two in B; ground-truth ordering disagrees on (1, 2).
    B = np.array([0.0, 2.0, 1.0])
    # Pair (1,2) and (2,1) disagree with |A_i - A_j| = 1.
    # Row 0 max = 0, row 1 max = 1, row 2 max = 1. Mean = 2/3.
    assert mmrv(A, B) == pytest.approx(2.0 / 3.0)


def test_compute_seed_metrics_includes_mmrv():
    rng = np.random.default_rng(0)
    A = np.linspace(0.0, 1.0, 20)
    B = A + rng.normal(scale=0.05, size=A.shape)
    metrics = _compute_seed_metrics_from_arrays(
        A=A, B=B, k_list=[1, 3, 5], eps=1e-8, tie_tol=0.0, nan_policy="omit",
    )
    assert "mmrv" in metrics
    assert "spearman" in metrics
    assert metrics["spearman"] > 0.9
    assert metrics["mmrv"] >= 0.0


def test_bootstrap_ci_of_mean_basic():
    rng = np.random.default_rng(1)
    vals = np.full(10, 0.5)
    lo, hi = bootstrap_ci_of_mean(vals, num_bootstrap=200, ci_level=0.95, rng=rng)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# tb_io
# ---------------------------------------------------------------------------


def test_cache_round_trip_via_loader():
    with tempfile.TemporaryDirectory() as cache_dir:
        train_payload = {
            "tb_dir": "/dev/null/train",
            "event_file": None,
            "use_only_latest": True,
            "tags": {
                "Valid/Loss": {"steps": [0, 1, 2], "values": [0.5, 0.3, 0.2]},
                "Rollout/Success_Rate/X-mean": {
                    "steps": [0, 1, 2], "values": [0.1, 0.4, 0.7],
                },
            },
        }
        seqval_payload = {
            "tb_dir": "/dev/null/seqval",
            "event_file": None,
            "use_only_latest": True,
            "tags": {
                "SequentialValid/PrefixSurvival_Score": {
                    "steps": [0, 1, 2], "values": [0.2, 0.5, 0.8],
                },
                "SequentialValid/ActionL2_mean": {
                    "steps": [0, 1, 2], "values": [0.9, 0.6, 0.3],
                },
            },
        }
        seed_name = "policy_seed_7"
        hp_dir = "tq0.75_lsetau1.0_ctf0.5_ta15_nds8_noshare_noskip_L2"
        group_dir = "seqcache_dsfoo_modesplit_splitvalid_vkeynone_efd10"

        save_cache_file(train_payload, cache_path_for_train(cache_dir, seed_name))
        save_cache_file(
            seqval_payload,
            cache_path_for_seqval(cache_dir, seed_name, hp_dir, group_dir),
        )

        loader = CachedScalarLoader(cache_dir)
        steps, values = loader.load_train_series(seed_name, "Valid/Loss")
        assert values.tolist() == [0.5, 0.3, 0.2]

        _, sv = loader.load_seqval_series(
            seed_name, hp_dir, group_dir, "SequentialValid/PrefixSurvival_Score",
        )
        assert sv.tolist() == [0.2, 0.5, 0.8]

        with pytest.raises(KeyError):
            loader.load_seqval_series(seed_name, hp_dir, group_dir, "missing/tag")


def test_save_cache_file_atomic_rename():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "a.json")
        payload = {"tb_dir": "/x", "tags": {}}
        save_cache_file(payload, out)
        assert os.path.exists(out)
        # No stray .tmp.
        assert not os.path.exists(out + ".tmp")
        # Round-trip preserves payload (plus extracted_at).
        loaded = load_cache_file(out)
        assert loaded["tb_dir"] == "/x"
        assert "extracted_at" in loaded
