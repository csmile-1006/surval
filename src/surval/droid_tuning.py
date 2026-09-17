"""CPU-only, finite-grid DROID tuning against the existing SURVAL formula.

No policy inference. Local scales use the shared full-val DINO reference;
quantiles remain independent for position and rotation. Global fallback is
deliberately unsupported by this fast path: every row must have enough neighbors.
"""

from dataclasses import replace
from itertools import product
from pathlib import Path
import json

import numpy as np

from .droid import action_space_config, score_policy_cache
from .local_threshold.rotation import rot6d_relative_delta
from .local_threshold.threshold import (compute_local_thresholds,
                                        query_threshold_neighbors)
from .sequential_validate import _aggregate_chunk_time, _compute_per_block_scaled_step_errors
from .tb_aggregate.metrics import _compute_seed_metrics_from_arrays


AXES = {"ta": list(range(1, 9)), "k": [10, 20, 35, 50, 75, 100, 150, 200],
        "quantile": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
        "scale_multiplier": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]}
FIXED = {"every_step": True, "num_samples": 8, "temporal_radius": 5,
         "min_neighbors": 10, "chunk_top_frac": 0.5, "lse_tau": 1.0,
         "exact_neighbors": True, "seed": 0}
VERSION = 1


def hp_grid(axes=AXES):
    if set(axes) != set(AXES):
        raise ValueError("Expected ta, k, quantile, scale_multiplier axes")
    if any(not values or len(values) != len(set(values)) for values in axes.values()):
        raise ValueError("Axes must be nonempty and unique")
    for ta, k, q, c in product(*(axes[name] for name in AXES)):
        if (ta != int(ta) or not 1 <= ta <= 8 or k != int(k) or k < 10
                or not 0 < q <= 1 or not np.isfinite(c) or c <= 0):
            raise ValueError("Invalid HP or chunk beyond cache horizon 8")
    return [dict(zip(AXES, values)) for values in product(*(axes[name] for name in AXES))]


def expert_distance_matrices(db, batch_size=64):
    """O(N^2) float32 storage; bounded broadcast work, same library geometry.

    # ponytail: max current DB is 4119 rows; switch to on-demand pair blocks
    # when a reference is too large for its two N-by-N distance matrices.
    """
    actions = db.records.actions
    matrices = []
    for name, sl in db.cfg.block_slice_dict().items():
        block = actions[:, sl]
        matrix = np.empty((len(block), len(block)), np.float32)
        for start in range(0, len(block), batch_size):
            a, b = block[start:start + batch_size, None], block[None]
            delta = (rot6d_relative_delta(a, b) if db.cfg.block_type_dict().get(name) == "rot6d"
                     else a - b)
            matrix[start:start + batch_size] = np.linalg.norm(delta, axis=-1)
        matrices.append(matrix)
    if not all(np.isfinite(m).all() for m in matrices):
        raise ValueError("Nonfinite expert distances")
    return matrices


def local_scale_table(db, matrices, k, quantiles, *, batch_size=32):
    cfg = replace(db.cfg, k_neighbors=k, temporal_exclusion_radius=5, min_neighbors_for_local=10,
                  quantiles=tuple(quantiles), pairwise_or_query_centered="pairwise")
    neighbors = query_threshold_neighbors(db, cfg, exact_neighbors=True)
    counts = np.array([len(row) for row in neighbors])
    if np.any(counts != k) or np.any(counts < 10):
        raise ValueError("Fast sweep requires exactly k eligible neighbors and no global fallback")
    neighbors = np.stack(neighbors)
    upper = np.triu_indices(k, 1)
    scales = np.empty((db.n_states, len(matrices), len(quantiles)), np.float32)
    for start in range(0, db.n_states, batch_size):
        nb = neighbors[start:start + batch_size]
        for bi, matrix in enumerate(matrices):
            pairs = matrix[nb[:, upper[0]], nb[:, upper[1]]]
            scales[start:start + len(nb), bi] = np.quantile(pairs, quantiles, axis=1).T
    return scales, cfg


def chunk_errors(cache, tas):
    """Factor only positive, row-constant scale out of top-k/sample means."""
    cfg = action_space_config(cache["droid_action_space"])
    # Presets expose dictionaries, not LocalThresholdConfig objects.
    names = cfg["block_names"]
    slices = cfg["block_slices"]
    raw = _compute_per_block_scaled_step_errors(
        cache["pred_actions"], cache["actions"][None],
        np.ones((len(cache["actions"]), len(names)), np.float32), names, slices,
        block_types=cfg.get("block_types"))
    if max(tas) > raw.shape[2]:
        raise ValueError("Requested Ta exceeds cached action horizon")
    return {ta: np.stack([_aggregate_chunk_time(raw[:, :, :ta, b], agg="top_k_mean", top_frac=0.5)
                         .mean(axis=0) for b in range(len(names))], axis=-1) for ta in tas}


def grouped_rows(cache, *, stride=1):
    ids, times = cache["demo_ids"], cache["index_in_demo"]
    groups = []
    for name in sorted(set(ids)):
        rows = np.flatnonzero((ids == name) & (times % stride == 0))
        if len(rows):
            groups.append(rows[np.argsort(times[rows], kind="stable")])
    return groups


def scores_from_errors(errors, scales, groups):
    """[configs,rows,blocks] -> scores; float64 gate/LSE/cumprod like scorer."""
    scaled = (errors / np.maximum(scales.astype(np.float32), 1e-12)).astype(np.float64)
    p = np.exp(-np.maximum(0.0, scaled - 1.0))
    neg = -p  # fixed LSE tau=1
    maximum = neg.max(axis=-1, keepdims=True)
    survival = -(np.log(np.exp(neg - maximum).mean(axis=-1)) + maximum[..., 0])
    result = np.zeros(len(survival), np.float64)
    for rows in groups:
        result += np.cumprod(survival[:, rows], axis=1).mean(axis=1)
    return result / len(groups)


def score_grid(cache, scales_by_k, axes=AXES):
    grid = hp_grid(axes)
    errors = chunk_errors(cache, axes["ta"])
    groups = grouped_rows(cache)
    q_index = {q: i for i, q in enumerate(axes["quantile"])}
    result = np.empty(len(grid), np.float64)
    for start in range(0, len(grid), 64):
        hps = grid[start:start + 64]
        scale = np.stack([scales_by_k[h["k"]][:, :, q_index[h["quantile"]]] *
                          np.float32(h["scale_multiplier"]) for h in hps])
        error = np.stack([errors[h["ta"]] for h in hps])
        result[start:start + len(hps)] = scores_from_errors(error, scale, groups)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite sweep scores")
    return result


def primary_metrics(success_rates, scores):
    """Use the existing report definitions, including tie handling."""
    success_rates, scores = np.asarray(success_rates), np.asarray(scores)
    m = _compute_seed_metrics_from_arrays(success_rates, scores, [], 1e-12, 0.0, "propagate",
                                         compute_kendall=False)
    out = {name: m[name] for name in ("nregret", "mmrv", "spearman")}
    span = float(np.ptp(success_rates))
    selected = int(np.argmax(scores))
    out.update(selected_index=selected, selected_success_rate=float(success_rates[selected]),
               regret_pp=float((max(success_rates) - success_rates[selected]) * 100))
    # delta_spearman may be undefined even when the primary metrics are valid.
    out["eligible"] = bool(span > 0 and np.ptp(scores) > 1e-8
                           and all(np.isfinite(out[name]) for name in ("nregret", "mmrv", "spearman")))
    out["balanced_loss"] = ((out["nregret"] + out["mmrv"] / span + (1 - out["spearman"]) / 2) / 3
                            if span > 0 else float("nan"))
    return out


def canonical_thresholds(db, quantiles, k, target):
    """Audit local quantiles through the original implementation.

    Global values are unused, explicitly zero placeholders: require NO fallback.
    Only scorer-required row/threshold files are written, not duplicate DINO images.
    """
    cfg = replace(db.cfg, quantiles=tuple(quantiles), k_neighbors=k,
                  temporal_exclusion_radius=5, min_neighbors_for_local=10)
    threshold = compute_local_thresholds(db, cfg, np.zeros((2, len(quantiles)), np.float32),
                                         exact_neighbors=True)
    if threshold.fallback_used.any() or np.any(threshold.n_neighbors_used != k):
        raise ValueError("Canonical audit requires exactly k neighbors with no fallback")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "demo_id_int.npy", db.records.demo_id_int)
    np.save(target / "t.npy", db.records.t)
    (target / "demo_id_str.json").write_text(json.dumps(db.records.demo_id_str_by_int))
    threshold.save(str(target / "thresholds"))
    return threshold


def canonical_score(cache, target, threshold, hp):
    """Canonical score with scale multiplication performed before division."""
    scaled = replace(threshold, thresholds=threshold.thresholds * np.float32(hp["scale_multiplier"]),
                     global_thresholds=threshold.global_thresholds * np.float32(hp["scale_multiplier"]))
    scaled.save(str(Path(target) / "thresholds"))
    summary, episodes = score_policy_cache(cache, str(target), quantile=hp["quantile"],
                                           ta=hp["ta"], num_samples=cache["pred_actions"].shape[0],
                                           every_step=True)
    if sum(e["T"] for e in episodes) != len(cache["actions"]):
        raise AssertionError("Canonical scorer skipped observation rows")
    return summary["PrefixSurvival_Score"]
