"""
Shared implementation for sequential-validation-from-cache scripts.

Both ``sequential_validate_from_cache.py`` (14-D standard) and
``sequential_validate_from_cache_dex.py`` (24-D dexterous) delegate their
core logic here.  Each entry-point script defines an *action_space_config*
dict, then calls ``run``.

This script expects caches where ``actions`` and ``pred_actions`` are
already unnormalized to raw physical units (see ``validate_rlds_loss.py``).
No further normalization is applied here, so block scales ``S_g`` and
per-block errors live directly in raw action units.

action_space_config schema
--------------------------
{
    "action_dim"          : int              total dimension to read from cache
    "block_names"         : list[str]        ordered block names
    "block_slices"        : dict[str, slice]
    "block_dims"          : dict[str, int]   raw slice dimension of each block
    "arm_pairs"           : list[(str,str)]  (left, right) pairs for covariance sharing
    "scale_groups"        : list[dict]       pooling groups for scale estimation;
                              each entry: {"blocks": [str], "summary_key": str}
    "summary_scale_fields": list[(str,str)]  (summary_key, scale_key) pairs
                              to expose in the result summary dict
    "block_types"         : dict[str, str]   (optional) special block type per block.
                              Supported values:
                                "rot6d" — 6D continuous rotation representation
                                  (Zhou et al., 2019).  Distances are computed in
                                  axis-angle space via R_a^T @ R_b; covariance is
                                  estimated on the 3-D axis-angle vectors.
                              Blocks not listed here default to raw L2 / Mahalanobis.
}
"""

from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tensorboardX import SummaryWriter

# ---------------------------------------------------------------------------
# File-discovery helpers
# ---------------------------------------------------------------------------

# Cache-mode specs: which on-disk filename glob and HDF5 attribute name to use
# for the time-axis. Different upstream cache writers use different conventions:
#   "step"  — surval_openpi (default): seqcache_step_*.hdf5  + f.attrs["step"]
#   "epoch" — droid_policy_learning, custom-robomimic:
#             seqcache_epoch_*.hdf5 + f.attrs["epoch"]
# The internal variable name is always ``step``; the mode only controls which
# file pattern is discovered, which HDF5 attr is read, and which key the
# JSON/group-summary output uses.
_CACHE_MODE_SPECS: dict[str, dict[str, str]] = {
    "step": {"glob": "seqcache_step_*.hdf5", "attr": "step"},
    "epoch": {"glob": "seqcache_epoch_*.hdf5", "attr": "epoch"},
}


def _cache_mode_spec(mode):
    spec = _CACHE_MODE_SPECS.get(str(mode))
    if spec is None:
        raise ValueError(f"Unknown cache_mode: {mode!r}; choose one of {sorted(_CACHE_MODE_SPECS)}")
    return spec


def _get_cache_files(cache_dir, *, mode="step"):
    spec = _cache_mode_spec(mode)
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    paths = sorted(Path(cache_dir).glob(spec["glob"]))
    return [str(p.resolve()) for p in paths if p.is_file()]


def _discover_cache_groups(cache_dir, *, mode="step"):
    """Discover dirs containing cache files for ``mode``, recursively.

    ``mode`` selects the filename pattern via ``_CACHE_MODE_SPECS`` (e.g. "step"
    matches ``seqcache_step_*.hdf5``; "epoch" matches ``seqcache_epoch_*.hdf5``).
    """
    spec = _cache_mode_spec(mode)
    root = Path(os.path.abspath(os.path.expanduser(cache_dir))).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"cache directory not found: {root}")
    group_to_files = {}
    for p in root.rglob(spec["glob"]):
        if not p.is_file():
            continue
        key = str(p.parent.resolve())
        group_to_files.setdefault(key, []).append(str(p.resolve()))
    groups = []
    for group_dir in sorted(group_to_files.keys()):
        rel_dir = os.path.relpath(group_dir, str(root))
        groups.append(
            {
                "group_dir": group_dir,
                "relative_dir": rel_dir,
                "cache_files": sorted(group_to_files[group_dir]),
            }
        )
    return groups


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _sort_demo_keys(demo_keys):
    def _key(d):
        if d.startswith("demo_"):
            try:
                return (0, int(d.split("_")[-1]))
            except Exception:
                pass
        return (1, d)

    return sorted(demo_keys, key=_key)


# ---------------------------------------------------------------------------
# 6D rotation utilities
# ---------------------------------------------------------------------------


def _rot6d_to_rotmat(r6d):
    """
    Convert 6D rotation representation to rotation matrix via Gram-Schmidt.

    Matches the convention in torch_utils.rotation_6d_to_matrix / matrix_to_rotation_6d:
    the 6D vector stores the first two *rows* of R, and the output has b1, b2, b3 as rows.
      R[..., 0, :] = b1,  R[..., 1, :] = b2,  R[..., 2, :] = b3

    r6d: (..., 6).  Returns: (..., 3, 3).
    """
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)  # (..., 3, 3) rows


def _rotmat_to_axis_angle(rot):
    """
    Convert rotation matrix to axis-angle vector.
    rot: (..., 3, 3).  Returns: (..., 3), with ||result|| = rotation angle in radians.
    Near-zero rotations map to the zero vector.
    """
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)  # (...)
    # Axis from skew-symmetric part of rot
    axis_raw = np.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        axis=-1,
    )  # (..., 3)
    sin_a = np.sin(angle)
    safe_denom = 2.0 * np.where(sin_a > 1e-8, sin_a, np.ones_like(sin_a))
    axis = axis_raw / safe_denom[..., None]
    # For near-zero angles the axis is ill-defined; set result to zero vector
    axis = np.where((angle < 1e-8)[..., None], np.zeros_like(axis), axis)
    return (angle[..., None] * axis).astype(np.float32)


def _rot6d_to_axis_angle(r6d):
    """Absolute axis-angle (from identity) of a 6D rotation. (..., 6) -> (..., 3)."""
    return _rotmat_to_axis_angle(_rot6d_to_rotmat(r6d))


def _rot6d_relative_delta(r6d_a, r6d_b):
    """
    Axis-angle of the relative rotation R_a^T @ R_b (from a to b).
    r6d_a, r6d_b: (..., 6).  Returns: (..., 3).
    ||result|| is the geodesic (angular) distance between the two rotations.
    """
    r_a = _rot6d_to_rotmat(r6d_a)  # (..., 3, 3)
    r_b = _rot6d_to_rotmat(r6d_b)
    r_rel = np.einsum("...ji,...jk->...ik", r_a, r_b)  # r_a^T @ r_b
    return _rotmat_to_axis_angle(r_rel)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


def _block_distance_l2(delta):
    """L2 norm of delta. delta shape (..., D)."""
    return np.linalg.norm(delta, axis=-1).astype(np.float32)


def _block_distance_mahalanobis(delta, inv_sigma):
    """
    Mahalanobis distance for a D-dimensional block.
    delta (..., D), inv_sigma (D, D).  Returns sqrt(delta' inv_sigma delta).
    """
    quad = (delta @ inv_sigma) * delta
    d_sq = np.sum(quad, axis=-1)
    return np.sqrt(np.maximum(d_sq, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------


def _compute_expert_block_covariances(
    demo_trajectories,
    block_names,
    block_dims,
    block_slices,
    arm_pairs,
    *,
    share_across_arms=True,
    reg=1e-6,
    block_types=None,
):
    """
    Compute per-block inv-covariance from expert trajectories.
    Returns dict: block_name -> (D_eff, D_eff) inv(Sigma + reg*I).

    For "rot6d" blocks the raw 6-D slice is converted to a 3-D axis-angle
    vector (absolute, from identity) before the covariance is computed, so
    the returned matrix is always 3x3 for those blocks.
    """
    block_types = block_types or {}
    pooled = {k: [] for k in block_names}
    for traj in demo_trajectories:
        if traj.shape[0] == 0:
            continue
        for k in block_names:
            raw = traj[:, block_slices[k]]
            if block_types.get(k) == "rot6d":
                raw = _rot6d_to_axis_angle(raw)  # (T, 3)
            pooled[k].append(raw)

    out = {}
    for k in block_names:
        d = 3 if block_types.get(k) == "rot6d" else block_dims[k]
        if len(pooled[k]) == 0:
            out[k] = np.eye(d, dtype=np.float32)
            continue
        arr = np.concatenate(pooled[k], axis=0).astype(np.float32)
        if arr.shape[0] < 2:
            out[k] = np.eye(d, dtype=np.float32)
            continue
        sigma = np.cov(arr, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]], dtype=np.float32)
        sigma = np.atleast_2d(sigma).astype(np.float32)
        sigma_reg = sigma + float(reg) * np.eye(sigma.shape[0], dtype=np.float32)
        try:
            out[k] = np.linalg.inv(sigma_reg).astype(np.float32)
        except np.linalg.LinAlgError:
            out[k] = np.eye(d, dtype=np.float32)

    if share_across_arms:
        for left, right in arm_pairs:
            if left in out and right in out:
                out[right] = out[left]
    return out


# ---------------------------------------------------------------------------
# Demo trajectory reconstruction
# ---------------------------------------------------------------------------


def _rows_to_demo_trajectories(actions, demo_ids, index_in_demo, action_dim):
    """
    Reconstruct per-demo expert trajectories from cached rows.
    Uses the first action (t=0) of each chunk at each unique start index.
    """
    if actions.shape[-1] < action_dim:
        raise ValueError(f"Expected at least {action_dim} action dims, got {actions.shape[-1]}")
    by_demo = {}
    for i in range(actions.shape[0]):
        demo = str(demo_ids[i])
        idx = int(index_in_demo[i])
        a0 = actions[i, 0, :action_dim].astype(np.float32)
        by_demo.setdefault(demo, {})[idx] = a0
    trajectories = []
    for demo in sorted(by_demo.keys()):
        idx_to_action = by_demo[demo]
        if not idx_to_action:
            continue
        sorted_idx = sorted(idx_to_action.keys())
        traj = np.stack([idx_to_action[k] for k in sorted_idx], axis=0).astype(np.float32)
        trajectories.append(traj)
    return trajectories


def _build_sequential_summary(
    all_e,
    all_e_mean,
    all_e_per_step,
    all_scores_chunk,
    all_tau_chunk,
    matched_prefix_lengths,
    elapsed_min,
    valid_loss=None,
    valid_off=None,
):
    """Build aggregate sequential-validation summary dict."""
    s_chunk_global = float(np.mean(all_scores_chunk))
    summary = {
        "PrefixSurvival_Score": s_chunk_global,
        "PrefixSurvival_Loss": 1.0 - s_chunk_global,
        "PrefixSurvival_Score_chunk": s_chunk_global,
        "PrefixSurvival_Loss_chunk": 1.0 - s_chunk_global,
        "PrefixSurvival_Score_transition": s_chunk_global,
        "PrefixSurvival_Loss_transition": 1.0 - s_chunk_global,
        "PrefixSurvival_Score_combined": s_chunk_global,
        "PrefixSurvival_Loss_combined": 1.0 - s_chunk_global,
        "PrefixSurvival_Score_mean_per_ep": s_chunk_global,
        "tau_chunk_mean": float(np.mean(all_tau_chunk)),
        "action_error_mean": float(np.mean(all_e)),
        "action_error_std": float(np.std(all_e)),
        "action_error_p50": float(np.percentile(all_e, 50)),
        "action_error_p90": float(np.percentile(all_e, 90)),
        "action_error_mean_over_time": float(np.mean(all_e_mean)),
        "action_error_mean_over_time_std": float(np.std(all_e_mean)),
        # Mean(sum_t s_t) per trajectory: larger means longer valid prefix on average.
        "PrefixSurvival_MatchedPrefixLen_MeanPerTraj": float(np.mean(matched_prefix_lengths)),
        "Time_SequentialValidation": elapsed_min,
    }
    if valid_loss is not None and not np.isnan(valid_loss):
        summary["valid_loss"] = float(valid_loss)
    if valid_off is not None and not np.isnan(valid_off):
        summary["valid_off_manifold_norm"] = float(valid_off)
    for step_idx in range(all_e_per_step.shape[1]):
        summary[f"action_error_per_step_{step_idx}"] = float(np.mean(all_e_per_step[:, step_idx]))
    return summary


# ---------------------------------------------------------------------------
# Block scale estimation
# ---------------------------------------------------------------------------


def _estimate_block_scales_from_expert_demos(
    demo_trajectories,
    block_names,
    block_slices,
    block_dims,
    scale_groups,
    arm_pairs,
    *,
    share_scales_across_arms=True,
    quantile=0.90,
    local_progress_window=0.08,
    distance_mode="L2",
    mahalanobis_reg=1e-6,
    block_types=None,
):
    """
    Estimate tolerance-like block scales from expert-expert local disagreement.

    Args:
        scale_groups: list of {"blocks": [name, ...], "summary_key": str}.
            Defines how to pool blocks together for shared scales.
    """
    if len(demo_trajectories) < 2:
        raise ValueError(f"Need at least 2 expert demos to estimate block scales; got {len(demo_trajectories)}")

    block_types = block_types or {}
    use_mah = distance_mode == "mahalanobis"
    block_inv_sigmas = (
        _compute_expert_block_covariances(
            demo_trajectories,
            block_names,
            block_dims,
            block_slices,
            arm_pairs,
            share_across_arms=share_scales_across_arms,
            reg=mahalanobis_reg,
            block_types=block_types,
        )
        if use_mah
        else None
    )

    per_block_pool = {k: [] for k in block_names}

    for i in range(len(demo_trajectories)):
        a_i = demo_trajectories[i]
        ti = a_i.shape[0]
        if ti < 1:
            continue
        p_i = np.linspace(0.0, 1.0, num=ti, dtype=np.float32)

        for j in range(i + 1, len(demo_trajectories)):
            a_j = demo_trajectories[j]
            tj = a_j.shape[0]
            if tj < 1:
                continue
            p_j = np.linspace(0.0, 1.0, num=tj, dtype=np.float32)

            progress_diff = np.abs(p_i[:, None] - p_j[None, :])  # [Ti, Tj]
            local_mask = progress_diff <= float(local_progress_window)
            if not np.any(local_mask):
                continue

            def _pairwise_block_dist(delta_ij, inv_sig):
                if use_mah and inv_sig is not None:
                    quad = (delta_ij @ inv_sig) * delta_ij
                    d_sq = np.sum(quad, axis=-1)
                    return np.sqrt(np.maximum(d_sq, 0.0)).astype(np.float32)
                return np.linalg.norm(delta_ij, axis=-1).astype(np.float32)

            for k in block_names:
                if block_types.get(k) == "rot6d":
                    # Geodesic distance in axis-angle space
                    delta_k = _rot6d_relative_delta(
                        a_i[:, None, block_slices[k]],
                        a_j[None, :, block_slices[k]],
                    )  # [Ti, Tj, 3]
                else:
                    delta_k = a_i[:, None, block_slices[k]] - a_j[None, :, block_slices[k]]
                inv_k = block_inv_sigmas[k] if block_inv_sigmas is not None else None
                dist = _pairwise_block_dist(delta_k, inv_k)
                masked = np.where(local_mask, dist, np.inf)
                row_min = np.min(masked, axis=1)
                col_min = np.min(masked, axis=0)
                if np.any(np.isfinite(row_min)):
                    per_block_pool[k].append(row_min[np.isfinite(row_min)])
                if np.any(np.isfinite(col_min)):
                    per_block_pool[k].append(col_min[np.isfinite(col_min)])

    def _pooled_quantile_or_fallback(parts, fallback_val=1.0):
        if not parts:
            return float(fallback_val)
        vals = np.concatenate(parts, axis=0).astype(np.float32)
        if vals.size == 0:
            return float(fallback_val)
        return float(np.quantile(vals, quantile))

    scales = {}
    for group in scale_groups:
        group_blocks = group["blocks"]
        summary_key = group["summary_key"]

        if share_scales_across_arms:
            all_parts = []
            for k in group_blocks:
                all_parts += per_block_pool[k]
            q = _pooled_quantile_or_fallback(all_parts)
            for k in group_blocks:
                scales[k] = q
            scales[summary_key] = q
        else:
            per_arm = []
            for k in group_blocks:
                q_k = _pooled_quantile_or_fallback(per_block_pool[k])
                scales[k] = q_k
                per_arm.append(q_k)
            scales[summary_key] = float(np.mean(per_arm))

    scales["estimated_from_expert_pairs"] = True
    scales["num_demos_used"] = len(demo_trajectories)
    if block_inv_sigmas is not None:
        scales["block_inv_sigmas"] = block_inv_sigmas
    return scales


def _estimate_block_scales_from_intra_demo_diffs(
    demo_trajectories,
    block_names,
    block_slices,
    block_dims,
    scale_groups,
    arm_pairs,
    ta,
    *,
    share_scales_across_arms=True,
    quantile=0.90,
    distance_mode="L2",
    mahalanobis_reg=1e-6,
    block_types=None,
):
    """
    Estimate block scales from within-demo action differences.

    For each demo, computes the distance between actions ta steps apart
    (i.e. ||a[t+ta] - a[t]|| for each t within the demo), where ta is the
    action chunk horizon.  All values across all demos are pooled and the
    given quantile is returned as the block scale.
    """
    block_types = block_types or {}
    use_mah = distance_mode == "mahalanobis"
    block_inv_sigmas = (
        _compute_expert_block_covariances(
            demo_trajectories,
            block_names,
            block_dims,
            block_slices,
            arm_pairs,
            share_across_arms=share_scales_across_arms,
            reg=mahalanobis_reg,
            block_types=block_types,
        )
        if use_mah
        else None
    )

    per_block_pool = {k: [] for k in block_names}

    for traj in demo_trajectories:
        t_len = traj.shape[0]
        if ta >= t_len:
            continue

        a_t = traj[:-ta]  # [T-ta, A]
        a_t1 = traj[ta:]  # [T-ta, A]

        for k in block_names:
            inv_k = block_inv_sigmas[k] if block_inv_sigmas is not None else None
            if block_types.get(k) == "rot6d":
                delta_k = _rot6d_relative_delta(
                    a_t[:, block_slices[k]],
                    a_t1[:, block_slices[k]],
                )  # [T-gap, 3]
            else:
                delta_k = a_t1[:, block_slices[k]] - a_t[:, block_slices[k]]  # [T-gap, D]

            if use_mah and inv_k is not None:
                quad = (delta_k @ inv_k) * delta_k
                dists = np.sqrt(np.maximum(np.sum(quad, axis=-1), 0.0)).astype(np.float32)
            else:
                dists = np.linalg.norm(delta_k, axis=-1).astype(np.float32)

            per_block_pool[k].append(dists)

    def _pooled_quantile_or_fallback(parts, fallback_val=1.0):
        if not parts:
            return float(fallback_val)
        vals = np.concatenate(parts, axis=0).astype(np.float32)
        if vals.size == 0:
            return float(fallback_val)
        return float(np.quantile(vals, quantile))

    scales = {}
    for group in scale_groups:
        group_blocks = group["blocks"]
        summary_key = group["summary_key"]

        if share_scales_across_arms:
            all_parts = []
            for k in group_blocks:
                all_parts += per_block_pool[k]
            q = _pooled_quantile_or_fallback(all_parts)
            for k in group_blocks:
                scales[k] = q
            scales[summary_key] = q
        else:
            per_arm = []
            for k in group_blocks:
                q_k = _pooled_quantile_or_fallback(per_block_pool[k])
                scales[k] = q_k
                per_arm.append(q_k)
            scales[summary_key] = float(np.mean(per_arm))

    scales["estimated_from_intra_demo_diffs"] = True
    scales["num_demos_used"] = len(demo_trajectories)
    if block_inv_sigmas is not None:
        scales["block_inv_sigmas"] = block_inv_sigmas
    return scales


# ---------------------------------------------------------------------------
# Local / global-from-DB threshold loading
# ---------------------------------------------------------------------------


def _threshold_subdir_for_method(method):
    """Map a block-scale method to its threshold artifact subdir under
    ``state_db_dir``. ``intra_demo_sc`` reads its own subdir so it can coexist
    on disk with the default ``local`` artifact (different
    LocalThresholdConfig.scale_source)."""
    if method == "intra_demo_sc":
        return "thresholds_intra_demo_sc"
    return "thresholds"


def _build_state_lookup(state_db_dir):
    """Build {(demo_id_str, t_int): state_idx} from a saved state DB directory."""
    demo_id_int = np.load(os.path.join(state_db_dir, "demo_id_int.npy"))
    t_arr = np.load(os.path.join(state_db_dir, "t.npy"))
    with open(os.path.join(state_db_dir, "demo_id_str.json")) as f:
        mapping = {int(k): v for k, v in json.load(f).items()}
    lookup = {}
    for idx in range(demo_id_int.shape[0]):
        lookup[(mapping[int(demo_id_int[idx])], int(t_arr[idx]))] = idx
    return lookup


def _resolve_scales_per_row(
    demo_ids,
    index_in_demo,
    block_names,
    method,
    *,
    expert_block_scales=None,
    state_db_dir=None,
    threshold_quantile=None,
    action_space_block_types=None,
):
    """Resolve per-row, per-block divisors for ``_compute_per_block_scaled_step_errors``.

    Returns:
        scales_per_row_block: (N, B) float32
        scales_repr:          (B,)   float32 — representative per-block scale for
                              the violation log / summary fields. For per-row
                              modes this is the mean over rows.
        info: dict with diagnostic fields ("source", "quantile",
              "fallback_rate", "missing_rows").
    """
    n = demo_ids.shape[0]
    b = len(block_names)

    if method in ("inter_demo", "intra_demo"):
        if expert_block_scales is None:
            raise ValueError(f"expert_block_scales is required for method={method!r}")
        scales = np.zeros((n, b), dtype=np.float32)
        scales_repr = np.zeros((b,), dtype=np.float32)
        for bi, k in enumerate(block_names):
            v = float(expert_block_scales[k])
            scales[:, bi] = v
            scales_repr[bi] = v
        return scales, scales_repr, {"source": method}

    if method not in ("local", "global_db", "intra_demo_sc"):
        raise ValueError(f"Unknown block_scale_method: {method!r}")
    if not state_db_dir:
        raise ValueError(f"--state-db-dir is required when --block-scale-method={method!r}")
    subdir = _threshold_subdir_for_method(method)
    threshold_dir = os.path.join(state_db_dir, subdir)
    if not os.path.isdir(threshold_dir):
        raise FileNotFoundError(
            f"thresholds dir not found: {threshold_dir} "
            "(run compute_local_thresholds.py first with the matching --scale-source)"
        )

    from surval.local_threshold.threshold import LocalThresholdMap

    m = LocalThresholdMap.load(threshold_dir)
    if tuple(m.block_names) != tuple(block_names):
        raise ValueError(
            f"block_names mismatch: cache action space has {list(block_names)}, threshold map has {list(m.block_names)}"
        )
    # If the threshold map advertises per-block distance types, require them to
    # match the consumer's action-space config so we don't compare e.g. an L2
    # threshold against a geodesic per-block error.
    if getattr(m, "block_types", None):
        consumer_types = {
            k: v for k, v in (action_space_block_types or {}).items() if v
        }
        map_types = {k: v for k, v in dict(m.block_types).items() if v}
        if consumer_types != map_types:
            raise ValueError(
                "block_types mismatch between threshold map and consumer ACTION_SPACE: "
                f"map={map_types}, consumer={consumer_types}"
            )
    if threshold_quantile is None:
        raise ValueError(f"--threshold-quantile is required when --block-scale-method={method!r}")
    qi = m.quantile_index(float(threshold_quantile))
    g = m.global_thresholds[:, qi].astype(np.float32)  # (B,)

    if method == "global_db":
        scales = np.broadcast_to(g[None, :], (n, b)).astype(np.float32).copy()
        return (
            scales,
            g,
            {"source": "global_db", "quantile": float(threshold_quantile)},
        )

    # method == "local" or "intra_demo_sc" (per-state lookup, different subdir).
    lookup = _build_state_lookup(state_db_dir)
    scales = np.zeros((n, b), dtype=np.float32)
    fallback = np.zeros((n,), dtype=bool)
    miss = 0
    for i in range(n):
        sidx = lookup.get((str(demo_ids[i]), int(index_in_demo[i])))
        if sidx is None:
            scales[i, :] = g
            fallback[i] = True
            miss += 1
        else:
            scales[i, :] = m.thresholds[sidx, :, qi]
            if bool(m.fallback_used[sidx]):
                fallback[i] = True
    repr_ = scales.mean(axis=0).astype(np.float32) if n > 0 else g
    info = {
        "source": method,
        "quantile": float(threshold_quantile),
        "missing_rows": int(miss),
        "fallback_rate": float(fallback.mean()) if n > 0 else 0.0,
    }
    return scales, repr_, info


# ---------------------------------------------------------------------------
# Step-error computation
# ---------------------------------------------------------------------------


def _group_rows_by_demo(rows):
    """Group rows by demo_id and sort by index_in_demo."""
    by_demo = defaultdict(list)
    for demo_id, idx, err in rows:
        by_demo[demo_id].append((idx, err))
    for demo_id in by_demo:
        by_demo[demo_id].sort(key=lambda x: x[0])
    return by_demo


def _aggregate_chunk_time(
    step_err,
    agg="max",
    top_frac=0.2,
    quantile=0.9,
    smooth_max_p=4.0,
):
    """
    Aggregate per-step errors over the chunk time axis (axis=2).
    step_err: [S, N, T]. Returns [S, N].
    """
    _, _, t_len = step_err.shape
    if t_len == 0:
        raise ValueError("Chunk time dimension T must be positive, got 0.")
    if agg == "max":
        return np.max(step_err, axis=2).astype(np.float32)
    if agg == "top_k_mean":
        k = max(1, min(t_len, int(np.ceil(t_len * top_frac))))
        topk = np.partition(step_err, -k, axis=2)[..., -k:]
        return np.mean(topk, axis=2).astype(np.float32)
    if agg == "quantile":
        return np.asarray(np.percentile(step_err, quantile * 100.0, axis=2), dtype=np.float32)
    if agg == "smooth_max":
        p = float(smooth_max_p)
        return np.asarray(np.power(np.mean(np.power(step_err, p), axis=2), 1.0 / p), dtype=np.float32)
    raise ValueError(f"chunk_time_agg must be one of: max, top_k_mean, quantile, smooth_max; got {agg}")


def _compute_per_block_scaled_step_errors(
    pred,
    gt,
    scales_per_row_block,
    block_names,
    block_slices,
    distance_mode="L2",
    block_inv_sigmas=None,
    block_types=None,
):
    """
    Per-block scaled step errors without cross-block aggregation.

    pred: [S, N, T, A], gt: [1, N, T, A]. Returns [S, N, T, B].
    scales_per_row_block: [N, B] float32 — per-row, per-block divisor. Allows
    state-conditional (local) thresholds; for the legacy constant-per-block
    case, broadcast a (1, B) scalar to (N, B).
    """
    block_types = block_types or {}
    use_mah = distance_mode == "mahalanobis" and block_inv_sigmas is not None

    def _dist(delta, k):
        if use_mah and block_inv_sigmas is not None:
            return _block_distance_mahalanobis(delta, block_inv_sigmas[k])
        return _block_distance_l2(delta)

    scales = np.maximum(scales_per_row_block.astype(np.float32), 1e-12)
    block_errors = []
    for bi, k in enumerate(block_names):
        if block_types.get(k) == "rot6d":
            delta = _rot6d_relative_delta(gt[..., block_slices[k]], pred[..., block_slices[k]])
        else:
            delta = pred[..., block_slices[k]] - gt[..., block_slices[k]]
        dist = _dist(delta, k)  # [S, N, T]
        scale = scales[None, :, None, bi]  # broadcast over S, T
        block_errors.append(dist / scale)

    return np.stack(block_errors, axis=-1).astype(np.float32)  # [S, N, T, B]


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------


def _load_cache_hdf5(path, *, load_obs_features=False, mode="step"):
    """
    Load one seqcache HDF5 file.

    ``obs_features`` is not used by sequential validation metrics; loading it
    can dominate I/O when many cache groups / large files. Default is to skip.

    ``mode`` selects the HDF5 attribute used for the time axis (``step`` by
    default; ``epoch`` for caches written by robomimic-side scripts). The
    returned tuple's 7th element (``step``) holds the integer regardless of
    name; -1 if the attribute is absent.
    """
    spec = _cache_mode_spec(mode)
    attr_name = spec["attr"]
    demo_ids = []
    index_in_demo_chunks = []
    actions_chunks = []
    pred_chunks = []
    obs_feat_chunks = []

    with h5py.File(path, "r") as f:
        checkpoint = str(f.attrs.get("checkpoint", ""))
        step = int(f.attrs.get(attr_name, -1))
        valid_loss = valid_off = None
        if "metrics" in f and "valid" in f["metrics"]:
            vg = f["metrics"]["valid"]
            if "Loss" in vg:
                valid_loss = float(vg["Loss"][()])
            if "off_manifold_norm" in vg:
                valid_off = float(vg["off_manifold_norm"]["sample_0"][()])
        if "data" not in f:
            raise RuntimeError(f"Cache missing 'data' group: {path}")
        data_grp = f["data"]
        for group_name in _sort_demo_keys(list(data_grp.keys())):
            grp = data_grp[group_name]
            demo_id = str(grp.attrs["demo_id"]) if "demo_id" in grp.attrs else group_name
            if "actions" not in grp or "pred_actions" not in grp:
                continue
            idx = (
                grp["index_in_demo"][:]
                if "index_in_demo" in grp
                else np.arange(grp["actions"].shape[0], dtype=np.int64)
            )
            actions = grp["actions"][:]  # [N, T, A]
            pred = grp["pred_actions"][:]  # [S, N, T, A] or [N, T, A]
            obs_feat = (grp["obs_features"][:] if "obs_features" in grp else None) if load_obs_features else None
            if pred.ndim == 3:
                pred = pred[None]
            if actions.ndim != 3 or pred.ndim != 4:
                raise ValueError(
                    f"Unexpected tensor dims in {path} / {demo_id}: actions {actions.shape}, pred {pred.shape}"
                )
            n_local = actions.shape[0]
            if idx.shape[0] != n_local or pred.shape[1] != n_local:
                raise ValueError(f"Length mismatch in cached demo {demo_id} at {path}")
            demo_ids.extend([str(demo_id)] * int(n_local))
            index_in_demo_chunks.append(idx.astype(np.int64))
            actions_chunks.append(actions.astype(np.float32))
            pred_chunks.append(pred.astype(np.float32))
            if load_obs_features and obs_feat is not None:
                if obs_feat.ndim == 3:
                    obs_feat = obs_feat[:, 0, :]
                if obs_feat.ndim != 2 or obs_feat.shape[0] != n_local:
                    raise ValueError(f"Unexpected obs_features shape in {path} / {demo_id}: {obs_feat.shape}")
                obs_feat_chunks.append(obs_feat.astype(np.float32))
            else:
                obs_feat_chunks.append(None)

    if not actions_chunks:
        return (
            np.array([], dtype="U1"),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0, 0), dtype=np.float32),
            np.zeros((0, 0, 0, 0), dtype=np.float32),
            None,
            checkpoint,
            step,
            valid_loss,
            valid_off,
        )

    min_t = min(a.shape[1] for a in actions_chunks)
    min_a = min(a.shape[2] for a in actions_chunks)
    min_s = min(p.shape[0] for p in pred_chunks)
    has_obs = all(x is not None for x in obs_feat_chunks)
    min_f = min(x.shape[1] for x in obs_feat_chunks) if has_obs else None

    actions = np.concatenate([a[:, :min_t, :min_a] for a in actions_chunks], axis=0).astype(np.float32)
    pred_samples = np.concatenate([p[:min_s, :, :min_t, :min_a] for p in pred_chunks], axis=1).astype(np.float32)
    obs_features = (
        np.concatenate([x[:, :min_f] for x in obs_feat_chunks], axis=0).astype(np.float32) if has_obs else None
    )
    max_len = max(len(x) for x in demo_ids)
    demo_ids_arr = np.array(demo_ids, dtype=f"U{max(1, max_len)}")
    index_in_demo = np.concatenate(index_in_demo_chunks, axis=0).astype(np.int64)
    return (
        demo_ids_arr,
        index_in_demo,
        actions,
        pred_samples,
        obs_features,
        checkpoint,
        step,
        valid_loss,
        valid_off,
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_from_cache(
    demo_ids,
    index_in_demo,
    actions,
    pred_actions_samples,
    ta,
    num_diffusion_samples,
    prefix_epsilon,
    prefix_tau,
    prefix_epsilon_soft,
    prefix_mode,
    prefix_scoring_mode,
    action_space_config,
    *,
    block_scale_quantile=0.99,
    block_scale_local_progress_window=0.08,
    block_share_scales_across_arms=True,
    block_scale_method="inter_demo",
    state_db_dir=None,
    threshold_quantile=None,
    distance_mode="L2",
    mahalanobis_reg=1e-6,
    chunk_time_agg="top_k_mean",
    chunk_top_frac=0.2,
    chunk_quantile=0.9,
    chunk_smooth_max_p=4.0,
    skip_intermediate_on_pass=True,
    block_prefix_vote_n=None,
    prefix_soft_aggregator="worst_n",
    prefix_soft_lse_tau=1.0,
    valid_loss=None,
    valid_off=None,
):
    """Returns (summary_dict, per_episode_list)."""
    if actions.ndim != 3 or pred_actions_samples.ndim != 4:
        raise ValueError(
            f"Expected actions [N,T,A] and pred_actions_samples [S,N,T,A], got {actions.shape} and {pred_actions_samples.shape}"
        )
    s_cached, n_rows, t_cached, a_cached = pred_actions_samples.shape
    if n_rows == 0 or t_cached == 0:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []
    if num_diffusion_samples <= 0:
        raise ValueError("num_diffusion_samples must be positive.")
    if num_diffusion_samples > s_cached:
        raise ValueError(
            f"Requested num_diffusion_samples={num_diffusion_samples} but cache has only {s_cached} samples."
        )

    action_dim = action_space_config["action_dim"]
    block_names = action_space_config["block_names"]
    block_slices = action_space_config["block_slices"]
    block_dims = action_space_config["block_dims"]
    arm_pairs = action_space_config["arm_pairs"]
    scale_groups = action_space_config["scale_groups"]
    summary_scale_fields = action_space_config["summary_scale_fields"]
    block_types = action_space_config.get("block_types", {})

    ta_eff = t_cached if ta is None else min(int(ta), t_cached)
    if ta_eff <= 0:
        raise ValueError(f"Effective Ta must be positive, got {ta_eff}")
    if a_cached < action_dim:
        raise ValueError(f"Expected action dim >= {action_dim}, got {a_cached}")

    gt = actions[None, :, :ta_eff, :action_dim]
    pred = pred_actions_samples[:num_diffusion_samples, :, :ta_eff, :action_dim]

    _scale_method = str(block_scale_method)
    expert_block_scales = None
    block_inv_sigmas = None
    if _scale_method in ("inter_demo", "intra_demo"):
        demo_trajectories = _rows_to_demo_trajectories(
            actions[:, :ta_eff, :action_dim], demo_ids, index_in_demo, action_dim
        )
        if _scale_method == "intra_demo":
            expert_block_scales = _estimate_block_scales_from_intra_demo_diffs(
                demo_trajectories=demo_trajectories,
                block_names=block_names,
                block_slices=block_slices,
                block_dims=block_dims,
                scale_groups=scale_groups,
                arm_pairs=arm_pairs,
                share_scales_across_arms=bool(block_share_scales_across_arms),
                ta=int(ta_eff),
                quantile=float(block_scale_quantile),
                distance_mode=str(distance_mode),
                mahalanobis_reg=float(mahalanobis_reg),
                block_types=block_types,
            )
        else:
            expert_block_scales = _estimate_block_scales_from_expert_demos(
                demo_trajectories=demo_trajectories,
                block_names=block_names,
                block_slices=block_slices,
                block_dims=block_dims,
                scale_groups=scale_groups,
                arm_pairs=arm_pairs,
                share_scales_across_arms=bool(block_share_scales_across_arms),
                quantile=float(block_scale_quantile),
                local_progress_window=float(block_scale_local_progress_window),
                distance_mode=str(distance_mode),
                mahalanobis_reg=float(mahalanobis_reg),
                block_types=block_types,
            )
        block_inv_sigmas = expert_block_scales.get("block_inv_sigmas")
    elif _scale_method in ("local", "global_db", "intra_demo_sc"):
        if str(distance_mode) != "L2":
            raise ValueError(
                f"--block-scale-method={_scale_method!r} only supports --distance-mode=L2 "
                "(local thresholds were computed in L2)."
            )
    else:
        raise ValueError(f"Unknown block_scale_method: {_scale_method!r}")

    scales_per_row, scales_repr, scales_info = _resolve_scales_per_row(
        demo_ids=demo_ids,
        index_in_demo=index_in_demo,
        block_names=block_names,
        method=_scale_method,
        expert_block_scales=expert_block_scales,
        state_db_dir=state_db_dir,
        threshold_quantile=threshold_quantile,
        action_space_block_types=block_types,
    )

    num_blocks = len(block_names)
    # Default N: strict majority of blocks (e.g. 3 of 4, 2 of 3).
    _default_vote_n = max(1, num_blocks // 2 + 1)
    vote_n = int(block_prefix_vote_n) if block_prefix_vote_n is not None else _default_vote_n
    vote_n = max(1, min(vote_n, num_blocks))

    # Per-block step errors, then block-aware (hard/soft) prefix survival aggregation.
    step_err_pb = _compute_per_block_scaled_step_errors(
        pred=pred,
        gt=gt,
        scales_per_row_block=scales_per_row,
        block_names=block_names,
        block_slices=block_slices,
        distance_mode=str(distance_mode),
        block_inv_sigmas=block_inv_sigmas,
        block_types=block_types,
    )  # [S', N, Ta_eff, B]

    row_stride = int(ta_eff) if skip_intermediate_on_pass else 1
    keep_idx = [i for i in range(n_rows) if (int(index_in_demo[i]) % row_stride) == 0]
    if not keep_idx:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    chunk_per_block = []
    for b in range(num_blocks):
        chunk_by_sample = _aggregate_chunk_time(
            step_err_pb[:num_diffusion_samples, :, :, b],
            agg=str(chunk_time_agg),
            top_frac=float(chunk_top_frac),
            quantile=float(chunk_quantile),
            smooth_max_p=float(chunk_smooth_max_p),
        )
        chunk_per_block.append(np.mean(chunk_by_sample, axis=0).astype(np.float64))
    chunk_errors_mean = np.mean(np.stack(chunk_per_block, axis=0), axis=0)

    rows_all_e = [(str(demo_ids[i]), int(index_in_demo[i]), float(chunk_errors_mean[i])) for i in keep_idx]
    by_demo_e = _group_rows_by_demo(rows_all_e)
    all_e = np.maximum(np.array([p[1] for pairs in by_demo_e.values() for p in pairs], dtype=np.float64), 0.0)

    by_demo_blocks = []
    for b in range(num_blocks):
        rows_b = [(str(demo_ids[i]), int(index_in_demo[i]), float(chunk_per_block[b][i])) for i in keep_idx]
        by_demo_blocks.append(_group_rows_by_demo(rows_b))

    demo_ids_sorted = sorted(set.intersection(*(set(d.keys()) for d in by_demo_blocks)))
    if not demo_ids_sorted:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []

    all_scores_chunk = []
    all_tau_chunk = []
    matched_prefix_lengths = []
    per_episode = []
    mean_block_pass_rates = []
    # Track ratio of (t, g) entries where e_{t,g} > S_g.
    # err_mat is already scaled by 1/S_g, so the test is err_mat > 1.0.
    violation_total = 0
    violation_count = 0
    per_block_total = np.zeros(num_blocks, dtype=np.int64)
    per_block_viol = np.zeros(num_blocks, dtype=np.int64)
    eps_s = float(prefix_epsilon_soft)
    eps_hard = float(prefix_epsilon)
    n = int(vote_n)
    mode = str(prefix_mode).lower()
    if mode not in ("hard", "soft"):
        raise ValueError(f"prefix_mode must be one of 'hard', 'soft'; got {prefix_mode}")

    for did in demo_ids_sorted:
        pairs0 = sorted(by_demo_blocks[0][did], key=lambda x: x[0])
        t_len = len(pairs0)
        if t_len == 0:
            continue
        idxs = [p[0] for p in pairs0]
        err_mat = np.zeros((num_blocks, t_len), dtype=np.float64)
        aligned = True
        for b in range(num_blocks):
            pb = sorted(by_demo_blocks[b][did], key=lambda x: x[0])
            if len(pb) != t_len or [x[0] for x in pb] != idxs:
                aligned = False
                break
            err_mat[b, :] = np.maximum(np.array([float(x[1]) for x in pb], dtype=np.float64), 0.0)
        if not aligned:
            continue

        viol_mask = err_mat > 1.0  # [num_blocks, t_len]
        violation_total += int(err_mat.size)
        violation_count += int(np.sum(viol_mask))
        per_block_total += int(t_len)
        per_block_viol += np.sum(viol_mask, axis=1).astype(np.int64)

        if mode == "soft":
            pooled = err_mat.ravel()
            if prefix_tau is not None:
                tau_ep = float(prefix_tau)
            else:
                tau_ep = max(
                    float(np.percentile(pooled, 75) - np.percentile(pooled, 25)),
                    1e-6,
                )
            e_t = np.zeros(t_len, dtype=np.float64)
            # Cross-block aggregation strategy: how to collapse per-block soft
            # probabilities p_b(t) into a single survival value e_t.
            #   'worst_n'   — N-th worst block (current default; integer K).
            #   'logsumexp' — Boltzmann smooth-min, continuous τ interpolating
            #                 between min (τ→∞, picks worst block) and mean
            #                 (τ→0). Matches Worst-K's intent without the
            #                 coarse integer grid.
            #   'product'   — Π_b p_b under independence. Hyperparameter-free
            #                 and consistent with SURVAL's prefix survival
            #                 product; over-penalizes when blocks are
            #                 correlated, but uniformly across checkpoints.
            soft_agg = str(prefix_soft_aggregator).lower()
            if soft_agg not in ("worst_n", "logsumexp", "product"):
                raise ValueError(
                    "prefix_soft_aggregator must be one of "
                    f"'worst_n', 'logsumexp', 'product'; got {prefix_soft_aggregator!r}"
                )
            lse_tau = max(float(prefix_soft_lse_tau), 1e-8)
            for t in range(t_len):
                p_b = np.exp(-np.maximum(0.0, err_mat[:, t] - eps_s) / max(tau_ep, 1e-8))
                if soft_agg == "worst_n":
                    # N-th smallest (ascending) soft prob = N-th worst block.
                    # Aligns soft with hard's semantic: hard declares "fail at
                    # t if >= N blocks violate" iff "the N-th worst block
                    # violates". For G=2, N=2 this reduces to max(p_pos,p_rot)
                    # rather than mean(worst-N).
                    n_use = min(n, num_blocks)
                    e_t[t] = float(np.partition(p_b, n_use - 1)[n_use - 1])
                elif soft_agg == "logsumexp":
                    # Smooth-min via LogSumExp:
                    #   v = -(1/τ) · log mean_b exp(-τ · p_b)
                    # τ→∞ ⇒ min(p_b) (worst block); τ→0 ⇒ mean(p_b).
                    neg = -lse_tau * p_b
                    m = float(neg.max())
                    log_mean_exp = float(np.log(np.mean(np.exp(neg - m)))) + m
                    e_t[t] = float(-log_mean_exp / lse_tau)
                else:  # soft_agg == "product"
                    e_t[t] = float(np.prod(p_b))
            s = np.cumprod(e_t)
            if str(prefix_scoring_mode) == "weighted_sum":
                w = 1.0 - np.arange(t_len, dtype=np.float64) / max(t_len, 1)
                score_d = float(np.dot(w, s))
            else:
                score_d = float(np.mean(s))
            all_tau_chunk.append(tau_ep)
            mean_block_pass_rates.append(float(np.mean(e_t)))
            per_episode.append(
                {
                    "demo_id": did,
                    "PrefixSurvival_Score_chunk": score_d,
                    "PrefixSurvival_Loss_chunk": 1.0 - score_d,
                    "PrefixSurvival_Score_transition": score_d,
                    "PrefixSurvival_Loss_transition": 1.0 - score_d,
                    "PrefixSurvival_Score_combined": score_d,
                    "PrefixSurvival_Loss_combined": 1.0 - score_d,
                    "T": t_len,
                    "tau_chunk": tau_ep,
                    "matched_prefix_length": float(np.mean(s)),
                    "mean_chunk_error": float(np.mean(err_mat)),
                    "BlockPrefix_e_t": e_t.tolist(),
                    "BlockPrefix_s_cumprod": s.tolist(),
                    "BlockPrefix_n_smallest_soft": n,
                }
            )
        elif mode == "hard":
            e_t = np.ones(t_len, dtype=np.float64)
            for t in range(t_len):
                e_b = (err_mat[:, t] < eps_hard).astype(np.float64)
                num_violate = int(num_blocks - np.sum(e_b))
                if num_violate >= n:
                    e_t[t] = 0.0
            s = np.cumprod(e_t)
            score_d = float(np.mean(s))
            all_tau_chunk.append(0.0)
            num_ok_per_t = np.sum(err_mat < eps_hard, axis=0)
            mean_block_pass_rates.append(float(np.mean(num_ok_per_t)) / float(num_blocks))
            per_episode.append(
                {
                    "demo_id": did,
                    "PrefixSurvival_Score_chunk": score_d,
                    "PrefixSurvival_Loss_chunk": 1.0 - score_d,
                    "PrefixSurvival_Score_transition": score_d,
                    "PrefixSurvival_Loss_transition": 1.0 - score_d,
                    "PrefixSurvival_Score_combined": score_d,
                    "PrefixSurvival_Loss_combined": 1.0 - score_d,
                    "T": t_len,
                    "tau_chunk": 0.0,
                    "matched_prefix_length": float(np.mean(s)),
                    "mean_chunk_error": float(np.mean(err_mat)),
                    "BlockPrefix_e_t": e_t.tolist(),
                    "BlockPrefix_s_cumprod": s.tolist(),
                    "BlockPrefix_n_violate_threshold": n,
                }
            )
        all_scores_chunk.append(score_d)
        matched_prefix_lengths.append(float(np.mean(s)))

    if not all_scores_chunk:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []

    if violation_total > 0:
        overall_ratio = violation_count / violation_total
        parts = [f"overall={overall_ratio:.4f} ({violation_count}/{violation_total})"]
        for b, bname in enumerate(block_names):
            if per_block_total[b] > 0:
                r_b = per_block_viol[b] / per_block_total[b]
                parts.append(f"{bname}={r_b:.4f}(S_g={float(scales_repr[b]):.4f})")
        print("[violation ratio e_{t,g} > S_g] " + ", ".join(parts), flush=True)

    step_err_combined = np.mean(step_err_pb[0], axis=-1).astype(np.float32)
    e_per_step_sub = step_err_combined[keep_idx_arr]
    e_mean_sub = np.mean(e_per_step_sub, axis=1)

    summary = _build_sequential_summary(
        all_e=all_e,
        all_e_mean=e_mean_sub,
        all_e_per_step=e_per_step_sub,
        all_scores_chunk=all_scores_chunk,
        all_tau_chunk=all_tau_chunk,
        matched_prefix_lengths=matched_prefix_lengths,
        elapsed_min=0.0,
        valid_loss=valid_loss,
        valid_off=valid_off,
    )

    summary["ErrorPrefixSurvival_Score"] = float(summary.get("PrefixSurvival_Score", np.nan))
    summary["ErrorPrefixSurvival_Loss"] = float(summary.get("PrefixSurvival_Loss", np.nan))
    summary["ErrorPrefixSurvival_Score_mean_per_ep"] = float(summary.get("PrefixSurvival_Score_mean_per_ep", np.nan))
    summary["ErrorPrefixSurvival_MatchedPrefixLen_MeanPerTraj"] = float(
        summary.get("PrefixSurvival_MatchedPrefixLen_MeanPerTraj", np.nan)
    )

    if mode == "soft":
        # Soft: e_t = N-th smallest (= N-th worst) block soft prob at t, s_t = cumprod(e).
        summary["BlockPrefixSoft_MinProbCumprod"] = 1.0
        summary["BlockPrefix_MeanPerBlockIsolatedSoftMean"] = (
            float(np.mean(mean_block_pass_rates)) if mean_block_pass_rates else 0.0
        )
    else:  # mode == "hard"
        summary["BlockPrefixVote_N"] = float(vote_n)
        summary["BlockPrefixVote_MeanFracBlocksPassing"] = (
            float(np.mean(mean_block_pass_rates)) if mean_block_pass_rates else 0.0
        )

    # Build a unified key→scalar map for summary_scale_fields. For expert-based
    # methods we reuse the original dict (which contains both per-block scales
    # and group summary keys like "s_joint"). For local / global_db modes we
    # only have per-block representative scales (mean across rows for local),
    # so summary_keys collapse to the mean across blocks.
    if expert_block_scales is not None:
        scales_lookup = expert_block_scales
    else:
        scales_lookup = {bname: float(scales_repr[bi]) for bi, bname in enumerate(block_names)}
        for group in scale_groups:
            blocks_in_group = group["blocks"]
            group_vals = [scales_lookup[k] for k in blocks_in_group if k in scales_lookup]
            if group_vals:
                scales_lookup[group["summary_key"]] = float(np.mean(group_vals))
    for summary_key, scale_key in summary_scale_fields:
        if scale_key in scales_lookup:
            summary[summary_key] = float(scales_lookup[scale_key])
    # `inter_demo` uses `estimated_from_expert_pairs`, while `intra_demo` uses
    # `estimated_from_intra_demo_diffs`. Keep this feature binary and default
    # to 0.0 for cases where the "pairs" estimate isn't used.
    summary["ActionBlockScale_EstimatedFromPairs"] = (
        1.0
        if (expert_block_scales is not None and bool(expert_block_scales.get("estimated_from_expert_pairs", False)))
        else 0.0
    )
    summary["ActionBlockScale_NumDemosUsed"] = (
        float(expert_block_scales.get("num_demos_used", 0.0)) if expert_block_scales is not None else 0.0
    )
    summary["BlockDistanceMode"] = 1.0 if distance_mode == "mahalanobis" else 0.0
    # Local / global-DB threshold diagnostics
    if scales_info.get("source") in ("local", "global_db", "intra_demo_sc"):
        if "quantile" in scales_info:
            summary["LocalThreshold_Quantile"] = float(scales_info["quantile"])
        if "fallback_rate" in scales_info:
            summary["LocalThreshold_FallbackRate"] = float(scales_info["fallback_rate"])
        if "missing_rows" in scales_info:
            summary["LocalThreshold_MissingRows"] = float(scales_info["missing_rows"])

    # -----------------------------------------------------------------
    # L1 / L2 (MSE) between predicted and GT actions
    # pred: [S, N, Ta_eff, A],  gt: [1, N, Ta_eff, A]
    # Computed per diffusion sample, then averaged / min-pooled over S.
    # -----------------------------------------------------------------
    delta = pred - gt  # [S, N, Ta_eff, A]
    l1_per_sample = np.mean(np.abs(delta), axis=(1, 2, 3))  # [S]
    l2_per_sample = np.mean(delta**2, axis=(1, 2, 3))  # [S]  (MSE)
    summary["ActionL1_mean"] = float(np.mean(l1_per_sample))
    summary["ActionL1_best"] = float(np.min(l1_per_sample))
    summary["ActionL2_mean"] = float(np.mean(l2_per_sample))
    summary["ActionL2_best"] = float(np.min(l2_per_sample))

    return summary, per_episode


# ---------------------------------------------------------------------------
# Checkpoint metrics summary printer
# ---------------------------------------------------------------------------

_SUMMARY_METRICS = [
    "PrefixSurvival_Score",
    "ActionL1_mean",
    "ActionL1_best",
    "ActionL2_mean",
    "ActionL2_best",
    "valid_loss",
    "valid_off_manifold_norm",
]


def _print_checkpoint_metrics(
    results: list,
    group_label: str = "",
    *,
    use_compact: bool = True,
    time_key: str = "step",
) -> None:
    """
    Print and return a per-metric summary across all checkpoints in a group.

    Organises results as::

        {metric: {<time_key>: value, ...}, ...}

    and prints both a compact JSON-style dict and a human-readable table.

    Args:
        results: list of per-checkpoint dicts produced by ``_run_single_cache_group``.
            Each entry contains ``time_key`` ("step" or "epoch"), ``"summary"``
            (seq metrics), and ``"valid_summary"`` ({"Loss": ...,
            "Off_Manifold_Norm": ...}).
        group_label: optional label shown in the header (e.g. relative_dir).
        time_key: the JSON field name that holds the integer time axis.
    """
    if not results:
        return

    sorted_results = sorted(results, key=lambda r: int(r[time_key]))
    steps = [int(r[time_key]) for r in sorted_results]

    per_metric: dict[str, dict[int, float]] = {m: {} for m in _SUMMARY_METRICS}
    for r in sorted_results:
        s = int(r[time_key])
        summary = r.get("summary", {})
        vsummary = r.get("valid_summary", {})

        per_metric["PrefixSurvival_Score"][s] = summary.get("PrefixSurvival_Score")
        per_metric["ActionL1_mean"][s] = summary.get("ActionL1_mean")
        per_metric["ActionL1_best"][s] = summary.get("ActionL1_best")
        per_metric["ActionL2_mean"][s] = summary.get("ActionL2_mean")
        per_metric["ActionL2_best"][s] = summary.get("ActionL2_best")
        per_metric["valid_loss"][s] = vsummary.get("Loss")
        per_metric["valid_off_manifold_norm"][s] = vsummary.get("Off_Manifold_Norm")

    header = f"\n{'=' * 72}\n[checkpoint metrics] {group_label}\n{'=' * 72}"
    print(header, flush=True)

    if use_compact:
        compact: dict[str, list] = {}
        for m in _SUMMARY_METRICS:
            vals = [per_metric[m].get(s) for s in steps]
            compact[m] = vals
        compact[f"{time_key}s"] = steps
        print(json.dumps(_to_jsonable(compact), indent=2), flush=True)

    col_w = 14
    step_w = 10
    active = [m for m in _SUMMARY_METRICS if any(v is not None for v in per_metric[m].values())]

    header_row = f"{time_key:>{step_w}}" + "".join(f"  {m[:col_w]:>{col_w}}" for m in active)
    sep = "-" * len(header_row)
    print(f"\n{header_row}\n{sep}", flush=True)
    for s in steps:
        row = f"{s:>{step_w}}"
        for m in active:
            v = per_metric[m].get(s)
            row += f"  {f'{v:.6f}' if v is not None else 'N/A':>{col_w}}"
        print(row, flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# Shared run loop
# ---------------------------------------------------------------------------


def _run_single_cache_group(group, args, action_space_config):
    """
    Process one cache group (one subdir under --cache-dir).

    Filename / attr conventions follow ``args.cache_mode`` (``step`` matches
    ``seqcache_step_*.hdf5`` + ``f.attrs["step"]``; ``epoch`` matches
    ``seqcache_epoch_*.hdf5`` + ``f.attrs["epoch"]``). Used sequentially or
    from worker processes when --num-workers > 1.
    """
    rel_dir = group["relative_dir"]
    group_dir = group["group_dir"]
    cache_files = group["cache_files"]
    cache_mode = str(getattr(args, "cache_mode", "step"))
    time_key = _cache_mode_spec(cache_mode)["attr"]  # "step" or "epoch"
    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    group_out_dir = out_dir if rel_dir == "." else os.path.join(out_dir, rel_dir)
    os.makedirs(group_out_dir, exist_ok=True)
    tb_dir = os.path.join(group_out_dir, "tb")
    writer = SummaryWriter(tb_dir)
    results = []
    prefix_scores = []

    prefix_mode = str(args.prefix_mode)
    effective_prefix_epsilon = 0.0 if prefix_mode == "soft" else 1.0
    load_obs = bool(getattr(args, "load_cache_obs_features", False))

    # print(f"\n[group] cache_dir={group_dir} (num_files={len(cache_files)})", flush=True)
    for cache_path in cache_files:
        (
            demo_ids,
            index_in_demo,
            actions,
            pred_actions_samples,
            obs_features,
            checkpoint,
            step,
            valid_loss,
            valid_off,
        ) = _load_cache_hdf5(cache_path, load_obs_features=load_obs, mode=cache_mode)
        if step < 0:
            step = len(results) + 1
        seq_summary, per_episode = _compute_from_cache(
            demo_ids=demo_ids,
            index_in_demo=index_in_demo,
            actions=actions,
            pred_actions_samples=pred_actions_samples,
            ta=args.ta,
            num_diffusion_samples=args.num_diffusion_samples,
            prefix_epsilon=effective_prefix_epsilon,
            prefix_tau=args.prefix_tau,
            prefix_epsilon_soft=1.0,
            prefix_mode=prefix_mode,
            prefix_scoring_mode=args.prefix_scoring_mode,
            action_space_config=action_space_config,
            block_scale_quantile=args.block_scale_quantile,
            block_scale_local_progress_window=args.block_scale_local_progress_window,
            block_share_scales_across_arms=args.block_share_scales_across_arms,
            block_scale_method=args.block_scale_method,
            state_db_dir=getattr(args, "state_db_dir", None),
            threshold_quantile=getattr(args, "threshold_quantile", None),
            distance_mode=args.distance_mode,
            mahalanobis_reg=args.mahalanobis_reg,
            chunk_time_agg=args.chunk_time_agg,
            chunk_top_frac=args.chunk_top_frac,
            chunk_quantile=args.chunk_quantile,
            chunk_smooth_max_p=args.chunk_smooth_max_p,
            skip_intermediate_on_pass=args.skip_intermediate_on_pass,
            block_prefix_vote_n=args.block_prefix_vote_n,
            prefix_soft_aggregator=args.prefix_soft_aggregator,
            prefix_soft_lse_tau=args.prefix_soft_lse_tau,
            valid_loss=valid_loss,
            valid_off=valid_off,
        )
        for key, value in seq_summary.items():
            if isinstance(value, int | float | np.floating):
                writer.add_scalar(f"SequentialValid/{key}", float(value), int(step))
        if valid_loss is not None and not np.isnan(valid_loss):
            writer.add_scalar("Valid/Loss", float(valid_loss), int(step))
        if valid_off is not None and not np.isnan(valid_off):
            writer.add_scalar("Valid/Off_Manifold_Norm", float(valid_off), int(step))

        if "PrefixSurvival_Score" in seq_summary:
            prefix_scores.append(float(seq_summary["PrefixSurvival_Score"]))
        # print(
        #     f"\n[cache] group={rel_dir} step={int(step)} file={cache_path}",
        #     flush=True,
        # )
        # print(json.dumps(seq_summary, indent=2, sort_keys=True), flush=True)
        results.append(
            {
                "cache_file": cache_path,
                "checkpoint": checkpoint,
                time_key: int(step),
                "valid_summary": {"Loss": valid_loss, "Off_Manifold_Norm": valid_off},
                "summary": seq_summary,
                "per_episode": per_episode,
            }
        )

    writer.flush()
    writer.close()

    out_json = os.path.join(group_out_dir, "sequential_validation_from_cache_results.json")
    with open(out_json, "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    # print(f"\nSaved TensorBoard logs to: {tb_dir}", flush=True)
    # print(f"Saved JSON results to: {out_json}", flush=True)

    _print_checkpoint_metrics(results, group_label=rel_dir, use_compact=args.use_compact, time_key=time_key)

    steps = [int(x[time_key]) for x in results]
    return {
        "relative_dir": rel_dir,
        "cache_dir": group_dir,
        "output_dir": group_out_dir,
        "num_cache_files": len(cache_files),
        "num_results": len(results),
        f"{time_key}_min": int(min(steps)) if steps else None,
        f"{time_key}_max": int(max(steps)) if steps else None,
        "mean_prefix_survival_score": (float(np.mean(prefix_scores)) if prefix_scores else None),
        "results_json": out_json,
        "tb_dir": tb_dir,
    }


def _cache_group_worker(payload):
    """Picklable entry for ProcessPoolExecutor: (group, args_dict, action_space_config)."""
    group, args_dict, action_space_config = payload
    args = Namespace(**args_dict)
    return _run_single_cache_group(group, args, action_space_config)


def run(args, action_space_config):
    cache_mode = str(getattr(args, "cache_mode", "step"))
    cache_groups = _discover_cache_groups(args.cache_dir, mode=cache_mode)
    if not cache_groups:
        spec = _cache_mode_spec(cache_mode)
        raise RuntimeError(
            f"No cache files found under {args.cache_dir} "
            f"(expected {spec['glob']} in this folder or its subfolders; "
            f"--cache-mode={cache_mode!r})."
        )

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    num_workers = int(getattr(args, "num_workers", 1) or 1)
    num_workers = max(1, min(num_workers, len(cache_groups)))

    args_dict = vars(args).copy()

    if num_workers == 1:
        group_summaries = [_run_single_cache_group(group, args, action_space_config) for group in cache_groups]
    else:
        # print(
        #     f"Processing {len(cache_groups)} cache groups with {num_workers} parallel workers (CPU-bound numpy; "
        #     "ensure OMP/BLAS threads stay low per worker, e.g. OMP_NUM_THREADS=1).",
        #     flush=True,
        # )
        payloads = [(g, args_dict, action_space_config) for g in cache_groups]
        group_summaries = [None] * len(cache_groups)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_cache_group_worker, p): i for i, p in enumerate(payloads)}
            for fut in as_completed(futures):
                idx = futures[fut]
                group_summaries[idx] = fut.result()

    summary_json = os.path.join(out_dir, "sequential_validation_from_cache_group_summary.json")
    with open(summary_json, "w") as f:
        json.dump(_to_jsonable(group_summaries), f, indent=2)
    # print(f"\nSaved group summary JSON to: {summary_json}")


# ---------------------------------------------------------------------------
# Shared argparse helpers
# ---------------------------------------------------------------------------


def add_common_args(parser):
    """Add all CLI args shared between the standard and dex entry scripts."""
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="step",
        choices=sorted(_CACHE_MODE_SPECS.keys()),
        help=(
            "Time-axis convention for the on-disk caches. "
            "'step': discover seqcache_step_*.hdf5 and read f.attrs['step'] "
            "(surval_openpi default). "
            "'epoch': discover seqcache_epoch_*.hdf5 and read f.attrs['epoch'] "
            "(robomimic-side scripts: droid_policy_learning, custom-robomimic). "
            "Wrappers can override the default with parser.set_defaults(cache_mode=...)."
        ),
    )
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--prefix-tau", type=float, default=1.0)
    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="soft",
        choices=["hard", "soft"],
        help=(
            "Prefix-survival aggregation mode. "
            "'hard': timestep fails if >= N blocks exceed eps (cliff, loses 'degree'). "
            "'soft': e_t = N-th smallest per-block soft-prob (= N-th worst block); smoothed hard."
        ),
    )
    parser.add_argument(
        "--prefix-scoring-mode",
        type=str,
        default="cumprod",
        choices=["cumprod", "weighted_sum"],
        help=(
            "Scoring aggregation mode. "
            "'cumprod': prefix-survival via cumprod (hard cutoff after first failure). "
            "'weighted_sum': linear decay weighting score = sum_t (1-t/T)*z_t / sum_t (1-t/T), "
            "where z_t is the hard indicator or soft exponential per timestep."
        ),
    )
    parser.add_argument(
        "--ta",
        type=int,
        default=None,
        help="Optional Ta used when slicing cached per-step errors and row stride.",
    )
    parser.add_argument(
        "--block-scale-quantile",
        type=float,
        default=0.99,
        help="Quantile q for expert-expert local-min disagreement scales.",
    )
    parser.add_argument(
        "--block-scale-local-progress-window",
        type=float,
        default=0.05,
        help="Normalized progress window w for local cross-demo matching.",
    )
    parser.add_argument(
        "--block-scale-method",
        type=str,
        default="inter_demo",
        choices=["inter_demo", "intra_demo", "local", "global_db", "intra_demo_sc"],
        help=(
            "Method for picking the per-block divisor S_g used in scaled errors. "
            "'inter_demo': cross-demo local disagreement at similar progress (default). "
            "'intra_demo': within-demo consecutive action differences. "
            "'local': per-state local thresholds saved by compute_local_thresholds.py "
            "with --scale-source=local (--state-db-dir + --threshold-quantile required; "
            "only supports --distance-mode=L2). "
            "'global_db': per-block global thresholds from the same 'local' artifact. "
            "'intra_demo_sc': per-state, state-conditional intra-demo chunk-motion "
            "thresholds saved by compute_local_thresholds.py with "
            "--scale-source=intra_demo_sc. Same requirements as 'local', but reads "
            "from the 'thresholds_intra_demo_sc/' subdir so it can coexist with "
            "'local' on disk."
        ),
    )
    parser.add_argument(
        "--state-db-dir",
        type=str,
        default=None,
        help=(
            "Path to a state DB directory produced by build_state_db.py "
            "(must contain a `thresholds/` or `thresholds_intra_demo_sc/` subdir "
            "from compute_local_thresholds.py). "
            "Required when --block-scale-method is 'local' or 'global_db'."
        ),
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=None,
        help=(
            "Which quantile column of the saved local-threshold map to use as the divisor "
            "(must match one of the quantiles stored at compute_local_thresholds time, "
            "e.g. 0.5 / 0.75 / 0.9 / 0.95 / 0.99). Required when --block-scale-method is "
            "'local' or 'global_db'."
        ),
    )
    parser.add_argument(
        "--block-share-scales-across-arms",
        dest="block_share_scales_across_arms",
        action="store_true",
        help="Share pos/rot/finger scales across left/right arms.",
    )
    parser.add_argument(
        "--block-no-share-scales-across-arms",
        dest="block_share_scales_across_arms",
        action="store_false",
        help="Use separate scales per arm/block.",
    )
    parser.set_defaults(block_share_scales_across_arms=False)
    parser.add_argument(
        "--distance-mode",
        type=str,
        default="L2",
        choices=["L2", "mahalanobis"],
        help="Block distance: L2 norm or Mahalanobis (expert Sigma^{-1/2} whitening).",
    )
    parser.add_argument(
        "--mahalanobis-reg",
        type=float,
        default=1e-6,
        help="Regularization added to expert covariance for Mahalanobis (Sigma + reg*I).",
    )
    parser.add_argument(
        "--chunk-time-agg",
        type=str,
        default="top_k_mean",
        choices=["max", "top_k_mean", "quantile", "smooth_max"],
        help="Aggregation over chunk time axis: max, top_k_mean, quantile, or smooth_max.",
    )
    parser.add_argument(
        "--chunk-top-frac",
        type=float,
        default=0.5,
        help="For top_k_mean: fraction of T to average (e.g. 0.2 = top-20%% mean).",
    )
    parser.add_argument(
        "--chunk-quantile",
        type=float,
        default=0.5,
        help="For quantile: q in (0,1] (e.g. 0.9 = 90th percentile over chunk).",
    )
    parser.add_argument(
        "--chunk-smooth-max-p",
        type=float,
        default=8.0,
        help="For smooth_max: p-norm exponent (larger => closer to max).",
    )
    parser.add_argument(
        "--skip-intermediate-on-pass",
        dest="skip_intermediate_on_pass",
        action="store_true",
        help=(
            "When enabled, only evaluate chunks at non-overlapping stride positions "
            "(row_stride=ta_eff). If a chunk passes, the intermediate states within "
            "it are skipped. When disabled (--no-skip-intermediate-on-pass), every "
            "demo step is evaluated as a chunk start (row_stride=1)."
        ),
    )
    parser.add_argument(
        "--no-skip-intermediate-on-pass",
        dest="skip_intermediate_on_pass",
        action="store_false",
        help="Evaluate every demo step as a chunk start (row_stride=1).",
    )
    parser.set_defaults(skip_intermediate_on_pass=True)
    parser.add_argument(
        "--block-prefix-vote-n",
        type=int,
        default=None,
        help=(
            "Hard: timestep fails if >= N blocks violate (err_b >= epsilon). "
            "Soft (with --prefix-soft-aggregator=worst_n): e_t is the N-th smallest "
            "(= N-th worst) per-block soft prob p_b(t); aligned with hard's semantic. "
            "For G=2, N=2 this reduces to max(p_pos, p_rot). "
            "Ignored by 'logsumexp' and 'product' aggregators. "
            "Default: strict majority ceil(num_blocks/2)."
        ),
    )
    parser.add_argument(
        "--prefix-soft-aggregator",
        type=str,
        default="logsumexp",
        choices=["worst_n", "logsumexp", "product"],
        help=(
            "How to collapse per-block soft probabilities p_b(t) into a single "
            "survival value e_t (only used when --prefix-mode=soft). "
            "'worst_n': N-th worst block's soft prob (current default; uses "
            "--block-prefix-vote-n; coarse integer K). "
            "'logsumexp': Boltzmann smooth-min "
            "v = -(1/τ) log mean_b exp(-τ p_b); τ→∞ ⇒ min (worst block), "
            "τ→0 ⇒ mean (uses --prefix-soft-lse-tau). "
            "'product': Π_b p_b under independence; hyperparameter-free, "
            "consistent with SURVAL's prefix survival product, may over-penalize "
            "when blocks are strongly correlated."
        ),
    )
    parser.add_argument(
        "--prefix-soft-lse-tau",
        type=float,
        default=1.0,
        help=(
            "Temperature τ (>0) for --prefix-soft-aggregator=logsumexp. "
            "Larger τ ⇒ closer to min (worst block); smaller τ ⇒ closer to mean. "
            "Continuous analog of the integer K in worst_n."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for cache *groups* (subdirs each with "
            "seqcache_*.hdf5). Default 1 (sequential). Set to the number of CPUs "
            "you allocate (e.g. SLURM --cpus-per-task) when you have many groups; "
            "keep OMP_NUM_THREADS=1 per process to avoid BLAS oversubscription."
        ),
    )
    parser.add_argument(
        "--load-cache-obs-features",
        action="store_true",
        help=(
            "Load obs_features from HDF5 cache (not used for metrics; only for debugging). "
            "Skipping them (default) reduces disk I/O when caches are large."
        ),
    )
    parser.add_argument(
        "--use-compact",
        action="store_true",
        help="Use compact JSON output for checkpoint metrics.",
    )
    parser.add_argument(
        "--no-use-compact",
        action="store_false",
        help="Use human-readable table output for checkpoint metrics.",
    )
    parser.set_defaults(use_compact=False)


def validate_common_args(args, num_blocks):
    """Validate args that are shared between both scripts. num_blocks = number of action blocks."""
    if args.num_diffusion_samples <= 0:
        raise ValueError("--num-diffusion-samples must be positive.")
    if args.ta is not None and args.ta <= 0:
        raise ValueError("--ta must be positive.")
    if not (0.0 < args.block_scale_quantile <= 1.0):
        raise ValueError("--block-scale-quantile must be in (0, 1].")
    if not (0.0 <= args.block_scale_local_progress_window <= 1.0):
        raise ValueError("--block-scale-local-progress-window must be in [0, 1].")
    if not (0.0 < args.chunk_top_frac <= 1.0):
        raise ValueError("--chunk-top-frac must be in (0, 1].")
    if not (0.0 < args.chunk_quantile <= 1.0):
        raise ValueError("--chunk-quantile must be in (0, 1].")
    if args.chunk_smooth_max_p <= 0.0:
        raise ValueError("--chunk-smooth-max-p must be positive.")
    vn = getattr(args, "block_prefix_vote_n", None)
    if vn is not None and (vn < 1 or vn > num_blocks):
        raise ValueError(f"--block-prefix-vote-n must be between 1 and {num_blocks} (num action blocks).")
    nw = int(getattr(args, "num_workers", 1) or 1)
    if nw < 1:
        raise ValueError("--num-workers must be >= 1.")
    method = str(getattr(args, "block_scale_method", "inter_demo"))
    if method in ("local", "global_db", "intra_demo_sc"):
        if not getattr(args, "state_db_dir", None):
            raise ValueError(f"--state-db-dir is required when --block-scale-method={method!r}")
        if not os.path.isdir(args.state_db_dir):
            raise FileNotFoundError(f"--state-db-dir does not exist: {args.state_db_dir}")
        sub = _threshold_subdir_for_method(method)
        if not os.path.isdir(os.path.join(args.state_db_dir, sub)):
            raise FileNotFoundError(
                f"{sub}/ subdir not found under {args.state_db_dir}; "
                "run scripts/compute_local_thresholds.py first "
                f"(with --scale-source={'intra_demo_sc' if method == 'intra_demo_sc' else 'local'})."
            )
        if getattr(args, "threshold_quantile", None) is None:
            raise ValueError(f"--threshold-quantile is required when --block-scale-method={method!r}")
        if not (0.0 < float(args.threshold_quantile) <= 1.0):
            raise ValueError("--threshold-quantile must be in (0, 1].")
        if str(getattr(args, "distance_mode", "L2")) != "L2":
            raise ValueError(
                f"--block-scale-method={method!r} only supports --distance-mode=L2 "
                "(local thresholds are computed in L2)."
            )
