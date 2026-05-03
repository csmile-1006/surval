"""
SURVAL — rollout-free sequential validation with regime-conditioned scale lookup.

This module wraps the core SURVAL evaluation pipeline, replacing the global
quantile scale with a regime-conditioned scale that adapts to whether a given
timestep is in a low-motion or high-motion regime.

Key change from the base implementation
-----------------------------------------
In the original code, each block has a single scalar scale:
    error_scaled = dist(pred, gt) / scales[block_name]

Here, scale depends on the expert action's magnitude regime at each timestep:
    regime   = classify_regime(|expert_action[group_idx]|, boundaries[block_name])
    scale_t  = scales[block_name][regime]
    error_scaled = dist(pred, gt) / scale_t

The survival product, worst-K aggregation, and trajectory scoring logic
are identical to the base implementation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .scale_estimation import _REGIME_LABELS
from .scale_estimation import classify_regime_array
from .scale_estimation import compute_regime_scales

# ---------------------------------------------------------------------------
# Regime-conditioned scale lookup
# ---------------------------------------------------------------------------


def get_scale(
    group_name: str,
    expert_action: np.ndarray,
    scales: dict[str, dict[str, float]],
    boundaries: dict[str, list[float]],
    group_idx: int,
) -> float:
    """
    Look up the regime-conditioned scale for a single expert action timestep.

    Args:
        group_name: block/group name (e.g. 'joint_0').
        expert_action: (A,) expert action at the current timestep.
        scales: {group_name: {regime_label: float}} from compute_regime_scales().
        boundaries: {group_name: list[float]} from compute_regime_scales().
        group_idx: action dimension index for this group.

    Returns:
        Scale value (float) for the current regime.

    Example:
        >>> import numpy as np
        >>> scales = {'joint_0': {'low': 0.05, 'high': 0.2}}
        >>> boundaries = {'joint_0': [0.3]}
        >>> get_scale('joint_0', np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), scales, boundaries, 0)
        0.05
    """
    mag = abs(float(expert_action[group_idx]))
    n_regimes = len(boundaries[group_name]) + 1
    labels = _REGIME_LABELS[n_regimes]
    idx = int(np.searchsorted(boundaries[group_name], mag, side="right"))
    return float(scales[group_name][labels[idx]])


def get_scale_array(
    group_name: str,
    expert_actions: np.ndarray,
    scales: dict[str, dict[str, float]],
    boundaries: dict[str, list[float]],
    group_idx: int,
) -> np.ndarray:
    """
    Vectorized regime-conditioned scale lookup for an array of expert actions.

    Args:
        group_name: block/group name.
        expert_actions: (..., A) array of expert actions.
        scales: {group_name: {regime_label: float}}.
        boundaries: {group_name: list[float]}.
        group_idx: action dimension index for this group.

    Returns:
        scale_arr: float32 array of shape (...,) with per-timestep scales.

    Example:
        >>> import numpy as np
        >>> scales = {'joint_0': {'low': 0.05, 'high': 0.2}}
        >>> boundaries = {'joint_0': [0.3]}
        >>> ea = np.array([[0.1, 0.0], [0.5, 0.0]])  # shape (2, 2)
        >>> s = get_scale_array('joint_0', ea, scales, boundaries, group_idx=0)
        >>> list(s)
        [0.05, 0.2]
    """
    mag = np.abs(expert_actions[..., group_idx])  # (...,)
    regime_idxs = classify_regime_array(mag, boundaries[group_name])  # (...,) int

    n_regimes = len(boundaries[group_name]) + 1
    labels = _REGIME_LABELS[n_regimes]
    scale_values = np.array([scales[group_name][lbl] for lbl in labels], dtype=np.float32)

    return scale_values[regime_idxs]  # (...,)


# ---------------------------------------------------------------------------
# Regime-conditioned per-block step errors
# ---------------------------------------------------------------------------


def compute_per_block_scaled_step_errors_regime(
    pred: np.ndarray,
    gt: np.ndarray,
    scales: dict[str, dict[str, float]],
    boundaries: dict[str, list[float]],
    group_config: dict[str, int],
    block_names: list[str],
    gripper_idx: int | None = 7,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """
    Per-block scaled step errors using regime-conditioned scales.

    For each group at each timestep, the scale is determined by the expert
    action's magnitude regime rather than a fixed global value.

    Args:
        pred: [s, n, t, a] predicted actions (s diffusion samples).
        gt:   [1, n, t, a] expert ground-truth actions.
        scales: {group_name: {regime_label: float}} from compute_regime_scales().
        boundaries: {group_name: list[float]} from compute_regime_scales().
        group_config: {group_name: action_index}, gripper excluded.
        block_names: ordered list of block names to process.
        gripper_idx: action dimension for gripper; if not None, appends a gripper
            gate error as the last block.  Set to None to skip gripper evaluation.
        gripper_threshold: open/close decision threshold for gripper gate.

    Returns:
        step_err_pb: [s, n, t, b] float32 array of scaled per-block errors.
            b = len(block_names) + (1 if gripper_idx is not None else 0).

    Example:
        >>> import numpy as np
        >>> s_dim, n_dim, t_dim, a_dim = 2, 3, 10, 8
        >>> pred = np.random.randn(s_dim, n_dim, t_dim, a_dim).astype(np.float32)
        >>> gt   = np.random.randn(1, n_dim, t_dim, a_dim).astype(np.float32)
        >>> scales = {f'joint_{i}': {'low': 0.1, 'high': 0.3} for i in range(7)}
        >>> boundaries = {f'joint_{i}': [0.2] for i in range(7)}
        >>> group_cfg = {f'joint_{i}': i for i in range(7)}
        >>> err = compute_per_block_scaled_step_errors_regime(
        ...     pred, gt, scales, boundaries, group_cfg, list(group_cfg.keys())
        ... )
        >>> assert err.shape == (s_dim, n_dim, t_dim, 7)
    """
    n_action_dims = pred.shape[-1]
    block_errors: list[np.ndarray] = []

    for k in block_names:
        idx = group_config[k]
        # Absolute difference for 1D group (magnitude == |delta|)
        delta = np.abs(pred[..., idx] - gt[..., idx])  # [s, n, t]

        # Per-timestep regime scale from expert action — broadcast over s
        # gt[0] shape: [n, t, a]  →  scale_nt shape: [n, t]
        scale_nt = get_scale_array(k, gt[0], scales, boundaries, group_idx=idx)  # [n, t]
        scale_snt = scale_nt[np.newaxis, :, :]  # [1, n, t]

        block_errors.append((delta / np.maximum(scale_snt, 1e-8)).astype(np.float32))

    if gripper_idx is not None and gripper_idx < n_action_dims:
        # Gripper: binary pass/fail — error = 0 if same state, 1 if different
        pred_state = pred[..., gripper_idx] > gripper_threshold    # [s, n, t] bool
        gt_state = gt[0, :, :, gripper_idx] > gripper_threshold    # [n, t] bool
        gate = (pred_state == gt_state[np.newaxis]).astype(np.float32)  # [s, n, t]
        block_errors.append(1.0 - gate)  # 0 = correct, 1 = wrong

    return np.stack(block_errors, axis=-1).astype(np.float32)  # [s, n, t, b]


# ---------------------------------------------------------------------------
# Trajectory-level survival scoring
# ---------------------------------------------------------------------------


def compute_trajectory_score(
    err_mat: np.ndarray,
    *,
    epsilon: float = 1.0,
    use_soft: bool = True,
    tau: float | None = None,
    epsilon_soft: float = 1.0,
    vote_n: int | None = None,
    scoring_mode: str = "mean",
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute trajectory-level SURVAL score from per-block error matrix.

    Args:
        err_mat: [b, t] float array of scaled per-block errors (chunk-aggregated).
        epsilon: hard gate threshold (in units of scale).
        use_soft: use soft gates instead of hard gates.
        tau: soft gate temperature; if None, estimated from IQR of err_mat.
        epsilon_soft: threshold for soft gate (errors below this pass with ~1).
        vote_n: number of blocks that must fail to kill survival.
            Default: strict majority (b // 2 + 1).
        scoring_mode: 'mean' averages cumprod(e_t); 'weighted_sum' uses
            linearly decreasing weights (earlier timesteps count more).

    Returns:
        score: scalar trajectory score in [0, 1].
        e_t: [t] per-timestep gate value.
        s: [t] cumulative survival product.

    Example:
        >>> import numpy as np
        >>> err = np.random.rand(4, 20)  # 4 blocks, 20 timesteps
        >>> score, e_t, s = compute_trajectory_score(err)
        >>> assert 0.0 <= score <= 1.0
    """
    num_blocks, t_len = err_mat.shape
    if t_len == 0:
        return 0.0, np.zeros(0), np.zeros(0)

    _vote_n = int(vote_n) if vote_n is not None else max(1, num_blocks // 2 + 1)
    _vote_n = max(1, min(_vote_n, num_blocks))

    if use_soft:
        if tau is not None:
            tau_ep = float(tau)
        else:
            pooled = err_mat.ravel()
            tau_ep = max(float(np.percentile(pooled, 75) - np.percentile(pooled, 25)), 1e-6)

        e_t = np.zeros(t_len, dtype=np.float64)
        for t in range(t_len):
            p_b = np.exp(
                -np.maximum(0.0, err_mat[:, t] - float(epsilon_soft)) / max(tau_ep, 1e-8)
            )
            n_use = min(_vote_n, num_blocks)
            smallest_n = np.partition(p_b, n_use - 1)[:n_use]
            e_t[t] = float(np.mean(smallest_n))
    else:
        e_t = np.ones(t_len, dtype=np.float64)
        for t in range(t_len):
            num_violate = int(np.sum(err_mat[:, t] >= float(epsilon)))
            if num_violate >= _vote_n:
                e_t[t] = 0.0

    s = np.cumprod(e_t)

    if scoring_mode == "weighted_sum":
        w = 1.0 - np.arange(t_len, dtype=np.float64) / max(t_len, 1)
        score = float(np.dot(w, s))
    else:
        score = float(np.mean(s))

    return score, e_t, s


# ---------------------------------------------------------------------------
# End-to-end SURVAL with regime scales
# ---------------------------------------------------------------------------


def evaluate_with_regime_scales(
    pred_actions: np.ndarray,
    expert_actions: np.ndarray,
    demo_ids: np.ndarray,
    index_in_demo: np.ndarray,
    group_config: dict[str, int],
    *,
    n_regimes: int = 2,
    scale_quantile: float = 0.99,
    min_samples_per_regime: int = 30,
    gripper_idx: int | None = 7,
    gripper_threshold: float = 0.5,
    epsilon: float = 1.0,
    use_soft: bool = True,
    tau: float | None = None,
    epsilon_soft: float = 1.0,
    vote_n: int | None = None,
    scoring_mode: str = "mean",
) -> tuple[dict, list[dict]]:
    """
    Full SURVAL evaluation pipeline with regime-conditioned scale estimation.

    Args:
        pred_actions: [s, n, t, a] predicted action samples.
        expert_actions: [n, t, a] ground-truth expert actions.
        demo_ids: [n] string array of demo identifiers.
        index_in_demo: [n] int array of timestep index within each demo.
        group_config: {group_name: action_index} for joint groups (gripper excluded).
        n_regimes: number of motion regimes for scale estimation.
        scale_quantile: quantile for regime scale computation.
        min_samples_per_regime: fallback threshold for per-regime scale estimation.
        gripper_idx: action index for gripper gate; None to disable.
        gripper_threshold: gripper open/close threshold.
        epsilon: hard gate threshold (fraction of scale).
        use_soft: use soft gates.
        tau: soft gate temperature (None = auto from IQR).
        epsilon_soft: soft gate threshold.
        vote_n: blocks that must fail to terminate survival.
        scoring_mode: 'mean' or 'weighted_sum'.

    Returns:
        summary: aggregate metrics dict.
        per_episode: list of per-demo dicts with individual scores.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> s_dim, n_dim, t_dim, a_dim = 2, 10, 20, 8
        >>> pred = rng.standard_normal((s_dim, n_dim, t_dim, a_dim)).astype(np.float32)
        >>> gt   = rng.standard_normal((n_dim, t_dim, a_dim)).astype(np.float32)
        >>> demo_ids = np.array([f'demo_{i // 2}' for i in range(n_dim)])
        >>> idx_in_demo = np.array([i % 2 for i in range(n_dim)], dtype=np.int64)
        >>> group_cfg = {f'joint_{i}': i for i in range(7)}
        >>> summary, eps = evaluate_with_regime_scales(pred, gt, demo_ids, idx_in_demo, group_cfg)
        >>> assert 'PrefixSurvival_Score' in summary
    """
    num_s, num_n, num_t, num_a = pred_actions.shape
    block_names = list(group_config.keys())

    # -------------------------------------------------------------------------
    # Reconstruct per-demo expert trajectories for scale estimation
    # -------------------------------------------------------------------------
    by_demo: dict[str, dict[int, np.ndarray]] = {}
    for i in range(num_n):
        did = str(demo_ids[i])
        t_idx = int(index_in_demo[i])
        by_demo.setdefault(did, {})[t_idx] = expert_actions[i, 0, :num_a].astype(np.float32)

    demo_trajectories: list[np.ndarray] = []
    for did in sorted(by_demo.keys()):
        sorted_steps = sorted(by_demo[did].keys())
        traj = np.stack([by_demo[did][k] for k in sorted_steps], axis=0)
        demo_trajectories.append(traj)

    if len(demo_trajectories) < 2:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []

    # -------------------------------------------------------------------------
    # Regime-based scale estimation
    # -------------------------------------------------------------------------
    scales, boundaries = compute_regime_scales(
        expert_demos=demo_trajectories,
        group_config=group_config,
        n_regimes=n_regimes,
        quantile=scale_quantile,
        min_samples_per_regime=min_samples_per_regime,
    )

    # -------------------------------------------------------------------------
    # Per-block scaled step errors: [s, n, t, b]
    # -------------------------------------------------------------------------
    gt = expert_actions[np.newaxis, :, :, :]  # [1, n, t, a]
    step_err_pb = compute_per_block_scaled_step_errors_regime(
        pred=pred_actions,
        gt=gt,
        scales=scales,
        boundaries=boundaries,
        group_config=group_config,
        block_names=block_names,
        gripper_idx=gripper_idx,
        gripper_threshold=gripper_threshold,
    )  # [s, n, t, b]

    num_blocks = step_err_pb.shape[-1]

    # -------------------------------------------------------------------------
    # Chunk-time aggregation: max over t → [s, n, b], then mean over s → [n, b]
    # -------------------------------------------------------------------------
    chunk_max = np.max(step_err_pb, axis=2)    # [s, n, b]
    chunk_mean_s = np.mean(chunk_max, axis=0)  # [n, b]

    # -------------------------------------------------------------------------
    # Group rows by demo, compute per-demo trajectory score
    # -------------------------------------------------------------------------
    by_demo_b: list[dict] = []
    for b in range(num_blocks):
        rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i in range(num_n):
            did = str(demo_ids[i])
            tidx = int(index_in_demo[i])
            rows[did].append((tidx, float(chunk_mean_s[i, b])))
        for did, row_list in rows.items():
            row_list.sort(key=lambda x: x[0])
        by_demo_b.append(dict(rows))

    demo_ids_sorted = sorted(
        set.intersection(*(set(d.keys()) for d in by_demo_b)) if by_demo_b else set()
    )
    if not demo_ids_sorted:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []

    all_scores: list[float] = []
    all_prefix_lengths: list[float] = []
    per_episode: list[dict] = []

    for did in demo_ids_sorted:
        pairs0 = sorted(by_demo_b[0][did], key=lambda x: x[0])
        t_len = len(pairs0)
        if t_len == 0:
            continue
        idxs = [p[0] for p in pairs0]

        err_mat = np.zeros((num_blocks, t_len), dtype=np.float64)
        aligned = True
        for b in range(num_blocks):
            pb = sorted(by_demo_b[b][did], key=lambda x: x[0])
            if len(pb) != t_len or [x[0] for x in pb] != idxs:
                aligned = False
                break
            err_mat[b, :] = np.maximum([float(x[1]) for x in pb], 0.0)
        if not aligned:
            continue

        score, e_t, s = compute_trajectory_score(
            err_mat,
            epsilon=epsilon,
            use_soft=use_soft,
            tau=tau,
            epsilon_soft=epsilon_soft,
            vote_n=vote_n,
            scoring_mode=scoring_mode,
        )
        all_scores.append(score)
        all_prefix_lengths.append(float(np.mean(s)))
        per_episode.append(
            {
                "demo_id": did,
                "PrefixSurvival_Score": score,
                "PrefixSurvival_Loss": 1.0 - score,
                "matched_prefix_length": float(np.mean(s)),
                "mean_block_error": float(np.mean(err_mat)),
                "BlockPrefix_e_t": e_t.tolist(),
                "BlockPrefix_s_cumprod": s.tolist(),
            }
        )

    if not all_scores:
        return {"PrefixSurvival_Score": 0.0, "PrefixSurvival_Loss": 1.0}, []

    mean_score = float(np.mean(all_scores))
    summary: dict = {
        "PrefixSurvival_Score": mean_score,
        "PrefixSurvival_Loss": 1.0 - mean_score,
        "PrefixSurvival_MatchedPrefixLen_MeanPerTraj": float(np.mean(all_prefix_lengths)),
        "n_regimes": n_regimes,
        "scale_quantile": scale_quantile,
    }

    # Expose per-group per-regime scales and boundaries in summary
    for g, regime_scales in scales.items():
        for lbl, val in regime_scales.items():
            summary[f"RegimeScale_{g}_{lbl}"] = float(val)
    for g, bvals in boundaries.items():
        for i, bval in enumerate(bvals):
            summary[f"RegimeBoundary_{g}_{i}"] = float(bval)

    return summary, per_episode
