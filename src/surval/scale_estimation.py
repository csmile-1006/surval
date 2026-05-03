"""
Regime-based scale estimation for SURVAL.

Estimates per-group, per-regime tolerance scales from expert demonstrations.
Instead of a single global quantile scale, computes separate scales for
low-motion and high-motion regimes based on action magnitude.

Algorithm overview
------------------
1. Compute per-group action magnitudes (|a[g]|) across all demos.
2. Determine regime boundaries via quantile splits of those magnitudes.
3. Collect per-regime errors using one of two methods:
   - ``inter_demo``: pairwise expert consistency errors within a progress
     window (row/col minimums, same as the base inter_demo scale method).
   - ``intra_demo``: within-demo action differences at stride ``ta``.
4. Return per-regime quantile scales; fall back to global scale when a
   regime has too few samples.
"""

from __future__ import annotations

import numpy as np


# Regime label lists indexed by n_regimes.
# n_regimes=1 has a single "all" regime — equivalent to the standard global scale.
_REGIME_LABELS: dict[int, list[str]] = {
    1: ["all"],
    2: ["low", "high"],
    3: ["low", "mid", "high"],
}


def compute_group_magnitudes(
    expert_demos: list[np.ndarray],
    group_config: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Compute per-group action magnitudes pooled across all demos and timesteps.

    Each group is 1-D, so magnitude = |a[group_idx]|.

    Args:
        expert_demos: list of N demo arrays, each shape (T_n, A).
        group_config: {group_name: action_index}.
            e.g., {'joint_0': 0, ..., 'joint_6': 6}

    Returns:
        magnitudes: {group_name: flat float32 array of |a[g]| values}
            Shape of each array: (sum of T_n,).

    Example:
        >>> demos = [np.random.randn(50, 8).astype(np.float32) for _ in range(3)]
        >>> mags = compute_group_magnitudes(demos, {'joint_0': 0})
        >>> assert mags['joint_0'].shape == (150,)
    """
    pools: dict[str, list[np.ndarray]] = {g: [] for g in group_config}
    for demo in expert_demos:
        if demo.shape[0] == 0:
            continue
        for g, idx in group_config.items():
            pools[g].append(np.abs(demo[:, idx]).astype(np.float32))  # (T_n,)

    return {
        g: np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        for g, parts in pools.items()
    }


def compute_regime_boundaries(
    magnitudes: np.ndarray,
    n_regimes: int = 2,
) -> list[float]:
    """
    Compute regime boundary values using quantile splits of the magnitude distribution.

    Args:
        magnitudes: flat float array of action magnitudes.
        n_regimes: number of regimes.
            2 → 1 boundary at median (50th percentile).
            3 → 2 boundaries at 33rd and 67th percentile.

    Returns:
        boundaries: list of (n_regimes - 1) sorted float boundary values.

    Example:
        >>> import numpy as np
        >>> b = compute_regime_boundaries(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), n_regimes=2)
        >>> assert len(b) == 1
        >>> assert abs(b[0] - 0.3) < 0.05
    """
    if n_regimes not in _REGIME_LABELS:
        raise ValueError(f"n_regimes must be one of {list(_REGIME_LABELS)}, got {n_regimes}")
    if n_regimes == 1:
        return []  # no boundaries; every magnitude falls in the single "all" regime
    if magnitudes.size == 0:
        return [0.0] * (n_regimes - 1)

    percentiles = {2: [50.0], 3: [33.0, 67.0]}[n_regimes]
    return [float(np.percentile(magnitudes, q)) for q in percentiles]


def classify_regime(mag: float, boundaries: list[float]) -> str:
    """
    Classify a single magnitude value into a regime label.

    Args:
        mag: action magnitude (scalar, non-negative).
        boundaries: sorted list of (n_regimes - 1) boundary values.

    Returns:
        Regime label string: 'low', 'mid', or 'high'.

    Example:
        >>> classify_regime(0.1, [0.3])
        'low'
        >>> classify_regime(0.5, [0.3])
        'high'
        >>> classify_regime(0.5, [0.3, 0.7])
        'mid'
    """
    n_regimes = len(boundaries) + 1
    labels = _REGIME_LABELS[n_regimes]
    idx = int(np.searchsorted(boundaries, mag, side="right"))
    return labels[idx]


def classify_regime_array(
    magnitudes: np.ndarray,
    boundaries: list[float],
) -> np.ndarray:
    """
    Vectorized regime classification for an array of magnitudes.

    Args:
        magnitudes: float array of any shape.
        boundaries: sorted list of (n_regimes - 1) boundary values.

    Returns:
        Integer regime indices (0 = low, 1 = mid/high, ...), same shape as magnitudes.

    Example:
        >>> import numpy as np
        >>> idxs = classify_regime_array(np.array([0.1, 0.5]), [0.3])
        >>> list(idxs)
        [0, 1]
    """
    return np.searchsorted(boundaries, magnitudes, side="right").astype(np.int32)


# ---------------------------------------------------------------------------
# Error pool builders
# ---------------------------------------------------------------------------


def _collect_errors_inter_demo(
    expert_demos: list[np.ndarray],
    group_config: dict[str, int],
    boundaries: dict[str, list[float]],
    labels: list[str],
    local_progress_window: float = 0.08,
) -> dict[str, dict[str, list[float]]]:
    """
    Collect per-regime pairwise errors using progress-window matching (inter-demo).

    For each pair of demonstrations, aligns timesteps by normalized progress
    within a local window and takes row/column minimums — identical to the
    approach used in ``_estimate_block_scales_from_expert_demos``.

    Regime is classified by the **anchor** action's magnitude (the timestep for
    which the row/col minimum is taken), which avoids a double-loop over the
    full [Ti, Tj] grid.

    Args:
        expert_demos: list of demo arrays, each (T_n, A).
        group_config: {group_name: action_index}.
        boundaries: {group_name: list[float]} regime boundaries per group.
        labels: ordered regime label list, e.g. ['low', 'high'].
        local_progress_window: max normalized-progress gap for matching.

    Returns:
        error_pools: {group_name: {regime_label: list[float]}}
    """
    error_pools: dict[str, dict[str, list[float]]] = {
        g: {lbl: [] for lbl in labels} for g in group_config
    }
    n_demos = len(expert_demos)

    for i in range(n_demos):
        a_i = expert_demos[i]  # (Ti, A)
        ti = a_i.shape[0]
        if ti < 1:
            continue
        p_i = np.linspace(0.0, 1.0, num=ti, dtype=np.float32)

        for j in range(i + 1, n_demos):
            a_j = expert_demos[j]  # (Tj, A)
            tj = a_j.shape[0]
            if tj < 1:
                continue
            p_j = np.linspace(0.0, 1.0, num=tj, dtype=np.float32)

            progress_diff = np.abs(p_i[:, None] - p_j[None, :])  # [Ti, Tj]
            local_mask = progress_diff <= float(local_progress_window)
            if not np.any(local_mask):
                continue

            for g, idx in group_config.items():
                ai_g = a_i[:, idx]  # (Ti,)
                aj_g = a_j[:, idx]  # (Tj,)

                err_mat = np.abs(ai_g[:, None] - aj_g[None, :])  # [Ti, Tj]
                masked = np.where(local_mask, err_mat, np.inf)

                row_min = np.min(masked, axis=1)  # [Ti] — best j match for each i
                col_min = np.min(masked, axis=0)  # [Tj] — best i match for each j

                # Classify row minimums by a_i magnitude
                valid_row = np.isfinite(row_min)
                if np.any(valid_row):
                    mags = np.abs(ai_g[valid_row])
                    errs = row_min[valid_row]
                    r_idxs = classify_regime_array(mags, boundaries[g])
                    for r_idx, lbl in enumerate(labels):
                        sel = r_idxs == r_idx
                        if np.any(sel):
                            error_pools[g][lbl].extend(errs[sel].tolist())

                # Classify col minimums by a_j magnitude
                valid_col = np.isfinite(col_min)
                if np.any(valid_col):
                    mags = np.abs(aj_g[valid_col])
                    errs = col_min[valid_col]
                    r_idxs = classify_regime_array(mags, boundaries[g])
                    for r_idx, lbl in enumerate(labels):
                        sel = r_idxs == r_idx
                        if np.any(sel):
                            error_pools[g][lbl].extend(errs[sel].tolist())

    return error_pools


def _collect_errors_intra_demo(
    expert_demos: list[np.ndarray],
    group_config: dict[str, int],
    boundaries: dict[str, list[float]],
    labels: list[str],
    ta: int = 1,
) -> dict[str, dict[str, list[float]]]:
    """
    Collect per-regime errors from within-demo action differences (intra-demo).

    For each demo, computes ``|a[t+ta, g] - a[t, g]|`` and classifies the error
    by the magnitude of the **current** timestep ``|a[t, g]|``.

    Args:
        expert_demos: list of demo arrays, each (T_n, A).
        group_config: {group_name: action_index}.
        boundaries: {group_name: list[float]} regime boundaries per group.
        labels: ordered regime label list.
        ta: stride for within-demo differences (action chunk horizon).

    Returns:
        error_pools: {group_name: {regime_label: list[float]}}
    """
    error_pools: dict[str, dict[str, list[float]]] = {
        g: {lbl: [] for lbl in labels} for g in group_config
    }

    for traj in expert_demos:
        t_len = traj.shape[0]
        if ta >= t_len:
            continue

        a_t = traj[:-ta]   # (T-ta, A) — current timestep
        a_ta = traj[ta:]   # (T-ta, A) — timestep ta ahead

        for g, idx in group_config.items():
            err = np.abs(a_t[:, idx] - a_ta[:, idx]).astype(np.float32)  # (T-ta,)
            mag = np.abs(a_t[:, idx])  # magnitude at current timestep

            r_idxs = classify_regime_array(mag, boundaries[g])  # (T-ta,)
            for r_idx, lbl in enumerate(labels):
                sel = r_idxs == r_idx
                if np.any(sel):
                    error_pools[g][lbl].extend(err[sel].tolist())

    return error_pools


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_regime_scales(
    expert_demos: list[np.ndarray],
    group_config: dict[str, int],
    n_regimes: int = 2,
    quantile: float = 0.99,
    min_samples_per_regime: int = 30,
    method: str = "inter_demo",
    local_progress_window: float = 0.08,
    ta: int = 1,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
    """
    Compute regime-conditioned scale estimates from expert demonstrations.

    Errors are collected by one of two methods:

    - ``"inter_demo"`` (default): pairwise expert consistency errors within a
      normalized-progress window, with row/col minimums — identical structure
      to the base ``inter_demo`` block-scale method.  Regime is classified by
      the anchor timestep's action magnitude.
    - ``"intra_demo"``: within-demo action differences ``|a[t+ta] - a[t]|``,
      classified by the magnitude of the current timestep ``|a[t]|``.

    A global (regime-agnostic) fallback scale is used for any regime whose
    error pool is smaller than ``min_samples_per_regime``.

    Args:
        expert_demos: list of N demo arrays, each shape (T_n, A).
            Gripper should be excluded from group_config.
        group_config: {group_name: action_index}, gripper excluded.
            e.g., {'joint_0': 0, ..., 'joint_6': 6}
        n_regimes: number of motion regimes (2 = low/high, 3 = low/mid/high).
        quantile: quantile for scale (default 0.99 → 99th percentile).
        min_samples_per_regime: minimum pool size per regime before fallback.
        method: error collection method — ``"inter_demo"`` or ``"intra_demo"``.
        local_progress_window: max normalized-progress gap for inter-demo
            matching (only used when ``method="inter_demo"``).
        ta: action-chunk stride for intra-demo differences
            (only used when ``method="intra_demo"``).

    Returns:
        scales: {group_name: {regime_label: float}}
            e.g., {'joint_0': {'low': 0.05, 'high': 0.18}, ...}
        boundaries: {group_name: list[float]}
            e.g., {'joint_0': [0.35], 'joint_1': [0.22], ...}

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> demos = [rng.standard_normal((100, 8)).astype(np.float32) for _ in range(6)]
        >>> group_cfg = {f'joint_{i}': i for i in range(7)}
        >>> scales, boundaries = compute_regime_scales(demos, group_cfg)
        >>> assert 'low' in scales['joint_0'] and 'high' in scales['joint_0']
        >>> assert len(boundaries['joint_0']) == 1
        >>> scales2, _ = compute_regime_scales(demos, group_cfg, method='intra_demo', ta=5)
        >>> assert 'low' in scales2['joint_0']
    """
    if len(expert_demos) < 2:
        raise ValueError(f"Need at least 2 expert demos, got {len(expert_demos)}")
    if n_regimes not in _REGIME_LABELS:
        raise ValueError(f"n_regimes must be one of {list(_REGIME_LABELS)}, got {n_regimes}")
    if method not in ("inter_demo", "intra_demo"):
        raise ValueError(f"method must be 'inter_demo' or 'intra_demo', got {method!r}")

    # n_regimes=1: single regime covering all magnitudes → global quantile scale,
    # identical in behaviour to the standard (non-regime) inter/intra_demo method.

    labels = _REGIME_LABELS[n_regimes]

    # -------------------------------------------------------------------------
    # Step 1: per-group magnitude distribution and regime boundaries
    # -------------------------------------------------------------------------
    all_magnitudes = compute_group_magnitudes(expert_demos, group_config)

    boundaries: dict[str, list[float]] = {
        g: compute_regime_boundaries(all_magnitudes[g], n_regimes)
        for g in group_config
    }

    # -------------------------------------------------------------------------
    # Step 2 & 3: collect per-regime errors
    # -------------------------------------------------------------------------
    if method == "intra_demo":
        error_pools = _collect_errors_intra_demo(
            expert_demos=expert_demos,
            group_config=group_config,
            boundaries=boundaries,
            labels=labels,
            ta=int(ta),
        )
    else:  # inter_demo
        error_pools = _collect_errors_inter_demo(
            expert_demos=expert_demos,
            group_config=group_config,
            boundaries=boundaries,
            labels=labels,
            local_progress_window=float(local_progress_window),
        )

    # -------------------------------------------------------------------------
    # Step 4: quantile-based scales with global fallback
    # -------------------------------------------------------------------------
    # Global fallback = quantile over all errors regardless of regime
    global_scales: dict[str, float] = {}
    for g in group_config:
        all_errs = [e for lbl in labels for e in error_pools[g][lbl]]
        global_scales[g] = float(np.quantile(all_errs, quantile)) if all_errs else 1.0

    scales: dict[str, dict[str, float]] = {}
    for g in group_config:
        scales[g] = {}
        for lbl in labels:
            pool = error_pools[g][lbl]
            if len(pool) < min_samples_per_regime:
                scales[g][lbl] = global_scales[g]
            else:
                scales[g][lbl] = float(np.quantile(pool, quantile))

    return scales, boundaries
