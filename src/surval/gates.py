"""
Gate functions for SURVAL evaluation.

Gates convert raw prediction errors into pass/fail or soft probabilities.
"""

from __future__ import annotations

import numpy as np


def soft_gate(
    e: np.ndarray,
    S: float,
    eps: float = 1.0,
    tau: float = 0.1,
) -> np.ndarray:
    """
    Soft gate: exp(-max(e/S - eps, 0) / tau).

    Smoothly transitions from 1.0 (well within tolerance) to ~0.0 (far outside).

    Args:
        e: error values (any shape, non-negative).
        S: scale (tolerance level).
        eps: threshold in units of S; errors below eps*S yield gate ≈ 1.
        tau: temperature controlling steepness of the transition.

    Returns:
        Gate values in (0, 1], same shape as e.

    Example:
        >>> import numpy as np
        >>> g = soft_gate(np.array([0.0, 1.0, 2.0]), S=1.0, eps=1.0, tau=0.1)
        >>> assert g[0] == 1.0 and g[1] == 1.0 and g[2] < 0.01
    """
    scaled = np.asarray(e, dtype=np.float64) / max(float(S), 1e-8)
    return np.exp(-np.maximum(scaled - float(eps), 0.0) / max(float(tau), 1e-8)).astype(np.float32)


def hard_gate(e: np.ndarray, S: float) -> np.ndarray:
    """
    Hard gate: 1[e < S].

    Returns 1.0 if error is strictly within scale, 0.0 otherwise.

    Args:
        e: error values (any shape, non-negative).
        S: scale threshold.

    Returns:
        Binary gate values (0.0 or 1.0), same shape as e.

    Example:
        >>> import numpy as np
        >>> g = hard_gate(np.array([0.1, 0.5, 1.5]), S=1.0)
        >>> assert list(g) == [1.0, 1.0, 0.0]
    """
    return (np.asarray(e, dtype=np.float64) < float(S)).astype(np.float32)


def gripper_gate(
    pred_grip: float,
    expert_grip: float,
    threshold: float = 0.5,
) -> float:
    """
    Binary gripper gate: 1.0 if both grippers are in the same open/close state, else 0.0.

    Gripper state is determined by thresholding the continuous gripper value.

    Args:
        pred_grip: predicted gripper action value.
        expert_grip: expert gripper action value.
        threshold: open/close decision boundary (default 0.5).

    Returns:
        1.0 if same state (both open or both closed), 0.0 if different.

    Example:
        >>> gripper_gate(0.8, 0.9)   # both open
        1.0
        >>> gripper_gate(0.2, 0.8)   # pred closed, expert open
        0.0
    """
    return 1.0 if (float(pred_grip) > threshold) == (float(expert_grip) > threshold) else 0.0
