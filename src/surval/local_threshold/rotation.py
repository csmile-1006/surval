"""Small numpy helpers for 6D continuous rotation distances.

Mirrors the convention used by ``surval.sequential_validate`` (the threshold
consumer) so that local-threshold computation produces values in the same
units as the per-block error the consumer measures at eval time. Keeping a
private copy here avoids importing the heavy ``sequential_validate`` module
from the threshold pipeline.

6D layout: ``r6d[..., :3]`` = first row of R, ``r6d[..., 3:6]`` = second row.
"""

from __future__ import annotations

import numpy as np


def rot6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 6D -> rotation matrix. (..., 6) -> (..., 3, 3)."""
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def rotmat_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle (||result|| = angle in radians)."""
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    axis_raw = np.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        axis=-1,
    )
    sin_a = np.sin(angle)
    safe_denom = 2.0 * np.where(sin_a > 1e-8, sin_a, np.ones_like(sin_a))
    axis = axis_raw / safe_denom[..., None]
    axis = np.where((angle < 1e-8)[..., None], np.zeros_like(axis), axis)
    return (angle[..., None] * axis).astype(np.float32)


def rot6d_relative_delta(r6d_a: np.ndarray, r6d_b: np.ndarray) -> np.ndarray:
    """Axis-angle of ``R_a^T @ R_b`` (from a to b). (..., 6), (..., 6) -> (..., 3).

    ``||result||`` is the geodesic (angular) distance between the two rotations.
    """
    r_a = rot6d_to_rotmat(r6d_a)
    r_b = rot6d_to_rotmat(r6d_b)
    r_rel = np.einsum("...ji,...jk->...ik", r_a, r_b)
    return rotmat_to_axis_angle(r_rel)
