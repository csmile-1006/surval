"""State vector composition (image + proprio) for local-threshold retrieval.

The double L2-normalization (per-modality block + final state vector) is the
core of the modality-balancing trick. Without it, the high-dimensional image
block dominates cosine distance. See spec §3.2 "Critical detail".
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle

import numpy as np

from .config import LocalThresholdConfig

_EPS = 1e-12


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, _EPS)


@dataclass
class ProprioNormalizer:
    """Per-dimension z-score. Fit on the val set, persisted to disk."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, proprio_array: np.ndarray) -> ProprioNormalizer:
        if proprio_array.ndim != 2:
            raise ValueError(f"proprio_array must be (N, d), got shape {proprio_array.shape}")
        self.mean = proprio_array.mean(axis=0).astype(np.float32)
        std = proprio_array.std(axis=0).astype(np.float32)
        # Guard against zero variance dims.
        self.std = np.where(std < _EPS, 1.0, std).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ProprioNormalizer.fit must be called before transform")
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path: str) -> ProprioNormalizer:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(mean=d["mean"], std=d["std"])


def flatten_proprio(
    proprio_dict: dict[str, np.ndarray],
    cfg: LocalThresholdConfig,
) -> np.ndarray:
    """Flatten the configured proprio keys into a single 1-D vector.

    Per-frame only; multi-frame state windowing is applied later via
    ``build_window_indices`` + ``compose_state_vectors_batch``.
    """
    parts = []
    for key in cfg.proprio_keys:
        v = np.asarray(proprio_dict[key], dtype=np.float32).reshape(-1)
        parts.append(v)
    return np.concatenate(parts, axis=0)


def build_window_indices(
    demo_id_int: np.ndarray,
    t: np.ndarray,
    K: int,  # noqa: N803 — K is conventional for window size
) -> np.ndarray:
    """For each state index i, return the K source indices to gather as the
    causal window (t-K+1 ... t), clamped at the start of the same demo.

    Returns: ``(N, K)`` int64. ``out[i, -1] == i``; ``out[i, k]`` walks back
    in time within the same demo, clamping at the earliest available frame.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    n = demo_id_int.shape[0]
    out = np.zeros((n, K), dtype=np.int64)
    offsets = np.arange(K - 1, -1, -1)  # [K-1, K-2, ..., 0]
    for demo in np.unique(demo_id_int):
        mask = demo_id_int == demo
        idx_demo = np.where(mask)[0]
        order = np.argsort(t[idx_demo])
        idx_sorted = idx_demo[order]  # state indices in temporal order
        n_demo = idx_sorted.shape[0]
        positions = np.arange(n_demo)
        back = np.maximum(positions[:, None] - offsets[None, :], 0)  # (n_demo, K)
        out[idx_sorted] = idx_sorted[back]
    return out


def _gather_aggregate(
    feats: np.ndarray,  # (N, D)
    window_indices: np.ndarray | None,  # (N, K) or None
    aggregation: str,
) -> np.ndarray:
    """Gather ``feats[window_indices]`` then aggregate to ``(N, D')``.

    - aggregation="concat" → ``D' = K * D`` (stacks frames along feature dim)
    - aggregation="mean"   → ``D' = D``     (mean-pools across the window)

    ``window_indices=None`` is a no-op (returns ``feats`` unchanged).
    """
    if window_indices is None:
        return feats
    gathered = feats[window_indices]  # (N, K, D)
    if aggregation == "concat":
        return gathered.reshape(gathered.shape[0], -1).astype(feats.dtype, copy=False)
    if aggregation == "mean":
        return gathered.mean(axis=1).astype(feats.dtype, copy=False)
    raise ValueError(f"unknown state_window_aggregation: {aggregation!r}")


def compose_state_vectors_batch(
    image_features_per_view: dict[str, np.ndarray],  # {view: (N, D_img)}
    proprio_array: np.ndarray,  # (N, d_proprio), already z-scored
    cfg: LocalThresholdConfig,
    window_indices: np.ndarray | None = None,  # (N, K) for state-window gather; None => single-frame
) -> np.ndarray:
    """Vectorized state composition for a batch of N states.

    If ``window_indices`` is provided (shape ``(N, K)`` with K = cfg.state_window_size),
    each modality's per-frame features are first gathered into windows and
    aggregated via ``cfg.state_window_aggregation`` before per-modality L2-norm.

    Returns: ``(N, D_state)``, float32, L2-normalized.
    """
    agg = cfg.state_window_aggregation
    blocks = []

    if cfg.use_image:
        view_arrays = []
        for view in cfg.image_views:
            feat = np.asarray(image_features_per_view[view], dtype=np.float32)
            feat = _gather_aggregate(feat, window_indices, agg)  # (N, D') after agg
            if cfg.l2_normalize_per_modality:
                feat = _l2_normalize(feat, axis=-1)
            view_arrays.append(feat)
        image_block = np.concatenate(view_arrays, axis=-1)
        if cfg.l2_normalize_per_modality:
            image_block = _l2_normalize(image_block, axis=-1)
        blocks.append(image_block)

    if cfg.use_proprio:
        proprio_block = proprio_array.astype(np.float32, copy=False)
        proprio_block = _gather_aggregate(proprio_block, window_indices, agg)
        if cfg.l2_normalize_per_modality:
            proprio_block = _l2_normalize(proprio_block, axis=-1)
        blocks.append(proprio_block)

    if not blocks:
        raise ValueError("At least one of use_image/use_proprio must be True")

    state = np.concatenate(blocks, axis=-1)
    state = _l2_normalize(state, axis=-1)
    return state.astype(np.float32, copy=False)


def modality_block_norms(
    image_features_per_view: dict[str, np.ndarray],
    proprio_array: np.ndarray,
    cfg: LocalThresholdConfig,
    window_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """For sanity check: per-row L2 norms of the per-modality blocks BEFORE the
    final renormalization. Both should be ~1.0 if l2_normalize_per_modality is
    on. Mirrors the windowing in ``compose_state_vectors_batch``.
    """
    agg = cfg.state_window_aggregation
    out: dict[str, np.ndarray] = {}
    if cfg.use_image:
        view_arrays = []
        for view in cfg.image_views:
            feat = np.asarray(image_features_per_view[view], dtype=np.float32)
            feat = _gather_aggregate(feat, window_indices, agg)
            if cfg.l2_normalize_per_modality:
                feat = _l2_normalize(feat, axis=-1)
            view_arrays.append(feat)
        image_block = np.concatenate(view_arrays, axis=-1)
        if cfg.l2_normalize_per_modality:
            image_block = _l2_normalize(image_block, axis=-1)
        out["image_block"] = np.linalg.norm(image_block, axis=-1)
    if cfg.use_proprio:
        proprio_block = proprio_array.astype(np.float32, copy=False)
        proprio_block = _gather_aggregate(proprio_block, window_indices, agg)
        if cfg.l2_normalize_per_modality:
            proprio_block = _l2_normalize(proprio_block, axis=-1)
        out["proprio_block"] = np.linalg.norm(proprio_block, axis=-1)
    return out
