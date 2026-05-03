"""Configuration for state-conditional (local) threshold computation.

Single source of truth for hyperparameters. All fields participate in the
deterministic ``to_hash()`` cache key, so changing any field produces a fresh
artifact directory.

Notes specific to this codebase:
- Image keys come from ``DroidRldsSequentialDataset``: ``image`` (exterior) and
  ``wrist_image`` (wrist). There are exactly 2 views.
- Proprio keys are ``joint_position`` (7), ``cartesian_position`` (6),
  ``gripper_position`` (1).
- Action is DROID 8D ``[joint_velocity (7), gripper (1)]``. Blocks are the 7
  joint scalars; gripper is excluded from threshold blocks (matches
  ``DROID_ACTION_SPACE`` in scripts/sequential_validate_from_cache_droid.py).
- Distance is L2 only (rotation/Mahalanobis intentionally omitted).
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class LocalThresholdConfig:
    # ----- Encoder -----
    encoder_name: str = "dinov2_vitb14"  # dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14
    encoder_pool: str = "cls"  # "cls" | "mean_patch"
    image_size: int = 224
    image_views: tuple[str, ...] = ("image", "wrist_image")

    # ----- State vector composition -----
    use_image: bool = True
    use_proprio: bool = True
    proprio_keys: tuple[str, ...] = ("joint_position", "cartesian_position", "gripper_position")
    # State window: gather past K frames (causal, clamped at demo start) for each
    # state. K=1 = current frame only (default, no behavior change).
    state_window_size: int = 1
    state_window_aggregation: str = "concat"  # "concat" (preserve temporal info, dim *= K) | "mean"
    l2_normalize_per_modality: bool = True

    # ----- Retrieval -----
    index_type: str = "flat_ip"  # "flat_ip" (cosine) | "hnsw"
    k_neighbors: int = 50
    same_demo_allowed: bool = True
    temporal_exclusion_radius: int = 5  # exclude same-demo neighbors with |t - t_q| <= this

    # ----- Threshold -----
    # Multiple quantile candidates are computed and stored simultaneously; the
    # consumer chooses one at use time.
    quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)
    min_neighbors_for_local: int = 10  # below this, fall back to the matching global quantile
    pairwise_or_query_centered: str = "pairwise"  # "pairwise" | "query_centered"
    # Source of the per-block scale distribution.
    #   "local"          — neighbor-action pairwise/query-centered L2 (default).
    #   "intra_demo_sc"  — within-demo chunk motion ||a[t+ta] - a[t]|| pooled over
    #                      state-NN neighbors. Same units as the consumer's
    #                      non-SC ``intra_demo`` mode, but state-conditional.
    scale_source: str = "local"
    # Chunk horizon used by ``intra_demo_sc``. Should match the seqval --ta.
    ta: int = 8

    # ----- Action grouping (DROID 8D) -----
    action_dim: int = 8
    block_names: tuple[str, ...] = (
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    )
    # Per-block slice as (start, stop). Stored as tuples (frozen-dataclass-friendly).
    block_slices: tuple[tuple[str, int, int], ...] = (
        ("joint_0", 0, 1),
        ("joint_1", 1, 2),
        ("joint_2", 2, 3),
        ("joint_3", 3, 4),
        ("joint_4", 4, 5),
        ("joint_5", 5, 6),
        ("joint_6", 6, 7),
    )

    # ----- Data loader -----
    data_dir: str = ""  # tfds data root
    dataset_name: str = "droid"
    droid_action_space: str = "joint_velocity"  # "joint_velocity" | "joint_position"
    action_chunk_size: int = 16  # only the t=0 step is consumed; matches existing pipeline
    max_samples: int | None = None  # cap for debugging
    # When True, the state DB pulls actions directly from a sequential-validation
    # eval cache (HDF5) instead of from raw RLDS, and only retains states whose
    # (demo_id, t) is present in that cache. This guarantees 1:1 alignment with
    # the consumer's --block-scale-method=intra_demo / local pipeline (same
    # action values modulo the Normalize→Unnormalize round-trip, same demo
    # subset modulo drop_last). The cache path itself is a build-time argument
    # (not in the hash) — toggling this bool forces a fresh state DB.
    actions_from_cache: bool = False

    # ----- Caching -----
    cache_root: str = "./cache/local_threshold"

    # ----- Encoder runtime -----
    encoder_batch_size: int = 128
    encoder_device: str = "cuda"

    def block_slice_dict(self) -> dict[str, slice]:
        """Convert the tuple form to ``{block_name: slice}``."""
        return {name: slice(start, stop) for name, start, stop in self.block_slices}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_hash(self) -> str:
        """Deterministic 16-char SHA-256 over all fields."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
