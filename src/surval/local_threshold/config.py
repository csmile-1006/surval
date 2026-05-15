"""Configuration for state-conditional (local) threshold computation.

Single source of truth for hyperparameters. All fields participate in the
deterministic ``to_hash()`` cache key, so changing any field produces a fresh
artifact directory. (One backward-compat shim: ``block_types`` is omitted
from the hash when empty so existing joint-velocity artifacts keep their
hash after the field was added.)

Notes specific to this codebase:
- Image keys come from ``DroidRldsSequentialDataset``: ``image`` (exterior) and
  ``wrist_image`` (wrist). There are exactly 2 views.
- Proprio keys are ``joint_position`` (7), ``cartesian_position`` (6),
  ``gripper_position`` (1).

Action layouts (selected via ``droid_action_space``):
- ``joint_velocity`` / ``joint_position`` — DROID 8D ``[joint (7), gripper (1)]``.
  Blocks are the 7 joint scalars; gripper is excluded. Per-block distance is
  raw L2. (Default — matches ``DROID_ACTION_SPACE`` in pre-2026 surval_openpi.)
- ``pos_rot6d`` — DROID 10D ``[pos (3), rot_6d (6), gripper (1)]``. Blocks are
  ``pos`` (L2 over 3D translation) and ``rot_6d`` (SO(3) geodesic distance over
  the 6D continuous rotation). Matches ``DROID_ACTION_SPACE`` in
  ``droid_policy_learning``'s ``sequential_validate_from_cache_droid.py``.

Adding a new action space:
1. Add a branch in ``_action_space_block_spec`` below.
2. Optionally extend the consumer ``ACTION_SPACE`` in the seqval wrapper to
   match (same ``block_names``, ``block_types``).
3. Run a hash diff before/after to confirm the new preset stamps a unique
   ``to_hash()``.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json


def _action_space_block_spec(
    droid_action_space: str,
) -> tuple[int, tuple[str, ...], tuple[tuple[str, int, int], ...], tuple[tuple[str, str], ...]]:
    """Return (action_dim, block_names, block_slices, block_types) for a preset.

    block_types entries are ``(block_name, type)`` pairs; absent blocks default
    to raw L2 in the consumer / threshold computation.
    """
    s = droid_action_space.lower()
    if s in ("joint_velocity", "joint_position"):
        block_names = tuple(f"joint_{i}" for i in range(7))
        block_slices = tuple((f"joint_{i}", i, i + 1) for i in range(7))
        return 8, block_names, block_slices, ()
    if s in ("pos_rot6d", "cartesian_position_and_rot6d"):
        # 10-D layout: [pos (0:3), rot_6d (3:9), gripper (9)]. Gripper is
        # excluded from threshold blocks to match the seqval consumer's
        # DROID_ACTION_SPACE in droid_policy_learning.
        block_names = ("pos", "rot_6d")
        block_slices = (("pos", 0, 3), ("rot_6d", 3, 9))
        block_types = (("rot_6d", "rot6d"),)
        return 10, block_names, block_slices, block_types
    raise ValueError(
        f"Unknown droid_action_space {droid_action_space!r}. "
        "Supported: joint_velocity, joint_position, pos_rot6d."
    )


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
    #   "local"          — neighbor-action pairwise/query-centered distances (default).
    #   "intra_demo_sc"  — within-demo chunk motion ||a[t+ta] - a[t]|| pooled over
    #                      state-NN neighbors. Same units as the consumer's
    #                      non-SC ``intra_demo`` mode, but state-conditional.
    scale_source: str = "local"
    # Chunk horizon used by ``intra_demo_sc``. Should match the seqval --ta.
    ta: int = 8

    # ----- Action grouping (default: DROID 8D joint_velocity) -----
    # ``build_state_db.py`` overrides these from ``--droid-action-space`` via
    # ``_action_space_block_spec`` so the preset and the explicit fields stay
    # in sync. For non-DROID consumers you can construct the config with
    # explicit block_names/block_slices/block_types and any action_dim.
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
    # Per-block distance type. Only blocks listed here use a non-default
    # distance; absent blocks default to raw L2 over the slice. Currently
    # supported types: "rot6d" (SO(3) geodesic on a 6D continuous rotation).
    # Stored as tuple-of-tuples to remain frozen-dataclass-friendly.
    block_types: tuple[tuple[str, str], ...] = ()

    # ----- Data loader -----
    data_dir: str = ""  # tfds data root
    dataset_name: str = "droid"
    droid_action_space: str = "joint_velocity"  # joint_velocity | joint_position | pos_rot6d
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

    def block_type_dict(self) -> dict[str, str]:
        """Convert the tuple form to ``{block_name: type}``."""
        return {name: typ for name, typ in self.block_types}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_hash(self) -> str:
        """Deterministic 16-char SHA-256 over all fields.

        ``block_types`` is dropped from the payload when empty so existing
        joint-velocity / joint-position state DBs (built before the field was
        added) keep their pre-existing hash.
        """
        payload_dict = self.to_dict()
        if not self.block_types:
            payload_dict.pop("block_types", None)
        payload = json.dumps(payload_dict, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @classmethod
    def for_droid_action_space(
        cls,
        droid_action_space: str,
        /,
        **overrides,
    ) -> LocalThresholdConfig:
        """Build a config with action-space-appropriate block layout.

        Mirrors what ``build_state_db.py`` does, exposed as a class method so
        downstream code (tests, ad-hoc scripts) can construct a properly-shaped
        config without duplicating the preset table.
        """
        action_dim, block_names, block_slices, block_types = _action_space_block_spec(
            droid_action_space
        )
        kwargs = dict(
            droid_action_space=droid_action_space,
            action_dim=action_dim,
            block_names=block_names,
            block_slices=block_slices,
            block_types=block_types,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# Re-export the preset table for callers that need the spec without constructing
# a full config (e.g. ``build_state_db.py``).
__all__ = ["LocalThresholdConfig", "_action_space_block_spec"]
