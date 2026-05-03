"""Local (state-conditional) threshold computation.

For every state ``s_q`` in the database, retrieve k nearest neighbors in
state space, drop self / temporally-close same-demo neighbors, then compute
per-block, per-quantile thresholds ``S_g^q(s_q)`` from action distances among
those neighbors.

Per user direction (2026-04-25):
- Distance is L2 only (no rotation, no Mahalanobis).
- Multiple quantile candidates (e.g. 0.5, 0.75, 0.9, 0.95, 0.99) are computed
  in a single pass and stored together; the consumer chooses one at use time.
- Global per-block-per-quantile thresholds are computed in the same pipeline
  (using the *same* L2 + state-NN formulation, but pooled across all neighbor
  pairs in the DB) so global and local share units and are saved together.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np

from .config import LocalThresholdConfig
from .database import StateDatabase

# -------- core distance computation --------


def _block_pair_distances(
    actions: np.ndarray,  # (M, action_dim) float32
    block_slice: slice,
    *,
    pairwise: bool,
    query_action: np.ndarray | None = None,  # (action_dim,), required if not pairwise
) -> np.ndarray:
    """Per-block L2 distances among M neighbor actions.

    pairwise=True  → upper-triangular pairwise distances, length M*(M-1)/2.
    pairwise=False → query-centered distances ||a_i - a_q||, length M.
    """
    block = actions[:, block_slice]  # (M, d_block)
    if pairwise:
        if block.shape[0] < 2:
            return np.empty((0,), dtype=np.float32)
        diff = block[:, None, :] - block[None, :, :]  # (M, M, d)
        dist = np.linalg.norm(diff, axis=-1)  # (M, M)
        iu = np.triu_indices(block.shape[0], k=1)
        return dist[iu].astype(np.float32)
    if query_action is None:
        raise ValueError("query_action required for query_centered mode")
    q_block = query_action[block_slice][None, :]  # (1, d)
    return np.linalg.norm(block - q_block, axis=-1).astype(np.float32)


# -------- LocalThresholdMap --------


@dataclass
class LocalThresholdMap:
    """Per-state, per-block, per-quantile thresholds.

    Storage shape: ``thresholds[N, n_blocks, n_quantiles]``.
    """

    thresholds: np.ndarray  # (N, n_blocks, n_quantiles) float32
    fallback_used: np.ndarray  # (N,) bool — True if local fell back to global
    n_neighbors_used: np.ndarray  # (N,) int32
    block_names: tuple[str, ...]
    quantiles: tuple[float, ...]
    global_thresholds: np.ndarray  # (n_blocks, n_quantiles) float32
    config_hash: str

    def get(self, state_idx: int, block: str, quantile: float) -> float:
        b = self.block_names.index(block)
        q = self.quantile_index(quantile)
        return float(self.thresholds[state_idx, b, q])

    def get_global(self, block: str, quantile: float) -> float:
        b = self.block_names.index(block)
        q = self.quantile_index(quantile)
        return float(self.global_thresholds[b, q])

    def quantile_index(self, quantile: float) -> int:
        for i, q in enumerate(self.quantiles):
            if abs(q - quantile) < 1e-9:
                return i
        raise ValueError(f"quantile={quantile} not in stored quantiles {self.quantiles}")

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez(
            os.path.join(path, "local_threshold_map.npz"),
            thresholds=self.thresholds,
            fallback_used=self.fallback_used,
            n_neighbors_used=self.n_neighbors_used,
            global_thresholds=self.global_thresholds,
        )
        with open(os.path.join(path, "local_threshold_meta.json"), "w") as f:
            json.dump(
                {
                    "block_names": list(self.block_names),
                    "quantiles": list(self.quantiles),
                    "config_hash": self.config_hash,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> LocalThresholdMap:
        npz = np.load(os.path.join(path, "local_threshold_map.npz"))
        with open(os.path.join(path, "local_threshold_meta.json")) as f:
            meta = json.load(f)
        return cls(
            thresholds=npz["thresholds"],
            fallback_used=npz["fallback_used"],
            n_neighbors_used=npz["n_neighbors_used"],
            block_names=tuple(meta["block_names"]),
            quantiles=tuple(meta["quantiles"]),
            global_thresholds=npz["global_thresholds"],
            config_hash=meta["config_hash"],
        )


# -------- main entry points --------


def compute_global_thresholds_from_db(
    db: StateDatabase,
    cfg: LocalThresholdConfig,
) -> np.ndarray:
    """Pool ALL filtered neighbor pairs across the DB into a single per-block
    distribution, then take each quantile.

    Returns: (n_blocks, n_quantiles) float32.

    Note: this differs from the legacy progress-window method in
    ``base.py:_estimate_block_scales_from_expert_demos``. Per user direction
    (compute global within the same pipeline), we use the same L2 formulation
    as the local pass, just pooled. This guarantees units match.
    """
    if db.records is None or db.embeddings is None:
        raise RuntimeError("DB not built")

    n = db.n_states
    block_slices = cfg.block_slice_dict()
    block_names = list(cfg.block_names)
    quantiles = np.asarray(cfg.quantiles, dtype=np.float64)

    buffer = max(2 * cfg.temporal_exclusion_radius + 2, 1)
    k = cfg.k_neighbors + buffer

    # Single FAISS query for all states.
    _, indices = db.query(db.embeddings, k=k)
    filtered = db.filter_neighbors(
        indices,
        db.records.demo_id_int,
        db.records.t,
        cfg=cfg,
    )

    actions = db.records.actions
    pools: dict[str, list[np.ndarray]] = {b: [] for b in block_names}

    for q in range(n):
        nb_idx = filtered[q]
        if nb_idx.size < 2:
            continue
        nb_actions = actions[nb_idx]
        for b in block_names:
            d = _block_pair_distances(
                nb_actions,
                block_slices[b],
                pairwise=cfg.pairwise_or_query_centered == "pairwise",
                query_action=actions[q] if cfg.pairwise_or_query_centered != "pairwise" else None,
            )
            if d.size > 0:
                pools[b].append(d)

    out = np.zeros((len(block_names), len(quantiles)), dtype=np.float32)
    for bi, b in enumerate(block_names):
        if not pools[b]:
            continue
        all_d = np.concatenate(pools[b], axis=0)
        out[bi, :] = np.quantile(all_d, quantiles).astype(np.float32)
    return out


def compute_local_thresholds(
    db: StateDatabase,
    cfg: LocalThresholdConfig,
    global_thresholds: np.ndarray,  # (n_blocks, n_quantiles)
) -> LocalThresholdMap:
    """For each state in db, compute per-block per-quantile thresholds.

    States with too few valid neighbors fall back to the matching global
    quantile.
    """
    if db.records is None or db.embeddings is None:
        raise RuntimeError("DB not built")

    n = db.n_states
    block_slices = cfg.block_slice_dict()
    block_names = list(cfg.block_names)
    quantiles = np.asarray(cfg.quantiles, dtype=np.float64)
    n_blocks = len(block_names)
    n_q = len(quantiles)

    if global_thresholds.shape != (n_blocks, n_q):
        raise ValueError(f"global_thresholds shape {global_thresholds.shape} != ({n_blocks}, {n_q})")

    buffer = max(2 * cfg.temporal_exclusion_radius + 2, 1)
    k = cfg.k_neighbors + buffer

    _, indices = db.query(db.embeddings, k=k)
    filtered = db.filter_neighbors(
        indices,
        db.records.demo_id_int,
        db.records.t,
        cfg=cfg,
    )

    actions = db.records.actions
    thresholds = np.zeros((n, n_blocks, n_q), dtype=np.float32)
    fallback = np.zeros((n,), dtype=bool)
    n_neighbors_used = np.zeros((n,), dtype=np.int32)

    pairwise = cfg.pairwise_or_query_centered == "pairwise"
    min_nb = cfg.min_neighbors_for_local

    for qi in range(n):
        nb_idx = filtered[qi]
        n_neighbors_used[qi] = nb_idx.size
        if nb_idx.size < min_nb:
            fallback[qi] = True
            thresholds[qi] = global_thresholds
            continue

        nb_actions = actions[nb_idx]
        q_action = actions[qi] if not pairwise else None
        for bi, b in enumerate(block_names):
            d = _block_pair_distances(
                nb_actions,
                block_slices[b],
                pairwise=pairwise,
                query_action=q_action,
            )
            if d.size == 0:
                # No valid pairs for this block at this state — fall back per-block.
                thresholds[qi, bi, :] = global_thresholds[bi, :]
            else:
                thresholds[qi, bi, :] = np.quantile(d, quantiles).astype(np.float32)

    return LocalThresholdMap(
        thresholds=thresholds,
        fallback_used=fallback,
        n_neighbors_used=n_neighbors_used,
        block_names=tuple(block_names),
        quantiles=tuple(float(q) for q in cfg.quantiles),
        global_thresholds=global_thresholds.astype(np.float32),
        config_hash=cfg.to_hash(),
    )


# -------- intra-demo (state-conditional) chunk-motion thresholds --------


def _gather_chunk_motion_deltas(
    nb_idx: np.ndarray,
    *,
    actions: np.ndarray,
    demo_id_int: np.ndarray,
    t_arr: np.ndarray,
    idx_lookup: dict[tuple[int, int], int],
    ta: int,
) -> np.ndarray:
    """For each neighbor j, compute ``a[idx_at(demo_j, t_j+ta)] - a[j]``.

    Returns an ``(M', A)`` array of valid chunk-motion deltas, where ``M' <= M``
    drops neighbors whose ``(demo_j, t_j+ta)`` is not present in the DB
    (demo end, or sampling gap).
    """
    if nb_idx.size == 0:
        return np.empty((0, actions.shape[1]), dtype=np.float32)
    deltas: list[np.ndarray] = []
    for j in nb_idx:
        d_j = int(demo_id_int[j])
        t_j_next = int(t_arr[j]) + int(ta)
        nxt = idx_lookup.get((d_j, t_j_next))
        if nxt is None:
            continue
        deltas.append(actions[nxt] - actions[j])
    if not deltas:
        return np.empty((0, actions.shape[1]), dtype=np.float32)
    return np.stack(deltas, axis=0).astype(np.float32)


def _intra_demo_sc_neighbors(
    db: StateDatabase,
    cfg: LocalThresholdConfig,
) -> list[np.ndarray]:
    """Run state-NN + the intra-demo-SC neighbor filter for every state in db.

    Returns ``filtered[q]`` = 1-D int64 array of retained neighbor db indices
    for query ``q`` (self + same-demo near-time excluded; same-demo far kept).
    """
    if db.records is None or db.embeddings is None:
        raise RuntimeError("DB not built")
    buffer = max(2 * cfg.temporal_exclusion_radius + 2, 1)
    k = cfg.k_neighbors + buffer
    _, indices = db.query(db.embeddings, k=k)
    return db.filter_neighbors_keep_same_demo_far(
        indices,
        db.records.demo_id_int,
        db.records.t,
        cfg=cfg,
    )


def compute_global_intra_demo_thresholds_from_db(
    db: StateDatabase,
    cfg: LocalThresholdConfig,
) -> np.ndarray:
    """Pool ALL state-NN-restricted, within-demo chunk-motion magnitudes across
    the DB into a single per-block distribution and take each quantile.

    Same units as ``_estimate_block_scales_from_intra_demo_diffs`` (the seqval
    ``intra_demo`` mode), but the pool is only the union of state-NN neighbors'
    deltas — i.e. state-conditional only at the population level. Local pass
    additionally restricts the pool to each query's own neighbors.

    Returns: ``(n_blocks, n_quantiles)`` float32.
    """
    if db.records is None:
        raise RuntimeError("DB not built")

    block_slices = cfg.block_slice_dict()
    block_names = list(cfg.block_names)
    quantiles = np.asarray(cfg.quantiles, dtype=np.float64)

    filtered = _intra_demo_sc_neighbors(db, cfg)
    actions = db.records.actions
    demo_id_int = db.records.demo_id_int
    t_arr = db.records.t
    idx_lookup = db.idx_lookup()
    ta = int(cfg.ta)

    pools: dict[str, list[np.ndarray]] = {b: [] for b in block_names}
    for q in range(db.n_states):
        deltas = _gather_chunk_motion_deltas(
            filtered[q],
            actions=actions,
            demo_id_int=demo_id_int,
            t_arr=t_arr,
            idx_lookup=idx_lookup,
            ta=ta,
        )
        if deltas.shape[0] == 0:
            continue
        for b in block_names:
            sl = block_slices[b]
            d = np.linalg.norm(deltas[:, sl], axis=-1).astype(np.float32)
            pools[b].append(d)

    out = np.zeros((len(block_names), len(quantiles)), dtype=np.float32)
    for bi, b in enumerate(block_names):
        if not pools[b]:
            continue
        all_d = np.concatenate(pools[b], axis=0)
        out[bi, :] = np.quantile(all_d, quantiles).astype(np.float32)
    return out


def compute_legacy_intra_demo_thresholds_from_db(
    db: StateDatabase,
    cfg: LocalThresholdConfig,
) -> np.ndarray:
    """Reference implementation of the seqval-side ``intra_demo`` formula on
    the state DB itself.

    For each demo group all states by ``t``, then pool every
    ``a[t+ta] - a[t]`` chunk-motion delta (no state-NN, no temporal exclusion,
    each pair counted exactly once). Per-block L2 norm; quantile per
    ``cfg.quantiles``.

    Returned shape: ``(n_blocks, n_quantiles)``. Use this as a sanity-check
    target for ``compute_global_intra_demo_thresholds_from_db``: with large k
    and small radius, the SC global should converge close to this.
    """
    if db.records is None:
        raise RuntimeError("DB not built")

    block_slices = cfg.block_slice_dict()
    block_names = list(cfg.block_names)
    quantiles = np.asarray(cfg.quantiles, dtype=np.float64)
    actions = db.records.actions
    demo_id_int = db.records.demo_id_int
    t_arr = db.records.t
    ta = int(cfg.ta)

    by_demo: dict[int, dict[int, int]] = {}
    for i in range(db.n_states):
        by_demo.setdefault(int(demo_id_int[i]), {})[int(t_arr[i])] = i

    pools: dict[str, list[np.ndarray]] = {b: [] for b in block_names}
    for t_to_idx in by_demo.values():
        ts_sorted = sorted(t_to_idx.keys())
        if len(ts_sorted) <= ta:
            continue
        deltas = []
        for t in ts_sorted:
            j_next = t_to_idx.get(t + ta)
            if j_next is None:
                continue
            j_cur = t_to_idx[t]
            deltas.append(actions[j_next] - actions[j_cur])
        if not deltas:
            continue
        d_arr = np.stack(deltas, axis=0).astype(np.float32)
        for b in block_names:
            sl = block_slices[b]
            pools[b].append(np.linalg.norm(d_arr[:, sl], axis=-1).astype(np.float32))

    out = np.zeros((len(block_names), len(quantiles)), dtype=np.float32)
    for bi, b in enumerate(block_names):
        if not pools[b]:
            continue
        all_d = np.concatenate(pools[b], axis=0)
        out[bi, :] = np.quantile(all_d, quantiles).astype(np.float32)
    return out


def compute_local_intra_demo_thresholds(
    db: StateDatabase,
    cfg: LocalThresholdConfig,
    global_thresholds: np.ndarray,  # (n_blocks, n_quantiles)
) -> LocalThresholdMap:
    """Per-state ``intra_demo`` chunk-motion thresholds.

    For each state ``q``:
      1. State-NN → filter out self and same-demo near-time (keep far).
      2. For each surviving neighbor ``j`` in some demo, look up state at
         ``(demo_j, t_j + ta)`` (skip if absent).
      3. Pool ``delta_j = a[next] - a[j]`` and take per-block L2 quantiles.
      4. If valid sample count < ``min_neighbors_for_local``, fall back to the
         matching ``global_thresholds`` row.

    Same output format (``LocalThresholdMap``) as ``compute_local_thresholds``,
    so consumers (sequential_validate_from_cache_*) need no changes.
    """
    if db.records is None or db.embeddings is None:
        raise RuntimeError("DB not built")

    n = db.n_states
    block_slices = cfg.block_slice_dict()
    block_names = list(cfg.block_names)
    quantiles = np.asarray(cfg.quantiles, dtype=np.float64)
    n_blocks = len(block_names)
    n_q = len(quantiles)

    if global_thresholds.shape != (n_blocks, n_q):
        raise ValueError(f"global_thresholds shape {global_thresholds.shape} != ({n_blocks}, {n_q})")

    filtered = _intra_demo_sc_neighbors(db, cfg)
    actions = db.records.actions
    demo_id_int = db.records.demo_id_int
    t_arr = db.records.t
    idx_lookup = db.idx_lookup()
    ta = int(cfg.ta)
    min_nb = int(cfg.min_neighbors_for_local)

    thresholds = np.zeros((n, n_blocks, n_q), dtype=np.float32)
    fallback = np.zeros((n,), dtype=bool)
    n_neighbors_used = np.zeros((n,), dtype=np.int32)

    for qi in range(n):
        deltas = _gather_chunk_motion_deltas(
            filtered[qi],
            actions=actions,
            demo_id_int=demo_id_int,
            t_arr=t_arr,
            idx_lookup=idx_lookup,
            ta=ta,
        )
        n_neighbors_used[qi] = deltas.shape[0]
        if deltas.shape[0] < min_nb:
            fallback[qi] = True
            thresholds[qi] = global_thresholds
            continue
        for bi, b in enumerate(block_names):
            sl = block_slices[b]
            d = np.linalg.norm(deltas[:, sl], axis=-1).astype(np.float32)
            thresholds[qi, bi, :] = np.quantile(d, quantiles).astype(np.float32)

    return LocalThresholdMap(
        thresholds=thresholds,
        fallback_used=fallback,
        n_neighbors_used=n_neighbors_used,
        block_names=tuple(block_names),
        quantiles=tuple(float(q) for q in cfg.quantiles),
        global_thresholds=global_thresholds.astype(np.float32),
        config_hash=cfg.to_hash(),
    )
