"""State embedding database with FAISS index + per-state action records.

The database is the central artifact of Phase 1. After it is built, Phase 2
operates entirely on its arrays (no more image encoding).

Layout on disk (under ``output_dir``):
    embeddings.npy           # (N, D_state) float32, L2-normalized
    actions.npy              # (N, action_dim) float32, the t=0 action
    demo_id_int.npy          # (N,) int32
    demo_id_str.json         # mapping {int_id: str_id}
    t.npy                    # (N,) int32, idx_in_demo
    faiss.index              # FAISS IndexFlatIP (or HNSW)
    proprio_normalizer.pkl   # ProprioNormalizer state
    config.json              # serialized LocalThresholdConfig
    state_db.complete        # sentinel
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np

from .config import LocalThresholdConfig


@dataclass
class StateRecords:
    """Parallel arrays of per-state metadata."""

    actions: np.ndarray  # (N, action_dim) float32
    demo_id_int: np.ndarray  # (N,) int32
    t: np.ndarray  # (N,) int32, idx_in_demo
    demo_id_str_by_int: dict[int, str]  # int -> str

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class StateDatabase:
    """FAISS-backed state embedding store. Build once, query many times."""

    def __init__(self, cfg: LocalThresholdConfig):
        self.cfg = cfg
        self.embeddings: np.ndarray | None = None
        self.records: StateRecords | None = None
        self._index = None  # faiss.Index
        self._idx_lookup: dict[tuple[int, int], int] | None = None

    # ---------- build / save / load ----------

    def attach(self, embeddings: np.ndarray, records: StateRecords) -> None:
        """Set arrays + build FAISS index. Used by build_state_db.py after the
        encoding pass completes."""
        if embeddings.dtype != np.float32:
            raise ValueError(f"embeddings must be float32, got {embeddings.dtype}")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
        if embeddings.shape[0] != len(records):
            raise ValueError(f"embeddings.shape[0]={embeddings.shape[0]} != len(records)={len(records)}")
        norms = np.linalg.norm(embeddings, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"embeddings must be L2-normalized; got norm range [{norms.min():.4f}, {norms.max():.4f}]")

        self.embeddings = embeddings
        self.records = records
        self._index = self._build_index(embeddings)

    def _build_index(self, embeddings: np.ndarray):
        import faiss

        d = embeddings.shape[1]
        if self.cfg.index_type == "flat_ip":
            index = faiss.IndexFlatIP(d)
        elif self.cfg.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efSearch = 64
        else:
            raise ValueError(f"Unknown index_type={self.cfg.index_type!r}")
        index.add(embeddings)
        return index

    def save(self, path: str) -> None:
        import faiss

        if self.embeddings is None or self.records is None or self._index is None:
            raise RuntimeError("Nothing to save; call attach() first")
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        np.save(os.path.join(path, "actions.npy"), self.records.actions)
        np.save(os.path.join(path, "demo_id_int.npy"), self.records.demo_id_int)
        np.save(os.path.join(path, "t.npy"), self.records.t)
        with open(os.path.join(path, "demo_id_str.json"), "w") as f:
            json.dump(
                {str(k): v for k, v in self.records.demo_id_str_by_int.items()},
                f,
                indent=2,
            )
        faiss.write_index(self._index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str, cfg: LocalThresholdConfig) -> StateDatabase:
        import faiss

        db = cls(cfg)
        db.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        actions = np.load(os.path.join(path, "actions.npy"))
        demo_id_int = np.load(os.path.join(path, "demo_id_int.npy"))
        t = np.load(os.path.join(path, "t.npy"))
        with open(os.path.join(path, "demo_id_str.json")) as f:
            mapping = {int(k): v for k, v in json.load(f).items()}
        db.records = StateRecords(actions=actions, demo_id_int=demo_id_int, t=t, demo_id_str_by_int=mapping)
        db._index = faiss.read_index(os.path.join(path, "faiss.index"))
        return db

    # ---------- query / filter ----------

    def query(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Cosine-similarity k-NN.

        Returns:
            distances: (Q, k) float32 — *cosine distance* (= 1 - inner product).
            indices: (Q, k) int64.
        """
        if self._index is None:
            raise RuntimeError("Index not built")
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        sims, indices = self._index.search(query_embeddings, k)  # IP == cosine for unit vectors
        distances = (1.0 - sims).astype(np.float32)
        return distances, indices.astype(np.int64)

    def filter_neighbors(
        self,
        indices: np.ndarray,  # (Q, k) int64
        query_demo_id_int: np.ndarray,  # (Q,) int32
        query_t: np.ndarray,  # (Q,) int32
        cfg: LocalThresholdConfig | None = None,
    ) -> list[np.ndarray]:
        """Apply same_demo_allowed and temporal_exclusion_radius.

        Returns a list of length Q. Each element is a 1-D int64 array of
        retained neighbor *db indices* (not positions within the row).
        """
        cfg = cfg or self.cfg
        if self.records is None:
            raise RuntimeError("DB not built")
        nb_demo = self.records.demo_id_int[indices]  # (Q, k)
        nb_t = self.records.t[indices]  # (Q, k)

        same_demo = nb_demo == query_demo_id_int[:, None]  # (Q, k)
        # Always exclude self (same demo, same t).
        is_self = same_demo & (nb_t == query_t[:, None])

        if not cfg.same_demo_allowed:
            mask_keep = ~same_demo
        else:
            # Within same demo, drop neighbors with |t - t_q| <= radius (this
            # also drops self).
            radius = int(cfg.temporal_exclusion_radius)
            close_in_time = same_demo & (np.abs(nb_t - query_t[:, None]) <= radius)
            mask_keep = ~close_in_time
        mask_keep = mask_keep & ~is_self

        out: list[np.ndarray] = []
        for q in range(indices.shape[0]):
            row = indices[q][mask_keep[q]]
            out.append(row)
        return out

    def filter_neighbors_keep_same_demo_far(
        self,
        indices: np.ndarray,  # (Q, k) int64
        query_demo_id_int: np.ndarray,  # (Q,) int32
        query_t: np.ndarray,  # (Q,) int32
        cfg: LocalThresholdConfig | None = None,
    ) -> list[np.ndarray]:
        """Drop self and same-demo neighbors with ``|t - t_q| <= radius``, but
        always keep same-demo neighbors farther in time and all cross-demo
        neighbors.

        Used by ``intra_demo_sc`` where same-demo far frames are needed so the
        chunk-motion ``a[t+ta] - a[t]`` lookup can succeed (cfg.same_demo_allowed
        is intentionally ignored here).
        """
        cfg = cfg or self.cfg
        if self.records is None:
            raise RuntimeError("DB not built")
        nb_demo = self.records.demo_id_int[indices]  # (Q, k)
        nb_t = self.records.t[indices]  # (Q, k)

        same_demo = nb_demo == query_demo_id_int[:, None]  # (Q, k)
        radius = int(cfg.temporal_exclusion_radius)
        close_in_time = same_demo & (np.abs(nb_t - query_t[:, None]) <= radius)
        mask_keep = ~close_in_time  # also drops self (same demo, |Δt|=0 ≤ radius)

        return [indices[q][mask_keep[q]] for q in range(indices.shape[0])]

    # ---------- (demo, t) lookup ----------

    def idx_lookup(self) -> dict[tuple[int, int], int]:
        """Return a cached ``{(demo_id_int, t): state_idx}`` mapping.

        Used by ``intra_demo_sc`` to resolve ``actions[idx_at(demo_j, t_j+ta)]``
        for chunk-motion deltas. Built lazily on first call.
        """
        if self.records is None:
            raise RuntimeError("DB not built")
        if self._idx_lookup is None:
            di = self.records.demo_id_int
            tt = self.records.t
            self._idx_lookup = {(int(di[i]), int(tt[i])): i for i in range(int(di.shape[0]))}
        return self._idx_lookup

    def idx_at(self, demo_id_int: int, t: int) -> int | None:
        """Return the db index for ``(demo_id_int, t)``, or None if absent."""
        return self.idx_lookup().get((int(demo_id_int), int(t)))

    # ---------- introspection ----------

    @property
    def n_states(self) -> int:
        return 0 if self.embeddings is None else int(self.embeddings.shape[0])

    @property
    def state_dim(self) -> int:
        if self.embeddings is None:
            raise RuntimeError("DB not built")
        return int(self.embeddings.shape[1])
