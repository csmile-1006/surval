"""HDF5 sequential-validation cache I/O.

The canonical schema produced by :func:`write_seqcache_hdf5` is what
``surval.sequential_validate._load_cache_hdf5`` consumes::

    File attrs:
      checkpoint        : str
      step              : int
      num_cache_samples : int
      num_rows          : int
      num_steps         : int    (T)
      ac_dim            : int
      val_loss          : float  (NaN if not measured)
      has_obs_features  : int    (0/1)
      obs_feat_dim      : int    (0 if absent)

    data/                          (group; attr ``total`` = num_rows)
      demo_<i>/                    (i = enumerated index 0..D-1)
        attrs:
          demo_id     : str        (original ID from dataset)
          num_samples : int
        datasets:
          index_in_demo : int64 [N_demo]
          actions       : float32 [N_demo, T, A]
          pred_actions  : float32 [S, N_demo, T, A]
          obs_features  : float32 [N_demo, F] or [N_demo, T, F]   (optional)

    metrics/valid/
      Loss : float64                                (scalar, NaN if not measured)
      off_manifold_norm/                            (group, optional)
        sample_<k> : float64                        (one per cache sample)

Filenames follow ``seqcache_step_{step:06d}.hdf5`` so that
``surval.sequential_validate._find_caches_in_dir``'s glob discovers them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np


def sort_demo_keys(demo_keys: Sequence[str]) -> list[str]:
    """Sort demo group names by their numeric suffix when they look like
    ``demo_<n>`` / ``row_<n>``, falling back to lexicographic order.
    """

    def _key(d: str) -> tuple[int, Any]:
        if d.startswith(("demo_", "row_")):
            try:
                return (0, int(d.split("_")[-1]))
            except (ValueError, IndexError):
                pass
        return (1, d)

    return sorted(demo_keys, key=_key)


def concat_trim_time(arr_list: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate ``[B, T_i, A_i]`` arrays along ``B`` after trimming to common
    minimum ``T`` and ``A``. Returns float32.
    """
    if len(arr_list) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    min_t = min(a.shape[1] for a in arr_list)
    min_a = min(a.shape[2] for a in arr_list)
    if min_t <= 0 or min_a <= 0:
        return np.zeros((sum(a.shape[0] for a in arr_list), 0, 0), dtype=np.float32)
    return np.concatenate([a[:, :min_t, :min_a] for a in arr_list], axis=0).astype(np.float32)


def concat_trim_time_feature(arr_list: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate ``[B, T_i, F_i]`` feature arrays along ``B`` after trimming
    to common minimum ``T`` and ``F``. Returns float32.
    """
    if len(arr_list) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    min_t = min(a.shape[1] for a in arr_list)
    min_f = min(a.shape[2] for a in arr_list)
    if min_t <= 0 or min_f <= 0:
        return np.zeros((sum(a.shape[0] for a in arr_list), 0, 0), dtype=np.float32)
    return np.concatenate([a[:, :min_t, :min_f] for a in arr_list], axis=0).astype(np.float32)


def group_rows_by_demo(
    demo_ids: Sequence[str] | np.ndarray,
    index_in_demo: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Group flat row indices by demo id, sorted within demo by ``index_in_demo``.

    Returns ``{demo_id: row_indices_into_input}``.
    """
    by_demo: dict[str, list[int]] = {}
    for i, did in enumerate(np.asarray(demo_ids).tolist()):
        by_demo.setdefault(str(did), []).append(i)
    index_in_demo_arr = np.asarray(index_in_demo)
    out: dict[str, np.ndarray] = {}
    for did, rows in by_demo.items():
        idxs = np.array(rows, dtype=np.int64)
        local_order = np.argsort(index_in_demo_arr[idxs])
        out[did] = idxs[local_order]
    return out


def write_seqcache_hdf5(
    out_path: str,
    *,
    demo_ids: Sequence[str] | np.ndarray,
    index_in_demo: Sequence[int] | np.ndarray,
    actions: np.ndarray,
    pred_actions_list: Sequence[np.ndarray],
    checkpoint: str,
    step: int,
    val_loss: float | None = None,
    off_manifold_norms: Mapping[int, float] | float | None = None,
    obs_features: np.ndarray | None = None,
) -> None:
    """Write the canonical sequential-validation cache HDF5 file.

    Parameters
    ----------
    out_path
        Destination ``.hdf5`` path.
    demo_ids
        Length-``N`` array/list of original demo identifiers (one per row).
    index_in_demo
        Length-``N`` integer array of within-demo positions.
    actions
        ``[N, T, A]`` ground-truth actions.
    pred_actions_list
        Sequence of ``S`` arrays, each ``[N, T, A]``, holding predicted
        actions per cache sample. The leading axis ``S`` of the stored
        ``pred_actions`` dataset comes from the length of this list.
    checkpoint
        Free-form checkpoint identifier (path or step descriptor).
    step
        Integer training step (or epoch, for frameworks that count by epoch).
        Stored as ``f.attrs["step"]`` so the reader doesn't need to parse the
        filename.
    val_loss
        Validation loss scalar. Stored under ``metrics/valid/Loss``;
        ``NaN`` if ``None``.
    off_manifold_norms
        Off-manifold norm value(s). Either ``dict[int, float]`` (per cache
        sample, key = sample index) or a single scalar (treated as
        ``{0: scalar}``). ``None`` skips the group entirely.
    obs_features
        Optional state-feature array, ``[N, F]`` or ``[N, T, F]``.
    """
    actions = np.asarray(actions)
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [N, T, A]; got {actions.shape}")
    if len(pred_actions_list) == 0:
        raise ValueError("pred_actions_list must contain at least one array")
    pred_arrs = [np.asarray(p) for p in pred_actions_list]
    for i, p in enumerate(pred_arrs):
        if p.shape != actions.shape:
            raise ValueError(
                f"pred_actions_list[{i}] shape {p.shape} does not match actions shape {actions.shape}"
            )

    demo_ids_arr = np.asarray(demo_ids)
    index_in_demo_arr = np.asarray(index_in_demo, dtype=np.int64)
    n_rows = actions.shape[0]
    if demo_ids_arr.shape[0] != n_rows or index_in_demo_arr.shape[0] != n_rows:
        raise ValueError(
            f"row mismatch: demo_ids={demo_ids_arr.shape[0]}, "
            f"index_in_demo={index_in_demo_arr.shape[0]}, actions={n_rows}"
        )

    if obs_features is not None:
        obs_features = np.asarray(obs_features)
        if obs_features.shape[0] != n_rows:
            raise ValueError(f"obs_features rows {obs_features.shape[0]} != actions rows {n_rows}")
        if obs_features.ndim not in (2, 3):
            raise ValueError(f"obs_features must be [N, F] or [N, T, F]; got {obs_features.shape}")
        obs_feat_dim = int(obs_features.shape[-1])
    else:
        obs_feat_dim = 0

    if off_manifold_norms is None:
        omn_map: dict[int, float] = {}
    elif isinstance(off_manifold_norms, Mapping):
        omn_map = {int(k): float(v) for k, v in off_manifold_norms.items()}
    else:
        omn_map = {0: float(off_manifold_norms)}

    val_loss_value = float(val_loss) if val_loss is not None else float("nan")
    by_demo = group_rows_by_demo(demo_ids_arr, index_in_demo_arr)
    has_obs_features = obs_features is not None

    with h5py.File(out_path, "w") as f:
        f.attrs["checkpoint"] = str(checkpoint)
        f.attrs["step"] = int(step)
        f.attrs["num_cache_samples"] = int(len(pred_arrs))
        f.attrs["num_rows"] = int(n_rows)
        f.attrs["num_steps"] = int(actions.shape[1])
        f.attrs["ac_dim"] = int(actions.shape[2])
        f.attrs["val_loss"] = val_loss_value
        f.attrs["has_obs_features"] = int(has_obs_features)
        f.attrs["obs_feat_dim"] = int(obs_feat_dim)

        data_grp = f.create_group("data")
        data_grp.attrs["total"] = int(n_rows)

        for enum_idx, demo_id in enumerate(sort_demo_keys(list(by_demo.keys()))):
            idxs = by_demo[demo_id]
            grp = data_grp.create_group(f"demo_{enum_idx}")
            grp.attrs["demo_id"] = str(demo_id)
            grp.attrs["num_samples"] = int(len(idxs))
            grp.create_dataset("index_in_demo", data=index_in_demo_arr[idxs], compression="gzip")
            grp.create_dataset("actions", data=actions[idxs], compression="gzip")
            pred_slice = np.stack([p[idxs] for p in pred_arrs], axis=0)
            grp.create_dataset("pred_actions", data=pred_slice, compression="gzip")
            if has_obs_features:
                grp.create_dataset("obs_features", data=obs_features[idxs], compression="gzip")

        metrics_grp = f.create_group("metrics")
        valid_grp = metrics_grp.create_group("valid")
        valid_grp.create_dataset("Loss", data=np.array(val_loss_value, dtype=np.float64))
        if omn_map:
            omn_grp = valid_grp.create_group("off_manifold_norm")
            for s_idx, omn_val in sorted(omn_map.items()):
                omn_grp.create_dataset(f"sample_{s_idx}", data=np.array(omn_val, dtype=np.float64))


class ReservoirSampler:
    """Bounded reservoir sampling for parallel arrays.

    Maintains uniformly random subsets of fixed ``capacity`` over a stream of
    ``observe`` calls. Tracks several parallel ``slots`` so several arrays
    (e.g. features, gt actions, predicted actions) stay aligned per row.

    Example::

        sampler = ReservoirSampler(
            capacity=1024, slots=("feat", "gt", "pred"), rng_seed=42,
        )
        for feat_b, gt_b, pred_b in stream:           # batched iteration
            for i in range(feat_b.shape[0]):
                sampler.observe(feat=feat_b[i], gt=gt_b[i], pred=pred_b[i])
        out = sampler.collect()  # {"feat": [...], "gt": [...], "pred": [...]}

    With ``capacity=None`` the sampler keeps every observation.
    """

    def __init__(
        self,
        capacity: int | None,
        *,
        slots: Sequence[str] = ("item",),
        rng_seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.slots = tuple(slots)
        self._rng = np.random.default_rng(rng_seed)
        self._buffers: dict[str, list[Any]] = {s: [] for s in self.slots}
        self._n_seen = 0

    def observe(self, **kwargs: Any) -> None:
        if set(kwargs.keys()) != set(self.slots):
            raise ValueError(
                f"observe() expected slot kwargs {set(self.slots)}, got {set(kwargs.keys())}"
            )
        self._n_seen += 1
        first_buf = self._buffers[self.slots[0]]
        if self.capacity is None or len(first_buf) < self.capacity:
            for s, v in kwargs.items():
                self._buffers[s].append(np.asarray(v).copy())
            return
        j = int(self._rng.integers(0, self._n_seen))
        if j < self.capacity:
            for s, v in kwargs.items():
                self._buffers[s][j] = np.asarray(v).copy()

    def collect(self) -> dict[str, list[np.ndarray]]:
        return {s: list(buf) for s, buf in self._buffers.items()}

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def __len__(self) -> int:
        return len(self._buffers[self.slots[0]])


def compute_off_manifold_errors(
    pred_actions: np.ndarray,
    state_features: np.ndarray,
    expert_actions: np.ndarray | None,
    k: int = 5,
) -> np.ndarray:
    """Project each predicted action onto the column space of expert actions
    at the ``k`` nearest neighbor states; return the residual norm per sample.

    Returns ``[B, T]`` if inputs are 3D, else ``[B]``. Returns zeros when
    ``expert_actions`` is ``None``.

    Requires scikit-learn (imported lazily).
    """
    if expert_actions is None:
        if pred_actions.ndim == 3:
            return np.zeros(pred_actions.shape[:2], dtype=pred_actions.dtype)
        return np.zeros(pred_actions.shape[0], dtype=pred_actions.dtype)

    had_temporal_dim = state_features.ndim == 3
    if had_temporal_dim:
        b_dim, t_dim, d_dim = state_features.shape
        state_features = state_features.reshape(b_dim * t_dim, d_dim)
        pred_actions = pred_actions.reshape(b_dim * t_dim, -1)
        expert_actions = expert_actions.reshape(b_dim * t_dim, -1)
    else:
        b_dim = state_features.shape[0]
        t_dim = 1

    n_total = state_features.shape[0]
    k_nn = min(k, n_total - 1)
    if k_nn < 1:
        if had_temporal_dim:
            return np.zeros((b_dim, t_dim), dtype=pred_actions.dtype)
        return np.zeros((b_dim,), dtype=pred_actions.dtype)

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_off_manifold_errors requires scikit-learn; "
            "install via `pip install scikit-learn`."
        ) from exc

    nbrs = NearestNeighbors(n_neighbors=k_nn + 1, algorithm="auto", n_jobs=-1)
    nbrs.fit(state_features)
    _, knn_indices = nbrs.kneighbors(state_features)
    knn_indices = knn_indices[:, 1 : k_nn + 1]

    a_mat = expert_actions[knn_indices]  # [N, k, ac_dim]
    if k_nn == 1:
        proj_a = a_mat[:, 0, :]
        errors = np.linalg.norm(pred_actions - proj_a, axis=1)
    else:
        gram = np.einsum("nka,nla->nkl", a_mat, a_mat)
        gram = gram + 1e-9 * np.eye(k_nn)
        rhs = np.einsum("nka,na->nk", a_mat, pred_actions)
        try:
            coeffs = np.linalg.solve(gram, rhs)
            proj_a = np.einsum("nka,nk->na", a_mat, coeffs)
        except np.linalg.LinAlgError:
            proj_a = a_mat.mean(axis=1)
        errors = np.linalg.norm(pred_actions - proj_a, axis=1)

    if had_temporal_dim:
        return errors.reshape(b_dim, t_dim)
    return errors


def compute_off_manifold_norm(
    pred_actions: np.ndarray,
    state_features: np.ndarray,
    expert_actions: np.ndarray | None,
    k: int = 5,
) -> float:
    """Mean off-manifold projection error (see :func:`compute_off_manifold_errors`)."""
    errors = compute_off_manifold_errors(
        pred_actions=pred_actions,
        state_features=state_features,
        expert_actions=expert_actions,
        k=k,
    )
    return float(errors.mean()) if errors.size > 0 else 0.0
