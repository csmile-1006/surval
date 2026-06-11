"""Reader for robomimic / robocasa / dexmimicgen source HDF5 datasets.

Schema (verified against dexmimicgen ``*_processed.hdf5``)::

    data/                      attrs: env_args (json), total
      demo_<i>/                attrs: num_samples, model_file
        actions   float [T_ep, A]
        obs/<key> float/uint8 [T_ep, ...]    (proprio + image keys)
        ... (states, rewards, dones, action_dict, datagen_info)
    mask/                      split name -> [bytes demo names]
      train, valid, ...
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

import numpy as np

from .base import Episode, EpisodeReader


class RobomimicHDF5Reader(EpisodeReader):
    """Iterate demos from a robomimic-format HDF5 file.

    Parameters
    ----------
    path
        Path to the source ``.hdf5``.
    split
        Optional mask name under ``mask/`` (e.g. ``"valid"``, ``"train"``).
        ``None`` (default) iterates every demo in ``data/``.
    obs_keys
        Observation keys to surface in ``Episode.obs`` (passed to the policy
        callbacks). ``None`` (default) loads every key under ``demo/obs``.
        Restrict this to skip loading large image arrays you don't need.
    state_keys
        Low-dimensional ``obs`` keys concatenated into ``Episode.state`` (the
        ``obs_features`` fallback). ``None`` leaves ``state`` empty — supply a
        ``feature_fn`` to the builder instead, or set this to your proprio keys.
    demo_order
        ``"natural"`` (default) sorts ``demo_<n>`` numerically; ``"file"`` keeps
        the HDF5 key order.
    """

    def __init__(
        self,
        path: str,
        *,
        split: str | None = None,
        obs_keys: Sequence[str] | None = None,
        state_keys: Sequence[str] | None = None,
        demo_order: str = "natural",
    ) -> None:
        self.path = path
        self.split = split
        self.obs_keys = list(obs_keys) if obs_keys is not None else None
        self.state_keys = list(state_keys) if state_keys is not None else None
        if demo_order not in ("natural", "file"):
            raise ValueError(f"demo_order must be 'natural' or 'file'; got {demo_order!r}")
        self.demo_order = demo_order
        self._demo_names = self._resolve_demo_names()

    def _open(self):
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError("RobomimicHDF5Reader requires h5py.") from exc
        return h5py.File(self.path, "r")

    @staticmethod
    def _natural_key(name: str) -> tuple[int, object]:
        if "_" in name:
            try:
                return (0, int(name.rsplit("_", 1)[-1]))
            except ValueError:
                pass
        return (1, name)

    def _resolve_demo_names(self) -> list[str]:
        with self._open() as f:
            if self.split is not None:
                if "mask" not in f or self.split not in f["mask"]:
                    avail = list(f["mask"].keys()) if "mask" in f else []
                    raise KeyError(
                        f"split {self.split!r} not found under mask/; available: {avail}"
                    )
                names = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in f["mask"][self.split][:]
                ]
            else:
                names = list(f["data"].keys())
        if self.demo_order == "natural":
            names = sorted(names, key=self._natural_key)
        return names

    @property
    def env_args(self) -> dict:
        """Parsed ``data.attrs['env_args']`` JSON (empty dict if absent)."""
        with self._open() as f:
            raw = f["data"].attrs.get("env_args")
        if raw is None:
            return {}
        return json.loads(raw)

    def __len__(self) -> int:
        return len(self._demo_names)

    def __iter__(self) -> Iterator[Episode]:
        with self._open() as f:
            data = f["data"]
            for name in self._demo_names:
                grp = data[name]
                actions = np.asarray(grp["actions"][:], dtype=np.float32)
                obs_grp = grp["obs"]
                keys = self.obs_keys if self.obs_keys is not None else list(obs_grp.keys())
                obs = {k: np.asarray(obs_grp[k][:]) for k in keys}
                state = None
                if self.state_keys:
                    parts = [np.asarray(obs_grp[k][:], dtype=np.float32) for k in self.state_keys]
                    parts = [p.reshape(p.shape[0], -1) for p in parts]
                    state = np.concatenate(parts, axis=1)
                yield Episode(demo_id=name, actions=actions, obs=obs, state=state)
