"""Reader for LeRobot datasets (v2.x parquet layout).

Reads the dataset directly with ``pyarrow`` + ``meta/info.json`` — it does NOT
import the ``lerobot`` package, keeping surval's dependency surface small. Layout
(verified against a v2.0 dataset)::

    meta/info.json     codebase_version, features{dtype,shape}, data_path template
    meta/episodes.jsonl, tasks.jsonl, stats.json
    data/chunk-{c:03d}/episode_{e:06d}.parquet   one episode per parquet

Each parquet has columns ``action`` ([A]), ``observation.state`` ([D]),
``episode_index``, ``index``, plus per-camera ``observation.images.*`` entries.

Limitation: only numeric (non-video) columns are loaded. Image/video
observations (``dtype="video"`` / ``"image"``) are skipped — decode them via the
``lerobot`` package if your policy needs pixels.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence

import numpy as np

from .base import Episode, EpisodeReader

_SKIP_DTYPES = ("video", "image")


class LeRobotReader(EpisodeReader):
    """Iterate episodes from a LeRobot dataset root.

    Parameters
    ----------
    root
        Dataset root directory (the folder containing ``meta/`` and ``data/``).
    episodes
        Optional explicit list of episode indices to iterate. ``None`` (default)
        iterates ``0 .. total_episodes-1``.
    obs_keys
        Numeric observation columns to surface in ``Episode.obs``. ``None``
        (default) loads every non-video ``observation.*`` feature.
    state_key
        Column used as ``Episode.state`` (the ``obs_features`` fallback).
        Default ``"observation.state"``; set ``None`` to leave state empty.
    action_key
        Column holding actions. Default ``"action"``.
    """

    def __init__(
        self,
        root: str,
        *,
        episodes: Sequence[int] | None = None,
        obs_keys: Sequence[str] | None = None,
        state_key: str | None = "observation.state",
        action_key: str = "action",
    ) -> None:
        self.root = root
        self.action_key = action_key
        self.state_key = state_key
        self.info = self._load_info()
        self.features = self.info.get("features", {})
        self._data_path = self.info["data_path"]
        self._chunks_size = int(self.info.get("chunks_size", 1000))

        if obs_keys is not None:
            self.obs_keys = list(obs_keys)
        else:
            self.obs_keys = [
                k
                for k, spec in self.features.items()
                if k.startswith("observation.")
                and spec.get("dtype") not in _SKIP_DTYPES
            ]

        total = int(self.info.get("total_episodes", 0))
        self.episodes = list(episodes) if episodes is not None else list(range(total))

    def _load_info(self) -> dict:
        with open(os.path.join(self.root, "meta", "info.json")) as fh:
            return json.load(fh)

    def _episode_path(self, ep_idx: int) -> str:
        chunk = ep_idx // self._chunks_size
        rel = self._data_path.format(episode_chunk=chunk, episode_index=ep_idx)
        return os.path.join(self.root, rel)

    @staticmethod
    def _col_to_2d(series) -> np.ndarray:
        """Stack a parquet column of per-row vectors into ``[T, D]`` float32."""
        arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in series])
        return arr

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[Episode]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LeRobotReader requires pyarrow; install the 'lerobot' extra "
                "(`pip install surval[lerobot]`) or `pip install pyarrow`."
            ) from exc

        for ep_idx in self.episodes:
            path = self._episode_path(ep_idx)
            df = pq.read_table(path).to_pandas()
            actions = self._col_to_2d(df[self.action_key])
            obs = {k: self._col_to_2d(df[k]) for k in self.obs_keys if k in df.columns}
            state = (
                self._col_to_2d(df[self.state_key])
                if self.state_key and self.state_key in df.columns
                else None
            )
            yield Episode(
                demo_id=f"episode_{ep_idx:06d}",
                actions=actions,
                obs=obs,
                state=state,
            )
