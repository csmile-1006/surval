"""
TensorBoard scalar extraction + JSON cache I/O.

The two-stage workflow keeps the slow `EventAccumulator` import + parse out of
the aggregate step, so cluster jobs can extract once and many quick metric
sweeps can iterate the cache without touching the original event files.

Cache layout (mirrors the on-disk structure):

    <cache_dir>/manifest.json
    <cache_dir>/train/<seed_name>.json
    <cache_dir>/eval/<seed_name>/<hp_dir>/<cache_group_dir>.json

Each per-tb-dir JSON file has the schema:

    {
      "tb_dir": "/abs/path/to/tb",
      "event_file": "/abs/path/to/events.out.tfevents.<...>",
      "use_only_latest": true,
      "tags": {
        "Valid/Loss": {"steps": [...], "values": [...]},
        "Valid/Off_Manifold_Norm": {"steps": [...], "values": [...]},
        ...
      }
    }
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from typing import Iterable

import numpy as np


def _load_event_accumulator(load_path: str):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "tensorboard is required for extraction (pip install tensorboard)."
        ) from exc
    acc = EventAccumulator(load_path)
    acc.Reload()
    return acc


def _latest_event_file(tb_dir: str):
    pattern = os.path.join(tb_dir, "events.out.tfevents.*")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_tb_scalars(
    tb_dir: str,
    *,
    tag_filter: Iterable[str] | None = None,
    use_only_latest: bool = True,
) -> dict:
    """
    Extract scalar tags from a TensorBoard log directory.

    Args:
        tb_dir: dir containing events.out.tfevents.* files.
        tag_filter: optional iterable of tag names to keep. None = keep all.
        use_only_latest: if True, load only the most-recent event file
            (matches the v7 robomimic aggregator semantics).

    Returns:
        dict with keys: tb_dir, event_file, use_only_latest, tags.
    """
    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"tb_dir not found: {tb_dir}")

    load_path = tb_dir
    event_file = None
    if use_only_latest:
        event_file = _latest_event_file(tb_dir)
        if event_file is not None:
            load_path = event_file

    acc = _load_event_accumulator(load_path)
    available = acc.Tags().get("scalars", [])
    if tag_filter is not None:
        wanted = set(tag_filter)
        tags = [t for t in available if t in wanted]
    else:
        tags = list(available)

    out_tags: dict = {}
    for tag in tags:
        events = acc.Scalars(tag)
        out_tags[tag] = {
            "steps": [int(e.step) for e in events],
            "values": [float(e.value) for e in events],
        }

    return {
        "tb_dir": os.path.abspath(tb_dir),
        "event_file": os.path.abspath(event_file) if event_file else None,
        "use_only_latest": bool(use_only_latest),
        "tags": out_tags,
    }


# ---------------------------------------------------------------------------
# Cache file I/O
# ---------------------------------------------------------------------------


def cache_path_for_train(cache_dir: str, seed_name: str) -> str:
    return os.path.join(cache_dir, "train", f"{seed_name}.json")


def cache_path_for_seqval(
    cache_dir: str, seed_name: str, hp_dir: str, cache_group_dir: str
) -> str:
    return os.path.join(
        cache_dir, "eval", seed_name, hp_dir, f"{cache_group_dir}.json"
    )


def cache_path_for_seqcache_metrics(
    cache_dir: str, seed_name: str, cache_group_dir: str
) -> str:
    """Per-(seed, dataset) cache for HDF5-derived metrics (HP-invariant)."""
    return os.path.join(
        cache_dir, "seqcache_metrics", seed_name, f"{cache_group_dir}.json"
    )


def save_cache_file(payload: dict, out_path: str) -> None:
    payload = dict(payload)
    payload.setdefault("extracted_at", _dt.datetime.now().isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, out_path)


def load_cache_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregate-side cached loader (mimics robomimic load_scalar_series signature)
# ---------------------------------------------------------------------------


class CachedScalarLoader:
    """
    Reads pre-extracted JSON cache files and serves scalar series.

    Used by the aggregate step so it doesn't need tensorboard installed.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(cache_dir)
        self._mem: dict = {}

    # ----- file lookup -----

    def train_cache_path(self, seed_name: str) -> str:
        return cache_path_for_train(self.cache_dir, seed_name)

    def seqval_cache_path(self, seed_name: str, hp_dir: str, cache_group_dir: str) -> str:
        return cache_path_for_seqval(self.cache_dir, seed_name, hp_dir, cache_group_dir)

    def seqcache_metrics_path(self, seed_name: str, cache_group_dir: str) -> str:
        return cache_path_for_seqcache_metrics(self.cache_dir, seed_name, cache_group_dir)

    # ----- payload loading (LRU-free; payloads cached in memory) -----

    def _load(self, path: str) -> dict:
        cached = self._mem.get(path)
        if cached is not None:
            return cached
        if not os.path.exists(path):
            raise FileNotFoundError(f"cache file not found: {path}")
        payload = load_cache_file(path)
        self._mem[path] = payload
        return payload

    def load_train(self, seed_name: str) -> dict:
        return self._load(self.train_cache_path(seed_name))

    def load_seqval(self, seed_name: str, hp_dir: str, cache_group_dir: str) -> dict:
        return self._load(self.seqval_cache_path(seed_name, hp_dir, cache_group_dir))

    def load_seqcache_metrics(self, seed_name: str, cache_group_dir: str) -> dict:
        """Load the per-(seed, dataset) seqcache-derived metrics payload."""
        return self._load(self.seqcache_metrics_path(seed_name, cache_group_dir))

    # ----- scalar series accessor -----

    @staticmethod
    def get_series(payload: dict, tag: str):
        """
        Return (steps, values) as numpy arrays for a tag from a cache payload.
        Raises KeyError if the tag isn't present.
        """
        tags = payload.get("tags", {})
        if tag not in tags:
            available = sorted(tags.keys())
            raise KeyError(
                f"tag '{tag}' not found in cache for {payload.get('tb_dir')}. "
                f"Available (first 30): {available[:30]}"
            )
        entry = tags[tag]
        steps = np.asarray(entry["steps"], dtype=np.float64)
        values = np.asarray(entry["values"], dtype=np.float64)
        return steps, values

    def load_train_series(self, seed_name: str, tag: str):
        return self.get_series(self.load_train(seed_name), tag)

    def load_seqval_series(
        self, seed_name: str, hp_dir: str, cache_group_dir: str, tag: str
    ):
        return self.get_series(self.load_seqval(seed_name, hp_dir, cache_group_dir), tag)
