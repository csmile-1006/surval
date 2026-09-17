"""
Extract per-checkpoint scalars from seqcache HDF5 files.

For each `<seqcache_root>/<seed_name>/<cache_group_dir>/seqcache_epoch_<n>.hdf5`,
this produces three (epoch -> value) series:

  Cache/Valid/Loss               : metrics/valid/Loss attribute (scalar)
  Cache/Valid/Off_Manifold_Norm  : metrics/valid/off_manifold_norm/sample_0 if
                                   present (new schema), else the flat
                                   metrics/valid/Off_Manifold_Norm scalar
                                   (legacy robomimic schema)
  Cache/Valid/MSE_mean_pred      : mean((mean_s(pred_actions) - actions)**2)
                                   across all demos / time / action dims. The
                                   per-sample mean is taken first, so this
                                   measures point-estimate accuracy without
                                   charging the policy for sampling variance.

These three are HP-invariant per (seed, dataset), so they're written to a
seed/dataset–keyed JSON cache:

    <output_cache_dir>/seqcache_metrics/<seed_name>/<cache_group_dir>.json

The aggregator (CachedScalarLoader) falls back to this file when a tag is not
present in the seqval-TB cache.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterable

import numpy as np

# Tag names emitted by this extractor.
LOSS_TAG = "Cache/Valid/Loss"
OMN_TAG  = "Cache/Valid/Off_Manifold_Norm"
# MSE baseline (lower-is-better):
#   MSE_mean_pred : E_{n,t,a}[(mean_s(pred) - gt)^2]  (Bayes-optimal point-est)
# The sample mean is taken before the error, so sampling variance is not
# charged against a stochastic policy.
MSE_MEAN_PRED_TAG = "Cache/Valid/MSE_mean_pred"
MSE_VARIANT_TAGS = [MSE_MEAN_PRED_TAG]

_EPOCH_RE = re.compile(r"epoch_(\d+)")
# openpi checkpoints are indexed by training step, not epoch; both filename
# conventions carry the same payload, so the extractor accepts either.
_STEP_RE = re.compile(r"step_(\d+)")
_INDEX_RE = re.compile(r"(?:epoch|step)_(\d+)")
_CACHE_GLOBS = {"epoch": "seqcache_epoch_*.hdf5", "step": "seqcache_step_*.hdf5"}


def _file_index(path: str) -> int:
    """Checkpoint index parsed from a cache filename (epoch or step)."""
    m = _INDEX_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def detect_cache_mode(group_path: str) -> str | None:
    """Return "epoch"/"step" for the caches present, or None when there are none.

    Raises when a directory mixes both conventions, since the two indices are
    not comparable and silently merging them would corrupt every series.
    """
    present = [m for m, g in _CACHE_GLOBS.items() if glob.glob(os.path.join(group_path, g))]
    if len(present) > 1:
        raise ValueError(f"Cache group mixes epoch and step files: {group_path}")
    return present[0] if present else None


def _list_epoch_files(group_path: str, mode: str | None = None) -> list:
    """Return seqcache cache files sorted by their checkpoint index.

    ``mode`` selects the filename convention; ``None`` auto-detects it so
    existing epoch-based callers keep working unchanged.
    """
    mode = mode or detect_cache_mode(group_path)
    if mode is None:
        return []
    if mode not in _CACHE_GLOBS:
        raise ValueError(f"Unknown cache mode {mode!r}; choose one of {sorted(_CACHE_GLOBS)}")
    files = glob.glob(os.path.join(group_path, _CACHE_GLOBS[mode]))
    return sorted(files, key=_file_index)


def _read_loss(h) -> float:
    if "metrics/valid/Loss" in h:
        v = float(h["metrics/valid/Loss"][()])
        return v
    if "val_loss" in h.attrs:
        return float(h.attrs["val_loss"])
    return float("nan")


def _read_omn(h) -> float:
    """
    Prefer the new sample_<k> group (use sample_0 to match the surval reader);
    fall back to the legacy flat scalar at metrics/valid/Off_Manifold_Norm.
    """
    grp_path = "metrics/valid/off_manifold_norm"
    if grp_path in h:
        node = h[grp_path]
        # If it's a Group (newer schema), pull sample_0 (or mean across samples
        # if sample_0 missing).
        if hasattr(node, "keys"):
            keys = list(node.keys())
            if not keys:
                return float("nan")
            if "sample_0" in node:
                return float(node["sample_0"][()])
            vals = [float(node[k][()]) for k in keys]
            return float(np.mean(vals)) if vals else float("nan")
        # Dataset (unusual but handle it):
        return float(node[()])
    legacy = "metrics/valid/Off_Manifold_Norm"
    if legacy in h:
        return float(h[legacy][()])
    return float("nan")


def _read_mse_variants(h) -> dict:
    """
    Lower-is-better MSE baseline computed from one checkpoint's HDF5 cache.
    Returns dict keyed by MSE_*_TAG names; NaN-on-failure per key.
    """
    keys = [MSE_MEAN_PRED_TAG]
    accum = {k: [0.0, 0] for k in keys}
    if "data" not in h:
        return {k: float("nan") for k in keys}
    for dk in h["data"].keys():
        demo = h["data"][dk]
        if "actions" not in demo or "pred_actions" not in demo:
            continue
        gt   = demo["actions"][...].astype(np.float64, copy=False)
        pred = demo["pred_actions"][...].astype(np.float64, copy=False)
        if pred.ndim == 3:
            pred = pred[None]                  # promote to [1, N, T, A]
        n = min(gt.shape[0], pred.shape[1])
        t = min(gt.shape[1], pred.shape[2])
        a = min(gt.shape[2], pred.shape[3])
        gt_s = gt[:n, :t, :a]
        pr_s = pred[:, :n, :t, :a]
        # Sample-mean prediction first, then the error.
        pm = pr_s.mean(axis=0)                  # [n, t, a]
        d = pm - gt_s
        accum[MSE_MEAN_PRED_TAG][0] += float((d * d).sum())
        accum[MSE_MEAN_PRED_TAG][1] += int(d.size)
    out = {}
    for k in keys:
        s, n = accum[k]
        out[k] = (s / n) if n > 0 else float("nan")
    return out


def _read_step(h, fallback_path: str) -> int:
    if "step" in h.attrs:
        try:
            return int(h.attrs["step"])
        except Exception:
            pass
    if "epoch" in h.attrs:
        try:
            return int(h.attrs["epoch"])
        except Exception:
            pass
    return _file_index(fallback_path)


def extract_seqcache_metrics_for_group(group_path: str, mode: str | None = None) -> dict:
    """
    Walk one cache_group_dir and return a JSON-cache-shaped payload with the
    three Cache/Valid/* tags as (steps, values) series sorted by checkpoint
    index (epoch or training step, whichever the cache files use).

    Returns a dict suitable for surval.tb_aggregate.tb_io.save_cache_file,
    including a "kind" marker so consumers can tell which extractor produced
    it.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required to read seqcache HDF5 files.") from exc

    files = _list_epoch_files(group_path, mode)
    steps: list = []
    loss_vals: list = []
    omn_vals: list = []
    mse_series: dict = {k: [] for k in MSE_VARIANT_TAGS}
    sources: list = []

    for path in files:
        try:
            with h5py.File(path, "r") as h:
                step = _read_step(h, path)
                loss = _read_loss(h)
                omn  = _read_omn(h)
                mse_vals = _read_mse_variants(h)
        except OSError as exc:
            print(f"  [seqcache] skip unreadable file: {path} ({exc})")
            continue
        steps.append(int(step))
        loss_vals.append(loss)
        omn_vals.append(omn)
        for k in MSE_VARIANT_TAGS:
            mse_series[k].append(mse_vals.get(k, float("nan")))
        sources.append(os.path.basename(path))

    tags = {
        LOSS_TAG: {"steps": steps, "values": loss_vals},
        OMN_TAG:  {"steps": steps, "values": omn_vals},
    }
    for k in MSE_VARIANT_TAGS:
        tags[k] = {"steps": steps, "values": mse_series[k]}
    return {
        "kind": "seqcache_metrics",
        "group_path": os.path.abspath(group_path),
        "source_files": sources,
        "tags": tags,
    }


def iter_seed_groups(seqcache_root: str, seed_glob_prefix: str = "") -> Iterable[tuple]:
    """
    Yield (seed_name, cache_group_dir, abs_group_path) for every seed/group
    combination that contains at least one seqcache_epoch_*.hdf5 file.
    """
    for seed_name in sorted(os.listdir(seqcache_root)):
        seed_path = os.path.join(seqcache_root, seed_name)
        if not os.path.isdir(seed_path):
            continue
        if seed_glob_prefix and not seed_name.startswith(seed_glob_prefix):
            continue
        for group_name in sorted(os.listdir(seed_path)):
            group_path = os.path.join(seed_path, group_name)
            if not os.path.isdir(group_path):
                continue
            if not glob.glob(os.path.join(group_path, "seqcache_epoch_*.hdf5")):
                continue
            yield seed_name, group_name, group_path
