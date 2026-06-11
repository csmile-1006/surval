"""
Extract per-checkpoint scalars from seqcache HDF5 files.

For each `<seqcache_root>/<seed_name>/<cache_group_dir>/seqcache_epoch_<n>.hdf5`,
this produces three (epoch -> value) series:

  Cache/Valid/Loss               : metrics/valid/Loss attribute (scalar)
  Cache/Valid/Off_Manifold_Norm  : metrics/valid/off_manifold_norm/sample_0 if
                                   present (new schema), else the flat
                                   metrics/valid/Off_Manifold_Norm scalar
                                   (legacy robomimic schema)
  Cache/Valid/MSE                : mean((actions - pred_actions)**2) across
                                   all demos / time / action dims, averaged
                                   over cache samples (axis 0 of pred_actions)

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
# Five MSE variants (lower-is-better):
#   MSE              : E_{s,n,t,a}[(pred - gt)^2]    (current, variance-included)
#   MSE_mean_pred    : E_{n,t,a}[(mean_s(pred) - gt)^2]  (Bayes-optimal point-est)
#   MSE_t0_only      : (mean_s(pred)[t=0] - gt[t=0])^2 over (n, a) — one-step
#   MSE_tlast        : same but t=-1 — last horizon step
#   MSE_per_dim_norm : ((mean_s(pred) - gt) / std_a)^2 averaged — scale-normalized
MSE_TAG  = "Cache/Valid/MSE"
MSE_MEAN_PRED_TAG = "Cache/Valid/MSE_mean_pred"
MSE_T0_TAG        = "Cache/Valid/MSE_t0_only"
MSE_TLAST_TAG     = "Cache/Valid/MSE_tlast"
MSE_PER_DIM_TAG   = "Cache/Valid/MSE_per_dim_norm"
MSE_VARIANT_TAGS = [
    MSE_TAG, MSE_MEAN_PRED_TAG, MSE_T0_TAG, MSE_TLAST_TAG, MSE_PER_DIM_TAG,
]

_EPOCH_RE = re.compile(r"epoch_(\d+)")


def _list_epoch_files(group_path: str) -> list:
    """Return seqcache_epoch_*.hdf5 files sorted by epoch number."""
    files = glob.glob(os.path.join(group_path, "seqcache_epoch_*.hdf5"))
    def _key(p):
        m = _EPOCH_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else -1
    return sorted(files, key=_key)


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


def _gather_action_std(group_path: str, n_demos_cap: int = 400):
    """
    Estimate per-dim std of the GT actions for this (seed, dataset) group.
    Used by the per-dim-normalized MSE variant. GT actions are the same across
    epoch checkpoints, so we only scan one file.
    """
    try:
        import h5py
    except ImportError:
        return None
    files = _list_epoch_files(group_path)
    if not files:
        return None
    try:
        with h5py.File(files[0], "r") as h:
            if "data" not in h:
                return None
            sums = None; sqs = None; cnt = 0
            for dk in list(h["data"].keys())[:n_demos_cap]:
                if "actions" not in h["data"][dk]:
                    continue
                a = h["data"][dk]["actions"][...].astype(np.float64, copy=False)
                flat = a.reshape(-1, a.shape[-1])
                if sums is None:
                    sums = flat.sum(axis=0)
                    sqs  = (flat * flat).sum(axis=0)
                else:
                    sums += flat.sum(axis=0)
                    sqs  += (flat * flat).sum(axis=0)
                cnt += flat.shape[0]
            if cnt == 0:
                return None
            mean = sums / cnt
            var  = sqs / cnt - mean * mean
            return np.sqrt(np.maximum(var, 1e-8))
    except OSError:
        return None


def _read_mse_variants(h, action_std) -> dict:
    """
    Five lower-is-better MSE variants computed from one epoch's HDF5 cache.
    Returns dict keyed by MSE_*_TAG names; NaN-on-failure per key.
    """
    keys = [MSE_TAG, MSE_MEAN_PRED_TAG, MSE_T0_TAG, MSE_TLAST_TAG, MSE_PER_DIM_TAG]
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
        # Variant 1: current — E_{s,n,t,a}[(pred - gt)^2]
        d = pr_s - gt_s[None, :, :, :]
        accum[MSE_TAG][0] += float((d * d).sum()); accum[MSE_TAG][1] += int(d.size)
        # Sample-mean prediction (used by variants 2/3/4/5)
        pm = pr_s.mean(axis=0)                  # [n, t, a]
        # Variant 2: mean_pred — Bayes-optimal point-estimate accuracy
        d = pm - gt_s
        accum[MSE_MEAN_PRED_TAG][0] += float((d * d).sum())
        accum[MSE_MEAN_PRED_TAG][1] += int(d.size)
        # Variant 3: t0_only
        d0 = pm[:, 0, :] - gt_s[:, 0, :]
        accum[MSE_T0_TAG][0] += float((d0 * d0).sum())
        accum[MSE_T0_TAG][1] += int(d0.size)
        # Variant 4: tlast
        dl = pm[:, -1, :] - gt_s[:, -1, :]
        accum[MSE_TLAST_TAG][0] += float((dl * dl).sum())
        accum[MSE_TLAST_TAG][1] += int(dl.size)
        # Variant 5: per-dim normalized
        if action_std is not None and len(action_std) >= a:
            d = (pm - gt_s) / action_std[:a][None, None, :]
            accum[MSE_PER_DIM_TAG][0] += float((d * d).sum())
            accum[MSE_PER_DIM_TAG][1] += int(d.size)
    out = {}
    for k in keys:
        s, n = accum[k]
        out[k] = (s / n) if n > 0 else float("nan")
    return out


def _read_mse(h) -> float:
    """Back-compat single-variant reader (current MSE)."""
    return _read_mse_variants(h, None)[MSE_TAG]


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
    m = _EPOCH_RE.search(os.path.basename(fallback_path))
    return int(m.group(1)) if m else -1


def extract_seqcache_metrics_for_group(group_path: str) -> dict:
    """
    Walk one cache_group_dir and return a JSON-cache-shaped payload with the
    three Cache/Valid/* tags as (steps, values) series sorted by epoch.

    Returns a dict suitable for surval.tb_aggregate.tb_io.save_cache_file,
    including a "kind" marker so consumers can tell which extractor produced
    it.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required to read seqcache HDF5 files.") from exc

    files = _list_epoch_files(group_path)
    action_std = _gather_action_std(group_path)
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
                mse_vals = _read_mse_variants(h, action_std)
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
