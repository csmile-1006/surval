"""Phase 2: compute local + global per-block thresholds from a state DB.

Reads a state DB built by ``build_state_db_from_cache.py`` (state embeddings =
the policy's cached ``obs_features``) and writes a ``LocalThresholdMap`` that the
``--block-scale-method local`` / ``global_db`` / ``intra_demo_sc`` consumers in
``surval.sequential_validate`` look up by ``(demo_id, index_in_demo)``.

``--scale-source`` (default from the DB's config): ``local`` writes to a
``thresholds/`` subdir; ``intra_demo_sc`` writes to ``thresholds_intra_demo_sc/``
so both can coexist on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.local_threshold.config import LocalThresholdConfig  # noqa: E402
from surval.local_threshold.database import StateDatabase  # noqa: E402
from surval.local_threshold.threshold import (  # noqa: E402
    compute_global_intra_demo_thresholds_from_db,
    compute_global_thresholds_from_db,
    compute_local_intra_demo_thresholds,
    compute_local_thresholds,
)


def _load_config(state_db_dir: str) -> LocalThresholdConfig:
    with open(os.path.join(state_db_dir, "config.json")) as f:
        d = json.load(f)
    d.pop("history_len", None)  # backward compat
    for k in ("image_views", "proprio_keys", "quantiles", "block_names"):
        if k in d:
            d[k] = tuple(d[k])
    for k in ("block_slices", "block_types"):
        if k in d:
            d[k] = tuple(tuple(x) for x in d[k])
    return LocalThresholdConfig(**d)


def _apply_overrides(cfg: LocalThresholdConfig, args) -> LocalThresholdConfig:
    overrides: dict = {}
    if args.quantiles is not None:
        overrides["quantiles"] = tuple(args.quantiles)
    if args.k_neighbors is not None:
        overrides["k_neighbors"] = int(args.k_neighbors)
    if args.temporal_exclusion_radius is not None:
        overrides["temporal_exclusion_radius"] = int(args.temporal_exclusion_radius)
    if args.same_demo_allowed is not None:
        overrides["same_demo_allowed"] = bool(args.same_demo_allowed)
    if args.min_neighbors_for_local is not None:
        overrides["min_neighbors_for_local"] = int(args.min_neighbors_for_local)
    if args.pairwise_or_query_centered is not None:
        overrides["pairwise_or_query_centered"] = args.pairwise_or_query_centered
    if args.scale_source is not None:
        overrides["scale_source"] = args.scale_source
    if args.ta is not None:
        overrides["ta"] = int(args.ta)
    return replace(cfg, **overrides) if overrides else cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-db-dir", required=True)
    ap.add_argument("--quantiles", nargs="+", type=float, default=None)
    ap.add_argument("--k-neighbors", type=int, default=None)
    ap.add_argument("--temporal-exclusion-radius", type=int, default=None)
    ap.add_argument("--same-demo-allowed", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    ap.add_argument("--min-neighbors-for-local", type=int, default=None)
    ap.add_argument("--pairwise-or-query-centered", choices=["pairwise", "query_centered"], default=None)
    ap.add_argument("--scale-source", choices=["local", "intra_demo_sc"], default=None)
    ap.add_argument("--ta", type=int, default=None, help="Chunk horizon for intra_demo_sc.")
    ap.add_argument("--output-subdir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    state_db_dir = os.path.abspath(args.state_db_dir)
    cfg = _apply_overrides(_load_config(state_db_dir), args)
    src = str(cfg.scale_source)
    subdir = args.output_subdir or ("thresholds_intra_demo_sc" if src == "intra_demo_sc" else "thresholds")
    out_dir = os.path.join(state_db_dir, subdir)
    if os.path.exists(os.path.join(out_dir, "local_threshold_map.npz")) and not args.force:
        print(f"[skip] thresholds already exist at {out_dir} (pass --force to recompute)")
        return

    print(f"[1/3] Loading state DB <- {state_db_dir}")
    db = StateDatabase.load(state_db_dir, cfg)
    print(f"      {db.n_states} states, dim={db.state_dim}; scale_source={src}, ta={cfg.ta}")

    print(f"[2/3] Computing global thresholds ({'intra-demo SC' if src == 'intra_demo_sc' else 'pooled neighbor pairs'})")
    t0 = time.time()
    global_t = (
        compute_global_intra_demo_thresholds_from_db(db, cfg)
        if src == "intra_demo_sc"
        else compute_global_thresholds_from_db(db, cfg)
    )
    print(f"      shape={global_t.shape}  elapsed={time.time() - t0:.1f}s")
    for bi, b in enumerate(cfg.block_names):
        for qi, q in enumerate(cfg.quantiles):
            print(f"      global[{b}, q={q}] = {float(global_t[bi, qi]):.6f}")

    print(f"[3/3] Computing local thresholds  (out_dir={out_dir})")
    t0 = time.time()
    m = (
        compute_local_intra_demo_thresholds(db, cfg, global_t)
        if src == "intra_demo_sc"
        else compute_local_thresholds(db, cfg, global_t)
    )
    print(f"      done in {time.time() - t0:.1f}s; fallback_rate={m.fallback_used.mean():.2%}; "
          f"mean valid neighbors/state={float(m.n_neighbors_used.mean()):.1f}")

    os.makedirs(out_dir, exist_ok=True)
    m.save(out_dir)
    print(f"      wrote thresholds to {out_dir}")


if __name__ == "__main__":
    main()
