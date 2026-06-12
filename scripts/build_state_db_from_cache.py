"""Build a `surval.local_threshold` state DB directly from a seqcache HDF5.

State embeddings are the policy's own per-row feature output (the ``obs_features``
already stored in the cache — a VLM prefix embedding for openpi/RLDS, or a model
encoder output for robomimic), L2-normalized: the feature computed once at
caching time is reused as the retrieval key, with no raw-image access. Per-state
actions come from ``actions[:, 0, :]`` (chunk t=0). The DB is keyed by
``(demo_id_str, index_in_demo)`` to match the consumer in
``surval.sequential_validate`` (``--block-scale-method state_inter`` etc.).

Block layout is selected via ``--block-layout`` so the same script serves
DROID / gripper / dex / humanoid; it must match the seqval consumer's
ACTION_SPACE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.local_threshold.config import LocalThresholdConfig  # noqa: E402
from surval.local_threshold.database import StateDatabase, StateRecords  # noqa: E402

# (action_dim, block_slices, block_types). Gripper/finger channels are excluded
# so the threshold blocks match the seqval consumers. DROID presets are sourced
# from LocalThresholdConfig.for_droid_action_space instead (see _make_config).
LAYOUTS = {
    "droid_joint": "joint_velocity",  # 8-D DROID single arm: 7 joint blocks (L2)
    "droid_pos_rot6d": "pos_rot6d",   # 10-D DROID: pos (L2) + rot_6d (geodesic)
    "gripper": dict(action_dim=14, block_slices=(
        ("L_pos", 0, 3), ("L_rot", 3, 6), ("R_pos", 7, 10), ("R_rot", 10, 13)), block_types=()),
    "dex": dict(action_dim=24, block_slices=(
        ("L_pos", 0, 3), ("L_rot", 3, 6), ("L_finger", 6, 12),
        ("R_pos", 12, 15), ("R_rot", 15, 18), ("R_finger", 18, 24)), block_types=()),
    "humanoid": dict(action_dim=30, block_slices=(
        ("R_pos", 0, 3), ("R_rot", 3, 9), ("L_pos", 9, 12), ("L_rot", 12, 18),
        ("R_finger", 18, 24), ("L_finger", 24, 30)),
        block_types=(("R_rot", "rot6d"), ("L_rot", "rot6d"))),
}


def _demo_sort_key(s: str):
    return (0, int(s.split("_")[-1])) if s.startswith("demo_") else (1, s)


def _load_cache(path: str):
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"Cache missing 'data' group: {path}")
        ac_dim = int(f.attrs.get("ac_dim", -1))
        demo_ids, actions_t0, obs_features, index_in_demo = [], [], [], []
        for demo_name in sorted(f["data"].keys(), key=_demo_sort_key):
            grp = f["data"][demo_name]
            if "actions" not in grp or "obs_features" not in grp:
                continue
            a = grp["actions"][:]            # (n, T, A)
            obs = grp["obs_features"][:]     # (n, F) or (n, T, F)
            idx = grp["index_in_demo"][:] if "index_in_demo" in grp else np.arange(a.shape[0], dtype=np.int64)
            if obs.ndim == 3:
                obs = obs[:, 0, :]
            n = a.shape[0]
            if obs.shape[0] != n or idx.shape[0] != n:
                raise ValueError(f"Length mismatch in {demo_name}: actions={n}, obs={obs.shape[0]}, idx={idx.shape[0]}")
            actions_t0.append(a[:, 0, :].astype(np.float32))
            obs_features.append(obs.astype(np.float32))
            index_in_demo.append(idx.astype(np.int64))
            demo_ids.extend([str(demo_name)] * n)
    if not demo_ids:
        raise RuntimeError(f"No demos with obs_features found in {path}. "
                           "Rebuild the cache so it stores the model's obs_features.")
    return demo_ids, np.concatenate(index_in_demo), np.concatenate(actions_t0), np.concatenate(obs_features), ac_dim


def _make_config(block_layout: str, args) -> LocalThresholdConfig:
    common = dict(
        k_neighbors=args.k_neighbors,
        same_demo_allowed=True,
        temporal_exclusion_radius=args.temporal_exclusion_radius,
        quantiles=tuple(float(q) for q in args.quantiles),
        min_neighbors_for_local=args.min_neighbors_for_local,
        pairwise_or_query_centered="pairwise",
        scale_source="local",
        ta=args.ta,
        index_type="flat_ip",
        actions_from_cache=True,
        dataset_name="seqcache",
    )
    spec = LAYOUTS[block_layout]
    if isinstance(spec, str):  # DROID preset
        return LocalThresholdConfig.for_droid_action_space(spec, **common)
    block_slices = spec["block_slices"]
    return LocalThresholdConfig(
        action_dim=spec["action_dim"],
        block_names=tuple(name for name, _, _ in block_slices),
        block_slices=tuple(block_slices),
        block_types=tuple(spec["block_types"]),
        droid_action_space="custom",
        **common,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-file", required=True, help="One seqcache_*.hdf5 file (must contain obs_features).")
    ap.add_argument("--output-dir", required=True, help="State DB output directory.")
    ap.add_argument("--block-layout", default="droid_joint", choices=sorted(LAYOUTS.keys()),
                    help="Action-block layout; must match the seqval consumer's ACTION_SPACE.")
    ap.add_argument("--k-neighbors", type=int, default=50)
    ap.add_argument("--min-neighbors-for-local", type=int, default=10)
    ap.add_argument("--temporal-exclusion-radius", type=int, default=5)
    ap.add_argument("--ta", type=int, default=8, help="Chunk horizon (used by intra_demo_sc).")
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.5, 0.75, 0.9, 0.95, 0.99])
    args = ap.parse_args()

    cfg = _make_config(args.block_layout, args)
    demo_ids, idx_in_demo, actions_t0, obs_features, ac_dim = _load_cache(args.cache_file)
    if ac_dim != cfg.action_dim:
        raise ValueError(f"Cache ac_dim={ac_dim} != layout action_dim={cfg.action_dim}")

    unique = sorted(set(demo_ids), key=_demo_sort_key)
    str_to_int = {s: i for i, s in enumerate(unique)}
    demo_id_int = np.array([str_to_int[s] for s in demo_ids], dtype=np.int32)

    # L2-normalize the model features -> unit-norm retrieval embeddings.
    norms = np.maximum(np.linalg.norm(obs_features, axis=-1, keepdims=True), 1e-8)
    embeddings = (obs_features / norms).astype(np.float32)

    records = StateRecords(
        actions=actions_t0, demo_id_int=demo_id_int,
        t=idx_in_demo.astype(np.int32), demo_id_str_by_int={i: s for s, i in str_to_int.items()},
    )
    db = StateDatabase(cfg)
    db.attach(embeddings, records)
    db.save(args.output_dir)

    with open(os.path.join(args.output_dir, "state_db.complete"), "w") as f:
        f.write("ok\n")
    summary = dict(
        cache_file=os.path.abspath(args.cache_file), feature_source="cache_obs_features",
        n_states=int(embeddings.shape[0]), state_dim=int(embeddings.shape[1]),
        n_demos=len(unique), action_dim=cfg.action_dim, block_names=list(cfg.block_names),
        config_hash=cfg.to_hash(),
    )
    with open(os.path.join(args.output_dir, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
