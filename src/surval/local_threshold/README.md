# SURVAL Local (State-Conditional) Thresholds

State-conditional version of the SURVAL block thresholds. Where the global
pipeline assigns a single per-block scale `S_g` over the entire val set, this
module computes a *per-state* threshold `S_g(s)` that adapts to the local
neighborhood of the state in **feature space**.

**Feature source: the policy's own output.** State embeddings are the per-row
`obs_features` already stored in the seqcache — a VLM prefix embedding for
openpi/RLDS, or a model encoder output for robomimic — L2-normalized. The
feature the model computed once at caching time is reused as the retrieval key,
so the state DB needs no raw-image access and no extra GPU pass.

## Pipeline

```
seqcache_*.hdf5  (with obs_features)               ┌─ embeddings.npy   (N, F)
   │                                               ├─ actions.npy      (N, A)
   ▼                                               ├─ demo_id_int.npy  (N,)
build_state_db_from_cache.py                       ├─ t.npy            (N,)
   │  L2-normalize cached obs_features  ─────────► ├─ faiss.index
   ▼                                               └─ config.json
state DB
   │                                               ┌─ thresholds       (N, n_blocks, n_quantiles)
   ▼                                               ├─ fallback_used    (N,)
compute_local_thresholds.py                        ├─ n_neighbors_used (N,)
   │  k-NN + filter + per-block dist + multi-q     ├─ global_thresholds(n_blocks, n_quantiles)
   ▼
threshold artifacts (per state, per block, per quantile)
```

Both scripts live in `surval/scripts/`.

## Phase 0 — assumptions

- **Feature source**: cached `obs_features` (model VLM / encoder output),
  L2-normalized to unit length. Cosine retrieval via FAISS inner-product.
- **Action layouts** (`--block-layout`): `droid_joint` (8-D DROID, 7 joint
  blocks, L2), `droid_pos_rot6d` (10-D, `pos` L2 + `rot_6d` geodesic), plus
  `gripper` (14-D), `dex` (24-D), `humanoid` (30-D, rot6d on the rotation
  blocks). Must match the seqval consumer's `ACTION_SPACE`.
- **Distance dispatch**: blocks default to L2; `block_types[...] = "rot6d"`
  selects SO(3) geodesic distance. Extend `_block_pair_distances` /
  `_block_chunk_motion_distance` in `threshold.py` for new types.
- **Quantiles**: default `(0.5, 0.75, 0.9, 0.95, 0.99)`, all computed in one
  pass; the consumer chooses which to apply at use time.

## Install

```bash
pip install -e .[db]   # faiss-cpu for the state DB index
```

State features come from the cache (`obs_features`), so no extra dependency.

## Usage

### Phase 1 — build the state DB from a cache

```bash
python scripts/build_state_db_from_cache.py \
    --cache-file /path/to/seqcache_step_000600.hdf5 \
    --output-dir ./cache/local_threshold/run0 \
    --block-layout dex            # droid_joint | droid_pos_rot6d | gripper | dex | humanoid
```

The cache **must** contain per-row `obs_features` (the model feature output). The
build L2-normalizes them into retrieval embeddings, pulls per-state actions from
`actions[:, 0, :]`, and keys the DB by `(demo_id, index_in_demo)` to match the
consumer.

### Phase 2 — compute local + global thresholds

```bash
python scripts/compute_local_thresholds.py --state-db-dir ./cache/local_threshold/run0
```

Outputs land at `<state-db-dir>/thresholds/` (`local_threshold_map.npz` with
`thresholds`, `fallback_used`, `n_neighbors_used`, `global_thresholds`).
Threshold-only knobs can be overridden without rebuilding the DB:

```bash
python scripts/compute_local_thresholds.py --state-db-dir <dir> \
    --quantiles 0.9 0.95 --k-neighbors 100 \
    --pairwise-or-query-centered query_centered --output-subdir thresholds_qc
```

### Phase 3 — score with state-conditional thresholds

Point the seqval consumer at the DB:

```bash
python -m your_wrapper --cache-dir <caches> --output-dir <out> \
    --block-scale-method state_inter --state-db-dir ./cache/local_threshold/run0 \
    --threshold-quantile 0.9
```

(`state_intra` reads the `thresholds_intra_demo_sc/` subdir — build it with
`compute_local_thresholds.py --scale-source intra_demo_sc`.)

## Multi-quantile lookup

```python
from surval.local_threshold.threshold import LocalThresholdMap
m = LocalThresholdMap.load("./cache/local_threshold/run0/thresholds")
s_local = m.get(state_idx=42, block="joint_3", quantile=0.95)
s_global = m.get_global(block="joint_3", quantile=0.95)
```

## Files

- `config.py` — `LocalThresholdConfig` (frozen dataclass + `to_hash`)
- `state_embed.py` — feature/proprio composition + `ProprioNormalizer`
- `database.py` — `StateDatabase` (FAISS + records)
- `threshold.py` — global + local threshold computation, `LocalThresholdMap`
- `sanity.py` — Phase 1/2 sanity checks (raise on failure)
- `../../../scripts/build_state_db_from_cache.py`, `compute_local_thresholds.py`
