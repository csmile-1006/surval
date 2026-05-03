# SURVAL Local (State-Conditional) Thresholds

State-conditional version of the SURVAL block thresholds. Where the existing
global pipeline assigns a single per-block scale `S_g` over the entire val set,
this module computes a *per-state* threshold `S_g(s)` that adapts to the local
neighborhood of the state in image+proprio space.

## Pipeline

```
RLDS val demos                                     ┌─ embeddings.npy   (N, D_state)
   │                                               ├─ actions.npy      (N, 8)
   ▼                                               ├─ demo_id_int.npy  (N,)
build_state_db.py                                  ├─ t.npy            (N,)
   │  DINOv2 (frozen) + ProprioNormalizer  ──────► ├─ faiss.index
   │                                               ├─ proprio_normalizer.pkl
   ▼                                               └─ config.json
state DB
   │                                               ┌─ thresholds       (N, n_blocks, n_quantiles)
   ▼                                               ├─ fallback_used    (N,)
compute_local_thresholds.py                        ├─ n_neighbors_used (N,)
   │  k-NN + filter + per-block L2 + multi-quantile├─ global_thresholds(n_blocks, n_quantiles)
   │                                               └─ meta.json
   ▼
threshold artifacts (per state, per block, per quantile)
```

## Phase 0 — assumptions baked into this implementation

- **Action**: DROID 8-D `[joint_velocity (7), gripper (1)]`. Block grouping
  matches `DROID_ACTION_SPACE` in
  `scripts/sequential_validate_from_cache_droid.py`: 7 single-dim joint blocks
  (`joint_0..joint_6`), gripper excluded from blocks.
- **Distance**: per-block L2 only (no rotation, no Mahalanobis).
- **Image views**: `image` (exterior_image_1_left) + `wrist_image`
  (wrist_image_left), 2 views.
- **Proprio**: `joint_position` (7) + `cartesian_position` (6) +
  `gripper_position` (1) = 14-D, z-scored.
- **Encoder**: DINOv2 (default `dinov2_vitb14`), frozen, CLS pooled.
- **Subsampling**: none — uses every `passes_filter` step in the val split.
  The DROID val split is expected to be ≤100,000 steps after upstream filtering.
- **Quantiles**: stored as a tuple, default `(0.5, 0.75, 0.9, 0.95, 0.99)`.
  All are computed in one pass; the consumer chooses which to apply.

## Install

```bash
uv sync --extra rlds --extra local_threshold
# or pip install: faiss-cpu>=1.8.0
```

DINOv2 is loaded via `torch.hub` from `facebookresearch/dinov2`. The first run
downloads weights to `~/.cache/torch/hub/`.

## Usage

### Phase 1 — build the state DB

```bash
python scripts/build_state_db.py \
    --data-dir /path/to/tfds_root \
    --dataset-name droid \
    --max-samples 5000 \
    --cache-root ./cache/local_threshold
```

Outputs go to `<cache-root>/<config_hash>/`. The hash is computed from every
field of `LocalThresholdConfig`, so any change to encoder / image_size /
proprio keys / quantile candidates produces a fresh artifact dir. Re-running
with the same config skips work via the `state_db.complete` sentinel.

Sanity report: `<config_hash>/sanity_phase1.md`. Includes encoder determinism,
modality balance, FAISS round-trip, and visual neighbor figures.

### Phase 2 — compute local + global thresholds

```bash
python scripts/compute_local_thresholds.py \
    --state-db-dir ./cache/local_threshold/<config_hash>/
```

Outputs land at `<state-db-dir>/thresholds/`:

- `local_threshold_map.npz` with arrays:
  - `thresholds`: `(N, n_blocks, n_quantiles)` float32
  - `fallback_used`: `(N,)` bool
  - `n_neighbors_used`: `(N,)` int32
  - `global_thresholds`: `(n_blocks, n_quantiles)` float32
- `local_threshold_meta.json` — `block_names`, `quantiles`, `config_hash`
- `global_thresholds.npz` — same global array, also exposed as a standalone file
- `sanity_phase2.md` + `figures/`

You can override threshold-pass-only knobs without rebuilding the DB:

```bash
python scripts/compute_local_thresholds.py \
    --state-db-dir <dir> \
    --quantiles 0.9 0.95 \
    --k-neighbors 100 \
    --pairwise-or-query-centered query_centered \
    --output-subdir thresholds_qc
```

## Multi-quantile lookup

```python
from surval.local_threshold.threshold import LocalThresholdMap
m = LocalThresholdMap.load("./cache/local_threshold/<hash>/thresholds")

s_local = m.get(state_idx=42, block="joint_3", quantile=0.95)
s_global = m.get_global(block="joint_3", quantile=0.95)
```

## Files

- `config.py` — `LocalThresholdConfig` (frozen dataclass + `to_hash`)
- `encoder.py` — `FrozenImageEncoder` (DINOv2 wrapper)
- `state_embed.py` — composition + `ProprioNormalizer`
- `database.py` — `StateDatabase` (FAISS + records)
- `threshold.py` — global + local threshold computation, `LocalThresholdMap`
- `sanity.py` — Phase 1 + Phase 2 sanity checks (raise on failure)
- `*_test.py` — pytest tests (encoder test marked `@pytest.mark.gpu`)

## Out of scope (per spec §7)

- Phase 3 (policy scoring with local thresholds) — separate task.
- Phase 4 (closed-loop validation).
- Modifications to global threshold computation in
  `sequential_validate_from_cache_base.py` (left untouched).
- 14-D / 24-D action spaces (DROID 8-D only).
