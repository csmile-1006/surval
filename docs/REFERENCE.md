# surval

Shared library for sequential-validation-from-cache metrics, state-conditional
threshold tooling, and the canonical HDF5 cache format that ties the two together.

Consumed by:

- `surval_openpi` — DROID/openpi sequential-validation pipeline
- `custom-robomimic` — gripper / dex / humanoid sequential-validation wrappers

## Install

The library is published from this GitHub repo and pinned in each downstream
project by commit SHA so behavior changes always require an explicit bump.

**uv project (e.g. `surval_openpi`):**

```toml
# pyproject.toml
[project]
dependencies = ["surval", ...]

[tool.uv.sources]
surval = { git = "https://github.com/csmile-1006/surval.git", rev = "<sha>" }
```

Then `uv sync` from the project root.

**conda / pip env (e.g. the `dmg` env used by `custom-robomimic`):**

```bash
pip install --upgrade --force-reinstall --no-deps \
  "surval @ git+https://github.com/csmile-1006/surval.git@<sha>"
```

`--no-deps` avoids re-resolving heavyweight transitive deps (torch, sklearn)
that the host env already manages.

**Local editable (for surval development only):**

```bash
pip install -e /home/changyeon/workspace/surval[full]
```

### Extras

- `encoder` — pulls in `torch`; required by `surval.local_threshold.encoder.FrozenImageEncoder`
- `db` — pulls in `faiss-cpu`; used lazily inside `surval.local_threshold.database`
- `sanity` — pulls in `matplotlib`; used by `surval.local_threshold.sanity` plot helpers
- `omn` — pulls in `scikit-learn`; required by `surval.cache_io.compute_off_manifold_norm`
- `full` — all of the above
- `dev` — `pytest` for the test suite

## Modules

| Module | Purpose |
|---|---|
| `surval.cache_io` | Canonical HDF5 cache writer + helpers shared by every cache producer |
| `surval.sequential_validate` | Reads caches, computes prefix-survival / per-block scaled error metrics |
| `surval.local_threshold` | State-conditional threshold maps (`local`, `global_db`, `intra_demo_sc` modes) |
| `surval.scale_estimation`, `surval.surval`, `surval.gates` | Regime-conditioned scale lookup and gate functions |

---

## `surval.cache_io` — canonical HDF5 cache I/O

The cache producers in `surval_openpi` (JAX/openpi) and `custom-robomimic`
(PyTorch/robomimic) used to maintain their own near-duplicate writers, with
subtle schema drift that left robomimic-produced caches partially invisible
to the surval reader. `cache_io` centralizes the writer and the I/O helpers
both producers need so the schema lives in one place.

### Canonical HDF5 schema

```
File attrs:
  checkpoint        : str    (path or step descriptor)
  step              : int    (training step, or epoch for epoch-based frameworks)
  num_cache_samples : int
  num_rows          : int
  num_steps         : int    (T, the action horizon)
  ac_dim            : int
  val_loss          : float  (NaN if not measured)
  has_obs_features  : int    (0/1)
  obs_feat_dim      : int    (0 if absent)

data/                            (group; attr "total" = num_rows)
  demo_<i>/                      (i = enumerated index 0..D-1, sorted)
    attrs:
      demo_id     : str          (original ID from the dataset)
      num_samples : int
    datasets:
      index_in_demo : int64   [N_demo]
      actions       : float32 [N_demo, T, A]
      pred_actions  : float32 [S, N_demo, T, A]
      obs_features  : float32 [N_demo, F] or [N_demo, T, F]   (optional)

metrics/valid/
  Loss : float64                                (scalar; NaN if not measured)
  off_manifold_norm/                            (group, optional)
    sample_<k> : float64                        (one per cache sample)
```

### Filename convention

Cache files **must** be named `seqcache_step_{step:06d}.hdf5` so they are
auto-discovered by `surval.sequential_validate._find_caches_in_dir`. For
frameworks that count by epoch (robomimic), pass `step=epoch` to the writer
and use the same naming pattern — the reader doesn't care what the integer
means semantically.

### Public API

| Symbol | Purpose |
|---|---|
| `write_seqcache_hdf5(out_path, *, demo_ids, index_in_demo, actions, pred_actions_list, checkpoint, step, val_loss=None, off_manifold_norms=None, obs_features=None)` | Writes the canonical schema. `pred_actions_list` is a sequence of `[N, T, A]` arrays (one per cache sample). `off_manifold_norms` accepts `dict[int, float]`, a single scalar (treated as `{0: scalar}`), or `None`. |
| `sort_demo_keys(keys)` | Sort `demo_<n>` / `row_<n>` keys numerically; lexicographic fallback. |
| `concat_trim_time(arr_list)` | Concatenate `[B, T_i, A_i]` arrays along `B` after trimming to common min `T`/`A`. |
| `concat_trim_time_feature(arr_list)` | Same but for `[B, T_i, F_i]` feature arrays. |
| `group_rows_by_demo(demo_ids, index_in_demo)` | `{demo_id: row_indices_sorted_by_index_in_demo}`. |
| `ReservoirSampler(capacity, *, slots, rng_seed)` | Bounded uniform sampling for parallel arrays. |
| `compute_off_manifold_norm(pred, state_feats, expert, k=5)` | Mean projection error onto k-NN expert action span. Requires `scikit-learn` (the `omn` extra). |
| `compute_off_manifold_errors(...)` | Same metric, returned per-sample (no mean). |

### Producer-side example

A minimal cache-producer loop using the unified writer:

```python
import numpy as np
from surval.cache_io import (
    ReservoirSampler,
    compute_off_manifold_norm,
    write_seqcache_hdf5,
)

# Collect predictions across one validation pass.
gt_chunks: list[np.ndarray] = []           # [B, T, A] per batch
pred_chunks: list[list[np.ndarray]] = [[] for _ in range(num_cache_samples)]
demo_ids_all: list[str] = []
index_in_demo_all: list[int] = []
omn_sampler = ReservoirSampler(
    capacity=65536,
    slots=("feat", "gt", "pred"),
    rng_seed=seed,
)

for batch in valid_loader:
    pred_samples, gt, demo_ids, idx_in_demo, state_feats = run_inference(batch)
    gt_chunks.append(gt)
    for s in range(num_cache_samples):
        pred_chunks[s].append(pred_samples[s])
    demo_ids_all.extend(demo_ids)
    index_in_demo_all.extend(idx_in_demo)
    for i in range(gt.shape[0]):
        omn_sampler.observe(feat=state_feats[i], gt=gt[i], pred=pred_samples[0][i])

actions = np.concatenate(gt_chunks, axis=0)
pred_actions_list = [np.concatenate(p, axis=0) for p in pred_chunks]

# Off-manifold norm on a uniform random subset of the dataset.
omn_data = omn_sampler.collect()
off_manifold_norms = {
    0: compute_off_manifold_norm(
        np.stack(omn_data["pred"]).reshape(len(omn_data["pred"]), -1),
        np.stack(omn_data["feat"]),
        np.stack(omn_data["gt"]).reshape(len(omn_data["gt"]), -1),
        k=5,
    )
}

write_seqcache_hdf5(
    f"/path/to/cache/seqcache_step_{step:06d}.hdf5",
    demo_ids=np.array(demo_ids_all),
    index_in_demo=np.array(index_in_demo_all, dtype=np.int64),
    actions=actions,
    pred_actions_list=pred_actions_list,
    checkpoint="/path/to/checkpoint",
    step=step,
    val_loss=val_loss,
    off_manifold_norms=off_manifold_norms,
)
```

### Reader-side example

The cache is consumed by `surval.sequential_validate`:

```python
from surval.sequential_validate import _load_cache_hdf5

(
    demo_ids,         # [N] str
    index_in_demo,    # [N] int64
    actions,          # [N, T, A] float32
    pred_samples,     # [S, N, T, A] float32 (always 4D — single-sample auto-wrapped)
    obs_features,     # [N, F] float32 or None
    checkpoint,       # str
    step,             # int (-1 if attr missing)
    valid_loss,       # float (NaN if not measured)
    valid_off,        # float or None (only sample_0 exposed by the reader)
) = _load_cache_hdf5("/path/to/seqcache_step_000042.hdf5")
```

For full sequential-validation metrics, use the wrapper-script entry-point
trio described in the next section.

### Migration notes

If you still have caches from the pre-unification era, two formats need
patching to be readable by the surval reader:

```bash
# 1. Robomimic used to name caches seqcache_epoch_*.hdf5; rename to step:
for f in seqcache_epoch_*.hdf5; do
    mv "$f" "${f/epoch_/step_}"
done

# 2. Robomimic used to write a flat metrics/valid/Off_Manifold_Norm scalar.
# The reader reads only metrics/valid/off_manifold_norm/sample_<k>. Any
# OMN values stored in the old format are silently dropped — regenerate
# the cache to recover them.
```

---

## `surval.sequential_validate` — metrics from a cache directory

Per-action-space wrappers (in `surval_openpi/scripts/` and
`custom-robomimic/robomimic/scripts/`) compose this module's entry-point trio
to turn a directory of `seqcache_step_*.hdf5` files into JSON summaries and
TensorBoard scalars:

```python
import argparse
from surval.sequential_validate import add_common_args, run, validate_common_args

ACTION_SPACE = {
    "action_dim": 14,
    "block_names": [...],
    "block_slices": {...},
    "block_dims": {...},
    "arm_pairs": [...],
    "scale_groups": [...],
    "summary_scale_fields": [...],
}

def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    args = parser.parse_args()
    validate_common_args(args, num_blocks=len(ACTION_SPACE["block_names"]))
    run(args, ACTION_SPACE)
```

`add_common_args` exposes `--cache-dir`, `--output-dir`, `--block-scale-method`
(including `local`, `global_db`, `intra_demo_sc`), `--state-db-dir`,
`--threshold-quantile`, plus the prefix-survival hyperparameters.

---

## `surval.local_threshold` — state-conditional threshold maps

```python
from surval.local_threshold.threshold import LocalThresholdMap
from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.database import StateDatabase, StateRecords
from surval.local_threshold.encoder import FrozenImageEncoder  # needs the `encoder` extra
```

These back the `local` / `global_db` / `intra_demo_sc` modes of the
`--block-scale-method` flag. See `src/surval/local_threshold/README.md` for
build / query workflow.

---

## Tests

```bash
cd /home/changyeon/workspace/surval
pip install -e .[full,dev]
pytest tests/ -v
```

`tests/cache_io_test.py` covers the writer ↔ reader round-trip and every
helper. The off-manifold tests are skipped when `scikit-learn` is not
installed; install the `omn` extra (or the `full` extra) to run them.
