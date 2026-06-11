# surval

**Sequential validation from cached policy rollouts** — a shared library for
ranking training checkpoints by how well a policy's predicted actions survive
against the ground-truth trajectory, *without running the environment*. It also
ships the state-conditional thresholding and the canonical HDF5 cache format
that make those scores comparable across frameworks.

New here? Read this page top-to-bottom. Deep API reference lives in
[`docs/REFERENCE.md`](docs/REFERENCE.md) and
[`src/surval/local_threshold/README.md`](src/surval/local_threshold/README.md).

---

## 1. What surval is for

You train a policy and, at each checkpoint, dump the policy's predicted actions
next to the ground-truth actions into an **HDF5 cache**. surval reads that cache
and computes a **PrefixSurvival score** — the headline metric — that tells you
which checkpoint / hyper-parameter is best, with no simulator in the loop.

surval is **only a library**. It never trains a policy and never talks to a
simulator. Two downstream projects produce caches and call into surval:

| Consumer | Stack | What it does |
|---|---|---|
| **`surval_openpi`** | JAX / openpi | DROID two-arm sequential-validation pipeline |
| **`custom-robomimic`** | PyTorch / robomimic | gripper / dex / humanoid wrappers |

The two consumers use surval **slightly differently** (install path + cache time
convention). Section 4 is the part you must not skip.

The **lean core** of surval is the PrefixSurvival metric. A handful of older
baseline proxies (validation loss, off-manifold norm, action MSE) are kept
around but are treated as **legacy** and documented separately in §10 — they are
*not* required for the main path.

---

## 2. The pieces you will touch

```
producer  ──►  HDF5 cache  ──►  surval reader  ──►  per-ckpt scores  ──►  multi-seed
(your repo)    seqcache_*.hdf5   sequential_validate    JSON / TB           aggregate
                                                                            vs. success rate
```

1. **Produce** — in *your* training repo, after a validation pass, call
   `surval.cache_io.write_seqcache_hdf5(...)` to write one cache file per
   checkpoint (§6).
2. **Score** — point `surval.sequential_validate` at a cache directory to get a
   PrefixSurvival score per checkpoint (§7).
3. **Aggregate** — the `scripts/` in this repo collapse many seeds / HPs / tasks
   into summary tables and **correlate the score against real success rate**
   (§8–9).

---

## 3. Install

surval is published from this GitHub repo and **pinned by commit SHA** in each
downstream project, so a behavior change always requires an explicit bump.

### `surval_openpi` (uv project)

```toml
# surval_openpi/pyproject.toml
[project]
dependencies = ["surval", ...]

[tool.uv.sources]
surval = { git = "https://github.com/csmile-1006/surval.git", rev = "<sha>" }
```

```bash
uv sync   # from the surval_openpi root
```

### `custom-robomimic` (conda / pip env, e.g. the `dmg` env)

```bash
pip install --upgrade --force-reinstall --no-deps \
  "surval @ git+https://github.com/csmile-1006/surval.git@<sha>"
```

`--no-deps` keeps pip from re-resolving heavyweight transitive deps (torch,
sklearn) the conda env already owns.

### Developing surval itself

```bash
pip install -e /home/changyeon/workspace/surval[full,dev]
pytest tests/ -v
```

**Optional extras:** `encoder` (torch — image encoder), `db` (faiss — state
database), `omn` (scikit-learn — *legacy* off-manifold norm), `sanity`
(matplotlib), `full` (all), `dev` (pytest).

---

## 4. surval_openpi vs. robomimic — the differences that bite

surval supports a **step-indexed** and an **epoch-indexed** cache convention.
openpi counts training *steps*; robomimic counts *epochs*. Get this wrong and
the reader silently finds **zero** cache files.

| | `surval_openpi` (openpi) | `custom-robomimic` (robomimic) |
|---|---|---|
| Install | uv git-source pin → `uv sync` | `pip install --no-deps git+...@<sha>` |
| Cache filename | `seqcache_step_{step:06d}.hdf5` | `seqcache_epoch_{epoch:06d}.hdf5` |
| Time attr in HDF5 | `f.attrs["step"]` | `f.attrs["epoch"]` |
| Reader flag | `--cache-mode step` (**default**) | `--cache-mode epoch` |
| Wrapper scripts live in | `surval_openpi/scripts/` | `custom-robomimic/robomimic/scripts/` |
| Action space config | DROID two-arm | gripper / dex / humanoid |

Two practical rules:

- **Producer side:** the integer you write is a step *or* an epoch — the reader
  only requires that the **filename, the time attr, and `--cache-mode` agree**.
- **Reader side:** robomimic callers pass `--cache-mode epoch` (or set
  `parser.set_defaults(cache_mode="epoch")` in their wrapper). openpi relies on
  the `step` default.

---

## 5. Cache file schema (what the HDF5 must contain)

Every cache file follows **one canonical HDF5 layout**. The reader is strict —
missing a required group/dataset raises. Below, `A`, `T`, `S`, `N`, `F` are
**variables** that change per run (action space, horizon, sampling, dataset
size, feature dim) — never hard-code them.

| Symbol | Meaning | Stored as |
|---|---|---|
| `A` | action dimension | `attrs["ac_dim"]` |
| `T` | action horizon (timesteps per row) | `attrs["num_steps"]` |
| `S` | number of prediction samples | `attrs["num_cache_samples"]` |
| `N` | total rows across all demos | `attrs["num_rows"]` |
| `F` | obs-feature dimension | `attrs["obs_feat_dim"]` |

### Required layout (lean core — what the surval metric needs)

```
/                                          (file root)
├── attrs
│   ├── checkpoint        : str    checkpoint path / descriptor
│   ├── step  OR  epoch    : int    time-axis index (§4)
│   ├── ac_dim            : int    = A
│   ├── num_steps         : int    = T
│   ├── num_cache_samples : int    = S
│   ├── num_rows          : int    = N
│   ├── has_obs_features  : int    = 1   (REQUIRED — always present)
│   └── obs_feat_dim      : int    = F
│
└── data/                              attrs: total = N
    ├── demo_0/                        attrs: num_samples = N_0, demo_id (opt)
    │   ├── actions       float32  [N_0, T, A]      ground-truth actions   (REQUIRED)
    │   ├── pred_actions  float32  [S, N_0, T, A]   predicted, S samples    (REQUIRED)
    │   ├── index_in_demo int64    [N_0]            row order within demo   (REQUIRED)
    │   └── obs_features  float32  [N_0, F]         state features          (REQUIRED)
    ├── demo_1/ ...
    └── demo_<D-1>/
```

**`obs_features` is required.** It powers the state-conditional threshold /
scale-estimation modes (`local`, `global_db`, `intra_demo_sc`) that surval's
method relies on (`[N_i, F]`, or `[N_i, T, F]` which the reader collapses to the
first timestep). Every current cache writes it (`has_obs_features = 1`).

### Hard rules the reader enforces

- **Dims must be exact:** `actions` is 3-D `[N_i, T, A]`; `pred_actions` is 4-D
  `[S, N_i, T, A]` (a 3-D `[N_i, T, A]` is auto-wrapped to `S = 1`). Otherwise
  `ValueError`.
- **Lengths must agree per demo:**
  `index_in_demo.shape[0] == actions.shape[0] == pred_actions.shape[1] == obs_features.shape[0]`.
- **Demo keys** sort numerically (`demo_0, demo_1, ...`); the `demo_id` attr, if
  present, overrides the displayed id.
- **Filename ↔ attr ↔ flag must agree** (§4). Mismatch ⇒ 0 files found, or the
  time index reads back as `-1`.

> Legacy proxy blocks (`metrics/valid/...`) are **not** part of the lean schema;
> see §10.

---

## 6. Producing a cache

In your training repo, after one validation pass:

```python
import numpy as np
from surval.cache_io import write_seqcache_hdf5

# `step` (openpi) or `epoch` (robomimic) — must match filename + --cache-mode (§4)
write_seqcache_hdf5(
    f"/path/to/cache/seqcache_step_{step:06d}.hdf5",   # ..._epoch_... for robomimic
    demo_ids=np.array(demo_ids_all),           # [N] str, one per row
    index_in_demo=np.array(idx_all, np.int64), # [N]
    actions=actions,                           # [N, T, A] ground truth
    pred_actions_list=pred_actions_list,       # list of S arrays, each [N, T, A]
    obs_features=obs_features,                 # [N, F] REQUIRED
    checkpoint="/path/to/checkpoint",
    step=step,
)
```

`write_seqcache_hdf5` regroups the flat `[N, ...]` arrays into `data/demo_<i>/`
by `demo_ids` and writes the canonical schema. A complete producer loop is in
[`docs/REFERENCE.md`](docs/REFERENCE.md#producer-side-example).

> Older callers also passed `val_loss=` / `off_manifold_norms=`; those feed the
> **legacy** proxy block (§10) and are optional for the surval metric.

---

## 7. Scoring a cache → the surval metric

The reader is exposed through an entry-point trio your wrapper composes
(`surval.sequential_validate`). The wrappers themselves live in the consumer
repos (`surval_openpi/scripts/`, `custom-robomimic/robomimic/scripts/`); a
minimal one is just:

```python
import argparse
from surval.sequential_validate import add_common_args, run, validate_common_args

ACTION_SPACE = {                 # define once per action space (DROID, gripper, dex, ...)
    "action_dim": 14,
    "block_names": [...],        # e.g. ["pos", "rot_6d", "grip"] per arm
    "block_slices": {...},
    "block_dims": {...},
    "arm_pairs": [...],
    "scale_groups": [...],
    "summary_scale_fields": [...],
}

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # robomimic wrappers add: parser.set_defaults(cache_mode="epoch")
    args = parser.parse_args()
    validate_common_args(args, num_blocks=len(ACTION_SPACE["block_names"]))
    run(args, ACTION_SPACE)

if __name__ == "__main__":
    main()
```

Run it against a directory of caches:

```bash
# openpi
python -m your_wrapper --cache-dir /path/to/caches --output-dir /path/to/out

# robomimic
python -m your_wrapper --cache-dir /path/to/caches --output-dir /path/to/out \
    --cache-mode epoch
```

**Output.** `run()` discovers every `seqcache_*.hdf5` under `--cache-dir`,
scores each, and writes one JSON file:

```
<output-dir>/sequential_validation_from_cache_group_summary.json
```

Each entry is a per-cache-group summary. The headline field is
**`PrefixSurvival_Score`** (higher = better; the policy's predicted action
prefixes stay inside the per-block scaled-error thresholds longer). Companion
fields include `PrefixSurvival_Loss = 1 - score`,
`PrefixSurvival_MatchedPrefixLen_MeanPerTraj`, and per-step action errors.

Key flags from `add_common_args`: `--cache-dir`, `--output-dir`,
`--cache-mode {step,epoch}`,
`--block-scale-method {raw,local,global_db,intra_demo_sc}`,
`--prefix-mode {soft,hard}`, `--threshold-quantile`, plus the ablation knobs
`--scale-groups-mode {grouped,flat}` and `--prefix-time-reduction {product,mean}`
(`tests/ablation_test.py`).

The consumer wrappers additionally log the score to TensorBoard as
`SequentialValid/PrefixSurvival_Score` across checkpoints — that scalar series is
what the aggregation pipeline below consumes.

---

## 8. Aggregating across seeds (e.g. robomimic, 10 seeds)

A single seed gives one score curve over checkpoints. To get a stable estimate
you aggregate across seeds. The `scripts/` here read **TensorBoard scalars**
(from both the training run and the seqval run), cache them to JSON once, then
compute per-seed metrics and bootstrap a confidence interval across seeds.

Expected layout (one subdir per seed, on both sides):

```
<train_root_dir>/<seed>/...            # training TB: success rate, val loss, ...
<eval_root_dir>/<seed>/<hp_dir>/<cache_group_dir>/...   # seqval TB: PrefixSurvival_Score
```

### Step 1 — extract scalars once into a JSON cache

```bash
python scripts/extract_tb_scalars.py \
    --train_root_dir <train_root_dir> \
    --eval_root_dir  <eval_root_dir> \
    --cache_dir      <cache_dir>
```

This walks every seed's TB event files and writes
`<cache_dir>/train/<seed>.json` and `<cache_dir>/eval/<seed>/<hp>/<group>.json`,
so later metric sweeps never re-read the (slow) event files.

### Step 2 — aggregate across seeds

```bash
python scripts/aggregate_seqval_hparams.py \
    --train_root_dir <train_root_dir> \
    --eval_root_dir  <eval_root_dir> \
    --cache_dir      <cache_dir> \
    --train_tag      "Valid/Loss"  --train_tag_is_rate false \
    --seqval_tag     "SequentialValid/PrefixSurvival_Score" \
    --out_json out/agg.json --out_csv out/agg.csv
```

For each `(HP, seqval_tag)` it computes per-seed metrics, averages over the
**common seeds** present in both roots, and bootstraps a 95% CI
(`--num_boot`, `--ci_level`). `bootstrap_ci_of_mean` and
`leave_one_seed_out_stats` (`--report_loo`) live in
`surval.tb_aggregate.metrics`.

For a per-task rollup over many tasks/datasets, the higher-level drivers wrap
this: `run_our_method_per_task.py` → `summarize_our_method.py` /
`report_task_dataset.py`, and `summarize_ablation.py` for the leave-one-out
ablations. Each has `--help`.

---

## 9. Correlating the score against **real success rate**

The point of the PrefixSurvival score is to *predict* real-world success without
rollouts. To measure how good a predictor it is, feed the **actual success
rate** as the ground-truth signal `A` and the seqval score as the proxy `B`; the
metrics quantify how well `B` ranks checkpoints the way `A` does.

The success rate is read from the **training TB** as a scalar series over
checkpoints (one value per checkpoint), e.g.
`Rollout/Success_Rate/<TaskName>-mean`:

```bash
python scripts/aggregate_seqval_hparams.py \
    --train_root_dir <train_root_dir> \
    --eval_root_dir  <eval_root_dir> \
    --cache_dir      <cache_dir> \
    --train_tag  "Rollout/Success_Rate/<TaskName>-mean" \
    --train_tag_is_rate true \
    --seqval_tag "SequentialValid/PrefixSurvival_Score" \
    --k_list 1 2 3 5 \
    --out_json out/corr.json --out_csv out/corr.csv
```

- `--train_tag_is_rate true` treats `A` as a rate and skips checkpoints whose
  value falls outside `[0, 1]`.
- Mixing a higher-is-better proxy (PrefixSurvival_Score) with lower-is-better
  ones (Loss / MSE / OMN) in one run? Negate the latter with
  `--negate_tag "Valid/Loss" "SequentialValid/ActionL2_mean"` so all proxies are
  oriented "higher = better" before correlating.

### What the metrics mean (`A` = success rate, `B` = proxy)

| Metric | Meaning | Better |
|---|---|---|
| `spearman` | rank correlation of `A` vs `B` over checkpoints | higher |
| `delta_spearman` | same on first differences (step-to-step trend) | higher |
| `kendall_tau` | rank concordance (toggle `--disable_kendall`) | higher |
| `hit@k` | does the top-`k` of `B` include the best-`A` checkpoint? | higher |
| `nregret` | success-rate gap between true-best and `B`-selected ckpt | lower |
| `rank_pct` | percentile rank of the true-best ckpt under `B` | lower |
| `mmrv` | Mean Maximum Rank Violation (Li et al. 2024, arXiv:2405.05941) | lower |

`hit@k` and `nregret` answer the practical question: *"if I pick the checkpoint
my proxy likes best, how close to the truly-best success rate do I land?"*

### Optional: posterior over success (small rollout budget)

If each success rate came from only a few hundred rollouts, the point estimate is
noisy. Pass `--use_posterior_success` with `--train_success_count_tag` (success
counts `k`) and `--n_rollouts n`; surval samples `A ~ Beta(k+a, n-k+b)` and
returns the MC expectation of every metric, so the CI reflects rollout noise too.

---

## 10. Legacy proxy metrics (validation loss, off-manifold norm, action MSE)

These predate the PrefixSurvival metric and are kept for **baseline comparison
only**. They are not needed for the lean path (§5–7); document them here so they
don't clutter the core schema.

### Where they live in the cache

```
metrics/
└── valid/
    ├── Loss                          float64  ()   validation loss (NaN if unset)
    ├── off_manifold_norm/                          ← nested form the reader reads
    │   └── sample_<k>                float64  ()
    └── Off_Manifold_Norm   (flat)    float64  ()   ← LEGACY robomimic form
```

Write them via the optional `write_seqcache_hdf5(..., val_loss=, off_manifold_norms=)`
arguments. `off_manifold_norms` accepts `{sample_idx: value}`, a bare scalar
(treated as `{0: value}`), or `None`.

> **⚠️ Off-manifold-norm format gotcha (observed in current robomimic caches).**
> The main reader reads OMN **only** from the nested group
> `metrics/valid/off_manifold_norm/sample_<k>`. Current robomimic caches instead
> store a **flat** scalar `metrics/valid/Off_Manifold_Norm` (capital, no
> sub-group), which the main reader **silently drops** (`valid_off = None`). The
> `tb_aggregate` seqcache path (`extract_seqcache_metrics.py`) reads the flat
> scalar directly, so cross-seed OMN aggregation still works; only the main
> reader is affected. Regenerate caches in nested form to use OMN through the
> main reader.

### Computing them across seeds

The action-MSE variants and the `Cache/Valid/*` series come from the HDF5
directly, not from TB:

```bash
python scripts/extract_seqcache_metrics.py --seqcache_root <root> --cache_dir <cache_dir>
```

This writes `<cache_dir>/seqcache_metrics/<seed>/<group>.json` with
`Cache/Valid/Loss`, `Cache/Valid/Off_Manifold_Norm`, and five MSE variants
(`MSE_t0_only`, `MSE_tlast`, ...; see `surval.tb_aggregate.seqcache_metrics`).
Then point `aggregate_seqval_hparams.py` at them with
`--force_seqcache_for_tag` / `--seqval_tag "Cache/Valid/Loss" ...`. The
per-task baseline driver `run_baselines_per_task.py` →
`summarize_baselines.py` wraps this end-to-end.

---

## 11. Module map

| Module | Purpose |
|---|---|
| `surval.cache_io` | Canonical HDF5 cache **writer** + helpers shared by every producer |
| `surval.sequential_validate` | Reads caches; computes PrefixSurvival / per-block scaled-error metrics |
| `surval.local_threshold` | State-conditional threshold maps (`local`, `global_db`, `intra_demo_sc`) — uses `obs_features` |
| `surval.scale_estimation`, `surval.surval`, `surval.gates` | Regime-conditioned scale lookup + gate functions |
| `surval.tb_aggregate` | TB / seqcache scalar caching + cross-seed proxy metrics (used by `scripts/`) |
| `scripts/` | Extract → aggregate → summarize/report pipeline (baselines, our method, ablations) |

---

## 12. Where to go next

- **Full cache schema, public `cache_io` API, producer/reader examples,
  migration notes:** [`docs/REFERENCE.md`](docs/REFERENCE.md)
- **Building / querying state-conditional thresholds:**
  [`src/surval/local_threshold/README.md`](src/surval/local_threshold/README.md)
- **Ablation knobs** (`--scale-groups-mode`, `--prefix-time-reduction`):
  `add_common_args` in `src/surval/sequential_validate.py`,
  `tests/ablation_test.py`

---

## Tests

```bash
cd /home/changyeon/workspace/surval
pip install -e .[full,dev]
pytest tests/ -v
```
