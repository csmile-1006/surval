# surval

Rank training checkpoints by how well a policy's predicted actions survive
against the ground-truth trajectory — the **PrefixSurvival** metric — computed
from a cached HDF5 of rollouts, with no simulator in the loop. surval is a
library: you produce a cache, surval scores it, and you correlate the score
against real success rate.

```
produce cache  ──►  score (PrefixSurvival / checkpoint)  ──►  correlate vs. success rate
```

## Install

```bash
pip install -e .[full,dev]      # dev: pytest; full: faiss + matplotlib + sklearn + pyarrow
```

Extras: `db` (faiss — state DB), `lerobot` (pyarrow — LeRobot reader), `sanity`
(matplotlib), `omn` (scikit-learn — legacy off-manifold norm), `full`, `dev`.
Downstream projects pin by commit SHA, e.g.
`surval = { git = "https://github.com/csmile-1006/surval.git", rev = "<sha>" }`.

## Cache schema

One canonical HDF5 per checkpoint. `A` (action dim), `T` (horizon), `S` (pred
samples), `N` (rows), `F` (obs-feature dim) are per-run variables — never
hard-code them.

```
/
├── attrs: checkpoint:str, step|epoch:int, ac_dim=A, num_steps=T,
│          num_cache_samples=S, num_rows=N, has_obs_features=1, obs_feat_dim=F
└── data/                                    attrs: total = N
    └── demo_<i>/                            attrs: num_samples, demo_id (opt)
        ├── actions       float32  [N_i, T, A]       ground truth   (required)
        ├── pred_actions  float32  [S, N_i, T, A]    predicted      (required; 3-D auto-wraps S=1)
        ├── index_in_demo int64    [N_i]             row order      (required)
        └── obs_features  float32  [N_i, F]          state features (required)
```

- **`obs_features`** is the policy's own per-row feature output (a VLM prefix
  embedding, a model encoder output, ...). It keys the state-conditional
  thresholds; always written.
- **Filename ↔ time attr ↔ `--cache-mode` must agree:** `seqcache_step_*.hdf5` +
  `attrs["step"]` + `--cache-mode step` (default), or the `epoch` triple.
  Mismatch ⇒ zero files found.
- Per-demo lengths must match across `actions` / `pred_actions` / `index_in_demo`
  / `obs_features`.

## Producing a cache

**`surval.ingest`** — give a dataset reader + your policy callbacks; surval slices
the ground-truth action chunks and writes the cache (the model-specific
`predict_fn`/`feature_fn` is the only part that can't be generic):

```python
from surval.ingest import RobomimicHDF5Reader, build_seqcache

reader = RobomimicHDF5Reader("demos.hdf5", split="valid",
                             obs_keys=[...], state_keys=[...])
build_seqcache(
    "out/seqcache_step_000600.hdf5", reader,
    predict_fn=my_policy,    # {obs_key: [B, ...]} -> [S, B, T, A]
    feature_fn=my_encoder,   # obs_batch -> [B, F]   (omit to use the dataset state)
    horizon=15, num_samples=8, step=600,
)
```

Readers: `RobomimicHDF5Reader` (robomimic/robocasa/dexmimicgen HDF5),
`LeRobotReader` (LeRobot v2.x parquet, `pip install .[lerobot]`); subclass
`EpisodeReader` for anything else. Runnable example:
[`examples/build_seqcache_robomimic.py`](examples/build_seqcache_robomimic.py).

**`surval.cache_io.write_seqcache_hdf5(...)`** — low-level, when you already have
the row arrays in memory (e.g. a model with its own video-aware data pipeline).
Examples: [`examples/build_seqcache_rlds_openpi.py`](examples/build_seqcache_rlds_openpi.py)
(openpi pi0/pi05 on RLDS DROID) and
[`examples/build_seqcache_gr00t.py`](examples/build_seqcache_gr00t.py) (NVIDIA
Isaac-GR00T N1.5 on its public `demo_data`, scored with `--action-space gr1`) —
both fully open-source, runnable on a GPU.

## Scoring + the metric (3 scripts, surval-only)

The proxy `B` (PrefixSurvival) comes from surval scoring the cache; the **only**
things read from TensorBoard are `success_rate` (ground truth `A`) and
`valid_loss`.

```bash
# 1. score a cache dir -> per-step proxy npz (PrefixSurvival, valid_loss, action_l2)
python scripts/score_seqcache.py --cache-dir <caches> --output-dir <out> \
    --action-space droid --block-scale-method inter

# 2. pull success_rate + valid_loss from the training TB -> gt.npz
python scripts/extract_tb_metrics.py --tb-dir <train_tb> --out gt.npz \
    --success-tag Success_Rate --valid-loss-tag Valid/Loss

# 3. correlate proxy (B) vs success_rate (A)   (multiple seeds: pass paired lists)
python scripts/correlate_metrics.py --gt gt.npz --proxy <out>/proxy_metrics.npz
```

`--action-space` is a built-in layout (`droid` / `gripper` / `dex` / `humanoid`).
`correlate_metrics` aligns by step and reports, for `A` = success rate, `B` = proxy:

| Metric | Meaning | Better |
|---|---|---|
| `spearman`, `kendall_tau` | rank correlation / concordance over checkpoints | higher |
| `hit@k` | top-`k` of `B` includes the best-`A` checkpoint? | higher |
| `nregret` | success-rate gap between true-best and `B`-selected checkpoint | lower |
| `mmrv` | Mean Maximum Rank Violation (Li et al. 2024, arXiv:2405.05941) | lower |

### `--block-scale-method` — `{inter, intra} × {state-free, state-conditional}`

The per-block tolerance `S_g` the errors are compared against:

- **`inter`** (default, state-free): cross-demo expert disagreement at similar progress.
- **`intra`** (state-free): within-demo expert chunk motion `||a[t+ta] - a[t]||`.
- **`state_inter` / `state_intra`** (state-conditional): the per-state version,
  looked up from a state DB keyed on `obs_features`. Build it with
  `scripts/build_state_db_from_cache.py` → `scripts/compute_local_thresholds.py`,
  then score with `--block-scale-method state_inter --state-db-dir <db>
  --threshold-quantile 0.9`. See
  [`src/surval/local_threshold/README.md`](src/surval/local_threshold/README.md).

Prefix survival is soft (LogSumExp smooth-min over blocks, `--prefix-soft-lse-tau`);
ablation knobs `--scale-groups-mode {grouped,flat}` and `--prefix-time-reduction
{product,mean}` (`tests/ablation_test.py`).

## Module map

| Module | Purpose |
|---|---|
| `surval.ingest` | Generic cache builder + dataset readers (robomimic/robocasa HDF5, LeRobot) |
| `surval.cache_io` | Canonical HDF5 cache writer + helpers |
| `surval.sequential_validate` | Reads caches; computes PrefixSurvival / per-block scaled-error metrics |
| `surval.action_spaces` | Built-in block layouts (droid/gripper/dex/humanoid/gr1) |
| `surval.local_threshold` | State-conditional threshold maps (`state_inter`, `state_intra`) |
| `surval.tb_aggregate` | TB scalar I/O + cross-seed proxy metrics |
| `scripts/` | `score_seqcache` → `extract_tb_metrics` → `correlate_metrics`; state DB: `build_state_db_from_cache` → `compute_local_thresholds` |

## Tests

```bash
pip install -e .[full,dev] && pytest tests/
```
