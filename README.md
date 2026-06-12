# surval

Open-source implementation of **SurVAL: Rollout-Free Survival Validation for
Robot Policies** ([citation](#citation)).

## Quickstart

```bash
pip install -e .[full,dev]
```

1. **Produce a cache** — at each checkpoint, dump the policy's predicted actions
   next to the ground-truth actions into one HDF5 (`surval.ingest.build_seqcache`
   for a standard dataset, or `surval.cache_io.write_seqcache_hdf5` if your model
   brings its own loader).
2. **Score it** → a SurVAL value per checkpoint (+ a tiny proxy `.npz`):
   ```bash
   python scripts/score_seqcache.py --cache-dir <caches> --output-dir <out> --action-space droid
   ```
3. **(optional) Validate** SurVAL against real success rate on your setup:
   ```bash
   python scripts/extract_tb_metrics.py --tb-dir <train_tb> --out gt.npz --success-tag Success_Rate
   python scripts/correlate_metrics.py --gt gt.npz --proxy <out>/proxy_metrics.npz
   ```

`--action-space` is a built-in robot layout (`droid` / `gripper` / `dex` /
`humanoid` / `gr1`). Everything is surval-only — TensorBoard is read only for the
ground-truth `success_rate` / `valid_loss` in step 3.

## Using your own data, model, or robot

surval is policy/dataset/robot-agnostic — you wire up at most three things, and
the scoring above is unchanged:

- **Dataset** — where ground-truth actions come from: subclass
  `surval.ingest.EpisodeReader`, or use the low-level writer if your model's
  loader already pre-chunks rows.
- **Model** — `predict_fn` (predicted action chunks) + `feature_fn` (the model's
  **own** per-row feature, stored as the required `obs_features`).
- **Action space** — a block layout added to `surval.action_spaces`.

Step-by-step how-to with snippets: [`docs/EXTENDING.md`](docs/EXTENDING.md).
Runnable end-to-end examples (all open-source, GPU):

| Example | Model | Data |
|---|---|---|
| [`build_seqcache_robomimic.py`](examples/build_seqcache_robomimic.py) | robomimic `flow_policy` | dexmimicgen HDF5 |
| [`build_seqcache_rlds_openpi.py`](examples/build_seqcache_rlds_openpi.py) | openpi pi0 / pi05 | RLDS DROID |
| [`build_seqcache_gr00t.py`](examples/build_seqcache_gr00t.py) | NVIDIA GR00T N1.5 | Isaac-GR00T `demo_data` |

---

The rest of this page is reference detail.

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
`predict_fn` / `feature_fn` is the only part that can't be generic):

```python
from surval.ingest import RobomimicHDF5Reader, build_seqcache

reader = RobomimicHDF5Reader("demos.hdf5", split="valid", obs_keys=[...], state_keys=[...])
build_seqcache(
    "out/seqcache_step_000600.hdf5", reader,
    predict_fn=my_policy,    # {obs_key: [B, ...]} -> [S, B, T, A]
    feature_fn=my_encoder,   # obs_batch -> [B, F]   (omit to use the dataset state)
    horizon=15, num_samples=8, step=600,
)
```

Readers: `RobomimicHDF5Reader` (robomimic/robocasa/dexmimicgen HDF5),
`LeRobotReader` (LeRobot v2.x parquet, `pip install .[lerobot]`); subclass
`EpisodeReader` for anything else.

**`surval.cache_io.write_seqcache_hdf5(...)`** — low-level, when you already have
the row arrays in memory (e.g. a model with its own video-aware data pipeline
that pre-chunks rows, like the openpi/RLDS and GR00T examples).

## The SurVAL metric & `--block-scale-method`

`score_seqcache.py` writes the per-checkpoint `SurVAL_Score`. Per timestep, the
per-block prediction error is compared against a tolerance `S_g`, turned into a
survival probability, aggregated across blocks by a soft LogSumExp smooth-min
(`--soft-lse-tau`) and over time by a cumulative product (`--time-reduction`,
ablation). `S_g` is picked by `--block-scale-method`, as
`{inter, intra} × {state-free, state-conditional}`:

- **`inter`** (default, state-free): cross-demo expert disagreement at similar progress.
- **`intra`** (state-free): within-demo expert chunk motion `||a[t+ta] - a[t]||`.
- **`state_inter` / `state_intra`** (state-conditional): the per-state version,
  from a state DB keyed on `obs_features`. Build it with
  `scripts/build_state_db_from_cache.py` → `scripts/compute_local_thresholds.py`,
  then score with `--block-scale-method state_inter --state-db-dir <db>
  --threshold-quantile 0.9`. See
  [`src/surval/local_threshold/README.md`](src/surval/local_threshold/README.md).

## Validating against success rate

To check SurVAL ranks checkpoints like real success rate, `extract_tb_metrics`
pulls the ground-truth `success_rate` (and `valid_loss`) from the training TB,
and `correlate_metrics` aligns it with the proxy by step (multiple seeds:
paired lists → bootstrap CI). For `A` = success rate, `B` = proxy:

| Metric | Meaning | Better |
|---|---|---|
| `spearman`, `kendall_tau` | rank correlation / concordance over checkpoints | higher |
| `hit@k` | top-`k` of `B` includes the best-`A` checkpoint? | higher |
| `nregret` | success-rate gap between true-best and `B`-selected checkpoint | lower |
| `mmrv` | Mean Maximum Rank Violation (Li et al. 2024, arXiv:2405.05941) | lower |

## Module map

| Module | Purpose |
|---|---|
| `surval.ingest` | Generic cache builder + dataset readers (robomimic/robocasa HDF5, LeRobot) |
| `surval.cache_io` | Canonical HDF5 cache writer + helpers |
| `surval.sequential_validate` | Reads caches; computes SurVAL / per-block scaled-error metrics |
| `surval.action_spaces` | Built-in block layouts (droid/gripper/dex/humanoid/gr1) |
| `surval.local_threshold` | State-conditional threshold maps (`state_inter`, `state_intra`) |
| `surval.tb_aggregate` | TB scalar I/O + cross-seed proxy metrics |
| `scripts/` | `score_seqcache` → `extract_tb_metrics` → `correlate_metrics`; state DB: `build_state_db_from_cache` → `compute_local_thresholds` |

## Tests

```bash
pip install -e .[full,dev] && pytest tests/
```

## Citation

If you use surval, please cite the SurVAL paper:

```bibtex
@inproceedings{surval2026,
  title     = {{SurVAL}: Rollout-Free Survival Validation for Robot Policies},
  author    = {Anonymous},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  note      = {Under review},
}
```
