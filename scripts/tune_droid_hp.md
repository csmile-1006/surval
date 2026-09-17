# DROID every-step HP sweep (CPU)

Uses existing caches and shared full-val DINO databases; no training, policy
inference, GPU allocation, W&B, new dependency, or external upload.
Run from `/workspace/surval` with the existing `.venv-droid` environment.

## Matrix

- Tasks: apple, pan, pet; splits: val5, val10, val20, val30; epochs5:5:50.
- Ta: 1,2,3,4,5,6,7,8 (existing cache horizon ceiling).
- Exact eligible k: 10,20,35,50,75,100,150,200.
- Quantile: .1,.25,.5,.75,.9,.95,.99,1.
- Common scale multiplier: .125,.25,.5,1,2,4,8; separate pos/rotation scales.
- Fixed: every_step=True, samples8, temporal exclusion5, min_neighbors10,
  chunk top fraction.5, LSE tau1, cumulative product, equal episode weights.
- 3584 configurations x12 dataset workers x10 checkpoints =430080 scores.
- Seed0 labels existing cached draws, NOT an independent training seed.
- At most2 CPU workers concurrently; each uses2 BLAS/OpenMP threads.

## List and launch

```bash
cd /workspace/surval
DRY_RUN=1 SKIP_ARTIFACT_CHECK=1 bash scripts/tune_droid_hp.sh --list
bash scripts/tune_droid_hp.sh --output-root outputs/droid_hp_every_step_20260913 --workers 2
```

List/dry-run reads only the baseline manifest, creates no output, and performs
no model/data checks. Launch long jobs under the instance supervisor, using a
private socket/config in the output root. Re-running a completed worker checks
source-cache, code, DINO, grid and result hashes; incompatible roots are rejected.

Optional deterministic sharding uses sorted dataset indices modulo shard count.
For a shared filesystem, shard control filenames are separate. Each block is
self-contained and uses the same grid/seed; no GPU slice applies to CPU jobs.

```bash
cd /workspace/surval
bash scripts/tune_droid_hp.sh --output-root outputs/droid_hp_every_step_20260913 --num-shards 2 --shard-idx 0 --workers 2
```

```bash
cd /workspace/surval
bash scripts/tune_droid_hp.sh --output-root outputs/droid_hp_every_step_20260913 --num-shards 2 --shard-idx 1 --workers 2
```

Overrides: `DROID_PYTHON`, `--baseline-json`, `--dino-root`, `--output-root`,
`--axes-json` (JSON with all four axis names), `--workers`, shard flags.
Boundary-extension runs must use a new output root and preserve the initial grid.

## Completed experiment and final report

The initial3584-HP run is preserved. A predeclared single boundary extension
adds k400 and scale multipliers.0625/16: 5184 configurations,622080 scores.
Its root is `outputs/droid_hp_every_step_extended_20260913`.

```bash
cd /workspace/surval
bash scripts/tune_droid_hp.sh \
  --output-root outputs/droid_hp_every_step_extended_20260913 \
  --axes-json outputs/droid_hp_every_step_extended_20260913/axes.json --workers 2
.venv-droid/bin/python scripts/report_droid_hp.py \
  --output-root outputs/droid_hp_every_step_extended_20260913 --audit
.venv-droid/bin/python scripts/verify_droid_hp.py
```

The ordinary scorer also reproduces a chosen setting. Example shared minimax HP;
`--cache-dir` may point to one condition or the entire120-cache collection:

```bash
cd /workspace/surval
.venv-droid/bin/python scripts/score_droid_cache.py \
  --cache-dir outputs/droid_cache_val5_10_20_30_20260910/conditions \
  --dino-db-root outputs/shared_dino_val \
  --output-dir outputs/droid_common_robust_reproduction \
  --every-step --exact-neighbors --ta 8 --k-neighbors 10 \
  --threshold-quantile 0.1 --scale-multiplier 0.25 \
  --num-samples 8 --min-neighbors 10 --temporal-radius 5 \
  --chunk-top-frac 0.5 --lse-tau 1
```

Use `--ta 7` for the shared macro-balanced alternative. Per-task balanced HPs:
apple Ta7/k20/q.25/x2; pan Ta1/k20/q.9/x1; pet Ta6/k10/q.25/x.125.
Only the shared DINO path supports the new exact-neighbor/scale flags.
No uploader exists in this repository; no remote publication is performed.

## Artifacts and interpretation

Root: run_manifest.json, run_status.json, orchestrator.log, events.jsonl.
Each conditions/<task>_<val>/seed_0 contains manifest.json, status.json, job.log,
and eval/{scores.npz,scales.npz,verification.json}. Scores keep every configuration
and all10 checkpoint values. Old stride8 and old buffered every-step controls
are retained. Every source cache is hashed before and after scoring.

Real-world tuning is post-hoc/resubstitution, not held-out evidence. Report
normalized regret and MMRV (lower), Spearman (higher). Choose per-task balanced
loss averaged over4 splits; common robust HP minimizes worst task balanced loss,
then macro mean. Balanced loss = (nregret + MMRV/task SR range + (1-rho)/2)/3.
Constant or nearly constant scores (range<=1e-8) are retained but not selectable.
Metric-specific bests, macro alternative and Pareto alternatives accompany this
explicit tradeoff. Val sizes are not tuned, and are not independent seeds.
