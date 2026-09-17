# DROID c=1, val30-only long-chunk search

Uses the existing `.venv-droid`; no new dependencies or training. Three tasks
(apple/pan/pet), epochs5..50 by5, 30 validation demos each. No val5/10/20 work.
The root defaults to `outputs/droid_c1_long_20260913`.

Fixed: c=1, all8 diffusion samples, every observation timestep, frozen shared
DINO representation, full-val30 reference, exact eligible k, same-demo temporal
exclusion radius5, minimum10 neighbors, seed0. Position L2 / rotation SO(3),
gripper excluded. Quantile normalization stays independent for both groups.

Grid: Ta1..15; k10/15/20/25/35/50/75/100/150/200/300/400;
q.01/.025/.05/.1/.15/.2/.25/.35/.5/.65/.75/.85/.9/.95/.99/1;
worst-action count1..Ta; LSE tau.03/.1/.3/1/3/10/30.
161,280 HPs ×3 tasks ×10 checkpoints =4,838,400 scores.
Worst-action count uses an interior fraction `(count-.5)/Ta` to ensure the
canonical `ceil(Ta*frac)` is exactly the intended integer.

## Inspect (does not create outputs)

```bash
cd /workspace/surval
bash scripts/tune_droid_long_hp.sh --list
DRY_RUN=1 bash scripts/tune_droid_long_hp.sh --num-shards 3 --shard-idx 1
```

## Run

The native model has Tp16/To2, so max15 real future actions; longer horizons
would change model semantics. The producer uses `--cache-ta 15`, not padding.
Existing H8 inputs are never overwritten. New H15 caches are checked against
the exact val30 RLDS row set and bitwise H8 expert-action prefix. New DINO DBs
keep all embeddings/firstactions unchanged and only extend expert chunks.

On this instance, use the private supervisor config in the output root:

```bash
cd /workspace/surval
supervisord -c outputs/droid_c1_long_20260913/supervisor.conf
supervisorctl -c outputs/droid_c1_long_20260913/supervisor.conf start long_tune
# After all three CPU scoring jobs complete:
supervisorctl -c outputs/droid_c1_long_20260913/supervisor.conf start long_report
```

For an already-running supervisor, skip its first command. The cache program
is a single shared-GPU worker, batch32 with a6% allocation cap, waiting for6GiB
free memory. CPU scoring waits for each task's10 caches and overlaps remaining
GPU inference. Two CPU workers default; `--workers` changes that count.
Sharding is deterministic `task_index % num_shards == shard_idx`. For an
external managed scheduler, the foreground entrypoints are:

```bash
cd /workspace/surval
.venv-droid/bin/python scripts/prepare_droid_long_cache.py
bash scripts/tune_droid_long_hp.sh --workers 2
bash scripts/tune_droid_long_hp.sh --report
.venv-droid/bin/python scripts/verify_droid_long_hp.py
```

All logs/manifests/status/events live under the same root, with inference under
`cache_generation/conditions/*/seed_0`, scoring under `tuning/conditions/*/seed_0`.
Scoring resumes at verified checkpoint boundaries; incompatible source/cache
hashes fail closed. Completed workers verify saved score/scale hashes.

Report outputs: `tuning/REPORT.ko.md`, `selected_metrics.csv`, `all_metrics.csv`,
per-task `eval/scores.npz`/`metrics.npz`, `canonical_scores.json`, `verification.json`.
The final verifier additionally checks all published selection objectives and CSV
rows, then emits `selected_checkpoint_scores.csv` (810 rows with scores/outcomes)
and `integrity_verification.json`. It works after either unsharded or sharded scoring.
No W&B run or uploader: this is local cached posthoc analysis, not a new seed.

Real-world outcomes are used for HP selection. Label all maxima **posthoc**;
they are not held-out generalization estimates. Balanced loss remains
`(NRegret + MMRV/SR_range + (1-Spearman)/2)/3`. Report both balanced and
metric-specific choices, task-specific and common HP, and matched Ta<=8 vs>8
using the same H15 predictions. Every published choice is re-scored by the
canonical implementation and every metric row is independently recomputed.
