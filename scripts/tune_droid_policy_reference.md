# Frozen policy epoch5/epoch25/epoch50: val30-only cached sweep

Uses existing `.venv-droid`; CPU only, no inference, new dependency or upload.
Three task jobs (apple/pan/pet), seed0 denotes cached diffusion draws, not a new
training seed. Epochs5..50 by5,30 validation demos each,30 trials/checkpoint.

Reference and query embeddings both come from each task's epoch25 EMA policy
obs encoder. The1024D vectors already fuse image/proprio and2-frame history;
L2-normalize once, do not append proprio/history again. Target checkpoint
features never enter retrieval. Reference actions are expert GT, not epoch25
predictions. Same-demo self/+-5 frames excluded, exact k, minimum10 neighbors.

Grid: Ta1..15; k10/15/20/25/35/50/75/100/150/200/300/400;
q.01/.025/.05/.1/.15/.2/.25/.35/.5/.65/.75/.85/.9/.95/.99/1.
Fixed worst=ceil(Ta/2), chunk_top_frac=.5, LSEtau1,c1,S8,every_step=True.
2880 HPs x3 tasks x10 checkpoints =86400 scores. Also report the subset
Ta2/4/8/10/12/14/15 without another sweep. No native16 padding.

MSE and both full-val OMNs use the SAME H8 prefix of the saved H15 predictions,
all8 samples; OMN k5. This isolates the representation change. Saved legacy
Loss/policy OMN retain first1600-row coverage and are labeled separately.
All old DINO caches/results remain unchanged. W&B target/uploader: none.

## Inspect without creating outputs

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --list
DRY_RUN=1 SKIP_ARTIFACT_CHECK=1 bash scripts/tune_droid_policy_reference.sh --num-shards 3 --shard-idx 1
```

## Run on this instance

The isolated supervisor config is stored under the new result root. It runs
two CPU workers through the existing process pool, never exposes a port, and
never controls the instance's management services.

```bash
cd /workspace/surval
supervisord -c outputs/droid_policy25_val30_20260915/supervisor.conf
supervisorctl -c outputs/droid_policy25_val30_20260915/supervisor.conf start policy25_tune
# After the three workers complete:
supervisorctl -c outputs/droid_policy25_val30_20260915/supervisor.conf start policy25_report
# After report verification:
supervisorctl -c outputs/droid_policy25_val30_20260915/supervisor.conf shutdown
```

Foreground entrypoints for another managed scheduler:

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --workers 2
bash scripts/tune_droid_policy_reference.sh --report
```

Optional deterministic CPU shards, each contains one task:

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --num-shards 3 --shard-idx 0 --workers 1
```

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --num-shards 3 --shard-idx 1 --workers 1
```

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --num-shards 3 --shard-idx 2 --workers 1
```

Default root: `outputs/droid_policy25_val30_20260915`; `--output-root` changes
it, `--source-root` selects the existing long-cache DINO run. Per-task logs,
status and resumable eval/epoch files live under `conditions/<task>_val30/seed_0`.
Code/cache/config/content signatures guard resume. No-write list is preferred
before running. All30 checkpoints and full row coverage are required.

Report: DINO at prior half/tau1-optimal HP; Policy25 at those SAME HP;
Policy25 re-tuned; common and task-specific aggregate for all/requested Ta.
Canonical threshold/scorer verification covers each selected configuration.
Real-world labels are used for selection: every optimum is posthoc, not a
held-out or statistically significant result. See root REPORT.ko.md and CSVs.

## Epoch50 comparison (same caches and grid)

No new launcher or inference: override the reference epoch and use a NEW root.
Default25 behavior is unchanged. The epoch50 report additionally loads the
preserved `outputs/droid_policy25_val30_20260915` results, checks matching cache
hashes and non-reference baselines, and separates50-at25-HP from50-retuned.
All/requested Ta and common/task-specific selection remain identical.

Inspect all3 CPU jobs without writing outputs:

```bash
cd /workspace/surval
DRY_RUN=1 SKIP_ARTIFACT_CHECK=1 bash scripts/tune_droid_policy_reference.sh \
  --reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915 --list
```

Run on this instance through the isolated supervisor (no network port or upload):

```bash
cd /workspace/surval
supervisord -c outputs/droid_policy50_val30_20260915/supervisor.conf
supervisorctl -c outputs/droid_policy50_val30_20260915/supervisor.conf start policy50_tune
# After3 jobs complete:
supervisorctl -c outputs/droid_policy50_val30_20260915/supervisor.conf start policy50_report
# After the report exits successfully:
supervisorctl -c outputs/droid_policy50_val30_20260915/supervisor.conf shutdown
```

Foreground equivalents for another managed scheduler:

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915 --workers 2
bash scripts/tune_droid_policy_reference.sh --reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915 --report
```

Optional one-task CPU shards (same root, shard-suffixed orchestration files):

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915 --num-shards 3 --shard-idx 0 --workers 1
```

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915 --num-shards 3 --shard-idx 1 --workers 1
```

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915 --num-shards 3 --shard-idx 2 --workers 1
```

Report labels `Policy50_same_HP` mean DINO-optimal HP; `Policy50_same_Policy25_HP`
mean epoch25-optimal HP. Epoch25 optimum metrics are reused with hash/scalar
verification; newly reported epoch50 configurations and DINO controls are
re-scored canonically. The historical epoch25 report's source signature remains
historical; its scientific result files are never rewritten by this comparison.
Both policy OMN values use matching H8/S8 data and are labeled by reference epoch.

## Epoch5 comparison

Same3 tasks,2880 HPs,30 caches and posthoc labels; no new inference. Reference
and query both use the cached epoch5 EMA conditioning. The report separates
`Policy5_same_Policy25_HP` from `Policy5_retuned`, retaining DINO controls
(`Policy5_same_HP` is the DINO-HP transfer). Default25 is unchanged. Use a new
root so existing25/50 outputs remain byte-identical.

No-write listing:

```bash
cd /workspace/surval
DRY_RUN=1 SKIP_ARTIFACT_CHECK=1 bash scripts/tune_droid_policy_reference.sh --reference-epoch 5 --output-root outputs/droid_policy5_val30_20260915 --list
```

This instance: isolated supervisor,2CPU workers,no public port/no upload:

```bash
cd /workspace/surval
supervisord -c outputs/droid_policy5_val30_20260915/supervisor.conf
supervisorctl -c outputs/droid_policy5_val30_20260915/supervisor.conf start policy5_tune
# After all3 jobs finish:
supervisorctl -c outputs/droid_policy5_val30_20260915/supervisor.conf start policy5_report
# After successful report verification:
supervisorctl -c outputs/droid_policy5_val30_20260915/supervisor.conf shutdown
```

Foreground equivalents for another managed scheduler:

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 5 --output-root outputs/droid_policy5_val30_20260915 --workers 2
bash scripts/tune_droid_policy_reference.sh --reference-epoch 5 --output-root outputs/droid_policy5_val30_20260915 --report
```

Optional disjoint one-task CPU shards:

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 5 --output-root outputs/droid_policy5_val30_20260915 --num-shards 3 --shard-idx 0 --workers 1
```

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 5 --output-root outputs/droid_policy5_val30_20260915 --num-shards 3 --shard-idx 1 --workers 1
```

```bash
cd /workspace/surval
bash scripts/tune_droid_policy_reference.sh --reference-epoch 5 --output-root outputs/droid_policy5_val30_20260915 --num-shards 3 --shard-idx 2 --workers 1
```

The prior25/50 reports retain their historical source signatures; their output
files are not rewritten. New5 scalar metrics and all newly selected5 scores are
checked against original metric/scorer implementations. Matched OMNs share
identical H8/S8 prediction inputs; only the fixed feature-space neighbors differ.
