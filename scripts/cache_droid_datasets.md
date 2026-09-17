# DROID validation cache queue

Uses the installed DROID/Octo producer. No retraining or score changes.
Dataset root: `/workspace/data`; task policy runs: `/workspace/droid_policy_learning/checkpoints`.
Matrix: all six task manifests, val5/10/20/30, omit absent val30 only.
Only checkpoints whose saved task name matches the dataset are used. All epochs are included.
Current inventory: 23 dataset cells; apple/pan/pet have 10 epochs each, giving 12 jobs / 120 caches.
The other three task policies are missing and remain explicitly pending.

Defaults: one GPU worker, 8 diffusion samples, batch 32, seed 0, checkpoint action horizon,
4 loader workers, 4 torch threads, validation Loss/OMN on at most 50 batches, manifold k=5.
Each split is inferred independently: slicing val30 would change stochastic and batch-local baselines.
Existing valid caches are reused only after the producer's exact provenance comparison.

```bash
cd /workspace/surval
.venv-droid/bin/python scripts/cache_droid_datasets.py \
  --output-root outputs/droid_cache_val5_10_20_30_20260910 --list
```

`--dry-run` or `DRY_RUN=1` also prints without creating files. No artifact check bypass is
needed to list missing checkpoints: they are reported in the matrix. Optional deterministic
sharding uses `--num-shards N --shard-idx I --gpu DEVICE`; each shard has separate control files.

Real foreground invocation (use the instance's supervisor for persistent execution):

```bash
cd /workspace/surval
.venv-droid/bin/python -u scripts/cache_droid_datasets.py \
  --output-root outputs/droid_cache_val5_10_20_30_20260910 --gpu 0
```

Output root contains `run_manifest.json`, `run_status.json`, `events.jsonl`, and supervisor's
`orchestrator.log`. Each `conditions/<task>_val<N>_<run_timestamp>/seed_0/` has the exact
command in `manifest.json` and `job.log`, atomic `status.json`, and producer files under `cache/`.
Here seed_0 is the evaluation RNG seed, not an inferred policy training seed.
Every completed job reopens all caches to check epoch/sample/episode coverage, cache contract,
and finite Loss/OMN. No score or outcome aggregation is launched by this wrapper.

Exit 0: all requested available task policies complete, no missing tasks.
Exit 1: execution/verification failure (queue stops; inspect the failure log).
Exit 2: available policy jobs completed but task checkpoint sets remain missing.
After supplying additional checkpoints under the checkpoint root, rerun the same command;
existing matching caches are skipped and missing task jobs are discovered.
Do not change samples/batch size/seed in the same output root; use a new output root.

No W&B run, external upload, or model download is created. The existing offline DistilBERT
cache is used. H100 requires the CUDA 11.8 variant of the existing torch 2.0.1 / torchvision
0.15.2 packages, installed only in `.venv-droid` (the shared environment stays unchanged).

## This instance's supervised queue

Private config: `outputs/droid_cache_val5_10_20_30_20260910/supervisord.conf`.
Only an owner-accessible Unix socket is exposed; existing services are unchanged.
Start once after the GPU probe passes:

```bash
cd /workspace/surval
/usr/local/bin/supervisord -c outputs/droid_cache_val5_10_20_30_20260910/supervisord.conf
```

Inspect or resume (do not start a second supervisor):

```bash
cd /workspace/surval
/usr/local/bin/supervisorctl -c outputs/droid_cache_val5_10_20_30_20260910/supervisord.conf status
# Only after the queue has exited / stopped:
/usr/local/bin/supervisorctl -c outputs/droid_cache_val5_10_20_30_20260910/supervisord.conf start droid-cache
```

Failures and completion are not automatically retried. The separate `gpu_probe/`
directory is capped and must not be used for experiments; score only `conditions/`.
