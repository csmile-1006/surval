# DROID diffusion-policy evaluation

Run these commands from `surval/`. The cache command imports the installed
`droid_policy_learning` fork of robomimic and the patched Octo checkout. The score
command uses only cached arrays, surval and FAISS; it does not load a policy.

## Environment

This instance now has an authorized compatibility environment at
`/workspace/surval/.venv-droid`. Use `.venv-droid/bin/python` for the DROID
commands. It reuses `/venv/droid_policy` packages read-only via system-site-packages
and overrides incompatible dependencies locally; it is not a standalone clone.
For the downloaded policy language encoder set
`HF_HOME=/workspace/surval/.venv-droid/model-cache/huggingface`.
The existing shared environment was not modified. See [DROID_REVIEW.md](DROID_REVIEW.md)
for the verified scope, actual smoke outputs and reference-manifold caveats.

Use a compatible DROID environment (the original training dependencies are
documented in `droid_policy_learning/README.md`). Install the local surval source
and its existing database extra into that environment when setting it up:

```bash
python -m pip install -e '.[db]'
```

The checked-in command scripts also add this repository's `src/` to the import
path, so an editable surval install is optional when its dependencies already
exist. TensorFlow 2.15 and the old DROID torch stack need a compatible NumPy 1.x
environment. Do not upgrade/downgrade a shared environment while training is
running; use an isolated environment for repairs. The default policy-feature
path needs no extra image encoder. The optional shared-DINO path below downloads
frozen DINOv2 weights once. DistilBERT is the existing policy language encoder
and is loaded only when the policy's batch preprocessing requests it.

## Cache checkpoints

```bash
python scripts/cache_droid_checkpoints.py \
  --checkpoint-dir /absolute/path/to/run/models \
  --data-dir /workspace/data \
  --dataset-name put_the_pet_on_the_shelve/droid/val30 \
  --cache-dir outputs/droid/cache \
  --num-cache-samples 8 --batch-size 32
```

`--dataset-name` defaults to the checkpoint's unique task-specific dataset; the
base `droid` co-training dataset is excluded from that automatic selection.
`--data-dir` defaults to the saved training root. TFDS `val` is used if present,
otherwise Octo's existing final-5%-of-train validation convention applies.

One epoch per file is written below
`<cache-dir>/<experiment>/<run timestamp>/<dataset group>/seqcache_epoch_XXXXXX.hdf5`.
`seqcache_manifest.json` records the source checkpoint and dataset. Each HDF5
records its source signatures, normalization, encoder selection, horizons,
sample count, validation settings and RNG seed. Existing files are reused only
when that provenance matches. Use a new output directory for different cache
settings, or explicitly supply `--overwrite` to replace the affected epoch files.
Publication is atomic so interrupted writes do not appear complete.

For a bounded initial check, add:

```bash
--target-epoch 5 --max-demos 2 --max-rows 64 \
--num-cache-samples 1 --batch-size 2 --valid-num-steps 1
```

`--max-demos` and `--max-rows` cap the input before image decoding. A row cap can
end within the first episode; the manifest reports how many episodes were
actually cached. `--cpu` supports small checks without GPU use. The normal path
still materializes the selected decoded validation rows once, as the original
DROID producer did; bound the selection if memory is limited.

Validation loss and the existing batch-local OMN are measured during caching,
using up to `--valid-num-steps` batches (default 50). They are not recoverable
from predicted actions alone. `--no-run-standard-validation` skips both and
leaves them missing. MSE variants are calculated later from the action cache.

## Local thresholds and scores

```bash
python scripts/score_droid_cache.py \
  --cache-dir outputs/droid/cache \
  --output-dir outputs/droid/evaluation \
  --threshold-quantile 0.95 --k-neighbors 50
```

For each checkpoint the command:

1. Reads its cached policy encoder vectors and L2-normalizes them for the
   existing cosine-neighbor retrieval.
2. Creates a state DB using that checkpoint's own features and aligned expert
   actions. The complete observation-horizon conditioning is used; EMA weights
   are selected when the inference policy uses EMA. Images, proprioception and
   any other configured inputs are already represented by the policy encoder;
   no separate feature extractor or additional proprio vector is appended.
3. Calls surval's existing local-threshold code with pos and rot6d blocks.
   Same-episode neighbors within `--temporal-radius` (default 5) are excluded.
   Fewer than `--min-neighbors` (default 10) triggers the library's global
   fallback. The fallback rate is reported; missing state mappings are errors.
4. Calls the library's continuous score and writes per-checkpoint results.

DB/threshold artifacts are stored under `evaluation/thresholds/epoch_<n>_<hash>/`.
The hash includes the source cache signature and threshold configuration, so a
different checkpoint, overwritten cache or changed k/quantile does not silently
reuse another encoder's thresholds.

There is no hard/soft switch in this workflow. It uses the current library path:
scaled block errors, top-fraction chunk reduction, continuous exponential gates,
logsumexp across blocks, and a cumulative product through the trajectory.
The library still calls that branch `soft` internally for legacy consumers.
The old DROID tau-dependent formula is not ported. `--lse-tau` controls only the
library's block logsumexp operation. Gripper is excluded from SURVAL blocks,
matching the existing DROID action-space definition.

`--ta` may shorten the cached action horizon; `--num-samples` selects a prefix of
the cached prediction samples (default all). Default score row stride is Ta;
`--every-step` uses every row. MSE baselines always use the full stored horizon
and all cached samples, independently of these score-only options.


## Shared DINO DB and full-split DINO OMN

Pass `--dino-db-root` to use a checkpoint-independent representation for both
SURVAL local thresholds and the new `Off_Manifold_Norm_DINO` baseline. Without
this argument the existing policy-encoder path is unchanged.

Build from existing full-validation DROID caches and raw RLDS images, once per
dataset. This loads saved config JSON, not policy weights, and does not rerun
diffusion sampling or modify HDF5 caches. DINOv2 ViT-B/14 uses frozen CLS features,
224x224 images, both saved exterior-camera views, and the cached observation
history (currently 2). History is causal and clamped within each episode; each
current image is encoded only once, then features are gathered into windows.
Views/history are concatenated with the existing library's L2 normalization.
No proprioception or language is appended. Current DB dimension is 3072.
The official DINO code revision is pinned; the weight digest and preprocessing
settings are persisted. The first build downloads the official source/weights
to `.venv-droid/model-cache/torch`.

For all existing apple/pan/pet val5/10/20/30 caches:

```bash
cd /workspace/surval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/workspace/surval/.venv-droid/model-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.venv-droid/bin/python scripts/build_droid_dino_db.py \
  --cache-dir outputs/droid_cache_val5_10_20_30_20260910/conditions \
  --data-dir /workspace/data \
  --output-dir outputs/shared_dino_val

.venv-droid/bin/python scripts/score_droid_cache.py \
  --cache-dir outputs/droid_cache_val5_10_20_30_20260910/conditions \
  --dino-db-root outputs/shared_dino_val \
  --dino-omn-k 5 \
  --output-dir outputs/droid_dino_all
```

The builder accepts a single condition directory too. It requires an uncapped
`droid_policy_learning` epoch cache and unchanged dataset files. DB layout is
`<root>/<dataset_name>/`, containing the standard StateDatabase files plus
`expert_action_chunks.npy` (physical units, full stored horizon) and
`shared_dino_meta.json`. A generic state DB without these alignment artifacts
is not silently accepted. Existing compatible DBs are reused; incompatible ones
require a new output root, never an implicit overwrite.

The scorer aligns by exact `(source demo ID, observation timestep)`, accepting
row permutations but rejecting missing/extra rows, changed GT, action layout,
normalization-derived targets, history or action-start offset. It never matches
by row number alone. A single `thresholds/dino_<hash>` artifact is reused across
checkpoints for a given DB/config. SURVAL's formula and per-action-group expert
difference quantiles are unchanged; only the state representation changes.

DINO OMN uses exact cosine kNN over the **entire same val split**:

- Query once per observation row, not per repeated action-horizon feature.
- Exclude self and same-demo observations within `--temporal-radius` (default 5).
  Retain the closest `--dino-omn-k` eligible rows (default 5).
- At chunk offset h, project the prediction onto neighboring expert actions at
  that same h. Preserve the existing OMN projection formula, including its
  nearest-action convention when only one neighbor exists.
- Use checkpoint-normalized action coordinates, all 10 dimensions including
  gripper, every saved sample and every chunk offset. `--ta`/`--num-samples`
  affect SURVAL only, not this baseline. Neighbor sets do not depend on inference
  batch composition or checkpoint encoder weights.
- Fewer than k neighbors are reported and used. If any query has no eligible
  neighbors, the checkpoint DINO OMN is null/blank, not zero.

JSON includes neighbor counts, per-sample OMN, DB fingerprint, units and protocol.
The original `Off_Manifold_Norm`, `Loss` and `MSE_mean_pred` remain unchanged.
Do not interpret old and DINO OMN as differing only in encoder: retrieval scope,
temporal exclusion and evaluation coverage also differ. The outcome aggregator
automatically includes `Off_Manifold_Norm_DINO` as a lower-is-better proxy when
the column exists; legacy CSVs remain supported.

Reference and query are both the selected full val split, following the chosen
custom-robomimic convention. Thus val5 and val30 have different reference sets;
the shared representation removes checkpoint/batch dependence, not this
validation-size dependence.

Verified on 2026-09-12: apple val5, 488 rows, all 10 cached checkpoints (epochs 5–50),
8 samples x 8 offsets. Every row had 5 eligible DINO neighbors; all 10 checkpoints
shared one threshold directory. Results:
`outputs/droid_dino_apple_val5_20260912/checkpoint_metrics.csv`.
The other 11 split DBs have not been generated by this smoke check; the commands
above build them and reuse the completed apple val5 DB.

## Time and action coordinates

For observation row `t`, the existing DROID training transform first removes
the first action in the Octo window with `action[1:]`. The diffusion policy
returns the slice beginning at `To - 1`. Cache GT now takes that same slice.
For the current `To=2` checkpoints, both prediction and GT therefore start at
original action `t+1`. `index_in_demo` remains the observation index `t`; the
cache records `action_start_offset=1`. End-of-episode actions repeat the final
action, following Octo's existing absolute-action padding convention.

Cache arrays remain in checkpoint-normalized coordinates. Evaluation does not
fit new action statistics: saved scale/offset normalize the selected dataset,
with clipping to the policy's [-1, 1] target range. Score and threshold code
inverse-transform those targets/predictions before computing translation and
SO(3) rotation distances. Thus denormalized GT is the clipped training target
in physical units, not an independently refitted validation coordinate system.

Legacy DROID caches lack the required alignment/normalization/encoder metadata
and must be regenerated for this path. Generic step caches and legacy consumers
continue to use the existing library APIs.

## Outputs and later custom outcomes

`checkpoint_metrics.csv` contains one row per run/dataset/checkpoint, with
`surval_score`, `Loss`, `Off_Manifold_Norm` and `MSE_mean_pred`. The MSE baseline
reuses the existing extractor and includes every action dimension, gripper
included; the sample mean is taken before the error, so sampling variance is not
charged against a stochastic policy.
`checkpoint_metrics.json` also contains per-episode scores, threshold diagnostics,
settings, artifact paths and provenance. Missing validation metrics are JSON
null / blank CSV fields, never zero.

### Baseline and SURVAL aggregation

All four proxies are included: SURVAL, validation Loss, Off_Manifold_Norm and
MSE_mean_pred. Keep custom measurements in a separate CSV with columns
`run,run_timestamp,dataset,epoch,outcome`. Copy the four key columns exactly from
`checkpoint_metrics.csv` and add your measurements under `outcome`. Extra columns
are ignored, so a separate copy of the metrics CSV with an added outcome column
also works. An outcome is higher-is-better; blank cells mean not measured.

```bash
python scripts/aggregate_droid_metrics.py \
  --metrics-csv outputs/droid/evaluation/checkpoint_metrics.csv \
  --outcomes-csv outputs/droid/custom_outcomes.csv \
  --output-json outputs/droid/evaluation/proxy_aggregate.json
```

This reuses the existing surval aggregation formulas: Spearman, Kendall tau,
delta-Spearman, hit@1/3/5, normalized regret, rank percentile and MMRV. Baselines
are negated internally because lower is better; SURVAL is higher-is-better.
Checkpoint rows are joined by the explicit four-column key, not CSV order.
Duplicate keys, non-finite numbers and outcome keys absent from the metrics file
are errors. Missing outcomes/proxy values are omitted per method and the exact
included/omitted epochs are reported; methods with fewer than two paired epochs
are skipped with a reason. Compare methods on the same epoch coverage when
making a fair ranking. Undefined correlations are null, not zero.

The JSON includes per-run/dataset selection metrics and selected/oracle epochs,
plus equal-run-weight means, population standard deviations and valid group
counts within each dataset. Checkpoints are not counted as independent seeds,
and raw scores/outcomes from different tasks are not pooled. Ties select the
earliest epoch, following the existing library convention. Delta correlations
use differences between adjacent available epochs, even if epoch gaps differ.
Only the aggregation command needs rerunning when custom outcomes change; it
does not regenerate caches, thresholds or scores, or modify the input CSVs.

## Loader-order investigation (2026-09-10)

Out-of-order input is compatible with sequential scoring: cache writing groups
rows by source episode and timestep, and SURVAL groups/sorts rows before its
cumulative calculation. The current cache producer requests a full action chunk
for each complete observation window, bypassing the rollout action queue.
The row-order regression test checks identical scores/per-episode results for a
fixed set of predictions and thresholds after permuting every cache row.

However, the two shuffle switches do different things:

- Octo/dlimp `shuffle=True` passes `shuffle_files=True` to TFDS. It does not itself
  increase `num_parallel_reads`, trajectory/frame mapping or interleave workers.
  dlimp also sets `options.deterministic=False` independently of file shuffle.
- PyTorch DataLoader shuffle only changes indexing after `TorchRLDSMapDataset`
  has already decoded/materialized the entire selected dataset. It does not
  accelerate that preceding RLDS read.

A CPU-only check used the existing openpi environment and local
`put_the_pet_on_the_shelve/droid/val5`: one validation shard, five episodes,
626 rows, two cameras, observation horizon 2, future action window 15, 128x128
decode/resize. Two passes per setting measured iterator consumption only
(graph construction and model inference excluded); OS data was already warm.
TensorFlow inter/intra-op thread limits were 4/2. Parallelism was applied to
reads, trajectory maps, flatten interleave and frame transforms together.

| File shuffle | Pipeline parallelism | Median seconds |
| --- | ---: | ---: |
| False | 1 | 0.928 |
| True | 1 | 0.944 |
| False | 4 | 0.317 |
| True | 4 | 0.308 |

Every pass had exactly the independently enumerated 626 source/timestep keys,
no duplicates, and identical per-key action-window and decoded-image hashes.
The diagnostic used RLDS's original 7-D actions, not the DROID 10-D policy
standardizer or policy inference. Thus this demonstrates loader correctness and
roughly 2.9x faster loading with four workers for this sample, not a full-cache
speedup. File shuffling one shard provides no useful extra file parallelism.

This optimization is now enabled by default for the robomimic producer through
`--loader-parallelism 4` (`1` for a serial comparison). It uses a cheap independent
metadata pass and then parallel decoding. Subsets are chosen in lexicographic
source-ID order, not read-arrival order. Provenance version 2 records expected
row counts/key hash and rejects prior traversal-order cache reuse. The pipeline is:

1. Establish the expected row keys after validation filters and any explicit
   subset selection. Use source `file_path`; `_traj_index` fallback is assigned
   by enumeration after reading and is not a stable ID under shuffled reads.
2. Build observation/action windows within each episode, then allow parallel
   decode/interleave. Keep the finite validation dataset: no training `.repeat()`
   or weighted sampling. Select any capped subset before nondeterministic work.
3. Assert every expected `(source_id, timestep)` occurs exactly once. Counting
   visited rows or checking for gaps within visited episodes alone cannot detect
   a completely missing episode or missing tail. Validate each prediction sample
   and feature array against the same keys before publishing a complete cache.
4. Sort materialized rows by a fixed source/timestep order before batching for
   validation and inference. Keep batch size, RNG seed and that order fixed.

The actual DROID 10-D loader was additionally checked on all 626 val5 rows:
worker counts 1 and 4 produced identical keys, actions and decoded observations.
One run measured 2.894 vs 0.935 seconds including the new metadata pass. This is
not an end-to-end GPU benchmark. Every prediction sample/feature array and the
saved cache itself are checked against the independent row manifest.

The final sort matters beyond scoring: reordered inference assigns diffusion
noise draws to different rows; the existing OMN uses batch-local neighbors;
limited validation steps may otherwise cover different examples. Canonical
ordering also avoids changing FAISS tie ordering merely because rows arrived
differently. These are not corrected just by sorting predicted outputs later.
Since decoded data is reused across checkpoints, measure loading separately from
GPU inference before expecting a comparable end-to-end speedup.

## Checks

```bash
PYTHONPATH=src:tests python -m unittest \
  droid_pipeline_test.DroidCacheContract droid_pipeline_test.DroidAggregationContract -v
PYTHONPATH=src python -m pytest tests/cache_io_test.py \
  tests/local_threshold/database_test.py tests/local_threshold/threshold_test.py \
  tests/ablation_test.py tests/tb_aggregate_test.py -q
```

The stdlib test file includes cache/score/aggregation contracts and a separate
`DroidRuntimeContract` that exercises real TensorFlow/Octo/DROID preprocessing
without loading a checkpoint or downloading encoder weights. Actual checkpoint
smoke execution is a separate validation step, not implied by these tests.
Run that additional class only in the compatible DROID environment:

```bash
PYTHONPATH=src:tests python -m unittest droid_pipeline_test.DroidRuntimeContract -v
```
