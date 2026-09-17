# Complete: c=1, val30-only DROID SurVAL tuning

L1–L5 all satisfied; no remaining implementation/verification gaps.
Only apple/pan/pet val30 were inferred or tuned. Original weights, historical
caches, DINO representation, real-world outcomes, and metric definitions remain
unchanged. Reused native producer, queue, geometry, threshold and metric helpers;
no new dependency, policy training, external upload, or unrelated process stop.

## Result

161,280 HPs ×3 tasks ×10 checkpoints =4,838,400 scores. Genuine15-step caches
respect Tp16/To2. c1 and every_stepTrue fixed; additional k/quantile/worst-action
count/LSE-temperature search. These are posthoc finite-grid optima, not held-out
generalization estimates or mathematical global maxima.

Long-only balanced choices (one HP tuple per task):

| Task | Ta | k | q | worst count | LSE tau | NRegret | MMRV | Spearman | Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apple |14|100|.95|14|10|0|.0133333|.9786345|40|
| pan |11|50|.75|7|30|0|.0433333|.7766372|30|
| pet |9|10|.05|2|30|0|.0766667|.7339758|30|

Apple/pan match their best short-chunk primary metrics. Pet improves balanced
Spearman .709510→.733976 with unchanged NRegret/MMRV. The shared balanced/minimax
optimum across ALL lengths is still Ta6/k300/q.35/worst3/tau10; long chunks are
not universally superior. Long-only common alternatives and every metric-specific
optimum are explicitly separated in the full report/CSV.

## Verification and commands

- 44 tests passed (30 unittest,14 existing pytest checks).
- `scripts/prepare_droid_long_cache.py` completed30 caches and3 frozenDINO DBs.
- `scripts/tune_droid_long_hp.py` completed3 workers and all scores.
- Completed `--worker-index 0` resume rechecked hashes and skipped recomputation.
- `scripts/tune_droid_long_hp.py --report`:280 canonical scores, maximumabsolute
 error2.3956020944737588e-8; metrics and selected epochs identical. Every483840
 primary-metric row independently recomputed using the original definition.
- `scripts/verify_droid_long_hp.py`:51 independent optima checked, full metric CSV
 validated,810 selected checkpoint rows exported, sources/code/outcomes unchanged.
- `git diff --check`, Bash syntax, dry-run/sharding and Python compilation passed.

Exact environments/commands/exits are in test-results-1/2/3.json. Two early smoke
invocation failures were environmental/CLI mistakes, corrected before inference;
no product repair or relaxed test was required.

Results: `/workspace/surval/outputs/droid_c1_long_20260913/tuning/REPORT.ko.md`,
`selected_metrics.csv`, `selected_checkpoint_scores.csv`, `all_metrics.csv`,
per-task `eval/scores.npz`, `verification.json`, `integrity_verification.json`.
Reproduction: `scripts/tune_droid_long_hp.md`.

The private supervisor was shut down after all three programs exited successfully;
unrelated services and training jobs were left untouched.

Loop artifacts: `/workspace/surval/.loop-engineering/20260913-c1-long-chunk/`.
