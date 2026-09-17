# DROID every-step HP tuning

1. Add opt-in exact-neighbor selection to shared threshold retrieval, default off.
   Reuse it in both local/global pairwise threshold functions. Preserve previous
   buffered behavior and old artifacts. Add exact-neighbor and scale-multiplier
   options to DROID threshold/scorer CLI so chosen HPs can be reproduced normally.
2. Add a focused tuning module. Reuse existing physical block distance and chunk
   reducers. Precompute per-checkpoint unscaled chunk errors and batched expert
   pairwise action-distance matrices. Compute all quantiles per k together.
   Positive row/block scales commute mathematically with top-k/sample averaging;
   exploit that for a batched equivalent of the current gate/LSE/cumprod/episode
   mean. Validate float32 roundoff against canonical scoring, not a new formula.
   Store every HP's10 checkpoint scores so ranking metrics can be recomputed.
3. Launch12 dataset workers on CPU (at most2 concurrent) using the existing
   cache_droid_datasets.py now/save_json/run_worker helpers. Add a thin Bash
   launcher and Markdown; --list/dry-run creates no files, deterministic shard
   selection. No new shell helper layer is necessary because leaf scripts only
   exec the reusable Python orchestration. No W&B, GPU allocation or uploads.
   Per condition/seed_0: manifest/status/job.log/eval. Run manifests/status/events
   include axes, source signatures, exact commands, and resume compatibility.
4. Exhaust the grid on all12 splits. Selected HP analysis uses predeclared
   equal-metric balanced loss plus independent best-per-metric outputs. Domain
   means weight4 splits equally; common minimax first balances worst task.
   Also report common macro optimum, Pareto alternatives and all original HP
   controls (old stride8, buffered every-step, exact every-step).
5. Verify all cell/checkpoint/sample/row coverage, exact k, input hashes, score
   equivalence and ranking agreement. Re-score unique reported winners using
   canonical score_policy_cache every_step=True. If score roundoff changes
   relevant rankings, repair precision/expand canonical audit before selecting.
   Inspect useful optimum boundary hits and extend k/scale once if needed.
6. Write complete CSV/JSON and Korean report with numerical task-specific and
   common recommendations, explicit tradeoffs, full axes and post-hoc caveat.

Checks: focused unittest/pytest for thresholds and tuning, Bash syntax/ShellCheck
where installed, dry-run matrix/shard coverage, harmless worker success/failure
probe, real single-cell timing/equivalence probe, full12-cell run and selected
canonical audit. No training or policy inference; GPU currently busy elsewhere.
