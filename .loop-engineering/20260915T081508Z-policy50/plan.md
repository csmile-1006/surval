# Plan

1. Hash preserved epoch25 output tree and inspect current code/skills/dirty worktree.
2. Reuse `tune_droid_policy_reference.py` and `.sh` unchanged with explicit
   `--reference-epoch 50 --output-root outputs/droid_policy50_val30_20260915`.
   Add epoch50 reference identity/target-feature invariance and CLI matrix tests
   in existing `tests/droid_policy_reference_test.py`.
3. Extend existing `report_droid_policy_reference.py` from hardcoded Policy25 to
   explicit supported25/50 labels. For50 load preserved25 score matrices/proofs,
   compare25-optimal and50-at25-HP to50-retuned with both all/requested scopes and
   common/task-specific balanced selection. Keep DINO control and target-at-DINO-HP.
   Validate identical prediction cache hashes, GT identity, input/fixed settings
   (reference epoch excepted), unchanged non-reference baselines, and old macro values.
4. Baselines use existing matched H8/S8 worker results. Add archived epoch25 OMN
   as a clearly named comparison. Reuse scalar metrics, original thresholds and
   scorer for all newly reported epoch50 HP and current DINO controls. Historical25
   optima were audited previously; validate their artifact hashes and scalar values.
5. Update existing `.md` with explicit50 commands,new root,3 CPU shards,private
   supervisor/no upload. Two CPU workers, no GPU/services exposed. Default25 unchanged.
6. Focused tests and no-write list before start; 35+ runnable regression tests,
   actual3-job completion,resume and canonical audit afterward. ShellCheck/legacy
   pytest absence is retained as a limitation,not a pass. Preserve report and old
   artifacts; final verification checks all requirements and shuts private supervisor.

Use existing source kernels/runner; no new production launcher/report copy.
The report's historical source signature may differ after extension, while its
saved scientific outputs must remain byte-identical. Repair limit3.
