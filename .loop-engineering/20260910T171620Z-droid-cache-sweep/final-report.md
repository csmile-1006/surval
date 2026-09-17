# DROID cache queue: launched, not fully complete

Requested: six tasks, val5/10/20/30, 23 existing dataset cells under /workspace/data.
Apple/pan/pet each have epochs5..50, yielding 12 runnable jobs / 120 checkpoint caches.
The other three task checkpoint sets are missing; their 11 existing cells remain pending.
No other task's policy is substituted.

One stdlib wrapper reuses the cache CLI, records matrix/commands/status/logs, runs one
GPU job at a time and checks each completed job. No score or reference semantics changed.
Runbook: scripts/cache_droid_datasets.md; tests: tests/droid_cache_sweep_test.py.

H100 probe found original torch2.0.1+cu117 lacks sm_90 kernels. Under prior dependency
approval, installed torch2.0.1+cu118 and torchvision0.15.2+cu118 into .venv-droid only.
Shared /venv/droid_policy remains cu117. No driver or system CUDA changes.
12 tests, actual CUDA operation and 32-row/8-sample policy probe passed.
First uncapped apple val5 cache independently verified:488rows,5demos,8samples,
8-step/10-D actions,1024-D features,finite Loss/OMN,independent row coverage.

Private supervisor launched at2026-09-10T17:25:40Z, survives shell exit, no public port.
Output:/workspace/surval/outputs/droid_cache_val5_10_20_30_20260910/
Live progress:run_status.json; matrix:run_manifest.json; aggregate log:orchestrator.log.
Only conditions/ contains experiment caches; gpu_probe/ is capped and excluded.
No full-sweep completion claim: inspect live status after the finite queue exits.
Exact checks:test-results-1.json; requirement status:verification-1.json.
No SURVAL score sweep or custom outcome aggregation was launched.
