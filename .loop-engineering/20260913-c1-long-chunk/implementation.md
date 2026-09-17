# Implementation

Latest user scope is val30 only. The original model/producer already supports
native15-step output, so no policy edits were required. Added a small preparer
that verifies stable row identity and historical expert H8 prefix and republishes
frozen DINO embeddings with H15 GT. No smaller-split cache derivation remains.

The queue's common verifier now accepts an explicit no-standard-validation job
flag (default remains strict), validates requested horizon, and writes missing
Loss/OMN as null. This lets the existing queue handle the actual SURVAL-only job.

New long-grid math reuses original group geometry, exact local scale tables,
episode grouping, threshold construction, and metric definitions. c is not an
axis. The grid enumerates unique integer worst-action counts and LSE temperatures.
All compared short/long horizons use the same fresh full-horizon prediction draws.

CPU workers prepare and start each val30 task as soon as its10 caches exist,
overlapping remaining GPU inference. Task-specific manifests avoid concurrent
writers; the full preparer later emits combined30-cache verification. Source and
cache hashes gate checkpoint-level resume. Supervised jobs never stop unrelated
OpenPI training and inference allocation is capped at6% GPU memory.

Selection: retain task/common balanced and per-metric optima for all/short/long
scope. Canonical audits cover every published choice and all10 checkpoints;
full original threshold construction covers every published k. All483840 metric
rows are independently checked against the existing scalar metric implementation.

No new dependency, policy training, change to labels/metrics, or external upload.
