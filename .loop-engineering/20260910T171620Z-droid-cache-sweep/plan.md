# DROID validation cache sweep

1. Inventory task directories and TFDS validation sizes, and match saved config.train.dataset_names to each task. Present: 23 dataset cells, three policy runs, 12 runnable cells, 120 checkpoint caches. Three tasks lack checkpoints; ask for their locations while proceeding with available tasks.
2. Run existing CPU cache/aggregation contracts, then a bounded GPU cache probe using the existing compatible environment. No package install, score change, new training, or checkpoint mutation.
3. Add one stdlib Python execution wrapper around scripts/cache_droid_checkpoints.py. It records an explicit matrix and per-job commands/logs/status under one output root, invokes one task/split at a time, and validates the resulting cache manifests. Reuse the existing atomic/provenance-checked cache producer. List/dry-run does not create outputs. No new generic process-pool library is needed for one GPU.
4. Run the wrapper under a private supervisor config/socket under the output root. No network endpoint or changes to existing instance services. The queue is finite, autorestart is disabled, and failures remain visible. Resume reruns the same producer arguments so complete matching epoch caches are skipped.
5. Verify the first real full-split cache before handoff, record commands and live status, and clearly distinguish completed caches, running work, and missing-checkpoint tasks. Bulk completion is not asserted at launch.

Fixed runtime: eight prediction samples, batch 32, action horizon from checkpoint, loader workers 4, torch threads 4, validation up to 50 batches, manifold k=5, seed 0, GPU 0. Offline Hugging Face dependencies use the already populated venv cache. Data root: /workspace/data. Output root: /workspace/surval/outputs/droid_cache_val5_10_20_30_20260910.

Validation: unittest DroidCacheContract + DroidAggregationContract; compile wrapper; list/dry-run matrix; synthetic success/failure worker check; GPU torch operation and actual capped policy cache; first full-split published HDF5 contract inspection.
