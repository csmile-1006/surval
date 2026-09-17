# Loader / review / runtime plan

1. Reuse Octo's finite filtered trajectory/windows; require stable source paths. Add surval.rlds_cache to enumerate expected metadata, select canonical subset, parallel decode and sort. Replace the redundant map wrapper with its already-materialized list; keep deterministic Torch batches.
2. Wire parallelism (default 4), manifest hash/counts and per-sample coverage checks into the DROID producer. Do not change training loading or score formulas. Producer provenance version 2 invalidates prior traversal-order caches.
3. Test synthetic omissions/caps and real val5 CPU rows at worker counts 1/4. Review split manifests, configs, old scorer and OMN definitions, documenting verified and non-applicable issues.
4. User authorized dependencies: create .venv-droid with read-only existing DROID site packages, install local NumPy 1.x/FAISS/sklearn and ABI companions. Run actual runtime contracts and a CPU epoch-5 bounded cache then SURVAL/baseline scoring; retain failures and bounded repairs.
5. Report remaining methodological decisions without implementing a new baseline definition.

Commands: PYTHONPATH=src:tests:../octo ../surval_openpi/.venv/bin/python -m unittest droid_pipeline_test.DroidLoaderContract -v; PYTHONPATH=src:tests .venv-droid/bin/python -m unittest droid_pipeline_test.DroidRuntimeContract -v; scripts/cache_droid_checkpoints.py --cpu --target-epoch 5 --max-demos 1 --max-rows 4 --num-cache-samples 1 --batch-size 2 --valid-num-steps 1; scripts/score_droid_cache.py --k-neighbors 5 --min-neighbors 2.
