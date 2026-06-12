"""State-conditional (local) threshold computation for SURVAL.

State features are the policy's own per-row feature output, stored as
``obs_features`` in the seqcache (a VLM prefix embedding for openpi/RLDS, or a
model encoder output for robomimic). See
README.md for the pipeline overview. The two entry-point scripts are
``scripts/build_state_db_from_cache.py`` (Phase 1: state DB from cache
``obs_features``) and ``scripts/compute_local_thresholds.py`` (Phase 2).
"""
