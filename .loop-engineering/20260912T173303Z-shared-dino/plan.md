# Shared DINO representation and OMN

Reuse FrozenImageEncoder, state composition, StateDatabase, local thresholds, row coverage,
and the existing projection formula. Add one droid_dino module and a build_droid_dino_db CLI.
Build reads saved config JSON plus cache normalization, not policy weights; reuse finite Octo
loading at224px. Encode current images once, compose existing causal history windows,
and save a checkpoint-independent DB under <root>/<dataset_name> with physical expert chunks.
Pin DINO source revision and record a digest of frozen weights; same image policy across splits.
Validate source signatures/GT alignment and publish atomically; existing incompatible DB is an error.

Scorer accepts optional --dino-db-root and --dino-omn-k. Resolve DB per dataset, align by exact
source/timestep, reject missing/extra rows or changed GT/horizon/offset/action layout, and compute
thresholds once per shared DB/config into output/thresholds. Existing default unchanged.

Default DINO OMN: full-split kNN with self/same-demo temporal exclusion, nearest5 retained states.
For chunk offset h, project each sampled predicted action onto neighboring experts' actions at
the same h. Use cache-normalized10D coordinates, including gripper, all stored samples/horizon.
Do not expand duplicated state features over the horizon for neighbor retrieval. Reuse legacy
projection semantics including k=1, but explicitly report no-neighbor rows as unavailable, not zero.
Store per-sample values and retrieval protocol in JSON; add separate optional CSV scalar.
Legacy stored OMN/Loss/MSE and prediction cache bytes stay untouched.

Extract only the shared least-squares projection helper from cache_io; preserve old callers.
Optional DINO OMN column joins custom outcomes as another lower-is-better proxy; old CSV supported.

Checks: synthetic two-checkpoint shared-DB invariance, row permutation/mismatch, history/GT offset,
projection math/permutation/no-neighbor behavior, optional aggregate column, legacy regression tests.
Then build one real apple val5 DB and score all10 checkpoints without policy inference; check
common threshold path, finite metrics, independent cache signature before/after. Full12-split
DINO build is not required to demonstrate the new capability; provide complete CLI commands.
