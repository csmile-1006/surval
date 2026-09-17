# DROID local validation handoff

Follow-up scope update: the user restored baseline aggregation after this
handoff. It is now implemented for all seven baselines and SURVAL, using explicit
custom-outcome CSV keys. See `followup-baselines-and-loader-check.json` and
`docs/DROID.md` for 66 passing checks (one optional skip), the RLDS loader
investigation and exact commands. Production shuffle settings remain unchanged;
the actual-policy environment blocker below still applies. The remainder of
this report preserves the original handoff status.

Implementation is present across surval, droid_policy_learning and Octo. Runtime
verification of the actual policy is pending a compatible dependency environment;
the overall request is not yet fully verified.

Implemented:

- Surval-owned cache and scoring commands, documented in `docs/DROID.md`.
- Octo source episode IDs/frame indices; DROID GT slicing aligned with the
  trained action[1:] then To-1 convention (observation t -> action t+1).
- Checkpoint normalization, atomic/provenance-checked epoch caches, inference
  encoder features (EMA when present), deterministic evaluation language and CPU paths.
- Per-checkpoint policy-feature local threshold artifacts using existing FAISS
  and threshold code; physical action units for SURVAL, normalized MSE/OMN.
- Library continuous score delegation and checkpoint JSON/CSV. Outcome and
  cross-run aggregation deferred as requested.

Verification: six cache/score contracts pass in the existing openpi environment,
including real FAISS threshold construction and synthetic checkpoint JSON/CSV.
The existing focused suite passed 36 tests with one optional sklearn-dependent
skip; the added k>N regression then passed with all four database tests. These
represent 43 distinct passing test cases across the final relevant files.
Real local RLDS val1 data preserves one source ID and 116 original frame indices
through chunking/flattening. Ten changed/new Python files compile on Python 3.10;
all three git diff --check commands pass. Exact commands/outcomes are in
`test-results-1.json` and `test-results-2.json`.

One product repair was needed: the missing-threshold-row guard used a wrong
diagnostic field. The guard now uses LocalThreshold_MissingRows and its failing
test passes. CPU checkpoint loading/DataParallel and training-language mode were
also checked during that repair. No tests were weakened.

Remaining gap: /venv/droid_policy has NumPy 2.2.6 with TensorFlow 2.15 and fails
import with _ARRAY_API / numpy.core.umath errors. The existing openpi environment
supports CPU surval/Octo checks but has incompatible newer diffusers and lacks
tensorflow-graphics for this policy. No dependencies were installed, no existing
environments changed, and no GPU jobs interrupted. No real policy checkpoint
cache or real checkpoint score/baseline output has been generated.

Next step after installation authorization: prepare an isolated surval-local
DROID-compatible environment, run DroidRuntimeContract, then cache one local
epoch with a bounded validation subset and score it. Retain any runtime failures
and continue the remaining two repair iterations if needed.

Artifact directory:
`/workspace/surval/.loop-engineering/20260910T045951Z-droid-local-validation/`

Permission boundary comes from the explicitly requested skill:
`/root/.codex/skills/loop-engineering/SKILL.md`: "Do not install dependencies
merely to make a check available unless the user authorized installation."
An asynchronous installation question remains unanswered at this handoff.
