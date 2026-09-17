# DROID loader and review — completed scoped handoff

R1 verified: parallel RLDS read/map/decode (default4), independent metadata/row-key manifest, deterministic source-key subset selection, canonical pre-inference ordering, strict per-sample feature/GT/prediction coverage and consumer-side key hash checks. Real626-row val5 contents match for1/4workers; actual12-row2-sample checkpoint predictions/features/Loss/OMN are bitwise identical.

R2 verified: see ../../docs/DROID_REVIEW.md (actual file /workspace/surval/docs/DROID_REVIEW.md). Checked6split manifests and3DROID runs/30local checkpoints. Openpi-specific blockers do not apply to current DROID execution. Methodological concerns remain, especially val-dependent calibration, batch-local repeated-horizon OMN neighbors, partial/unweighted validation and non-EMA/EMA mixing. No reference/baseline redefinition was made.

R3 verified for bounded execution: user authorized new environment. .venv-droid reuses base DROID packages read-only and installs local compatibility overrides; it is not a standalone clone. Original NumPy2.2.6 unchanged. Initial offline DistilBERT miss retained in test-results-1.json; downloaded required existing model artifacts and actual CPU checkpoint smoke succeeded.

72 distinct final tests passed (14DROID-specific,58existing regressions), no skips. Final Python3.10 compile and three repository diff checks passed. Actual output: /workspace/surval/outputs/droid_smoke_20260910/evaluation_multi/checkpoint_metrics.csv. Smoke12rows/2samples: SURVAL0.05517062358, Loss0.00863732956, OMN0.13738597929, MSE0.00349384826. This is not a full validation-set result or measured real-world correlation.

Other-session openpi step/joint-action changes were preserved. No commits, pushes, checkpoint remote operations, training or GPU jobs were performed. Normal Torchvision initialization downloaded ResNet50 to standard Torch cache; DistilBERT resides in the new venv model-cache.

Deferred: decide fixed train reference vs val self-calibration and explicit OMN definition; then full3task×5split×10checkpoint sweep plus measured custom outcomes. Exact runtime commands and failures are in test-results-1.json; follow docs/DROID.md for current commands.
