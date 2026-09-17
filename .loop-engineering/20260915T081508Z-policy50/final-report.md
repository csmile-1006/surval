# Frozen epoch50 comparison: complete

R1–R5 satisfied. Reused the existing worker,score/threshold/OMN kernels and thin
launcher without modification. Extended only the existing comparison report,
focused tests,and matching runbook. Saved results under
`outputs/droid_policy50_val30_20260915/`; default25 unchanged.

Same full-val30 data,10checkpoints per apple/pan/pet,2880Ta/k/q settings,
half-chunk,LSE1,c1,S8,every-step exact neighbors. Reference/query both frozen
to each task's epoch50 EMA image/proprio conditioning. No model inference.

All-Ta macro results (NRegret,MMRV,Spearman):

- Common25 optimum: .076923,.061111,.755901.
- Common50 at25 HP: .076923,.064444,.741629.
- Common50 retuned: .076923,.062222,.752145 (Ta3,k300,q.2).
- Task-specific25 optimum:0,.044444,.817645.
- Task-specific50 at25 HP: .261111,.078889,.732894.
- Task-specific50 retuned:0,.051111,.801077.

Thus25 is better on SurVAL MMRV/Spearman in this matched finite-grid comparison;
both retuned variants attain the same regret. Requested7-Ta results also favor25.
Policy OMN50 improves MMRV/Spearman over25 but worsens regret; DINO remains higher
in Spearman. Non-reference baselines unchanged. This is posthoc analysis of
supplied real-world labels,not held-out validation of the reference epoch.

Validation:36 supported offline tests,3actual CPU jobs,3completed-worker resume
checks,8640 scalar HP metric rows,370 original-scorer points (maxerror8.756e-9,
thresholderror0,same metrics/selections),independent macro means and exact audit
coverage. All181 epoch25 files byte-identical. Private supervisor stopped after
both programs exited0. Exact commands and outcomes in test-results-1/2.json.

Optional unverified checks: ShellCheck absent; legacy pytest modules cannot run
without pytest. No dependencies installed merely to enable them. A test patch
initially required the zero-context git apply option; corrected and retested.
No product repair or failed runtime acceptance criterion remained.

Artifacts: `.loop-engineering/20260915T081508Z-policy50/`.
