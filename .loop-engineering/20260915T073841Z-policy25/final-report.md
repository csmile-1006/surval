# Frozen epoch25 policy reference: complete

R1–R6 satisfied. One bounded repair: prevent an alternative reference epoch from
being mislabeled Policy25 in the comparison report. No original input, DINO
result, model, dependency, service, commit, or external storage mutation.

Implementation reuses StateDatabase, exact-neighbor expert quantiles, existing
half-chunk/survival reduction, OMN projection, metric definitions, and CPU runner.
The reference encoder is each task's epoch25 EMA conditioning (image + proprio,
512D x2 history), shared by all its checkpoint queries. No new policy inference.

35 offline regression tests passed. Three task jobs and all completed-worker
resume checks passed. 8640 HP metric rows checked against original scalar metrics;
310 selected checkpoint scores checked with original SurVAL: maximum error
1.056173055957732e-8, threshold error0, unchanged metrics/checkpoint choices.
Source/output hashes and prior DINO aggregate values verified. Private supervisor
shut down after successful sweep and report; CPU only. Exact commands/outcomes
are in test-results-1.json and test-results-2.json.

Optional checks not executed: ShellCheck absent; two legacy pytest modules failed
collection because pytest is absent. These remain explicitly unverified; no test
was weakened. Canonical full-data threshold/score checks cover the changed path.

Results: `outputs/droid_policy25_val30_20260915/REPORT.ko.md` and accompanying CSVs.
All-Ta common Policy25 optimum: Ta4,k400,q.2; macro NRegret .076923,
MMRV .061111,Spearman .755901. Task-specific retuning: NRegret0,MMRV .044444,
Spearman .817645. DINO task-specific control:0,.051111,.801077.
Matched Policy25 OMN Spearman .129357 vs DINO OMN .531050: not an improvement.

These are posthoc finite-grid results using supplied real-world outcomes, not
held-out generalization or statistically significant gains. Full legacy Loss
re-evaluation and image-only policy feature ablation remain outside this request.

Artifacts: `.loop-engineering/20260915T073841Z-policy25/`.
