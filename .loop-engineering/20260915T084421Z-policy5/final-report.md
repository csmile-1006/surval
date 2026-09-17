# Frozen epoch5 comparison: complete

R1–R5 satisfied. Existing worker,score/threshold/OMN kernels,thin launcher and
default25 retained. Existing report/tests/runbook extended for5/25/50 with correct
labels and preserved output roots. No inference,training,dependency install,
external upload,commit or unrelated service mutation.

Same val30,3tasks x10checkpoints,2880Ta/k/q combinations;half-chunk,LSE1,c1,S8,
every-step exact neighbors. Reference/query use each task's epoch5 EMA fused
image/proprio conditioning,not target-checkpoint features.

All-Ta macro (NRegret,MMRV,Spearman):

- Common25 optimum: .076923,.061111,.755901.
- Common5 at25 HP: .387179,.090000,.665742.
- Common5 retuned: .076923,.063333,.729203 (Ta6,k300,q.35).
- Task-specific25 optimum:0,.044444,.817645.
- Task-specific5 at25 HP: .133333,.083333,.714738.
- Task-specific5 retuned:0,.042222,.923031.

Thus5 improves task-specific retuned MMRV/Spearman,while25 remains preferable
for common HP. Requested7-Ta task-specific5 result:.025641,.045556,.906720;
the Spearman/MMRV improvement there trades off higher regret versus25.
Policy5 OMN:.454701,.112222,.527208;Policy25:.421368,.194444,.129357.

Important tie caveat: pan's task-specific all-Ta maximum scores tie at epochs
30/35/40/50. Original np.argmax chooses the earliest,30,yielding NRegret0.
No tie-breaking or metric definition changed; Spearman uses average ranks.
This is disclosed in REPORT.ko.md and verification metadata. All numeric CSVs
were byte-identical before/after the documentation-only report rerun.

Validation:36 supported tests,3actual CPU jobs,3resume checks,8640scalar HP
metrics,400original-scorer points,maxerror1.1376310449229265e-8,threshold error0,
identical primary metrics/selected checkpoints. Independent artifact verifier
checks exact coverage,macro means,hashes,ties and all387prior25/50 files preserved.
Worker and both report runs exited0;private supervisor shut down.

Optional checks remain unverified: ShellCheck absent;legacy pytest modules need
absent pytest. No product failure/repair remained. These are posthoc finite-grid
and reference-epoch comparisons using supplied real-world outcomes,not held-out
generalization or statistically significant gains.

Results: `outputs/droid_policy5_val30_20260915/REPORT.ko.md`,HP/checkpoint CSVs.
Commands/evidence: test-results-1/2.json and verify_artifacts.py in this directory.
