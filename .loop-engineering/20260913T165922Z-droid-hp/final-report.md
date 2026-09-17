# DROID HP tuning: complete

R1–R5 satisfied. The initial3584-HP grid and one predeclared extension to5184
were evaluated on apple/pan/pet xval5/10/20/30 x10 checkpoints, every_step=True.
No prediction cache or shared DINO DB was changed; no policy inference rerun.

Task balanced optima: apple Ta7/k20/q.25/x2; pan Ta1/k20/q.9/x1;
pet Ta6/k10/q.25/x.125. Shared minimax: Ta8/k10/q.1/x.25;
shared macro-balanced: Ta7/k10/q.1/x.25. These are exploratory finite-grid
in-sample optima, not held-out generalization results.

Implementation: opt-in exact-k retrieval in shared threshold functions;
scale multiplier andexact-k flags inDROID scorer; cached CPU grid module;
reusable launcher/report/verifier andfocused tests. Existing helpers reused.
No dependencies, GPU allocation, external upload, or unrelated edits required.

Validation:38 tests;12+12 successful workers;1180 canonical score comparisons
(maximum absolute error2.69e-8,allprimarymetrics/selectionunchanged);
62208 independent metric rows exactly agree;initial430080 scores bitwise
unchanged;120 sourcecache SHA256 unchanged;ordinaryCLI10checkpointreproduction.
Commands andoutputs: test-results-1.json. Requirement-by-requirement audit:
verification-1.json. No repair wasnecessary. ShellCheck unavailable; noted.

User deliverables: outputs/droid_hp_every_step_extended_20260913/REPORT.ko.md,
summary.json,selections.json,selected_cell_metrics.csv,all_*_metrics.csv,
checkpoint_scores.csv.gz,pareto_common.csv,canonical_audit.csv,verification.json.
Reproduction: scripts/tune_droid_hp.md. Initialoutputs preserved at
outputs/droid_hp_every_step_20260913. Dedicated supervisors are shut down
after their own worker/report processes have exited; shared instance services untouched.
