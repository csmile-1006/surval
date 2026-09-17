# c=1 long-chunk search

1. Reuse existing robomimic producer --cache-ta15 (Tp16,To2). No policy source
   edits. Smoke batch32/rows32/sample8 first; inspect available GPU capacity,
   never stop unrelated training. Run three val30 workers under ownsupervisor,
   tenepochs each,seed0,batch32,skipstandardvalidation (SURVAL-onlyrequest).
2. Use generated val30 caches directly (30 checkpoint files, three datasets).
   Check stable row keys, fullGT first8prefix,15steps,8samples,all10epochs.
   Historical baselines remain explicitly historical. Publish three new DINO
   DBs with unchanged embeddings/firstactions and longGT; no DINO inference.
   User scope update: never generate or tune val5/10/20.
3. Keep prior HP production module/runner unchanged so oldprovenance remains
   replayable. Add focused long-HP module reusingdistance/scaletable/rowgroup
   helpers. c=1assertion; actualtop-k count replacesduplicatefractionaxis.
   Scores reuse rawchunk errors; LSE tau varies, every-step/product unchanged.
4. Reuse existing queue/manifest helpers in thinlongrunner andBash/Markdown.
   Produce161280HPx3val30tasksx10scores, atmost2CPUworkers. Store everycheckpoint
   score andvectorized primarymetrics. NoGPUduringHPscoring,nonewdependency.
5. Reuse metric/selection definitions; efficientfrontier ratherthan quadratic
   all-grid scan. Report percell/task/common andshort-only/long-only optima.
   Audit publishedchoices with canonical score_policy_cache using matching
   top-fraction andLSEtau,c1; verify quantiles exactly andrankmetrics unchanged.
6. Independently check coverage,sourcehashes,c1,allmetricrows,selectionoptimality,
   literalH>8samples,canonical agreement. Write Koreanresults andreproduction.

Tests: synthetic15-step top-k/LSE equivalence includingtiny/large tau; c!=1
rejection; HDF5val30identity/GT/rowchecks; dryrun/shards,nooutputs; real
smokecache andnativeprefixcheck; completedworkersresume; finalcanonicalaudit.
No upload/training/model-extrapolation. If denseCPUsearch is slow, keep the
fullreviewedobjective andsupervisedprocess,not a silent smallergrid.
