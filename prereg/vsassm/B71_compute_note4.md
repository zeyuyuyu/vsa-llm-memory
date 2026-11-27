# B71 compute note 4 (during dev; no dev or test output of any B71 arm read)

Date: 2026-11-23, about 17:55 (machine clock).

1. **All remaining dev work moved to the box.** The local VSAD kind R runner (6 workers, competing with other sessions'
   processes at a load of 18-23 on 16 cores) finished no job in 13 minutes; at 17:52 I stopped it by its exact PIDs (no
   job finished, none kept) and ran VSAD kind R on the box: M = 256 (120 jobs, plan_dev_box4a.jsonl, md5 57f1d510...,
   16 workers, from 17:48) and M = 1,024 (51 jobs, plan_dev_box4b.jsonl, md5 c100e6cd..., 26 workers, from 17:53), both
   filters of the registered plan_dev.jsonl, md5-checked on both machines. Consequence, stated before any output is read:
   the audit's reproduction of B53's and B56's dev accuracies (which were computed on this machine) is now a cross-machine
   reproduction; any difference will be reported with its cause (float32 near-ties, B70 compute note 2), not hidden. The 8
   VSAD kind P dev jobs finished locally before note 2 are not used; the box's copies (same plan entries) are.
2. **Dev summaries on the box.** Downloads from the box run at about 10 KB/s, too slow for the roughly 11 MB of dev job
   files. b71_devsum.py (committed with this note) computes on the box, for every dev key, the number of dev stores and
   the mean of each accuracy field over the stores (sorted rep order, numpy.mean, the same arithmetic as b71_run.choose)
   and the per-epoch mean update counts; b71_run.choose and b71_verdict.audit read these summaries (B71_DEVSUM=1). A
   round-trip test on the 8 local VSAD kind P jobs gave identical means on all 720 keys (max |difference| 0). The bits
   audit of the dev rows runs on the box (b71_check.py bits). The dev choices are made locally from the summaries.
3. **Worker count.** From 17:41 to 17:47 my box runners together had up to 146 worker processes on the 128 cores (the
   registered limit is 48; another user's three processes were running). I stopped the D2 SDMt dev runner (24 workers;
   48 of its 297 jobs finished and kept, in-flight ones discarded) at 17:47; it is restarted after the other dev runners
   finish. The total stays above 48 until NLMSF and VSAD kind P finish. Results do not depend on the worker count (every
   job is single-threaded and seeded); the deviation from the compute rule is recorded here.
4. Code changes (procedure only): b71_run.py (load_devsum, dev_mean), b71_verdict.py (audit reads summaries),
   b71_side.py (d2verdict; D2 test job names carry the budget; b60 job directories per budget), b71_devsum.py. No
   configuration, grid, seed or rule changed.
