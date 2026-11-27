# B71 compute note 3 (during dev; no dev or test output of any B71 arm read)

Date: 2026-11-23, about 17:50 (machine clock).

**A job-name collision in the NLMSF dev plan (procedure only).** The plan groups each store's R1 budgets by the store's
chosen (prep, fmap, stored share); NLMSF does not use the share, and its job name omits it, so where the final design's
or VSAPc's chosen share differs between budgets of one (M, V) with the same (prep, fmap), two NLMSF specs had the same
name and only one of them could be written (18 names: qwen8 NLMSFP M = 1,024 V = 16, mistral NLMSFP M = 256 V = 16, e5
NLMSFR M = 1,024 V = 16, bge NLMSFP M = 256 V = 16; found by listing the plan's names, not from any output). Remedy, fixed
before any dev output is read: (i) b71_run.job_name appends an optional tag; (ii) a supplementary plan
(b71_out/plan_dev_fix.jsonl, md5 d0220892..., 18 jobs, tag 'u') re-runs those stores with the union of the colliding
specs' budgets and the registered grid, on the box; (iii) load_dev accepts a dev row computed twice only if its accuracy
is identical (|difference| < 1e-12; same machine, same seeds) and raises otherwise; (iv) the TEST plan groups NLMSF jobs by
(prep, fmap) only, so no test name collides. No configuration, seed, grid or rule changed. b71_run.py md5 after the fix
8590f141....
