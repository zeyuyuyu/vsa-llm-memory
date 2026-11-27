# B71 compute note 1 (during dev; no dev or test output of any B71 arm read)

Date: 2026-11-23, about 17:36 (machine clock).

1. **Pre-registration time.** The header says "about 18:10"; that was an estimate written before the commit and is wrong.
   The pre-registration was committed at 17:28:12 (417dab0) and its OpenTimestamps proof at 17:28:19 (00f4bd7), before
   any B71 development run; the commit times are the record. The time box (3 hours from 17:10) is unchanged.
2. **Placement of the dev runs.** The link to the box is slow (copying 11 code files and the 200 KB plan took about four
   minutes), and the test predictions must come back, so the registered dev plan (b71_out/plan_dev.jsonl, 1,230 jobs,
   md5 8d8b2aac...) is split by arm, without changing any job: VSAPR (600 jobs, the heavy feature maps; plan_dev_box.jsonl,
   md5 73b3da00..., made on the box by the same filter and md5-checked) on the box (~/b71_run, 48 workers, nice 5, one
   thread per process; started 17:35, pid 4021485), and NLMSF + VSAD with the audit (630 jobs; plan_dev_local.jsonl, md5
   a3d921d4...) locally (8 workers, nice 10; started 17:35). Every job is computed from registered seeds only, so its output
   does not depend on where or in which order it runs. Code on the box md5-identical to the repository: b32_common d6e1231b,
   b32_run bf5510f5, b36_common 39dc8029, b49_common f2228be9, b53_run 22fa3829, b56_run bbf61d39, b60_run 520af3db, b62_run
   ca210944, b66_lsfc 94114309, b71_run 787e65fa; panel codes copied from ~/b66_run's B49 copies, all 9 md5 ok.
3. The verdict and check code (b71_verdict.py, b71_check.py) are written during dev and committed with the dev choices,
   before any test run. No design, grid, seed or rule has changed.
