# B66 compute note 2 (during dev; no test output of any B66 arm read)

Date: 2026-11-23, 13:45 (machine clock).

- **(C) draws (dev rows only):** b66_grids.py draws ran 13:43-13:44 on the existing dev rows; its choices and the list
  of needed test runs are committed with this note (results/b66_C_choices.json, results/b66_C_need.json); test plan
  b66_out/grids/plan_test.jsonl md5 a718a567, 935 jobs (LAMC 345, SDMT 200, KF6 200, KF6IDX 190).
- **Cut (registered rule: within (C), KF6 first), decided now, before any test output is read:** the KF6 and KF6IDX
  test jobs are NOT run. Reasons: B36's five families cost about 5 CPU-hours here (B36's Setup per worker, 22-350 s per
  test store), more than the local machine has inside the time box next to (A) and (B); and the index family needs
  faiss, which exists only in the box's b49 environment, whose PQ / OPQ fit caches are partial (a refit there may not
  match B32's saved fits, and verifying that is not possible in the time box). Consequence: (C) reports STORE (its 72
  configurations are its full grid), LAMC and SDMT, and the matched-grid contrasts STORE - LAMC72, VSAR - SDMT72,
  VSA - SDMT72; KF6's draws, BEST7_72, VSA - BEST7_72 and VSAR - BEST7_72 are not reported.
- **Local load / one incident:** the local machine reached load 50 (16 cores) with the clean-room dev processes, LSfc
  dev, (D)'s analysis (started 13:42, output not read) and the SDMTCR test runner. The SDMTCR test runner (local, 2
  workers, 0 jobs written) was stopped and restarted on the box with the same plan (12 workers); my `pkill` pattern also
  matched its own shell (the shell died, no other process was affected; ps afterwards showed no runner).
