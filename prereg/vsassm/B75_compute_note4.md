# B75 compute note 4 (00:56 machine clock, written before any test output is read; only job-completion counts seen)

- **Deadline cut.** The coordinator moved the deadline: the report is due within 20 minutes. Under the registered cut
  order, applied now before any test output is read: **H-SBC-J at x16 (launch R2b, 21 of 320 jobs done) is cut** and
  stopped by its exact PIDs; **(D) on the fresh stores (launch F, not started) is cut** (its chain branch is stopped so
  it never starts); **the determinism re-run is not run** (no time). R1 (U-LS, O2-BCH, O2-LDPC, U-J at every registered
  condition; 330 of 1,790 jobs done at 00:54) is allowed to finish; R2a (H-SBC-J at x1 and x8) finished at 00:50:04
  (530 jobs, rc 0). If R1 has not finished by about 01:08, the report says so and gives only R2a's results.
- Consequences fixed now: store-best-J at x16 is U-J alone; no fresh-store result; the determinism check is reported as
  not run.
