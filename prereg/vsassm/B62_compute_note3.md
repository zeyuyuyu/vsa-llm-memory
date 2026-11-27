# B62 compute note 3 (test finished; written before any test output is read and before the verdict is run)

Written 2026-11-23 about 03:09 (machine clock).

1. **Test run.** plan_test.jsonl (640 jobs, md5 5f9e2de3...) started on the box at 02:48:23, after the dev choices were
   committed (02:47:35, e5db971), with 48 workers; the last test job file was written at 03:08:03; 640 / 640 jobs, no failure or traceback in the log, the runner exited. The test was run once; no job was re-run.
2. **No cut.** The test ended within the 2-hour budget (02:19-04:19); every registered arm, encoder and cell was run.
   The order of the remaining steps is fixed now: pull the test jobs; checks that need no verdict (bits on all rows,
   float64 re-derivation with genuine primal recursions on the registered sample, started on the box at 03:08 with
   4 threads; it compares its predictions with the logged ones and computes no region statistic); then the verdict,
   the predictions, the plain-loop scorer and the sample reading. Test jobs pulled back to b62_out/jobs (640 files).
3. **Disclosed before the test (compute note 2):** Q6 and the "mu = 1" part of Q7 already fail on the committed dev
   choices. The bits audit was run on the 834,624 dev rows before the test finished (0 over budget, 0 not maximal;
   output to the scratchpad, not logged); it is re-run on all rows for the check file.
