# B65 compute note 2 (2026-11-23, 13:25; during dev, no test run exists)

- **Another user's jobs arrived on our GPUs.** At about 05:20Z another user started five small processes
  (imagenet_int8_eval_batched.py, 1.1-2.1 GB each) on GPUs 1, 2, 4, 5 (which our first-grid workers had claimed while
  idle at 05:12-05:18Z) and on GPU 6 (idle until then; it was the GPU the third OPS worker, RWo3, would have claimed).
  None of their processes was touched. RWo3 claimed GPU 6 only at 05:23:44Z (briefly idle), found the OPS queue
  already empty and ended at once, so the OPS grid ran on two GPUs, not three.
- **Out of memory on the OPS grid.** With four of our lanes (about 19.5 GB each) plus the other user's 1.3 GB on GPUs 2 and 5,
  six OPS write-back runs (all VSA: vsabm 3e-4, 3e-3, 1e-2; vsapub 1e-3, 3e-3, 1e-2) died at their first training step
  (torch.OutOfMemoryError, rc 1, 05:18:59-05:21:31Z), before any dev number beyond step 0 and before any train.json
  was written. These six runs are re-run once, unchanged (same command lines, same seed), by appending them to the OPS
  grid queue; they are taken by the surviving lanes of the two OPS workers on the GPUs those workers already hold, at
  three lanes per GPU (3 x 19.5 + 1.3 GB fits in 79 GB). This is one re-run of jobs that never trained, not a retry
  loop and not a selection: the dev rule reads only complete train.json files. The LAST grid (GPUs 1, 4, 7) had no
  failure.
- From here on every worker runs at most three lanes per GPU (b65_orch.sh already used three for the dev2 and test
  phases), and the claim patience of new workers is raised from about 15 minutes to about 2 hours (b65_worker.sh;
  procedure only). New workers still start only on GPUs idle by the registered rule (< 1 GB, no compute process), so
  GPUs where the other user's processes run are not claimed anew.
- The OPS dev2 waiter: my first waiter was stopped by its PID and replaced (b65_wait_dev2.sh now takes the worker
  count); a second one with count 2 was also stopped by PID before it could fire, because RWo3 had already logged its
  end, and replaced by one waiting for all three OPS workers' ends. The LAST waiter is unchanged.
- Time box: the registration was at 13:09; the cut order of the pre-registration (first seed 2 of OPS, then OPS) is
  applied, if needed, in a later compute note written before any test output is read.
