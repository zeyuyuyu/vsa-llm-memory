# B65 compute note 4 (2026-11-23; written after the test runs finished and BEFORE any test output is read)

- Dev choices committed 4593fea at 14:06:01; the test gate file TEST_OK was created on the box at 14:06:10 (after the
  commit); the chained queues then started the test evaluations.
- LAST: 9 registered test evaluations (3 arms x 3 seeds) and the 3 determinism re-runs (seed 0 of each arm, into
  preds_rerun/) ran 06:06:17-06:10:13Z on GPUs 1 and 7 (the workers that already held them), rc 0.
- OPS: 9 registered test evaluations ran 06:25:05-06:28:04Z on GPUs 2 and 5 (the workers that already held them; the other
  user's small processes were still on both), rc 0.
- No cut: all 18 registered evaluations ran, at both read positions, 3 seeds each; the time box was not exceeded at
  this point (14:29; registration 13:09).
- Nothing was re-run on the test split; no test file was opened, scored or summarised before this note (the lane
  logs, which print a running accuracy, were not read; only the queue command logs with return codes and the file
  listings were looked at).
- The seed-1 / 2 dev curves (best step only) are appended to the dev record in the same commit as this note.
- GPU time so far: 13.51 lane-hours summed over processes (logs/gpu_hours.txt: write-back training 12.45, readers 0.47,
  evaluations 0.55, caches 0.05) and 4.82 GPU-hours of claimed GPU wall time (logs/gpu_ledger.txt, six GPUs at most,
  up to four lanes sharing a GPU). All B65 workers have ended and released their GPUs.
