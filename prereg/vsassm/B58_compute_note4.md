# B58 compute note 4 (22:11 machine clock; test running, no test output read - only job counts and memory)

- The i.i.d. test of the protected store (700 jobs, 9 workers) completes about 2.3 jobs per minute on the ten
  6,000-pair test stores; with the mid-size and small stores and the extension passes it will not finish before
  about 23:15-23:30. The E-LDPC re-score (1,108 jobs, 6 workers) runs beside it.
- Per compute note 2, the 16 damage conditions of part (a) had to finish their window and extensions by 23:20. They
  cannot start before the i.i.d. test ends, so **they are not run** (part (a) damage = not run; H3-P / H4-P, Q5, Q6 =
  not evaluated). This is decided now, before any test output is read. The i.i.d. part (a) and part (c) (re-score of
  the 16 Fig. 5b conditions and the 7 i.i.d. conditions) run to completion as registered, even past the time budget.
- The driver script b58_run.sh was briefly edited while two copies were running and restored byte-identical to the
  committed version within a minute (bash reads scripts incrementally); the geometry-ordered damage option went to a
  separate copy, b58_run2.sh, which will not be used.
- Memory: 3-5 GB free with 15 workers (test workers 0.6-1.4 GB on the large stores); no worker has failed.
