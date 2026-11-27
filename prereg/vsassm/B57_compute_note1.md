# B57 compute note 1 (during dev; no test store built)

- dev1 ran 17:16-17:58 (machine clock); queue parallelism raised from 4 to 5 (17:19) and to 7 (17:40) as memory
  allowed. All 98 jobs returned rc = 0; every MF grid file holds its 18 dev stores.
- dev2 (MF choices, then the LM / RT / LR engines) was started at 17:55, while the last four dev1 jobs (minilm lamm 3;
  e5 / bge / minilm NLMS dev) were still running; dev2 reads only the finished MF grid files. Its parallelism was
  lowered from 5 to 4 at 18:00 (qwen8 M = 1,024 engine jobs; 6.9 GB available, briefly 2.5 GB).
- No scope cut was needed so far (the test phase is expected to start before 20:00, the pre-registered cut time).
- Gate run on B49's existing dev results (no B57 query): b57_run.lam_choice reproduces B49's LAM choice in 112 of 112
  (encoder, cell, accounting) cases for qwen8 and e5, so the per-map LAM re-selection uses B49's rule.
- Nothing about design, grids, seeds or rules changed. I looked at the dev mean accuracy of the VSAP matched-filter
  choices (dev data) before the engine runs; no choice was altered.
- dev2 ran 17:55-18:18, parallelism raised back to 6 at 18:05; all 64 jobs rc = 0; choices written 18:19. The test phase starts before 20:00: no scope cut.
