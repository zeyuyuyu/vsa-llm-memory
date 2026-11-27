# B72 compute note 6 (2026-11-23, about 20:32 machine clock; no test output read)

- (B) development finished on the box at about 20:26 (4 x 288 jobs after the restart, 0 failures; 3,261 box dev jobs
  in all, 0 failures). Language-model (A) test finished locally (465 jobs, 0 failures, 20:13-20:30).
- **Bug fixed before any (B) choice or test** (procedure; no rule changed): in b72_run.py's (B) dev choice and (B) test
  plan, the position of prep in a configuration was taken as index 0 for every memory; for LAMC and NLMSC prep is at
  index 2 (their grids put scale and target first). The first full choice therefore stopped with "missing dev rows"
  (it looked for whitened LAMC rows at K = 1e3, which the plan correctly never ran) and wrote nothing; the plan would
  have produced no LAMC / NLMSC test job. Fixed (prep index 2 for LAMC / NLMSC), the box copy replaced atomically
  (md5 7760b416...), the full choice re-run by the registered rules: 252 choices per language model; the (A) / (D)
  entries are identical to step 1's committed ones on all four (checked key by key); no (B) cell is without a
  feasible configuration.
- (B) test: launched on the box from the committed choices (test_b.sh: plan, four per-encoder runners of 12 workers,
  then the (B) verdict, bits audit and packing of the small part files).
