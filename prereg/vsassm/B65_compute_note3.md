# B65 compute note 3 (2026-11-23, 13:33; during dev, no test run exists)

- Because the other user's processes now sit on GPUs 1, 2, 4, 5 and 6, a NEW worker could claim only GPU 7 under the
  registered idle rule once the first grids end, which would have serialised everything that follows on one GPU and
  broken the 3-hour time box. Instead the workers that already hold GPUs 1, 2, 4, 5, 7 (claimed while idle at
  05:12-05:18Z) keep their lanes: b65_chain.py lines were appended to their live queues (one "step" and five "hold"
  lines per position; q/R_wb_last, q/R_wb_ops). A step waits until the position's grid is complete, runs the
  registered rule (b65_plan.py W, under the choices lock), appends any edge extensions (and a further step), then the
  seed-1/2 readers, then the seed-1/2 write-backs, and finally, only after a file TEST_OK exists (created after the dev
  choices are copied and committed), that position's 9 test evaluations (and, for LAST, the 3 determinism re-runs).
  Holders block a lane until the next generation is released, so at most six lanes per position stay alive. No new GPU
  is claimed and no other user's process is touched; we keep using GPUs we already held when their jobs arrived.
- The two dev2 waiters (b65_wait_dev2.sh) were stopped by their PIDs before they fired, so dev2 is not also run by
  b65_orch.sh (it would have re-run the seed readers and write-backs).
- b65_chain.py was exercised on fabricated dev files in a scratch directory (grid complete -> an edge extension ->
  interior -> F -> readers -> write-backs -> TEST_OK -> test lines); the dev rules themselves are b65_plan.py's,
  unchanged (md5 2b7ca5fd...).
