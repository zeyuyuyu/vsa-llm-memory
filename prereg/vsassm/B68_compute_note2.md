# B68 compute note 2 (2026-11-23 07:21 UTC; scope cut by the registered cut order; no bar, rule or prediction changed)

**Measured throughput.** At 07:20 UTC 99 of the 1,152 dev runs had finished, at a mean of 274 s per run with 42 lanes
(9.3 runs per minute), slower than the projection from B42's timings (B42 ran about 3 lanes per GPU; B68 runs 7). At
that rate, with B and C runs about 0.55 and 0.4 times the cost of A runs (B42's table), the full design would end its
last test run near 09:00 UTC and leave too little of the time box (record by 09:45 UTC) for the second scorer, the
sample reading and the notes.

**Cut (registered order: the reverse of the priority order, arena C first, Phi-2 before Qwen3-8B).** Five arena-C cells
are cut: phi2L2_C, phi2L4_C, phi2L0_C, qwen8L2_C, qwen8L4_C. Their 320 dev lines were removed from q/tune at 07:20:57 UTC
(q/tune.before_cut2 keeps the original); none of them had started, so they have no dev or test output and are
reported as "not run". qwen8L0_C (the last C cell in the cut order) stays: it tests the C-by-construction reading at the
input embeddings of one model. Arenas A and B run in full (12 cells), as P10 requires.

No test output of any cell exists at this time (q/final is empty; no dev choices have been committed).
