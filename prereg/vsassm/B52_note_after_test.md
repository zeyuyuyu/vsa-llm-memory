# B52 note after the test runs (2026-11-20; no design change, no number replaced)

Scope actually run: (b) and (c) in full; (a) only the primary cell qwen8L27 x C (compute notes 2-4; the other three cells
are "not run"). Verdicts: results/b52b_verdict.txt, results/b52c_verdict.txt, results/b52a_verdict.txt.

## (b) R1 across random maps: ROBUST

- Per map (VSA re-selected on dev per map, BEST = B36's choice with every random part re-seeded): R1 VSA - BEST +8.13
  [+7.20, +9.07], +8.33 [+7.41, +9.28], +8.59 [+7.62, +9.56] points; mean +8.35, between-map SD 0.23; two-level bootstrap
  (maps x entity clusters) [+7.47, +9.23]. Against B36's original BEST predictions: +8.31 / +8.55 / +8.50.
- B36's published map is the lowest of four (+6.77; SD 0.81 over four maps, range +6.77 to +8.59). The gap is a
  selection effect, not the map: B32's committed dev choices at M = 256, V = 2, 4 and 8 bits per fact (centre, linear,
  q = 2 / 4) test at 60.1 / 60.7, while every new map's dev choice there (mostly whiten, q = 1-2) tests at 63.6-70.0.
  BEST moves by at most 0.3 points between maps (38.55-38.85 R1 mean).
- Gates: G-b1 12 / 12 dev choices equal B32's; G-b2 per-query agreement 1.000 with B36's test predictions (VSA and BEST,
  all R1 bpf, rep 0 of each (M, V)). Checks: numpy float64 re-derivation of the VSA reads on 10% of test stores 1.000
  (23,856 queries); re-scoring |difference| 1.4e-14 points. Hand reading (M 1,024, V 2, 4 bits per fact, the cell with the
  largest between-map range, 61.6-64.9): the queries that flip between maps are near-ties of the two value scores, at
  alias-label cosines from 0.06 to 0.62; nothing systematic.
- Predictions: PB1-PB6 all HELD.

## (c) One-day device rates: ranking unchanged

- TABLE WIN at every noise multiplier at one day, as at 30 days; best arms identical per condition (E-BCH at k1-k4 and
  e05, E-LDPC at k8-k32; V-last best VSA except V-Snt at k8). Best table vs best VSA read, cells per stored pair (30 days
  in parentheses): k1 19 vs 76 (19 vs 76), k2 19 vs 76 (19 vs 76), k4 23 vs 76 (27 vs 76), e05 38 vs 91 (38 vs 91),
  k8 54 vs 108 (54 vs 128), k16 181 vs 304 (215 vs 431), k32 609 vs 1,448 (724 vs 1,722); Delta -8 / -8 / -7 / -5 / -4 /
  -3 / -5 grid steps (30 days: -8 / -8 / -6 / -5 / -5 / -4 / -5). The ratio narrows slightly at k8 and k16 (2.0x and 1.7x
  at one day vs 2.4x and 2.0x at 30 days).
- No extension pass was needed (every N* bracketed by the first test window). N*(1 day) <= N*(30 days) for every arm and
  condition (first verdict pass flagged 13 spurious violations: it compared the exact grid budget, e.g. 76.1, with the
  30-day table's printed 76; fixed to compare grid steps before anything was concluded). k1 (rate 0 at both horizons)
  equals B38's 30-day N* for every arm.
- Predictions: PC1-PC4 HELD; PC5 (closed-form flags) see the closed-form line in results/b52c_verdict.txt (if absent, the
  check had not finished by the report deadline).
- PC5 NOT HELD: the closed-form check flags 37 of 445 cells (registered: at most 10). All 37 are E-BCH / E-CAM with the
  simulation ABOVE the closed form (the registered closed form counts a key as failed whenever one of its codewords
  fails, a lower bound for bounded-distance decoding; B38 had 108 of 953 flags for the same reason). 0 flags for the VSA
  reads and T-rep (310 cells); near the 99.5% point (336 cells with closed form >= 95% or simulation >= 99%) 4 flags,
  largest |difference| 3.1 points. The N* comparison is unaffected.

## (a) qwen8L27 x C (the only cell run; qwen8L0 x C, qwen8L9 x A, qwen8L0 x A NOT RUN, compute notes 2-4)

- Naming head at layer 27: 0.998 (B31 explore1). Every label-free arm FAILS: vq 0.481 [0.451, 0.512] (R -0.06), dkvb
  0.485 (R -0.06), gumbel 0.516 [0.483, 0.548] (R +0.01 [-0.12, +0.12], the best), st 0.457 (R -0.11), cw 0.505 (R
  -0.01), bind+pc 0.491 (R -0.04), bind+pz 0.509 (R -0.01); the binding-loss map itself 0.491. Answer for this cell: NO
  label-free interface reaches the naming head read from the same layer. Every arm is 0.48-0.54 below that naming head.
- Geometry: centring / whitening remove the common-mode cross-talk (cw different/same cosine 0.016 / 0.006; the plain
  binding-loss map 0.87 / 0.90) but leave the codes inconsistent across mentions (vq mention consistency 0.16-0.24,
  gumbel 0.37-0.39); clean-up correct at hop 1 in only 3-14% of instances for every arm. Hand reading (cw, seed 42, 12
  failures): 10 of 12 fail at the hop-1 clean-up (the query role and the same role in the context do not get matching
  codes), 2 at hop 2 - the same pattern B42 found at the final layer (23 of 24).
- Gate G1 PASS (gumbel and st with the identity loss reach dev 1.000 = naming head). Second scorer (b42_indep.py):
  per-instance agreement 1.000 for vq, dkvb, gumbel, st (all seeds); every accuracy re-derived against dictionary-walk
  labels. RECOMPUTED != LOGGED for vq (seed 42: 0.462 vs 0.470) and gumbel (0.510 / 0.530 vs 0.482 / 0.470): ties at the
  threshold with 6-decimal scores, as in B42; recomputed numbers reported; FAIL under either.
- Predictions: PA1 HELD for C (A not run), PA3 HELD, PA7 HELD; PA2, PA4, PA5, PA6 not tested (cells not run).

## Compute (ledgers: box logs/cpu_ledger.txt, logs/c1d_ledger.txt; b52_out/compute_ledger.txt)

- A800 box CPU, part (a): 10 lanes x 6 threads 03:27-05:00 UTC and 4 lanes x 6 threads 04:05-05:00 UTC (about 115
  allocated core-hours, nice 10, shared with other users' load); scope cut at 03:44 UTC (logs/cpu_ledger.txt "refocus";
  the compute notes' approximate clock times are superseded by the ledger). Input-embedding extraction about 1 core-minute.
- A800 box CPU, part (c): 24 workers 03:29-04:57 UTC (about 35 core-hours).
- Local 16-core CPU, part (b): about 1.3 hours wall (about 20 core-hours).
- GPU: 0 GPU-hours (the idle-GPU worker never found an idle GPU; GPU 3 never touched). PACE: 0 (paid tier unused). No
  paid API.

## Failed predictions

PC5 (closed-form flags, lower-bound direction only). Not tested: PA2, PA4, PA5, PA6.
