# B41 note after the test (written after the test verdict was read; nothing here changes a registered bar)

Test run: 2026-11-18 15:36-16:18 UTC (b41_test_pipeline.sh), once, seeds 1-3, 1,000 queries per (seed, N, rho),
3,000 per cell; the verdict uses the dev-chosen v* per cell (results/b41_dev_choice.json).

## Outcome

PRIMARY VERDICT: TABLE DOMINATES (results/b41_verdict.txt). No VSA-WIN in any of the 10 primary cells or any of the
45 cells under per-query ops, memory traffic or state size. In every cell every VSA configuration costs more per
query than every table configuration, and the best table is more accurate than the best VSA configuration
(A-gap 95% CIs exclude 0 in all 45 cells). C-ratio 2.5e-6 to 1.5e-5.

## Gates

- G0 (second implementation; results/b41_indep_summary.txt, results/b41_g0_ties.json): 45 runs, 891 (run, config)
  pairs, 373,905 (query, h) comparisons. Tables, EXACT and ENUM: answers agree 100% and every ops / traffic / state
  count matches (0 mismatches). VSA: costs match everywhere; answers agree >= 99% except 14 (run, config) pairs, all
  at N = 32 with kb = 1 (97.1-98.9%). Follow-up (b41_g0_ties.py): every one of the 115 disagreements has an EXACT
  score tie at or before the read-out hop (top-2, or the c-th/(c+1)-th score of the clean-up), where torch.topk
  (float32) and numpy's stable argsort (float64) break the tie differently; with kb = 1 and small D the clean-up
  scores are sums of +-1 terms, so ties among noise candidates are common. None of the 115 answers is correct in
  either implementation (all are noise trajectories), so no accuracy depends on the tie rule. G0 is read as passed
  with this explanation; the affected configurations are not v* in any cell.
- G1 (dev): PASS.
- G2: realised test per-mention argmax error 0.034 / 0.069-0.071 / 0.260-0.263 vs dev 0.036 / 0.071 / 0.262 (within
  0.03). PASS.
- Second scorer (b41_rescore.py): gold answers regenerated from the seeds equal the stored ones in all 45 runs;
  4,050 pooled cell-config accuracies recomputed, 0 mismatches.
- Hand-reading: results/b41_handread_notes.txt (table failures are near misses forced by confident misreadings;
  VSA failures are mostly crosstalk picks of unrelated entities).

## Predictions (results/b41_report.txt, mechanical)

Held: P2, P5, P7, P8 (primary), P9. Failed: P1 (dev error range starts at 0.036, not 0.02), P3 (EXACT - ARGMAX only
+0.09 / +0.12 at N = 100,000 and +0.34 / +0.36 at N = 32,000, below the +0.30 bar at 100,000 - dense graphs leave
less chain-breaking evidence), P4 (BEAM(4,16) within 0.01 of EXACT in 36/45 cells, not 40: at high noise the truth
falls outside a mention's top 4; BEAM(16, inf) is within 0.01 everywhere), P6 (most accurate VSA beats ARGMAX by
0.12-0.15 at N = 3,200, below 0.20), P10 (EXACT itself is 0.117 at N = 100,000, h = 6, high noise, so the gap is
0.112), P11 (no D on the grid recovers >= 90% of EXACT - ARGMAX at high noise, h = 6).

## Exploratory E1 (D = 2^18; registered before the test, not in the bar)

Quadrupling D to 262,144 raises the VSA by 1-10 points at N <= 3,200 and by up to 0.45 at N = 100,000 (low noise,
h = 2: 0.553 -> 0.847), but it stays below BEAM(4,16) and EXACT in every cell (best gap: low noise, h = 2,
N = 320: 0.928 vs 0.941) at 2.2e9-6.5e9 ops per query against about 4e3-7e3 for BEAM(4,16).
