# NSR0e pre-registration: confirm the derivation-voting table against V-delta on fresh seeds

Date: 2026-11-06. Written after NSR0d was scored; committed before any NSR0e run.

## Why

NSR0d's registered bar D1 passed: V-delta beat T-stage (the symbolic arm its tie rule selected) in L_p0.1_m1 and
F_p0.2_m1. The other registered symbolic arm, T-stage+vote (same parameters; it could not separate from T-stage on
validation because derivation voting cannot break a 1:1 tie at k <= 3, so it lost on grid order), scored above V-delta
in all 27 cells on seeds 20-29 (e.g. L_p0.1_m1 0.777 vs 0.653, L_p0.2_m1 0.621 vs 0.436, F_p0.2_m1 0.767 vs 0.581).
That comparison was reported but was not the registered bar. NSR0e makes it one, on new data.

## Design

- Arms, parameters fixed in advance (no tuning): **T-stage+vote** (s2 = 1, sc = 3, q = 0.5; NSR0d's validation
  choice); **V-delta-d** (tau = 0.3, alpha = 0.3; NSR0d's validation choice); **V-delta-0** (tau = 0.4, alpha = 0.3;
  NSR0's validation choice). Two V-delta settings so that neither tuning criterion is the reason V-delta loses.
- Cells: every NSR0 S, C, L and F cell; **fresh seeds 30-39**. Code: NSR0d's TableStage and derivation-voting CYK,
  NSR0's learning, sampling and noise, all unchanged.

## Bars

- **E1 (a VSA store beats the voting table):** either V-delta arm beats T-stage+vote by >= 0.02 overall in some cell,
  paired difference > 0 in >= 9 of 10 seeds, V-delta >= 0.50 there.
- **E2:** T-stage+vote >= each V-delta arm - 0.02 in every cell (no usefulness floor).

## Predictions

E1 fails and E2 holds in all 27 cells. If E1 passes anywhere, I report that cell as a regime where the vector store
helps.
