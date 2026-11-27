# VSA1 verdict (2026-11-15): a resonator network replaces the solver's search only on the smallest van problems
Prereg: prereg/VSA1_prereg.md (commit 0587e89). Items: data/vsa1_items.jsonl (65 fresh vans items, seeds 9300+,
sha256 4f0fe348, gold valid on all 65; commit after the prereg). Runner: vsa/test.py; scores: out/vsa/test_score.json.

## Outcome (one CPU core, 120 s per item, the same symmetry fix in both arms)
| level | parcels | factors | log10 search space | resonator V | solver C | C median s |
|---|---|---|---|---|---|---|
| 1 | 6 | 12-18 | 8-16 | **10/10** (median 0.4 s) | 10/10 | 0.01 |
| 2 | 9 | 24-32 | 18-27 | **10/10** (median 33 s) | 10/10 | 0.02 |
| 3 | 13 | 32-42 | 22-34 | 0/10 | 10/10 | 0.03 |
| 4 | 18 | 52-64 | 37-49 | 0/10 | 10/10 | 0.05 |
| 5 | 24 | 86-94 | 57-65 | 0/10 | 10/10 | 0.11 |
| 6 | 50 | 157-167 | 109-117 | 0/5 | 5/5 | 0.9 |
| 7 | 80 | 314-328 | 205-214 | 0/5 | 5/5 | 11 |
| 8 | 120 | 534-594 | 328-362 | 0/5 | 5/5 | 28 |

Validity gate G (subset sum in Kymn et al.'s regime, standard resonator): 26/40 (n = 10, 20, 30, 40: 9, 8, 5, 4).

## Bars
- G: **PASS** (26 >= 24). The implementation reproduces the published subset-sum behaviour.
- B1 (parity at level 1): **PASS**, 10 vs 10.
- B2 (C - V >= 20 over levels 2-5): **PASS**, 30.
- B3 (against the expectation): not triggered (V 0 at every level from 3 up).

## Predictions against outcomes
- V >= 8/10 at level 1: held (10). V >= 5/10 at level 2: held (10). V <= 3/10 at level 3: held (0). V 0 at 5-8: held.
- C 10/10 at levels 1-5: held. C "some failures at 6-8": **missed**. With the symmetry fix and 120 s, CP-SAT on one
  core solved all 15 (in dev, without the fix and with 60 s, it had solved 0 of 5 burned level-8 items).

## Checks from other angles
- Exploratory (not registered; vsa/explore_budget.py, out/vsa/explore_budget.jsonl): the 10 level-3 items with 5x the
  time (600 s) solved 0/10 at D = 10,000 and 0/10 at D = 20,000. The failure at 13 parcels is a limit of the search,
  not of the time budget.
- The encoding is exact: the gold loading reproduces the target on all NSR8 vans items (vsa/selftest.py), and every
  resonator success was checked by the family's own rules.
- The resonator settings were the best of four literature variants on burned items; without noise (dev1) it solved 0
  of 6 burned level-2 items, so the stochastic update matters.

## Literature check
- **Supports:** Frady et al. and Kent et al. (2020): the operational capacity of a resonator grows roughly with the
  square of the dimension for random codebooks (the constant was not checked here). Our successes reach search spaces of 10^27
  at level 2, helped by the structured (phase-code) codebooks, and stop between 10^22 and 10^34. Kymn et al. (2025)
  solved subset sum, a single sum constraint, and compared only with an exponential exact algorithm.
- **Conflicts / limits:** Langenegger et al. (2023) report up to five orders of magnitude more capacity from a
  stochastic in-memory resonator with winner-take-all; we used software noise without winner-take-all, so hardware
  resonators may reach somewhat larger problems. Even five orders of magnitude would not bridge levels 4-8 (10^37 to
  10^362).
- **Not found:** any VSA/HDC solver for bin packing, scheduling or multi-constraint CSPs.

## What this means
- On the one primitive a vector symbolic search could supply, finding a solution, a resonator network matches the
  solver at six and nine parcels and fails from 13 parcels up, while the solver finds all 65 loadings within seconds.
- The search is incomplete: it cannot prove that no loading exists or that one is optimal or unique, which the
  engine's certificates need. So VSA cannot currently take the solver's place in the engine; at most it could serve
  small find-steps, or run on hardware where its speed and energy matter more than its reach.
- For the paper: the open question becomes a measured answer (Discussion + Supplementary Note).
