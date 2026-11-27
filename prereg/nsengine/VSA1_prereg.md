# VSA1 prereg (2026-11-15): can a vector symbolic search take the constraint solver's place?
Written after development on burned items only (out/vsa/dev1.jsonl, dev2.jsonl) and before any test item exists.

## Why
The VSA paper (rewrite r15, Discussion) now reports the solver experiment and ends on an open question: resonator
networks search over combinations of codes held in superposition; can such a search take the solver's place? The
user asked (11-15) to research that question instead of leaving it open.

## What the literature already shows (checked 11-15)
- Resonator networks factor a bound product into codebook entries (Frady et al. 2020, Kent et al. 2020, Neural
  Computation 32(12)); their operational capacity grows roughly with the square of the dimension, about two orders of
  magnitude above the optimisation methods they compared.
- Kymn et al. (Neural Computation 37(1), 2025) code integers with residue phase codes, so that binding adds them, and
  solve subset sum with a resonator: one binary factor per element, target = code of the sum. They compare only with
  an exponential exact algorithm (faster for more than 28 elements), not with a dynamic program or a CP/SAT solver.
- Variants: self-attention updates (Yeung et al. 2024, arXiv 2403.13218); stochastic resonators with device noise and
  winner-take-all (Langenegger et al. 2023, Nature Nanotechnology 18:479), up to five orders of magnitude more
  capacity.
- Higher-order oscillator Ising machines solve k-SAT (Bybee et al. 2023, Nature Communications 14:6033, same group),
  but they are Ising machines, not VSA algebra.
- No work was found that applies VSA/HDC to bin packing, scheduling, rostering or general multi-constraint CSPs.

## A limit that holds before any experiment
A resonator is an incomplete (Las Vegas) search: it can find a solution and have it checked, but it cannot prove that
no solution exists, that an answer is the only one, or that it is optimal. The engine's certification needs exactly
these proofs (unique, optimize, all_opt, impossible with a MUS). A vector symbolic search could therefore replace at
most the solver's "find" step. This study tests that step.

## Task
Vans (bench/heldout/vans.py), the family in which the engine abstained most: is there a loading into k* vans, where
k* is the family's own optimum? Every test item is feasible at k*, so success = the arm returns a loading that the
family's rules accept (vsa/resonator.py vans_valid, the same rules as check()).

## Encoding (vsa/resonator.py, self-test vsa/selftest.py: the gold loading reproduces the target on 30/30 NSR8 items)
FHRR phasors, D = 10,000. Channels: weight of van v and spaces of van v (residue phase codes, moduli m-1, m, m+1 with
(m-1)^3 above the largest partial sum, weights in units of 10 kg), and a counter mod 3 for every apart pair and van.
One factor per group of must-travel-together parcels (codebook = one entry per van, binding its weight, its spaces
and its apart counters); one slack factor per van for weight (0..W) and for spaces (0..S); one slack factor (0 or 1)
per apart pair and van. Target: every van's load plus slack equals its limit, every apart counter plus slack equals 1.
A choice reaches the target iff the loading is valid. Symmetry: the heaviest group is fixed to van 1.

## Arms (each item, one CPU core, 120 s wall clock)
- **V (resonator):** the dev winner "att10n": self-attention updates (softmax over similarities, beta = 10) with
  Gaussian noise of std 0.05 on the similarities (a stochastic resonator), restarts of 50 sweeps until the time limit.
  Dev (burned NSR8 vans, 120 s, 14 jobs in parallel on 16 cores): level 1 6/6, level 2 6/6, level 3 1/6. Others:
  att20 6/1/1, att20n 6/5/0, stdn 6/5/1; dev1 without noise and with 200-sweep restarts: 5/0/0 at best.
- **C (solver):** the same feasibility problem in CPMpy, OR-Tools CP-SAT, one worker, the same symmetry fix.

## Validity gate (G)
Fresh subset-sum instances in Kymn et al.'s regime (integers 0..5000, target = sum of a random half), n = 10, 20,
30, 40, 10 each, D = 10,000, standard resonator, at most 10 restarts of 100 sweeps. G passes if at least 24 of 40 are
solved (dev: 15/20). If G fails, my resonator is not a faithful implementation and V is not interpreted.

## Test items
Fresh vans items from the family's generator, seeds 9300-9309 at levels 1-5 (10 per level) and 9300-9304 at levels
6-8 (5 per level): 65 items, generated after this file is committed. Sizes: 6, 9, 13, 18, 24, 50, 80, 120 parcels.

## Predictions
- V: at least 8 of 10 at level 1; at least 5 of 10 at level 2; at most 3 of 10 at level 3; 0 at levels 5-8.
- C: 10 of 10 at every level 1-5; at levels 6-8 some failures (dev: 0 of 5 burned level-8 items in 60 s).

## Bars
- B1 (VSA can replace the solver's find step at small sizes): V >= C - 1 at level 1. PASS = parity at six parcels.
- B2 (the gap with size): over levels 2-5, C - V >= 20 (of 40).
- B3 (against my expectation): if V solves at least 5 of 10 at level 3 or higher, or more than C at levels 6-8, the
  conclusion "a resonator cannot yet take the solver's place beyond the smallest problems" is wrong and will not be
  written.

## Disclosure
- The resonator's settings were chosen on burned NSR8 vans items (levels 1-3) among four variants from the literature
  (dev2). The solver arm was not tuned.
- Merging together-groups is the family solver's own preprocessing; the CP-SAT arm states together/apart directly.
