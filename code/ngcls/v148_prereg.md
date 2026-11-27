# V-148 pre-registration — V-147 re-run on a convergence test that works in float32, with bars that can move

Written and committed BEFORE the V-148 sweep, and before any V-148 seed exists. This supersedes nothing
in V-147's data, which stands and is reported; it replaces V-147's **instrument** and its **bar
placement**, both of which were mine and both of which were wrong.

## Why there is a re-run at all

V-147 ran the registered grid and `v147_score.py` was run verbatim. The reading it produced is **void**,
for three faults in what I committed:

1. **The exclusion rule voids the arm it was meant to protect.** The prereg says a cell whose
   iteration-cap rate exceeds 0.02 "is reported as truncated and excluded". The VSA arm hit the cap on
   0.13--0.50 of runs in every cell, so all 48 VSA cells were excluded and the grid printed `trunc`
   throughout. The cause is the convergence test, not the dynamics: both arms test
   `||x_{t+1} - x_t|| < 1e-9`, the localist arm runs in float64 and settles at a step norm of ~3e-17
   (seven orders below the test, so it always registers), and the VSA arm runs in float32 and plateaus at
   ~1.5e-8 (an order **above** the test, dipping under it only by chance). The flag separates the two
   dtypes, not two behaviours. Measured, not assumed: `tolcheck.py`.
2. **The cap was costing nothing.** The V-147 truncation control re-ran the transition cells with the cap
   raised ten-fold and recorded the read-out at t = 200 / 1,000 / 4,000 / 10,000 / 40,000:

   | cell | t=200 | t=1,000 | t=4,000 | t=10,000 | t=40,000 | dead |
   |---|---|---|---|---|---|---|
   | d=4096, n=6 | 0.67 | 0.70 | 0.70 | 0.70 | 0.70 | 0/30 |
   | d=4096, n=7 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0/30 |
   | d=4096, n=8 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0/30 |
   | d=65536, n=6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/30 |
   | d=65536, n=7 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0/30 |
   | d=65536, n=8 | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 | 0/30 |

   Every t=4,000 value equals the registered sweep's value for the same cell, no run died, and a ten-fold
   budget moves nothing.

   So V-147's excluded numbers are the same numbers a
   longer run gives; what is void is the scorer's verdict, not the measurement.
3. **The scorer applies the rule inconsistently, and the bars sit where they cannot move.**
   `v147_score.py` excludes cells from the table and from `n*` (P2) but computes P1 and P3 from those same
   excluded cells. And P1/P3 are anchored at "the largest `n`", which the grid puts deep in the saturated
   region (`n = 12`, where VSA is 0.00--0.03 at every `d`): P1 passes trivially at a gap of +1.00, and P3
   asks dimension to recover half a gap at a point where no dimension recovers anything. A bar that cannot
   fail and a bar that cannot pass are not bars.

## What is kept

The mechanism, the graphs, the scoring of a run (any valid isomorphism, not the planted permutation), the
localist gold standard, `p = 0.5`, `PYTHONHASHSEED=0`, and the attribution: the formulation is Gayler &
Levy (ANALOGY-2009) and the open problem is Kleyko's Survey Part II sec 3.1.2.3. **The contribution is the
failure boundary, measured.** No new method is claimed.

## What changes

- **Convergence is tested on the read-out, not on the iterate.** A run stops when the arg-max mapping has
  been unchanged over 200 consecutive iterations --- sampled every 25 iterations and required unchanged
  across 8 consecutive samples, since reading the mapping out of the GPU every iteration costs more than
  the iteration --- with a cap of 40,000. This is the quantity the score reads, it is
  meaningful in both dtypes, and the V-147 control shows the read-out is what settles. The fraction of runs
  reaching the cap is recorded per cell and **the same exclusion rule is applied to every bar or to none**.
- **Bars are anchored at `n = 8`**, fixed here and taken from the 10-seed PILOT (pilot: localist 0.95--1.00,
  VSA 0.20 at d=4096 and 0.30 at d=16384), not from the V-147 grid. `n = 8` is where the pilot put the
  transition, so it is where dimension has room to move the result in either direction.
- **Fresh seeds.** `200 + s`, not V-147's `100 + s`, so the bars are not tested on the runs that motivated
  them.

## Design

- `n in {4, 5, 6, 7, 8, 9, 10, 12}`, `p = 0.5`, planted isomorphism, `d in {2048, 4096, 8192, 16384, 32768,
  65536}`, **30 seeds per cell**, seeds `200 + s`. Localist replicator at every `n`, same graphs.
- No language model, no API, no training.

## Bars

**Q0 --- validity gate.** Localist >= 0.90 at every `n`. If it fails, nothing is read.

**Q1 --- there is a boundary.** At `n = 8`, `d = 4096`: localist minus VSA >= **0.30**, 95% bootstrap CI
over seeds excluding zero.

**Q2 --- dimension moves the boundary.** With `n*(d)` the largest `n` whose VSA success is >= 0.70:
`n*(65536) - n*(2048) >= 2`. Stated as a **difference in `n`**, not a ratio: V-147's ratio form asked
`n*` to double, which at `n* = 5` demands reaching `n = 10`, a size the pilot never suggested was in reach.

**Q3 --- the cause is superposition, not the algorithm.** At `n = 8`, raising `d` from 2048 to 65536
recovers at least **half** the gap between VSA and localist.

**Q4 --- descriptive, no bar.** Full grid; iterations-to-settled against `n` and `d`; `n^2` candidate
mappings at the boundary against this project's banked storage capacity (`k* = 32/64/128/256` at
`d = 512/1024/2048/4096`), to see whether the mapping boundary obeys the storage law.

## What I expect (recorded before the sweep, separately from the bars)

- Q0 passes.
- **Q1 holds.** V-147's own grid, read raw, gives localist 0.97 against VSA 0.07 at `n = 8`, `d = 4096`.
  This is the one I am confident in, and it is confident because V-147 already measured it -- which is
  precisely why it is the weakest of the four as evidence.
- **Q2 holds, narrowly.** Read raw, V-147 gives `n*(2048) = 5` and `n*(65536) = 7`: a difference of 2,
  exactly at the bar. A single cell moving either way decides it. I am registering a bar I expect to land
  on the line, and saying so in advance rather than after.
- **Q3 holds.** Raw V-147 at `n = 8`: localist 0.97, VSA 0.07 at d=2048 and 0.57 at d=65536 -- 56% of the
  0.90 gap. Also close to its bar.
- If Q2 and Q3 both fail, the limit is structural in the dynamics rather than superposition noise, and that
  is the outcome worth chasing.

## Scope

Random graphs are a proxy. Real analogical episodes carry typed, higher-order relations, and the classics
(Water Flow--Heat Flow, Solar System--Atom) were handled by Slipchenko & Rachkovskij (2009) with a
re-representation technique this sweep does not test. A held bar says where *this* mechanism stops on
*these* graphs. It says nothing about language models; the six preregistered runs on that question were all
negative and are not reopened.

## Protocol

`v148_score.py` frozen and committed with this file before the sweep runs, and run verbatim. The exclusion
rule is applied to every bar. V-147's data, scorer output and the faults above are kept and reported, not
replaced.
