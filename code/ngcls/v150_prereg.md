# V-150 pre-registration — does this project's cross-talk instrument predict analogical mapping, or stop at storage?

Written and committed BEFORE the V-150 sweep and before any V-150 seed exists.

## Why this experiment, and whose idea it is

The advisor (N.I.) pointed at **Kanerva 2009, Cogn Comput 1:139--159**, saying it shows VSA does *reasoning*
and not only memory. Read in full 2026-09-04 (ledger 125b29c). Two of its three demonstrations are already
ruled out as targets: the "Dollar of Mexico" is deflated by Kleyko Survey Part II sec 3.1.2 as a *simple
proportional analogy* of the word2vec class and must never be led with (V-146 was invalidated for testing
exactly that), and the one-shot grandmother rule needs role--filler structure that our store does not keep,
since it binds `hash(subject)` to `hash(object)` and **discards the relation** -- a store change, not an
experiment.

What does connect is the part of Kanerva 2009 that is qualitative: he names two failure modes, a
**cleanup-selection** failure (a tautological term in the antecedent correlates more highly than the intended
conclusion) and **superposition cross-talk** (holistic substitution destroys the roles where records agreed).
This project has an instrument for the second one -- the cosine between the vectors different symbols feed to
the store, whose threshold **0.0989** (V-95) orders every arena row in the manuscript. The manuscript's
strongest structural claim is that one measured quantity predicts every arm.

**This registration asks whether that claim extends from storage to mapping, or stops.** It is a scope test of
our own instrument, against the reasoning task the advisor's pointer identifies.

## The reason to doubt it, stated before the run

V-148 (closed, ccaaa1d) measured the mapping boundary across `d`. Candidate mappings at the boundary,
`n*^2`, went 25 / 25 / 36 / 36 / 36 / 49 across a 32-fold increase in `d`: an exponent of **0.19**. This
project's banked storage capacity `k* = 32/64/128/256` at `d = 512/1024/2048/4096` is **linear in `d`**. A
pure accumulated-cross-talk argument predicts an exponent of **0.5** (interference per pair ~ `1/sqrt(d)`,
load ~ `n^2`). The measured 0.19 matches neither. So the honest prior is that cross-talk does **not** govern
mapping, and this registration is expected to return a negative.

## Definitions, fixed here

- `tau` -- the geometric cross-talk, defined to match V-95: the mean of `|cos(P_a, P_b)|` over distinct
  candidate-mapping hypervectors `P_a = S_i * T_j`. Measured per run, not assumed.
- `Lambda = n^2 * tau` -- the load-weighted cross-talk: number of superposed candidate mappings times the
  interference each pair contributes. This is the quantity an accumulated-noise account says should govern
  the boundary.
- Success -- unchanged from V-147/V-148: the read-out mapping is a valid isomorphism of the two graphs.

## Design

`n in {4,5,6,7,8,9,10,12}`, `d in {2048,4096,8192,16384,32768,65536}`, `p = 0.5`, planted isomorphism,
**30 seeds per cell**, seeds `300 + s` (fresh; V-147 used 100+, V-148 used 200+). Convergence as in V-148:
arg-max mapping unchanged over 200 iterations, sampled every 25, cap 40,000; cap rate recorded per cell and
the exclusion rule applied to every bar or to none. Localist gold standard at every `n`. `PYTHONHASHSEED=0`.
No language model, no API, no training.

## Bars

**R0 --- validity gate.** Localist >= 0.90 at every `n`, and the VSA success grid reproduces V-148 within a
95% bootstrap CI at `n = 8, d = 4096` and `n = 8, d = 65536`. If the grid does not reproduce on fresh seeds,
nothing is read.

**R1 --- does one quantity order the grid?** Pooling every run across every cell, `Lambda` as a single
scalar classifier of per-run success reaches **AUC >= 0.90**. That is the bar the manuscript's arena claim
meets, so it is the bar the extension must meet.

**R2 --- is it the same threshold?** The success/failure crossover in `Lambda` (the value at which pooled
success passes 0.5) lies within **[0.0791, 0.1187]**, i.e. 0.0989 +/- 20%. R2 is read only if R1 holds; if
one quantity does not order the grid, asking where its threshold sits is meaningless.

**R3 --- the exponent, on fresh seeds.** Fitting `n*^2 ~ d^alpha` by least squares on log-log, the 95%
bootstrap CI for `alpha` **excludes 0.5**. This confirms or refutes, on data that did not motivate it, the
V-148 finding that the boundary follows neither the storage law (1.0) nor the cross-talk law (0.5).

**R4 --- descriptive, no bar.** Of a candidate-invariant family fixed here and not extended after seeing the
data --- `n^2/sqrt(d)`, `n^2/d`, `n^3/sqrt(d)`, `n^2/d^0.19`, `n/sqrt(d)`, `log(n)/log(d)` --- report the
coefficient of variation of each at the boundary, and name the most constant. Descriptive because choosing an
invariant by fit and then calling it a result is the fishing this project does not do; a winner here is a
hypothesis for a later registration, not a finding.

## What I expect (recorded before the sweep, separately from the bars)

- R0 passes.
- **R1 FAILS.** `tau` is ~`1/sqrt(d)` and independent of `n`, so `Lambda` is nearly `n^2/sqrt(d)`; at the
  measured boundary that quantity takes 0.552 at `d=2048` and 0.191 at `d=65536`, a factor of 2.9 apart. A
  quantity that moves threefold along the boundary it is supposed to mark will not reach AUC 0.90.
- **R2 not read**, since R1 is expected to fail.
- **R3 holds**, with `alpha` near 0.19 and well below 0.5.
- The outcome I expect is therefore: **the manuscript's one-quantity claim is scoped to storage and does not
  extend to analogical mapping.** That is a limitation of our own instrument, found by testing it where the
  advisor's pointer sends us, and it is worth more than another confirmation would be. If R1 unexpectedly
  holds, the instrument unifies storage and reasoning and that is a much stronger result -- which is why the
  bar is set at the manuscript's own 0.90 rather than somewhere easier.

## Scope

Random graphs at `p = 0.5`, not typed higher-order analogical episodes; the classics were handled by
Slipchenko & Rachkovskij (2009) with a re-representation technique this sweep does not test. The mechanism is
Gayler & Levy (ANALOGY-2009); the open problem is Kleyko Survey Part II sec 3.1.2.3. No new method is
claimed. Nothing here enters the npj submission, whose six preregistered language-model runs were all
negative and are not reopened; this is next-paper material.

## Protocol

`v150_score.py` frozen and committed with this file before the sweep runs, and run verbatim. R2 is gated on
R1 inside the scorer, so a threshold cannot be quoted from a quantity that failed to order the grid.
