# V-147 pre-registration — where the VSA replicator stops finding the mapping, and whether dimension buys it back

Written and committed BEFORE the registered sweep. A 10-seed pilot exists and is reported below as the
reason for the design; the bars are set against what that pilot leaves open, not against what it showed.

## What this is, and what it is not

This is **not** a new method. Gayler & Levy (ANALOGY-2009, "A distributed basis for analogical mapping")
pose analogical mapping as approximate graph isomorphism, convert it to Pelillo's maximum-clique
formulation, and solve it with replicator dynamics carried entirely in hypervectors:

    x  = (sum of source vertices) * (sum of target vertices)     all candidate mappings, ONE product
    w  = (source edge set) * (target edge set)                   all edge compatibilities, ONE product
    pi = x * w                                                   consistent mappings reinforce
    x <- normalise(x AND pi)                                     iterate to convergence

Kleyko's Survey Part II (sec 3.1.2.3) records the open problem in one line: the approach "was not
demonstrated to be scalable to large analogical episodes". **The contribution here is the failure boundary,
measured. The phenomenon is theirs and must be cited as theirs.**

## What the pilot already established, and why it is not enough

Both implementations were built and run (10 seeds, `p = 0.5`, scored as ANY valid isomorphism rather than
the planted permutation --- scoring against the planted one alone gave 0.00--0.15 and was my error, since a
graph with automorphisms has several correct answers):

| n | localist (gold standard) | VSA d=4096 | VSA d=16384 |
| ---: | ---: | ---: | ---: |
| 4 | 1.00 | 1.00 | 1.00 |
| 6 | 1.00 | 0.80 | 0.80 |
| 7 | 0.95--1.00 | 0.20 | 0.60 |
| 8 | 0.95--1.00 | 0.20 | 0.30 |

Iteration truncation is ruled out: raising the cap from 200 to 4000 changed no cell and no run hit the cap.

Three things make the pilot unusable as a result: 10 seeds per cell, two dimensions only, and no bars fixed
in advance. This prereg fixes all three.

## Design

- `n in {4, 5, 6, 7, 8, 9, 10, 12}` vertices, edge probability `p = 0.5`, planted isomorphism.
- `d in {2048, 4096, 8192, 16384, 32768, 65536}`.
- **30 seeds per cell**, seeds fixed as `100 + s`.
- The **localist replicator is run at every n** as the gold standard, on the same graphs.
- Success = the read-out mapping is a valid isomorphism of the two graphs. Convergence cap 4000, and the
  fraction of runs hitting it is recorded per cell; a cell above 0.02 is reported as truncated and excluded.
- `PYTHONHASHSEED=0`. No language model, no API, no training.

## Bars

**P0 --- validity gate.** The localist replicator reaches >= 0.90 at every `n`. It is the reference the VSA
arm is measured against; if it fails, the association graph is wrong and nothing is read.

**P1 --- there is a boundary at all.** At the largest `n`, at `d = 4096`: localist minus VSA >= **0.30**,
with the 95% bootstrap CI over seeds excluding zero.

**P2 --- dimension moves the boundary.** With `n*(d)` the largest `n` whose VSA success is still >= 0.70:
`n*(65536) / n*(2048) >= 2`.

**P3 --- the cause is superposition, not the algorithm.** At the largest `n`, raising `d` from 2048 to 65536
recovers at least **half** the gap between VSA and localist.

**P4 --- descriptive, no bar.** The full grid; iterations to convergence against `n` and `d`; the number of
candidate mappings `n^2` at the boundary, compared against this project's banked capacity results
(`k* = 32/64/128/256` facts at `d = 512/1024/2048/4096`) to see whether the mapping boundary obeys the same
law as the storage boundary.

## What I expect (recorded before the sweep, separately from the bars)

- P0 passes; the pilot already shows 0.95--1.00.
- **P1 holds.** The pilot's 0.20 against 0.95+ at n=8 is far past a 0.30 gap, and 30 seeds should not move it.
- **P2 is the one I am least sure of.** The pilot shows dimension helping at n=7 (0.20 -> 0.60) but barely at
  n=8 (0.20 -> 0.30). If capacity saturates in `d` the way this project's storage capacity saturated past
  4096, `n*` may move by less than the factor of two the bar asks, and P2 fails.
- **P3 holds if P2 does**, and the two failing together would say the limit is not superposition noise but
  something structural in the dynamics --- which would be the more interesting outcome and the one I would
  want to chase.

## Scope

Random graphs are a proxy. Real analogical episodes carry typed, higher-order relations
(`cause(bite, flee)`), and the classics --- Water Flow-Heat Flow, Solar System-Atom, Schools --- were handled
by Slipchenko & Rachkovskij (2009) with a different, re-representation technique that this sweep does not
test. A held bar here says where *this* mechanism stops on *these* graphs, and nothing about analogical
mapping in general. It also says nothing about language models; the six preregistered runs on that question
were all negative and are not reopened here.

## Protocol

Scorer `v147_score.py` frozen and committed with this file before the registered sweep is run, and run
verbatim. The pilot artifact is kept and labelled as a pilot. V-147 is a fresh ID: V-138 and V-139 were
already closed on 2026-09-17, V-145 and V-146 are this session's, and the highest previously used was V-144.
