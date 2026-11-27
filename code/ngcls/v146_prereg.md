# V-146 pre-registration — VSA as inference, not storage: analogical mapping under load and under competition

Written and committed BEFORE any instance is generated.

## Why this axis, after five negatives

V-133 (depth), V-134 (knowledge location), V-135/V-137 (volume) and V-145 (cost) each fixed bars in advance
and each came back negative. Read together they say something narrower than "the algebra does not help":
**every one of them tested VSA as storage** --- bind facts in, unbind them out. Retrieval over supplied
evidence is exactly what a long-context reader is good at, so five negatives on that axis are not a
surprise in hindsight.

The literature check (standing rule: check after a conclusion) returns the same shape from outside this
project: AMA-Bench reports that agent-memory techniques beat long-context baselines on dialogue benchmarks
and **fall short of those baselines on long-horizon agentic tasks**. Our result is not an outlier.

What none of the five touched is the operation Kanerva (2009) actually points at and the one the VSA
literature still calls open: **holistic transformation** --- learn a mapping from correspondences and apply
it to a new episode in one vector operation, with no search and no training. Kleyko's Survey Part II names
two open problems verbatim, and a search in September 2026 finds them still open:

1. the approach "was not demonstrated to be **scalable to large analogical episodes**";
2. current models "**lack interaction and competition of consistent alternative mappings**", improvable
   "by using an approach involving the associative memory".

Those are a **capacity** question and a **cleanup-competition** question. They are the two things this
project has built instruments for, and the contribution would be in the measurement, not the phenomenon ---
the phenomenon is a mature subfield and must not be presented as new.

**The blocker recorded against this direction is now gone.** It was that our store bound
`hash(subject)` to `hash(object)` and discarded the relation, so there was no role--filler structure to map
over. V-133's store binds `(subject (x) relation) -> object`; the structure exists.

## Instrument

An analogical episode is a source domain and a target domain with the same relational structure and disjoint
vocabularies. With `n` corresponding entity pairs, the mapping is one vector:

`M = SUM_i vec(source_i) (x) vec(target_i)`

and a correspondence is queried holistically as `cleanup(M (x) vec(source_j)) ?= target_j`, with cleanup over
the target vocabulary. No training, no search, one bind per query. Relations carry over from the V-133 store.

Swept: `n in {4, 8, 16, 32, 64, 128, 256}` episode size, `d in {1024, 2048, 4096, 8192}`, three codebook
seeds. No language model anywhere, so this is cheap and fully deterministic.

**Competition condition.** A second mapping `M2` is built over a fraction `rho in {0, .25, .5, .75}` of the
same source entities pointing to *different* targets, and the two are superposed. Both are internally
consistent; only one is intended. This is open problem 2 stated as a measurement.

## Bars

**P0 --- validity gate.** At `n = 4`, `d = 4096`, `rho = 0`, mapping accuracy >= 0.95. If one bind cannot
carry four correspondences, the instrument is wrong and nothing is read.

**P1 --- the capacity question (open problem 1).** The episode size `n*` at which mapping accuracy crosses
0.70 grows **at least linearly in `d`** over 1024 to 8192, i.e. `n*(8192) / n*(1024) >= 6`. Retrieval
capacity in this project followed `k* = Theta(d/tau)` and saturated past 4096; whether *mapping* obeys the
same law is exactly what "not demonstrated to be scalable" leaves open.

**P2 --- the competition question (open problem 2).** At `rho = 0.5`, `d = 4096`, `n = 16`, the intended
mapping is selected on at least **0.60** of queries. Chance between two consistent mappings is 0.50, so this
asks whether superposed mappings do better than a coin flip at all.

**P3 --- does one quantity govern both regimes?** The cross-talk tolerance measured for *retrieval* in this
project is 0.099 (V-95). Imposing cross-talk by hand on the mapping's codebook, the accuracy of holistic
mapping crosses its own floor within **0.03** of that same 0.099. A single tolerance governing storage and
inference would be the result worth having; two different tolerances would be worth having too, and would
say the two operations are not interchangeable.

**P4 --- descriptive, no bar.** The full `n x d` grid; mapping accuracy against the number of *relations*
rather than entities; and whether errors concentrate on entities that occupy both roles (the degree effect
measured in V-133's diagnostics, where readout ran .632 to .957 at constant `k` and `d`).

## What I expect (recorded before the data, separately from the bars)

- P0 passes.
- **P1 holds but saturates**, as retrieval capacity did past `d = 4096`. If it saturates below the 6x the bar
  asks, P1 fails and the honest report is that mapping inherits storage's ceiling.
- **P2 is the one I expect to FAIL.** The survey says these models *lack* competition among alternative
  mappings; a superposition of two consistent mappings most likely returns whichever has more energy, not the
  intended one, which would land near chance. I would rather register that and be wrong.
- **P3 I genuinely do not know**, which is why it is worth running. It is the only bar here whose outcome I
  cannot guess, and either answer is publishable.

## Scope

Synthetic domains, one VSA, no language model. This measures the algebra's own inference operation. It says
nothing about whether an LLM benefits from it --- five runs now say it does not benefit on retrieval, and
that question is not reopened here.

## Protocol

Scorer `v139_score.py` frozen and committed with this file before any data exist, run verbatim.
`PYTHONHASHSEED=0`. No bare `set` iteration. V-146 is a fresh ID.
