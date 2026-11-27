# V-149 pre-registration — is the store result HRR's, or the VSA family's?

Written and committed BEFORE the sweep and before any V-149 seed exists, together with the frozen
`v149_score.py` and the sweep code `v149.py`.

## Why this experiment, and where it comes from

Every store number this project has banked comes from **one** model: HRR with FFT circular convolution,
the involution inverse, sum bundling and cosine cleanup. V-65, V-100, V-101, V-102, V-106, V-130 and
`store_rebase` all share that algebra. The npj manuscript's claim is about "a vector symbolic memory" —
a **family**-level claim carried by single-model evidence. That is the same shape of defect as the
`.974` re-base (a number from `v65.hrr_answer` under text describing `v102.sharded_two_hop`), and it was
caught there only because someone read the module against the sentence.

The gap is not mine to name. Kleyko, Rachkovskij, Osipov & Rahimi, *A Survey on Hyperdimensional
Computing aka Vector Symbolic Architectures, Part I* (ACM Computing Surveys 55(6), 2022), sec 4.1:

> "in general it is not clear whether there is one best model, which will dominate all others, or
> whether each model has its own applicability scope"

and the survey asks for exactly this: systematic experimental comparison across models on the same task
(it cites Kelly et al. 2013 and Schlegel et al. 2021 as the only prior attempts, both on intrinsic
properties rather than a downstream task). **The contribution here is a measurement, not a method.** No
new model is proposed and no model is being advocated.

Paper provenance: the two-part survey was recommended by the co-author (N. Imam) on 2026-09-25 as the
main reading for this project. Part II sec 3.1.2.3 is already the registered source of V-147/V-148's
open problem; this registers Part I sec 4.1 as the source of V-149's.

## Arms — four models, from Survey Part I sec 2.3, everything else held fixed

| arm | Survey | vectors | binding | unbinding | bundling | similarity |
|---|---|---|---|---|---|---|
| HRR | 2.3.3 | real, normalised Gaussian | circular convolution (FFT) | involution | sum | cosine |
| MAP | 2.3.7 | bipolar {-1,+1} | Hadamard product | self-inverse | sum (MAP-I, unclipped) | cosine |
| BSC | 2.3.6 | binary {0,1} | XOR | self-inverse | **majority rule** | 1 - normalised Hamming |
| FHRR | 2.3.5 | unit-modulus complex | elementwise product | conjugate | sum | normalised Re<.,.> |

Held fixed across arms: the rows, the gold triples, the `blake2b` hash-to-codebook identity, the
200,003-row codebook, the pooling, the shard rule, the two-hop procedure, the cleanup-by-argmax
protocol, the seeds and `PYTHONHASHSEED=0`. **Only the four algebra functions change.**

Protocol choices recorded rather than defaulted: MAP bundling is the unclipped sum (MAP-I), not MAP-C;
BSC majority ties on an even stack are broken by a fair coin drawn from the run's own seeded generator.

## Design

- **M1, capacity law by model.** V-101's protocol verbatim except the algebra: 2WikiMultihopQA validation
  slice of 400 (all 400 usable), `n = 200` instances, pool sizes `k in {32,64,128,256,512}`,
  `d in {512,1024,2048,4096}`, seeds `{0,1,2}`, codebook seed `1000+s`, pooling rng `7000+s`, one
  unsharded store, two hops, exact match. `k*(d)` = the largest `k` whose accuracy is `>= 0.70`
  (V-101's level, not a new one).
- **M2, task readout by model.** `store_rebase`'s protocol verbatim except the algebra: the 300 banked
  rows (`rows300.json`), `d = 2048`, shard at 64, seeds `{42,123,456}`, per-row hits banked.
- No language model, no API, no training, no fine-tuning.

## Bars

**P0 — validity gate, both halves. If either fails, nothing below is read.**
- P0a: the HRR arm reproduces V-101 at `d=2048`: `k=128 -> 0.7417` and `k=256 -> 0.2417`, each within
  `+/- 0.05`.
- P0b: the HRR arm reproduces `store_rebase`'s mean EM `0.9778` within `+/- 0.01`.

**P1 — the claim is family-level.** At least **3 of the 4** models reach mean EM `>= 0.90` on M2. If
fewer do, the manuscript's "a vector symbolic memory" must be narrowed to the models that clear it.

**P2 — the models are not interchangeable.** The spread `max - min` of M2 mean EM across the four arms
is `>= 0.05`, with a 95% bootstrap CI over the 300 rows excluding zero. If the spread is under five
points, the choice of model is **not** a lever on this task and the paper should say so.

P1 and P2 pull against each other on purpose: P1 asks the arms to agree, P2 asks them to differ. Both
holding means a real but bounded model effect; P1 holding with P2 failing means the algebra is not a
design decision here; P1 failing means the family claim was too wide.

**P3 — the capacity law is a family property.** Fit `k*(d) = c * d^alpha` per model over
`d in {512..4096}`. Every model's `alpha` lies in `[0.85, 1.15]`. HRR's banked law is
`k* = 32/64/128/256`, i.e. `alpha = 1.0`. If any model falls outside the band, the manuscript's
"capacity linear in d" must be scoped to the models that obey it.

**P4 — descriptive, no bar.** Per-model seconds per query and arithmetic class, tabulated with the
accuracy. BSC is the only integer-only arm (XOR + popcount, no multiply anywhere), which is the corner
that Yilmaz, *ArrowFlow* (arXiv 2604.04087, also recommended 2026-09-25) argues is worth paying accuracy
for. This sweep does not run ArrowFlow and makes no claim about it; it only reports what the
integer-only corner of **our own** store costs on **our own** task, so that the argument can be checked
here rather than imported.

## What I expect, recorded before the sweep and separately from the bars

- **P0 passes.** The HRR path is a transcription of the banked one; if it does not reproduce, the
  transcription is wrong and that is what the gate is for.
- **P1 holds.** I expect HRR, MAP and FHRR all above 0.90. BSC is the arm I would bet against: its
  bundling is majority rule, which discards magnitude, and Kleyko, Gayler & Osipov's commentary on the
  Science Robotics paper (arXiv 2003.11458 sec IV.B — the third paper recommended on 2026-09-25) warns
  specifically that XOR-bound members of a set cancel when their vectors are similar. Our object vectors
  are random and near-orthogonal, so that warning may not bite; I am recording that I think it will not,
  and that BSC's loss will come from majority bundling instead.
- **P2 FAILS, and this is the prediction I am least comfortable with.** I expect HRR, MAP and FHRR
  within about two points of each other, so the spread will be decided entirely by BSC. If BSC clears
  0.90 the spread is probably under five points and P2 fails. I am registering a bar I expect to miss
  because the informative outcome is "the algebra does not matter here", and that outcome is only worth
  anything if the bar could have caught the opposite.
- **P3 holds for HRR, MAP and FHRR; BSC is where I expect the deviation**, because majority-rule
  bundling has a different capacity law from sum bundling — the case the Kleyko/Gayler/Osipov commentary
  works out in closed form in its sec V, after correcting the Science Robotics paper's own version of it.
- If P1 holds and P2 fails, the reading is: *on this task the result is a property of the family and not
  of HRR, and choosing among dense VSA models is not a lever.* That is a negative, and it is the one
  that most changes what the manuscript is allowed to say.

## Scope

One task (2WikiMultihopQA two-hop over gold triples), one identity scheme (hash to a fixed codebook),
one cleanup rule (argmax over the candidate set). Four models out of the eleven Part I catalogues —
the sparse families (SBDR, Sparse Block Codes) and the matrix binders (MBAT, VTB) are **not** tested and
nothing here speaks for them. A held bar says what these four do on this task; it says nothing about
which model is better in general, which is the question the survey says is open and which one task
cannot close.

## Protocol

`v149_prereg.md`, `v149.py` and the frozen `v149_score.py` are committed together before the sweep runs,
and the scorer is run verbatim on `v149_logs/v149.json`. `PYTHONHASHSEED=0`. Imported modules are
checksummed into the output. If the scorer has no branch for the outcome that occurs, the missing
reading is written in the ledger, not by editing the frozen file.
