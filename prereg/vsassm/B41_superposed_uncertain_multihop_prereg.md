# B41 pre-registration: computing in superposition under uncertain naming - a VSA against table arms on multi-hop composition, accuracy and cost

Date: 2026-11-18, about 15:30 UTC (10:30 US Eastern; box clock Asia/Shanghai). Written, with all B41 code
(b41_names.py, b41_feats.py, b41_head.py, b41_data.py, b41_engine.py, b41_indep.py, b41_verdict.py), before any run
on a test question. What exists at registration: the train and dev mention banks and their Qwen3-8B features (no
test bank exists; it is generated after the dev record is committed, with the registered seeds below), a naming head
being fitted on train with dev for the layer/epoch choice, and a CPU smoke run of the engine and of the second
implementation on MOCK logits (random numbers, not the LLM; used only to debug; numbers not reported). No engine run
on real naming distributions has been made. Timing note: the naming head's dev statistics (layer 9, T = 0.927, per-rho dev argmax error 0.036 / 0.071 / 0.115 / 0.187 / 0.262 / 0.329 / 0.402 at rho .03-.3) were printed minutes before this commit; no line of this file was changed after they were read (P1 below was written before).

## Why

The co-author's question for the paper: what does a VSA give an LLM that a table cannot? One candidate is computing in
superposition. When the LLM's reading of a symbol is uncertain (a distribution over candidate names), a VSA can bundle
the candidates weighted by that distribution and carry the bundle through several binding/unbinding steps in a vector
of fixed width, cleaning up only at the end (or per hop). A table that commits to the argmax at every hop compounds
reading errors; a table that enumerates candidate paths pays k^h. But a table can also carry the distribution: exact
probability propagation by sparse products over the stored facts (the forward algorithm / TensorLog form), or a beam
over entity states with merging, whose cost is about k per hop, not k^h. B41 measures accuracy and cost of all of
these on one task with real LLM naming uncertainty.

Prior (written down before any real-data run, so it can fail): a back-of-envelope analysis says the tables should win
on cost everywhere. (i) The VSA's crosstalk noise per hop grows with the number of facts in a memory, so D must grow
roughly linearly with facts per memory; (ii) without clean-up, noise compounds geometrically over hops (the smoke run
with noise-free readings agrees: clean-up at the end fails at h = 4 even at D = 4,096 with ~12 facts per memory);
(iii) clean-up costs E x D per hop (E = 4,096 codebook rows), which alone exceeds a beam's whole cost by orders of
magnitude; (iv) the exact sum-product is what the VSA computes in superposition, plus crosstalk, so the VSA's accuracy
should not exceed the exact arm's beyond noise. The literature check (results/b41_literature.txt, 28 verified items;
14 support, 5 conflict, 9 gap) agrees: VSA cost wins in the literature are against exhaustive enumeration of a product
space or against committing to one path, not against per-hop marginalisation; no paper runs this comparison with
uncertain symbol readings. If the analysis holds, B41's value for the paper is a bounded negative: what computing in
superposition buys over an argmax table, and what it costs relative to a probability table.

## Naming distributions (real LLM uncertainty)

- Entities: E = 4,096 names generated in b41_names.py from fixed syllable lists and numpy seeds (no machine files):
  1,024 clusters of four confusable names (a base name and three variants that differ by one surname letter).
  Relations: 32 fixed relation words, read exactly (relation uncertainty is not tested).
- Mentions: the name passed through a character typo model with rate rho (per character, prob. rho of one of
  substitution / deletion / insertion / transposition). Banks (deterministic; seeds [41, split, entity, rho, j]):
  train 32 variants per entity (rho in {0, .03, .06, .1, .15, .2, .25, .3}, four each); dev 3 per entity per rho in
  {.03, .06, .1, .15, .2, .25, .3}; test 8 per entity per registered test rho (made after the dev record).
- Reader: frozen Qwen3-8B (the B31 encoder), mention in the carrier "Note: {name} attended the meeting.", span mean of
  hidden_states[L] over the name's tokens (offset mapping); L in {9, 18, 27, 36}. Naming head = linear 4,096-way head
  (B31's recipe: identity labels, cross-entropy, AdamW 1e-3, wd 1e-4, batch 2,048, up to 30 epochs; features
  standardised with train statistics); the epoch per layer and the layer are chosen by pooled dev NLL; one temperature
  T is fitted on the pooled dev bank (NLL). Naming distribution of a mention = softmax(logits / T) over all 4,096.
- Noise levels: three test rho levels chosen on dev from the dev grid, the rho whose dev per-mention argmax error is
  nearest to 0.02 (low), 0.08 (mid) and 0.25 (high) (distinct rhos). The realised test error is reported per cell.

## Task

b41_data.py. A context holds N facts (s, r, o), functional in (s, r), N in {32, 320, 3,200, 32,000, 100,000}. Q query
chains of 6 hops are planted first (Q = max(1, min(1000, floor(N / 18)))); with probability 1/2 a decoy chain is
planted alongside, starting at another member of the start's confusable cluster and passing through other members of
the clusters of every intermediate entity, with the same relations (a misreading at any hop can jump to a parallel
chain with a different end). The remaining facts are random distractors. Queries of h in {2, 4, 6} hops are the first
h relations of a chain; the answer is e_h. Every mention (the query's start and both arguments of every fact) is an
independent draw of a bank variant of the true entity at the cell's rho; the same graph and draws are used at every
rho (rho levels are paired). Cell = (rho level, h, N): 45 cells. Test: seeds 1, 2, 3, 1,000 queries per seed per
(N, rho) (3,000 per cell). Dev: seed 101, 500 queries. Context counts follow from Q (N = 32: 1,000 contexts per seed;
N >= 18,000: one context per seed). The number of entities appearing in context grows with N (about 50 at N = 32,
all 4,096 at N = 100,000); the codebook is the fixed 4,096.

## Arms (all read the same per-mention distributions; b41_engine.py)

Tables (top-k truncation by zeroing, never renormalised; float64):
- ARGMAX: each mention stored as its arg-max; state = one entity; dead chain = wrong.
- BEAM(k, B): each mention stored as its top-k candidates with probabilities; posting lists (entity, relation) ->
  facts; state = top-B entities, paths that reach the same entity merged by summing. k in {1, 2, 4, 8, 16},
  B in {1, 4, 16, 64, inf} (ARGMAX = BEAM(1, 1)).
- ENUM(k), k in {2, 4, 8, 16}: the same top-k readings, every path enumerated without merging (the k^h form);
  accuracy = BEAM(k, inf), cost counted from path counts.
- EXACT: full distributions, dense sum-product: w_f = <p, ps_f>, p' = sum_f w_f po_f over the hop's facts (the
  strongest exact baseline; any truncation of it is a BEAM arm).
VSA (bipolar codebook PHI of width D, seeded per seed and D; one memory per (context, relation),
M = sum_f code(s_f) * P(code(o_f)), P a fixed coordinate permutation so subject and object are not interchangeable;
code of a mention = sum over its top-kb candidates of p(e) PHI[e]; hop x <- P^-1(x * M_r), x /= |x|):
- VSA-END(D, kb): clean-up (arg-max of PHI x) only when the answer is read.
- VSA-HOP(D, kb, c): clean-up after every hop, keep the top-c entities weighted by max(score, 0), re-bundle.
- D in {256, 1,024, 4,096, 16,384, 65,536}, kb in {1, 4, 16}, c in {1, 4, 16}: 60 VSA configurations.
Relations are read exactly, so one memory (VSA) or posting list (tables) per relation is allowed to both.

## Cost model (per query; identical definitions in the second implementation)

One multiply-add, comparison, hash probe or normalisation step = 1 op; memory traffic = 32-bit words read; state =
words carried between hops (max over hops).
- Every arm reads the start distribution: E ops (top-k selection or the dense vector), E words.
- Tables per hop: probes = |state|; posting MACs = sum over state entities of matching postings; scatter MACs =
  sum over matched facts of stored object candidates; normalise = |new support|; top-B selection = |new support| when
  it exceeds B. Traffic = 2 |state| + 2 postings + 2 scatter. Final arg-max = |support|. State = 2 |support|.
- ENUM: per hop, one probe per path plus, per matched (path, fact), 1 + |stored object candidates|; final merge =
  number of paths; state = 2 x paths.
- EXACT per hop: 2 x (facts of the hop's relation) x E + E ops and words; state E.
- VSA: start kb D (bundle) + 2D (normalise); per hop 3D ops (unbind + normalise), 2D words (memory row + state);
  clean-up E D + E ops and E D words; re-bundle c D + 2D ops, c D words; state D.
- Reported apart, per context: store build ops and store size (tables: top-k selection 2E per fact, stored entries;
  EXACT: N (1 + 2E) words; VSA: N (2E + 2 kb D + D) ops, slots x D words plus the shared E x D codebook). The naming
  head's own forward (same for every arm) is not counted.
Primary cost = per-query ops (query time; the VSA's larger store build is NOT charged to it, which favours the VSA).

## Dev and test

Dev chooses: the layer and T (head), the three rho levels, and per cell the VSA configuration v* with the largest dev
A-gap under ops (ties: higher dev accuracy, then lower ops); v_acc = the most accurate VSA configuration on dev is
recorded as a secondary. The dev record (choices + dev tables) is committed before the test bank exists. Test runs
once, every arm, every cell; nothing is chosen on test. Tables are reported as their full grids (eligibility at equal
cost uses the test means of each configuration; the max over eligible tables is taken inside every bootstrap
replicate).

## Primary question, bar and verdict

Per cell and cost metric m (v = v*):
- A-gap(m) = acc(v) - max{acc(t) : table t with mean m-cost <= v's};  WIN-ACC if the 95% CI lower bound > 0.
- A-gap10(m) = acc(v) - max{acc(t) : mean m-cost <= 10 x v's};  WIN-COST ("equal accuracy at >= 10x lower cost")
  if the 95% CI lower bound > -0.01.
- CIs: 2,000 bootstrap replicates, queries resampled within seed (paired across arms).
Primary regime (registered as the one most favourable to the VSA a priori): rho = high, h in {4, 6}, every N
(10 cells). PRIMARY VERDICT = "VSA REGIME FOUND" if any primary cell is WIN-ACC or WIN-COST under per-query ops;
otherwise "TABLE DOMINATES". The same bars under traffic and state, and over all 45 cells, are secondary
(descriptive; with 45 cells x 3 metrics a single marginal win would be read as noise unless it repeats in adjacent
cells). Also reported: C-ratio (ops of the cheapest table within 1 point of v's accuracy / ops of v), the value of
carrying uncertainty (EXACT - ARGMAX), and the share of it each D recovers (best VSA config per D).

## Gates

- G0 (second implementation, b41_indep.py: numpy float64, explicit posting lists, path lists and loops, own counters):
  on the first 150 queries of every test run at N <= 3,200 and on 100 queries at N = 32,000 / 100,000 for the tables
  (VSA D <= 4,096, kb in {1, 4}, END/HOP1/HOP4): answers agree on >= 99% of (query, h) per config and every cost
  count matches to 1e-6 relative; disagreements listed. Failing G0 voids the cell until explained.
- G1 (noise-free code check, dev): with one-hot readings, VSA-HOP1 at D = 4,096 answers >= 0.99 at N = 320.
- G2: realised test per-mention argmax error within 0.03 of the dev value at each level.
- Hand-reading: 15 failures each of ARGMAX, BEAM(4, 16), EXACT and v* in the primary regime (mention texts, top-3
  readings, chain, decoy), notes in results/b41_handread_notes.txt.

## Predictions (numeric, checked mechanically where possible)

- P1 Dev per-mention argmax error at the chosen layer spans 0.02 to >= 0.25 within rho <= 0.3; most errors fall
  inside the true name's cluster (>= 60%).
- P2 ARGMAX at high noise, h = 6: <= 0.10 in every N; at low noise, h = 2: >= 0.80.
- P3 EXACT - ARGMAX >= +0.30 at high noise, h in {4, 6}, every N.
- P4 BEAM(4, 16) within 0.01 of EXACT in >= 40 of 45 cells, at <= 1/100 of EXACT's ops for N >= 3,200.
- P5 VSA-END is never v*'s mode at h = 6; its best accuracy at h = 6 is <= ARGMAX's at N >= 3,200.
- P6 The most accurate VSA config beats ARGMAX by >= 0.20 at high noise, h in {4, 6}, N <= 3,200.
- P7 No VSA config exceeds EXACT by more than its CI in any cell (VSA <= EXACT + 0.02 everywhere).
- P8 PRIMARY: TABLE DOMINATES; no VSA-WIN in any of the 45 cells under ops, traffic or state.
- P9 C-ratio <= 1e-3 in every cell (a table within 1 point of v*'s accuracy costs <= 1/1,000 of v*'s ops).
- P10 At N = 100,000, h = 6, high noise, the most accurate VSA config is below EXACT by >= 0.20.
- P11 The smallest D on the grid at which the best VSA config recovers >= 90% of EXACT - ARGMAX (high noise, h = 6)
  grows with N (non-decreasing across N where defined; undefined = never reached on the grid).

## What the paper may say, per outcome

- TABLE DOMINATES (expected): "A VSA can carry an LLM's naming uncertainty through multi-hop composition in one
  fixed-width vector and recovers X of the Y points that committing to the arg-max loses (with clean-up per hop,
  D >= ...), but a table that carries the same distribution - exact sparse propagation or a top-k beam - is at least as
  accurate at 10^a-10^b times fewer operations, less memory traffic and a smaller state in every cell tested. Computing
  in superposition is therefore not what VSAs give an LLM over a table on a digital processor." Plus the measured D
  scaling. It must not say the VSA beats tables or that superposition avoids k^h relative to a probability table.
- VSA REGIME FOUND: state the cell(s), the metric and the size, with the caveat of the cost model (query time only,
  store build excluded), and that it is one registered regime.
- If gates fail: report the failure; no claim from the affected cells.

## Compute

Features: three Qwen3-8B passes over ~300k short sentences (~15 GPU-min). Engine: 45 test runs + 15 dev runs, about
2-6 GPU-hours on idle A800 GPUs (never GPU 3); the second implementation on CPU. Cap 30 GPU-hours; no paid API.
