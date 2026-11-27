# NSR2 pre-registration: a neuro-symbolic engine that knows when the read state may be wrong

Date: 2026-11-09. Written after development on dev, the retired test and nsr1 (all three are development data now;
nsr1 was spent by NSR1), committed before the held-out split nsr2 exists.

## Question

NSR1 showed that every miss of the Readable Memory pipeline is a misread state fact, that the product's own confidence is
1.0 for every fact (no signal), and that a plain graph query composes as well as anything. The engine can only add value
by reasoning about which facts may be wrong. NSR2 asks: does a neuro-symbolic engine, which puts a learned reliability on
every read fact and answers by exact probabilistic inference over alternative places, know when its answers are wrong,
better than simple confidence heuristics?

## Engine (nsr/nsr2.py at the prereg commit)

- **Neural part.** The shipped reader (unchanged) writes the state. A logistic-regression reliability model scores each
  state fact from features anyone can observe in the log: whether a human message or a system feed wrote it, the channel,
  whether that message carried more than one move, how many tracked names it mentions, how late it came. Trained on the
  facts of dev + retired test + nsr1 (label: does the fact match the truth).
- **Symbolic part.** Every thing gets a distribution over places: the place read (probability = reliability q); the rest
  (1 - q) goes to the place the memory held before that write with weight w_prev (learned from the same data: how often a
  wrong fact's truth is the previous place; 0.35-0.38 in development), and is spread uniformly over the other places of a
  kind that may hold the thing (a containment-type table learned from the truth histories of the same data). Impact
  questions are answered by exact inference over these distributions and the site catalog (containment chains recurse
  through the containers' own distributions): P(x affected) for every candidate x, the MAP answer {x : P >= 0.5}, and a
  question confidence prod_x max(P, 1 - P).

## Arms (confidence rankings of the same deterministic answers unless stated)

1. **NS engine** confidence (above).
2. Baselines: (a) min reliability over the facts the deterministic answer depends on (the reliability model without
   inference); (b) minus the number of those facts written by human messages; (c) minus the answer-set size.
3. **ProbLog** (2.2.6) encoding of the same distributions and rules: a second, independent implementation of the inference.
4. **VSA substrate** (secondary): the same soft state stored as one FHRR superposition sum_x sum_l P(x at l) bind(x, l),
   probabilities read back by unbinding and similarity, then the same inference; D in {256, 1024, 2048}.
5. **Negative control**: reliabilities permuted among the facts of each log.
Deterministic answers = NSR1's graph query on the read state; the NS MAP answers are also scored.

## Data

Held-out **nsr2**: 40 fresh logs (10 per scenario, seeds 500+) generated after this commit with the vendored generator
(nsr/nsr1_gen.py, apps/real at 707351b) and Qwen3-32B (rev 9216db57) on the A800 box, read by the shipped reader
(nsr/nsr1_reader.py). The same impact questions as NSR1 (nsr1_catalog.questions). Nothing is tuned on nsr2.

## Measures and bars (nsr2, all questions)

- **Floor (else the run is void):** fact-level AUROC of the reliability model >= 0.75 and at least 8 wrong facts.
- **B1:** NS question-level AUROC (confidence vs the deterministic answer being right) >= 0.72.
- **B2:** NS AUROC - the best of baselines 2a-2c >= 0.03.
- **B3:** NS selective accuracy at 80% coverage - deterministic accuracy on all questions >= +0.025.
- **B4:** ProbLog marginals equal the engine's on every question (max |difference| <= 1e-6).
- **B5:** negative control AUROC <= 0.62.
- **B6 (secondary):** VSA-substrate AUROC within 0.02 of the exact engine at D = 2048.
Reported: MAP accuracy, selective accuracy at 70/80/90% coverage, fact-level ECE, per-scenario numbers.
Independent scorer: nsr/nsr2_score.py (own AUROC, recomputes gold from the logs), written before nsr2 exists.

## Development numbers (seen; train on two splits, evaluate on the third)

Fact AUROC 0.805 (eval nsr1) / 0.845 (eval test). Question AUROC: NS 0.761 / 0.749; best baseline 0.672 / 0.725.
Selective accuracy at 80% coverage: NS 0.957 / 0.897 vs all-question deterministic 0.930 / 0.857. MAP answers = the
deterministic answers on every development question.

## Predictions

Floor holds (about 0.8). B1 holds (about 0.75). B2 is uncertain (development +0.09 and +0.02); I predict a pass with
about 60% confidence. B3 holds (about +0.03). B4 holds. B5 holds. B6 holds at D = 2048 and fails at D = 256. MAP accuracy
equals deterministic accuracy (the engine flags errors; it does not repair them).
What a pass would support: "a neuro-symbolic engine that models read reliability can tell which impact answers to send
to a human". What it would not: that it repairs the state, or that the vector substrate is needed (B6 is a parity test).

## Amendment 1 (2026-11-09, development data only; nsr2 is still being generated and has not been read)

Smoke runs of the full pipeline on the development splits (train on two of dev / test / nsr1, evaluate on the third)
found three design problems, fixed here before any held-out data exist.
1. **Engine: the containment-type table now also constrains the read and previous places** (a read place that breaks the
   table gets reliability 0). Without it, a reader error such as "rack R-07 in box CB-007" creates a cycle among the
   candidate places, my recursion cut it at a depth limit, and ProbLog (exact on cycles) disagreed by up to 6e-4. After
   the change ProbLog agrees to 3e-12 on both development directions. In development every type-violating fact was wrong.
2. **B5 was miscalibrated and is replaced.** Permuting reliabilities among a log's facts left the AUROC unchanged (0.765
   vs 0.764; 0.764 vs 0.753): the learned reliability acts through the log's overall messiness combined with the
   question's structure, not by pinpointing the wrong fact. New arms: **ns_const** (every fact gets the training mean
   reliability: the probabilistic structure alone) and **log_rel** (the log's mean reliability as every question's
   confidence: the reliability model alone, no inference). New **B5: NS - ns_const >= 0.03** (does the neural part add
   anything?). **B5b** (reported with a prediction, no bar): NS - permuted, a test of fact-level targeting.
3. **B2 includes log_rel** among the simple baselines. **B6 averages 3 codebook seeds** (one seed varied by 0.02).
Development numbers with these changes (eval nsr1 / eval test): NS AUROC 0.764 / 0.753; ns_const 0.688 / 0.678; permuted
0.765 / 0.764; log_rel 0.585 / 0.625; best simple baseline 0.672 / 0.725; selective accuracy at 80% NS 0.957 / 0.897 vs
deterministic 0.930 / 0.857; VSA D=2048 (one seed) 0.700 / 0.760.
**Updated predictions:** floor, B1, B3, B4 hold; B5 holds (about +0.07); B2 uncertain (development +0.09 and +0.03);
B5b about 0 (no fact-level targeting); B6 uncertain (development -0.06 and +0.01). What a pass of B1-B5 would support:
the engine's confidence comes from the interaction of learned reliability and the question's probabilistic structure;
neither alone reaches it. What B5b near 0 would mean: it does not know WHICH fact is wrong, only how risky a question is.
