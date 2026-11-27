# NSR0 pre-registration: does the VSA rule memory add anything over a composition table? (CLUTRR, gold facts)

Date: 2026-11-06. Written and committed before any NSR0 code. Brief: docs/neurosymbolic_brief.md. Background: R1
(prereg/R1_prereg.md, results/r1_verdict.txt). With gold facts, R1's VSA rule memory reached 0.975 overall and 1.000 at
k = 10 on CLUTRR gen_train23_test2to10, but in `r1_oracle.py` the rules are first voted into a Python dict and only then
written into the VSA vector; the "known rule" test also reads the dict. So the R1 VSA arm is the table read back through
a vector, and it cannot differ from the table except by crosstalk. NSR0 asks whether a VSA store that learns *without*
a side dictionary adds anything on any axis: sample efficiency, noise robustness, size, or confidence.

## Data and splits

- CLUTRR v1 `gen_train23_test2to10` (local copy `r1_data/`). Facts are gold: the relation chain `f_comb`; the label is
  `target_text`. No reader is involved.
- Train: 9,074 stories (k = 2: 4,076 over 62 distinct chains; k = 3: 4,998 over 321 distinct chains; every distinct
  chain has >= 10 stories, so every budget below is satisfiable). Train contains 7 inherently ambiguous k = 3 chains
  (e.g. wife-daughter-grandmother -> mother 7 / mother-in-law 8); they are kept as they are.
- Validation (k = 2, 3; 2,020 stories): used only to choose the two free parameters of the VSA learners (below).
- Test (k = 2..10, 1,146 stories): reporting only. Disclosure: R1's oracle diagnostic already ran on this split; NSR0
  has no component tuned on it.

## Shared learning and inference (identical for every arm; only the store differs)

Each arm is a *store* with three operations: `write(a, b, v)`, `read(a, b) -> (value, confidence)`, `known(a, b)`.
1. Stage 1: every sampled k = 2 story writes (r1, r2) -> target.
2. Stage 2 (closure, 3 rounds, as in R1): for each sampled k = 3 story (a, b, c -> t), if `known(a, b)` and the pair
   (read(a, b), c) is not known at the start of the round, write (read(a, b), c) -> t; symmetrically for (b, c).
   `known` is evaluated on a snapshot taken at the start of the round.
3. Inference: CYK over the chain with any bracketing (R1's `compose`), using `known` and `read`. Several distinct
   results -> "AMBIG"; none -> "UNKNOWN". Both count as wrong; their rates are reported.
4. Confidence of an answer: the minimum `read` confidence over the rule applications on the winning derivation.

## Arms

- **T (table).** Count table: `write` adds a vote; `read` returns the majority value (ties: lexicographic) with
  confidence = winning vote share; `known` = at least one vote.
- **V-dict (R1's design).** T's learning, then the voted rules are written once each into an FHRR superposition
  (three independent codebooks; D = 2048) and `read` goes through the vector (cleanup). `known` uses the dict.
- **V-sum (pure VSA, additive).** M += bind(first(a), second(b)) * value(v) for every write; `read` = cleanup of
  unbind(M, key) over the value codebook; `known` = top cleanup similarity >= tau (similarity in units of one write).
  Confidence = top similarity / sum of positive similarities. No side dictionary anywhere.
- **V-delta (pure VSA, delta rule).** M += alpha * key * (value(v) - unbind(M, key)); otherwise as V-sum.
- Free parameters, chosen on validation before any test run: tau in {0.3, 0.4, 0.5, 0.6, 0.7} for V-sum and V-delta,
  alpha in {0.3, 0.5, 1.0} for V-delta; criterion = mean validation accuracy over the two cells (m = 3, clean) and
  (m = 3, label noise 0.2), 3 seeds. Then frozen.

## Cells (10 seeds each; a seed varies the sampled stories, the noise and the codebooks)

- **S (sample efficiency):** m in {1, 3, 10} stories per distinct training chain, and "all".
- **C (coverage):** N in {50, 100, 200, 400, 800} stories drawn at random from train (k = 2 and 3 pooled; some rules
  are then missing and CYK must route around them).
- **L (label noise):** training target replaced by a uniformly random different target with probability p in
  {0.1, 0.2, 0.3}, at m in {1, 3, 10}.
- **F (fact noise):** one relation of the training chain replaced by a random different chain relation with
  probability p in {0.1, 0.2, 0.3}, at m in {1, 3, 10}.
- **Z (size):** full training data, clean; V-dict and V-sum at D in {32, 64, 128, 256, 512, 1024, 2048}.
  Bytes: complex64 = 8 bytes per dimension; also 4-bit phase quantisation (0.5 byte per dimension), evaluated by
  quantising the stored vector. Table bytes: packed (rule count x 3 x ceil(log2 #relations) bits), counted exactly.
- **Q (test-fact noise, sanity):** clean full training; each test chain relation corrupted with probability 0.05.

## Measures

Accuracy on the test split (all 1,146; per k; overall = mean over stories as in R1; k >= 7), UNKNOWN and AMBIG rates,
learned rule count, and, in L and F at p = 0.3, AUROC of answer confidence for correctness.

## Claims and bars (each is "VSA adds something on this axis")

- **A1 sample efficiency:** a pure VSA arm (V-sum or V-delta) beats T by >= 0.02 overall in some S or C cell, with the
  paired per-seed difference > 0 in >= 9 of 10 seeds.
- **A2 noise:** the same bar in some L or F cell.
- **A3 size:** the smallest D at which V-dict or V-sum is within 0.01 of T's overall accuracy (cell "all", clean)
  uses fewer bytes (either precision) than T's packed table.
- **A4 confidence:** a VSA arm's AUROC exceeds T's by >= 0.05 in the L or F cell at p = 0.3, m = 3, averaged over seeds.
- **Usefulness floor (overrides A1/A2):** a "win" counts only if the winning VSA arm's overall accuracy in that cell
  is >= 0.50; wins in cells where every arm is near the floor do not count.
- **Negative controls (must hold, or the run is void):** NC1, T with its learned rule values permuted among keys
  (seeded): overall <= 0.20. NC2, T trained on labels shuffled across training stories: overall <= 0.20.
- **Reproduction check (must hold):** T and V-dict in cell "all", clean, D = 2048 reproduce R1's 0.975.

## Verification (multi-angle)

1. Independent scorer `nsr/nsr0_score.py`, written and committed before the run: reads the per-story prediction file
   and recomputes every accuracy from `test.csv` itself; the run's own summary must match it exactly.
2. Second implementation of T (`nsr/nsr0_table2.py`, a memoised recursive interval composition written separately,
   no shared code): must reproduce T's per-story predictions in cell "all", clean, and in one noisy cell.
3. Manual inspection of 20 sampled disagreements between T and the best VSA arm, if any.
4. Literature check on the conclusion (SUPPORTED / CONFLICTS / NO PRECEDENT), recorded in the verdict.

## Predictions (mine, written before the run)

- A1, A2, A3, A4 all FAIL: on relation-level symbols a count table is the optimal learner (majority vote is what
  superposed writes approximate), so the VSA stores can at best tie it; any difference comes from crosstalk and is
  against VSA.
- V-dict ties T exactly at D = 2048 in every cell. V-sum ties T at small m and loses at "all" if rare keys drown in the
  crosstalk of ~9,000 writes (predicted noise std ~ sqrt(9000 / 2048) ~ 2 writes). V-delta loses to T under label
  noise at m = 10 (a recency-weighted vote, not a majority).
- A3: T is >= 10x smaller than the smallest adequate VSA vector.
- Negative reading if all fail: on CLUTRR with gold facts the VSA rule memory adds no accuracy, robustness, size or
  confidence over a table; its remaining distinct properties are a fixed-size superposed store and graded similarity.
  Then no NSR claim may attribute reasoning to VSA, and NSR1 must carry the table/graph arm in every comparison.
- Out of scope, noted in advance: FHRR binding is commutative, so a learned code with code(r1) * code(r2) ~ code(r1 o r2)
  cannot represent kinship composition (father o sister = aunt but sister o father = father). A VSA that generalises to
  unseen relation pairs would need non-commutative binding; that is a separate experiment, not NSR0.

## Amendment 1 (2026-11-06, before tuning and before any registered test run)

1. **V-delta write corrected.** The registered form M += alpha * key * (value(v) - unbind(M, key)) is wrong for a
   single FHRR vector: key * conj(key) = 1 elementwise, so it equals M <- (1 - alpha) M + alpha * key * value(v) and
   erases everything but the last write (smoke: 1 rule known, every answer UNKNOWN). The product's delta rule is on a
   matrix memory, where the same expression is correct. Corrected form, the delta rule in the value-code subspace:
   s_r = Re<value(r), unbind(M, key)> / D for every relation r; M += alpha * key * (value(v) - sum_r s_r value(r)).
2. **V-sum tau grid widened** to {0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 3, 5}. On validation (m = 3, clean, tau = 0.5) V-sum
   marked 397 of 324+ pairs as known: the crosstalk of the superposed writes is larger than one write, so every
   registered tau sits below the noise. A wider grid can only strengthen the VSA arm, i.e. it works against my
   prediction. Selection rule unchanged (validation only; ties: first in grid order).
3. **Disclosure.** My first smoke test ran on the test split (full training data, untuned tau = 0.5, alpha = 0.5,
   one seed): T 0.9747, V-dict 0.9747 (the registered reproduction check), V-sum 0.1283 (400 pairs known, 999 AMBIG),
   V-delta 0.0 (the bug in item 1). No parameter is chosen from these numbers. Every later smoke test uses validation:
   corrected V-delta 0.9762 at m = 3 and 0.9698 on all data; T 0.9762 on both.
