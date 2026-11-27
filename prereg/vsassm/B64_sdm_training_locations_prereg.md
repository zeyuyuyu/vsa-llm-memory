# B64 pre-registration: sparse distributed memory addressed by the training split's own codes (SDMt) against the store

Date: 2026-11-23 (machine clock Asia/Shanghai, about 03:20). Written and committed, then OpenTimestamps-stamped,
before any B64 development or test run. The paper is not edited by B64.

## What has been seen before this file (disclosed)

- **A clean-room re-implementation (exploratory, not pre-registered; scratchpad cleanroom4_keyfree/cr.py,
  compare_qwen8.txt, compare_llama.txt).** Another agent re-implemented the store and B36's six unsupervised
  key-free memories from the paper's text only, on its OWN stores (own seeds) drawn from the same dev and test
  entities, for Qwen3-8B and Llama-3.1-8B, R1 only, ideal accounting. Its SDM with hard locations at the training
  split's label and alias codes (5,621 codes; activate the a nearest, a in {1, 4, 16, 64}, chosen 1-16; one-hot value
  counts or value-bit sums charged at their empirical entropy) reached 43.7 (Qwen3-8B) and 45.5 (Llama) over R1,
  against 35.6 / 36.2 for the pipeline's SDM (random Gaussian locations); with it, the clean room's first-design store
  led its best of six by +2.3 [1.6, 3.1] (Qwen3-8B) and +2.1 [1.3, 2.9] (Llama), its final design by +4.7 / +2.9,
  where the pipeline reports +6.8 / +7.1 (first) and +10.2 / +9.2 (final). **So the direction of the Qwen3-8B and
  Llama outcomes is known in advance (on other stores of the same test entities); Granite, Mistral and the three
  sentence encoders are blind.** No pipeline store has been answered by any SDMt.
- **B64 runs before this file (no dev or test store):** `b64_run.py selftest` on random toy data (budgets
  respected; the fast incremental implementation equals a slow, separately written full-sort path on 176
  (configuration, budget, accounting) triples, N identical and predictions >= 0.99 identical; exact-key read 1.000);
  `b64_run.py timing` on one TRAINING-split store (Qwen3-8B, M = 1,024, V = 16: 35 s for the full grid, one thread);
  `b64_run.py disjoint` (below), which found that under accent-stripping tokenizers the training label 'Rio' has
  exactly the code of the dev / test string 'Río' (e5, bge, MiniLM), so code-equal locations are now removed too.

## Why

The paper's one positive equal-bit result for superposition is that the store (first design, random value codes) beat
the best of six unsupervised key-free memories (per-fact index, static function, count-min sketch, Bloom filter, fly
Bloom filter, SDM; B36's families, B49's panel) by +6.1 to +7.1 points over R1 on the four language-model codes. B36's
SDM places its hard locations at random Gaussian directions (Kanerva 1988). The clean room shows that an SDM whose hard
locations are the training split's own codes is far stronger. The paper's rules let every memory keep, for free, fixed
components fitted on the training split (PQ codebooks, ITQ rotations, learned hashing, the store's own whitening), so
this is a legitimate, stronger baseline. B64 tests it properly on all seven encoders, with the pipeline's stores,
queries, budgets, accounting and choice rule.

## Fixed (B36 / B49 / B53 / B60's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (b49_out/b32_data,
md5s as B49 / B60), splits, B32's make_store stores (dev 6 + 3 per (M, V) at M = 256 / 1,024, test 20 + 5), queries
(unseen aliases), V in {2, 16}, R1 budgets 4, 8, 16 bits per fact (R1 only; R0 / RH / R2 not run), accountings (ideal
primary, real secondary). B36's conventions for a key-free family: the default value (the store's plurality value) is
answered when nothing votes and costs ceil(log2 V) bits; b36_common.decide (ties to the default value, then the smallest
value); nothing fits -> the default value, 0 bits (as B36's SDM). The other arms are NOT re-run: their test predictions
come from the merged records b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz (VSAR = B49's store, first design; VSA =
B53's store, final design; IDX, SF, CM, BF, FBF, SDM = B36's families as run in B36 / B46 / B49; BEST36 = B36's best of
six). The six families' dev accuracies (for the best of seven) are B49's (results/b49_choice_{enc}.json 'dev_by_fam',
copied there from the B36-protocol choices); b64_run.choose asserts that they reproduce B36's BEST of six in every cell.

## SDMt (b64_run.py; spec: the clean room's description; own code)

- **Locations.** A fixed set of training-split codes: 'LA' = every training label and alias (5,621), 'L' = the labels
  (2,818), 'A' = the aliases (2,803); minus any string that is a dev or test label or alias (none) and any location
  whose code row equals a dev or test code row (one, 'Rio', on e5 / bge / MiniLM). Fixed order: a permutation from seed
  ('B64LOC', set). The memory uses the first N locations of the order. Address of location l = P(x_l), P = the
  configuration's prep (b32_common.Prep 'center' or 'whiten', fitted on the training split; rows unit-norm). Free, like
  B36's PQ codebooks (fixed, fitted on the training split, independent of the store's facts).
- **Write.** Each stored fact (the first round(share x M) facts of B36's store order) activates the a locations among
  the first N with the largest cosine to P(key) (all N if N < a). Counters per location: 'onehot' = V counters, the
  number of activating facts with each value; 'bits' = ceil(log2 V) counters, the sum of the activating facts' bipolar
  value bits (2 bit_j(v) - 1).
- **Counter type.** q 'cnt' exact; 'clip3' saturating (onehot min(c, 3); bits clip to [-3, 3]); 'tern' (onehot
  min(c, 1), one presence bit per value; bits sign(c)).
- **Read.** The query activates its a nearest of the same N locations; onehot: score(v) = sum over its active locations
  of Q(C[l, v]); bits: score(v) = sum_j (sum_l Q(C[l, j])) (2 bit_j(v) - 1); answer = b36_common.decide.
- **Bits.** ideal (primary): N c H0 + ceil(log2 V), c counters per location, H0 the zeroth-order empirical entropy of
  the stored counter symbols (B36's SDM rule). real (secondary): fixed width per counter (onehot: cnt ceil(log2(max +
  1)) + 8 header bits, clip3 2, tern 1; bits: cnt ceil(log2(2 max|c| + 1)) + 8, clip3 3, tern 2) + ceil(log2 V).
  **N** = the largest point of a fixed geometric grid (160 points from 1 to the set size, rounded, unique) whose bits
  fit the budget (the cost is not monotone in N, so every grid point is tested; consecutive grid points differ by at
  most about 6%).
- **Grid** (order = tie order): prep (center, whiten) > location set (LA, L, A) > a (1, 2, 4, 8, 16, 32, 64) > counters
  (onehot, bits) > q (cnt, clip3, tern) > share (1, 1/2): **504 configurations** per cell. (B36's SDM: 40; the clean
  room's: 64 with random and training locations; B36's whole six-family set: 468.)
- **Dev choice (B36's rule).** Per cell (encoder, M, V, bpf, acct): the configuration with the highest mean dev
  accuracy over the cell's dev stores; within 0.1 points tied, first in grid order. Test: the chosen configuration only,
  run once, after the choices are committed.
- **BEST7** = B36's BEST rule over IDX, SF, CM, BF, FBF, SDM, SDMT: per cell the family whose chosen configuration has
  the highest dev accuracy (exact ties in that order); test predictions = that family's.

## Comparisons (ideal accounting; 95% cluster bootstrap, entity within store, resampled within each cell; 10,000 draws)

Region statistic = mean over cells of the cell mean (b36_verdict.boot_region), on every encoder, over **R1** (12 cells)
and, reported separately, **R1-V2** and **R1-V16** (6 cells each):
1. **VSAR - SDMT** (first design minus SDMt);
2. **VSAR - BEST7** (first design minus the best of seven);
3. **VSA - BEST7** (final design minus the best of seven).

**Decision rule (B60 / B49):** "store leads" iff the 95% lower bound > 0; "memory leads" iff the upper bound < 0;
otherwise "level". Counts over the 4 language models and the 3 sentence encoders, for each comparison and region.

**The paper's pre-specified bar for the key-free claim (B36 H1, B49 H1-B49):** over R1, store - BEST >= +2.0 points
and the 95% interval above 0 = "passes"; <= -2.0 with the interval below 0 = "loss"; otherwise "tie". Applied to
comparisons 2 and 3 on every encoder; panel counts (of 7, of 4 language models, of 3 sentence encoders).

Secondary (2,000 draws unless stated): VSAR - BEST36 and VSA - BEST36 (continuity; must reproduce B49 / B53's numbers,
same predictions); SDMT - SDM (the effect of the locations); comparisons 1-3 under real accounting; BEST7's dev family
per R1 cell; SDMt's dev choice composition.

## Readings (fixed now) for the paper's claims

- **"The store beat unsupervised classical memories on language-model codes (+6.1 to +7.1)"** (Results, Fig. 4,
  abstract "the store beat key-free memories on language-model codes"): SURVIVES (with the new range) only if VSAR -
  BEST7 passes the bar on all 4 language models. If it does not pass on all 4 but "store leads" on all 4: it must be
  restated as a lead of x to y points over the best of seven unsupervised key-free memories (with SDMt named), below
  the pre-specified +2-point bar on the encoders where it fails, and +6.1 to +7.1 must go. If "level" or "memory leads"
  on any language model: the claim must go for the first design on those encoders, and the paper must say that an SDM
  addressed by training codes matched or beat the store there.
- **The final design (B53)** may be stated against BEST7 only with its own labels and bar results; it cannot replace
  the first design's number silently (it is a different store, and its comparison is reported as such).
- **Table 1, superposition row, "against key-free memories +6.1 to +7.1 (language models)":** replaced by the VSAR -
  BEST7 range (and VSA - BEST7 range) over the 4 language models with the labels and the bar count.
- **Per-ingredient verdict for superposition ("(ii) stored similar keys compactly at a few bits per fact against
  key-free memories"):** survives only if VSAR - BEST7 or VSA - BEST7 passes the bar on all 4 language models (the
  design named); otherwise restated by the labels. In every case B36's pre-fixed reading applies: SDMt is itself a
  superposed memory (each location sums the value counts of every fact that activates it), so if SDMt is the best
  of seven and closes the gap, "a superposition memory with a different feature map (data-placed hard locations) does
  as well as the dense MAP store; the 'no bits on keys' mechanism stands, the VSA-specific claim does not".
- Whatever B64 shows, the store's numbers are unchanged (only its key-free baseline is strengthened), and the
  comparisons with least squares (B60) are not affected.

## Predictions (stated so that they can be wrong)

P1. SDMT - SDM over R1 >= +5 points on each of the 4 language models (clean room +8.2 / +9.3 with a smaller grid).
P2. VSAR - SDMT over R1 between -1.0 and +3.0 on each of the 4 language models.
P3. VSAR - BEST7 over R1: the bar passes on at most 1 of the 4 language models; "store leads" on at least 2 of 4.
P4. VSA - BEST7 over R1: "store leads" on all 4 language models; the bar passes on at least 3 of 4.
P5. VSA - BEST7 over R1-V2 >= +4 on all 4 language models; over R1-V16 between -2 and +4 on each.
P6. Sentence encoders: VSAR - BEST7 over R1 "memory leads" on all 3; VSA - BEST7 passes the bar on none of the 3.
P7. BEST7 = SDMT (dev choice, ideal) in at least 8 of the 12 R1 cells of each language model.
P8. SDMt dev choices (ideal, R1, 84 cells pooled over the 7 encoders): a <= 16 in at least 75%; location set 'LA' in at
    least half; onehot counters in at least half of the 42 V = 16 cells.
P9. VSAR - BEST36 over R1 reproduces B49's numbers within 0.01 points on every encoder.
P10. Checks pass: plain-loop scorer |difference| < 1e-9; 0 dev or test rows over budget (and the audit's own
    re-computation of the chosen N agrees); disjointness 0 string and 0 code-row overlaps, and no location with raw
    cosine >= 0.9999 to a dev or test code; float64 re-derivation >= 0.99 agreement on every re-derived store.

## Checks (b64_check.py; second angle)

- **Plain-loop scorer:** every R1 cell accuracy of VSAR, VSA, SDMT, BEST7, BEST36 and the point statistics of
  comparisons 1-3 recomputed with plain Python loops from the job files and the merged records; |difference| from the
  verdict < 1e-9.
- **Bits audit:** every dev and test row: bits <= budget; for the test rows of a sample of stores, the bits and N
  recomputed by the independent implementation below (own grid, own counters, own entropy).
- **Location-set disjointness:** strings, accent / case-normalised strings, exact code rows, and the maximum raw
  cosine between every location and every dev / test code, per encoder and set.
- **Float64 re-derivation (independent code):** one test store per (encoder, M, V) at 4 and 16 bits per fact, ideal,
  the chosen SDMt configuration recomputed in float64 (own cosine, own full stable sort for the a nearest, own
  counters by loops over facts, own entropy, own N scan, own read and tie rule); agreement with the logged predictions
  >= 0.99 (near-ties of float32 cosines may flip an activation).
- **Sample reading by the AI-assisted process:** Qwen3-8B and e5-base-v2, the R1 cell with the lowest VSAR - SDMT:
  10 queries only SDMt answers right and 10 only the store answers right (alias, label, gold, both answers, the
  location strings the alias activated), with notes.

## Compute and time rule

CPU only, local machine (16 cores, shared; at most 10 worker processes, one thread each, nice 10); the A800 box only if
the local machine is too slow (its /data0 is 100% used, 7.7 GB free; nothing written there unless needed). Time box:
2 hours from 03:04 (machine clock). Priority if time is short: the 4 language models first (dev, choice, test,
verdict), then the 3 sentence encoders; real accounting last. Any cut is written into a dated compute note before any
test output is read.

## Files

b64_run.py (SDMt, plans, dev choice, selftest, disjointness), b64_verdict.py, b64_check.py; results/b64_*; job
outputs under b64_out/ (not committed).
