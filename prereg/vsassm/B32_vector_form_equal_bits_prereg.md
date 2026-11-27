# B32 pre-registration: the vector form against an equally engineered vector index at the same number of bits

Date: 2026-11-18 (US Eastern; box clock Asia/Shanghai). Written before the B32 dev grid (b32_run.py dev) and before
any B32 memory has answered a test query.

Done before this file (nothing on test queries):
- Data: b32_data.py built the entity pool from PopQA (b32_data/entities.json); b32_embed.py encoded every label and
  alias (train, dev and test strings; no memory, no answer) with five sentence encoders (CPU) and Qwen3-8B (A800
  GPU 4, 89 s; 18,292 strings).
- Encoder choice by the criterion below (dev only): b32_lm_select.py (results in this file).
- Literature check (below).
- Three pilots on the dev split only (b32_pilot.py, b32_pilot2.py, b32_pilot3.py; outputs b32_out/pilot*.json/log);
  their numbers are summarised at the end. They informed the region R1, the grid and the predictions.
- Code: b32_common.py, b32_run.py (dev grid, choice, test, merge), b32_verdict.py (bars, mechanical), b32_check.py
  (independent scorer and re-derivation), b32_handread.py, b32_run_gpu.sh. b32_run.py selftest passes on CPU.

## Why

Four referees of the npj Unconventional Computing manuscript agree that in every benchmark of the paper a plain table
filled by the same reader gives the same answers as the VSA store, that a trigram table matched the trigram VSA (B26a),
that a repetition-coded table beat the store on simulated phase-change devices (B26b), and that every working
neural-to-VSA interface is discrete. Their main request: "make the vector form, not exactness, the variable that is
manipulated, test it against an equally engineered table, and show at least one case where the vector form wins",
for example "a neural reader that writes continuous codes directly into the store, with no argmax and no string
hashing, for open-vocabulary or unseen symbols where a table has no key". B32 is that test. It goes into the paper
whatever it shows.

## What theory and the literature predict (checked before the design; verified bibliography in the B32 report)

- A superposed store is a lossy compression of a key-value list. With unlimited bits an index over the same codes
  that returns the nearest stored key (a MAX over keys) is at least as good as a VSA read, which is a kernel-weighted
  SUM over all stored keys (Schlag, Irie & Schmidhuber 2021; Zoology, Arora et al. 2023; Based, Arora et al. 2024;
  Frady et al. 2021 on similarity kernels). So the vector form can only win at a fixed budget.
- On exact keys, static functions (Bloomier filter, Chazelle et al. 2004; ribbon retrieval, Dillinger & Walzer 2021)
  store M facts in about M log2 V bits without keys, below what one-shot superposition achieves (about 0.5 bits per
  number; Frady, Kleyko & Sommer 2018). So exact keys are not where the vector form can win (B26b agrees).
- With continuous keys queried by aliases, a static function has no key to hash. An index must spend bits per fact
  on a key code; a VSA spends none on keys (the query brings its own code) but pays crosstalk, which grows with the
  number of stored facts relative to the dimension of the key space and with the number of keys that share a value.
  Teeters, Kleyko, Kanerva & Olshausen (2023) find superposition smallest only when the item memory is not stored,
  which is the case here (keys come from the encoder). The expected winner region is therefore: few distinct values
  (small V), few bits per fact, keys with a high-dimensional code. No published equal-bits comparison of a VSA
  against PQ / binary hashing with noisy continuous keys was found.
- Setting (c), holistic queries, is not chosen: a table answers an analogy or role query exactly in two look-ups
  (Kanerva 2010's "dollar of Mexico" is a one-step mapping, not an accuracy advantage), and set-overlap estimates are
  the domain of MinHash and Bloom-type sketches, which are themselves superposition structures (Kleyko et al. 2022:
  Bloom filters are a subclass of HDC/VSA) and beat bundles at equal bits.

## Setting chosen: (a) and (b) together

Open-vocabulary recall at a fixed memory budget. A store holds M facts "entity -> value". The entities are real
(PopQA subjects with Wikidata aliases), the values are random (so no value can be guessed from a name, and only memory
helps), each fact is written under the entity's label, and each query names the entity by one of its Wikidata aliases,
which never occurs in the store. Keys are continuous codes that a frozen neural reader writes directly (no argmax, no
string hashing). Both memories receive the same codes, the same budget in bits, the same training data for any fitted
transform, and a dev-tuned grid of comparable size. The manipulated variable is the form of the memory: one superposed
vector (VSA) or per-fact entries in a vector index.

## Data (b32_data.py; b32_data/entities.json)

- PopQA (Mallen et al. 2023; https://huggingface.co/datasets/akariasai/PopQA, test.tsv, 14,267 rows): one entity per
  Wikidata URI (12,244); its key string is PopQA's subject label. An alias is any of PopQA's s_aliases whose
  normalized form (lower case; every run of non-alphanumerics -> one space) differs from the label's.
- Dropped: 2,966 entities whose normalized label is shared with another entity (a store could not tell them apart);
  1,140 aliases whose normalized form equals another entity's label or alias (ambiguous gold). Kept: 9,014 aliases.
- Split by a hash of the URI: train 2,818 entities (1,075 with aliases, 2,803 aliases), dev 1,838 (706, 2,152), test
  4,622 (1,743, 4,059). Train strings fit every transform (centring, whitening, PCA, ITQ, PQ codebooks); dev tunes;
  test is scored once.
- Stores (b32_common.make_store): M entities drawn from one split (seeded per split, M, V, replicate), each given a
  value drawn uniformly from V symbols. Entities without aliases stay in the store as distractors. Every alias of a
  stored entity is one query; its gold answer is that entity's value.
- Limitations stated now: the facts are synthetic (random values) although names and aliases are real; many aliases
  are hard (translations, nicknames, former names): the exact-cosine index finds the right entity for about 70% of dev
  aliases; aliases are from one source (Wikidata via PopQA).

## Readers (encoders) and how the primary one was chosen

Criterion, fixed before any memory comparison: exact-cosine entity retrieval over the whole dev pool (every dev label
stored, every dev alias a query; codes centred on the training mean) (b32_lm_select.py). Highest = primary encoder.

| encoder | dev entity retrieval |
| --- | --- |
| Qwen3-8B, mean of layer-9 hidden states over the name's tokens in "Name: {x}" (4,096 numbers) | **0.7035** |
| intfloat/e5-base-v2 ("query: " prefix, 768) | 0.6980 |
| BAAI/bge-small-en-v1.5 (384) | 0.6547 |
| all-MiniLM-L6-v2 (384) | 0.6296 |
| all-mpnet-base-v2 (768) | 0.6264 |
| paraphrase-multilingual-mpnet-base-v2 (768) | 0.6227 |
| Qwen3-8B, other poolings / layers (mean L18 / L27 / L36; PromptEOL last token L9 / L18 / L27 / L36) | 0.474 / 0.621 / 0.409; 0.255 / 0.234 / 0.464 / 0.442 |

- **Primary: Qwen3-8B, mean-pooled layer 9** (the paper's RULER reader, frozen; stored as float16 for transfer, md5
  1e4f5a91b57bd8987c899b770ff0c915; its dev score in float16 is the same 0.7035). **Replication: e5-base-v2**, the best
  sentence encoder. Both run the whole protocol (dev grid, choice, test).

## Memories (b32_common.py, b32_run.py)

Budget: B = bpf x M bits per store (bpf = bits per fact). Free for every memory: the encoder, the value vocabulary (V
symbols), and fixed transforms fitted on the train split or drawn from seeds (centring, whitening, PCA, ITQ rotations,
PQ/OPQ codebooks, random projections, random value codes). Everything that depends on the store's facts is counted.

- **VSA** (the vector form). Preprocess the key code (centre, or centre + ZCA-whiten with eps 0.1, i.e.
  U diag(1/sqrt(eigenvalue + 0.1 x mean eigenvalue)) U^T of the training covariance; normalise). A fixed feature map
  phi into d dimensions: "lin" (Gaussian random projection), "sgn" (its sign) or "sgn3" (product of three
  independent signs: the angular kernel cubed, a sharper kernel). Write: S = sum over stored facts of
  phi(key) * c(value), elementwise, c a random +-1 code per value (MAP binding). Store S quantized at q in {1, 2, 4}
  bits per number (q = 1: sign; otherwise Max's uniform quantizer on S / std, per block of 8,192 numbers, each block
  with a 32-bit scale). Read: score(v) = sum_j Q(S)_j phi_j(query) c_j(v), answer argmax over the V values. d is the
  largest that fits B (32 bits per block charged for every q). The projections and the value codes are drawn once per
  (encoder, preprocessing, phi) from fixed seeds, block by block, and are the same for every store (a fixed part of
  the memory, like the index's LSH hyperplanes or PQ codebooks); a shorter vector is the prefix of a longer one.
- **IDX** (the equally engineered table). Per-fact entries: a b-bit code of the key and the value; read: the stored
  entry nearest to the query (asymmetric: the query stays exact), answer its value. Code families: SimHash (random
  hyperplanes, scored by the query's projections), ITQ (PCA to b dimensions + iterative quantization), PCA + 1-bit
  scalar quantization, FAISS PQ with 4- or 8-bit sub-quantizers (ADC), OPQ + 8-bit PQ (ADC). Each family uses the
  largest code it can make with at most b bits.
- **Both may store a share f in {1, 1/2, 1/4, 1/8} of the facts** (the same facts for both: prefixes of one seeded
  permutation of the store, independent of which entities are queried), so that at small budgets the table can trade
  coverage for longer codes; the VSA gets the same option.
- **Accounting (primary: ideal).** "ideal" gives each memory its information-theoretic best: the table's entries are
  an unordered multiset of (code, value) pairs stored with ideal enumerative coding, log2 C(2^b V + M_s - 1, M_s) bits
  (order carries no information, so a set of codes costs about M_s (b + log2 V - log2 M_s + 1.44) bits, which gives
  the table up to 8 more key bits per fact than fixed-width storage); the VSA's numbers cost the entropy of the
  quantizer's output for a Gaussian (1, 1.904, 3.602 bits for q = 1, 2, 4). "raw" (secondary) is fixed-width
  storage: the table pays b bits per entry plus the values (per entry, ceil(log2 V) bits, or implied by grouping the
  entries by value, V ceil(log2(M_s + 1)) bits, whichever is cheaper); the VSA pays q bits per number.
- **Sketches are not a table here.** A Bloom filter or count-min sketch per value is a superposition structure (a VSA
  subclass, Kleyko et al. 2022); the question is superposition against per-fact storage.
- **EXACT**: the index with float32 codes (unconstrained bits): the upper bound of similarity look-up with this reader.
- **String look-ups** (unconstrained, context; encoder-independent): DICT (exact table on normalized labels: answers no
  alias by construction), TRI (nearest label by character-trigram count cosine: B26's N) and WR (rapidfuzz WRatio).
- **Oracle dictionary** keyed by the true entity: 100% by construction (the ceiling if aliases were resolved).

## Grid, development and choice (the only tuning)

- Cells: M in {256, 1024} x V in {2, 16, 128, 1024} x bpf in {4, 8, 16, 32, 64} x accounting {ideal, raw}.
- VSA grid per cell: prep {center, whiten} x phi {lin, sgn, sgn3} x q {1, 2, 4} x f {1, 1/2, 1/4, 1/8} = 72.
  IDX grid: prep {center, whiten} x family {lsh, itq, sq1, pq4, pq8, opq8} x f {1, 1/2, 1/4, 1/8} = 48.
  EXACT: prep (2).
- Dev stores: 6 replicates at M = 256 and 3 at M = 1024 for each V (dev split, about 3,000-3,500 queries per cell).
- Choice (b32_run.py choose): per encoder, cell and memory, the configuration with the highest mean dev accuracy over
  the dev stores; configurations within 0.1 points of the best count as tied, and ties go to the first in grid order.
  Written to results/b32_choice_{qwen8,e5}.json and committed before any test run.

## Test

- Stores: 20 replicates at M = 256 and 5 at M = 1024 for each V, drawn from the test split (about 4,500 queries per
  cell; the same stores for both encoders and all memories). Each memory answers with its chosen configuration.
  Per-query predictions: b32_out/test/preds_{enc}_M{M}_V{V}.jsonl.gz.

## Metric and statistics

- Accuracy: share of queries whose answer equals the gold value. Chance = 1 / V.
- Cell difference: accuracy(VSA) - accuracy(IDX), paired on the same queries; 95% interval from a cluster bootstrap
  (clusters = an entity within a store; 10,000 resamples); a Bonferroni interval over the 40 cells of one encoder and
  accounting (99.875%); exact two-sided sign test on the queries that exactly one memory answers right.
- Region difference: the mean of its cells' differences; clusters resampled within each cell independently.
- Lift: (accuracy - 1/V) / (EXACT - 1/V), the share of the exact index's above-chance accuracy that a memory keeps
  (descriptive).

## Bars (primary: Qwen3-8B codes, ideal accounting)

- **R1** (the region where the pilots and theory place a VSA win, fixed now): V in {2, 16}, bpf in {4, 8, 16}, both
  M: 12 cells. **R2** (where they place an index win: many distinct values, so the VSA's clean-up must tell apart as
  many candidates as the index's search): V = 1,024, every bpf and M: 10 cells.
- **H1 (primary):** over R1, VSA - IDX >= +2.0 points and the 95% interval lies above 0.
- **H2:** over R2, VSA - IDX <= -2.0 points and the 95% interval lies below 0.
- **Replication:** H1's bar on the e5 codes. **Secondary:** H1's bar under raw accounting.
- **Cell map:** a cell is a VSA win if its difference is >= +2 points and its Bonferroni interval excludes 0; an index
  win if <= -2 points and its Bonferroni interval excludes 0; otherwise tie / inconclusive.
- **Reporting rule:** no advantage of either memory is claimed in a cell or region whose interval includes 0.

## How each outcome will be read (fixed now)

- **H1 passes and replicates:** the vector form beats an equally engineered index at equal bits when few values are
  stored and the budget is a few bits per fact, with two different neural readers writing continuous codes. The paper
  may say so, with the region, the accuracies (and their distance from EXACT and from chance) and the mechanism
  (no bits spent on keys); it must also say where the index wins (H2) and that with unlimited bits the index is the
  upper bound.
- **H1 passes but does not replicate on e5:** a win for one reader only; the paper says so and names the reader.
- **H1 passes only under raw accounting:** the win rests on the table storing its entries in a fixed width; with an
  information-theoretically efficient table it disappears; the paper may not claim a win.
- **H1 fails (tie):** no advantage over an equally engineered index even where the vector form should be strongest.
- **H1 fails with the interval below 0:** the index wins there too; the vector form brings no accuracy at equal bits.
- **H2 fails:** with 1,024 values the index does not win by the margin; reported as found.
- Usefulness, registered now: a win counts as useful only if the VSA's accuracy in the winning cells is at least 10
  points above chance; otherwise it is reported as a win in a regime where both memories are near chance.

## Predictions (stated so they can be wrong; informed by the dev pilots)

P1. H1 passes on the Qwen3-8B codes: R1 mean VSA - IDX between +4 and +11 points (pilot 3, one dev store per cell:
    +7.2).
P2. H1 replicates on e5, with a smaller margin (+2 to +7 points): the VSA's crosstalk grows with the number of facts
    per dimension of the key space (768 against 4,096 numbers).
P3. H1 also passes under raw accounting, with a margin at least as large as under ideal accounting.
P4. H2 passes: with 1,024 values the index leads by 3 to 9 points (pilot 3: -5.2).
P5. Cell map (Qwen3-8B, ideal): at least 4 of the 12 R1 cells are VSA wins at the Bonferroni level, including all
    three at M = 1,024, V = 16 (pilot: +12 to +14); no V = 1,024 cell is a VSA win and at least 3 are index wins; at
    V = 2 and 64 bits per fact the index is ahead or level (the VSA's crosstalk floor; pilot -3.3 and -2.2).
P6. Usefulness: in every R1 cell that is a VSA win, the VSA is at least 10 points above chance (pilot: V = 2 about
    0.63-0.73 against 0.5; V = 16 about 0.26-0.42 against 0.06).
P7. Both memories stay far below EXACT at these budgets: in R1 the VSA keeps at most 60% of EXACT's above-chance
    accuracy at 4-16 bits per fact.
P8. Dev choices: the VSA mostly takes phi = lin, q = 1, whitened codes, all facts stored; the index mostly takes LSH
    on the Qwen3-8B codes and ITQ / PCA-sign on e5, and stores only a share of the facts (f < 1) at 4-8 bits per fact
    in most cells; sgn3 and q = 4 are never chosen at 4-16 bits per fact.
P9. EXACT (unconstrained) is at least 10 points above TRI and WR at every M and V; DICT answers nothing.
P10. Independent re-derivation agrees on >= 99.5% of queries for each memory; the bits audit finds no configuration
    over budget.

## Checks (second angle)

- b32_check.py score: an independent scorer (no shared code) recomputes every cell and region from the dumps
  (difference from the verdict must be < 1e-9) and gives a coarser store-level bootstrap interval for R1 and R2.
- b32_check.py rederive: an independent implementation (torch, blocks materialised, codes re-derived from the written
  seed specification; LSH / ITQ / SQ re-implemented; PQ / OPQ through FAISS's codes and centroids with our own ADC)
  re-derives the predictions of a random 10% of test stores for every chosen configuration; agreement >= 99% of
  queries per memory is required, and any lower agreement is investigated before the verdict is reported.
- b32_check.py bits: every chosen configuration fits its budget under its accounting.
- Hand reading (b32_handread.py): 10 queries where only the VSA is right and 10 where only the index is right, in the
  R1 cell with the largest difference and in one R2 cell, with alias, label and the exact index's nearest label.
- Literature after the result: support / conflict / unrelated for each claim (HD for open-vocabulary retrieval,
  vector databases and compressed nearest-neighbour search, capacity of linear associative memories, Kanerva 2009/2010).

## Compute

- A800 (idle GPUs only, never GPU 3, b32_run_gpu.sh claims and logs them): the Qwen3-8B encoding (GPU 4, 89 s) and
  the VSA part of the Qwen3-8B dev grid and test (b32_run.py --part vsa --device cuda; the codes are already on the
  box, and uploads to it run at a few KB/s). Local CPU (16 cores, torch / FAISS): the index part for Qwen3-8B and both
  parts for e5. The same code runs on both (torch float32, TF32 off; selftest on each device). Ledger:
  b32_out/gpu_ledger.txt (wall-clock x GPUs, every B32 GPU process, failed ones included). Cap: 40 GPU-hours
  (the coordinator allows up to 1.5x if it speeds things up; any use above 40 is recorded). No paid API.

## Files

- Code: b32_data.py, b32_embed.py, b32_lm_select.py, b32_common.py, b32_run.py, b32_verdict.py, b32_check.py,
  b32_handread.py, b32_run_gpu.sh; pilots b32_pilot.py, b32_pilot2.py, b32_pilot3.py.
- Data: b32_data/ (entities, strings, codes; codes not committed: regenerable with b32_embed.py), b32_out/ (pilots,
  dev grid, test predictions).
- Results: results/b32_choice_{qwen8,e5}.json, results/b32_verdict.{json,txt}, results/b32_rederive.json.

## Pilot record (dev split only, before this file)

All on dev stores (dev split), one store per cell unless stated; accuracy = share of alias queries answered right.
- **Pilot 1** (b32_pilot.py, bge-small codes centred, every fact stored, raw accounting, M = 1,024; b32_out/
  pilot1_bge_center.json). EXACT 0.855 / 0.704 / 0.681 / ~0.68 for V = 2 / 16 / 256 / 1,024. At 8 bits per fact the
  VSA was level with the best index at V = 2 (0.585 vs 0.593) and ahead at V = 16 (0.151 vs 0.092) and V = 256 (0.073
  vs 0.060); from 32 bits per fact the index led everywhere (V = 2 at 128: 0.834 vs 0.692; V = 16 at 128: 0.657 vs
  0.305), and the VSA with lin / sgn features stayed near 0.59 at V = 2 at every budget (the SUM crosstalk floor of
  centred, non-whitened codes). This led to adding whitening, the stored share f for both memories, and the ideal
  accounting (the table's strongest form) before any comparison was registered.
- **Pilot 2** (b32_pilot2.py, e5 codes, M = 1,024; stopped part-way; b32_out/pilot2_e5_M1024_V{lo,hi}.json).
  Ideal accounting: V = 2 at 4 / 8 / 16 bits per fact: VSA 0.677 / 0.661 / 0.685 vs index 0.603 / 0.603 / 0.680;
  V = 128: 0.075 / 0.120 / 0.225 vs 0.084 / 0.137 / 0.243. Raw: V = 2 at 32: 0.683 vs 0.732; V = 16 at 4 / 8: 0.202 /
  0.225 vs 0.168 / 0.205. EXACT 0.873 (V = 2), 0.728 (V = 128).
- **Pilot 3** (b32_pilot3.py, Qwen3-8B codes, ideal accounting, reduced grid: no sgn3, q in {1, 2}, families LSH /
  ITQ / SQ1 / PQ4; b32_out/pilot3_qwen8.{json,log}). VSA - index at 4 / 8 / 16 / 32 / 64 bits per fact:
  - M = 1,024: V = 2 +7.9 / +7.7 / +7.2 / +2.3 / -3.3; V = 16 +12.2 / +12.3 / +13.7 / +10.8 / +4.0; V = 128 -0.1 / -0.6
    / +2.2 / +3.6 / +1.5; V = 1,024 -1.5 / +1.2 / -3.9 / -3.9 / -2.3.
  - M = 256 (about 280 queries per store, so about +-3 points of noise): V = 2 +5.0 / +8.6 / +7.9 / -0.4 / -2.2; V = 16
    +0.4 / +8.3 / -4.3 / +6.8 / -4.7; V = 128 +0.7 / -1.4 / -3.1 / +1.0 / -5.9; V = 1,024 -5.7 / -5.7 / -9.6 / -9.1 / -11.0.
  - EXACT (whitened codes) 0.923 / 0.836 / 0.801 / 0.827 (M = 1,024) and 0.910 / 0.888 / 0.871 / 0.789 (M = 256).
- The pilots used b32_common.vsa_answer (per-store projections, PCA whitening); the registered run uses b32_run.py
  (fixed per-encoder projections, ZCA whitening: the same inner products and the same distribution of features).

## Dev record (2026-11-18 US Eastern, dev split only; written before any test accuracy was read)

- Dev grid: b32_out/dev_{qwen8,e5}_{vsa,idx}.jsonl (36 dev stores per encoder; VSA 120 configurations per store,
  index 480 plus EXACT). Choices: results/b32_choice_{qwen8,e5}.json (summaries results/b32_devsum_{qwen8,e5}.txt).
  The Qwen3-8B VSA choices were committed first (results/b32_choice_qwen8_vsa.json, 109d524) while the index grid
  was still running (compute notes 1-2); the final choice file has the same 80 VSA entries.
- What the grid chose (ideal accounting, 40 cells per encoder): VSA phi = lin in 39 (Qwen3-8B) and 37 (e5) cells,
  q = 1 in 36 and 39; all facts stored in 30 and 29. Index: LSH in 35 Qwen3-8B cells (ITQ 1, PCA-sign 3, PQ4 1);
  on e5 ITQ 19, PCA-sign 14, LSH 5, PQ8 1, OPQ8 1. The index stores only a share of the facts (f < 1) in 15 of 16
  (Qwen3-8B) and 13 of 16 (e5) cells at 4-8 bits per fact. P8 holds except that sgn3 was chosen once (e5, M = 1,024,
  V = 2, 64 bits per fact, outside 4-16 bits) and q = 4 once at 8 bits per fact (Qwen3-8B, M = 256, V = 2).
- Dev differences, VSA - index (points; selected configurations, so optimistic for both memories): R1 mean +8.5
  (Qwen3-8B; every R1 cell between +4.3 and +13.6) and +1.2 (e5; -5.4 to +6.5). V = 1,024: -1.3 to -6.8 (Qwen3-8B),
  -3.1 to -15.0 (e5). e5 at 32-64 bits per fact: the index leads in every cell (-5.6 to -24.9).
- The test of the e5 codes and the VSA test of the Qwen3-8B codes ran after their choices were committed (69b81f3,
  109d524); no test accuracy was read before this record. The e5 re-derivation (b32_check.py rederive, 25% of the
  e5 test stores) was run and only its agreement column read: VSA lin 100%, sgn 98.9%, sgn3 98.6%; index LSH 100%,
  PCA-sign 100%, ITQ 85.6%, PQ4 57.0%, PQ8 37.8%, OPQ8 64.6%. The families with iterative training (ITQ, PQ, OPQ)
  do not reproduce bit for bit: the checker fits them again from float64 codes and lands in different local optima;
  the sign-feature VSA disagreements are projections within rounding of zero. Per the registered rule this is
  investigated in the verdict (accuracy of the re-fitted index against the logged one), not hidden.
