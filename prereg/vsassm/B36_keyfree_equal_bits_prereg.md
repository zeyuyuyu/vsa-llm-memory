# B36 pre-registration: the VSA store against classical memories that keep no keys, at the same number of bits

Date: 2026-11-18 (US Eastern; box clock Asia/Shanghai). Written before the B36 dev grid of the new families and before
any B36 memory answers a test query.

Done before this file (nothing on test queries):
- Literature check (results/b36_literature.txt, 45 verified entries; summary below).
- Code: b36_common.py (the new families), b36_run.py (dev grid, choice, test, merge; B32's VSA and index grids at 1-2
  bits per fact by B32's own code), b36_verdict.py (bars, mechanical), b36_check.py (independent scorer, bits audit,
  independent re-derivation), b36_handread.py, b36_devsum.py. `b36_run.py selftest` passes.
- B32's VSA and index dev grids at 1 and 2 bits per fact (b36_run.py dev32; B32's grids and code unchanged; dev split
  only; b36_out/dev32_*.jsonl). They involve no B36 design choice.
- Pilots on the dev split only, Qwen3-8B codes, three dev stores (M = 256, V = 2; M = 1,024, V = 16; M = 1,024,
  V = 1,024; rep 0), every new family's whole grid (b36_out/pilot_qwen8.jsonl). Numbers at the end. They fixed code
  faults (multi-probe with fewer sub-quantizers than probes; column limits; a cache race), and led to adding the count
  content to the LSH sketch and PQ k = 4 (16-cell) signatures; the grids below are the final pilot grids. The PQ
  codebooks of every pool sub-space were then fitted once per encoder and preprocessing (b36_run.py pqfit; training
  split only) and are read from b36_out/pqcache.
- Disclosure: the VSA's and the index's test predictions at 4-64 bits per fact are B32's and were read in B32 (and
  their R1 difference, +8.06, is public in results/b32_verdict.txt). What B36 has not seen is every new family's and
  BEST's test accuracy, and the VSA's and index's at 1-2 bits per fact.

## Why

B32 found that the VSA store beats a dev-tuned per-fact vector index at equal bits when few values are stored
(R1: V in {2, 16}, 4-16 bits per fact; +8.06 [7.31, 8.81] points on Qwen3-8B codes, +3.31 on e5) and attributed it to
the store spending no bits on keys. A referee: "the competitor stores a key code with every fact. Untested are the
classical structures that store a key->value map without storing keys: Bloomier filters and ribbon/XOR retrieval at
about log2 V bits per key, Bloom/count-min sketches and sparse distributed memory, each keyed by LSH or PQ signatures.
Required: full rate-accuracy curves against these structures under multi-table LSH or PQ, on both encoders." Also: the
win is narrow and in a low-accuracy regime (60-74% at V = 2). B36 is that test. It goes into the paper whatever it
shows.

## What the literature says (results/b36_literature.txt; tags there)

- Static functions (Bloomier [CKRT04], ribbon [DW21, DHSW22], XOR [GL20]) store n keys' r-bit values in about n r bits
  (the lower bound; BuRR within 1%), and return an arbitrary value for any key outside the set. Keyed by an LSH
  signature, an alias is answered from memory only if its signature equals the label's (or a probed neighbour's).
- LSH-keyed Bloom filters work only for small perturbations (eps = O(1/log n), [KM06]); fuzzy keys turn N pairs into
  N^(1+O(eps)) bucket pairs [PWZ21]; zero-error distance-sensitive filters cost as much as storing the set [GPSS17].
- Count-min with block-sparse keys IS a VSA (sparse block codes) [CUY23]; Bloom filters are a special case of HDC
  [TDR21, KRRO22, KRGO20]. The referee's arms and the store are one family with different feature maps; B36 asks
  which member is most bit-efficient for alias queries.
- At exact keys a static function returns ~0.74-0.99 bits per stored bit, a superposed store with one-shot clean-up at
  most ~0.5 [FKS18]: the VSA should not win there, and the one careful equal-resource comparison [KPS10] favoured
  look-up tables in most regimes. LSH count arrays already give sub-linear near-neighbour sketches [CBS20].
- Gap: no published equal-bits comparison of a VSA/HRR/HDC store with Bloomier/ribbon/XOR, Bloom, count-min or SDM,
  with or without LSH/PQ keys; no paper keys a static function by LSH signatures.

## Measured before the design (dev split, Qwen3-8B, 1,838 dev labels and their aliases)

Cosine between an alias's code and its label's code (preprocessed): median 0.51 (centred) / 0.43 (whitened), 10th
percentile 0.11 / 0.04; between two random labels: median 0.0, 99th percentile 0.33 / 0.05. So an alias is far from its
label in absolute terms but far nearer than any other entity. The chance that all k SimHash bits of an alias equal its
label's: 0.46 / 0.23 / 0.12 / 0.07 / 0.04 / 0.01 at k = 2 / 4 / 6 / 8 / 10 / 16 (whitened 0.44 ... 0.01). A structure
that needs an exact signature match must therefore either use short signatures (crowded buckets) or many tables and
probes; a similarity read (the VSA, the index) uses the graded similarity directly. This is the regime the bar tests.

## Setting (B32's, unchanged)

PopQA entities with Wikidata aliases, random values from V symbols, M facts per store written under the label, every
alias of a stored entity a query (unseen string). Keys: frozen Qwen3-8B mean layer-9 codes (primary) and e5-base-v2
(replication). B32's splits, dev stores (6 at M = 256, 3 at M = 1,024 per V) and test stores (20 / 5 per V; about 4,500
queries per cell), same seeds: every memory answers the same queries.

## Memories

Budget B = bpf x M bits per store, bpf in {1, 2, 4, 8, 16, 32, 64}; V in {2, 16, 128, 1024}; M in {256, 1024}. Free for
every memory (as in B32): the encoder, the value vocabulary, preprocessing fitted on the training split (centre; centre
+ ZCA whitening), and fixed random or training-fitted components drawn from seeds (projection pools, hash seeds, value
codes, PQ codebooks fitted on training codes). Counted: everything that depends on the store's facts. Every new family
also pays ceil(log2 V) bits for a default value (the store's plurality value), answered when nothing votes; ties go to
the default value, then to the smallest value.

Signatures (shared by SF, CM, BF; b36_common.py): codes are preprocessed and unit-norm; a fixed Gaussian projection pool
(65,536 columns, seeded per block). SimHash ("sim"): table t with k bits takes the signs of pool columns [t k, t k + k).
PQ ("pq"): sub-quantizer s = 16 centroids (k-means on the training codes: 16 seeded training points, 20 Lloyd
iterations) in the 16-dimensional sub-space of pool columns [16 s, 16 s + 16); a k-bit signature is k/4 consecutive sub-quantizers. Multi-probe (Lv et al. 2007): probe 0 is the
exact signature (vote weight 1); probes 1..T-1 flip the least confident bits (smallest |projection|) or replace one
sub-quantizer's code by its second-, third- or fourth-nearest centroid (smallest distance gap), vote weight 0.5.

- **VSA** (B32's store, unchanged): B32's dev-chosen configurations at 4-64 bits per fact (B32's committed choices and
  test predictions); at 1-2 bits per fact B32's grid (prep x {lin, sgn, sgn3} x q {1, 2, 4} x stored share {1, 1/2,
  1/4, 1/8}: 72) and choice rule on the new dev runs.
- **IDX** (B32's per-fact index, unchanged; key codes stored per fact; SimHash / ITQ / PCA-sign / PQ / OPQ; ideal set
  coding): same as the VSA (B32's choices at 4-64, B32's grid and rule at 1-2).
- **SF, static-function retrieval** (standard ribbon, width 64, Dillinger & Walzer 2021): L tables; table t stores, for
  each distinct signature among the stored facts (sim or pq, k bits), the plurality value of the facts that share it
  plus a 4-bit fingerprint of the signature; a configuration keeps the top f in {0, 1, 2, 4} fingerprint bits (bit
  slices are independent, so one build serves every f). A query probes T in {1, 4} signatures in each table; an
  output votes for its value if its fingerprint matches (always, if f = 0). Tables are added while the budget lasts;
  if not even one table fits, one table stores the largest prefix of the store's (B32) fact order that fits.
  Grid: prep (2) x {sim: k in 6, 8, 10, 12, 14, 16; pq: k in 4, 8, 12, 16, 20} x f (4) x T (2) = 176.
- **CM, LSH bucket sketch** (count-min / count sketch / RACE-style counters keyed by LSH): L tables of 2^k buckets;
  each bucket holds its stored facts' plurality value or EMPTY ("plural"), or for V <= 16 the set of values present
  (one Bloom bit per value, "pres"), or the vector of per-value counts ("count"); entropy-coded. A query adds its
  probed buckets' votes (count: the counts). Grid: prep x {sim: k in 2, 4, ..., 14; pq: k in 4, 8, 12, 16} x content
  {plural, pres, count} x T {1, 4} = 132 for V <= 16 (44, plurality only, for V >= 128).
- **BF, Bloom filter of (table, signature, value)**: one bit array of m = B - ceil(log2 V) bits, the distinct items of
  L tables, h = round(ln 2 m / items) hash functions (1..16, double hashing); a query scores each value by the number
  of tables where its (table, signature, value) tests present. Grid: prep x {sim: k in 8, 12, 16, 20; pq: k in 8, 16}
  x L {1, 2, 4, 8, 16} = 60.
- **FBF, fly Bloom filter per value** (Dasgupta et al. 2018; FlyNN, Sinha & Ram 2021): winner-take-all fly hash = the
  rho largest of the first m pool projections; one m-bit array per value; a key sets its winners in its value's array;
  read = number of the query's winners set in each array. m: largest that fits. Grid: prep x rho {1, 2, 4, 8, 16, 32} = 12.
- **SDM, sparse distributed memory** (Kanerva 1988) on the continuous codes: hard locations with Gaussian addresses
  (pool columns), active if the projection exceeds the (1 - p) quantile of N(0, 1); nd counters per location; counter
  u stores the sum of random +-1 value codes c_u(value) of the stored facts that activate its location; ternary
  (sign) or exact counts; read: score(v) = sum over the query's active counters of Q(S_u) c_u(v). D (counters) = the
  largest prefix that fits, at most min(2 B, 65,536). Grid: prep x p {0.003, 0.01, 0.03, 0.1, 0.3} x nd {1, 8} x q
  {tern, cnt} = 40.

Grid sizes: VSA 72; competitors IDX 48 + SF 176 + CM 132 + BF 60 + FBF 12 + SDM 40 = 468 (generous to the competitors).
Limits: every LSH family reads at most the 65,536 pool columns (at most 65,536 / k SimHash tables, 4,096 / (k / 4) PQ
tables); CM at most 16,384 tables; SDM at most 65,536 counters; FBF at most 65,536 cells.
Not included, with reasons: M = 4,096 (the dev split has 1,838 entities, so no dev store of that size exists, and test
stores of 4,096 of 4,622 test entities would share 89% of their entities); supervised (alias-trained) hashing or
metrics (they would change the reader, which all memories share); a least-squares (pseudo-inverse) associative memory
(a superposition memory, not a key-free classical structure; noted as a limitation).

## Accounting

- **ideal (primary)**: each memory at its information-theoretic best. VSA: B32's (entropy of the Gaussian quantizer's
  output, plus 32 bits per block). IDX: B32's (unordered multiset of (code, value) pairs, enumerative coding). SF:
  n_t (r + f) bits per table, r = log2 V (the retrieval lower bound; more generous than the ribbon actually built).
  CM: 2^k H0 per table (empirical zeroth-order entropy of the bucket symbols, no header). BF: m. FBF: V m H2(fill).
  SDM: D H0 of the stored counter symbols.
- **real (secondary)**: what is stored. VSA / IDX: B32's raw accounting. SF: the ribbon actually built, m_t (r + f) + 32
  per table (m_t >= 1.03 n_t + 4). CM: 2^k ceil(log2(V + 1)) (plural), 2^k V (pres), 2^k V ceil(log2(max count + 1))
  (count). BF: m. FBF: V m. SDM: 2 bits per ternary counter, ceil(log2(2 max|S| + 1)) per exact counter, + 8.

## Development and choice

- Dev: every configuration of every new family on B32's 36 dev stores per encoder (b36_run.py dev), both accountings.
- Choice (b36_run.py choose): per (encoder, M, V, bpf, accounting) and family, the configuration with the highest mean
  dev accuracy over the dev stores; within 0.1 points tied, first in grid order. VSA and IDX: B32's rule (B32's
  committed choices at 4-64 bits per fact). **BEST**: per cell, the competitor family (IDX, SF, CM, BF, FBF, SDM) whose
  chosen configuration has the highest dev accuracy (ties: that order). Written to results/b36_choice_{qwen8,e5}.json and
  committed before any test run.

## Test

Each family answers B32's test stores with its chosen configurations (b36_run.py test; B32's VSA / IDX at 1-2 bits per
fact by b36_run.py test32); merged with B32's test predictions (4-64 bits per fact) into b36_out/test/all_*.jsonl.gz.
Run once.

## Metric and statistics

- Accuracy = share of alias queries answered with the gold value. Cell difference VSA - BEST paired on the same queries;
  95% cluster bootstrap (clusters = an entity within a store; 10,000 resamples); Bonferroni interval over the 56 cells
  of one encoder and accounting; exact sign test on discordant queries. Region difference = mean of its cells'
  differences, clusters resampled within each cell.
- Regions: **R1** = V in {2, 16}, bpf in {4, 8, 16}, both M (B32's, 12 cells); **R0** = V in {2, 16}, bpf in {1, 2}
  (8 cells, new); **RH** = V in {2, 16}, bpf in {32, 64} (8 cells); **R2** = V = 1,024, every bpf (14 cells).
- **Rate at equal accuracy** (the compression statement): for each memory, (M, V), accounting and encoder, the bits per
  fact at which the running maximum of its test accuracy curve first reaches chance + lift x (EXACT - chance), lift in
  {0.25, 0.5}, log-linear interpolation between grid points ("<= 1" and "> 64" reported as bounds); ratio BEST / VSA
  with a 95% cluster-bootstrap interval (1,000 resamples).
- Sensitivity: ORACLE = per cell the competitor family with the highest TEST accuracy (selection on test; the most
  favourable competitor possible), reported for every region; VSA minus each family's own chosen configuration.

## Bars (primary: Qwen3-8B codes, ideal accounting)

- **H1 (primary):** over R1, VSA - BEST >= +2.0 points and the 95% interval lies above 0.
- **Replication:** H1's bar on the e5 codes. **Secondary:** H1's bar under real accounting; the same difference over R0.
- **Cell map:** VSA WIN if the cell difference >= +2 points and its Bonferroni interval excludes 0; BEST WIN if <= -2 and
  the Bonferroni interval excludes 0; otherwise tie / inconclusive. A VSA win counts as useful only if the VSA is at
  least 10 points above chance there.
- **Reporting rule:** no advantage of either side is claimed in a cell or region whose interval includes 0.

## How each outcome will be read (fixed now)

- **H1 passes and replicates on e5:** at equal bits, in the few-values, low-budget region, the VSA store beats the best
  dev-tuned classical memory that keeps no keys (static functions, sketches, Bloom filters, SDM, each keyed by
  LSH/PQ signatures with multi-table and multi-probe reads) as well as the per-fact index, with two neural readers.
  The paper may say so with the region, accuracies, lifts, the rate ratio, the mechanism (aliases share little
  absolute similarity with their labels, so exact-signature structures rarely match; the store reads graded
  similarity without spending bits on keys) and where it loses (V = 1,024; high budgets; exact keys per [FKS18]).
- **H1 passes on Qwen3-8B only:** the win depends on the reader; the paper names the reader.
- **H1 passes only under real accounting:** the win rests on how the competitors are stored; no win is claimed.
- **H1 fails (tie):** the key-free structures close the gap B32 found; B32's claim must be withdrawn as "no bits on
  keys" and restated as a tie with the best classical memory.
- **H1 fails with the interval below 0:** a classical key-free memory beats the store where it was strongest; the paper
  reports that and drops the bit-efficiency claim.
- If BEST is a superposition-type sketch (CM count, SDM) and wins, the paper says that a superposition memory with a
  different feature map beats the dense MAP store; the "no bits on keys" mechanism stands, the VSA-specific claim does
  not.

## Predictions (stated so they can be wrong; informed by the pilots)

Accuracies in points; Qwen3-8B, ideal accounting unless stated.

P1. H1 passes on Qwen3-8B: R1 mean VSA - BEST between +4 and +10 (B32's VSA - IDX was +8.06; pilots put the key-free
    families ahead of the index only at M = 1,024, V = 16, 4 bits per fact, still 5 points behind the VSA).
P2. On e5 the R1 difference lies between -1 and +4 (B32's e5 VSA - IDX: +3.31). Whether it clears the +2 bar is not
    predicted.
P3. BEST is the index in at least 8 of the 12 R1 cells; a key-free family is BEST in at most 4.
P4. Every key-free family on its own trails the VSA over R1 by at least 5 points (VSA - family >= +5); BF and SDM by
    at least 8.
P5. R0 (1-2 bits per fact): VSA - BEST between -4 and +4; at M = 1,024, V = 16 and 1 bit per fact BEST is a key-free
    family (a coarse PQ sketch or static function) and leads the VSA by at least 2 points (pilot: 17.0 vs 12.0).
P6. R2 (V = 1,024): VSA - BEST <= -2 with the interval below 0 on both encoders; BEST is the index in at least 12 of
    the 14 cells.
P7. RH (32-64 bits per fact, V <= 16): VSA - BEST between -1 and +7 on Qwen3-8B, below -4 on e5.
P8. Rate at equal accuracy (lift 0.25): in at least 3 of the 4 settings (M, V in {2, 16}) BEST needs at least 1.5
    times the VSA's bits per fact.
P9. At 64 bits per fact the best key-free family (its own dev choice) is at least 10 points below the index in every
    (M, V) (pilot: 14-21 points).
P10. Dev choices: at <= 8 bits per fact SF and CM choose coarse signatures (PQ k <= 8 or SimHash k <= 6) in most
    cells; BF is never BEST.
P11. Checks: independent scorer |difference| < 1e-9; 0 configurations over budget; re-derivation agreement >= 99% for
    SF, CM, BF, FBF and >= 98% for SDM (activations at the float32 threshold may flip between implementations).

## Checks (second angle)

- b36_check.py score: an independent scorer recomputes every cell's VSA - BEST from the dumps (|difference| from the
  verdict < 1e-9) and gives a store-level bootstrap for R1.
- b36_check.py bits: every logged bit count of every new-family configuration on every test store is within its
  budget; VSA dimensions and IDX key bits re-checked from the written rules.
- b36_check.py rederive: an independent implementation (pure-Python splitmix64, ribbon, sketches and Bloom filter;
  float32 preprocessing re-implemented; only the PQ codebooks read from the fit cache) re-derives the predictions of a
  random 10% of test stores (at least one per (M, V)) for every family's chosen configuration at every budget under
  ideal accounting and at 4 and 16 bits per fact under real accounting; agreement >= 99% per family is required, and
  lower agreement is investigated before the verdict is reported. Partial SF tables are not re-derived (counted).
- b36_handread.py: 10 queries only the VSA answers right and 10 only BEST answers right, in the R1 cell with the
  largest difference and in the cell where BEST leads most, with alias, label, exact index's nearest label and cosine.
- Literature after the result: results/b36_literature.txt (support / conflict / gap), updated.

## Compute

Local CPU (16 cores, shared) and, when it is idle enough, CPU cores of the A800 box (no GPU needed; never GPU 3; no
other process touched). No paid API. Ledger: b36_out/compute_ledger.txt.

## Files

Code: b36_common.py, b36_run.py, b36_verdict.py, b36_check.py, b36_handread.py, b36_devsum.py. Data: b36_out/ (dev grid,
test predictions; not committed if large). Results: results/b36_literature.txt, results/b36_choice_{enc}.json,
results/b36_devsum_{enc}.txt, results/b36_verdict.{json,txt}, results/b36_rederive_{enc}.json, results/b36_check.txt,
results/b36_handread_{enc}.txt, results/b36_predictions.txt.

## Pilot record (dev split only, before this file)

All on dev stores (dev split), Qwen3-8B codes, ideal accounting, one dev store per cell for the new families
(about 280 queries at M = 256, 1,000-1,200 at M = 1,024: +-3 / +-1.5 points of noise); VSA and IDX are their 6- / 3-store
dev means (B32's dev files; dev32 at 1-2 bits per fact). b36_out/pilot_qwen8.jsonl (final code); an earlier run with
the first code (b36_out/pilot_qwen8_v1.jsonl) differed only by the missing count content and PQ k = 4.

| cell | bpf 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M 256, V 2: VSA | 59.1 | 61.4 | 62.4 | 64.2 | 70.1 | 77.3 | 79.8 |
| IDX | 55.1 | 56.4 | 55.4 | 59.9 | 62.5 | 71.1 | 76.2 |
| best key-free (family) | 59.7 SDM | 54.0 SF | 56.1 SF | 57.9 FBF | 61.2 SDM | 62.2 SF | 61.9 SF |
| M 1024, V 16: VSA | 12.0 | 14.4 | 21.4 | 26.5 | 35.9 | 47.6 | 56.9 |
| IDX | 9.9 | 10.7 | 12.5 | 18.1 | 26.2 | 38.2 | 51.5 |
| best key-free (family) | 17.0 CM | 16.4 SF | 16.2 CM | 17.3 CM | 19.8 FBF | 27.7 FBF | 34.1 FBF |
| M 1024, V 1024: VSA | 0.5 | 1.3 | 4.8 | 9.6 | 16.3 | 29.1 | 42.6 |
| IDX | 3.1 | 3.7 | 8.3 | 12.0 | 17.6 | 33.5 | 45.6 |
| best key-free (family) | 1.6 CM | 3.6 SDM | 2.3 SF | 5.0 SF | 7.2 CM | 12.3 FBF | 24.7 FBF |

- The key-free families' best configurations at <= 4 bits per fact use coarse buckets (PQ k = 4-12, SimHash k = 4,
  multi-probe T = 4); their accuracy rises little with budget (the alias rarely shares a fine signature with its
  label), while the VSA and the index keep rising.
- Run time per dev store (one process, shared CPU): 173 s (M 256, V 2), 342 s (M 1024, V 16), 461 s (M 1024, V 1024;
  BF 197 s, SDM 167 s before the SDM code cache).

## Dev record (2026-11-19, box clock; dev split only; written before any B36 test run)

- Dev grid: b36_out/dev_{qwen8,e5}.jsonl (36 dev stores per encoder, every configuration of SF / CM / BF / FBF / SDM
  under both accountings), after amendment 1 (prereg/B36_amendment1.md). B32 arms at 1-2 bits per fact:
  b36_out/dev32_*. Choices: results/b36_choice_{qwen8,e5}.json; per-family dev bests: results/b36_devsum_{enc}.txt.
- Dev differences, VSA - BEST (selected configurations; optimistic for both sides): R1 mean +6.83 (Qwen3-8B, ideal),
  +8.47 (Qwen3-8B, real), -0.14 (e5, ideal), +2.37 (e5, real). V = 1,024: VSA behind in 13 of 14 Qwen3-8B cells
  (-1.2 to -6.8). e5 at 32-64 bits per fact: VSA behind in every cell (-4.6 to -24.9).
- BEST family (ideal; 56 cells): Qwen3-8B IDX 36, CM 11, SF 4, FBF 3, SDM 2; in R1 IDX 4, FBF 3, CM 3, SF 1, SDM 1.
  e5: IDX 39, CM 13, FBF 4; in R1 IDX 5, CM 5, FBF 2. Key-free families are BEST mostly at <= 8 bits per fact.
  So prediction P3 (index BEST in >= 8 of 12 R1 cells) already fails on the dev choice (4 of 12); recorded now.
- Choices at <= 8 bits per fact: CM always coarse (k = 4 in 24 / 20 of 32 cells, PQ in 22 / 17); SF coarse in 22 / 21
  of 32 (PQ k 4-8 or SimHash k 6), multi-probe T = 4 in 23 / 24; BF mostly PQ k = 8; FBF mostly rho = 1-4; SDM mostly
  ternary with p = 0.01-0.03 and one counter per location (Qwen3-8B / e5). P10 holds on the dev choices.
- No test accuracy of any new family, of BEST, or of the VSA / index at 1-2 bits per fact has been computed.
