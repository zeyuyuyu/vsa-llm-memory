# V5 specification packet (sent to gpt-6.1-sol)

Assembled 2026-11-23 from the text of the paper "What vector symbolic memory adds to frozen language models beyond
exact tables" (npj Unconventional Computing submission): the Methods subsections on the equal-bit test and the
equal-cell device simulation, Box 1, Supplementary Table "Implementation details" and the Supplementary descriptions of
the store, the corrected least squares (B60), the one-pass delta rule (B62), the device model, the coded tables, the
sparse-block store (B61) and the cuckoo insertable table (B63). Every result number of the paper (accuracies,
differences, cells-per-pair requirements, ratios, counts of outcomes) is removed and shown as [redacted]. Parameters,
grids, definitions and data sizes are kept. Where the paper's text names a choice without its values, the value is
filled in from the study's pre-registration and marked [filled from prereg]. Sections 3 and 4 (data paths, formats,
deliverables) were written for this packet. Part A and Part B are sent as two separate tasks.

---------------------------------------------------------------------------------------------------------------------

## PART A. Equal-bit fact memories: the store against corrected least squares and a one-pass delta rule

### A.1 Paper text (Methods, "Equal-bit fact memories"; trimmed to what Part A needs, numbers of results redacted)

Entities are PopQA subjects with their Wikidata aliases (9,014 unambiguous aliases); the entities are split by a hash
into training (2,818 entities), development (1,838) and test (4,622) sets. A store holds M = 256 or 1,024 entities, each
with a value drawn uniformly from V symbols; the fact is written under the entity's label and queried by every alias of
the entity, which the memory has never seen. A name's code is the mean hidden state of one layer of a frozen encoder
over the name's tokens (already computed; files below). The VSA store centres the key code, or centres and whitens it,
with statistics of the training split, normalizes it, and maps it into d dimensions by a fixed Gaussian random
projection, the projection's sign, or the product of three independent such signs, with d the largest that fits the
budget; it binds the result elementwise to a +-1 code of its value (multiply-add-permute binding), sums the bound pairs
and keeps the sum at 1, 2 or 4 bits per number (each block of 8,192 numbers with a 32-bit scale); a read scores every
value against the query by a matched filter and returns the best. Its 72 configurations (preprocessing, map, bits per
number, share of facts stored) [...]. Every memory may store a share of the facts and is charged, in the primary
accounting, the information-theoretic minimum for what it stores [...]. Each memory's configuration was chosen per
condition on development stores (288 configurations for the corrected least-squares memory) and scored once on test.
The least-squares memory: a pre-specified run (B60) re-derived the comparisons on a corrected memory (principal
directions before whitening; V-1 or V columns; per-column or shared scales; six regularisation strengths); B62 wrote the
corrected memory one fact at a time by a normalised delta rule. The pre-specified region is V = 2 or 16 at 4 to 16
bits per fact (12 conditions); intervals are cluster bootstraps over entities.

### A.2 Box 1 (rows that Part A needs)

- Algebra: MAP (multiply-add-permute, bipolar) in the equal-bit test.
- Stores: the equal-bit store (MAP, PopQA facts): final design with antipodal (two values) or orthogonal (sixteen
  values) value codes and a matched-filter read.
- VSA recipe: random high-dimensional codes; bound key-value pairs summed by a Hebbian write, each pair added once, and
  read by a matched filter; clean-up to the nearest code.
- Least-squares associative memory: a linear map from key features to values fitted by ridge regression: a superposed
  memory whose write decorrelates the keys.
- Ideal accounting: each memory charged the information-theoretic minimum for what it stores (here: the entropy of
  each quantized number for a Gaussian input, plus the 32-bit scales).

### A.3 Supplementary text on the store's final design (B53) and the corrected memory (B60) and the delta rule (B62)

The store (final design): antipodal value codes for two values and orthogonal (Sylvester-Hadamard) codes for sixteen,
configuration chosen on development data over the same 72 configurations. The store maps each key code through a fixed
random feature map, binds it elementwise to a +-1 code of its value, adds the results into one vector and keeps that
vector at 1, 2 or 4 bits per number; a read scores every value against the query and takes the best. Either memory may
store only a share of the facts.

The corrected comparison (B60): the corrected memory uses principal directions computed from the raw codes' training
covariance before whitening, V-1 or V target columns, per-column or shared scales and six regularisation strengths in
the grid (288 configurations), chosen per condition on development data and scored once on test. The primary
comparison is the store's final design minus the corrected memory over the region, with two and with sixteen values
separately.

Writing one fact at a time (B62): one pass of a normalised delta rule (NLMS) in a fixed random order with the corrected
memory's feature, target and scale options, its step size and averaging chosen on development data by the same rule.

### A.4 Supplementary Table "Implementation details" (rows that Part A needs, verbatim)

- Equal-bit quantizer: One bit: the sign, zeros set to +-1 at random. Two or four bits: a uniform mid-rise quantizer
  with Max's Gaussian-optimal step on the block's standard deviation. Each block of 8,192 numbers carries a 32-bit
  scale; ideal accounting charges the output's entropy for a Gaussian input (1, 1.904 and 3.602 bits for one, two and
  four bits).
- Sixteen-value codes: c_j(v) = r_j H_16[v, j mod 16], with r a random +-1 vector and H_16 the Sylvester-Hadamard
  matrix of order 16.
- Equal-bit stores: Per condition six (256 facts) or three (1,024) development stores and twenty or five test stores;
  entities drawn without replacement from the split, values uniform, every alias of a stored entity a query.
- Ties: Configurations within 0.1 points of the best development accuracy count as tied and go to the first in grid
  order (preprocessing, map or family, bits, stored share); reads take the first maximum.
- Code preprocessing: Statistics from the codes of the training split's labels and their aliases. Centre: subtract
  their mean, then normalize to unit length. Whiten: subtract the mean, multiply by the symmetric (ZCA) matrix
  U diag(1/sqrt(w_i + 0.1 w_bar)) U^T of their covariance (eigenvalues w_i, mean w_bar; the constant is added to every
  eigenvalue), then normalize.
- Corrected least squares (B60): Ridge W = F^T (F F^T + lambda I)^{-1} T over the store's facts,
  lambda = alpha * mean diag(F F^T), alpha in {1e-4, 1e-2, 0.1, 1, 10, 100}; read: the value maximizing
  f(q)^T W_hat t(v). Features: random (a dense Gaussian projection of the preprocessed code) or principal (the
  preprocessed code's coordinates on the top-p eigenvectors of the raw codes' centred training covariance, computed
  before whitening; p <= the code's dimension). Targets: centred one-hot with V-1 columns (value 0 scored as minus
  their sum) or all V columns. Scales: one 32-bit scale per column and per block of 8,192 rows, or one per block of
  8,192 numbers of the p x c matrix flattened column by column. Quantizer per scale group: one bit, the sign times the
  group's mean |w|; two or four bits, Max's uniform step on the group's standard deviation. p: the largest that fits
  c p H_q + 32 x (number of scale groups) bits. Grid, in tie order: scale (per column, flat) > target (V-1, V) >
  (preprocessing, features) (centre and random, centre and principal, whiten and random, whiten and principal) >
  alpha > q (1, 2, 4): 2 x 2 x 4 x 6 x 3 = 288 configurations, ties within 0.1 points to the first.
- Online least squares (B62): One pass of the normalised delta rule in a fixed random order over the store's facts,
  with the corrected memory's features, targets and scales; step size and averaging chosen on development data.

### A.5 Values the text leaves unstated [filled from prereg]

1. The store's stored shares: f in {1, 1/2, 1/4, 1/8}; a store with share f writes floor(f M) of its M facts (at
   least 1; a seeded random subset of the store), its dimension d still comes from the full budget b M bits, and the
   aliases of every stored entity (written or not) are still queried. The 72 configurations are preprocessing {centre,
   whiten} x map {lin = Gaussian projection, sgn = its sign, sgn3 = product of three independent signs} x q {1, 2, 4} x
   share {1, 1/2, 1/4, 1/8}, in that tie order.
2. The store's dimension: d = the largest d with d H_q + 32 ceil(d / 8,192) <= b M (ideal accounting; H_1 = 1,
   H_2 = 1.9037, H_4 = 3.6024). Write: S = sum over written facts of phi(key) * c(value) (elementwise); quantize S block
   by block (blocks of 8,192 numbers); read: score(v) = sum_j Q(S)_j phi_j(query) c_j(v), answer = argmax over the V
   values. Value codes: V = 2: c(0) = r, c(1) = -r (r a random +-1 vector); V = 16: Hadamard as above. All random parts
   (projections, r, the share subsets, quantizer tie signs) come from seeds fixed per (encoder, preprocessing, map) and
   per store, never from the test data.
3. Max's (1960) Gaussian-optimal uniform-quantizer steps, in units of the standard deviation: 4 levels (q = 2) 0.9957,
   16 levels (q = 4) 0.3352. Mid-rise: level index k = clip(floor(x / step), -L/2, L/2 - 1), output (k + 0.5) step,
   L = 2^q.
4. LAMc and NLMSc store all M facts (no share dimension). Random features: a dense Gaussian projection with p up to
   65,536; principal features p up to the code's dimension n. For whitened codes the principal map is the whitened,
   unit-normalized code's coordinates on the raw training covariance's top-p eigenvectors (eigenvectors of the raw
   centred training covariance, descending eigenvalue); for centred codes, the centred, unit-normalized code's
   coordinates on the same eigenvectors. 'V' targets: column v holds e_v - 1/V. If p = 0 fits, answer the store's
   majority value. ties at exactly 0 for q = 1: random signs.
5. NLMSc (B62): W_0 = 0; for t = 1 .. M in one fixed random order of the store's facts:
   e_t = y_t - W_{t-1}^T f_t, W_t = W_{t-1} + mu f_t e_t^T / (f_t^T f_t + 1e-12), y_t = fact t's target row ('V-1' or
   'V' targets as LAMc). Output W_M ('last') or the Polyak average (1/M) sum_t W_t ('avg'). Then LAMc's quantizer, bits
   and read. Grid, in tie order: scale (col, flat) > target (V-1, V) > (prep, feat) (centre rp, centre pca, whiten rp,
   whiten pca) > mu (1, 0.5, 0.25, 0.1, 0.03) > output (last, avg) > q (1, 2, 4): 480 configurations. Exactly one pass,
   every fact written once.
6. Dev rule (all three memories): per condition (M, V, bits per fact), the configuration with the highest mean dev
   accuracy over the condition's dev stores; configurations within 0.1 points of the best are tied and go to the first
   in grid order. Test: the chosen configuration only, run once on the test stores.
7. Statistic: per condition, test accuracy = share of the condition's test queries (pooled over its test stores)
   answered correctly. Region statistic = mean over the conditions of the per-condition accuracy difference.
   Interval: 95% cluster bootstrap, clusters = (store, entity), resampled within each condition, region mean per draw.

### A.6 Data (paths on the machine that runs the script)

- Directory D = /path/to/vsassm/b49_out/b32_data (read-only).
- D/entities.json: {"entities": [{"uri", "name", "aliases": [str], "props", "split": "train" | "dev" | "test"}, ...],
  ...}: 9,278 entities (train 2,818, dev 1,838, test 4,622); aliases per split: train 2,803, dev 2,152, test 4,059.
  5,754 entities have no alias: they can be stored but give no query.
- D/emb_strings.json: a JSON list of 18,292 distinct strings (every entity name and alias). Row i of an embedding file is
  the code of string i.
- Codes (numpy .npy, float16, shape 18,292 x 4,096):
  Qwen3-8B: D/emb_qwen8_mean_L9_f16.npy (md5 1e4f5a91b57bd8987c899b770ff0c915);
  Llama-3.1-8B: D/emb_llama31i_mean_L24_f16.npy (md5 a55892a0ae069b160cb2dfaedce914a4).
- Training statistics (means, covariances, eigenvectors): the codes of the training split's entity names and their
  aliases (5,621 rows).
- Stores: draw them yourself with your own seeds: for each (M, V) in {256, 1,024} x {2, 16}: 6 (M = 256) or 3
  (M = 1,024) dev stores from the dev split and 20 or 5 test stores from the test split; M entities without
  replacement; each gets a value uniform in 0 .. V-1; the fact is written under the entity's name code; every alias of a
  stored entity is a query with that entity's value as the answer. The same stores serve every memory and every bits
  value. Budgets: b in {4, 8, 16} bits per fact, B = b M bits per store.

### A.7 Deliverable for Part A

One standalone Python 3.10 script (numpy 2.2, scipy 1.15, numba 0.68, joblib, the standard library; CPU only). It must:
- take --enc {qwen8, llama}, --out DIR, --workers N (default 14), and --smoke (one condition, one dev and one test store,
  a handful of configurations, to finish in under 2 minutes);
- run dev selection and the test run for STORE (final design), LAMc and NLMSc on the 12 R1 conditions of the encoder;
- write DIR/{enc}_result.json with, per condition (M, V, b): each memory's chosen configuration, its mean dev accuracy,
  its test accuracy, the number of test queries; and per memory pair (STORE - LAMc, STORE - NLMSc, NLMSc - LAMc) the
  region mean over the 12 conditions, the V = 2 half (6 conditions) and the V = 16 half (6), each with its 95% cluster
  bootstrap interval (2,000 draws), and the bits used by each chosen test configuration against its budget;
- print progress, and finish in at most about 75 minutes per encoder on 14 cores with at most 12 GB of RAM.

---------------------------------------------------------------------------------------------------------------------

## PART B. Equal-cell device simulation: cells per stored pair for exact recall

### B.1 Paper text (Methods, "Equal-cell device simulation"; trimmed; numbers of results redacted)

We rebuilt 50 RULER stores (4,776 probed keys) and placed every stored bit of every memory, a table's directory and
parity included, in a cell of the statistical phase-change device model of Joshi et al. (Nat. Commun. 11, 2473, 2020),
a differential pair of devices read as a sign, 30 days after writing, with every noise source multiplied by k. Tables
found their rows through a stored minimal perfect hash, built with every key known, or (B63) a cuckoo table of 12- to
18-bit key fingerprints filled one pair at a time (four slots per bucket, load at most 0.95, every slot on the cells),
and were protected by repetition, shortened BCH codes (lengths 63 to 1,023, any strength) or LDPC codes (rates 1/4 to
9/10, decoded by belief propagation); the store was read by rules among them one that marks each key's last value. For
each memory we found, on synthetic development stores and then once on the test stores, the smallest number of cells
per stored key-value pair at which it read at least 99.5% of the probed keys exactly, under independent cell errors.
B61 replaced the dense bipolar store by a store of sparse block codes (one active position per block, binding by
block-wise index addition, superposition by OR, read by overlap). The store's codebook and the tables' index map come
from the same vocabulary and are free to both.

### B.2 Box 1 (rows that Part B needs)

- Algebra: bipolar codes on the devices; sparse block codes in the sparse-block store, whose pairs are superposed by OR
  and read by overlap.
- Stores: the device store (dense bipolar codes on simulated devices) and the sparse-block store (one active position
  per block).
- Error-corrected table: a table whose entries are protected by an error-correcting code (BCH or LDPC).
- Clean-up, codebook: replacing a noisy read by the nearest code in the codebook, the table of known codes.

### B.3 Supplementary Table "Implementation details" (rows that Part B needs, verbatim)

- Device model: Joshi et al., Supplementary Note 2, Eqs. 8-17, as implemented: target 25 uS (SET) or 0 (RESET);
  programming noise of standard deviation 1.2 uS (SET) or 0.2 uS (RESET) at two events, 27.36 and 55.39 s after
  targets defined at 20 s; a per-device drift exponent, Gaussian with the measured mean and spread and truncated at
  zero; 1/f read noise with Q_s = min(0.0388 G^-0.5729, 0.2) (t/T_Q)^(-nu_Q(G)), T_Q = 1.089e5 s, integrated from
  1/(t + T_read) to 1/(2 T_read) with T_read = 250 ns; conductances clipped at zero; an 8-bit converter and global drift
  compensation. "Times k" multiplies the drift-exponent spread and the programming-noise and read-noise standard
  deviations by k.
- Probed keys: Up to 150 keys per store, every floor(n/150)-th key in store order.
- Store layout and load: S = ceil(B n / 8,192) vectors of d = floor(B n / S) cells; the load follows from the budget,
  with no load search.
- Last-value marker: The key's last value is bound with a LAST code; clean-up runs over each value and value x LAST,
  and a read stops when a LAST variant wins. Tables store a last-value flag per row.
- Tables: Minimal perfect hash by displacement, with rows of (last flag, value index). BCH: the logical bits in
  shortened binary codewords (n = 63 to 1,023, any t), each coded bit repeated rho >= 1 times, a majority per bit with
  ties as erasures, errors-and-erasures bounded-distance decoding, (n, t) chosen per store by closed form at the
  condition's error rate. LDPC: IRA codewords of at most 1,024 information bits, rate chosen on development stores,
  soft combining of copies, sum-product decoding for 60 iterations.
- Budget: Every store must fit the budget in cells per pair; the 99.5% share of exact reads is pooled over the 4,776
  probed keys; the requirement is the smallest grid budget at which the share holds there and at every larger budget
  evaluated.
- Device design choice: Per memory, noise level and budget, the design with the highest development accuracy; exact
  ties go to the higher development accuracy at the next harder noise level at the same budget, then to the first in
  the memory's design list. BCH's information bits are the true dimension of the narrow-sense code (from its
  cyclotomic cosets), not n - mt.
- Stores of sparse block codes (B61): Blocks with one active position each, binding by adding the active indices of
  the key, the value's slot and the value block by block (modulo the block length), superposition of a shard's pairs by
  OR on binary cells, the block load chosen on development stores; a query scores every value of the vocabulary by its
  overlap with the cells and stops on the last-value code.
- Insertable directory (B63): Bucketized cuckoo table: four slots per bucket, two candidate buckets, partial-key cuckoo
  hashing, load at most 0.95, 500 moves per insertion; each slot holds an f-bit key fingerprint (f in {12, 14, 16, 18}),
  the last flag and the value index; every slot on the cells, coded by the BCH or LDPC codes above; two or more
  fingerprint matches fail the key.

### B.4 Supplementary text (redacted)

Every bit on the devices: every stored bit of every memory, including a table's directory, is a differential pair of
cells on the device model. A key is read exactly when all its values come back, in order, with nothing extra. The
tables were searched over code lengths, rates and repetition, the store over its load where it has one. The value
vocabulary (index -> value string) is a shared output dictionary, free to every memory; every memory returns
vocabulary indices. The dense store's codes are regenerated digitally from the strings (seeded hashes; no cells).

The sparse store (H-SBC): every key and value code has one active position in each of its blocks; binding adds the
codes' active positions block by block, and the pairs of a shard are superposed by OR onto binary cells; a query
scores every value by the overlap of its bound code with the cells and stops on the code that marks each key's last
value. The block load was chosen on the 12 synthetic development stores, and the test ran once.

Tables with an insertable directory (B63): a bucketized cuckoo table with four slots per bucket, two candidate buckets,
partial-key cuckoo hashing, load at most 0.95 and 500 moves per insertion, filled one pair at a time. Each slot holds an
f-bit fingerprint of the key, the last-value flag and the value index; every slot, empty or not, and every fingerprint
bit is on the cells and protected by the BCH or LDPC codes unchanged; a lookup with two or more fingerprint matches
fails the key; f in {12, 14, 16, 18} and the code were chosen on the development stores.

### B.5 Values the text leaves unstated [filled from prereg]

1. Cells and errors. One cell = one stored bit (a differential pair: +1 -> (25 uS, RESET), -1 -> (RESET, 25 uS); read
   as the sign of the converter's output for G+ - G-; an 8-bit symmetric mid-tread converter over [-25, +25] uS, 255
   levels; an output of exactly zero is a random sign). The conditions of Part B are independent cell (sign) errors at
   the device model's 30-day cell-error rate p(k): estimate p(k) yourself from the device model on at least 10^7 random
   cells per k, then flip each memory's cells independently at that rate. Binary 0/1 cells (the sparse store) are the
   same sign cells (0 -> -1).
2. Device-model equations (Joshi et al. SI; conductance in uS, time in s; G_T the target): drift-exponent mean
   mu_nu(G_T) = max(-0.0045 G_T + 0.08, min(0.06, max(-0.00169 G_T + 0.0825, 0.055))); spread
   sd_nu(G_T) = 1 / (8 G_T + 30.032); nu_Q(G_T) = min(0.081, 0.04 G_T + 0.02) - [G_T > 13.21] min(0.035,
   -0.2048 + 0.0155 G_T); read-noise exponent of the 1/f spectrum gamma = 1.21, so the integral term is
   sqrt((f_max^(1-gamma) - f_min^(1-gamma)) / (1 - gamma)), f_max = 1/(2 T_read), f_min = 1/(t + T_read). Programming:
   nu ~ max(mu_nu + k sd_nu N(0,1), 0) per device; G(T0) = max(G_T (T0/T_ref)^(-nu) + k sigma_prog N, 0);
   G(T1) = max(G(T0) (T1/T0)^(-nu) + k sigma_prog N, 0); drift to t: G_d(t) = G(T1) (t/T1)^(-nu); read:
   G(t) = max(G_d(t) + k Q_s(G_T, t) * integral * G_d(t) * N, 0). t = 30 days.
3. RULER stores' content (files below): a store is a list of (key string, value string, j), j = the value's position
   within its key (0, 1, ...); a key's values are its pairs in increasing j. n = the number of pairs. N = the number of
   distinct value strings in the store (the vocabulary). Probed keys: the distinct keys in order of first appearance,
   every max(1, floor(K/150))-th, at most 150 (K distinct keys); a probed key is exact iff the memory returns exactly its
   list of values in order. Test: 50 stores, 4,776 probed keys in all (check your count).
4. Budget grid: B = 2^(i/4) cells per pair, i = 12 .. 48 (8 to 4,096; one step = 19%). On a store with n pairs a memory
   may use at most floor(B n) cells; every bit it keeps between write and read is on them.
5. Dense bipolar store (the store with the last-value marker; no threshold). Random +-1 codes of length 8,192 for each
   key string, each position j, each value string, and one LAST code, from seeded hashes of the strings, truncated to
   the first d components. Pair code = key * pos_j * value (elementwise), times LAST if j is the key's last position.
   Key -> shard by a seeded hash of the key string mod S. A shard's cells = the sign of the integer sum of its pairs'
   codes (ties to +-1 at random). Read of a probed key: for j = 0, 1, ..., maxj (maxj = the store's largest j): unbind
   the key's shard with key * pos_j, score each of the 2N candidates (each value code; each value code * LAST) by
   inner product, take the best (first maximum; the LAST variant only if strictly higher than every plain one), append
   its value, and stop if the LAST variant won.
6. Sparse-block store (H-SBC). Geometry as the dense store (S, d, key -> shard). Block length
   Lb = max(2, round((n/S) / lam)), nb = floor(d / Lb) blocks (the last d - nb Lb cells of a shard are unused and
   counted). Each key, position j, value and LAST has one active index in each block (in 0 .. Lb-1), from seeded hashes.
   A pair's code: per block, (key + pos_j + value [+ LAST on the key's last value]) mod Lb; a shard's cells are the OR
   of its pairs' one-hot blocks. Read: for j = 0, 1, ...: for each of the 2N candidates (value; value + LAST), the
   overlap = the number of blocks whose candidate position reads 1; argmax (first index; LAST only if strictly
   higher); the dense store's stop rule. Block load lam in {0.35, 0.5, 0.7, 1.0}, chosen on the dev stores per
   (noise level, budget).
7. BCH static table (E-BCH). A minimal perfect hash over the n (key, j) pairs by hash-and-displace (on average 5 pairs
   per bucket), its displacement array on the cells; one row per pair: the last flag (only if some key has more than
   one value) and the value index in ceil(log2 N) bits. Read of a probed key: for j = 0, 1, ...: locate the row of
   (key, j) through the decoded displacements, append its value index, stop when its last flag is set (or after
   maxj + 1 rows). All logical bits (displacements, then rows) are cut into shortened binary narrow-sense BCH codewords
   (n in {63, 127, 255, 511, 1023}; any t with information length k >= 2, k the true dimension); each coded bit is
   repeated rho >= 1 times (rho or rho + 1 copies, spread so that the whole budget is used); majority per coded bit, a
   tie is an erasure; errors-and-erasures bounded-distance decoding (a word with 2e + s <= 2t is corrected; an
   undecodable word returns its received information bits). (n, t) per store, budget and noise level by closed form:
   the design that minimizes the expected number of failed probed keys at the condition's cell-error rate (ties: the
   expectation at the next harder noise level; then the shorter code). Write your own BCH encoder and decoder.
8. Cuckoo insertable table (O-BCH and O-LDPC; report each and the better of the two). Bucketized cuckoo table over the n
   (key, j) pairs filled one pair at a time: two candidate buckets per (key, j) (partial-key cuckoo hashing: the second
   bucket is the first XOR a hash of the fingerprint), four slots per bucket, number of buckets the smallest with
   n <= 0.95 x 4 x buckets, at most 500 moves per insertion (a failed insertion: report it). A slot holds the f-bit
   fingerprint of (key, j) (reserve one fingerprint value for "empty"), the last flag and the value index; all slots,
   empty or not, are on the cells, coded by the BCH scheme of item 7 (O-BCH) or the LDPC scheme of item 9 (O-LDPC).
   Lookup of (key, j): the 8 slots of its two buckets; exactly one fingerprint match -> its value and last flag; zero
   or two or more matches -> the key fails. Design: f in {12, 14, 16, 18} and (for O-LDPC) the rate, chosen on dev;
   (n, t) for O-BCH by the closed form of item 7.
9. LDPC (O-LDPC). Systematic irregular repeat-accumulate (IRA) codes, H = [H1 | dual-diagonal], at most 1,024
   information bits per codeword, rate R in {1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 9/10} with information-column weight
   dv = 6, 6, 5, 6, 5, 4, 5 respectively; each coded bit repeated rho >= 1 times; the copies' votes combined into a
   channel log-likelihood ratio at the condition's cell-error rate; sum-product belief propagation, flooding schedule,
   60 iterations. Write your own encoder and decoder (numpy / numba).
10. Development and test procedure. Dev: the 12 synthetic development stores (file below). For each memory, noise level
   and grid budget, evaluate every design, keep the design with the highest pooled dev share of exact probed keys (a
   design counts at a budget only if it fits on all 12 dev stores); dev requirement = the smallest grid budget at which
   the dev share is >= 99.5% there and at every larger dev budget evaluated. Test: every grid budget within 4 grid
   steps of the dev requirement, each with that budget's dev design; while the test requirement is not bracketed (a
   passing budget whose every larger evaluated budget also passes, with the budget one step below it evaluated and
   failing, or the grid's lower end), extend by 4 more steps in the needed direction, up to 4,096. A memory that never
   reaches 99.5% by 4,096 is reported as "> 4096". Channel draws: one uniform per cell per (store, memory, budget,
   design), shared across noise levels.

### B.6 Data

- Test stores: /path/to/vsassm/b34_out/test_stores.json.gz (gzip JSON): a list of 50 entries [store_id, pairs],
  pairs = [[key, value, j], ...]; store ids like "131072/niah_multikey_2/259411" (five task types: multi-key needles
  with 2 and 3 keys, multi-value needles, variable tracking; up to about 6,000 pairs per store, a few with as few as
  9 pairs).
- Dev stores (synthetic, no test data): /path/to/vsassm/b34_out/dev_stores.json.gz, the same format, 12 entries.
- Noise levels in scope: k = 1, 8 and 16 at 30 days. Vocabulary uncharged (free).

### B.7 Deliverable for Part B

One standalone Python 3.10 script (numpy 2.2, scipy 1.15, numba 0.68, joblib, the standard library; CPU only; your own
BCH and LDPC codecs, no coding-theory package). It must:
- take --memory {dense, sbc, bch, cuckoo, all}, --k (comma list, default 1,8,16), --out DIR, --workers N (default 14),
  and --smoke (two small dev stores, two budgets, k = 8 only, to finish in under 3 minutes);
- estimate and write p(k) (30 days) from the device model;
- for each memory and k: the dev table (design and pooled share per budget), the dev requirement, the test budgets
  evaluated with their designs and pooled test shares (exact keys / 4,776), and the test requirement in cells per pair;
  for the cuckoo table also O-BCH and O-LDPC separately and the number of failed insertions;
- write DIR/b_result.json, print progress, and finish within about 90 minutes on 14 cores with at most 12 GB of RAM.
