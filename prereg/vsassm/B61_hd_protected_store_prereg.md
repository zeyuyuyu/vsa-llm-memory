# B61 pre-registration: HD-native protected stores (sparse block codes, explaining away, an error-correcting item codebook) against the uncoded store and the best table at equal cells

Date: 2026-11-20 (US Eastern, 21:33; machine clock Asia/Shanghai 2026-11-21, 10:33). Written and committed, then
OpenTimestamps-stamped, before any B61 development or test run. The B61 code (b61_hd.py, the box driver b61_box.sh and a
first verdict script b61_verdict.py) is committed with this document; b61_hd.py was self-tested on synthetic pairs only
(record at the end). Nothing in B61 has read a dev or test store (only the test stores' pair counts were listed, for
compute planning: 10 stores of about 6,000 pairs, 10 of 1,500-2,000, 10 of about 1,000, 20 small). The check code
(b61_check.py) and any change to the verdict script are committed before any test output is read.

## Why (referee)

B54's scope paragraph: "the HD-native protected variant of the store (sparse block code, error-correcting item codebook
or iterative clean-up) asked for by the referee is **not run**". B58 and B59 (part 2) protected the store's bits with the
tables' channel codes (BCH, LDPC); coding never helped: searched over the uncoded store U and both code families at every
budget, the store needed 76 / 76 / 76 / 91 / 128 / 431 / 1,722 cells per stored pair at k1 / k2 / k4 / e05 / k8 / k16 /
k32, against the best table's 19 / 19 / 27 / 38 / 54 / 215 / 724 (B59 part 2, H1-P TABLE WIN). Under damage in the
natural layout the uncoded store won. The referee asks for the protected variant built from hyperdimensional codes, the
arm that would most favour the VSA side. B61 asks: **does an HD-native protected store need fewer cells than U, and does
it close the gap to the best table?**

## Fixed (B59 part 2's setting, prereg/B58_protected_store_prereg.md and prereg/B59_..._prereg.md; unchanged)

- Stores, keys, metric: B38's 50 RULER test stores (b34_out/test_stores.json.gz), 4,776 probed keys; a key is exact when
  all its values come back in order with nothing extra. Dev: B34's 12 synthetic dev stores (b34_out/dev_stores.json.gz,
  1,254 probed keys).
- Cells: every stored bit is one cell, B38's differential pair of PCM devices read as a sign (binary cells; B54's
  multi-level cells are not used, as in B58 / B59). Conditions: i.i.d. cell errors at the device model's 30-day
  bit-error rates k1 0, k2 1.3e-4, k4 0.0208, e05 0.05, k8 0.1226, k16 0.2760, k32 0.3744 (B38's Channel: one uniform
  per cell, seeded by store, arm tag and budget); e05 is reported and not in the labels (as B59). Damage (secondary):
  B38's failed lines and 16 x 16 stuck clusters, natural layout (cells in the arm's natural order, codes and shards
  contiguous), f = 2 eps of the matched i.i.d. condition, at k4 and k8 (B59's four damage conditions).
- Budget B = cells per stored (key, j) pair; at most floor(B n) cells on a store of n pairs; every bit kept between write
  and read is on them. Grid B = 2^(i/4), 8 ... 4,096. N* = the smallest grid budget with >= 99.5% of probed keys exact
  there and at every larger evaluated budget; a budget counts only when every store was evaluated there (B38 / B59).
- Free for every arm (B38): hashing, seeded code generation from strings, binding, similarity search, decoders, the value
  vocabulary (shared output dictionary, uncounted for every memory; B58 part (b) audit). Design constants fixed at write
  time from n and B (shard count, shard length, block length) are known to the reader, as the tables' code parameters.
- Comparators, as registered and already run (not re-run): U = B38's V-last (results/b38_test.json.gz) with B59's T1
  budgets (results/b59_test_T1_U.json.gz), N* 76 / 76 / 76 / 91 / 128 / 431 / 1,722; the best table = the smallest N*
  among T-rep, E-BCH, E-LDPC, E-CAM (B38's i.i.d. rows, b58_verdict.b38_counts), 19 / 19 / 27 / 38 / 54 / 215 / 724; in
  damage the best non-HD table with B58's re-scored E-LDPC* (as B59's damage table). The per-store failed-key counts of
  these rows enter the bootstrap below (b59_store_verdict.counts).

## Arms (b61_hd.py)

**H-SBC: sparse block code store (candidate a; Laiho et al. 2015, Frady et al. 2021).** Every key, position j, value and
the LAST marker is a sparse block code: nb blocks of length Lb, one active position per block, positions from seeded
hashes of the strings. Binding = block-wise addition of the active indices mod Lb (block-wise circular convolution, the
SBC binding). A pair's code is key + position + value (+ LAST on a key's last value), as B38's V-last. A shard's
superposition is the OR of its pairs' codes (one cell per position: active +1, inactive -1); on binary cells this is the
natural storage of a sparse superposition (a Bloom-filter-like / Willshaw-type memory). Geometry: B38's shards, S =
ceil(floor(B n) / 8,192), d = floor(floor(B n) / S), key -> shard by B38's hash; Lb = max(2, round((n / S) / lam)),
nb = floor(d / Lb) (the last d - nb Lb cells of a shard are unused and counted). Read of probed key a: for j = 0, 1, ...:
unbind block-wise with the key's and j's indices, score each of the 2 N_values candidates (value; value bound with LAST)
by its overlap = the number of blocks whose candidate position reads active, argmax, and B38's V-last stop rule (stop
when the LAST variant wins; ties: argmax's first index, LAST only if strictly higher). Laiho's block-wise winner-take-all
clean-up is defined for one bound vector; on an OR superposition with several active positions per block it is replaced
by the overlap score, which is the same decision when one item is stored and the maximum-likelihood decision under
i.i.d. cell errors (all blocks exchangeable). **Free parameter: the block load lam = (pairs per shard) / Lb in LAMS =
{0.35, 0.5, 0.7, 1.0}** (lam = ln 2 is the Bloom optimum without cell errors; about 0.5 maximizes the overlap SNR under
heavy errors). No censoring (the read is cheap).

**H-SIC: explaining away / successive interference cancellation (candidate c; Kleyko et al. 2023).** The same stored
cells and the same channel draw as H-SBC (same arm tag), read twice. Pass 1 = H-SBC's read of every probed key. Pass 2:
every position set by another item decoded in pass 1 (all probed keys' decoded (key, j, value, LAST) items) is
explained and carries no evidence (the slot's own pass-1 item is excluded); at unexplained positions a candidate scores
log((1 - e) / q) if the cell reads active and log(e / (1 - q)) if inactive (e = max(the condition's cell-error rate,
1e-3), q = the shard's active fraction among unexplained cells); argmax and the same stop rule. Two passes, fixed. The
reader explains away only what it is given at read time: the probe batch (150 of 1,500-6,000 stored keys per large dev
store, every key of the small ones). **Free parameter: its own lam choice** (from the same evaluations as H-SBC).
*Not run (stated as a limitation):* explaining away over every stored key (an oracle key list): the reader is given only
the query keys; with the stored-key list free, a table would need no key directory either (T-rep's and the codes'
displacement bits), so the comparison would no longer be at equal information. *Resonator decoding* (Frady et al.
2020, Kent et al. 2020) factorizes a product of several unknown factors; here the key and the position are given at read
time and only the value is unknown, so it reduces to the clean-up and is not a separate arm.

**H-HAD: error-correcting item codebook (candidate b).** B38's V-last store (dense bipolar codes, the sign of the
bundle, B38's shards and read) with the value codes replaced by codewords of the first-order Reed-Muller code RM(1, m)
(signed Walsh-Hadamard rows; 2^(m+1) codewords, minimum distance 2^(m-1), the largest possible for that many words), 2^(m-1)
< d <= 2^m, restricted to a seeded set of d of the 2^m coordinates and multiplied by a seeded sign vector; values take
distinct seeded (sign, row) slots among rows 0 ... 2^(m-1) - 1; LAST = Hadamard row 2^(m-1), so a value bound with LAST is
again a codeword and differs from every value's (at d = 2^m all 2 N_values candidates are orthogonal or antipodal).
Values beyond 2^m (none expected: d > 4,096 on every large store) would get B38's random codes (counted in the output).
The clean-up correlates the unbound read with the vocabulary's codewords: the restriction of RM(1, m)'s
maximum-likelihood decoder (the fast Hadamard transform) to the vocabulary (equality checked in the self-test). Keys and
positions keep B38's random codes. **No free parameter.** Dev / test censoring caps 7 / 40 failed keys per store (B58).

## Development (dev stores only; i.i.d. conditions; B58's two-pass procedure)

- Pass 1: H-SBC / H-SIC every lam in LAMS at the octave budgets 8, 16, 32, ..., 4,096 (one evaluation gives both reads);
  H-HAD every grid budget within 4 grid steps of U's dev N* (76 / 76 / 76 / 91 / 128 / 431 / 1,722) per condition.
- Pass 2 (b61_hd.dev2): every lam at the quarter steps within 3 grid steps of each (H-SBC or H-SIC, condition)'s pass-1
  N* (the top 4 budgets if not reached).
- Choice per (arm, condition, budget) = B38 / B58's rule (b61_hd.choose, over pass 1 + pass 2): the design with the
  highest dev share; ties the higher share at the next harder condition at the same budget, then lam ascending;
  censored designs below completed ones. Dev N* per arm. The choices are committed before any test run.

## Test (run once)

Per (arm, condition): H-SBC and H-SIC at every grid budget from 8 up to the matched i.i.d. dev N* + 4 grid steps (the
whole grid if dev never reached the bar), each budget with the arm's dev design there (nearest dev budget's design where
dev did not evaluate it), at the seven i.i.d. and the four natural-layout damage conditions (matched i.i.d. designs);
H-HAD at dev N* +- 4 steps, i.i.d. only. Then B38's extension rule (b61_hd.extend: 4 more steps up while the top
evaluated budget fails; 4 down while N* is the lowest evaluated budget), repeated until bracketed or the grid ends.
Every test cell runs once.

## Measures and decision rule (fixed now)

Steps = grid index (one step = x 2^(1/4) = 19%); a family that never reaches the bar counts as 37 (B59).
- **N*(arm, k)** on the test stores, per arm.
- **G (fewer cells than U?)** G(arm, k) = steps(U) - steps(arm): "fewer cells than U" if G >= +2, "more cells than U" if
  G <= -2, "tie" otherwise.
- **H1-HD (against the best table)** Delta(arm, k) = steps(best table) - steps(arm): "store fewer cells" if >= +2, "table
  fewer cells" if <= -2, "tie" if |Delta| <= 1; label TABLE WIN / VSA WIN / MIXED / TIE over k1, k2, k4, k8, k16, k32
  (B38's H1). **Closes the gap** at k: Delta >= -1. **Narrows the gap** at k: G >= +2 and Delta <= -2.
- **HD-best** = per k the smallest of the three HD arms' own N* (selected with knowledge of the test; favours the store);
  **Store-best** = the smaller of U and HD-best. PRIMARY: G and H1-HD for HD-best (and for each arm, secondary).
- **Bootstrap** (B59's store-cluster method): 2,000 resamples of the 50 stores (seed 61), per resample each arm's, U's and
  every table's N* by b58_verdict.nstar_counts, HD-best and the best table recomputed; 95% percentile ranges of G and
  Delta for HD-best and each arm.
- **Damage (secondary)**: H-SBC, H-SIC (and HD-best of the two) vs U and vs the best non-HD table (E-LDPC* re-scored) in
  the four natural-layout conditions, B59's label rule; position independence = within 1 step of the matched i.i.d. N*.

## How the paper should read each outcome (fixed now)

- **G "fewer cells than U" at some k, H1-HD TABLE WIN:** the HD-native sparse store is the store's best form at those k;
  "coded tables needed two to four times fewer cells" is restated against the best store design with the new factor
  (and "two to four" kept only for the dense store); "coding the store's bits never helped" is restricted to the channel
  codes (BCH, LDPC) on the dense store's bits, adding that an HD-native sparse code did help (by the factor); "the store
  needed more cells than coded tables at equal reliability" stands, with HD-best's numbers.
- **H1-HD tie or store fewer at some k:** name the k and the arm; "the store needed more cells than coded tables at equal
  reliability" must be restricted to the other k. If H-SIC is the arm, the paper must add that it reads the probe batch
  jointly.
- **G tie at every k for every arm:** the three sentences stand; the paper adds that HD-native codes (sparse block codes,
  explaining away, a Reed-Muller item codebook) did not lower the store's requirement.
- **H-HAD tie with U:** the item codebook's distances are not what limits the store (the cross-talk of the other bundled
  pairs is).

## Predictions (from a closed-form model only; no B61 data; checked against the verdict, reporting only)

The model (scratch calculation, no store read): per slot, the true value's overlap ~ Binomial(nb, 1 - eps) against
2 x 6,000 - 1 wrong candidates ~ Binomial(nb, rho'), rho' = rho (1 - eps) + (1 - rho) eps, rho = 1 - exp(-lam), nb = lam
B, ties half; a key fails if either of 2 slots fails; N* = the smallest grid budget with key failure <= 0.5%. The same
calculation for the dense sign store (Gaussian overlap, SNR^2 = (2 / pi) B (1 - 2 eps)^2) gives 76 / 76 / 91 / 108 / 152
/ 431 / 1,218 against U's measured 76 / 76 / 76 / 91 / 128 / 431 / 1,722 (within 2 steps). For H-SBC it gives 32 / 32 /
45 / 54 / 91 / 304 / 1,024 (best lam 0.5-0.7).
- **R1** H-SBC test N* within 2 steps of the model at every k (32 / 32 / 45 / 54 / 91 / 304 / 1,024).
- **R2 (G)** H-SBC needs fewer cells than U (G >= +2) at >= 5 of the 7 k, and G >= +1 at every k (model: +5 / +5 / +3 /
  +3 / +2 / +2 / +3).
- **R3 (H1-HD)** H-SBC: TABLE WIN; Delta <= -2 at every k and >= -4 at every k (model -3 / -3 / -3 / -2 / -3 / -2 / -2):
  the sparse store narrows the gap (to about 1.4-2.4 x) but does not close it at any k.
- **R4** H-SIC within 1 step of H-SBC at every k (the probe batch is at most 10% of a large store's keys).
- **R5** H-HAD within 1 step of U at every k (orthogonal item codes remove only the value-value term of the clean-up
  noise, about 1 / (pairs per shard) of its variance).
- **R6** HD-best = H-SBC or H-SIC at every k; HD-best's labels are R2 and R3's.
- **R7** dev lam at dev N*: 0.5 or 0.7 at every k.
- **R8 (damage, natural layout)** H-SBC within 1 step of its matched i.i.d. N* in all 4 conditions (position
  independent, as U) and fewer cells than the best non-HD table in all 4 (VSA WIN), and fewer cells than U in >= 2 of 4.
- **R9 (checks)** second scorer 0 differences; 0 evaluations over budget; N* by the second rule 0 mismatches; H-SIC's
  pass 1 identical to H-SBC on every sampled cell.

## Checks (fixed now; b61_check.py)

- **Second scorer by plain loops**: on a sample of test cells (every arm; i.i.d. and damage; stores of every family) the
  evaluation is repeated and the read recomputed by explicit loops (H-SBC: per key, position, candidate and block; H-HAD:
  one dot product per candidate), the exact-read test re-implemented from the store's own pairs (not Store38.want);
  failed-key lists must be identical.
- **Bits and cells audit**: every test evaluation's cells <= floor(B n) (count over budget must be 0) and the share used;
  H-SBC's density of active cells per evaluation reported.
- **N* by a second rule**: N* recomputed from the per-store failed-key counts by a separately written rule (0 mismatches).
- **Sample reading by the AI-assisted process** (results/b61_sample_reading.txt): failed keys one grid step below N*
  for H-SBC at k1, k8 and k16 and H-HAD at k8: the true value's score against the winner's, a tie or not, a wrong value
  or a wrong LAST decision, blocks and density.

## Compute, scope and time cut (registered now)

CPU only. The local machine is shared with other sessions (load average about 30 on 16 cores at 10:27), so dev and test
run on the A800 box's CPUs (b61_box.sh: own directory ~/b61_run, code and stores MD5-identical to the local commit,
nice 10, one BLAS thread per worker, at most 44 workers, every ssh command in the background, no retry loops around
state-changing commands, no other user's process touched); dev and test of every arm run on the same machine (the box's
numpy / BLAS differ from the local ones; B59 compute note 2). Checks run locally. Time box: 2.5 hours from 10:03
machine clock (until 12:33). Cut rule, applied by a compute note committed before any test output is read: if the test
cannot finish by 12:05, cut in this order: damage, e05, k32, k2 (k1, k4, k8 and k16 are kept); H-HAD's i.i.d. budgets are
cut before H-SBC's. Not run: multi-level cells, the interleaved layout and other damage geometries, H-HAD under damage,
explaining away over the stored-key list, resonator factorization, other read rules.

## Files

Code: b61_hd.py (arms, dev, dev2, choose, plan, run, extend, self-test), b61_box.sh, b61_verdict.py, b61_check.py.
Results: results/b61_dev1.json.gz, results/b61_dev2.json.gz, results/b61_choice.json, results/b61_plan*.json, results/b61_test*.json.gz,
results/b61_verdict.{txt,json}, results/b61_check.txt, results/b61_sample_reading.txt; notes prereg/B61_compute_note*.md,
prereg/B61_note_after_test.md.

## Self-test record (synthetic pairs only; b61_hd.py selftest, 10:27 machine clock, PASS)

- Synthetic store (700 pairs, 500 values, 150 probed keys): every evaluation within its budget (cells <= floor(B n));
  active-cell density 0.29 / 0.39 / 0.50 / 0.63 at lam 0.35 / 0.5 / 0.7 / 1.0 (1 - exp(-lam) = 0.30 / 0.39 / 0.50 /
  0.63); every stored pair's positions read active; the numba read (with exact pruning) equals an explicit-loop read
  on 25 probed keys; H-SBC and H-SIC read 149-150 of 150 keys exactly at B = 32 (k1) and 150 of 150 at 64 (k1), 128
  (k4), 256 (k8) for every lam; H-SIC with no explained cell and weights (1, 0) equals H-SBC.
- H-HAD: at d = 2^13 every pair of value codewords is orthogonal or antipodal and value x LAST is orthogonal to every
  value; the fast Hadamard transform (RM(1, m)'s maximum-likelihood decoder) gives the clean-up's correlations to
  3e-13; on d = 7,777 of 8,192 coordinates the largest |inner product| among 300 value codes is 85 (random codes:
  427); 149-150 of 150 keys exact at B = 76 (k1), 128 (k1, k8).
- A smoke run of the job runner (synthetic store; H-SBC / H-SIC 150 / 150 at B 64 k1 and B 512 k16; 146 / 145 at B 64
  k8; H-HAD 149 at 76 k1) was seen; nothing else.
- Timing (local machine under load; synthetic 6,000 pairs and 6,000 values, lam 0.5): H-SBC + H-SIC 1.3 s at B 64 (k1),
  15 s at 512 (k16), 103 s at 2,048 (k32); H-HAD 19-25 s per evaluation (independent of B). Hence the two-pass dev
  and the box.
- Code fixed during the self-test (before this commit): the slot kernel replaced by an exact-pruning decision kernel
  (same decisions; checked against the explicit loops), numpy-version-independent parity for the Hadamard rows (the box
  has numpy 1.26), finer jobs (one per store, arm, lam and budget; slow conditions split).
