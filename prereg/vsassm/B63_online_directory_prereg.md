# B63 pre-registration: the best coded tables with a directory that accepts online insertion, against the dense store U and the sparse block code store H-SBC at equal cells

Date: 2026-11-22 (US Eastern, about 13:35; machine clock Asia/Shanghai 2026-11-23, about 02:35). Written and
committed, then OpenTimestamps-stamped (the .ots committed), before any B63 development or test run. The B63 code
(b63_online.py) is committed with this document; it was self-tested on synthetic pairs only (record at the end). No dev
or test store has been read by any B63 code; only the pair counts of the dev and test stores were listed, for compute
planning (dev: 9, 9, 23, 23, 1,500, 1,500, 1,700, 1,700, 6,000, 6,000, 6,002, 6,002; test: 20 of 9-23, 20 of
1,465-1,723, 10 of about 5,990). The verdict and check code (b63_verdict.py, b63_check.py) are committed before any
test output is read.

## Why (referee)

The best error-corrected tables of B38 (E-BCH at k1-e05, E-LDPC at k8-k32; 19 / 19 / 27 / 38 / 54 / 215 / 724 cells per
stored pair) find their rows through a stored minimal perfect hash (MPHF, hash and displace, lambda = 5). An MPHF is
built with every key known and cannot accept an online insertion: adding a key means rebuilding it, which needs all
keys. The stores can be appended: H-SBC's OR superposition takes a new pair by setting its positions (exactly, on
binary cells); U (the sign of the bundle) takes one exactly only if the integer sums are kept (they are not on its
cells; see Scope). The referee asks to state this static / online asymmetry and to report the best table with an ONLINE
directory. B63 asks: **with a directory that supports online insertion, how many cells per pair do the best coded
tables need, and how does that compare with U and H-SBC?** The paper currently says (B61 note after test): "coded
tables needed 1.4 to 2 times fewer cells than the sparse block code store and 2 to 4 times fewer than the dense store".

## Fixed (B59 part 2 / B61's setting; unchanged)

- Stores, keys, metric: B38's 50 RULER test stores (b34_out/test_stores.json.gz), 4,776 probed keys; a key is exact when
  all its values come back in order with nothing extra. Dev: B34's 12 synthetic dev stores (b34_out/dev_stores.json.gz,
  1,254 probed keys).
- Cells: every stored bit is one cell (B38's differential PCM pair read as a sign, binary). i.i.d. 30-day cell errors at
  k1 0, k2 1.3e-4, k4 0.0208, e05 0.05, k8 0.1226, k16 0.2760, k32 0.3744 (B38's Channel: one uniform per cell, seeded
  by store, arm tag and budget, shared by the conditions); e05 is reported and not in the labels. No damage conditions
  (the question is the i.i.d. requirement).
- Budget B = cells per stored (key, j) pair; at most floor(B n) cells on a store of n pairs, and every bit kept between
  write and read is on them. Grid B = 2^(i/4), 8 ... 4,096 (one step = 19%). N* = the smallest grid budget with >= 99.5%
  of the probed keys exact there and at every larger evaluated budget; a budget counts only where every store was
  evaluated (B38 / B59; b58_verdict.nstar_counts).
- Free (B38): hashing the query, seeded codes, decoders, the value vocabulary (shared output dictionary). Design
  constants fixed at write time from n and B (here: the number of buckets m, the fingerprint width f, the code) are
  known to the reader, as the tables' code parameters and H-SBC's block length are.
- Comparators, already run, not re-run: U = B38 V-last + B59 T1 (76 / 76 / 76 / 91 / 128 / 431 / 1,722 at k1 / k2 / k4
  / e05 / k8 / k16 / k32); H-SBC = B61's test rows at its dev design (32 / 32 / 45 / 54 / 91 / 304 / 1,448); the MPHF
  tables = B38's T-rep, E-BCH, E-LDPC, E-CAM rows (best 19 / 19 / 27 / 38 / 54 / 215 / 724); E-CAM (B34's hashed-key
  CAM in BCH codewords, h-bit hashes searched over every row) is reported as the reference online-capable table already
  measured (its h was chosen collision-free with all keys known, so it is not counted as an online arm). Per-store
  failed-key counts of these rows enter the bootstrap (b59_store_verdict.counts, b61_verdict.b61_counts).

## Arms (b63_online.py)

**The online directory: a bucketized cuckoo hash table with fingerprints (Fan, Andersen, Kaminsky and Mitzenmacher
2014, the cuckoo filter's table, holding the value rows).** m = ceil(n / (4 x 0.95)) buckets of 4 slots (load <= 0.95,
two candidate buckets per item). Partial-key cuckoo hashing: i1 = h(key, j) mod m, fingerprint fp = 1 + h'(key, j) mod
(2^f - 1) (0 marks an empty slot), i2 = (H(fp) - i1) mod m (an involution, so a stored entry can be moved knowing only
its fingerprint and its current bucket; the subtraction form needs no power-of-two m). Insertion (Fan et al.'s
algorithm): a free slot in i1, else in i2, else evict a random entry of a random candidate bucket and move it to its
alternate bucket, up to 500 kicks; pairs are inserted one at a time in the store's pair order, and an insertion sees
only the table so far and the new pair (online; the self-test checks that a build continued from a half-built table
equals the full build). An entry still homeless after 500 kicks is lost (table full) and its key then reads wrong.
Each slot holds [fp (f bits) | last flag | value index] - the MPHF table's row plus the fingerprint. **Every slot of
every bucket is on the cells, empty or not**; there is no other directory. The table's bits (bucket-major) are the
logical bitstream that B38's codes protect, unchanged:
- **O-BCH**: B38's E-BCH machinery on the cuckoo bits (shortened binary BCH, (n, t) per store, budget and condition by
  B38's closed form - minimum expected failed keys over every stored key at the condition's rate, ties at the next
  harder rate, then the shorter code; a key's needed bits = the fingerprints of all slots of its two buckets at each j
  and its own slot's payload; each coded bit repeated rho >= 1 times; errors-and-erasures bounded-distance decoding).
  Free parameter: f.
- **O-LDPC**: B38's E-LDPC machinery on the cuckoo bits (IRA LDPC, <= 1,024 information bits per codeword, rate R in
  B38's seven rates, soft combining of the copies at the condition's rate, the ldpc package's sum-product decoder, 60
  iterations). Free parameters: (f, R).
- **f in FS = {12, 14, 16, 18}** for both arms (the free parameter; a 2-bit grid moves the cells by at most 1 bit per
  slot, about 4%, below one grid step).

**Read** (B38's table loop, j = 0, 1, ...; stop at the last flag): from (key, j) compute i1, fp, i2; compare fp with the
decoded fingerprints of the (up to) 8 candidate slots. Exactly one exact match: read that slot. **Two or more exact
matches (a fingerprint false match): the key's read fails** (the reader cannot tell which entry is the key's; it does
not guess). No exact match (residual bit errors): the unique nearest slot in Hamming distance; a tie fails. Every read
is scored by the exact-key test, so false matches, lost insertions and decoding failures all count as errors.

**OT-best** = per k the smaller of O-BCH's and O-LDPC's own N* (selected with knowledge of the test, which favours the
table, as HD-best favoured the store in B61).

**Second design not run (closed-form note, registered now).** A quotient filter (Bender et al. 2012) holding the same
rows needs r + 3 metadata bits per slot (r + 2.125 in the rank-and-select QF, Pandey et al. 2017) with a false-match
rate of about load x 2^-r per lookup, against the cuckoo table's about 7 x load x 2^-f; at equal false-match rate r = f -
2.8, so its slot is within one bit of the cuckoo slot (f + 0.2, or f - 0.7) - at most 4% of the cells, below one grid
step - and both run at load about 0.95. It is reported as not run.

## Development (dev stores only; i.i.d.; B58 / B61's two-pass procedure)

- Pass 1: every design (O-BCH: 4 f; O-LDPC: 4 f x 7 R) at the octave budgets 8, 16, ..., 4,096, the seven conditions.
- Pass 2 (b63_online.dev2): every design at the quarter steps within 3 grid steps of each (arm, condition)'s pass-1 N*
  (the top 4 budgets if not reached).
- Choice per (arm, condition, budget) (b63_online.choose, over both passes): the design with the highest pooled dev
  share, a design counting at a budget only where it was feasible on all 12 dev stores; ties the higher share at the
  next harder condition at the same budget, then list order (f ascending, then B38's rate order). Dev N* per arm. The
  choices are committed before any test run.

## Test (run once)

Per (arm, condition): every grid budget within 4 grid steps of the dev N* (B38's table rule; the top 5 budgets if dev
never reached the bar), each with the arm's dev design there (the nearest dev budget's design where dev did not
evaluate it, ties the larger budget). Then B38's extension rule (b63_online.extend: 4 steps up while the top evaluated
budget fails, 4 down while N* is the lowest evaluated budget; budgets already planned are not requested again),
repeated until bracketed or the grid ends. i.i.d. conditions only. Every test cell runs once.

## Measures and decision rule (fixed now)

Steps = grid index; a family that never reaches the bar counts as 37 (B59 / B61).
- **N*(arm, k)** for O-BCH, O-LDPC, OT-best; with U, H-SBC, the MPHF best table and E-CAM beside them.
- **Delta_S(k) = steps(OT-best) - steps(S)** for S = H-SBC (PRIMARY) and S = U (secondary); + = the store needs fewer
  cells. Per k: "store fewer cells" if Delta >= +2, "table fewer cells" if <= -2, "tie" if |Delta| <= 1. Label over k1,
  k2, k4, k8, k16, k32 (B38's H1): TABLE WIN ("table fewer" at >= 1 k, "store fewer" at none), VSA WIN (the reverse),
  MIXED (both), TIE (every k a tie). **H1-ON** = the label against H-SBC; **H1-ON-U** against U. Per arm (O-BCH, O-LDPC)
  the same, secondary.
- **Online penalty Pen(k) = steps(OT-best) - steps(MPHF best table)** (cells the online directory costs, in steps).
- **Bootstrap** (B59 / B61's store-cluster method): 2,000 resamples of the 50 stores (seed 63); per resample every arm's
  N* by the vectorized rule (checked against b58_verdict.nstar_counts on the full sample), OT-best and the MPHF best
  table recomputed; 95% percentile ranges of Delta vs H-SBC, Delta vs U, Pen and N*(OT-best).

## How the paper should read each outcome (fixed now)

- **H1-ON TIE** (|Delta| <= 1 at every labelled k against H-SBC): "coded tables needed 1.4 to 2 times fewer cells than
  the sparse block code store" **must carry the static qualifier**: it holds for tables whose directory is a minimal
  perfect hash built with all keys known; with a directory that accepts online insertion (a cuckoo table with
  fingerprints) the best coded table needs about as many cells as the sparse block code store (within one grid step,
  19%) at every noise level, with the numbers.
- **H1-ON TABLE WIN**: the sentence survives for online tables, with the online factor stated beside the static one.
- **H1-ON VSA WIN or MIXED**: name the k where the sparse store needs fewer cells than every online coded table; the
  paper may then say that among memories that accept online insertion the sparse block code store needs fewer cells at
  those k (and must say the tables still need fewer when built statically).
- **Against U**: "2 to 4 times fewer than the dense store" is restated for online tables with the measured factor (or
  restricted if H1-ON-U is not a TABLE WIN).
- **The static-directory caveat is stated in every outcome**: the best tables' directory (MPHF) is static; the online
  alternative costs Pen(k) grid steps; H-SBC's superposition is exactly appendable on binary cells; U is appendable only
  with its integer sums (not on its cells), so U's N* is that of a batch-written store.

## Predictions (from the self-test's logical-bit counts on synthetic pairs only; reporting only)

Model: on synthetic stores of 700-6,000 pairs, the cuckoo table holds 1.75 times the MPHF table's logical bits per pair
at f = 14 (28.4 against 16.2 at n = 6,000; 1.62 at f = 12, 1.88 at f = 16), i.e. +2.8 to +3.6 grid steps, and
fingerprint false matches occur at 4e-4 per lookup at f = 14 (model 7 x load x 2^-f). If the codes need the same
redundancy per logical bit as on the MPHF rows, N*(OT-best) is the MPHF best table's N* plus about 3 steps: 32 / 32 / 45
/ 64 / 91 / 362 / 1,218.
- **S1** OT-best N* within 1 step of the model (MPHF best + 3 steps) at every k; Pen(k) in {+2, +3, +4} at every k.
- **S2 (H1-ON)** TIE against H-SBC: |Delta| <= 1 at every labelled k (model 0 / 0 / 0 / 0 / +1 / -1; e05 +1).
- **S3 (H1-ON-U)** TABLE WIN against U: Delta <= -2 at >= 4 of the 6 labelled k, never >= +2 (model -5 / -5 / -3 / -2
  / -1 / -2).
- **S4** O-BCH is OT-best (smaller or equal N*) at k1, k2, k4 and O-LDPC at k8, k16, k32, as for the MPHF tables.
- **S5** dev f at dev N* is 14 or 16 in >= 10 of the 14 (arm, k) cells.
- **S6** no lost insertion on any test store at any f (bucket 4, load <= 0.95).
- **S7 (checks)** second scorer 0 differences on the same decoded bits; 0 evaluations over budget; N* second rule 0
  mismatches; insertion audit clean on every test store.

## Checks (fixed now; b63_check.py)

- **Second scorer by plain loops**: on a sample of >= 16 test cells (both arms; k1, k4, k8, k16 and one budget below N*;
  stores of every size family) the evaluation is repeated locally; on the decoded bitstream, a separately written
  reader re-extracts every slot's fingerprint and payload bit by bit, recomputes the candidate buckets with its own
  hash code and applies the read rule; the exact-read test is re-implemented from the store's own pairs (not
  Store38.want); failed-key lists must be identical to the first reader's on the same bits. Box vs local agreement of
  the failed-key lists is reported separately (the box and local ldpc / numpy versions differ, B38 / B59).
- **Cells audit**: every test evaluation's cells <= floor(B n) (count over budget must be 0); the logical bits equal m x
  4 x (f + payload bits) (empty slots included); the share of the logical bits that are fingerprints and empty slots.
- **N* by a second rule**: recomputed from the per-store failed-key counts by a separately written rule (0 mismatches).
- **Insertion audit**: on every test store and every f used, every placed pair is in one of its two buckets under its
  own fingerprint, placed + lost = n, and the online property (half build + continued insertion = full build).
- **Sample reading by the AI-assisted process** (results/b63_sample_reading.txt): failed keys one grid step below N*
  for OT-best at k1, k8 and k16: the cause (fingerprint false match, nearest-slot fallback after an undecoded codeword,
  wrong value or last flag under an exact match, lost insertion).

## Scope

- U as built is the sign of the bundle's integer sums with ties broken at random; appending a pair to the signs alone
  is not the same store (it needs the sums, or an approximate update rule); no online U is run. H-SBC's geometry (shard
  count, block length) is fixed from n and B, as the cuckoo table's m is: both are provisioned for the final n; neither
  needs the keys in advance. Growing beyond the provisioned n (resizing) is not run for either.
- Not run: the quotient filter (closed-form note above), other bucket sizes and loads, a stash, damage conditions,
  multi-level cells, a dynamic retrieval structure without fingerprints.

## Compute and time cut (registered now)

CPU only, on the A800 box's CPUs (b63_box.sh: own directory ~/b63_run, code and stores MD5-identical to the local
commit, nice 10, one BLAS thread per worker, at most 64 workers; every ssh command in the background; no retry loops
around state-changing commands; no other user's process touched); dev and test on the box (as B38's E-LDPC rows), checks
locally. Time box: 2 hours from 02:16 machine clock (until 04:16). Cut rule, applied by a compute note committed before
any test output is read: if the test cannot finish by 04:00, cut in this order: k32, e05, k2, then the extension (k1,
k4, k8 and k16 are kept).

## Files

Code: b63_online.py, b63_box.sh, b63_verdict.py, b63_check.py. Results: results/b63_dev1.json.gz, results/b63_dev2.json.gz,
results/b63_choice.json, results/b63_plan*.json, results/b63_test*.json.gz, results/b63_verdict.{txt,json},
results/b63_check.txt, results/b63_sample_reading.txt; notes prereg/B63_compute_note*.md, prereg/B63_note_after_test.md.

## Self-test record (synthetic pairs only; b63_online.py selftest, 02:24 machine clock, PASS)

- Insertion (synthetic stores of 150 / 700 / 3,000 / 6,000 pairs, f = 12 / 14 / 16): load 0.938-0.950, 0 lost, every
  placed pair in one of its two buckets under its own fingerprint, half build + continued insertion = full build in 12 /
  12; logical bits per pair 21.3-30.5 against the MPHF table's 11.1-16.2 (n = 6,000: 26.3 / 28.4 / 30.5 at f = 12 / 14
  / 16 against 16.2).
- Noiseless reads: every failure is an ambiguous lookup (fingerprint false match), e.g. 2 / 1 / 0 of 150 probed keys at
  f = 12 / 14 / 16 (700 pairs); false matches per lookup on a 20,000-pair store 0.030 / 0.0077 / 0.0017 at f = 8 / 10 /
  12 (model 0.026 / 0.0065 / 0.0016).
- Noisy runs (700 pairs, f = 14): O-BCH 148 / 150 at B 38 (k1, k4; 2 ambiguous), 128 at B 128 (k8), 6 at B 128 (k16);
  O-LDPC 148 / 150 at B 38 (k1, k2, R 5/6), B 128 (k8, R 1/3), B 512 (k16, R 1/4); every evaluation within its budget.
- Timing (local, under load; 6,000 pairs, f = 14): 4-18 s per job of one evaluation (store construction included).
