# B69 pre-registration: the equal-cell device comparison under both vocabulary accountings, and a sweep of the vocabulary size N

Date: 2026-11-23 (machine clock Asia/Shanghai, about 15:20; US Eastern 02:20). Written and committed, then
OpenTimestamps-stamped (the .ots committed), before any B69 development, test or part (A) computation. The B69 code
(b69_vocab.py) and the closed-form predictions (results/b69_theory.json) are committed with this document. The code
was self-tested on synthetic small-N stores only (record at the end). No B69 sweep store of the dev or test split
exists yet; part (A)'s inputs are the registered per-store failed-key counts of B38 / B59 / B61 / B63, which earlier
experiments read, but no charged requirement has been computed from them. The check code (b69_check.py) is committed
before any B69 test output is read.

## Why (referee M7)

On the equal-cell device comparison (Section 2.3, Fig. 5, Supplementary Note 'hardware'): "Report both accountings
(vocabulary charged and uncharged) in the main text. State that at x1 there are no cell errors. Add a sweep of
vocabulary size N, since the store's need scales with ln(N/eps) and the table's with log2 N, or else limit the claim to
this workload. Qualify the insertion result as logical rather than device-simulated." The paper states the uncharged
numbers (the value vocabulary, index -> value string, is shared and free for every memory; B58 part (b) audit) and, in
SI Note 2, that charging the vocabulary "makes the ratios ties pooled over the 50 stores" (B58: capacity bound, the
dense store and the best static table only, the largest per-pair vocabulary over all stores added to both).

## Fixed (B38 / B59 / B61 / B63's setting unless stated)

Cells: every stored bit is one cell (B38's differential PCM pair read as a sign, binary). Conditions: i.i.d. cell sign
errors at the device model's 30-day bit-error rates (results/b34_ber.json) k1 0, k2 1.3e-4, k4 0.0208, k8 0.1226, k16
0.2760, k32 0.3744, and the 5% control e05 (reported, not labelled). Budget B = cells per stored (key, j) pair, at most
floor(B n) cells on a store of n pairs. Grid 2^(i/4), 8 ... 4,096 (one step = 19%). Metric: a probed key is exact when
all its values come back in order with nothing extra; N* = the smallest grid budget with >= 99.5% of the probed keys
exact there and at every larger evaluated budget, a budget counting only where every store in scope was evaluated
(b58_verdict.nstar_counts; re-implemented in b69_vocab.nstar_counts). Steps = grid index; never reached = 37. Delta =
steps(table) - steps(store) (+ = the store needs fewer cells); per condition "store fewer" if Delta >= +2, "table
fewer" if <= -2, "tie" otherwise; labels TABLE WIN / VSA WIN / MIXED / TIE as B38's H1.

Memories: **U** = the dense bipolar store (B38's V-last); **H-SBC** = the sparse block code store (B61, block load chosen
on dev); **static** = the best static coded table, the smallest N* among T-rep, E-BCH, E-LDPC (and E-CAM in part A), all
with B38's minimal-perfect-hash directory; **insert** = the best insertable table, the smaller N* of O-BCH and O-LDPC
(B63's cuckoo directory with fingerprints, filled one pair at a time). Comparisons: static vs U, static vs H-SBC,
insert vs H-SBC, insert vs U.

## The two accountings (fixed now)

- **Uncharged (registered):** the value vocabulary is the shared output dictionary, free for every memory: the store
  regenerates its value codes from the strings, a table maps its decoded index to the string (B58 part (b)).
- **Charged (new):** on every store, the vocabulary's strings are stored on the same cells, in addition to the memory's
  own cells. The memory keeps its own requirement N* (uncharged, on the same scope), and the vocabulary goes in the
  remaining floor(B n_s) - floor(N* n_s) cells of every store s in scope; **N*_charged** = the smallest grid budget B at
  which that remainder holds the store's protected vocabulary on every store in scope. (A memory that needs fewer cells
  for its own content cannot become worse by charging; the charge is the same per store for every memory.)
- **Vocabulary bits V_s** (format utf8, primary): UTF-8 bytes x 8, the cheaper of a fixed width (the longest string, a
  write-time design constant) and terminated strings (one 0 byte each): 56 bits per value for the 7-digit numbers, 288
  for the UUIDs (B58's figures). **Format compact** (secondary): 7-digit decimal strings as 24-bit integers, version-4
  UUIDs as their 122 random bits, other strings as utf8.
- **Protection** (the same for every memory, because any memory may decode a channel code - B38's free logic - and
  neither store has a code of its own for a flat string list; the store regenerates its codebook from the decoded
  strings, a table returns the decoded string): **cap** = ceil(V_s / (1 - H2(eps))) cells, the binary-symmetric-channel
  capacity, a bound no code reaches (lower bracket; B58's charge); **bch** = the fewest cells over the tables' own
  codes by B38's closed form (b38_eq.q_rep / bch_cw_logfail, exact for i.i.d. errors): repetition with r or r + 1 copies
  (majority, a tie half wrong) or a B38 BCH design (n, k, t) with r or r + 1 copies per coded bit, such that P(any of
  the store's vocabulary strings is wrong) <= 1% (union bound over bits or codewords, every codeword at full length).
  At k8 and above the tables protect their rows with LDPC, which would hold the vocabulary in fewer cells than BCH;
  LDPC has no closed form here and is not simulated, so **cap and bch bracket** the charged requirement there. At k1 (no
  cell errors) both are exactly V_s cells.

## Part (A): both accountings on the registered runs (computation only, no new evaluation)

Inputs: b63_verdict.all_counts (results/b63_choice.json, results/b63_test1.json.gz): U = B38 V-last + B59 T1 rows,
H-SBC = B61's test rows at its dev design, T-rep / E-BCH / E-LDPC / E-CAM = B38's test rows at their registered designs,
O-BCH / O-LDPC = B63's test rows at its dev designs; per-store failed-key counts at every evaluated budget; the 50 RULER
test stores' vocabularies (b34_out/test_stores.json.gz). Conditions k1, k2, k4, e05, k8, k16, k32. Scopes: **pooled**
(50 stores, 4,776 keys; primary), and per family (requirements and charges on that family's stores and keys only):
**mk2** (niah_multikey_2, 20 stores, 56 vocabulary bits per pair), **mk2-6k** (its 10 stores of about 6,000 pairs, the
sweep's N = 6,000 analogue), **mk3** (niah_multikey_3, 10 stores, 288 bits per pair), **small** (multivalue and
variable tracking, 20 stores of 9-23 pairs). Accountings: free, cap-utf8, bch-utf8 (primary charged), cap-compact,
bch-compact. Output: results/b69_partA.{txt,json}: N* per (scope, accounting, memory, condition), Delta and labels per
comparison, the difference in cells per pair. Registered check: the free pooled N* reproduces the registered numbers
(U 76 / 76 / 76 / 91 / 128 / 431 / 1,722; H-SBC 32 / 32 / 45 / 54 / 91 / 304 / 1,448; static 19 / 19 / 27 / 38 / 54 /
215 / 724; insert 32 / 32 / 38 / 64 / 91 / 304 / 1,218).

## Part (B): vocabulary sweep

**Stores** (b69_vocab.make_store, fixed now): B34's synthetic generator as for its mk2big2 dev stores (6,000
adjective-noun keys, two of them with two values, so 6,002 pairs and a last flag; 7-digit values, as RULER's
niah_multikey_2), with an explicit vocabulary of **N in {256, 1,024, 6,000, 24,000}** distinct 7-digit values. Every one
of the N values is a clean-up candidate of the stores (2 N candidates with the LAST variants) and an index of
ceil(log2 N) bits in the tables (b25_pcm.codes is extended to append the unused values; checked: every store's Nv = N).
For N >= 6,002 the stored values are a sample without replacement (N = 24,000: 18,000 vocabulary values are never
stored, as in a fixed dictionary larger than the store); for N < 6,002 every value is used floor or ceil(6,002 / N)
times. n is fixed so that only N changes (the per-pair directory cost of the tables and the stores' load per cell do
not depend on N). 300 probed keys per store (B25's evenly spaced rule with NKEYS = 300). Dev: 3 stores per N (900
probed keys); test: 8 stores per N (2,400 probed keys; the bar allows 12 failures). Seeds 'b69:dev:N:r', 'b69:test:N:r'.

**Memories and designs** (code unchanged: b38_eq.run_job / run_job_auto, b61_hd.sbc_evaluate with explaining away
off, b63_online.run_job): U (no free parameter); H-SBC lam in {0.35, 0.5, 0.7, 1.0}; T-rep (none); E-BCH (B38's closed
form, none); E-LDPC rate R; O-BCH fingerprint width f; O-LDPC (f, R). **Restricted grids** (fixed now from B38's and
B63's dev choices on other stores, to fit the time box): R in {5/6, 9/10} at k1, {2/3, 3/4, 5/6} at k4, {1/4, 1/3, 1/2}
at k8 and k16; f in {12, 14, 16} (18 was never chosen at a B63 dev requirement). A dev choice at an inner edge of a
restricted grid at the dev requirement is reported as a limitation. E-CAM is not run (never the best table in B38).
**Conditions:** k1, k8, k16 (primary), k4 (secondary; first to be cut).

**Development** (dev stores only): every design at the 9 budgets within 4 grid steps of the model prediction (below)
per (memory, N, condition); one extension round (4 steps up for every design where the chosen curve never reaches the
bar, 4 down where the dev requirement is the lowest evaluated budget). Choice per (memory, N, condition, budget): the
design with the highest pooled dev share among designs evaluated on all 3 dev stores; ties the higher share at the
next harder condition (k1 -> k4 -> k8 -> k16) at the same budget, then list order (b69_vocab.choose). The choices are
committed before any test run.

**Test (run once):** per (memory, N, condition) every budget within 4 grid steps of the dev requirement (of the model
prediction for U, T-rep, E-BCH, which have no dev; the top 5 dev budgets if dev never reached the bar), each with the
dev design at that budget (the nearest dev budget's where dev did not evaluate it, ties the larger); then B38's
extension rule (4 steps up while the top evaluated budget fails, 4 down while the requirement is the lowest evaluated
budget), repeated until bracketed, at most 3 rounds.

**Accountings on the sweep:** as part (A), the sweep store's vocabulary being its N 7-digit values (utf8 56 N bits,
compact 24 N bits, on 6,002 pairs: 2.4 / 9.6 / 56 / 224 utf8 bits per pair at N = 256 / 1,024 / 6,000 / 24,000).

## Measures, bars and readings (fixed now)

- **(A)** per scope and accounting: N*, Delta, labels over k1, k2, k4, k8, k16, k32, and the cells-per-pair difference.
- **(B) N*(memory, N, condition)** on the 8 test stores per N; static and insert as minima; Delta per (N, condition) for
  the four comparisons; label per N over k1, k8, k16 (k4 reported beside it).
- **B-H1 (does the uncharged claim hold across N?)**, per comparison: **holds** if its uncharged label is the same at all
  four N; otherwise the N where it differs are named.
- **B-H2 (scaling)**: per memory and condition the step change of N* from N = 256 to 24,000; per comparison and
  condition the change of Delta: "stable" if within 1 step.
- **B-H3 (charged)**: the labels per N under cap-utf8 and bch-utf8; the smallest N at which each comparison is a TIE.
- Readings for the paper: (i) if B-H1 holds for static vs U, static vs H-SBC and insert vs H-SBC, the uncharged device
  sentences may be stated for vocabularies of 256 to 24,000 values (stores of 6,000 pairs), not only for this workload;
  for any comparison where it fails, the sentence is limited to this workload and the N where it changes are named.
  (ii) The charged numbers are stated beside the uncharged ones, with the bracket (cap, bch) where they differ, pooled and
  per family; whether the charged comparison ties depends on the vocabulary's bits per pair (N times the string length
  over the pairs), which is stated. (iii) x1: the measured 30-day error rate at x1 is 0, so no cell errs and every x1
  requirement is the memory's logical content (check C0). (iv) Insertion: B63's insertable table and B61's sparse store
  were both written in one pass as their final contents (the cuckoo moves, the codewords' parity updates and the OR
  insertions were carried out logically, before the cells were programmed), so the insertion result is logical, not
  device-simulated: per-insertion rewrites, parity updates, unequal cell ages and write endurance were not modelled.

## Predictions (closed-form models only; results/b69_theory.json; reporting only)

Models (no store read): U - Gaussian clean-up, true similarity N(sqrt(2B/pi)(1 - 2 eps), 1) against 2N - 1 wrong N(0, 1);
H-SBC - B61's binomial overlap model with lam in the grid, uniform tie break; requirement = smallest grid budget with
slot failure <= 0.5%; the store predictions are the workload's measured N* shifted by the model's step change from N =
6,000. Tables: exact logical bits per pair of a synthetic sweep store at each N (MPHF table 12.4 / 14.2 / 17.2 / 19.2 at
N = 256 / 1,024 / 6,000 / 24,000; cuckoo table at f = 12: 22.1 / 24.2 / 27.4 / 29.5), the workload's cells per logical
bit kept (T-rep at k1: one cell per logical bit). Predicted test N* (k1 / k4 / k8 / k16) at N = 256, 1,024, 6,000, 24,000:
U 54/64/108/362, 64/76/108/362, 76/76/128/431, 76/91/152/512; H-SBC 23/32/76/215, 27/38/76/256, 32/45/91/304,
38/45/108/362; static 13/23/45/181, 16/23/45/181, 19/27/54/215, 23/32/64/256; insert 27/32/76/256, 32/38/91/304,
32/38/91/304, 38/45/108/362.

- **Q1** test N* within 2 grid steps of its prediction in >= 85% of the (memory, N, condition) cells run.
- **Q2** the N = 6,000 sweep reproduces the workload: N* within 1 step of the registered workload N* for every memory at
  every condition run (mk2-like stores of 6,002 pairs against the pooled workload).
- **Q3 (B-H1)** uncharged labels at every N: static vs U TABLE WIN (Delta <= -4 at every N and condition); static vs
  H-SBC TABLE WIN (Delta -1 to -3, predicted -3 / -3 / -3 / -3 at k1); insert vs H-SBC TIE (|Delta| <= 1 everywhere);
  insert vs U TABLE WIN. B-H1 holds for all four.
- **Q4 (B-H2)** from N = 256 to 24,000 every memory's N* rises by +1 to +4 steps at every condition (predicted +2, H-SBC
  +2 to +3, T-rep k1 +3), so every Delta is stable (within 1 step) except possibly static vs U at k1 (predicted -8 ->
  -7). Both needs grow logarithmically: the store's with ln(N / eps) through the 2N candidates it must reject, the
  table's with log2 N through its index; neither ratio drifts by more than one step over a 94-fold range of N.
- **Q5 (B-H3)** charged (bch-utf8): at N = 256 every label equals the uncharged label; at N = 24,000 (224 bits per pair)
  every comparison is a TIE at k1, k8 and k16; at N = 6,000 static vs U is still "table fewer" at k1 (Delta <= -3) and
  static vs H-SBC a tie at k1, k8, k16. Under cap-utf8 the same at N = 256 and N = 24,000.
- **Q6 (A)** the free accounting reproduces the 28 registered pooled N*; pooled and charged (utf8, cap or bch), every
  comparison is a TIE at every labelled k (the 288-bit UUID strings of niah_multikey_3 set the charge, as B58); in mk2
  (56 bits per pair) static vs U stays TABLE WIN under cap-utf8 (Delta <= -2 at k1-k8) and static vs H-SBC is a tie at
  k1 (32 + 56 against 19 + 56 cells); with the compact format the mk2 static vs U label is TABLE WIN under both
  protections.
- **Q7 (checks)** 0 evaluations over budget; Nv = N and 300 probed keys on every sweep store; the independent recount
  agrees on every requirement; no flipped cell at x1 in 2 x 10^8 device-model cells (30 days and 1 year).

## Checks (fixed now; b69_check.py, committed before any test output is read)

- **C0 (x1 has no cell errors):** B34's device model (b34_inmem.signs_over_times, b25_pcm's programming noise, drift,
  read noise and ADC) at k = 1 on 2 x 10^8 random cells (10 seeds x 2 x 10^7, new seeds 69000-69009), read at 30 days and
  1 year: flipped cells counted (95% upper bound 1.5e-8 per cell if none), and the smallest |G+ - G-| margin relative
  to G_ON. Plus, mechanically: the i.i.d. channel at k1 (eps = 0) returns the cells unchanged (b38_eq.Channel.apply), and
  every memory's k1 requirement equals its k2 requirement (1.3e-4) in part (A).
- **C1 bits audit:** every sweep evaluation's cells <= floor(B n) (count over budget must be 0); every sweep store's
  vocabulary size Nv = N, index bits ceil(log2 N), stored-value counts as specified, 300 probed keys; logical bits per
  pair of each table (MPHF rows and directory; cuckoo slots = m x 4 x (f + 1 + ceil(log2 N))) reported per N; the
  vocabulary bits V_s recomputed by a second method (characters counted directly, not via the encoder) for every store
  in (A) and (B).
- **C2 independent recount:** every requirement in (A) and (B), uncharged and charged, recomputed by separately written
  code: raw rows -> per-budget pooled failures with explicit loops (not aggregate / nstar_counts), its own capacity
  formula, and for bch its own binomial tail sums (math.comb, not scipy) re-evaluating the chosen vocabulary designs'
  failure probability; 0 mismatches expected.
- **C3 sample recount:** for 8 requirements drawn by a seeded rule (both parts), the per-budget failed-key totals around
  N* listed from the raw rows (results/b69_check.txt).
- **C4 cross-machine:** 6 sweep test jobs re-run locally (seeded choice), failed-key lists compared with the box's.

## Compute, scope and time cut (registered now)

CPU only, on the A800 box's CPUs (b69_box.sh: own directory ~/b69_run, code and stores MD5-identical to the local
commit, nice 10, one BLAS thread per worker, at most 64 workers; other agents' runs share the box; every ssh command
in the background; no retry loops around state-changing commands; no other user's process touched); part (A) and the
checks locally. Time box 3 hours from 14:48 machine clock (until 17:48). Cut rule, applied by a compute note committed
before any test output is read: if the test cannot finish by 17:05, cut in this order: k4, N = 256 (keeping the current
workload's N = 6,000 and one size on each side, 1,024 and 24,000), test extension rounds after the first. Part (A) and
the checks are not cut. Not run: LDPC-protected vocabulary (bracketed instead), multi-level cells, damage conditions,
other n, explaining away (H-SIC), E-CAM, device-level simulation of insertions.

## Files

Code: b69_vocab.py, b69_box.sh, b69_check.py. Results: results/b69_theory.json, results/b69_partA.{txt,json},
results/b69_dev*.json.gz, results/b69_choice.json, results/b69_plan*.json, results/b69_test*.json.gz,
results/b69_verdictB.{txt,json}, results/b69_check.txt. Stores b69_out/sweep_{dev,test}.json.gz (regenerated by
`b69_vocab.py stores`; MD5 in a compute note). Notes prereg/B69_compute_note*.md, prereg/B69_note_after_test.md.

## Self-test record (synthetic stores of a 'selftest' and a 'timing' split only)

- b69_vocab.py selftest (local, 15:00): synthetic stores of 402 pairs with N = 64 and N = 512 (110 unused values
  appended): Nv = N, index bits ceil(log2 N), 50 probed keys; every memory's job ran at one budget at k1 and k8 within
  its budget (cells <= floor(B n)), 45-50 of 50 keys exact; vocabulary bit counts (56 / 24 bits per 7-digit value; fixed
  width vs terminated) and the N* rule on a hand-made curve correct; vocabulary cost per bit for 336,000 bits: k1 1.000
  (both), k4 cap 1.171 / bch 1.604 (BCH 1023 / 638, t = 42, one copy), k8 cap 2.159 / bch 7.02, k16 (489,600 bits) cap
  6.67 / bch 29.5. PASS.
- The models' numerics: the sparse-store tie formula was first computed by direct subtraction, which cancels to garbage
  below 1e-12 (non-monotone in B); rewritten in log space before the predictions above were made.
- Timing (local, under load) on one synthetic store of a separate 'timing' split (N = 24,000, 6,002 pairs, not a dev or
  test store): one job (store construction and one evaluation) took U 59-62 s (k1 at 76, k8 at 152 cells per pair),
  H-SBC 19 s (k8, 108), E-LDPC 18 s (k8, 64); hence U's jobs at large N are split one budget per job. The printed
  shares are of that timing store only and inform nothing registered here.
