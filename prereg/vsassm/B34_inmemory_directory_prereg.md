# B34 pre-registration: the VSA store against tables whose key directory lives on the same noisy devices

Registered 2026-11-18 (US Eastern), before any test run. Written before any B34 arm touched a test store: the code
(b34_inmem.py, b34_check.py, b34_verdict.py) was written and self-tested on synthetic pairs, the device model's cell
bit-error rate was measured on random cells (results/b34_ber.json), and the free design parameters of the tables were
chosen on synthetic dev stores only (record below). The 50 test stores were read once to copy their pairs into
b34_out/test_stores.json.gz (no arm, no device draw); nothing else about them was looked at.

## Why

B25/B26b placed the RULER stores (50 stores, 4,776 probed keys, 16 pairs per vector, bipolar store, d = 8,192) on the
Joshi et al. 2020 phase-change device model. With the store's clean-up kept digital, a table that spent the store's cell
count on a repetition code was more robust than the VSA store (30 days: 98.8% vs 0.15% at x16; 59.5% vs 0.0% at x32).
But both B26b tables kept their key -> row directory and each key's value count digital ("which favours the tables").
A VSA store keeps neither: keys are bound into the content and a read is by similarity. B34 asks whether that changes
the answer when every stored bit, including whatever a table needs to find a key's row, lives on the same noisy devices,
at an equal device count.

## The setting (fixed now)

- **Stores, keys, metric, device, grid:** B26b's. The 50 B22/B25 stores (b21_dl/b21_test), the same 4,776 probed keys,
  16 pairs per vector; B25's cell model (programming noise, per-device drift exponent, 1/f read noise, 8-bit ADC; every
  stored bit one differential pair, +1 -> (G_ON, RESET), -1 -> (RESET, G_ON), read as a sign, ADC zero -> random);
  times 60 s, 1 h, 1 day, 30 days, 1 year; noise multiplier k = 1 (measured model), 2, 4, 8, 16, 32 applied to every
  noise source at once; common random numbers across k. Metric: the share of the 4,776 probed keys read exactly (all
  values, in order, nothing extra).
- **What counts as stored state, on the devices in every arm:** every bit an arm keeps between write and read: the VSA
  store's vectors; a table's rows, its key codes / CAM words, its hash displacement array, its last-value flags.
  **Stateless logic, allowed in every arm:** hashing or encoding the query key string (the VSA's key, position and
  value codes are themselves seeded hashes of strings), binding/unbinding, majority gates, comparisons, Hamming-distance
  search over rows, address arithmetic. **Per-store digital design parameters, allowed in every arm:** sizes and widths
  fixed at write time (the VSA's shard count S and loop bound; a table's field widths, repetition factors, hash-table
  size, the store's largest j).
- **Shared output dictionary (not counted, same for every arm):** the value vocabulary (value index -> value string).
  Every arm returns vocabulary indices, as in B25/B26b. The VSA's digital clean-up regenerates its codebook from these
  strings (V-S); V-SC instead stores the codebook on the devices.
- **Device budget:** cells (each = 2 devices). Primary budget C = the VSA store's cell count, S x 8,192 (as B26b). A
  table at budget C uses at most C cells; each arm's actual share of C is reported.
- **Queries are probed keys, which are always stored** (as in B25/B26b). No arm is scored on rejecting an absent key.
  The tables therefore accept their nearest match without an acceptance threshold (this favours the tables); V-Snt
  gives the VSA the same assumption for its first value.

## Arms (fixed now; one row per stored (key, j) pair for every table)

VSA (B25/B26b code, unchanged):
- **V-S** (primary, held as in B26b): the bipolar store on the devices, digital clean-up, threshold 0.5 at every j,
  stop at the first j below. B26b's seeds, so it must reproduce B26b's VSA counts exactly (gate G1).
- **V-Snt** (secondary): the same device reads, but the first value is taken by argmax without the threshold (the
  tables' "the key is present" assumption); the stop at j >= 1 keeps the 0.5 threshold.
- **V-SC** (secondary): store AND clean-up codebook (N_values x 8,192 differential pairs) on the devices, B25's B1-SC
  path with global drift compensation (GDC) on the clean-up crossbar. Its device count is (S + N_values) x 8,192.
- **V-SCnt** (descriptive): V-SC with V-Snt's first-value rule.

Tables (in-memory: every stored bit on the devices):
- **T-i, CAM + exact match over the keys' native codes.** CAM word = the key's UTF-8 bytes zero-padded to the store's
  longest key, then j in ceil(log2(max j + 1)) bits, one cell per bit; a row matches only if every cell of its word
  equals the query's bit. RAM part = the value index (ceil(log2 N_values) bits) repeated r_v = floor((W - w_key) / b)
  times with W = floor(C / n_pairs) cells per row, majority (ties at random). Read j = 0, 1, ... until (key, j) finds
  no match. (Repeating the CAM word would only hurt exact matching; the leftover cells go to the values.)
- **T-iih, CAM with repetition-coded native keys, majority matching.** Word and value bits each repeated
  r = floor(W / (w_key + b)) times; each bit majority-decoded; exact match of the decoded word; stop on no match.
- **T-iis, CAM with repetition-coded native keys, Hamming matching.** Word repeated r_d times, searched by the nearest
  Hamming distance over all rows (soft, ties at random); a last-value flag (when some key has more than one value) and
  the value bits repeated r_p times, majority; stop at the last flag. r_d = max(1, round(s W / w_key)) for the dev-chosen
  share s, r_p = floor((W - r_d w_key) / (flag + b)).
- **T-iix, CAM with hashed keys (added: a stronger non-hyperdimensional directory).** As T-iis with the native word
  replaced by an h-bit SHA-256 hash of (key, j), h = the smallest h >= max(8, ceil(2 log2 n_pairs)) for which the
  store's pairs get distinct codes (about 26-27 bits for the largest stores): a low-dimensional code whose redundancy
  comes only from repetition.
- **T-v, a hash-to-row mapping stored as data (added: the strongest table we could design).** A minimal perfect hash
  (hash and displace, CHD; Belazzougui, Botelho & Dietzfelbinger 2009): (key, j) -> bucket by a stateless hash; each
  bucket stores a displacement sigma (db bits) on the devices; row = (h1 + (sigma mod R) h2 + sigma div R) mod R over
  R = the smallest prime >= n_pairs rows (the next prime if a bucket finds no displacement). Rows hold the last flag and
  the value bits. Every logical bit (displacements and rows) repeated r = floor(C / logical bits) times, majority
  (uniform repetition is optimal when every bit is hard-decoded and equally needed). No key is stored: it cannot reject
  an absent key (not scored here; stated in the reading). Average keys per bucket lambda: dev choice.
- **T-iiia, CAM with random high-dimensional key codes (hyperdimensional: the boundary case).** Row code =
  HD(key) * HD(pos j) (elementwise product of seeded random +-1 codes) of w_k = round(s W) cells, nearest Hamming search
  over all rows; last flag + value bits repetition-coded with the remaining cells.
- **T-iiib, random high-dimensional key AND value codes (hyperdimensional).** T-iiia's key field (w_k = round(s W)) plus
  a value field of W - w_k cells holding HD(value), multiplied by HD(last) on a key's last value when some key has more
  than one value, decoded by the nearest code over the store's vocabulary (x 2 last variants), digitally like V-S's
  clean-up. These are the VSA's two ingredients (random high-dimensional codes, similarity search) without superposition.
- **T-vhd, the stored hash mapping with high-dimensional values (hyperdimensional values; added after the first dev
  grid).** T-v's directory (displacements repetition-coded with a share s of the cells) and, in each row, T-iiib's value
  code (HD(value), times HD(last) on a key's last value) over the remaining cells, decoded digitally over the
  vocabulary. No stored key, no superposition.
- **T-vhdd, every field high-dimensional (the boundary case; added after the dev full run).** T-vhd with each
  displacement split into two blocks of ceil(db / 2) bits, each block stored as a random code of w_b cells and decoded
  by the nearest of its 2^ceil(db/2) candidate codes; share s of the cells to the displacement codes. Random
  high-dimensional codes with similarity decoding in every field, no superposition, no stored key: the VSA's
  ingredients minus superposition.
- **T-iv, reference (not in-memory):** B26b's Trep reproduced exactly (digital directory and counts; value bits
  repeated r = floor(C / (n_pairs b)) times). Must reproduce B26b's banked Trep counts (gate G1).

Equal total devices for V-SC: the five non-hyperdimensional in-memory tables again at C' = (S + N_values) x 8,192
cells, with every cell's sign flipped independently at the device model's measured bit-error rate p(k, t) (one uniform
per cell shared by every k and t). This is exact in distribution: the model's cells are i.i.d. (independent per-device
draws, no spatial correlation, symmetric for +1 and -1), and it keeps 17-fold larger arrays affordable.

## Why this is a fair and strong competition

- Every table keeps every bit it needs on the same devices, at most the VSA store's cell count, with the same reads,
  times and noise. The VSA's own address computation (shard = a stateless hash of the key) and its codes (hashes of
  strings) are stateless logic; the tables get the same freedom (hashing, majority gates, Hamming search, arithmetic).
- The user-named designs are covered: (i) T-i; (ii) T-iih (majority) and T-iis (Hamming); (iii) T-iiia/T-iiib;
  (iv) T-iv. Two stronger non-hyperdimensional designs are added (T-iix, T-v), because native key codes are a weak
  directory (two keys that share a noun differ in a few bits) and a stored key is not needed at all to find a row.
- Each soft-search table's one free parameter (the directory's share of a row) was tuned on synthetic dev stores; the
  VSA has no tuned parameter. The best non-hyperdimensional table is the maximum over five designs in every cell.
- Not included, with the reason: algebraic codes with bounded-distance decoding (BCH, as in memory ECC) correct at most
  a fraction t/n < 1/4 of a codeword at the rates needed here (e.g. BCH(511, 19) t = 119, 23%), so at x16 and x32
  (cell error 24-38%) they fail where repetition with majority (maximum-likelihood decoding of the repetition code)
  still works, and at x8 and below every design here is near 100% anyway. Maximum-likelihood decoding of a long code
  over its codebook is exactly the high-dimensional clean-up that T-iiib tests.

## Bars (fixed now)

Delta(k, t) = arm - best non-hyperdimensional in-memory table (max over T-i, T-iih, T-iis, T-iix, T-v), in points.
Per noise level k >= 4 at the primary time, 30 days:
- "VSA ahead" if Delta >= +5; "table ahead" if Delta <= -5; "tie" if |Delta| <= 2; otherwise "small difference".
- **VSA WIN:** no k >= 4 with Delta < -2, and at least one k >= 4 with Delta >= +5.
  **VSA WIN (strict, the user's example bar):** Delta >= +5 at every k >= 4.
- **TABLE WIN:** no k >= 4 with Delta > +2, and at least one k >= 4 with Delta <= -5.
- **TIE:** |Delta| <= 2 at every k >= 4. **MIXED:** VSA ahead at some k >= 4 and table ahead at another.
  Otherwise **UNRESOLVED**.
- **H1 (primary):** V-S vs the best non-HD table at budget C.
- **H2:** V-Snt vs the best non-HD table at budget C.
- **H3:** V-SC vs the best non-HD table at equal total devices C' (descriptive also at budget C, which favours V-SC).
- **H4 (boundary):** V-S and V-Snt vs the best hyperdimensional table (max of T-iiia, T-iiib, T-vhd, T-vhdd) at budget
  C, same rule.
- Each bar is also read at the other four times (secondary: "holds at every time" or not).
- **Gates (checks):** G0 at k = 1 every arm >= 99.5% at every time (all designs are exact on noiseless cells).
  G1 V-S and T-iv reproduce B26b's banked counts (results/b26_pcm_table.json) for every store, k and time.

## How the paper should read each outcome (fixed now)

- **H1 TABLE WIN:** "Even with every bit, including the table's directory, on the same noisy devices, a table at the
  same device count is more robust than the VSA store." Name the design (expected: the stored hash mapping, T-v). The
  paper may not claim a robustness advantage for the vector form; the B26b sentence is extended, not withdrawn.
- **H1 VSA WIN:** "Once the directory a table needs must also live on the noisy devices, the VSA store is more robust
  than every table design we built, because it stores no directory." Then H4 says where the robustness comes from: if
  a hyperdimensional table (T-iiia/b, T-vhd, T-vhdd) is within 2 points of the VSA or better at every k, the
  robustness belongs to random high-dimensional codes with similarity search (the VSA ingredient), not to
  superposition; if the VSA is ahead of every hyperdimensional table by >= 5 at some k, superposition itself (storing
  no per-pair address) adds robustness there.
- **H1 MIXED:** report which designs win at which noise levels, and which ingredient explains it; no one-line claim.
- **If only the user-named native-code CAMs (T-i, T-iih, T-iis) lose to the VSA while T-iix or T-v match or beat
  it:** the VSA's advantage over CAM tables is not needing to store the keys, which any hashed design shares; the paper
  may say the VSA beats a CAM that holds the keys, not that it beats a table.
- **H2 VSA WIN (with H1 TABLE WIN):** "Read as published (fixed threshold 0.5), the VSA store loses to a hashed table
  in memory; given the tables' own assumption that a probed key is present (first value by argmax), the VSA store is
  more robust than every non-hyperdimensional table at the noise levels where Delta >= +5." The paper must then give
  H4nt: if a hyperdimensional table without superposition (T-vhdd) matches or beats V-Snt, the advantage belongs to
  random high-dimensional codes with similarity decoding, which a table can use too, not to the vector store as such.
- **If V-S and V-Snt read differently:** the VSA's fixed acceptance threshold, not its code, sets its noise limit;
  the paper reports both and says which assumption each needs.
- **H3:** V-SC at equal total devices is the fully in-memory VSA; its codebook costs N_values / S (about 16) times the
  store's cells, and the reading says so.

## Predictions (informed by the synthetic dev runs; see the dev record at the end)

At 30 days unless stated; percentages of the 4,776 keys. Checked mechanically by b34_verdict.py (reporting only).
- **P1** native exact matching dies first: T-i <= 10% and T-iih <= 15% at x4 (dev 1.0 / 4.2; at x4, 30 days, a cell
  fails with p = 2.1%, so a 233-bit word survives with 0.979^233 = 0.7%; T-iih gets only r = 2 copies, and a 2-copy
  majority with random ties is no better than one copy).
- **P2** T-iis (native keys, Hamming search): >= 98 at x4, 80-95 at x8, 20-50 at x16, <= 10 at x32 (dev 99.5 / 88.2 /
  34.7 / 3.2): keys that share a word differ in a few bits of their native code.
- **P3** T-iix (hashed keys): >= 99.5 at x4 and x8, 60-85 at x16, 5-20 at x32 (dev 100 / 99.8 / 73.5 / 10.4).
- **P4** T-v (stored hash mapping) is the best non-hyperdimensional table at x16 and x32: >= 99.5 at x8, 85-97 at x16,
  10-30 at x32 (dev 100 / 91.2 / 17.8).
- **P5** H1 is a TABLE WIN: |Delta| <= 2 at x4 and x8, Delta <= -80 at x16 (V-S 0.15%, B26b's banked value), and a
  TABLE WIN at every time (dev: Delta 0 / -1.5 / -91 / -18).
- **P6** V-Snt: >= 99.5 at x8, 97-100 at x16, 65-90 at x32 (dev 100 / 99.5 / 79.0): without the threshold on the first
  value, the store survives where V-S's fixed threshold fails (true similarity about 0.8 (1 - 2p): 0.36 at x16 against
  the 0.5 threshold, but far above the best wrong value, about 0.18).
- **P7** H2 is a VSA WIN (not strict): ties at x4 and x8, Delta +2..+12 at x16, >= +30 at x32, at every time (dev 0 / 0 /
  +8.4 / +61).
- **P8** the hyperdimensional tables: T-iiib, T-vhd and T-vhdd >= 99.5 at x16; at x32 T-iiib 35-60, T-vhd 45-70,
  T-vhdd 70-90; T-iiia 80-95 at x16 (its repetition-coded values limit it) (dev 100 / 100 / 100; 46.7 / 59.9 / 82.1;
  87.8).
- **P9** H4 (V-S vs the best HD table) is a TABLE WIN; for H4nt, |Delta| <= 2 at x4, x8, x16 and V-Snt - T-vhdd in
  [-8, +2] at x32 (dev -3.1): a table with random high-dimensional codes in every field and no superposition matches
  or beats the VSA store everywhere.
- **P10** V-SC <= 5 at x8 (B25's GDC gain bias on the clean-up crossbar); V-SCnt 95-100 at x8 and <= 10 at x16; H3 a
  TABLE WIN; at equal total devices T-v@tot and T-iix@tot >= 99 at x32 (dev 0.3; 98.6, 1.0; 100, 99.8).
- **P11** G0 and G1 pass (G1: 3,000 store x k x time counts, 0 mismatches).
- **P12** closed form: no flagged cell for the exact-formula arms (T-iv, T-i, T-iih, T-v); at most 10 flagged cells in
  all.

Expected reading if these hold: with B26b's VSA read (V-S), the stored-hash table T-v is more robust in a fully
in-memory setting too (TABLE WIN); the native-key CAMs the referee would think of first lose to the VSA store at x8
(T-iis 80-95 vs 98.6) but the hashed directories do not; given the tables' own "key present" assumption, the VSA store
(V-Snt) is more robust than every non-hyperdimensional table at x16 and x32, and a table that uses random high-dimensional codes in every field (T-vhdd) matches it, so the ingredient is the
random high-dimensional code with similarity decoding, and superposition is not needed for the robustness.

## Checks from other angles (fixed now)

- G1 reproduction (above), and the noiseless self-test (every table reads every key exactly on noiseless cells).
- **Closed form (b34_check.py closed):** for every arm except V-SC, the expected share of keys read exactly computed
  from the noiseless codes and the measured cell bit-error rate p(k, t) alone, without simulating a cell: binomial
  majority for the repetition-coded fields (exact), the exact distribution of every row's Hamming distance for the
  nearest-row search (rows are independent; ties approximated), independence across vocabulary candidates for T-iiib's
  value decode, and a Gaussian approximation of the clean-up similarities for V-S / V-Snt. Flag any cell where
  simulation and closed form differ by more than max(1 point, 4 standard errors).
- **Hand-read failures (b34_check.py handread):** at least three failed keys per arm in the cells that decide the bars,
  with the cause (which cells flipped, which row was chosen, which similarity fell under the threshold).
- **Literature (after the result):** Karunaratne et al. 2020 (in-memory HDC on PCM), Ni et al. 2019 (FeFET TCAM),
  Kleyko et al. 2022 (Proc. IEEE, VSAs for emerging hardware), and work on CAM / associative-memory robustness and on
  HD codes as channel codes; each recorded as supporting, conflicting or no precedent.

## Compute

CPU only: the A800 box's CPU cores (no GPU; the user allowed the box and PACE) or the local machine. Estimated 6-8
CPU-hours for the test grid. No paid API.

## Files

- Code: b34_inmem.py (arms, dev, test), b34_check.py (closed form, hand reading), b34_verdict.py (tables, bars,
  predictions), b34_box.sh (runner).
- Data: b34_out/dev_stores.json.gz, b34_out/test_stores.json.gz.
- Results: results/b34_ber.json, results/b34_dev_grid.json, results/b34_dev_full.json, results/b34_choice.json,
  results/b34_inmem.json (test), results/b34_check.txt, results/b34_handread.txt, results/b34_verdict.txt,
  results/b34_literature.txt.

## Dev record (2026-11-18 US Eastern, before any test run; synthetic stores only)

- **Cell bit-error rate of the device model** (b34_inmem.py ber; 2e7 random cells; results/b34_ber.json), 60 s / 30 days:
  x1 0 / 0; x2 0 / 0.013%; x4 0.15% / 2.08%; x8 7.56% / 12.26%; x16 24.04% / 27.60%; x32 36.98% / 37.44%. These rates
  feed the equal-total (i.i.d.) arms and the closed form; B26b's VSA bit errors (e.g. 0.123 at x8, 30 days) agree.
- **Dev stores** (b34_inmem.py dump_dev; b34_out/dev_stores.json.gz): 12 synthetic stores, two of each kind: 6,000
  adjective-noun keys from RULER's wonderwords lists with 7-digit values; the same with two keys holding 2 values;
  1,500 adjective-noun keys; 1,700 uuid keys with uuid values; 18 keys holding 4, 2, 2 and 1 values; 9 keys of 3-5
  random letters with 5-digit values. 1,254 probed keys (B25's probe rule). No reader, no test data.
- **MPHF lambda** (b34_inmem.py mphf_lambda; results/b34_dev_mphf_lambda.json): overhead bits per pair (displacements +
  empty rows) 3.20 / 2.90 / 2.76 / 2.95 / 2.80 for lambda = 3 / 4 / 5 / 6 / 8 -> **lambda = 5** (displacement width
  db = 16 bits on the largest dev stores).
- **Share grid** (results/b34_dev_grid.json for T-iis, T-iix, T-iiia, T-iiib; results/b34_dev_grid_vhd.json
  and results/b34_dev_grid_vhdd.json for T-vhd and T-vhdd, run with lambda = 5), rule in b34_choose.py (highest mean over x4-x32 at
  30 days among shares exact at x1; ties within 0.1 to the smaller share): **T-iis 0.1** (r_d = 1: the native word
  once, 20 copies of each value bit on the largest stores; dev mean 56.7), **T-iix 0.4** (h = 26-28 bits x 7-8
  copies; 70.8), **T-iiia 0.4** (76.8), **T-iiib 0.5** (256 + 256 cells; 87.1), **T-vhd 0.5** (89.9), **T-vhdd 0.3**
  (96.2). Choice file: results/b34_choice.json.
- **Reproducibility:** the dev grid run on the A800 box (numpy 1.26.4) and locally (numpy 2.2.6) gave identical
  counts in all 900 cells. (The box run was started by a launch command whose ssh step I believed had failed; same
  code and data, so it served as this check.)
- **Arms added during dev, with the reason (both hyperdimensional boundary cases; no non-HD arm or bar changed):**
  T-vhd after the first grid (T-iiia was limited by its repetition-coded values and T-iiib paid for a key search the
  hash mapping avoids); T-vhdd after the first full dev run showed V-Snt ahead of T-iiib and T-vhd by 19-32 points at
  x32 (the first full dev run is kept as b34_out/box/dev_full_pre_vhdd on the box; the second, with T-vhdd, gave
  identical counts for every other arm).
- **Full dev run with the chosen designs** (all arms, all k and times; results/b34_dev_full.json, about 4 minutes on 48
  box cores): 30 days, x4 / x8 / x16 / x32:
  V-S 100 / 98.5 / 0.1 / 0; V-Snt 100 / 100 / 99.5 / 79.0; V-SC 100 / 0.3 / 0 / 0; V-SCnt 100 / 98.6 / 1.0 / 0.2;
  T-iv 100 / 100 / 98.5 / 58.0; T-i 1.0 / 0 / 0 / 0; T-iih 4.2 / 1.4 / 0.5 / 0; T-iis 99.5 / 88.2 / 34.7 / 3.2;
  T-iix 100 / 99.8 / 73.5 / 10.4; T-v 100 / 100 / 91.2 / 17.8; T-iiia 100 / 100 / 87.8 / 17.2;
  T-iiib 100 / 100 / 100 / 46.7; T-vhd 100 / 100 / 100 / 59.9; T-vhdd 100 / 100 / 100 / 82.1;
  equal total devices: T-v@tot 100 at every k; T-iix@tot 99.8 at x32. Bars on dev: H1 TABLE WIN, H2 VSA WIN (not
  strict), H3 TABLE WIN, H4 TABLE WIN, H4nt UNRESOLVED at 30 days (x32 Delta -3.1) and TABLE WIN at 60 s (-6.9).
- **Closed-form check on dev** (a check of the check code, not a gate): on one large dev store (dev/mk2big2/0, 150
  keys, 30 days) the closed form and the simulation agree within sampling error for every arm (e.g. x16: T-v 128.2 vs
  137, T-iix 105.4 vs 110, T-iiia 122.7 vs 124, V-S 0.2 vs 0, V-Snt 149.9 vs 150; x32: T-vhdd 112.4 vs 104, V-Snt
  111.1 vs 117, T-iiib 52.4 vs 52); the full dev comparison runs on the box (results/b34_dev_check.txt).
- Not yet run on the test stores: anything. The test run is b34_box.sh test (b34_inmem.py test, all arms, all k and
  times, 300 jobs), then b34_box.sh test-check, b34_verdict.py and b34_check.py handread.

### Hand reading (fixed now)

For each arm, three failed keys at the lowest k >= 4 where the arm reads < 99.5% at 30 days (the first failing stores
in store order), and additionally V-S at x16, V-Snt at x32, T-v at x16 and T-vhdd at x32: printed by
b34_check.py handread with the cause, into results/b34_handread.txt.
