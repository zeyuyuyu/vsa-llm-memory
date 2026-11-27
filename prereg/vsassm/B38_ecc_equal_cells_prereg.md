# B38 pre-registration: cells per stored fact for exact reads, VSA store vs error-corrected tables, under device noise and localized damage

Registered 2026-11-18 (US Eastern), before any test run. The code (b38_eq.py, b38_design.py, b38_check.py,
b38_verdict.py) was written and self-tested on synthetic pairs, the LDPC column weights were chosen on synthetic
random codewords, and every free design parameter was chosen on B34's 12 synthetic dev stores (record below). The 50
RULER test stores (b34_out/test_stores.json.gz, copied by B34) have not been read by any B38 arm.

## Why

A strict referee on B34: "Required: comparisons with tables protected by real error-correcting codes (Hamming, BCH,
LDPC), not only repetition; the cells per fact each memory needs for 99.5% exactness at each noise level, with the item
memory counted. The RULER stores run far below capacity (16 pairs in d = 8,192), so exactness at low precision is
expected, not a finding." B34 compared at one budget (the store's cells at 16 pairs per vector) against
repetition-coded tables only, and only under i.i.d. cell errors. B38 asks the referee's question directly, and also
tests the classic claim for VSAs that a distributed (holographic) code degrades gracefully, "irrespective of position"
(Kanerva 2009), under localized damage, against ECC tables that may interleave.

## Setting (fixed now)

- **Stores, keys, metric.** B34's 50 RULER stores (131072/niah_multikey_2, 131072/niah_multikey_3,
  131072/niah_multivalue, 131072/vt, 32768/niah_multikey_2; 10 each) and the same 4,776 probed keys (B25's probe rule).
  A key is read exactly when all its values come back, in order, with nothing extra. Share of probed keys read exactly.
- **Cells.** Every stored bit is one cell (a differential pair of PCM devices read as a sign, B25/B34's cell model).
  B34 showed the device model's sign errors are i.i.d. per cell and symmetric (programming noise, per-device drift,
  read noise, ADC; independent per device), so the i.i.d. conditions flip each cell independently at the device
  model's measured 30-day bit-error rate p(k) from results/b34_ber.json: k1 0, k2 0.000132, k4 0.0208, k8 0.1226,
  k16 0.2760, k32 0.3744 (k = noise multiplier on every noise source), plus a control at 5% (e05). A check re-reads a
  subset with the device model itself.
- **Budget.** B = cells per stored (key, j) pair. On a store with n pairs an arm may use at most floor(B n) cells, and
  every bit it keeps between write and read is on them: the VSA's vectors, a table's directory (minimal-perfect-hash
  displacements or CAM hash codes), its rows, its parity bits, and the VSA's clean-up codebook where stored (V-cb).
  Grid: B = 2^(i/4), i = 12 ... 48 (8 ... 4,096; quarter octaves, one grid step = 19%).
- **Free for every arm (stateless logic):** hashing the query key, generating seeded codes from strings, binding,
  majority, decoders (BCH, belief propagation), similarity search, address arithmetic; per-store design parameters
  fixed at write time (sizes, code parameters, loop bound). **Shared output dictionary (not counted, same for every
  arm):** the value vocabulary (index -> value string), as in B25/B34; every arm returns vocabulary indices.
- **Item memory.** The VSA store needs, at read time, the key, position, LAST and value codes. In V-pub/V-Snt/V-am/
  V-last they are regenerated digitally from the strings (seeded hashes; 0 cells; the price is computing N_values x d
  code bits and dot products per query, which the paper must state). In V-cb the value codebook (N_values x d) is
  stored on the same cells and counted (key, position and LAST codes are generated from the query, as a table hashes
  the query key). The HD table T-vhdd also regenerates its codes from seeds (0 cells). The tables need no item memory
  beyond the shared vocabulary.

## Error models (fixed now)

(a) **i.i.d.** cell sign errors at p(k) (above), one uniform per cell per (store, arm, budget, design) shared by all
rates (monotone coupling).

(b) **Localized damage**, damage only (healthy cells read correctly, as at k1): a damaged cell reads a fixed random sign,
independent of what it stores (a stuck or disconnected device pair), so its error rate is 1/2. The damaged fraction is
f = 2 eps, which matches the overall cell-error rate eps of an i.i.d. condition; f = 0.0416, 0.10, 0.2453, 0.5520 for
eps = p(k4), 0.05, p(k8), p(k16). Physical array: cells in rows of 256 (a word line), tiles of 256 x 256 cells; an
arm's cells fill the array in its natural order or through a random interleaver. Geometries:
- **lines**: dead word lines and bit lines; every row and every (tile, column) is dead independently with probability
  q, 1 - (1 - q)^2 = f.
- **clust**: clusters of stuck cells: aligned 16 x 16 blocks, each dead with probability f.
- **contig**: a contiguous fraction f of the arm's array is lost (a band of round(f N) cells in physical order,
  starting at a uniform cell, wrapping).
The damage map is a property of the physical array (seeded per store, budget and geometry; shared by every arm; nested
in f). Layouts:
- **il (primary):** a seeded random interleaver over the whole array, available to every arm (as a competent engineer
  would interleave ECC words; Karunaratne et al. 2020 spread hypervectors across the crossbar for the same reason).
- **nat (secondary):** natural order with no interleaver: a VSA shard vector's d cells contiguous (32 rows of 256 at
  d = 8,192), a table's codewords contiguous with the copies of each bit adjacent (B34's layout), T-vhdd's codes
  contiguous.
- **known (secondary, lines + il only):** every reader is given the dead cells' locations (erasures; ECP-style
  detection, Schechter et al. 2010): the VSA ignores them in its similarities, majority ignores them, BCH treats an
  all-erased or tied bit as an erasure, LDPC gives it zero log-likelihood ratio, T-vhdd ignores them.
Not rigged for either side: the matched rate equalizes the number of wrong cells; the interleaver is free for all; the
designers know the overall rate eps but not where damage falls (damage conditions use the matched i.i.d. design).

## Arms (fixed now)

VSA (B22/B25 bipolar store: the sign of the bundle; shard = a seeded hash of the key; at budget B, S = ceil(B n /
8,192) shards of d = floor(B n / S) <= 8,192 cells, i.e. load n / S pairs per vector; codes truncated to d components):
- **V-pub** B25's read as published: threshold 0.5 on the rms-scaled similarity at every j (the per-shard rms is a
  digital scalar, as in B25/B34, not counted).
- **V-Snt** B34's: first value by argmax, threshold 0.5 at j >= 1.
- **V-am** first value by argmax; a later value accepted if its best similarity >= c x the first value's (no stored
  scale); c in {0.5, 0.6, 0.7, 0.8, 0.9} chosen on dev per (condition, budget).
- **V-last** the store with each key's last value bound with a LAST code; clean-up over 2 N_values candidates
  (value, value x LAST); the read stops when the LAST variant wins. No threshold: the VSA analogue of the tables' last
  flag (added by us, the strongest VSA read we could design).
- **V-cb** V-last with its value codebook on the same cells (counted); pairs per vector m in {1, 2, 4, 8, 16, 32,
  64} chosen on dev; d = floor(B n / (S + N_values)).
Best VSA = the smallest N* among V-pub, V-Snt, V-am, V-last (all with digitally regenerated codes).

Tables (all directory bits on the cells; one row per stored (key, j)):
- **T-rep** B34's T-v: a minimal perfect hash (hash and displace, lambda = 5; displacement array on the cells) and rows
  (last flag, value index); every logical bit repeated r = floor(B n / logical bits) times, majority.
- **E-BCH** T-rep's logical bits (displacements, then rows) cut into shortened binary narrow-sense BCH codewords
  (n = 63, 127, 255, 511, 1023; any t with k >= 2; our own codec - generator polynomial from the cyclotomic cosets,
  systematic encoder, syndromes / Berlekamp-Massey / Chien search - over the same primitive field polynomials as the
  galois library's BCH codes, so the generator polynomials and encodings are identical and the decoders can be
  compared word by word); each coded bit repeated rho >= 1 times (r or r + 1 copies, spread evenly, using the whole
  budget); majority per coded bit, a tie is an erasure; errors-and-erasures bounded-distance decoding (both erasure
  fillings; a word with 2e + s <= 2t is corrected, as guaranteed; an undecodable word returns its received information
  bits). (n, t) per store, budget and condition by
  the closed form (exact for i.i.d. errors): the design minimizing the expected number of failed keys, over every
  stored key, at the condition's rate (ties: at the next harder rate; then the shorter code).
- **E-LDPC** the same logical bits in systematic IRA LDPC codewords (H = [H1 | dual-diagonal], info columns of weight
  dv placed on the lightest rows; <= 1,024 information bits per codeword; rate R in {1/4, 1/3, 1/2, 2/3, 3/4, 5/6,
  9/10}; dv = 6, 6, 5, 6, 5, 4, 5 chosen on synthetic BSC codewords, record below); each coded bit repeated rho >= 1
  times; the copies' votes combined into a channel log-likelihood ratio at the condition's rate; sum-product belief
  propagation (ldpc package, flooding, 60 iterations). R chosen on dev per (condition, budget). Soft decoding of the
  repetition is the competent design at high noise, where bounded-distance codes cannot work (t/n < 1/4).
  (t = 1 designs are the Hamming codes the referee names; they are in the design space and lose to larger t here.)
- **E-CAM** B34's T-iix rows (an h-bit SHA-256 hash of (key, j), h the smallest collision-free h >= 2 log2 n; last
  flag; value index) in BCH codewords as E-BCH; every row decoded, exact match of the decoded hash (nearest Hamming if
  none).
- **T-vhdd** (HD table, B34's boundary case): MPHF displacements as random HD codes in two blocks, value rows as
  random HD codes (times HD(last) on a key's last value), decoded by similarity; share s in {0.1, ..., 0.5} of the
  cells to the displacement codes, chosen on dev per (condition, budget).
Best table = the smallest N* among T-rep, E-BCH, E-LDPC, E-CAM (non-HD). T-vhdd is reported separately (H2).
Reference: a directory-free table at the binary symmetric channel's capacity, (log2 N_values + last flag) /
(1 - H2(eps)) cells per pair (max over stores), a bound no finite code reaches.

## Measures (fixed now)

- **M1 (primary): N*(arm, condition)** = the smallest grid budget at which >= 99.5% of the probed keys are read
  exactly, at that budget and at every larger budget evaluated for that arm and condition. Designs per (condition,
  budget) from dev (rules above). Test budgets: dev N* of the matched i.i.d. condition +- 4 grid steps, plus B = 512;
  if dev never reached 99.5% by 4,096, the top 5 grid budgets; extension (b38_design.py extend, repeated by
  b38_box.sh extend-loop) by 4 more steps in the needed direction while the test N* is not bracketed (an arm that
  never reaches 99.5% is followed up to 4,096). This removes the over-provisioning objection: every memory is
  read at the smallest budget where it is still exact.
- **M2: equal cells.** Share read exactly at B = 512 (B34's budget) and at B = the best VSA's own N*, for every arm.
- **M2b: equal cells across budgets (i.i.d.).** Every arm at every grid budget from 8 steps below the smallest dev N*
  to 4 steps above the largest (VSA reads, non-HD tables, T-vhdd), so the memories can be compared at equal cells
  where they are under-provisioned (budgets below the best table's N*).
- **M3: graceful degradation at equal cells, fixed designs.** B_g = dev N* of the best VSA read at k8; every arm at
  B_g with its dev design for k8 (V-am's c, V-cb's m, T-vhdd's s, E-LDPC's R; E-BCH/E-CAM designed for p(k8)), read at
  i.i.d. eps in {p(k8), 0.16, 0.20, 0.24, p(k16), 0.32, p(k32)} and at lines / contiguous damage in the natural layout
  with f = 2 eps. The code parameters stay fixed; E-LDPC's channel log-likelihood ratio uses the condition's actual
  cell-error rate (a decoder can estimate it, e.g. from disagreeing copies), which favours the table.

## Bars (fixed now)

Delta(c) = grid steps of N*(best table) minus grid steps of N*(best VSA) (positive: the VSA needs fewer cells; a
budget never reached counts as one step beyond 4,096). Per condition: **"VSA fewer cells"** if Delta >= +2 (the table
needs >= 41% more), **"table fewer cells"** if Delta <= -2, **"tie"** if |Delta| <= 1.
- **H1 (primary)**, i.i.d. k1 ... k32: **VSA WIN** if "VSA fewer cells" at >= 1 k and "table fewer cells" at none;
  **TABLE WIN** the reverse; **MIXED** if both occur; **TIE** if every k is a tie. e05 is reported, not in the label.
- **H1cb**: the same with V-cb (codebook counted) in place of the best VSA.
- **H2**: best VSA vs T-vhdd (does a table with HD codes and no superposition need fewer cells?).
- **H3**: localized damage, interleaved, unknown locations (12 conditions): the same rule. **H4**: natural layout.
  **H5**: known locations.
- **Position independence** per arm: |steps N*(damage) - steps N*(matched i.i.d.)| <= 1, counted over the damage
  conditions per layout.
- **H6 (graceful degradation, M3)**: at each eps beyond k8, best VSA share - best table share at B_g: "VSA ahead"
  >= +5 points, "table ahead" <= -5, "tie" within 2.
- **H7 (under-provisioned, M2b)**: at each budget below the best table's N* (down to 8 steps below), best VSA share -
  best table share: "VSA reads more" >= +5 points, "table reads more" <= -5; counted over k1 ... k32 and e05.
- Store-cluster bootstrap (2,000 resamples) of Delta at k4, k8, k16, k32 (reported, not a bar).

## How the paper should read each outcome (fixed now)

- **H1 TABLE WIN:** "Counting every stored bit and reading each memory at the smallest budget where it is still exact,
  error-corrected tables need fewer cells per fact than the VSA store at every noise level where the two differ" -
  with the factor per level and the code that achieves it. The paper may not claim a storage or robustness advantage for
  the vector store against coded tables; B34's equal-device sentence is superseded.
- **H1 MIXED / VSA WIN:** name the noise levels where the VSA needs fewer cells (Delta >= +2) and the code it beats
  there; if only at the highest noise, say that the advantage is where every memory needs hundreds of cells per fact,
  and that it belongs to soft (similarity) decoding of random codes, which the HD table also uses (H2).
- **H1cb:** state the cells per fact with the codebook on the devices; if the codebook dominates, say so with the
  number (it is N_values x d, about one vector per fact).
- **H2:** if T-vhdd needs no more cells than the best VSA read (Delta <= +1), the vector store's remaining edge over
  coded tables is its random high-dimensional code with similarity decoding, not superposition; if the VSA needs >= 2
  steps fewer, superposition itself saves cells there.
- **H3/H4/H5:** if the interleaved tables are position-independent (within one step of their i.i.d. N*) and H3 is a
  TABLE WIN or TIE, the paper may not cite graceful degradation under localized damage as a VSA advantage over
  interleaved ECC. If in the natural layout the VSA is position-independent and the tables are not (H4 VSA WIN), the
  paper may say: "stored without an interleaver, the VSA store tolerates localized damage that cripples a table's
  codewords, because superposition spreads every fact over a whole vector; an interleaver gives the table the same
  property at no cost in cells." Contiguous loss is reported separately (it removes whole vectors in the natural
  layout).
- **H6:** report which side degrades more gracefully beyond its design point at equal cells, and by how much; if the
  VSA is ahead beyond the point where both fall below 99.5%, the paper may say "degrades more gracefully", with the
  numbers, not "is more robust".
- **H7:** if the VSA reads more keys than every coded table at budgets too small for either to be exact, the paper may
  say "when under-provisioned, the VSA store fails gradually (fewer facts lost at the same cells), while coded tables
  fail in whole codewords" - and must add that this is a regime of inexact memory (below 99.5%), and whether T-vhdd
  (no superposition) shows the same.

## Predictions (informed by the dev runs; checked mechanically by b38_verdict.py, reporting only)

At 30 days; shares of the 4,776 probed keys; N* in cells per stored pair (grid steps of 19%). From the dev record
below (synthetic stores, where the small stores hold 3% of the probed keys against 6% on the test set), with margins.
- **P1 (H1)** TABLE WIN: Delta <= -3 at every k in k1 ... k32 (dev -8, -8, -6, -5, -4, -5); the best table is E-BCH or
  T-rep at k1/k2, E-BCH or E-LDPC at k4, E-LDPC at k8, k16, k32; the best VSA read is V-last or V-Snt at every k.
- **P2** N* within +-2 grid steps of dev for V-last, E-LDPC, E-BCH and T-vhdd at every i.i.d. k (dev V-last 76 / 76 / 76
  / 128 / 431 / 1722, E-LDPC 23 / 23 / 27 / 54 / 215 / 724, E-BCH 19 / 19 / 27 / 108 / 431 / 1722, T-vhdd 32 / 32 / 38 /
  91 / 256 / 1024 at k1 / k2 / k4 / k8 / k16 / k32); the VSA at k1 (no cell errors) needs >= 3x the best table's cells.
- **P3 (H1cb)** TABLE WIN; V-cb's N* >= 8 grid steps (>= 4x) above the best table's at every k, and V-cb does not
  reach 99.5% by 4,096 at k16 and k32.
- **P4 (H2)** TABLE WIN: T-vhdd needs fewer cells than the best VSA read at every k (Delta <= -1 everywhere, <= -2 at
  >= 4 of the 6 k; dev -5, -5, -4, -2, -3, -3).
- **P5 (H3)** interleaved damage is a TABLE WIN (every geometry, every f); V-last, E-LDPC, E-BCH, T-rep and T-vhdd are
  position-independent (within one step of their matched i.i.d. N*) in >= 10 of the 12 interleaved conditions each.
- **P6 (H4)** natural layout is a VSA WIN for lines and clusters: V-last within 2 steps of its matched i.i.d. N* in
  lines-nat and clust-nat at k4, e05 and k8, while every non-HD table and T-vhdd needs >= 4 steps more than its
  matched i.i.d. N* (or never reaches 99.5%) in every lines-nat and clust-nat condition from e05 up.
- **P7** contiguous loss in the natural layout: no arm reaches 99.5% by 4,096 at any f (each loses about the lost
  fraction of its facts); at B = 512 V-last reads within 3 points of 1 - f (dev 95.2 / 88.6 / 73.9 / 46.3% at f = 4.2
  / 10 / 24.5 / 55%) and at least as many keys as T-rep, E-BCH, E-LDPC and T-vhdd (E-CAM, whose rows are short,
  can be ahead).
- **P8 (H5)** known locations: TABLE WIN.
- **P9 (H6, M3 at B_g = 128)** an E-LDPC table designed for k8 stays ahead of every VSA read (>= +5) at eps = 0.20 and
  0.24, and falls behind (VSA ahead >= +20 points) at p(k16) = 0.276 and 0.32 (the code's threshold cliff); in the
  natural-layout lines condition the VSA is ahead at every eps; H6 counts VSA ahead at >= 8 of 20 conditions.
- **P10 (H7)** under-provisioned budgets (below the best table's N*): the VSA reads more keys (>= +5 points) at >= 2/3
  of the budgets, the table at <= 1/4 (dev 38 / 9 / 2 of 49).
- **P11 (checks)** closed form: at most 10 flagged cells of the closed-form arms; BCH: generator polynomials and
  encodings identical to galois, decoders agree on >= 99.9% of words and the bounded-distance shortcut is correct on
  every word it covers; LDPC: own decoder agrees with the ldpc package on >= 99% of codewords' information bits; device
  model vs i.i.d. flips: no flagged cell.

## Checks from other angles (fixed now)

- **Closed form vs simulation (b38_check.py closed):** for every simulated i.i.d. test cell of T-rep (exact
  binomial), E-BCH and E-CAM (exact errors-and-erasures bound per codeword; a failed codeword counted as a failed key,
  so the closed form is a lower bound up to sampling error) and V-pub, V-Snt, V-last (Gaussian approximation of the
  clean-up similarities, as B34): flag a cell if |simulation - closed form| > max(1 point, 4 SE); flags reported.
- **Second decoder implementations (b38_check.py decoders):** the galois library's BCH (generator polynomial,
  encoder, decoder) against our codec on every codeword of re-simulated E-BCH test cells near N*, including whether
  the bounded-distance shortcut (2e + s <= 2t -> corrected) holds for the library decoder; an own numpy sum-product
  LDPC decoder against the ldpc package (the run's decoder) on the same channel inputs. Agreement rates reported.
- **Device model (b38_check.py device):** a subset of test cells (the 10 131072/niah_multikey_2 stores at B = 256, k8
  and k16; V-Snt, V-last, T-rep, T-vhdd, E-BCH, E-LDPC, E-CAM, V-cb) re-read with B25's device model itself at k and
  30 days instead of i.i.d. flips; shares must agree within max(1 point, 4 SE).
- **Hand reading (b38_check.py handread):** at least three failed keys per primary arm at its N* minus one step at k8
  and k16 and in the natural-layout lines condition, with the cause (codeword errors vs t; similarity of the true value
  vs the winner; damage on the key's vector).
- **Reproducibility:** dev and test run on the A800 box's CPUs (Python 3.9, numpy 1.26.4, ldpc 2.3.9); the self-test
  ran on the box and locally (Python 3.10, numpy 2.2.6, ldpc 2.4.1) with identical counts in every printed cell.

## Compute

CPU only, on the A800 box (128 cores; no GPU) and locally. No paid API.

## Files

Code: b38_eq.py, b38_design.py, b38_check.py, b38_verdict.py. Data: b34_out/dev_stores.json.gz,
b34_out/test_stores.json.gz. Results: results/b38_dev.json (dev), results/b38_choice.json, results/b38_plan*.json,
results/b38_test*.json, results/b38_check_*.json/txt, results/b38_handread.txt, results/b38_verdict.txt,
results/b38_literature.txt.

## Dev record (before any test run; synthetic stores and synthetic codewords only)

- **Self-tests** (b38_eq.py selftest; synthetic pairs): every arm reads every probed key exactly on noiseless cells
  (V-cb at 4,096 cells per pair with load 4; V-am with c >= 0.8 rejects some true later values even without noise);
  damage fractions and i.i.d. rates as specified; identical counts on the box (numpy 1.26.4, ldpc 2.3.9) and locally
  (numpy 2.2.6, ldpc 2.4.1).
- **LDPC column weights** (synthetic all-zero codewords on a BSC, 1,024 information bits, 40 frames per point; no
  store): frame error rates at 0.5 / 0.6 / 0.7 / 0.8 x the capacity crossover for dv in {3, 4, 5, 6, 8}; chosen dv =
  6, 6, 5, 6, 5, 4, 5 for R = 1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 9/10. High-rate IRA codes of this length have error floors
  (FER 0.03-0.3 at half the capacity crossover for R >= 3/4); BCH covers that regime.
- **BCH codec vs galois** (b38_bch_codec_check.py, results/b38_bch_codec_check.txt; 17 random designs from n = 63 to
  1023): generator polynomials and systematic encodings identical in 17 / 17, decoding of 30 words each with 0 ... 2t + 2
  random errors identical in 510 / 510. (A first version used other primitive field polynomials than galois's BCH
  codes - equivalent codes, different generators; the E-BCH/E-CAM dev runs were redone with the matched fields.)
- **Changes during dev (no test data):** the BCH design rule first minimized expected failures of the probed keys; it
  now uses every stored key (the probe set is a sample); V-cb loads 1 and 2 were added after loads 4-64 left the
  choice at the grid edge (4); a budget window common to all arms (M2b) and the iterative extension were added after
  the first full dev run showed natural-layout tables failing up to the largest evaluated budget; E-BCH / E-CAM jobs
  are split finer (decoding of failing words dominates their time).
- **Dev grid** (results/b38_dev.json.gz: 12 dev stores, 1,254 probed keys, every design x 37 budgets x 7 i.i.d.
  conditions; choice in results/b38_choice.json). Dev N* (k1 / k2 / k4 / e05 / k8 / k16 / k32): V-pub 108 / 108 / 128 /
  152 / 861 / - / -; V-Snt 108 / 108 / 91 / 108 / 128 / 431 / 3444; V-am 91 / 91 / 91 / 108 / 152 / 512 / 2048 (c = 0.9,
  the top of its grid, at every N*); V-last 76 / 76 / 76 / 91 / 128 / 431 / 1722; V-cb 431 / 431 / 512 / 609 / 1448 / - /
  - (load 1-2); T-rep 19 / 19 / 91 / 152 / 304 / 1024 / 3444; T-vhdd 32 / 32 / 38 / 54 / 91 / 256 / 1024 (s = 0.3);
  E-BCH 19 / 19 / 27 / 38 / 108 / 431 / 1722; E-LDPC 23 / 23 / 27 / 38 / 54 / 215 / 724 (R = 5/6, 5/6, 2/3, 1/2, 1/3,
  1/3, 1/4); E-CAM 45 / 45 / 64 / 91 / 256 / 1218 / 4096. B_g = 128 (V-last / V-Snt at k8).
- **Full dev run with the registered plan** (all conditions, windows, M2b, M3, extension until bracketed; results in
  results/b38_devfull.json.gz, verdict in results/b38_dev_verdict.txt): H1 TABLE WIN (Delta -8 / -8 / -6 / -5 / -5 / -4
  / -5; bootstrap CI k8 [-5, -5], k16 [-6, -3]); H1cb TABLE WIN; H2 TABLE WIN; H3 TABLE WIN (interleaved damage behaves
  like i.i.d. errors for every arm); H4 VSA WIN (natural layout: V-last 91 / 91 / 152 / 609 for lines at k4 / e05 / k8 /
  k16 and 76 / 108 / 181 / 724 for clusters, while T-rep, E-BCH, E-LDPC, E-CAM and T-vhdd need 512 to > 4096; the
  contiguous loss defeats every arm); H5 TABLE WIN; H6: at B_g = 128 with k8 designs E-LDPC reads 100 / 100 / 97.0% at
  eps 0.16 / 0.20 / 0.24 and then 15.8 / 4.6% at 0.276 / 0.32, against V-Snt 97.9 / 94.4 / 84.2 / 67.4 / 41.2%
  (T-vhdd 69.1% at 0.276); H7: VSA reads more at 38 of 49 under-provisioned budgets (e.g. k16 at 128 cells: +52 points).
- Not yet run on the test stores: anything.
