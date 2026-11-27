# B54 pre-registration: multi-level cells (2 and 3 bits per cell) for the VSA store and the error-corrected tables; cells per stored fact, absolute and relative to each memory's own noise-free requirement

Registered 2026-11-20 (US Eastern), before any test run. The code (b54_mlc.py, b54_check.py, b54_verdict.py) was
written and self-tested on synthetic pairs, the cell channel was calibrated from B25's device model (no store), and
every free design parameter was chosen on B34's 12 synthetic dev stores (record below). The 50 RULER test stores
(b34_out/test_stores.json.gz; md5 6c83632c0492fec3b3ee7a7f17f1808b, identical on the box) have not been read by any
B54 arm.

**Scope (reduced by the coordinator on 2026-11-20, before any run):** only the multi-level-cell (MLC) arm for the
store and the tables, and the overhead of every memory relative to its own noise-free requirement. The HD-native
protected variant of the store (sparse block code, error-correcting item codebook or iterative clean-up) asked for by
the referee is **not run**; it is a stated limitation.

## Why

Referee (9th read), M2b-d on B38 (equal cells): (b) the tables were searched over code length, rate, repetition and
interleaving, the store only over four read rules and its load; (c) only binary cells were compared, while in-memory HD
claims rest on analog / multi-level conductances, and sign quantization of a superposition discards SNR, so a
multi-level-cell arm is needed for both the store and the tables; (d) robustness and storage efficiency are
conflated: normalized by each memory's own noise-free requirement, the store degrades least (B38, k8 / k16 overhead:
store 1.7x / 5.7x, LDPC 2.3x / 9.3x, random-code table 2.8x / 9.5x, BCH 5.7x / 22.7x). B54 adds the MLC arm and
reports both measures.

## Setting (fixed now; as B38 unless stated)

- **Stores, keys, metric, budget.** B38's: 50 RULER stores, 4,776 probed keys, a key is read exactly when all its
  values come back in order with nothing extra; N* = the smallest grid budget (cells per stored (key, j) pair, every
  stored bit on the cells) with >= 99.5% of probed keys exact there and at every larger evaluated budget. Grid
  extended downwards: B = 2^(i/4), i = 4 ... 48 (2 ... 4,096 cells per pair), because MLC tables need fewer than 8.
- **Cell.** B38's differential pair of PCM devices (B25's device model, Joshi et al. 2020: programming noise, per-device
  drift exponent, 1/f read noise, 8-bit ADC over +-G_ON; every noise source x k). With b bits per cell the
  differential target takes L = 2^b equally spaced levels in [-G_ON, +G_ON] (positive levels on the + device with
  the - device RESET, and vice versa; b = 1 is B38's cell). Times: 30 days and 1 day after programming.
- **Channel.** For every (b, k, time) the read value of 2^21 cells per level is simulated once from the device model
  (b54_mlc.py calib, results b54_out/b54_channel.npz); arms' cells are then read i.i.d. by inverse-CDF sampling from
  these per-level distributions (one uniform per cell shared across k and time; quantiles on a logit grid to
  5e-7). The reader is **calibrated on reference cells** (as MLC chips do): the decision threshold between adjacent
  levels minimizes their pairwise error (from the per-level distributions); the analog read is mapped through the
  per-level mean read values. This calibration is ideal (it favours MLC). At b = 1 the channel reproduces B34's
  bit-error rates (30 d: k2 1.31e-4, k4 0.0207, k8 0.1226, k16 0.276, k32 0.374; 1 d: k4 0.0098, k8 0.1017).
- **Packed mode (every table; and the VSA store's sign vector).** The arm's bits (bit budget b x cells, built by B38's
  own builders) are spread over the cells by a seeded random interleaver (free for every arm), b bits per cell,
  Gray-labelled levels; the read decides a level and returns its Gray bits. E-LDPC, and the packed VSA read, also get the
  exact a-posteriori log-likelihood ratio of every bit given the decided level (from the calibrated confusion matrix;
  E-LDPC sums it over copies; the VSA uses tanh(LLR / 2) as the component). T-rep, E-BCH and T-vhdd read hard bits
  (as in B38). Table contents are not whitened (their skewed bit statistics put more cells on the outer levels, which
  lowers their raw bit-error rate slightly; this favours the tables).
- **Analog mode (VSA store only).** One component of the integer bundle per cell (B38's V-last store: the key's last
  value bound with a LAST code), uniformly quantized to L levels with the outermost level at c x the shard's rms
  (exact zeros get a random half step); read as an analog value (soft), clean-up by similarity as V-last (no
  threshold). This is the multi-level store the referee asks for (sign quantization replaced by b-bit quantization).
- **Noise-free requirement.** For an MLC memory: N* on ideal cells (levels read exactly; condition m{b}-ideal). For a
  binary memory: B38's N* at k1 (bit-error rate exactly 0).

## Arms and tuning grids (fixed now)

| memory | b = 1 (binary) | b = 2, 3 (MLC, this run) | grid per (b, condition, budget) |
|---|---|---|---|
| VSA store V-last | B38 / B52 (c) | packed, or analog with c in {1.0, 1.5, 2.0, 2.5, 3.0} | 6 designs |
| T-rep (MPHF table, repetition) | B38 / B52 (c) | packed | 1 |
| E-BCH (BCH + repetition) | B38 / B52 (c) | packed; (n, t) by B38's closed form at the condition's calibrated mean bit-error rate | closed form over 223 BCH designs (not tuned on dev) |
| E-LDPC (IRA LDPC + repetition) | B38 / B52 (c) | packed, LLR input | 7 rates |
| T-vhdd (random-code HD table) | B38 (30 d); 1 d run here | packed | 5 shares |

Bits per cell b in {1, 2, 3} is a further choice per memory and condition ("best over b"). In B38 the binary grids
were: VSA 8 read configurations (V-pub, V-Snt, V-am with 5 c, V-last; V-last alone reported here), T-rep 1, E-BCH 223
(closed form), E-LDPC 7, T-vhdd 5. Not run at MLC: V-pub, V-Snt, V-am (V-last was the best or tied-best VSA read at
every k in B38 and B52 (c)), V-cb, E-CAM (never the best table in B38), localized damage, M3 / M2b.

Conditions per b in {2, 3}: ideal; k1, k2, k4, k8, k16, k32 at 30 days and at 1 day (13). At b = 1: T-vhdd at the
six 1-day rates (B52 (c) did not run it).

## Dev / test discipline (fixed now)

Dev (B34's 12 synthetic dev stores): pass 1 every design at the octave budgets (2, 4, ..., 4,096) and every
condition; pass 2 every design at the quarter steps within 3 steps of each (arm, b, condition)'s pass-1 N* (top 4
budgets if not reached). Design per (arm, b, condition, budget) = highest dev share (ties: share at the next harder
condition, then grid order), B38's rule. Test: every (arm, b, condition) at dev N* +- 4 grid steps (top 5 budgets if
dev never reached 99.5%), each budget with its dev design (nearest dev budget's design where dev did not evaluate it);
then extension by 4 steps in the needed direction (b54_mlc.py extend) until the test N* is bracketed or the grid
edge is reached. Each test cell is run once.

## Measures and bars (fixed now)

- **M1** N*(memory, b, condition), cells per stored pair (Tables 1a/1b).
- **M2** overhead N*(k) / N*(noise-free, same b) (Tables 2a/2b) - the referee's normalization.
- **M3** best over b per memory (Tables 3a/3b), with its overhead relative to the memory's best noise-free requirement.
- **H1-MLC** (primary), per time: Delta = grid steps of the best table's N* (best of T-rep, E-BCH, E-LDPC) minus the
  VSA store's, at b = 2, at b = 3, and with every memory at its best b; per k "VSA fewer cells" if Delta >= +2,
  "table fewer cells" if <= -2, "tie" within 1; label VSA WIN / TABLE WIN / MIXED / TIE as B38's H1 over k1 ... k32.
- **H2-MLC** per memory: does MLC (best of b = 2, 3) need >= 2 steps fewer cells than binary at k (counted over k and
  times)?
- **H4** (robustness vs efficiency): at k8 and k16, per time, per b and at the best b: is the VSA store's overhead
  the smallest of the five memories?

## How the paper should read each outcome (fixed now)

- **H1-MLC TABLE WIN (at the best b):** "With multi-level cells available to every memory, error-corrected tables still
  need fewer cells per fact than the vector store at every noise level" - with the factor and the b. B38's binary
  sentence stands; MLC does not rescue a storage advantage.
- **H1-MLC MIXED / VSA WIN:** name the k (and b) where the store needs fewer cells.
- **H4 (the store's overhead smallest):** the paper may say "relative to its own noise-free requirement, the vector
  store's cell count grows least with device noise (x at k8, y at k16, against ... for the tables)", and must add in
  the same sentence that in absolute cells it needs more than the coded tables at every k (if Table 1 shows that):
  graceful growth from a higher base, not fewer cells. If H4 fails at MLC, say for which b.
- **MLC benefit:** if MLC lowers a memory's N* only at low noise (k1, k2) and raises it from k4 up, say so: "extra
  levels pay only while the device noise is well below the level spacing".

## Predictions (informed by the dev runs; checked against the verdict, reporting only)

On the test stores (binary rows from B38 / B52 (c)); one grid step = 19%.
- **P1 (H1-MLC, every memory at its best b)** TABLE WIN at 30 days and at 1 day, Delta <= -4 at every k (dev, 30 d:
  VSA 27 (b3) / 38 (b2) / 76 / 128 (b1) / 431 (b1) / 1722 (b1) against the best table 10 / 13 / 27 / 54 / 215 / 724
  at k1 / k2 / k4 / k8 / k16 / k32).
- **P2 (H1-MLC at b = 2 and at b = 3 separately)** TABLE WIN at both times, Delta <= -3 at every k where the best table
  reaches 99.5% by 4,096.
- **P3 (where MLC pays)** for every memory, the best of b = 2, 3 needs >= 2 steps fewer cells than binary at k1 (both
  times); at k8, k16, k32 (30 d) no memory's MLC N* is >= 2 steps below its binary N*.
- **P4 (H4)** the VSA store's overhead N*(k) / N*(noise-free) is the smallest of the five memories at k8 and k16, at
  b = 1, 2, 3 and at the best b, at both times (16 of 16 cases; dev 16 / 16, e.g. 30 d k8 b = 2: store 4.0x, E-LDPC
  6.9x, T-vhdd 8.0x, E-BCH 21.5x, T-rep 61x).
- **P5** at k1 (30 d) the VSA store at b = 3 needs >= 4 steps (>= 2x) fewer cells than at b = 1 (dev 27 vs 76), and the
  packed design is chosen there (b = 2 and 3).
- **P6 (stability)** test N* within +-2 grid steps of dev N* for V-last, E-LDPC, E-BCH and T-vhdd at every MLC
  condition where dev reached 99.5%.
- **P7 (checks)** device-model re-read: no flagged cell; second demapper and second scorer agree on every bit / key;
  0 N* mismatches; closed form: every E-BCH flag has the simulation above the closed form, and at most 10% of T-rep
  cells are flagged.

## Checks from other angles (fixed now)

- **Closed form vs simulation** (b54_check.py closed): T-rep (binomial majority) and E-BCH (errors-and-erasures bound;
  a lower bound for a bounded-distance code) at the calibrated mean bit-error rate, every simulated MLC test cell within
  +-2 steps of its test N*; flag if |sim - closed form| > max(1 point, 4 SE). The closed form assumes uniform levels and
  i.i.d. bits, so flags with the simulation above the closed form are expected (skewed table contents, B38's lower bound).
- **Device model on the actual cells** (b54_check.py device): the 10 131072/niah_multikey_2 test stores, every arm,
  b = 2, 3, k1 / k2 / k4 at 30 days, at one grid step below each test N*, re-read with B25's device model on the arm's
  own cells (fresh devices) instead of the calibrated i.i.d. channel; shares must agree within max(1 point, 4 SE).
- **Second implementations** (b54_check.py second): a scalar second level demapper (bisect + explicit Gray table)
  against the vectorized one on 200,000 cells per condition (must agree on every bit); a second exactness scorer
  (decoded value strings against the store's own pairs, not Store38.want) on re-run test cells at m2-k4-30d (must agree
  on every key); N* recomputed from the per-store failed-key lists by a separately written rule (0 mismatches).
- **Hand reading** (b54_check.py handread): failed keys at one step below N* for the VSA store and the tables at
  b = 2 / 3, with the cause.

## Compute

CPU only: the A800 box (128 cores, nice 10, about 96 workers) and locally. No GPU, no paid API.

## Files

Code: b54_mlc.py, b54_check.py, b54_verdict.py, b54_box.sh. Channel: b54_out/b54_channel.npz (calib log
b54_out/calib.log). Results: results/b54_dev1.json.gz, results/b54_dev2.json.gz, results/b54_choice.json,
results/b54_test*.json.gz, results/b54_check_*.json/txt, results/b54_handread.txt, results/b54_verdict.txt.

## Dev record (before any test run; synthetic dev stores and synthetic pairs only)

- **Channel calibration** (b54_out/calib.log; no store). Mean bit-error rate per stored bit (uniform levels, Gray):
  30 d: b = 2: k1 1.7e-4, k2 0.028, k4 0.142, k8 0.268, k16 0.367, k32 0.452; b = 3: k1 0.031, k2 0.126, k4 0.247,
  k8 0.350, k16 0.422, k32 0.472. 1 d: b = 2: 4.4e-5 / 0.019 / 0.123 / 0.255 / 0.359 / 0.449; b = 3: 0.024 / 0.113 /
  0.232 / 0.342 / 0.417 / 0.469. b = 1 reproduces B34 (above). Self-test: on a synthetic store the packed channel's
  simulated bit-error rate matches the calibration (T-rep's skewed contents run slightly lower at high k, e.g. m2-k8-30d
  0.260 vs 0.268); the device model on the actual cells agrees with the calibrated channel (bit-error rate m2-k2-30d
  0.0263 vs 0.0272; exact keys 51 vs 50 / 150, V-last analog m2-k4-30d 138 vs 136 / 150).
- **Self-tests** (b54_mlc.py selftest): every arm reads every probed key exactly on ideal cells at b = 2 and 3, within
  its cell budget; packed round trip identical.
- **Bug found in dev (fixed before the test):** T-rep's design sentinel (None) was taken for "infeasible" in pass 1, so
  every T-rep cell was skipped; T-rep pass 1 was re-run with the sentinel "rep" (logs/dev1_trep.log) and merged
  (b54_dev1m.json) before pass 2. No other arm affected.
- **Dev grid** (pass 1: 1,836 jobs, every design x 12 octave budgets x 13 conditions x b = 2, 3, plus T-vhdd b = 1 at
  1 d; pass 2: 3,708 jobs, the quarter steps within 3 of each pass-1 N*). Dev N* (cells per stored pair):

| memory | b | ideal | k1 | k2 | k4 | k8 | k16 | k32 (30 d) | k1 | k2 | k4 | k8 | k16 | k32 (1 d) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V-last | 2 | 38 | 38 | 38 | 76 | 152 | 512 | 3444 | 38 | 38 | 64 | 128 | 431 | 1722 |
| V-last | 3 | 27 | 27 | 45 | 76 | 181 | 724 | 3444 | 27 | 38 | 64 | 128 | 512 | 1722 |
| T-rep | 2 | 10 | 10 | 54 | 152 | 609 | - | - | 10 | 45 | 152 | 431 | - | - |
| T-rep | 3 | 7 | 38 | 76 | 256 | - | - | - | 32 | 76 | 256 | - | - | - |
| E-BCH | 2 | 10 | 10 | 13 | 64 | 215 | - | - | 10 | 13 | 54 | 181 | - | - |
| E-BCH | 3 | 7 | 10 | 38 | 128 | - | - | - | 10 | 32 | 108 | 2896 | - | - |
| E-LDPC | 2 | 11 | 11 | 13 | 27 | 76 | 215 | 861 | 11 | 13 | 27 | 64 | 181 | 861 |
| E-LDPC | 3 | 7 | 10 | 16 | 32 | 91 | 256 | 1218 | 8 | 13 | 27 | 76 | 256 | 1024 |
| T-vhdd | 2 | 16 | 16 | 23 | 45 | 128 | 431 | 3444 | 16 | 23 | 45 | 128 | 362 | 2896 |
| T-vhdd | 3 | 10 | 16 | 32 | 76 | 215 | 1024 | - | 13 | 27 | 64 | 215 | 724 | - |

  T-vhdd b = 1 at 1 d: 32 / 32 / 38 / 76 / 256 / 861. ('-' = not reached by 4,096.) Binary dev (B38): V-last 76 / 76 /
  76 / 128 / 431 / 1722, E-LDPC 23 / 23 / 27 / 54 / 215 / 724, E-BCH 19 / 19 / 27 / 108 / 431 / 1722.
- **Designs chosen at dev N*:** V-last packed at every condition up to k8 at b = 2 (analog c = 1.0 at k32) and at b = 3
  up to k2 (analog c = 1.0 at k4-30d, k16, k32 and at k4 / k8 1 d): packing b sign components per cell beats one
  b-bit component of the bundle wherever the noise is low; the analog store is chosen only where every MLC store is
  worse than the binary one. c = 1.0 is the lower edge of the c grid (smaller c tends to the binary sign store, which is
  in the comparison as b = 1). E-LDPC rate 5/6 -> 1/4 as noise grows; T-vhdd share 0.3 almost everywhere.
- Test plan (b54_plan_test.json): 477 (budget, design) evaluations, 1,127 (budget, condition) cells per store.
- Not yet run on the test stores: anything.
