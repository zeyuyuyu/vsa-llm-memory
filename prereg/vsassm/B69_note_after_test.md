# B69 note after the test (test run once, no cut; results/b69_partA.{txt,json}, results/b69_verdictB.{txt,json}, results/b69_check_*.json, results/b69_check_recount.txt)

## What was run

Pre-registered (4449bc4, 15:07 machine clock; OpenTimestamps proof 41bf357; check code e587f61 at 15:10), run as
registered. Part (A): a computation on the registered per-store failed-key counts of B38 / B59 / B61 / B63 (local,
15:10-15:11). Part (B): synthetic stores of 6,002 pairs (B34's mk2big2 generator, 7-digit values) with vocabularies of
N = 256, 1,024, 6,000 and 24,000 values, dev on 3 stores per N (box, 15:17-15:30, one extension round), choices
committed (21644f4), test once on 8 stores per N (2,400 probed keys per N; box, 15:31-15:39, 8,056 store evaluations),
B38's extension rule added nothing, no cut. Deviations, all disclosed in compute notes 1-4 and none touching a rule:
stores generated on the box from exported word lists (content MD5-identical), the verdict's labels restricted to the
registered k1 / k8 / k16 before any output existed, and two malformed identifiers in the workload that make the
registered compact format fall back to utf8 on two niah_multikey_3 stores.

Accountings. Uncharged: the value vocabulary is the shared output dictionary, free for every memory (as registered).
Charged: every store also holds its vocabulary strings on the same cells (the memory keeps its own requirement, the
strings go in the remaining cells of every store), protected at the channel-capacity bound ('cap', a lower bracket no
code reaches) or by the tables' own repetition / BCH codes ('bch', closed form, P(any string wrong) <= 1%; an upper
bracket at k8 and above, where the tables' LDPC would be cheaper). Format utf8 (primary; 56 bits per 7-digit value,
288 per identifier).

## Part (A): both accountings on the registered runs

Cells per stored pair for >= 99.5% exact, k1 / k2 / k4 / e05 / k8 / k16 / k32 (30-day i.i.d. cell errors).

Pooled over the 50 stores (4,776 keys); the charge is set by the 288 bits per pair of the niah_multikey_3 identifiers:

| memory | uncharged | charged, cap | charged, bch |
|---|---|---|---|
| dense store U | 76 / 76 / 76 / 91 / 128 / 431 / 1,722 | 431 / 431 / 431 / 512 / 861 / 2,435 / >4,096 | 431 / 431 / 609 / 1,024 / 2,435 / >4,096 / >4,096 |
| sparse store H-SBC | 32 / 32 / 45 / 54 / 91 / 304 / 1,448 | 362 / 362 / 431 / 512 / 724 / 2,435 / >4,096 | 362 / 362 / 609 / 861 / 2,435 / >4,096 / >4,096 |
| best static table | 19 / 19 / 27 / 38 / 54 / 215 / 724 | 362 / 362 / 431 / 512 / 724 / 2,435 / >4,096 | 362 / 362 / 512 / 861 / 2,435 / >4,096 / >4,096 |
| best insertable table | 32 / 32 / 38 / 64 / 91 / 304 / 1,218 | 362 / 362 / 431 / 512 / 724 / 2,435 / >4,096 | 362 / 362 / 512 / 1,024 / 2,435 / >4,096 / >4,096 |

Uncharged (the registered numbers, reproduced 28 of 28): static vs dense TABLE WIN (-8 to -4 grid steps), static vs
sparse TABLE WIN (-4 to -2), insertable vs sparse TIE, insertable vs dense TABLE WIN. Charged, pooled: every comparison
is a TIE at every noise level under cap and under bch (|Delta| <= 1 step; the dense store's charged requirement is one
step above the tables' at k1, 431 against 362).

niah_multikey_2 (20 stores, 3,000 keys, 56 vocabulary bits per pair; the family that resembles the sweep):

| memory | uncharged | charged, cap | charged, bch |
|---|---|---|---|
| U | 76 / 76 / 76 / 91 / 128 / 431 / 1,722 | 152 / 152 / 152 / 181 / 256 / 861 / 3,444 | 152 / 152 / 181 / 256 / 609 / 2,435 / >4,096 |
| H-SBC | 32 / 32 / 45 / 64 / 91 / 304 / 1,448 | 91 / 91 / 128 / 152 / 215 / 724 / 2,896 | 91 / 91 / 152 / 256 / 512 / 2,048 / >4,096 |
| static | 19 / 19 / 27 / 38 / 54 / 215 / 724 | 76 / 76 / 108 / 128 / 181 / 609 / 2,048 | 76 / 91 / 128 / 215 / 512 / 2,048 / >4,096 |
| insertable | 32 / 32 / 38 / 64 / 91 / 304 / 1,218 | 91 / 91 / 108 / 152 / 215 / 724 / 2,435 | 91 / 91 / 152 / 256 / 512 / 2,048 / >4,096 |

Charged on niah_multikey_2: static vs dense stays TABLE WIN (cap -4 / -4 / -2 / -2 / -2 / -2 / -3; bch -4 / -3 / -2 /
-1 / -1 / -1 / 0: the table needs 2.0 times fewer cells without errors and 1.4 times at four times the noise); static
vs sparse comes within one step (19%) at k1-k16 (cap: TABLE WIN only through k32's two steps; bch: TIE); insertable vs
sparse TIE. With the compact format (24 bits per value) static vs sparse is two steps at k1 and k2 again (cap and
bch). niah_multikey_3 (288 bits per pair): every comparison ties except static vs dense at k1 (-2: 304 against 431).
The 20 small stores (9-23 pairs): uncharged, the insertable table needs two to three steps more than the sparse store
at k1, k2, e05, k16 and k32, one at k4 and k8 (27 against 19 cells without errors: a cuckoo table of a few buckets); charged, all ties.

**x1 has no cell errors (check C0).** At k = 1 the device model flipped none of 2 x 10^8 simulated cells, read 30 days
and one year after programming (95% upper bound 1.5 x 10^-8 per cell); the smallest read margin |G+ - G-| was 12.7% of
the full conductance at 30 days (7.2% at one year), and the mean margin 9.6 standard deviations (8.7). The i.i.d.
channel at eps = 0 returns the cells unchanged, and every compared memory needs the same cells at k1 and at k2 (1.3 x
10^-4). So every x1 requirement is the memory's logical content: cross-talk for the stores (76 / 32 cells per pair),
rows and directory for the tables (19 static, 32 insertable).

## Part (B): vocabulary sweep (N = 256, 1,024, 6,000, 24,000 values; 6,002 pairs per store)

Uncharged test requirements, k1 / k4 / k8 / k16 (k4 secondary):

| memory | N = 256 | N = 1,024 | N = 6,000 | N = 24,000 |
|---|---|---|---|---|
| dense store U | 54 / 64 / 108 / 304 | 64 / 76 / 128 / 362 | 76 / 91 / 128 / 431 | 91 / 91 / 152 / 512 |
| sparse store H-SBC | 27 / 32 / 76 / 256 | 32 / 38 / 91 / 304 | 32 / 45 / 108 / 362 | 38 / 54 / 108 / 431 |
| best static table | 13 / 19 / 38 / 152 | 16 / 23 / 45 / 181 | 19 / 27 / 54 / 215 | 23 / 32 / 64 / 256 |
| best insertable table | 23 / 32 / 76 / 256 | 27 / 38 / 76 / 304 | 32 / 45 / 91 / 304 | 32 / 45 / 91 / 362 |

(static = T-rep at k1, E-BCH or E-LDPC at k4, E-LDPC at k8 and k16; insertable = O-BCH at k1 for N <= 1,024 and
24,000, O-LDPC otherwise.) Delta in grid steps (table minus store), uncharged:

| comparison | N = 256 | N = 1,024 | N = 6,000 | N = 24,000 | label at every N |
|---|---|---|---|---|---|
| static vs dense | -8 / -7 / -6 / -4 | -8 / -7 / -6 / -4 | -8 / -7 / -5 / -4 | -8 / -6 / -5 / -4 | TABLE WIN |
| static vs sparse | -4 / -3 / -4 / -3 | -4 / -3 / -4 / -3 | -3 / -3 / -4 / -3 | -3 / -3 / -3 / -3 | TABLE WIN |
| insertable vs sparse | -1 / 0 / 0 / 0 | -1 / 0 / -1 / 0 | 0 / 0 / -1 / -1 | -1 / -1 / -1 / -1 | TIE |
| insertable vs dense | -5 / -4 / -2 / -1 | -5 / -4 / -3 / -1 | -5 / -4 / -2 / -2 | -6 / -4 / -3 / -2 | TABLE WIN |

- **B-H1 holds for all four comparisons**: the uncharged label is the same at every N. As ratios: the dense store
  needs 4.0 times the static table's cells without errors at every N (2.4-2.8 at k8, 2.0 at k16), the sparse store
  1.7-2.0 times, and the insertable table ties the sparse store (within one step; at N = 24,000 it is one step below
  the sparse store at every noise level).
- **B-H2 (scaling): stable.** From N = 256 to 24,000 every memory's requirement rose by 1-3 grid steps (U +2 to +3,
  H-SBC +2 to +3, static +3, insertable +1 to +2), and no Delta moved by more than one step. Both needs grow
  logarithmically in N, the store's through the 2N candidates its clean-up must reject (ln(N / eps)), the table's
  through its ceil(log2 N)-bit index; over a 94-fold range of N the ratio between them barely changes.
- **B-H3 (charged, utf8 56 bits per value, 2.4 / 9.6 / 56 / 224 bits per pair).** cap: static vs dense TABLE WIN at
  every N but shrinking (k1 -8, -6, -4, -2: 4.0, 2.8, 2.0, 1.4 times); static vs sparse TABLE WIN up to 6,000 and a TIE
  at 24,000 (304 against 256 cells without errors); insertable vs sparse TIE at every N. bch: the same, except static vs
  sparse is already a TIE at N = 6,000 (-1 / -1 / 0 / 0). The smallest N with a tie: static vs sparse 24,000 (cap) or
  6,000 (bch); static vs dense none (it ties at k4-k16 from 24,000, but k1 stays two steps); insertable vs sparse
  every N.
- Reproduction: at N = 6,000 every memory's requirement is within one step of the workload's (Q2).
- Design notes: at k4 the LDPC tables' rate sat at 2/3, the inner edge of the restricted grid (compute note 3), so
  their k4 numbers may be one step high (against the tables); H-SBC's block load at k16 sat at the grid's lower edge
  (0.35), as in B61.

## Predictions

Q1 held (112 of 112 test requirements within two steps of the closed-form prediction: 75 on it, 37 one step off,
none two steps off). Q2 held (N = 6,000 within one step of the workload for every memory and condition; +1 for U at k4, H-SBC at k8
and k16, T-rep at k8, E-BCH at k16, O-LDPC at k4). **Q3 failed in part**: the four labels held at every N and static vs
dense stayed <= -4, but static vs sparse reached -4 (N = 256 and 1,024 at k1 and k8, N = 6,000 at k8), outside the predicted -1 to -3.
Q4 held (rises of +1 to +3 steps, every Delta within one step). **Q5 failed in part**: at N = 256 every charged label
equals the uncharged one (held); at N = 6,000 static vs dense stays "table fewer" at k1 (-4) and static vs sparse ties
under bch (held); but at N = 24,000 static vs dense (and insertable vs dense) is still two steps at k1 under both
protections, so not every comparison ties there. Q6 held (28 of 28 reproduced; pooled charged ties everywhere;
niah_multikey_2 static vs dense TABLE WIN under cap with <= -2 at k1-k8; static vs sparse a tie at k1; compact TABLE WIN
for static vs dense under both protections). Q7 held (checks below).

## Checks

- C0: above (0 flips of 2 x 10^8 cells at x1).
- C1 bits audit (results/b69_check_audit.json): 13,073 dev and test evaluations, 0 over budget; every sweep store has
  Nv = N, 300 probed keys and 6,002 pairs; cuckoo logical bits = m x 4 x (f + 1 + ceil(log2 N)) in every O- evaluation;
  logical bits per pair of the static tables 12.0-12.4 / 14.0-14.4 / 17.0-17.4 / 19.2-19.4 at the four N (the theory
  file's 12.4 / 14.2 / 17.2 / 19.2) and of the cuckoo tables 22.1-33.7; vocabulary bits recomputed by direct character
  counts on every sweep and workload store, 0 differences.
- C2 independent recount (results/b69_check_recount.{json,txt}; own design lookup, own N* loops, own capacity formula,
  own math.comb tail sums re-verifying every bch vocabulary design): (B) 112 uncharged and 448 charged requirements, 0
  mismatches; (A) 1,120 charged requirements, 0 mismatches.
- C3 sample recount: 8 seeded requirements listed with pooled failures per budget (e.g. H-SBC N = 6,000 k4: 143 / 37 /
  12 / 0 / 0 failed of 2,400 at 32 / 38 / 45 / 54 / 64 cells; N* 45).
- C4 cross-machine (results/b69_check_cross.json): 6 seeded test jobs (E-LDPC, O-LDPC, two O-BCH, two U; N = 256,
  1,024, 6,000) re-run locally (numpy 2.2 / ldpc 2.4 against the box's): failed-key lists identical in 61 of 61
  (budget, condition) cells.
- The dev choice and the test plan were reproduced byte for byte locally from the fetched dev files.

## Reading against the literature (from memory, not re-searched in this session)

The logarithmic growth of a superposition store's dimension with the number of candidates it must reject at a fixed
error rate is the standard capacity result for VSAs (Frady, Kleyko and Sommer, Neural Computation 2018; Thomas,
Dasgupta and Rosing, JAIR 2021; Clarkson, Ubaru and Yang, arXiv 2301.10352), and the sparse store's 1.44 log2(1 / eps)
Bloom-type cost was matched in B61. A table's index cost of log2 N bits is elementary. Supports; no conflicting result
known. The charged comparison has no published counterpart that we know of.

## What the paper's device claim may say (plain sentences)

- **Accountings.** "With the value vocabulary treated as a shared dictionary, free to every memory, coded tables with a
  static directory needed 2 to 4 times fewer cells than the dense store and 1.4 to 2 times fewer than the sparse-block
  store. If every memory must also hold its vocabulary strings on the same cells, the comparison depends on how many
  bits those strings add per stored pair: on all 50 stores, where 36-character identifiers add 288 bits per pair,
  every comparison becomes a tie; on the stores with 7-digit values (56 bits per pair) the static table still needs
  2 times fewer cells than the dense store without errors and 1.4 times at four times the noise, and about as many as
  the sparse-block store."
- **Across N.** "In a sweep of the vocabulary from 256 to 24,000 values (6,002 pairs per store), every memory's need
  grew by one to three budget steps, and no uncharged ratio moved by more than one step (dense store 4 times the
  static table's cells without errors at every size, sparse store 1.7 to 2 times, insertable table level with the
  sparse store), so the uncharged comparison is not specific to this workload; charged, the tables' advantage shrinks
  as the vocabulary grows relative to the store, to a tie with the sparse store at 24,000 values."
- **x1.** "At the measured noise (x1) no cell errs: the device model flipped none of 2 x 10^8 simulated cells after 30
  days, so the x1 requirements are set by each memory's content alone."
- **Insertion.** "The insertion comparison is logical, not device-simulated: both the cuckoo table and the sparse-block
  store were programmed once with their final contents; the rewrites that insertions would cause on the devices
  (cuckoo moves, parity updates of the codewords touched, cells of different ages, write endurance) were not
  simulated."

Must not say: that charging the vocabulary favours the stores (it dilutes every ratio equally, never reverses one);
that the charged ties hold for every vocabulary (they depend on the strings' bits per pair); that the sweep covers
other store sizes, value formats, damage, multi-level cells or LDPC-protected vocabularies (not run); that the
insertion result includes device effects of writing.

## Limitations

Synthetic stores of one size (6,002 pairs) and one value format (7-digit numbers) in the sweep; 8 test stores per N;
restricted rate and fingerprint grids (the k4 LDPC rate at the restricted grid's inner edge); E-CAM and explaining
away not run in the sweep; the charged accounting brackets LDPC protection of the vocabulary between the capacity bound
and BCH rather than simulating it; the per-pair charge uses the most expensive store in scope (B58's convention), so
pooled charged numbers are set by one family; budgets are on a 19% grid, so a charged difference in cells per pair is
quantized.
