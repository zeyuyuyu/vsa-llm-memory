# B73 (equal cells) note after the test (test run once, no cut; results/b73_cells_verdict.{txt,json}, b73_cells_check.txt, b73_cells_check_*.json, b73_cells_leak.json, b73_cells_sample_reading.txt, b73_cells_curves_vs_ruler.txt)

## What was run

Pre-registered (8deea4e, 19:29:27 machine clock; OpenTimestamps proof 9c2e552, 19:29:34; the header's "about 19:35" was
written ahead, compute note 1); fresh stores generated after the proof commit; verdict and check code committed before
the test output was read (7fcccdc, 19:33); test once on this machine 19:29:58-19:59:30 (1,830 jobs, 9,450 store
evaluations, 0 failures, 8 workers, 3.93 CPU-hours); B38's extension rule added nothing; no cut (compute note 2,
6d0387d). The external (OSF) registration is pending the author.

50 fresh stores, 92,094 pairs, 4,779 probed keys (RULER: 92,080 and 4,776), keys and values from the system dictionary,
capital-letter codes, hexadecimal and six-digit strings that no RULER, development or B69 store contains (0 exact
string matches). Every design frozen at the paper's development choice per (condition, budget) (results/
b73_cells_frozen.json); independent cell errors at the 30-day rate x 1, 8, 16; value vocabulary uncharged.

## Pooled requirements (cells per stored pair for >= 99.5% exact; fresh [paper])

| memory | x1 | x8 | x16 |
|---|---|---|---|
| dense store U | 76 [76] | 128 [128] | 431 [431] |
| sparse-block store H-SBC | 32 [32] | 91 [91] | **362 [304]** |
| T-rep | 19 [19] | 256 [256] | 1,024 [1,024] |
| E-BCH | 19 [19] | 108 [108] | 431 [431] |
| E-LDPC | 23 [23] | 54 [54] | 215 [215] |
| O-BCH | 32 [32] | 181 [181] | 724 [724] |
| O-LDPC | 32 [32] | 91 [91] | 304 [304] |
| best static table | 19 [19] (T-rep) | 54 [54] (E-LDPC) | 215 [215] (E-LDPC) |
| best insertable table | 32 [32] | 91 [91] (O-LDPC) | 304 [304] (O-LDPC) |

20 of 21 requirements are identical to the RULER stores' and the 21st is one grid step higher (H-SBC at x16: 0.9925 of
probed keys at 304 cells, against 0.9956 on the RULER stores). The share curves themselves differ in the third decimal at
every budget (results/b73_cells_curves_vs_ruler.txt), so the identity is the 19% grid quantising two different but
equally sized workloads, not a copy. At x1 every table's requirement is the lowest budget at which its content fits every
store (below it the large stores' tables are infeasible; every feasible store read 100%), as on the RULER stores.

## Claims (registered criteria; Delta = grid steps table - store; ratio = store / table)

| claim | paper (Delta x1 / x8 / x16) | fresh Delta | fresh ratio | label | criterion | exact range | verdict |
|---|---|---|---|---|---|---|---|
| D1 static tables vs dense store, "2 to 4 times fewer" | -8 / -5 / -4 | -8 / -5 / -4 | 4.00 / 2.38 / 2.00 | TABLE WIN | [-9, -3] | held | **CONFIRMED** |
| D2 static tables vs sparse-block store, "1.4 to 2 times fewer" | -3 / -3 / -2 | -3 / -3 / -3 | 1.68 / 1.68 / 1.68 | TABLE WIN | [-5, -1] | held | **CONFIRMED** |
| D3 insertable table vs sparse-block store, "within one grid step" | 0 / 0 / 0 | 0 / 0 / -1 | 1.00 / 1.00 / 1.19 | TIE | [-1, +1] | held | **CONFIRMED** |
| D4 insertable table vs dense store, "1.4 to 2.4 times fewer" | -5 / -2 / -2 | -5 / -2 / -2 | 2.38 / 1.41 / 1.41 | TABLE WIN | [-6, -1] | held | **CONFIRMED** |
| O1 ordering static < sparse-block < dense, insertable < dense | 19<32<76; 54<91<128; 215<304<431 | 19<32<76; 54<91<128; 215<362<431 | | | | | **CONFIRMED** |

All five claims are confirmed and every magnitude lies inside the paper's own range (no margin needed). Store bootstrap
(secondary, 2,000 draws within family): D1, D2, D3, D4 hold in 100% of draws and O1 in 99.7%; the exact ranges hold in
99.8% (D1), 100% (D2, D3) and 74% (D4: the insertable-vs-dense ratio at x8 and x16 sits at the range's lower edge, 1.41).

**Per family (secondary).** On each of the three large families every label equals the pooled one. On the 20 small
stores (mv, 23 pairs; tiny, 9-13 pairs) the static table still needs the fewest cells (D1, D2 TABLE WIN), but the cuckoo
table needs 0-3 steps MORE than the sparse-block store (mv +1 / +3 / 0, tiny +2 / +3 / +2: label VSA WIN) and on tiny
one step more than even the dense store at x8 and x16: a cuckoo table of a few buckets (3 at the smallest), as the RULER
small stores showed (B69's note). The pooled claims are set by the large stores.

## Predictions

Q1 held (21 of 21 within one step; 20 identical). Q2 held (D1-D4 and O1 confirmed, D1 and D2 in their exact ranges). Q3
held (static at x1 = T-rep / E-BCH at 19; H-SBC's x1 requirement set by the large families, 32, against 23 and 19 on the
small ones). Q4 held (no lost cuckoo insertion on any fresh store, the small ones included; B63 had lost one on a RULER
development store). Q5 held (wp6k's labels equal the pooled ones). Nothing I predicted failed; the one surprise is the
H-SBC step at x16, which moved D2 at x16 from -2 to -3 and D3 at x16 from 0 to -1, both inside the criteria.

## Checks (results/b73_cells_check.txt)

Plain-loop scorer: 9,450 rows, exact counts = n - |failed| on all; 27 pooled requirements recomputed by plain loops, 0
differences. Cells audit: 8,739 feasible rows, 0 over budget, 0 off the grid; 711 infeasible rows (tables below their
content at x1, cuckoo / LDPC codes that do not fit), not counted as registered. Frozen-design audit: 0 rows with a
non-frozen design; the eight reused code files MD5-identical to the frozen table. Determinism: 92 jobs (5%, every
memory), 496 store evaluations re-run, 496 identical. Leakage: 0 exact key or value strings shared with the RULER test,
development or B69 stores (after case and punctuation folding, 1-10 coincidental word pairs per set, e.g. 'crazy/hedge'
vs 'crazy-hedge'; codes come from exact strings). Sample reading by the AI-assisted process (results/b73_cells_sample_
reading.txt): failures one step below each requirement at x8 are ordinary single-value keys of the large stores; the
stores and repetition / BCH tables lose tens of keys there, the LDPC tables over a thousand (their waterfall); no scoring
artefact.

## Literature

No new conclusion: B73 reproduces the RULER-store device comparison on fresh stores; the supporting and conflicting
sources are those of results/b38_literature.txt and B61 / B63 / B69 (sparse block codes and Bloom-filter bounds,
error-corrected phase-change storage, cuckoo filters). A replication on independently generated keys with frozen
designs is standard practice and changes none of those readings.

## What the paper may and must not say

May say (plain sentences, the authors' wording decides):
1. On 50 new key-value stores drawn by an independent generator (dictionary-word, letter-code and hexadecimal keys and
   values, sizes matched to the RULER stores), with every memory's design frozen at its development choice, the cells per
   stored pair needed at 1, 8 and 16 times the noise equalled those on the RULER stores in 20 of 21 cases; the
   sparse-block store needed one grid step more at 16 times the noise (362 against 304).
2. Tables with a static directory again needed 4.0, 2.4 and 2.0 times fewer cells than the dense store and 1.7 times
   fewer than the sparse-block store at 1, 8 and 16 times the noise; the insertable table needed as many cells as the
   sparse-block store at 1 and 8 times the noise and one step fewer at 16 times, and 1.4 to 2.4 times fewer than the
   dense store.
3. The device claims "2 to 4 times", "1.4 to 2 times", "within one grid step" and "1.4 to 2.4 times" were confirmed by a
   pre-registered run on fresh stores with frozen designs at 1, 8 and 16 times the noise.

Must not say: that B73 tested every noise level (2, 4, 32 times, 5% errors, line and cluster damage, the charged
vocabulary, one day after writing or multi-bit cells were not re-run; "at every level" remains a RULER-store
statement); that the insertable table matches the sparse-block store on small stores (on the 20 small fresh stores the
sparse-block store needed 1 to 3 steps fewer, as on the RULER ones); that the fresh stores test other store or vocabulary
sizes (they match the RULER sizes by design; B69's sweep is the vocabulary test); that x1 was re-simulated on the device
model (x1 is the error-free i.i.d. channel; B69's device check C0 was not re-run); that B73 is externally registered.

## CPU

This machine only: test 3.93 CPU-hours (8 workers, 29.5 minutes), determinism re-run about 0.25, self-test and checks
about 0.2; about 4.4 CPU-hours in all. No GPU, no use of the A800 box.
