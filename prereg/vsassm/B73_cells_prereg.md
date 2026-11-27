# B73 (equal cells) pre-registration: the device comparison on 50 fresh key-value stores, every design frozen

Date: 2026-11-23, about 19:35 machine clock (Asia/Shanghai; US Eastern 06:35). Written and committed, then
OpenTimestamps-stamped and the proof committed, BEFORE any fresh test store is generated. This file registers the
equal-cell half of B73, the confirmatory run requested by the round-23 referee ("with every configuration frozen, on
fresh data ... new key-value stores with an independent value format"). The equal-bit half is registered separately in
prereg/B73_confirmatory_prereg.md. **An external registration (OSF) needs the human author and is pending; this
internal registration (git commit + OpenTimestamps proof) does not replace it.** The paper is not edited by B73.

## What has been seen before this file (disclosed)

- Every result of the paper's device runs on the 50 RULER stores (B34 / B38 / B58 / B59 / B61 / B63 / B69): the pooled
  N*, the per-family N*, the dev choices. The claims tested below were written from them.
- b73_cells.py's self-test (synthetic stores from the same generator with the seed tag **b73cells-selftest**, which no
  test store uses; 2 stores per family): generator invariants, and one evaluation per memory at its development N* at
  k1 and k8 on one store per family, to check that every frozen design runs and stays within its cells and to time
  the jobs. Seen: every memory read 150 / 150 (or every key of the small stores) except H-SBC 149 / 150 on one
  1,473-pair store at 32 and at 90.5 cells per pair; every evaluation used <= its budget. Nothing was chosen from it.
- No fresh test store exists; the test seed tag (**b73cells**) has never been run.

## Fresh stores (b73_cells.make_store; frozen)

50 stores, 10 per family, mirroring the sizes of the paper's five RULER families (RULER: 10 x ~5,990 pairs with 2-5
two-value keys, 10 x ~1,480, 10 x ~1,700, 10 x 20-23 multi-value, 10 x 9-13 variable-tracking; 92,080 pairs, 4,776
probed keys), with key and value formats that RULER and B34's synthetic generator never use. Words come from the system
dictionary /usr/share/dict/american-english (MD5 16de2454dee65e9ceed77f9c1cd8a15e; lower-case alphabetic words of 4-8
letters, 34,912 words), not from wonderwords (RULER's and B34's source).

| family | stores | keys | values per key | key format | value format |
|---|---|---|---|---|---|
| wp6k | 10 | 5,975-5,995 (uniform) | 1; 2-5 random keys have 2 | `word/word` | `word word` (two dictionary words) |
| code1k5 | 10 | 1,460-1,495 | 1 | `word/word` | `ABCD-123` (4 capitals, hyphen, 3 digits) |
| hex1k7 | 10 | 1,695-1,725 | 1 | 24 hex digits | 24 hex digits |
| mv | 10 | 18 | 15 x 1, 2 x 2, 1 x 4 (23 pairs) | `word/word` | `word word` |
| tiny | 10 | rep 0: 9; reps 1-9: 10 | rep 0: 9 x 1; else 7 x 1, 3 x 2 (13 pairs) | `xy_7` | 9 six-digit numbers shared across keys |

Keys distinct within a store; a key's values distinct; values otherwise drawn independently (repeats possible, rare
except in tiny). Seed: random.Random('b73cells:test:{family}:{rep}'). Probed keys: B25's unchanged rule (at most 150
per store, every key of a small store; expected about 4,780 in all).

## Memories and frozen designs (results/b73_cells_frozen.json, committed with this file)

All memory code is the paper's, imported unchanged (b38_eq, b61_hd, b63_online, through b69_vocab.run_job; MD5s in the
frozen table): U (the dense store, B38's V-last), H-SBC (B61's sparse block code store), T-rep, E-BCH, E-LDPC (B38's
static-directory tables: minimal perfect hash + rows, repetition / BCH / LDPC coded), O-BCH, O-LDPC (B63's cuckoo tables
that insert keys one at a time). **static** = per condition the smallest N* of T-rep, E-BCH and E-LDPC; **insert** = the
smaller of O-BCH and O-LDPC (the paper's "best static table" and "best insertable table"). E-CAM, part of the paper's
static minimum, is not run: it was never the best static table at any noise level on the RULER stores, and leaving a
candidate out of a minimum can only raise the static N*, i.e. it works against claims D1 and D2.

**Design-mapping rule.** The design used at (memory, condition, budget) is the paper's development choice there:
B61's lam for H-SBC (results/b61_choice.json), B38's rate R for E-LDPC (results/b38_choice.json), B63's f for O-BCH
and (f, R) for O-LDPC (results/b63_choice.json); at a budget not evaluated on development, the design of the nearest
development budget (ties: the larger; b61_hd.design_at's rule). U and T-rep have no design parameter; E-BCH uses B38's
closed-form per-store rule ('auto') as registered. These parameters are chosen per (condition, budget), not per store
size, so a frozen design exists at every budget for every store. Everything that depends on the store (H-SBC's shards and
block length, the cuckoo table's bucket count, the hash's displacements, E-BCH's closed-form code choice, codeword
counts) is computed by the unchanged code from the store and the budget, as on the RULER stores. Where a design does not
fit a store at a budget (the code reports it infeasible), that store is not evaluated there and that budget does not
count for the pooled curve (B38 / B69's convention). No development selection is made on fresh stores.

**Budgets.** The paper's grid (8 to 4,096 cells per stored pair in quarter octaves, one step = x 2^(1/4), about 19%).
Per (memory, condition): every grid budget within 4 steps of the paper's development N* (B38 / B63's test-window rule;
B61's H-SBC test ran from the grid bottom, here the same +- 4 window for every memory), then ONE round of B38's extension
rule (4 steps up if the top evaluated budget fails; 4 steps down if N* is the lowest evaluated budget). An N* still not
bracketed after that round is reported as censored ('> top' = not reached, counted as not reached).

**Conditions.** Independent cell errors at the device model's 30-day bit-error rate multiplied by 1, 8 and 16 (b38_eq.IID
k1 = 0, k8 = 0.1226, k16 = 0.2760; at x1 no cell errs). The value vocabulary is uncharged (the paper's headline
accounting). Not run: k2, k4, e05, k32, damage conditions, the charged accounting.

**N\*** = the smallest grid budget whose pooled share of probed keys read exactly (every value of the key, in order) is
>= 99.5% there and at every larger evaluated budget, pooled over the 50 stores (b69_vocab.nstar_counts = B38 / B58's rule).

## The claims, the paper's values (RULER stores, pooled, uncharged; results/b69_partA.json) and the criteria

Paper N* (k1 / k8 / k16): U 76 / 128 / 431; H-SBC 32 / 91 / 304; static 19 / 54 / 215; insert 32 / 91 / 304
(T-rep 19 / 256 / 1,024; E-BCH 19 / 108 / 431; E-LDPC 23 / 54 / 215; O-BCH 32 / 181 / 724; O-LDPC 32 / 91 / 304).
Delta = steps(table) - steps(store); per condition "table fewer" if Delta <= -2, "store fewer" if >= +2, else tie;
label over k1, k8, k16 = B58's h_label (TABLE WIN / TIE / VSA WIN / MIXED).

| claim | paper text | paper value (Delta k1 / k8 / k16; ratio) | criterion: CONFIRMED iff, at each of k1, k8, k16 | exact range (reported) |
|---|---|---|---|---|
| D1 static vs dense | "2 to 4 times fewer cells than the dense store" | -8 / -5 / -4; 4.0 / 2.4 / 2.0 | label TABLE WIN and Delta in [-9, -3] (ratio 1.68-4.76: the range +- one grid step) | Delta in [-8, -4] (2-4 times) |
| D2 static vs sparse | "1.4 to 2 times fewer than the sparse-block store" | -3 / -3 / -2; 1.68 / 1.68 / 1.41 | label TABLE WIN and Delta in [-5, -1] (1.19-2.38) | Delta in [-4, -2] (1.41-2) |
| D3 insertable vs sparse | "as many, within one grid step" | 0 / 0 / 0 | \|Delta\| <= 1 (label TIE); no further margin, the claim is its own tolerance | the same |
| D4 insertable vs dense | "1.4 to 2.4 times fewer than the dense store" | -5 / -2 / -2; 2.38 / 1.41 / 1.41 | label TABLE WIN and Delta in [-6, -1] (1.19-2.83) | Delta in [-5, -2] |
| O1 ordering | static < sparse < dense | 19 < 32 < 76; 54 < 91 < 128; 215 < 304 < 431 | N*(static) < N*(H-SBC) < N*(U) and N*(insert) < N*(U) | - |

A claim is **confirmed** if its criterion holds at all three conditions, otherwise **not confirmed** (a censored N*
that leaves a criterion undecidable counts as not confirmed). "Exact range held" is reported beside it and decides only
whether the paper's numbers may stand unchanged (readings below). Nothing else counts toward a verdict.

Secondary (reported, no decision): per-family N* and Deltas; the share of 2,000 store-bootstrap draws (stores
resampled with replacement within family; seed 'b73cells-boot') in which each criterion holds; cuckoo lost insertions;
each memory's N* against the paper's.

## Predictions (stated so that they can be wrong)

- Q1. Each of the 21 (memory, condition) pooled N* is within one grid step of the paper's in at least 19 of 21.
- Q2. D1, D2, D3, D4 and O1 are all confirmed; D1 and D2 also hold in their exact ranges.
- Q3. At k1 the static table's N* is set by T-rep / E-BCH (19 or 23 cells per pair) and H-SBC's by the wp6k family.
- Q4. No cuckoo insertion is lost on a store of >= 1,000 pairs (losses on the small stores are possible, as in B63).
- Q5. On the wp6k family alone, the labels of D1-D4 equal the pooled ones.

## Checks (b73_cells_check.py)

- **Plain-loop scorer**: every pooled share and N* recomputed with plain loops from the raw result rows (independent
  of b69_vocab / b58_verdict code), and each row's exact count recomputed from its failed-key list; 0 differences.
- **Cells audit**: every feasible row's cells <= floor(B x n_pairs) and B on the grid; 0 over budget.
- **Determinism re-run**: a seeded sample of >= 5% of the test jobs (at least 20, covering every memory) re-run from
  scratch; failed-key lists identical.
- **Leakage / independence**: no key string and no value string of any fresh store equals a key or value string of the
  50 RULER test stores, the 12 development stores (b34_out/*_stores.json.gz) or B69's sweep stores; reported counts.
- **Frozen-design audit**: every counted row used the design of results/b73_cells_frozen.json at its (condition,
  budget); the code files' MD5s equal the frozen table's.
- **Sample reading by the AI-assisted process**: for each memory at the grid step below its k8 N*, which probed keys
  fail (family, key, number of values), with notes.

## Readings for the paper (fixed now)

- "error-corrected tables with a static directory needed two to four times fewer cells than the dense store": stands
  as written if D1 is confirmed with the exact range held; if confirmed only within the margin, the paper must add the
  fresh-store ratios beside it; if not confirmed, the sentence must be restricted to the RULER stores.
- "1.4 to 2 times fewer than the sparse-block store": the same with D2.
- "the sparse-block store ... as many, within one grid step, as tables that accept new keys": stands if D3 is
  confirmed; otherwise it must be restricted to the RULER stores and the fresh-store Deltas given.
- "1.4 to 2.4 times fewer than the dense store" (insertable): the same with D4.
- B73 tests k1, k8 and k16 only: a confirmed claim may be called "confirmed on fresh stores with frozen designs at 1, 8
  and 16 times the noise"; "at every noise level" stays a RULER-store statement.
- The paper must not present B73 as externally registered until the author registers it.

## Compute and time rule

This machine only, at most 8 worker processes, single-threaded BLAS (OMP / OPENBLAS / MKL / NUMBA threads = 1); outputs
under b73_out/cells/ (results/b73_cells_* small files committed). The A800 box and GPUs are not used. Order: commit this
file, the frozen table and b73_cells.py; stamp; commit the proof; generate the test stores; run the test once; the
extension round; a compute note BEFORE any test output is read (only job-completion lines and the extension count may be
seen); then verdict, checks, sample reading, note after the test. Time: all done by 22:40 machine clock. If the test
(with its extension) has not finished by 21:45, k16 is cut first (then k8), in a dated compute note written before any
test output is read.

## Files

b73_cells.py (generator, frozen table, plans, runner), results/b73_cells_frozen.json, b73_cells_verdict.py,
b73_cells_check.py (written after this file, before the test output is read), b73_out/cells/ (stores, plans, raw rows),
results/b73_cells_*.{txt,json}, prereg/B73_cells_compute_note{k}.md, prereg/B73_cells_note_after_test.md.
