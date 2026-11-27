# B63 note after the test (test run once; results/b63_verdict.{txt,json}, results/b63_check.txt, results/b63_sample_reading.txt)

## What was run

Pre-registered (d4cea81, 02:28 machine clock; OpenTimestamps proof a3d40f2) and run as registered, with no change to
the code after the pre-registration commit. Dev on the 12 dev stores (two passes, choices committed acfceb7 before the
test); the test once on the 50 RULER test stores (900 jobs on the box, 02:53-02:57); the extension rule added nothing;
no cut. Setting unchanged from B59 part 2 and B61: binary PCM cells at the 30-day bit-error rates, 4,776 probed keys,
>= 99.5% exact, budget grid 8-4,096 cells per stored pair. U, H-SBC and the MPHF tables are B38 / B59 / B61's
registered rows (reproduced exactly by the verdict code with no B63 data).

Arms: the best coded tables of B38 with their minimal perfect hash replaced by a directory that accepts online
insertion: a cuckoo table (Fan et al. 2014; 4 slots per bucket, two candidate buckets, partial-key cuckoo hashing, load
<= 0.95, 500 kicks), each slot [f-bit fingerprint | last flag | value index], every slot and every fingerprint bit on the
cells, pairs inserted one at a time; **O-BCH** (f chosen on dev, BCH by B38's closed form) and **O-LDPC** ((f, R) chosen
on dev). A fingerprint false match fails the key.

## Result

Cells per stored pair for >= 99.5% of the 4,776 facts exact (i.i.d. 30-day cell errors; steps of 2^(1/4) = 19%):

| k | online table (OT-best) | H-SBC (sparse store) | U (dense store) | static MPHF table | Delta vs H-SBC | Delta vs U | online penalty | bootstrap Delta vs H-SBC |
|---|---|---|---|---|---|---|---|---|
| k1 (0) | 32 (O-BCH = O-LDPC) | 32 | 76 | 19 (E-BCH) | 0 (tie) | -5 (U 2.4x) | +3 (x1.7) | [0, 0] |
| k2 | 32 (both) | 32 | 76 | 19 | 0 | -5 (2.4x) | +3 | [0, 0] |
| k4 (0.021) | 38 (O-LDPC) | 45 | 76 | 27 (E-BCH) | -1 (tie) | -4 (2.0x) | +2 (x1.4) | [-1, 0] |
| 5% control | 64 (both) | 54 | 91 | 38 (E-BCH) | +1 (tie) | -2 (1.4x) | +3 | [0, +1] |
| k8 (0.123) | 91 (O-LDPC) | 91 | 128 | 54 (E-LDPC) | 0 | -2 (1.4x) | +3 | [0, 0] |
| k16 (0.276) | 304 (O-LDPC) | 304 | 431 | 215 (E-LDPC) | 0 | -2 (1.4x) | +2 | [-1, +1] |
| k32 (0.374) | 1,218 (O-LDPC) | 1,448 | 1,722 | 724 (E-LDPC) | -1 (tie) | -2 (1.4x) | +3 | [-3, -1] |

(Delta = steps(online table) - steps(store); + = the store needs fewer cells. O-BCH alone: 32 / 32 / 45 / 64 / 181 /
724 / 2,435.)

- **With a directory that supports online insertion, the best coded table needs as many cells as the sparse block code
  store: H1-ON TIE** (|Delta| <= 1 at every k; Delta 0 at k1, k2, k8, k16). The bootstrap keeps Delta within one step at
  k1-k16; at k32 it ranges [-3, -1], leaning to the table (1,218 against 1,448, one step).
- **Against the dense store U: TABLE WIN** (the online table needs 2.4x fewer cells at k1-k2, 2.0x at k4 and 1.4x at
  e05-k32; bootstrap Delta [-5, -4] to [-2, 0]).
- **The online directory costs 2-3 grid steps (x1.4-1.7) against the static MPHF table** at every k (bootstrap +2 to
  +4): fingerprints and empty slots are 49-55% of the logical bits on the 6,000-pair stores (61% median over all
  evaluations; 27.4 / 31.6 logical bits per pair there at f = 12 / 16), where the MPHF's displacement array costs about
  2 bits per pair (16.2 logical bits per pair on synthetic stores of that size, self-test).
- At N* the fingerprint width is not what binds: 5 ambiguous lookups (false matches) per 4,776 keys at f = 12, 0-1 at
  f = 14-16, against the 23 failures the 99.5% bar allows. At k1 the step below N* (27 cells) cannot hold the cuckoo rows
  on 40 of the 50 stores (27.4 logical bits per pair on the 6,000-pair stores at f = 12 and load 0.95), so k1 = 32 is the
  row content itself; at k8 / k16 the failures one step below N* are residual bit errors of undecoded LDPC codewords
  (6-8% of the logical bits) in the key's own fingerprint or payload (sample reading).

## Predictions

S1 held (OT-best within 1 step of the model "MPHF best + 3 steps" at every k: 0 / 0 / -1 / 0 / 0 / -1 / 0; Pen +2 to
+3); S2 held (TIE vs H-SBC); S3 held (TABLE WIN vs U, Delta <= -2 at 6 of 6); **S4 failed** (O-LDPC already beats O-BCH
at k4, 38 against 45; it ties at k1, k2, e05); **S5 failed** (dev f at dev N* was 14 or 16 in 8 of 14 cells, 12 in the
other 6: false matches at f = 12 are rarer than the noise errors at those budgets); S6 held (0 lost insertions on the
test stores; one on a 23-pair dev store, compute note 2); S7 held (checks below).

## Checks

Second scorer: 24 / 24 sampled test cells identical on the same decoded bits (independent hash code, bit-by-bit slot
extraction, explicit candidate loops, targets from the stores' own pairs), box vs local 24 / 24; 5,350 test
evaluations, 0 over budget, logical bits = m x 4 x (f + payload bits) in every one (empty slots included); N* by a
second rule 0 / 14 mismatches; insertion audit 200 (store, f) builds, every pair in one of its buckets under its own
fingerprint, 0 lost, a half build continued by online insertion equals the full build in every build. Sample reading by
the AI-assisted process (results/b63_sample_reading.txt; not independent): k1 nothing to read (the step below N* is
infeasible on 40 stores); k8 and k16: 11 / 18 and 13 / 18 sampled failures have the key's own fingerprint hit by residual
bit errors (the nearest-slot fallback then picks another slot, ties, or lands on a damaged payload), 7 / 18 and 5 / 18
an exact fingerprint match with a damaged payload; no fingerprint false match among the sampled keys.

## Reading against the literature (conclusion -> literature check)

- Supports the asymmetry being structural: static retrieval (a function stored for a fixed key set, as the MPHF table
  here) needs n v (1 + o(1)) bits for n keys of v-bit values (Bloomier filters, Chazelle et al. 2004; ribbon retrieval,
  Dillinger and Walzer 2021), whereas any dynamic retrieval structure needs n v + Omega(n log log(u / n)) bits
  (Mortensen, Pagh and Patrascu 2005; Demaine, Meyer auf der Heide, Pagh and Patrascu 2006, arXiv cs/0512081). Online
  insertion has an unavoidable cost per key.
- Partly conflicting with how large we measured it: that lower bound is about log log(u / n), a handful of bits per
  key, while the cuckoo table pays about 11-15 bits per pair over the MPHF table (fingerprint and empty slots, f = 12-16). The
  theoretical dynamic structures that approach the bound (Demaine et al. 2006) were not built; an online table near the
  bound could need fewer cells than ours, so "as many cells as the sparse store" is a statement about the standard
  practical design (fingerprints, Fan et al. 2014), not about every possible online table.
- The tie has a simple reading: the sparse store is a Bloom-type memory (B61's literature note), and the online table
  is a cuckoo-filter-type memory holding its values; Fan et al. report that cuckoo filters need less space than
  space-optimized Bloom filters at moderately low false-positive rates, but here the sparse store rejects about 12,000
  candidate values per slot by superposition (about 31 cells per pair at k1, Bloom bound) while the table stores the
  value index explicitly beside its fingerprint (27.4 bits per pair at f = 12) - two routes to about the same budget.

## Limitations

One online design was run (a 4-way bucketized cuckoo table at load <= 0.95, the standard practical one); the quotient
filter was not run (registered closed-form note: within one bit per slot of the cuckoo table, below one grid step),
nor larger buckets, a stash, or the theoretical dynamic retrieval structures. f was searched on a 2-bit grid. Both the
cuckoo table and H-SBC are provisioned for the final n (m and the shard geometry are fixed from n and B); growing
beyond it was not run for either. U's N* is that of a batch-written store: appending to the signs alone is not the same
store (it needs the integer sums), so U is not an online memory on its cells as built. i.i.d. errors at 30 days only;
no damage conditions for the online tables. OT-best is selected per k with knowledge of the test, but it equals O-LDPC's
own N* (its dev-chosen designs) at every k (O-BCH ties at k1, k2 and e05), so nothing is gained by the selection.

## What the paper may and must not say

- **"Coded tables needed 1.4 to 2 times fewer cells than the sparse block code store and 2 to 4 times fewer than the
  dense store"** - **must carry the static qualifier**; it holds for tables whose directory is a minimal perfect hash
  built with every key known. The paper may say: "Coded tables whose rows are found through a minimal perfect hash -
  built once with every key known, and rebuilt to add a key - needed 1.4 to 2 times fewer cells than the sparse block
  code store and 2 to 4 times fewer than the dense store. With a directory that accepts insertions one at a time (a
  cuckoo table with fingerprints, every slot and fingerprint bit counted), the best coded table needed as many cells as
  the sparse block code store, within one budget step at every noise level (32 against 32 cells per fact without
  errors, 91 against 91 at k8, 304 against 304 at k16), and 1.4 to 2.4 times fewer than the dense store."
- **The static-directory caveat** must be stated where the comparison is: the best tables' advantage over the sparse
  store is the static directory's (online insertion costs the tables 1.4-1.7 times the cells); the sparse store accepts
  a new pair exactly (OR on binary cells) within its provisioned size; the dense store as measured is batch-written (an
  exact append needs its integer sums, which are not on its cells).
- **"The store needed more cells than coded tables at equal reliability"** must be restricted: true for the dense store
  against every coded table (static and online), and for the sparse store against static tables; against online coded
  tables the sparse store ties.
- Must not say: that the sparse store needs fewer cells than online tables (a tie; at k32 the bootstrap leans to the
  table); that VSA stores have a storage advantage over coded tables in general; that online tables need 1.4-1.7 times
  the cells of static ones in general (one practical design; the theoretical bound is lower); that the dense store is
  an online memory as measured.
