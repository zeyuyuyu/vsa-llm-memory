# B75 note after the test (RULER stores; results/b75_verdict.{txt,json}, b75_check_recount.json, b75_check_leak.json, b75_sample_reading.txt)

## What was run
Pre-registered eaa649d (22:48:43), OTS proof 504b449 (22:48:51), **Bitcoin block 977225** (upgraded 23:37:58, a005f02).
Dev on the box 23:38:28-00:36:53 (after the attestation; 3,000 jobs, 0 failures); choices committed a575bab (00:42:49);
test once on the 50 RULER stores from 00:43:19: R1 (U-LS, O2-BCH, O2-LDPC, U-J; 1,790 jobs, done 00:58) and R2a
(H-SBC-J at x1, x8; 530 jobs, done 00:50), 0 failures. **Cut (compute notes 3-4, before any test output was read):
H-SBC-J at x16 (stopped by PID after 21 of 320 jobs), (D) the fresh stores, the determinism re-run, and B38's
extension round** (deadline). Store-best-J at x16 is U-J alone.

## Results (cells per stored pair, >= 99.5% exact; '*' = lowest evaluated budget, bracket open below)
| memory | x1 | x2 | x4 | x8 | x16 | x32 |
|---|---|---|---|---|---|---|
| U (dense, registered) | 76 | 76 | 76 | 128 | 431 | 1,722 |
| **U-LS** (least-squares write) | 54 | 54 | 64 | 108 | 609 | 1,722 |
| **U-J** free / charged | 32 / 45 | - | - | 64 / 91 | 215 / 304 | - |
| H-SBC (registered) | 32 | 32 | 45 | 91 | 304 | 1,448 |
| **H-SBC-J** free / charged | 23 / 32 | - | - | 54 / 76 | cut | - |
| static table (registered) | 19 | 19 | 27 | 54 | 215 | 724 |
| cuckoo table (registered) | 32 | 32 | 38 | 91 | 304 | 1,218 |
| **O2-best** (short fingerprints + stash) | 27* | 27* | 32 | 76 | 256 | 861 |

Deltas (steps, X - Y): U-LS vs U -2/-2/-1/-1/+2/0 (MIXED); U-LS vs H-SBC +3/+3/+2/+1/+4/+1; U-LS vs static +6/+6/+5/+4/+6/+5.
Joint free: U-J vs U -5/-4/-4; H-SBC-J vs H-SBC -2/-3; J-best vs static +1/0/0 (TIE). Joint charged: J-best vs H-SBC
0/-1/0 (TIE); vs static +3/+2/+2 (static 1.4-1.7x fewer); vs cuckoo 0/-1/0. O2-best vs H-SBC -1/-1/-2/-1/-1/-3
(table fewer at x4 and x32); vs cuckoo -1 (x1-x16), -2 (x32); vs static +2/+2/+1/+2/+1/+1. O2: 19.6-20.9 logical bits per
pair (median over stores, by design; 22.7-24 on the 6,000-pair stores) against the cuckoo table's ~27 and the static
table's ~17: 5.7-6.9 bits beyond the static table. 0 lost insertions, 0 stash overflows.

## Predictions
PA1 FAILED (U-LS -2/-2/-1/-1 at x1-x8, but +2 at x16: the dev-chosen LS design at 431-512 cells is worse than U on
test). PA2 held. PB1 held. PB2 held. PC1 held. **PC2 failed in part** (Deltas at x1/x8/x16 as predicted, but the table is
two steps fewer at x4 and three at x32, not "x1 only"). PC3 held. PC4 held (0 lost). PD not run (cut).

## Checks
Plain-loop recount: 29,400 rows, exact counts 0 mismatches, 28 pooled N* recomputed, 0 differences; cells audit 0 over
budget, 0 off grid, O2 logical bits = m x bucket width + stash entries x entry width in all 4,360 O2 rows. Leakage:
source audit of every decode function clean (no stored value referenced); incumbent rows identical to the registered
rows on the same budgets (U-J T=0 vs U 1,650 / 1,650; U-LS (inf, 0) vs U 1,000 / 1,000; H-SBC-J T=0 vs H-SBC 850 / 850);
scrambled-values re-run done on synthetic stores only (self-test), not on test stores. Determinism re-run: NOT run (cut).
Sample reading by the AI-assisted process (results/b75_sample_reading.txt): one step below N*, U-LS fails ~1% of keys
mostly on the 6,000-pair stores; O2-LDPC's failures are undecoded LDPC codewords (nearest-slot fallback ties); O2-BCH's
are BCH decoding failures; the joint decoders fail a few keys per large store.

## Literature
Joint decoding with interference cancellation is known to raise VSA capacity (Kleyko et al., Neural Computation 2023:
1.20 -> 1.40 bits per dimension) - supports (B). Kuszmaul et al. (SODA 2025, arXiv 2410.10002): insert-only retrieval
can reach O(1) bits per key of redundancy when values are ~log n bits, so O2 (about 6 bits over the static table) is not
optimal; a better insertable table may exist.

## What the paper may and must not say
- Dense store, decorrelating write: may say that a least-squares write lowered the dense store's requirement by two
  steps (76 -> 54) at 1 and 2 times the noise and by one step at 4 and 8 times, not at 16 or 32 times (it was worse at
  16 times as chosen on development). Must not say it reaches the sparse-block store or the tables.
- "2-4 times fewer than the dense store": stands against the dense store read by its matched filter (U and U-LS: the
  static tables need 2.0-2.8 times fewer than U-LS). Must be qualified for joint decoding: with the key list charged at
  its minimum the jointly decoded dense store needs 45 / 91 / 304 cells (static tables 1.4-2.4 times fewer); with the
  key list free it matches the static tables within one step.
- Joint decoding: may say that decoding every stored pair jointly lowers the stores' requirement (dense -4 to -5 steps,
  sparse -2 to -3 steps) when the stored-key list is free, and that once that list is charged at its
  information-theoretic minimum the best jointly decoded store ties the sparse-block store as read (0 / -1 / 0 steps),
  ties the cuckoo table, and still needs 1.4-1.7 times the cells of the static tables. Must not say joint decoding beats
  the tables; must not report H-SBC-J at 16 times the noise (cut).
- "1.4-2 times fewer than the sparse-block store": stands for the static tables against the sparse-block store as read
  and against joint decoding with the key list charged (1.4-1.7x); must not be stated against joint decoding with a free
  key list (tie).
- "The sparse-block store matched a table accepting new keys": **must change.** It holds within one grid step against
  B63's cuckoo table; against an insertable table with short fingerprints, a conflict stash and semi-sorted buckets the
  table needs one step fewer at 1, 2, 8 and 16 times the noise (x1 and x2 bracket open: maybe two) and two to three
  steps fewer at 4 and 32 times. Say "an insertable table needed as many or up to 1.4 times fewer cells (2 times at the
  highest noise)", with this design named; must not say the sparse store matches every insertable table.
- Static tables: still need the fewest cells of everything run, except that the jointly decoded stores with a free
  key list tie them.
- Must not say: that B75 tested fresh stores, H-SBC-J at x16, or determinism (all cut); that O2 is an optimal
  insertable table.
