# B59 note after the test (test run once; results/b59_store_verdict.{txt,json}, results/b59_check.txt, results/b59_store_sample_reading.txt)

## Part 1 (map x entity intervals): SUPERSEDED

Part 1 ran to completion (dev 02:13-02:38, test 02:42-02:52, verdict read 03:03, committed 03:04 as 3c676ef) before
the B49 LAM defect was reported (compute note 4: the least-squares memory's "pca" basis is computed on already-whitened
codes for prep = whiten; V - 1 target columns; per-column 32-bit scales). Every part 1 comparison is against that LAM
(store - LAM directly; store - BEST through BEST's dev choice and LAM's share of it), so the part 1 results, bar B1 and
predictions P1-P7 are superseded and are not findings; B60 redoes the comparison with a corrected LAM. The files stay
in the repository for the record only.

## Part 2 (primary of this report): the protected store on the full grid, uncoded store inside the search

Scope actually run: dev = B58's pass 1 + P-BCH pass 2 (reused) + P-LDPC pass 2 by the registered rule (new; 8,400
evaluations), U from B38's dev; test on every grid budget 90.5-4,096 for each code family and 8-4,096 for U (B38's U
rows + 4,900 new U evaluations), both code families at every k, 1,594 B58 evaluations reused (20 / 20 recomputed
identical), damage (natural layout, failed lines / stuck clusters, k4 / k8) with B38's extension rule. Budgets not
evaluable on every store (the dev design does not fit; listed in the verdict's coverage table) lie below each family's
N* or are exact-fill budgets that do not fit 23-32 of the 50 stores (compute notes 5, 6). No cut was needed.

Cells per stored pair for >= 99.5% of the 4,776 facts exact (i.i.d. 30-day cell errors):

| k | U (uncoded) | P-BCH | P-LDPC | P-best (= P-any = P-dev) | best table | Delta (steps) | bootstrap Delta | bootstrap P-best |
|---|---|---|---|---|---|---|---|---|
| k1 (0) | 76 | 91 | 91 | 76 | 19 (E-BCH) | -8 | [-8, -7] | [64, 76] |
| k2 | 76 | 91 | 91 | 76 | 19 (E-BCH) | -8 | [-8, -7] | [64, 76] |
| k4 (0.021) | 76 | 91 | 91 | 76 | 27 (E-BCH) | -6 | [-6, -6] | [76, 76] |
| 5% control | 91 | 128 | 108 | 91 | 38 (E-BCH) | -5 | [-5, -4] | [76, 91] |
| k8 (0.123) | 128 | 152 | 181 | 128 | 54 (E-LDPC) | -5 | [-5, -5] | [128, 128] |
| k16 (0.276) | 431 | 1,448 | 724 | 431 | 215 (E-LDPC) | -4 | [-4, -3] | [362, 431] |
| k32 (0.374) | 1,722 | > 4,096 | 2,896 | 1,722 | 724 (E-LDPC) | -5 | [-6, -4] | [1,448, 2,048] |

**H1-P: TABLE WIN** at every k (Delta -4 to -8 grid steps; the best table needs 2.0 (k16) to 4.0 (k1, k2) times fewer
cells). With the uncoded store inside the search, the store's best design at every budget is the uncoded store: at
every k the budget at which P-best first passes is passed by U, and the dev-selected union (P-dev) chose U at every
N* as well (dev N* P-dev = U's: 76 / 76 / 76 / 91 / 128 / 431 / 1,722). **Coding helps the store at 0 of 7 k**: one
grid step worse at k1-k8 (the parity a code adds over the store's 76 information bits per pair), three steps worse at
k16 and k32 (724 / 2,896 with LDPC against 431 / 1,722 uncoded). The full grid and the P-LDPC pass 2 moved the codes'
own N* relative to B58 (P-LDPC e05 108, k8 181, k16 724, k32 2,896; P-BCH e05 128, k8 152, k16 1,448) but none below
U's. The store-cluster bootstrap never brings Delta above -3.

Damage (natural layout; P-any = the smallest family N*; best non-HD table with B58's re-scored E-LDPC*):
lines k4 76 vs 128 (+3), clusters k4 76 vs 128 (+3), lines k8 152 vs 2,896 (+17), clusters k8 152 vs 609 (+8):
VSA WIN in the natural layout, and the store's best form is again the uncoded store (codes 91-1,722). Protected,
the store loses position independence at k8 (lines / clusters: P-BCH 724 / 1,722 against 152 i.i.d.; P-LDPC 256 / 431
against 181 i.i.d.); uncoded it keeps it (152 vs 128, one step). The registered per-budget-maximum P-best is ill-defined in damage because
U has only B38's budgets there (compute note 6; it would read 724 / 1,722 at k8).

## Predictions (part 2)

- Q1 held: TABLE WIN, Delta <= -4 at every k (-4 to -8).
- Q2 held: coding helps at 0 of 7; P-best = U at every k.
- Q3 held: P-LDPC pass 2 lowered P-LDPC's dev N* at 4 of 7 k (e05 128 -> 108, k8 256 -> 152, k16 1,024 -> 861, k32
  4,096 -> 2,896).
- Q4 FAILED in part: P-LDPC <= P-BCH at k16 (724 vs 1,448) and k32 (2,896 vs > 4,096), P-BCH <= P-LDPC at k8 (152 vs
  181), but not at e05 (P-BCH 128, P-LDPC 108).
- Q5 held: P-dev = P-best at 7 of 7.
- Q6 held on P-any (= U in all four natural-layout conditions); on the literal per-budget maximum it fails at k8 (the
  coverage artefact of compute note 6).

## Checks

Second scorer by plain loops (full decode without the shard cache) on 10 B59 test cells: 0 differences. Accounting:
16,642 coded evaluations, 0 over budget, 0 information bits over floor(b n), budgets filled exactly. N* recomputed by a
separately written rule (families and P-best, 7 conditions): 0 mismatches of 28. Reuse: 20 / 20 reused B58 evaluations
recomputed identical; U path: 20 / 20 B38 V-last evaluations recomputed identical by the code that ran T1. Sample
reading by the AI-assisted process (not independent): results/b59_store_sample_reading.txt. Compute: part 2 on the
A800 box's CPUs (<= 48 cores, own processes only) and the local CPU; no GPU, no paid API. Disclosed deviations:
compute notes 1-6 (shard move, staged start, damage tier reinstated by the registered time rule, plan rounding defect
and its T6 repair, the damage P-best artefact).

## What the paper can and must not say (part 2)

Can say: searched over the uncoded store and both of the tables' channel codes (BCH by the tables' closed form, LDPC
over seven rates and four information budgets, development-selected at every budget) on the full budget grid, the
store's best design is the uncoded store at every noise level, and the best table still needs 2 to 4 times fewer
cells per stored pair (19 vs 76 cells with perfect cells; 54 vs 128 at k8; 724 vs 1,722 at k32). Channel coding
never helps the store (one grid step worse up to k8, three at k16 and k32). Under failed lines and stuck clusters in
the natural layout the uncoded store needs fewer cells than every non-interleaved table (1.7 times at 4% damaged
cells: 76 vs 128).

Must not say: anything from B59 part 1 (superseded; B60); that coding helps the store at any tested noise level; that
the protected store keeps the uncoded store's position independence (it does not at k8); that P-LDPC is better than
P-BCH everywhere (P-BCH is better at k8); anything about multi-level cells, interleaved damage or other store reads
(not run).
