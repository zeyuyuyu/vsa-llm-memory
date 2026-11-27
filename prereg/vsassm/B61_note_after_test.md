# B61 note after the test (test run once; results/b61_verdict.{txt,json}, results/b61_check.txt, results/b61_sample_reading.txt, results/b61_predictions.txt)

## What was run

Pre-registered (c2b548b, 10:32; OpenTimestamps proof 584e991) and run as registered, with one fix before any dev
output was read (compute note 1: H-SIC scores from integer counts, so ties follow the registered argmax rule rather than
floating-point rounding; found by the independent scorer on synthetic pairs). Dev on the 12 dev stores (two passes,
B58's rule; choices committed 47b08b7 before the test); the test once on the 50 RULER test stores (4,600 jobs on the
box, 10:54-11:00); B38's extension rule added nothing; no cut. Setting unchanged from B59 part 2: binary PCM cells at the
30-day bit-error rates, 4,776 probed keys, >= 99.5% exact, budget grid 8-4,096 cells per stored pair; U and the tables
are B38 / B59's registered rows.

Arms: **H-SBC**, a sparse block code store (one active position per block, block-wise index addition as binding, OR
superposition of a shard's pairs on the cells, overlap clean-up, B38's V-last stop rule; the block load lam chosen on
dev); **H-SIC**, the same cells read a second time with explaining away over the probe batch; **H-HAD**, the dense store
with a first-order Reed-Muller (Walsh-Hadamard) item codebook whose clean-up is the code's maximum-likelihood decoder
restricted to the vocabulary.

## Result

Cells per stored pair for >= 99.5% of the 4,776 facts exact (i.i.d. 30-day cell errors; steps = grid steps of 2^(1/4)):

| k | H-SBC | H-SIC | H-HAD | U (dense, uncoded) | best table | G = U - H-SBC (steps) | Delta = table - H-SBC (steps) | bootstrap G / Delta (H-SBC) |
|---|---|---|---|---|---|---|---|---|
| k1 (0) | 32 | 32 | 76 | 76 | 19 (E-BCH) | +5 (U needs 2.4x) | -3 (table 1.7x fewer) | [+4, +5] / [-3, -3] |
| k2 | 32 | 32 | 76 | 76 | 19 (E-BCH) | +5 (2.4x) | -3 (1.7x) | [+4, +5] / [-3, -3] |
| k4 (0.021) | 45 | 45 | 76 | 76 | 27 (E-BCH) | +3 (1.7x) | -3 (1.7x) | [+3, +4] / [-3, -2] |
| 5% control | 54 | 54 | 91 | 91 | 38 (E-BCH) | +3 (1.7x) | -2 (1.4x) | [+2, +3] / [-3, -2] |
| k8 (0.123) | 91 | 91 | 128 | 128 | 54 (E-LDPC) | +2 (1.4x) | -3 (1.7x) | [+2, +2] / [-3, -3] |
| k16 (0.276) | 304 | 304 | 431 | 431 | 215 (E-LDPC) | +2 (1.4x) | -2 (1.4x) | [+1, +2] / [-3, -2] |
| k32 (0.374) | 1,448 | 1,448 | 1,722 | 1,722 | 724 (E-LDPC) | +1 (1.2x, tie) | -4 (2.0x) | [+0, +2] / [-5, -4] |

- **Does an HD-native protected store need fewer cells than U? Yes, the sparse block code store does**: "fewer cells
  than U" (G >= +2) at 5 of the 6 labelled k (k1-k16; tie at k32, +1 step), and at the 5% control; the bootstrap keeps G
  >= +1 at k1-k16 and >= +2 at k1-k8. The noise-free requirement falls from 76 to 32 cells per pair.
- **Does it close the gap to the best table? No**: H1-HD **TABLE WIN** for H-SBC, H-SIC, H-HAD and HD-best; the best
  table still needs 2-4 grid steps (1.4-2.0x) fewer cells at every k (bootstrap Delta never above -2). Against U the
  table needed 2.0-4.0x (B59); against the sparse store 1.4-1.7x at k1-k16 and 2.0x at k32 (U: 2.4x there).
- **Explaining away over the probe batch adds nothing on the test** (H-SIC = H-SBC at every k; on dev one step at k16):
  the reader is given 150 of up to 6,000 stored keys per large store.
- **The error-correcting item codebook does nothing** (H-HAD = U at every k, bootstrap G within +-1): the clean-up noise
  is the other bundled pairs' cross-talk (spread sqrt(d) in the sample reading), not the distance between item codes.
- **Damage, natural layout** (failed lines / stuck clusters, k4 / k8): H-SBC 45 / 54 / 91 / 128, H-SIC 45 / 45 / 91 /
  108, U 76 / 76 / 152 / 152, best non-HD table 128 / 128 / 2,896 / 609: **VSA WIN** (store fewer cells in 4 of 4, now
  by +5 to +20 steps), and the sparse store needs fewer cells than U there too (G +3 / +2 / +3 / +1). H-SBC is position
  independent in 3 of 4 (not under stuck clusters at k8: 128 against 91 i.i.d., two steps); H-SIC in 4 of 4.

## Predictions

R1 held (H-SBC within 2 steps of the closed-form model at every k: 0 at k1-k16, +2 at k32); R2 held (G >= +2 at 6 of 7
k counting e05, >= +1 at all 7); R3 held (TABLE WIN, Delta -2 to -4); R4 held (H-SIC = H-SBC); R5 held (H-HAD = U);
R6 held (HD-best = H-SBC / H-SIC); **R7 failed** (dev lam at dev N* was 0.35, the grid's lower edge, at k32); **R8
failed in part** (fewer cells than the best non-HD table in 4 of 4 and than U in 3 of 4 held; position independence
failed at clust-nat-k8). R9 (checks) held.

## Checks

Second scorer 18 / 18 sampled test cells identical (independent numpy / float64 implementations, targets from the
stores' own pairs, box vs local machine); 14,550 evaluations, 0 over budget; N* by a second rule 0 / 29 mismatches;
sample reading by the AI-assisted process (results/b61_sample_reading.txt): H-SBC's k1 failures are full-score
false positives (a wrong value whose 18 positions are all set by other pairs, decided against the true LAST item by the
V-last tie rule), its k16 failures are overlap-SNR errors; H-HAD's k8 failures are wrong values above the true one
with the candidates' correlations spread at sqrt(d).

Exploratory, dev only (compute note 3; not registered, no test run): lam = 0.25, below the grid, is not better than
the chosen lam on dev at k16 / k32 (k32: >= 99.5% from 1,722 against 1,448), so the grid's lower edge is not what holds
H-SBC at k32.

## Reading against the literature (conclusion -> literature check)

- Supports: an OR superposition of sparse codes on binary cells is a Bloom-filter-like memory, a known link (Clarkson,
  Ubaru and Yang, "Capacity analysis of vector symbolic architectures", arXiv 2301.10352, JAIR 2026: binary sparse
  VSAs relate to Bloom filters), and sparse binary associative memories are known to store more per cell than dense
  ones (Willshaw-type memories; Hersche et al. 2023, arXiv 2303.13957, cite the high information capacity of sparse
  block codes). The noise-free requirement matches the Bloom-filter bound: rejecting about 2 x 6,000 wrong candidates
  per slot at a per-key failure of 0.5% needs a false-positive rate near 3e-7, i.e. about 1.44 log2(1 / 3e-7) = 31
  cells per pair, against the measured 32.
- The remaining gap is structural, not a tuning artefact: the table addresses each pair directly (value index of about
  13 bits, a LAST flag and a few hash-directory bits: 19 cells at k1), while any superposition read must reject every
  other vocabulary item. No conflicting published result (a sparse VSA store beating a coded table at equal cells) was
  found in the searches made.

## Limitations

H-SIC explains away only the probe batch; explaining away over every stored key (an oracle key list) was not run,
because with the key list free a table would also need no key directory. Resonator decoding reduces to the clean-up
here (one unknown factor). The sparse store uses B38's shards (<= 8,192 cells); at large budgets they hold few pairs and
short blocks (compute note 2), which may cost H-SBC at k16 / k32; longer shards were not run. Binary cells at 30 days
only (no multi-level cells); interleaved and other damage geometries not run for the HD arms. HD-best is selected per k
with knowledge of the test (it equals H-SBC's own dev-chosen design at every k, so nothing is gained by the selection).

## What the paper may and must not say

- **"Coded tables needed two to four times fewer cells"** - **must change**. It holds against the dense bipolar store
  (B38-B59). Against the strongest HD-native store, a sparse block code, the best table needs **1.4 to 2 times** fewer
  cells (2-4 grid steps, at every noise level, bootstrap never closer than 2 steps). The paper may say: "coded tables
  needed 1.4 to 2 times fewer cells than the best vector store we built (sparse block codes), and 2 to 4 times fewer than
  the dense bipolar store." The bare "two to four times" must not be stated about "the store" in general.
- **"Coding the store's bits never helped"** - **survives only restricted**: true for channel codes (BCH, LDPC) on the
  dense store's bits (B58, B59) and for an error-correcting item codebook (H-HAD = U at every k), but an HD-native
  change of representation did help: the sparse block code store needs 1.4-2.4 times fewer cells than the uncoded
  dense store at k1-k16 (tie at k32). The sentence must name the channel codes and add the sparse result; it must not be
  generalized to "no protection helps the store".
- **"The store needed more cells than coded tables at equal reliability"** - **survives**, now with every HD-native
  variant tested: TABLE WIN for H-SBC, H-SIC and H-HAD at every k. The paper may add that the gap is that of a
  superposition read against direct addressing (the Bloom-filter bound), not of a missing code.
- New statements the paper may make: the store's 76-cell noise-free requirement belongs to the dense bipolar code; a
  sparse block code lowers it to 32 (the table: 19). Under natural-layout damage the sparse store needs even fewer cells
  than the dense one (45 / 54 / 91 / 128 against 76 / 76 / 152 / 152) and still far fewer than every non-HD table, but
  it is not position independent under stuck clusters at k8 (two steps), whereas the dense store is (within one).
- Must not say: that explaining away or resonator decoding narrows the gap (not shown: the probe batch alone changes
  nothing, the oracle version was not run); that the sparse store reaches the tables at any k; that the Reed-Muller
  item codebook helps.
