# B46 dev record (dev split only; written before any B46 test run)

Date: 2026-11-19, 09:20 box clock. No test store has been built or scored on the B46 codes.

## What ran (b46_out/logs/steps.log; local CPU)

B32's VSA dev grid (4-64 bits per fact, 72 configurations x 36 dev stores), B32's index dev grid (48 + EXACT) after
all 44 PQ / OPQ fits were cached, B32's grids at 1-2 bits per fact (b36_run.dev32), B36's PQ codebooks and its dev grid
of the new families (SF 176, CM 132 / 44, BF 60, FBF 12, SDM 40; 36 stores, both accountings), then B32's choice and
B36's choice (results/b46_choice_b32.json, results/b46_choice_b36.json; the same 168 / 904 choice keys as B36's files).
Execution notes (order and parallelism only; no grid, rule, seed or code changed):
- 06:56 the first driver (b46_dev.sh) filled memory (14 single-thread OPQ fits + B36 PQ codebooks + VSA grid; swap
  8 / 8 GB). The seven whiten OPQ fits were stopped before writing anything and rerun after the centre fits; the driver
  was replaced by b46_dev3.sh (same steps; B36's family grid started after the VSA grid with 3 workers instead of 4).
- 08:45 dev32 was started by hand in parallel with the index grid, when every fit was already cached (so no process
  wrote a cache file); the driver's own dev32 call later found every store done. Line counts: VSA 216, index 36,
  dev32 216 + 36, new families 36 (no duplicates).

## Dev results (selected configurations; optimistic for every memory)

results/b46_devrecord.txt, results/b46_devsum.txt.
- R1 dev VSA - BEST: **+6.54** (ideal; cells +3.2 to +11.0), +7.97 (real). For reference, B36's dev: Qwen3-8B +6.83,
  e5 -0.14 (ideal).
- R0 +0.91, RH +1.85, R2 -4.76 (ideal).
- BEST family (ideal, 56 cells): IDX 34, CM 17, SF 4, SDM 1; in R1: CM 7, SF 2, IDX 3. Key-free families are the dev
  BEST in 9 of the 12 R1 cells (P4 holds on the dev choice; B36: 8 / 12 Qwen3-8B, 7 / 12 e5).
- VSA choice: lin, q = 1 in 50 of 56 ideal cells; index: LSH 33, PQ8 12, PCA-sign 7, ITQ 3, PQ4 1.
- Choices at <= 8 bits per fact: the LSH bucket sketch (CM) mostly with coarse PQ k = 4-8 or SimHash k = 4-6
  signatures, as in B36.

## Next (registered)

Test once (b46_test.sh: B32 test, B36 test, test32, merges, b46_verdict.py), then the checks (b46_check.py score / bits /
rederive / rederive32), hand reading (b46_handread.py), predictions (b46_predcheck.py).
