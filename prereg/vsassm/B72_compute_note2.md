# B72 compute note 2 (2026-11-23, about 19:50 machine clock; no development or test output read)

Procedure changes forced by the slow link to the box (about 3-4 KB/s), decided before any output is read:

- **(B) verdict on the box.** The finite-K contrasts (STORE - LAMC, VSAR - SDMT, STORE - NLMSC at K = 1e3, 1e4, 1e5)
  are computed on the box by b72_bverdict.py from the (B) test jobs at the committed choices, with the test records
  rebuilt from B32's make_store (qgold, qent per test store: the same records the merged files hold) and the registered
  statistics (b66_storeboot.analyse: one-level and two-level, 2,000 draws), a plain-loop recomputation of every point
  statistic, and the bits audit of every (B) dev and test row (b72_check.bits' rules). Only the small part files
  (results/b72_bpart_{enc}.json, b72_check_bits_box.json) come back. The K = infinity row of each (B) table is computed
  locally from the existing arms (b72_verdict.py part, 'Binf').
- **Consequence (a cut of a registered secondary):** the cost-of-charge secondaries STORE(K) - STORE and LAMC(K) - LAMC
  need the K = infinity predictions, which are not on the box; they are reported as point differences of region means
  (from the cell accuracies on both sides) WITHOUT an interval, for every (B) memory.
- **Transport of (A) / (D) test outputs.** The language models' OPT and LAMP test jobs are packed on the box
  (b72_pack.py: rows without predictions + predictions as uint8 in one compressed npz) and rebuilt locally into the
  same job files; b72_verdict.py then reads them unchanged.
- Code added or changed after the pre-registration commit (procedure only; no arm, grid, rule, statistic or bar
  changed): b72_bverdict.py, b72_pack.py (new); b72_verdict.py (part computes K = infinity only for (B); cpart = (C) on
  the four headline contrasts from saved outputs alone; panel merges the box's (B) part); b72_check.py (sample-reading
  material command). Box copies md5-checked (b36_verdict, b66_storeboot, b71_verdict, b72_verdict, b72_check,
  b72_bverdict).
