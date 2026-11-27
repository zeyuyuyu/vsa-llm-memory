# B61 compute note 2 (10:55 machine clock; dev choices written; no test run yet, no test output exists)

- **Dev ran on the box** (b61_box.sh, 44 workers, code MD5-identical to 504680d): pass 1 10:37:10-10:45:37 (1,956
  jobs, 4,116 (design, budget, condition) evaluations x the 12 dev stores, every job rc = 0); pass 2 (b61_hd.dev2, the
  registered rule) 10:47:36-10:51:40 (1,152 jobs). Files pulled back MD5-identical (results/b61_dev1.json 232545ab...,
  results/b61_dev2.json 37454616...; committed gzipped).
- **Dev choices** (results/b61_choice.json, b61_hd.choose over pass 1 + pass 2). Dev N* (cells per stored pair; k1 / k2 /
  k4 / e05 / k8 / k16 / k32): H-SBC 32 / 32 / 45 / 54 / 91 / 362 / 1,448; H-SIC 32 / 32 / 45 / 54 / 91 / 304 / 1,448;
  H-HAD 76 / 76 / 91 / 91 / 128 / 362 / 2,048 (U's dev N*: 76 / 76 / 76 / 91 / 128 / 431 / 1,722). lam at dev N*:
  H-SBC 0.7 / 0.7 / 0.7 / 0.5 / 0.5 / 0.5 / 0.35; H-SIC 0.7 / 0.7 / 0.5 / 0.5 / 0.5 / 0.5 / 0.35.
- **Observations on dev (reported; nothing changed):** (1) at k32 the chosen lam, 0.35, is the lower edge of the
  registered grid. (2) At large budgets B38's shards (<= 8,192 cells) hold few pairs (a 6,000-pair store at B = 4,096 has
  3,000 shards of about 2 pairs, so Lb = round(2 / lam) is 2-6), and the larger lam lose share there (pass 1, k32 at
  B = 4,096: 1.000 / 0.982 / 0.938 / 0.733 at lam 0.35 / 0.5 / 0.7 / 1.0; k8 at 4,096: 1.000 / 1.000 / 0.998 / 0.963). A
  smaller lam or longer shards for the sparse store might lower its N* at k16 / k32; neither is run (the registered grid
  and geometry stand; stated as a limitation in the note after the test).
- **Test plan** (results/b61_plan_test.json, b61_hd.plan): 92 (arm, design, budget) entries, 291 evaluations per test
  store (14,550 store evaluations): H-SBC / H-SIC budgets 8 ... dev N* + 4 at the seven i.i.d. and four natural-layout
  damage conditions, H-HAD dev N* +- 4 (i.i.d.). Runs on the box (44 workers). Then the extension rule.
- **Verdict script**: the bootstrap in b61_verdict.py was vectorized (per-store count matrices; its point N* is asserted
  equal to b58_verdict.nstar_counts for every arm, U and table). The script and b61_check.py's N* rule were exercised on
  fabricated rows only (all failures set by hand; no store read); the U and table numbers they print are B59's
  (U 76 / 76 / 76 / 91 / 128 / 431 / 1,722; best table 19 / 19 / 27 / 38 / 54 / 215 / 724; damage best non-HD table
  128 / 128 / 2,896 / 609).
- Compute note 1's header says 10:38; it was committed at 10:36.
