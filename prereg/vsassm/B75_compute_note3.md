# B75 compute note 3 (00:45 machine clock; development done, choices committed with this note, BEFORE the test is launched; no test output exists)

- **Development** ran once on the box, 23:38:28-00:36:53 (3,000 jobs, 0 failures, 48 workers; the last H-SBC-J jobs at
  x16 and lam 1.4 took up to 25 minutes each). The full output (results/b75_dev.json, MD5 f15e8785..., 12.8 MB) stays
  on the box; committed here: results/b75_dev_compact.json.gz (the same rows with each failed-key list replaced by its
  length, MD5 3d817ef9..., fetched in verified pieces), results/b75_choice.json (bf3444c2...) and
  results/b75_plan_test.json (b3fddec7...), both computed on the box by `b75_strong.py choose` / `testplan` and
  recomputed on this machine from the compact file: byte-identical.
- **Development N\*** (cells per stored pair, 12 dev stores; design at dev N*): U-LS 54 / 54 / 64 / 108 / 431 / 1,722
  at x1 / 2 / 4 / 8 / 16 / 32 (U's dev 76 / 76 / 76 / 128 / 431 / 1,722); U-J free 32 / 64 / 215 and charged 45 / 91 /
  304 at x1 / 8 / 16 (T = 3, 3, 2); H-SBC-J free 23 / 54 / 181 and charged 32 / 76 / 256 (T = 2; H-SBC's dev 32 / 91 /
  362); O2-BCH 27 / 27 / 32 / 152 / 609 / 2,048; O2-LDPC 27 / 27 / 32 / 76 / 256 / 861 (B63's O-LDPC dev 32 / 32 / 45 /
  91 / 304 / 1,218). These are development numbers only.
- **Test plan**: 168 (memory, parameter, rate, budget) entries, 234 evaluations per store, 2,520 jobs on the 50 RULER
  stores. Estimated from the development job times on the box: about 26 CPU-hours, of which H-SBC-J at x16 about 11.
- **Launch** (b75_test_chain.sh, committed with this note): the registered plan run once, split by memory and condition
  only (the rows are those of a single launch): R1 = U-LS, O2-BCH, O2-LDPC, U-J (24 workers) concurrently with R2a =
  H-SBC-J at x1 and x8 (24 workers), then R2b = H-SBC-J at x16 (24 workers); after R1, F = (D) on the fresh stores.
- **Cuts (registered order; decided now, before any test output exists).** (D) for the joint decoders (U-J, H-SBC-J)
  is cut: at about 20 CPU-hours more it cannot fit the time box (until 03:26) together with the RULER test, the
  determinism re-run and the write-up. (D) runs U-LS, O2-BCH and O2-LDPC at x1, x8, x16 only. Mechanical clock rules:
  if R2b (H-SBC-J at x16) has not finished at 02:05 machine clock, it is stopped by its exact PIDs and H-SBC-J at x16 is
  reported as cut (its partial rows are not read); if F has not finished at 02:10, it is stopped and (D) is reported as
  cut. R1 and R2a are not cut.
