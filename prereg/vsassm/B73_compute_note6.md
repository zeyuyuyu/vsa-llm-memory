# B73 (equal bits) compute note 6 (about 22:00 machine clock; after the test output was read; no rule or number changed)

- The box's post-test steps finished at 21:43 (b73_box_post.sh): the four per-encoder part files, the bits audit
  (4,200 rows, 0 / 0 / 0) and the determinism re-run (24 files identical) were written; the panel step then crashed while
  writing its JSON (TypeError: a numpy int64 in a claim count, from the margin rule's factor-of-two branch, which the
  sentence encoders had never reached), and the box's plain-loop scorer, which read that JSON, failed with it.
- Fixes (reporting code only): b73_verdict.py casts the claim counts and the margin flag to int / bool; b73_check.py's
  scorer compares the raw files against the per-encoder part files (the verdict's only input) when they exist. The part
  files were downloaded (MD5-checked tarball, 5b1f82bf...), the panel re-run locally on all seven part files, and the
  fixed scorer re-run on the box: 132 region statistics, max |difference| 1.0e-17. No statistic, label, bar or rule
  changed; the verdict text is the one the registered code produces from the same part files.
- Box logs copied to b73_out/box_logs/ (GPU ledger, code MD5s, unit completion logs).
