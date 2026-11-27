# B73 (equal cells) compute note 1 (19:34 machine clock; test running; no test output has been read)

- **Header time.** The pre-registration's header says "about 19:35"; it was committed at 19:29:27 (8deea4e) and stamped,
  the proof committed at 19:29:34 (9c2e552). The header was written ahead of the commit; nothing else differs.
- **Fresh stores** generated after the proof commit (b73_cells.py stores test, code MD5 6178a0b2... = 8deea4e):
  b73_out/cells/test_stores.json.gz (MD5 31ba48c5...), 50 stores, 92,094 pairs (RULER: 92,080). Not committed (2 MB;
  regenerable from the committed generator and seed tag).
- **Plan** b73_out/cells/plan_test1.json (MD5 9f61297e...): 27 (memory, design) entries, 189 evaluations per store, 1,830
  jobs, 9,450 store evaluations.
- **Test** started 19:29:5x on this machine (b73_cells.py run, 8 worker processes, nice 5, OMP / OPENBLAS / MKL / NUMBA
  threads = 1; log b73_out/cells/test1.log, output b73_out/cells/test1.json). Only job-completion lines (store, memory,
  seconds, count) are read while it runs.
- **Leakage check** (b73_cells_check.py leak, a property of the stores only, no result row read): 0 of 91,977 fresh key
  strings and 0 of 92,056 fresh value strings equal any key or value string of the 50 RULER test stores, the 12
  development stores or B69's sweep stores (results/b73_cells_leak.json). After lower-casing and deleting every
  non-alphanumeric character, 1 to 10 fresh strings per reference set coincide (e.g. the fresh key 'crazy/hedge' and a
  RULER 'crazy-hedge'): common adjective-noun pairs drawn independently from two word lists; every code is generated
  from the exact string, so they share nothing.
- **Verdict and check code** (b73_cells_verdict.py, b73_cells_check.py) committed with this note, before any test output
  exists in full; both were exercised only on fabricated rows on the self-test stores (seed tag b73cells-selftest),
  where the verdict reproduced the paper's N* when fed the paper's curves and the plain-loop recount agreed (0 of 27
  differences).
