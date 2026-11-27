# B73 (equal cells) compute note 2 (20:00 machine clock; test finished; written before the verdict is computed and before any test row is read)

- **Test ran once** on this machine, 19:29:58-19:59:30 (b73_cells.py run, 8 worker processes, single-threaded BLAS,
  plan b73_out/cells/plan_test1.json MD5 9f61297e..., code = 8deea4e), 1,830 jobs, 9,450 store evaluations, every job
  finished (no traceback or error line in the log). Summed job time 3.93 CPU-hours. Output b73_out/cells/test1.json
  (MD5 5fbccf0a...), committed gzipped with this note as results/b73_cells_test1.json.gz (MD5 fadef32d...).
- Read so far: job-completion lines (store, memory, seconds, count) and the summed job seconds only.
- **Extension** (b73_cells.py extend, B38's rule, one round, run mechanically): 0 additions (only the count was printed),
  so no extension run; b73_out/cells/plan_x1.json is empty.
- **No cut**: k1, k8 and k16 are all in; the 21:45 cut time did not bind.
- Next: b73_cells_verdict.py (committed 7fcccdc, unchanged), then b73_cells_check.py recount / det / sample, the sample
  reading and the note after the test.
