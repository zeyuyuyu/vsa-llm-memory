# B63 compute note 3 (03:00 machine clock; written before the verdict is computed; no test output has been read)

- **Test** ran once on the box, 02:53:27-02:57:42 (b63_box.sh test, results/b63_plan_test.json, 900 jobs, rc 0, output
  results/b63_test1.json, MD5 6fcedf4d...). The only lines read so far are job-completion counts (done / total / error
  count / the steps log).
- **Extension** (b63_online.extend, the registered rule, run mechanically on the box): 0 additions (it prints only the
  number), so no extension run; results/b63_plan_x1.json is empty.
- **No cut**: the test finished at 02:57, before the registered 04:00 limit; every registered condition (k1-k32, e05)
  is in.
- **Insertion audit** (b63_check.py insert) was run locally on the 50 test stores while the test ran; its output (a
  structural check of the cuckoo builds, which shows lost insertions) has not been read and is read with the checks.
- Next: fetch the test output, run b63_verdict.py (committed in 4cbdcec, unchanged), then the checks.
