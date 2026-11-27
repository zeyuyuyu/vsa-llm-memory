# B69 compute note 4 (15:41 machine clock; test finished; no test share or requirement has been read)

- **Test** ran once on the box, 15:31:38-15:39:19 (b69_box.sh run, 60 workers, results/b69_plan_test1.json, 1,016
  jobs, 8,056 store evaluations on the 32 test stores). The only lines read are job-completion counts and the steps
  log.
- **Extension** (b69_ext.sh, B38's rule run mechanically right after the test): round 1 requested 0 evaluations (it
  prints only the count), so the test is bracketed everywhere; results/b69_plan_x1.json is empty.
- **No cut**: the whole registered test (k1, k4, k8, k16; N = 256, 1,024, 6,000, 24,000; all seven memories) finished
  before the registered 17:05 limit.
- Next: copy results/b69_test1.json.gz back (MD5 of the decompressed file compared with the box's), run
  b69_vocab.py verdictB (committed code, with compute note 2's label fix), then the checks C1-C4 (b69_check.py,
  committed e587f61) and the note after the test.
