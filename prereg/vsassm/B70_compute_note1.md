# B70 compute note 1 (during dev; no dev or test output of NLMSc / SDMt read)

Date: 2026-11-23, about 17:00 (machine clock). Pre-registration committed d9f66ef, OpenTimestamps proof committed
f8844cb, both before any B70 job.

- Placement: the A800 box, fresh directory ~/b70_run (b70_box.sh setup). Code copied from this repository; md5 on the box
  equal to the local files: b32_common d6e1231b, b36_common 39dc8029, b49_common f2228be9, b60_run 520af3db, b62_run
  ca210944, b64_run 4e7708d0, b70_run f86152c6; results/b70_region.json 85062efd; plan_dev.jsonl 241648c8. Panel codes
  copied from ~/b66_run's B49 copies, all 9 md5 ok.
- Dev plan (b70_run.py plan dev): 216 jobs = NLMSc 144 (M = 256: 9 settings x 6 stores x 2 preps; M = 1,024: 6 x 3 x 2)
  + SDMt 72 (9 x 6 + 6 x 3), every configuration of the registered grids at the setting's R2 budgets in {32, 64} and at
  16 bits per fact (reproduction check). Started 16:58 on the box, 48 workers, nice 5, one thread per process, one ssh
  command in the background (pid 3992215 on the box). No retry loops.
- Local: b70_verdict.py and b70_check.py written during dev, not run on anything.
- No cut.
