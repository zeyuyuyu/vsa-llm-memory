# B70 compute note 4 (test ran once; written before the verdict is computed and before any test output is read)

Date: 2026-11-23, about 17:22 (machine clock).

- **Test, NLMSc:** b70_out/plan_test_nlms.jsonl (210 jobs, md5 8c3783a1; the NLMSc lines of plan_test.jsonl, md5
  4489e912), run once on the box (~/b70_run, b70_run.py md5 037c173d and the 7 choice files md5-identical to the
  committed ones), 48 workers, 17:14-17:19, 210 / 210 job files, 0 failures; runner pid 4004892 exited on its own.
  Job files pulled back as a tar (md5 checked on both sides).
- **Test, SDMt (the 4 language models):** b70_out/plan_test_sdmt.jsonl (95 jobs, md5 a5259ef4), run once locally
  (where its dev ran; note 3), 6 workers, one thread each, nice 5, 17:13-17:20, 95 / 95 job files, 0 failures.
- **Cut (registered order, note 3):** SDMt and BEST7 on e5-base-v2, bge-small and MiniLM-L6 (dev reproduction failed).
  Nothing else is cut: every R2 cell of every encoder has STORE, VSAPcX, VSAR, LAMc, NLMSc, BEST36, BESTc, LHI, IDX and
  EXACT; SDMt and BEST7 on the 4 language models. Predictions involving SDMt / BEST7 (Q5, Q7) are scored on the language
  models only.
- What I have read of the test runs: job counts, run times and failure counts only.
- Code changes after the dev choices, before the test output: b70_verdict.py and b70_check.py skip the cut arms on the
  sentence encoders (no statistic or rule changed).
- Time: prereg 16:55, dev 16:58-17:10, choices 17:13, test 17:13-17:20 (2.5-hour box from 16:43). No GPU, no paid API.
