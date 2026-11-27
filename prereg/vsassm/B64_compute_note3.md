# B64 compute note 3 (test ran once; before the verdict is computed and before any test output is read)

Written 2026-11-23 about 03:33 (machine clock).

1. **Test.** b64_out/plan_test.jsonl (350 jobs, md5 d0f39a63...), run once locally, 8 workers (--byenc), 03:27:46 to
   about 03:32; 350 of 350 job files, no FAILED line, no traceback. Code = b64_run.py md5 4e7708d0... (the committed
   2781ced). Only the job log lines (job name, seconds, row count) have been looked at.
2. **No cut.** Every registered encoder, region (R1, R1-V2, R1-V16), accounting and arm is computed; the time box
   (to 05:04) is not binding.
3. **Checks run before the verdict:** b64_check.py disjoint (does not read test predictions): 0 string overlaps, 0
   locations with raw cosine >= 0.9999 to a dev / test code on every encoder and set (max 0.956-0.999); the
   accent / case-normalised string match is 'Rio' (training label) vs 'Río' on the language models, whose codes differ
   there (cosine < 0.9999), and it was removed by code equality on the sentence encoders before the pre-registration.
4. Next, in this order: b64_verdict.py part (7 encoders) and panel; b64_check.py scorer, bits, rederive; sample
   reading; note after the test.
