# B64 compute note 1 (during dev; no dev or test output read)

Written 2026-11-23 about 03:24 (machine clock), before the dev choices are made and before any test store is answered
by SDMt.

1. **Pre-registration time.** The header says "about 03:20"; the pre-registration was committed at 03:14:47 (44cbfbb)
   and its OpenTimestamps proof at 03:14:55 (ddcdade). The commit times are the record.
2. **Placement.** Local CPU (the registered first choice): the local load was 5-6 of 16 cores; the A800 box had load
   20-58 and /data0 at 100% (7.7 GB free), so nothing was placed there. Dev plan b64_out/plan_dev.jsonl (126 jobs = 7
   encoders x V 2 / 16 x 9 dev stores, every configuration at 4 / 8 / 16 bits per fact, both accountings; md5
   c5dc108d...), started 03:15:06, 8 worker processes, one thread each, nice 10; code = the committed b64_run.py (md5
   90cbe5e9...).
3. **Verdict and check code** (committed at 03:22:41, 1803029) were written during the dev run and exercised only on
   fabricated rows (SDMT := B36's SDM test predictions, BEST7 := BEST36 on Qwen3-8B, which reproduces B49's VSAR -
   BEST36 and B53's VSA - BEST36 exactly) and, for the independent float64 re-derivation, on two TRAINING-split stores
   (16 configuration-budget pairs: predictions, N and bits identical to b64_run).
4. **Observation from the training-split exercise (not a result):** at V = 2 and 16 bits per fact some SDMt
   configurations reach the whole location set (N = 5,621 or 2,818) below the budget, so their budget is partly
   unused (the location set, not the budget, binds). This is a property of the registered arm (locations = the
   training split's codes); the verdict reports N for every chosen configuration.
5. No cut of the registered design so far. Next: B36's choice rule (b64_run.py choose) for all 7 encoders, committed
   before the test plan is written.
