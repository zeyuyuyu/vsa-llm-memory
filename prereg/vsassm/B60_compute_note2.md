# B60 compute note 2 (test running; the verdict has not been run)

Written 2026-11-21 about 09:06 (machine clock), before b60_verdict.py has been run on any test output.

1. **Test start.** Dev choices committed at 08:19:04 (899b8fa). The first attempt to start the box test at about
   08:20 timed out while the plan files were being uploaded (slow link; partial plan files, no runner started, 0 jobs).
   The plans were sent again compressed, md5-checked on the box against the local plans (plan_test f3d06bbb...,
   plan_maptest e58a751f..., plan_testR2 e3f8c3f8...), and started at 08:25:21: main test (535 jobs, 24 workers),
   map test (1,460 jobs, 16 workers), R2 test (345 jobs, 8 workers). VSAPc's test reads ran locally from 08:19 (28 jobs).
   Main test, R2 test and VSAPc test finished (no failure or traceback). After the main test finished (08:53) the map
   runner was restarted with 40 workers (313 jobs kept; jobs in flight discarded).
2. **Checks run before the verdict.** The float64 re-derivation and the step5 agreement check were run after the main
   test had finished and before the map test had (no choice remains that they could influence). Their lines are in
   results/b60_check.txt.
3. **Cut (time).** At 09:03 the map test had 496 of 1,460 jobs; at the observed rate (about 17 jobs a minute, the
   remaining jobs being M = 256 stores whose chosen configurations at 32-64 bits per fact use up to 65,536 random
   features) it would have ended at about 10:00-10:20, against the restarted 3-hour budget ending at 11:06. Under the
   registered priority (maps third; their R1 part first, per the design's "at least for R1 on all 7 encoders"), the map
   test of the remaining 964 jobs is cut to the R1 budgets (4, 8, 16 bits per fact): 784 jobs (180 jobs whose cells
   are all outside R1 dropped), plan_maptest_r1.jsonl (md5 e043c18b...), 48 workers, started 09:04:31. Consequences,
   fixed now: map statistics are computed for R1, R1-V2 and R1-V16 only (the registered map regions R0 and RH are not
   reported); the 496 already-finished map jobs keep all their budgets but only their R1 rows are used. The registered
   map, every other arm, every region and both accountings are unaffected. b60_verdict.py is changed accordingly (map
   arms read for R1 budgets only; the map table and P9 use R1, R1-V2, R1-V16) before it is first run.
