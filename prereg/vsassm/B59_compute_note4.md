# B59 compute note 4 (03:09 machine clock): part 1 superseded (LAM defect); part 2 continues as registered; damage tier reinstated

**Part 1 is superseded.** At about 03:07 the coordinator reported a defect in B49's least-squares memory (LAM), on
which every part 1 comparison depends: b49_common.LamFeat computes its "pca" basis on the already-whitened codes, so for
prep = whiten the "principal" basis is not principal (overlap 0.20 with the true top-104 eigenvectors on Qwen3-8B); two
weaker choices add to it (V - 1 target columns; per-column 32-bit scales). A new pre-registered run (B60) redoes the
store - LAM and store - BEST comparisons with a corrected LAM on all encoders.

This note could NOT be written before part 1's test output was read: part 1 had already finished. Its test ran
02:42-02:52 (81 jobs, all rc = 0), its verdict was computed and read at 03:03, and its result, checks and sample
reading were committed at 03:04 (commit 3c676ef), before the message arrived. Those results
(results/b59_verdict.{txt,json}, results/b59_part_*.json, the part 1 lines of results/b59_check.txt,
results/b59_sample_reading.txt) are hereby **marked superseded**: every quantity that involves LAM (store - LAM; store
- BEST, because BEST's dev choice includes LAM and LAM is the best of ten in most cells; LAMRF, which uses the same
ridge code with V - 1 columns and per-column scales) must not be used. Quantities without LAM (the store's own
accuracies per map, store - BEST36) are not affected by this defect but are not to be cited from B59 either; B60 owns
the map comparison. Part 1's predictions and its bar B1 are not evaluated as findings.

A post-test diagnostic of part 1 (the map pipeline re-run with the registered seeds on Granite dev data, started on
the box at 03:05) was stopped at 03:07 unfinished (own processes only, killed by PID / non-self-matching patterns); no
output of it was read. No part 1 job is running.

**Part 2 has priority and continues as pre-registered** (it involves no LAM). Tiers T1-T4 finished at 03:08 (T4
03:07:08, P-BCH T3 03:07:32). No part 2 test output has been read (only job counts in the logs).

**Damage tier reinstated.** Compute note 3 cut the damage conditions (T5) on the forecast that T1-T4 could not finish
before 03:40. That forecast was wrong: they finished at 03:08, so the registered rule ("only if T1-T4 finish before
03:40") requires T5. It runs now: failed lines and stuck clusters, natural layout, k4 and k8; each code family with its
matched i.i.d. dev designs at the matched i.i.d. dev N* +- 4 grid steps (results/b59_plan_dmg.json: 19 (family,
design, budget) entries, 2,100 store evaluations), then B38's extension rule (b58_store.extend, unchanged); U = B38's
V-last rows for the same conditions. If T5 has not finished by 04:00 it is cut by the registered time rule and
reported as not run (no partial N*).

Two operational events, both before any part 2 output was read: (1) a P-LDPC T2 run started at 02:47 without the
family filter (it would have repeated P-BCH T2) was stopped after about 10 seconds, its output deleted, and restarted
with --only P-LDPC; (2) P-BCH T3 started locally at 02:42 ran too slowly (6 of 300 jobs in 14 minutes) and was stopped
at 02:58; it was run on the box 02:55-03:08 (results/b59_test_T3_bch.json from the box only).
