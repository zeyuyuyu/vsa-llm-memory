# B52 compute note 3 (2026-11-20 about 04:08 UTC; deadline; procedure only, no bar, rule or prediction changed)

Part (a), qwen8L27 x C: runs take 7-17 minutes each on the shared box CPU (load about 100 of 128), so three seeds in one
process would not finish before the deadline. (1) The finals are queued as one process per seed
(out/final_seed/<cell>_<fam>_s<seed>.json, same code, configuration and seed values) and merged into B42's layout
(out/final/<cell>_<fam>.json, runs in seed order 42 / 123 / 456, side files keyed by seed) by `b52_merge_seeds.py`
before the second scorer and the verdict (`b52_a_post.sh`). (2) Four more CPU lanes (6 threads each, nice 10) serve the
same queues. (3) `b52_verdict_a.py` takes `--cells` and reports only qwen8L27_C (compute note 2); a misplaced `global`
statement was fixed (the file did not compile). No test output of any B52 cell had been written when this was done.
Part (b) finished its test run at 03:49 UTC (verdict generated mechanically); its float64 re-derivation check is running.
