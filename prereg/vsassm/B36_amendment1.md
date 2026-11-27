# B36 amendment 1 (before any dev result of the registered grid was written or read)

Date: 2026-11-18, 23:35 box clock. The registered dev grid had run for about three minutes (from 23:32) and written no line
(b36_out/dev_{qwen8,e5}.jsonl empty) when it was stopped.

Change: the plurality value of a bucket (CM "plural") or of a distinct signature (SF) broke ties toward the smallest
value index. With crowded buckets (e.g. 16 facts in a 16-cell PQ bucket at V = 16) ties are frequent, and summed over
many tables the rule biases every vote toward low value indices, which handicaps the two families (found while
validating the independent checker on one dev store; dev split only). Ties are now broken uniformly at random:
argmax of count + U(0, 0.5), the jitter drawn from seed ('tie', store tag, family, hash, k, table) over the sorted
distinct signatures (b36_common.tie_argmax; the checker re-implements it). Nothing else changes: grids, accounting,
choice rule, bars and predictions stay as registered. The score-level rule (ties between summed votes go to the store's
plurality value, then the smallest index) is unchanged.

Checker validation (dev store e5, M = 256, V = 16, rep 0; b36_check.py functions against b36_run.run_store for two
configurations of every family at 1, 4, 16 bits per fact): agreement 1.000 for SF, CM, BF, FBF and SDM, before and
after the change (partial SF tables are not re-derived, as registered).
