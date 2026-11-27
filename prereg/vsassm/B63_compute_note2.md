# B63 compute note 2 (02:54 machine clock; dev choices made; committed before any test run)

- **Dev pass 1** 02:33:38-02:43:42 (672 jobs, rc 0); **dev pass 2** 02:45:51-02:50:00 (1,536 jobs, rc 0), both on the
  box with the code of d4cea81 (MD5-identical). The test stores' copy on the box finished after dev pass 1 started; all
  ten files (code and both store files) are MD5-identical to the local ones.
- **Choice** (b63_online.choose over both passes, run on the box): results/b63_choice.json; re-running choose and plan
  locally from the fetched dev files gives byte-identical files (MD5 bae1f324... and 27f04aa2...).
- **Dev N*** (12 dev stores, 1,254 probed keys; cells per stored pair, k1 / k2 / k4 / e05 / k8 / k16 / k32):
  O-BCH 32 / 32 / 45 / 64 / 181 / 724 / 2,435 (f at dev N*: 16 / 16 / 14 / 12 / 14 / 12 / 12);
  O-LDPC 32 / 32 / 45 / 64 / 91 / 304 / 1,218 ((f, R) at dev N*: (12, 9/10) / (12, 9/10) / (16, 3/4) / (16, 1/2) /
  (14, 1/3) / (12, 1/4) / (16, 1/4)).
  For reference (dev, not used by any rule): H-SBC dev 32 / 32 / 45 / 54 / 91 / 362 / 1,448; B38's dev E-BCH 19 / 19 / 27
  / 38 / 108 / 431 / 1,722 and E-LDPC 23 / 23 / 27 / 38 / 54 / 215 / 724.
- **Lost insertions on dev**: 1 entry in 48 (store, f) builds (dev/mv/1, 23 pairs, m = 7 buckets, f = 12); the tiny
  stores (9-23 pairs) have few buckets, so 500 kicks can fail below load 0.95. This is the registered online behaviour
  (the entry's key reads wrong); prediction S6 (no lost insertion on any test store) may fail on the tiny test stores.
- **Test plan** (b63_online.plan): results/b63_plan_test.json, 85 (arm, design, budget) entries, 125 evaluations per
  store (dev N* +- 4 grid steps per arm and condition, i.i.d.). Test to start on the box next (60 workers); the cut rule
  (prereg: 04:00) is not expected to bind.
