# B69 compute note 3 (15:34 machine clock; test running on the box since 15:31; no test output has been read)

- **Dev** ran once on the box: pass 1 15:17:10-15:27:33 (504 jobs, 7,776 store evaluations, every job finished, no
  error line), the registered extension round (b69_vocab.devext: 13 new (design, budget, condition) entries, 39 store
  evaluations; E-LDPC at N = 256 and 24,000, O-BCH at N = 256, 1,024 and 24,000, all 4 steps below a dev requirement
  at the bottom of its window) 15:29-15:30. Choices over both passes (b69_vocab.choose): results/b69_choice.json
  (MD5 e14e192b...), committed in 21644f4 before the test; the test plan results/b69_plan_test1.json (MD5 77fb87f4...,
  96 (memory, N, design) entries, 1,007 evaluations per set of four stores, one of each N).
- **Restricted grids at the dev requirement** (registered as a limitation if at an inner edge): at k4 every E-LDPC and
  O-LDPC choice is R = 2/3, the lower edge of the restricted set {2/3, 3/4, 5/6} and an inner edge of B38's full set
  (1/2 below it); at k16 R = 1/4 (the full set's edge); at k8 1/3 (inside); at k1 5/6 or 9/10. f at the dev requirement
  is 12 or 14 (inside the grid's {12, 14, 16}; 12 is also the full B63 grid's lower edge). H-SBC lam 0.35 at k16 and at
  N = 24,000 k8 (the grid's lower edge, as in B61). The k4 rate restriction can only raise the LDPC tables' k4
  requirement (against the tables); it is reported with the k4 numbers, which are secondary.
- **Test** started 15:31 on the box (b69_box.sh run, 60 workers; 1,016 jobs, 8,056 store evaluations on the 32 test
  stores), the code and plan MD5-checked on the box. The box's load average was 45-100 during dev (other users' and
  agents' processes, not touched); this changes only the wall time.
- The dev result files are being copied back (slow link); a local re-run of choose / testplan on them is compared with
  the box's files by MD5 before the verdict.
- **Check C0** ran on the box 15:18:32-15:19:05 (b69_check.py c0, 2 x 10^8 device-model cells at k = 1). Its result is a
  check, not a comparison output, and was read: no cell flipped at 30 days or at 1 year (95% upper bound 1.5e-8 per
  cell); the smallest read margin |G+ - G-| was 12.7% of G_ON at 30 days (7.2% at 1 year), the mean margin 9.6 (8.7)
  standard deviations.
- **Part (A)** was read after compute note 2 (results/b69_partA.txt). One finding about the registered compact format:
  two niah_multikey_3 test stores each contain one malformed identifier (a 35-character string, not a valid UUID:
  '1178627c-84ad-473b-b5c8-800b254c7a4' in /189443 and 'eebd9a4-24ba-42d4-9dd1-0e4f2e8a1327' in /43865), so by the
  registered per-store rule those two stores fall back to utf8 (288 bits per pair) and the pooled and mk3 compact
  charges equal the utf8 ones. Reported as registered; no rule changed.
