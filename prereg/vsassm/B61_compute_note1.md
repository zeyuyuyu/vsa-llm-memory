# B61 compute note 1 (10:38 machine clock; development restarted; no dev or test output has been read)

- **Header time.** The pre-registration's header says 10:33 machine clock; it was committed at 10:32 (c2b548b) and
  stamped and the proof committed at 10:33 (584e991).
- **Dev pass 1, first start stopped.** Dev pass 1 started on the box at 10:33:56 (b61_box.sh dev1, 40 workers, code
  MD5-identical to c2b548b). At about 10:36, while the check code (b61_check.py, written after the pre-registration
  commit) was being validated **on synthetic pairs only**, its independent numpy scorer disagreed with the committed
  H-SIC read on one synthetic cell (lam 0.5, B 48, k8). Cause: H-SIC's scores are sums of floating-point weights
  log((1 - e) / q) and -log((1 - q) / e) accumulated in candidate-specific order (and the exact-pruning bound compared
  in floating point), so candidates with equal counts of unexplained active and inactive cells could differ by rounding;
  ties were then broken by rounding instead of by the registered "argmax first index". **Fix:** both reads compute a
  candidate's score from its integer counts as na A - ni Cc (one formula, so equal counts give equal scores; the
  pruning bound (na + remaining) A - ni Cc is the same monotone formula, so pruning stays exact). The H-SBC read
  (A = 1, Cc = 0) is unchanged in its decisions. After the fix the committed read and the independent scorer agree on 7
  of 7 synthetic cells for both H-SBC and H-SIC (H-HAD by float64 dot products: 1 of 1). The registered rule (weights,
  argmax, stop rule) is unchanged.
- The stopped run was killed by PID (bash driver, main process, its pool workers by parent PID); its log (moved to
  logs/dev1_stopped.out) showed only job-completion lines (store name, seconds, count); its partial output file was
  deleted unread. Dev pass 1 restarts with the fixed code (b61_hd.py committed with this note).
- Check code b61_check.py committed with this note (second scorer, bits audit, N* second rule, sample reading).
