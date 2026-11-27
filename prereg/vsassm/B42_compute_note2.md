# B42 compute note 2 (2026-11-18 about 22:50 UTC; no design change, no bar or prediction changed)

**A bug in the second scorer's input, found by the second scorer.** For the key-value arms (dkvb and the labelled
diagnostic dkvbor), the side file that keeps the dev-selected epoch's learned values for `b42_indep.py` stored
`values.detach().cpu()`. On the CPU finals (compute note 1) `.cpu()` returns the live parameter itself, so every
epoch's snapshot was the same tensor and the side file holds the LAST epoch's values. The second scorer then rebuilt
those arms' chains with the wrong values (per-instance agreement 0.51-0.72 on the first shards; the vq, gumbel and st
shards agree 1.000, max score difference <= 1e-6). The main results are not affected: scores, thresholds and accuracies
are computed inside every epoch from the live model.

**Fix and check.** `b42_arms.py` now clones every snapshot (values, codebook, table). The six affected arena-C finals
(Phi-2 and Qwen3-8B: dkvb, dkvbor hard, dkvbor st) are rerun on the CPU with the fixed code, same arguments and seeds,
into out/final_rerun/. The rerun must reproduce the original per-instance test scores exactly (same device, same
seeds); the second scorer then uses the rerun's side files. If a rerun does not reproduce its original, the verdict says
so and those arms keep only the recomputation check. The verdict reads the original finals in out/final/ in every case.
A and B finals run with the fixed code from the start.
