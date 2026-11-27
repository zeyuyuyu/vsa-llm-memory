# B42 note after the arena-C test run (2026-11-18 about 23:20 UTC; no design change, no number replaced)

1. **Recomputed vs logged accuracies (ties at the threshold).** The verdict recomputes every accuracy from the saved
   per-instance scores, which are stored to 6 decimals, against the saved training threshold, which is not rounded. For
   snapped codes, many chains end with two identical codebook rows, so the score is 1 up to float error and many test
   scores sit exactly at the threshold (up to 500 of 500 for Qwen3-8B st). There rounding decides the side, and the
   recomputed and logged accuracies differ in 7 of 48 arm-seed cells, by at most 0.042 (Phi-2 st seed 123: logged
   0.516, recomputed 0.558; Qwen3-8B gumbel seed 123: 0.534 / 0.572; the others <= 0.010). The verdict reports the
   recomputed number, as registered, and flags the arm ("RECOMPUTED != LOGGED"). Every flagged arm FAILS under either
   number; the largest flagged arm mean is 0.515 recomputed and 0.498 logged (Phi-2 st). The second scorer rounds its float64
   scores the same way and agrees with the recomputation on every instance.
2. **Run-to-run spread of the CPU finals.** The six key-value finals rerun for the second scorer (compute note 2)
   reproduced their originals exactly for the three hard clean-up runs, but not for the three straight-through runs
   (Phi-2 dkvb, Phi-2 and Qwen3-8B dkvbor st): multi-threaded CPU gradients are not bit-reproducible, and the
   trajectories drift apart. The reruns' test means differ from the originals by 0.020 (Phi-2 dkvb: 0.487 vs 0.507),
   0.014 (Phi-2 dkvbor st: 0.733 vs 0.719) and 0.011 (Qwen3-8B dkvbor st: 0.730 vs 0.741). The verdict keeps the
   originals; for these three arms the second scorer's chain re-derivation is of the rerun (agreement 1.000), and the
   original is checked by recomputation only.
3. **Hand-reading** (b42_out/handread_{phi2,qwen8}_C.txt; the label-free arm with the highest selected dev accuracy:
   Phi-2 dkvb, Qwen3-8B cw; seed 42; 12 failures each): in 23 of 24 failures the hop-1 clean-up already picks the wrong
   level-0 filler, i.e. the query's role and the same role in the context (identical hex string) do not get matching
   codes; the hex / decimal alias at hop 2 is never reached. In the Phi-2 dkvb cases the query and the context role
   share 0-6 of their 8 head keys.

# Addendum after the arena-A/B test run (2026-11-18 about 23:45 UTC; no design change, no number replaced)

4. **Scope.** A and B ran in full before the cut-off (tuning on GPUs 22:26-23:15 UTC, finals on GPUs 23:15-23:31 UTC).
   Total compute: 3.96 A800 GPU-hours (tuning, A/B finals, smokes) plus 28.4 CPU core-hours (the arena-C finals and the
   six reruns, compute notes 1-2). No PACE, no paid API.
5. **Gate G1 fails in arena A** (dev): the gumbel code path with the identity loss reaches 0.522 (Phi-2) and 0.512
   (Qwen3-8B) against the naming head's 0.976 and 0.800; st with the identity loss passes on Phi-2 (0.992) and fails on
   Qwen3-8B (0.538). G1 passes in B and C on both encoders. Diagnosis (b42_g1diag.py, training split, after the test
   run, exploratory): through the relaxed (gumbel) or straight-through code the answer loss sends the head a gradient
   10-20x the identity loss's (Phi-2 A gumbel: 28.6 vs 2.5 at initialisation, 48.8 vs 5.3 after 200 steps), and
   gradient clipping at 1.0 then scales both by 0.02-0.03. In C the read-out scale and bias have their own learning
   rate (0.1), the scale falls (20 -> 12.5 in 200 steps) and the identity loss still drops (4.31 -> 2.50); in A the
   published optimiser keeps scale and bias in the single group at 3e-4 (they stay at 20 / -4) and the identity loss
   barely moves (5.48 -> 5.35). The B31 naming head never has this conflict: its snapped codes pass no answer-loss
   gradient into the head. Reading: in arena A the gumbel / st arms are not shown to be able to carry identity even
   with labels, so their arena-A failure cannot be attributed to the missing labels alone. Arena A is secondary; the
   arena-C bar is unaffected (G1 passes there on both encoders). P1 is NOT HELD.
6. **Ties in A and B.** Recomputed vs logged accuracies differ in 7 of 48 arena-A cells (max 0.017) and 1 of 48 arena-B
   cells (0.008), for the same reason as item 1.
7. **Arena B (no headroom).** Most label-free families sit at the floor (Phi-2 0.947-0.960 against 0.956; Qwen3-8B
   0.983-0.997 against 0.994), but the Phi-2 key-value bottleneck reaches only 0.726 (train 1.000: it memorises), and
   post-hoc centring and ZCA whitening of the binding-loss map HURT in B (Phi-2 0.961 -> 0.848 and 0.595; Qwen3-8B
   0.996 -> 0.993 and 0.904): the B solution of that map lives partly in the common direction that centring removes.
   P10 NOT HELD.
8. **Consistency check of the rerun binding-loss map on GPU:** in A and B (GPU finals) the plain reading reproduces
   B31's per-seed test accuracies exactly (A 0.463 / 0.481 / 0.443 and 0.489 / 0.497 / 0.488; B 0.954 / 0.962 / 0.966
   and 0.996 x 3). In C (CPU finals) it is within 0.028 of B31's per seed (Phi-2 0.444 / 0.462 / 0.464 vs 0.460 / 0.474
   / 0.470; Qwen3-8B 0.466 / 0.464 / 0.462 vs 0.462 / 0.484 / 0.490), as compute note 1 anticipated.
