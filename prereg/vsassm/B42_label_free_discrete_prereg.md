# B42 pre-registration: label-free discrete interfaces (VQ, discrete key-value bottleneck, Gumbel-softmax, straight-through arg-max, centred / whitened map) against the naming head, on B31's frozen states

Date: 2026-11-18, about 22:30 UTC (06:30 on 11-19 on the box clock, Asia/Shanghai; US Eastern 2026-11-18 17:30). Written,
with all B42 code, gates, selection rule and verdict generator, before any registered B42 run reads a test label.
What was done before this file (train and dev splits, code smokes) is listed under "Done before registration"; two
code smokes in final mode did score the test split and are disclosed there.

## Why (referee M5)

The paper says a frozen LM can use a VSA memory only once each symbol is named by a supervised naming head (identity
labels from the generator), because maps trained only through the memory's answers collapse (B31: different-symbol
cosine 0.92-0.999) and label-free InfoNCE codes are inconsistent across mentions (same-symbol cosine 0.26-0.83). A
referee: "Label-free discrete bottlenecks are missing: VQ with a commitment loss, Gumbel-softmax or straight-through
keys, a discrete key-value bottleneck (Trauble et al. 2023), and simple centering or whitening are all absent. The
imposed cross-talk is common-mode, so subtracting the codebook mean would remove it." B42 asks whether any label-free
discrete interface reaches the naming head, on the same arenas, encoders, features, splits and scoring as B31. Its
result goes into the paper whatever it shows.

## Scope (coordinator, 2026-11-18 about 21:45 UTC: deliver within 24 h)

- **Primary: arena C on both encoders (Phi-2, Qwen3-8B), three seeds.** C runs first: gates, tuning, selection,
  finals.
- **Secondary: arenas A and B**, same design, queued only after the C finals have started. If the A and B finals are not
  all complete by 2026-11-19 14:00 UTC, the arenas that are not complete are reported as "not run" (their dev tuning,
  if any, is reported as dev only) and no test output of them is read.

## What is reused exactly (B31, prereg/B31_encoder_equalised_baselines_prereg.md)

- **Features:** B31's cached span-mean final-layer states, `/path/to/work/b31/feats/{phi2,qwen8}_{A,B,C}_{train,dev,
  test}.pt` (md5 of the text checked at load, as in B31). Phi-2 fp16 `last_hidden_state` pooled as published; Qwen3-8B
  base, bf16, after the final norm.
- **Splits:** A 2,000 / 500 / 1,000, B 1,000 / 500 / 500, C 800 / 500 / 500 (train / dev / test), as B31.
- **Chains:** `b31_arms.Arm.chain` (A: three hops, clean-up after hops 1-2; B: one hop; C: two hops, clean-up after
  each), d = 2,048, **perm** encoding (value argument permuted, private seed + 7001), **named** convention (every symbol,
  including the query and the candidate, is read from its own span; no codebook row is chosen by an index).
- **Training and scoring:** AdamW (A: one group; B, C: scale and bias at 0.1), batch 8 / 4 / 4, gradient clip 1.0, 10
  epochs, the closed-form training-accuracy-maximising threshold on the final cosine, dev accuracy after every epoch,
  test read at each seed's dev-best epoch. Seeds 42, 123, 456 (tuning: seed 42).
- **Geometry:** the B31 / V-165 cross-talk protocol (first 120 test instances, up to 4 occurrences per symbol, first 40
  symbols; mean cosine between different symbols and between occurrences of one symbol; labels group the vectors for
  this measurement only).
- **References (B31's banked per-instance test outputs, not rerun):** floor F = equalised probe; N = naming head + VSA
  (perm, named); exact symbols + VSA; binding-loss map (perm, named); InfoNCE map (perm, named).

| encoder | arena | F (probe) | N (naming head) | gap N - F | MATCH needs mean >= |
|---|---|---|---|---|---|
| Phi-2 | C | 0.472 | 1.000 | 0.528 | 0.947 |
| Qwen3-8B | C | 0.512 | 0.983 | 0.471 | 0.936 |
| Phi-2 | A | 0.485 | 0.978 | 0.493 | 0.929 |
| Qwen3-8B | A | 0.492 | 0.751 | 0.259 | 0.725 |
| Phi-2 | B | 0.956 | 0.996 | 0.040 | no headroom |
| Qwen3-8B | B | 0.994 | 0.796 | -0.198 | no headroom |

## Found before registration (training and dev splits only)

1. **The published chain gives the answer loss no path to most codes** (`b42_gradpath.py`, Phi-2, binding-loss map at
   initialisation, first 64 training instances; out/gradpath_phi2.json). Fraction of mentions whose code receives a
   non-zero gradient:

   | arena | clean-up | query | hop-1 keys | hop-1 values | hop-2 keys | hop-2 values | hop-3 keys / values | candidate |
   |---|---|---|---|---|---|---|---|---|
   | C | hard (published) | 0 | 0 | 0 | 0 | 0.031 (the winner) | - | 1 |
   | C | straight-through | 1 | 1 | 1 | 1 | 1 | - | 1 |
   | A | hard (published) | 0 | 0 | 0 | 0 | 0.063 (the winner) | 1 / 1 | 1 |
   | A | straight-through | 1 | 1 | 1 | 1 | 1 | 1 / 1 | 1 |
   | B | (no clean-up) | 1 | 1 | 1 | - | - | - | 1 |

   In C the score is the cosine between the final clean-up winner's code and the candidate's code; the arg-max clean-up
   blocks every other path. So in C a map "trained through the memory's answers" with the published chain gets no
   signal on the query, the roles, or the level-0 fillers, and cannot learn that a symbol written as a decimal filler
   in level 0 ("{63}") and as a hex role in level 1 ("[F3F]") is one symbol, which the chain needs at hop 2. B31's
   binding-loss and InfoNCE maps were trained that way. **Every B42 grid therefore includes a straight-through
   clean-up for training** (`--cleanup st`: the forward pass is the published hard chain; the backward pass also goes
   through softmax(cosine / 0.05) over the candidates, so the loss reaches every hop). Evaluation always uses the
   published hard chain.
2. **Gate G0 on dev passes in C:** the B42 code path with the binding-loss map reproduces B31's tuning runs epoch by
   epoch (Phi-2 lr 3e-4 and Qwen3-8B lr 1e-3: max |dev difference| over 10 epochs 0.000; out/g0/).
3. **VQ needs centring on these states:** without it every mention snapped to one code after one epoch (Phi-2 C smoke:
   different / same cosine 1.000 / 1.000, one code used), because the frozen states share one dominant direction. The
   registered VQ centres z by the mean over all training mentions (refreshed every 50 steps) before l2-normalisation
   (smokes: cross-talk 0.02 / 0.02 on Phi-2 C, 0.01 / 0.01 on Qwen3-8B C with input whitening).
4. **Smokes** (1-3 epochs, train + dev; out/smoke/, logs/smoke*.lane*.log): every family's code path ran on Phi-2 C,
   Phi-2 A (vq) and Phi-2 B (dkvb), Qwen3-8B C (vq). Dev accuracies 0.45-0.54 in C for every label-free family after
   1-3 epochs, except the labelled diagnostic dkvbor (surface-form keys, straight-through clean-up): dev 0.708 at
   epoch 2 (train 0.991). Post-hoc centring of the binding-loss map on Phi-2 C dev (G0 run): cross-talk 0.29 / 0.12 at
   epoch 1 (from 0.96 / 0.95), dev 0.49-0.55, no better than without it.
5. **Disclosure: two smokes in final mode scored the test split** (Phi-2 C, seed 42, one epoch): `st` (lr 1e-3, div
   0.1, st clean-up), test 0.476; and the B31-form binding-loss map (lr 3e-4), whose test number was not printed to me
   and is not read. They were run to check the code path that saves test outputs and the second scorer. Neither is a
   registered run, nothing was chosen from them, and their outputs stay in out/smoke/ and are excluded from the
   verdict.

## Arms (label-free: no identity label anywhere; trained only through the memory's answer)

All arms: `b42_arms.py`, one code per mention, perm / named, published chain, the B31 binding-loss map's answer loss
(cross-entropy on scale x score + bias), plus the arm's own label-free terms below.

| key | arm | code of a mention | extra loss / updates |
|---|---|---|---|
| vq | VQ bottleneck | z = W x (x = state, or its ZCA-whitened version), centred by the training mean, l2-normalised; snapped to the nearest code (cosine) of a per-kind codebook (C: one shared codebook); straight-through gradient; evaluation uses the code | commitment beta 0.25 (van den Oord et al. 2017); codebook by EMA (0.99), no gradient; spherical k-means++ initialisation on training mentions; dead-code restart every 50 steps (codes with EMA usage < 0.1 per batch reset to random current encoder outputs) |
| dkvb | discrete key-value bottleneck (Trauble et al. 2023) | x standardised per dimension; H fixed Gaussian random projections (256 / H dims each); per head, the nearest of a set of frozen keys (k-means++ and 20 Lloyd steps on 30,000 training mentions, label-free); each key holds a learned D-vector; code = l2-normalised mean over heads of the selected values | values learned by the answer loss only (lr 1e-3) |
| gumbel | Gumbel-softmax over the fixed codebook | the naming head's architecture (linear head over the span state, arena A/B role mentions read over the 64 roles) without labels; training: Gumbel-softmax sample over the fixed random codebook rows the naming head snaps to, temperature annealed 2 -> 0.1 (exponential over the first 70% of steps); evaluation: arg-max row (snapping) | optional diversity term div x (1 - H(mean softmax over the batch's mentions) / log K) per kind (Baevski et al. 2020) |
| st | straight-through arg-max | same head; training: forward one-hot arg-max row, backward through softmax (Bengio et al. 2013); evaluation: arg-max row | same optional diversity term |
| cw | binding-loss map, centred / whitened | z = W x (x raw or ZCA-whitened, statistics of all training mentions), then code = l2n(z - mean) (global training mean), l2n(z - mean of the instance's own codes) ("subtracting the codebook mean"), or l2n(ZCA(z)); statistics of all training mentions refreshed every 50 steps, no gradient through them | none |
| bind+pc, bind+pz | the B31 binding-loss map, post-hoc centring / ZCA whitening | the B31 map (rerun, its B31 dev-selected lr), evaluated with codes centred (pc) or ZCA-whitened (pz) by the statistics of the training mentions at that epoch; epoch chosen on dev per reading | none (the referee's exact proposal: subtract the mean) |

Diagnostics that use labels (never label-free; reported apart, not under the bar):
- **G1 positive control (dev):** gumbel and st with the naming head's identity loss added (weight 1, lr 3e-4, hard
  clean-up). Passes if its best dev accuracy is within 0.02 of the B31 naming head's best dev on the same cell.
- **dkvbor (arena C only, final, 3 seeds, test reported):** dkvb with oracle keys = the gold surface form (kind and
  symbol: 64 hex roles and 64 decimal fillers are different keys), learned values, hard and st clean-up. It is the best
  any label-free clustering of written forms could do, and says whether the remaining problem is clustering or the
  hex/decimal alias.

## Tuning grids (seed 42, train + dev, 10 epochs; `b42_queues.grid`, in tie-break order)

| family | grid | size per encoder x arena |
|---|---|---|
| vq | clean-up {hard, st} x codes per symbol {1, 4} x input {raw, ZCA} x lr {3e-4, 1e-3} | 16 |
| dkvb | clean-up {hard, st} x keys per symbol {1, 4} x heads {1, 8}; lr 1e-3 | 8 |
| gumbel | clean-up {hard, st} x div {0, 0.1, 1} x lr {3e-4, 1e-3} | 12 |
| st | clean-up {hard, st} x div {0, 0.1, 1} x lr {3e-4, 1e-3} | 12 |
| cw | clean-up {hard, st} x input {raw, ZCA} x code {global centring, per-instance centring, ZCA}; lr = B31's dev-selected binding-loss lr for the cell (Phi-2 3e-4; Qwen3-8B A and C 1e-3, B 3e-4) | 12 |

"Codes / keys per symbol" multiplies the number of symbols the naming head separates per kind (C 64; A 64 roles and
256 per other kind; B 64 / 256). For comparison: the naming head had no tuning (one configuration, B31), B31's
binding-loss map 2 and InfoNCE map 6 configurations. The B42 grids are larger, which favours the label-free arms.

**Selection (`b42_select.py`):** per family, encoder and arena, the configuration with the highest best-epoch dev
accuracy (perm/named); ties within 0.002 go to the configuration listed first. **Finals:** that configuration, seeds
42 / 123 / 456, test at each seed's dev-best epoch (the final epoch is reported beside it). bind+pc / bind+pz: the B31
binding-loss configuration, 3 seeds, the epoch chosen on dev for each reading.

## Measurements (per encoder x arena x arm)

Accuracy per seed (Wilson 95%) and mean (bootstrap over test instances, 10,000 resamples, paired across seeds and
with the B31 references); recovery R = (mean - F) / (N - F) with a paired bootstrap interval; cross-talk (different-
and same-symbol cosine, B31 protocol); for discrete arms, on all test mentions: mention consistency (share of mentions
on their symbol's most common code), code purity (share of mentions on their code's most common symbol), codes used,
and in C the alias rate (share of symbols whose most common code as a hex role equals that as a decimal filler);
clean-up correctness at hop 1 and hop 2 (A, C) from the saved winners. Labels are used for these measurements only.

## Bars and readings (fixed now)

- **Primary bar (arena C):** a label-free arm (vq, dkvb, gumbel, st, cw, bind+pc, bind+pz) **MATCHES** if its mean
  recovery in arena C is >= 0.90 **on both encoders** (the per-seed condition of B27/B31, R_s >= 0.80 in 3/3, is
  reported as a flag). PARTIAL: R >= 0.25 (on an encoder); FAIL: R < 0.25.
- **Answer:** "a label-free interface reaches the naming head" = YES iff at least one arm MATCHES. Seven arms are
  tested; each is dev-selected within its family, and the bar (90% of a 0.47-0.53 gap on two encoders) is far above
  what noise across seven arms could produce.
- **Secondary:** A by the same rule (Phi-2 and Qwen3-8B gaps 0.49 and 0.26); B has no headroom on either encoder (the
  probe is within 0.04 of, or above, the naming head), so B reports accuracy against F and N only.
- **Readings:**
  - YES: the paper must drop "only once each symbol is named by a supervised naming head" for arena C, name the
    interface that suffices, and recast the naming head as one way to supply codes. If the matching arm needs the
    straight-through clean-up, the paper says the coupling needs a gradient through the clean-up, not identity labels.
  - NO, with every arm FAIL: the paper may say that none of the label-free interfaces tested (VQ with commitment loss,
    EMA and dead-code restart; a discrete key-value bottleneck; Gumbel-softmax and straight-through keys over the
    naming head's codebook; centred or whitened maps, trained or post hoc) recovers 25% of the floor-to-naming-head gap
    in arena C on either encoder, give the best recovery with its interval, and use the geometry to say why (e.g.
    centring removes the common-mode cross-talk but leaves the codes inconsistent across mentions).
  - PARTIAL somewhere: the paper reports the recovery and its interval and does not say "label-free codes fail".
  - dkvbor: if surface-form keys with the st clean-up reach high accuracy while label-free arms do not, the missing
    piece is consistent label-free clustering of the written forms; if they do not, even a perfect label-free
    clustering of written forms is not enough in C, because the hex / decimal alias needs identity information or a
    task signal it does not get.

## Predictions (stated so they can be wrong)

- P1 gates G0 and G1 pass in every cell run.
- P2 no label-free arm MATCHES in arena C on both encoders (answer NO).
- P3 no label-free arm MATCHES in arena C on either encoder.
- P4 every label-free arm's arena-C recovery is below 0.25 on both encoders, except at most one (arm, encoder) cell.
- P5 discrete families (vq, dkvb, gumbel, st) in C: alias rate <= 0.20 in every seed.
- P6 trained centred / whitened map in C: different-symbol cosine <= 0.25 and same-symbol cosine <= 0.60 in every seed.
- P7 post-hoc centring changes the binding-loss map's arena-C accuracy by at most 0.03 on each encoder.
- P8 dkvbor in C: st clean-up exceeds hard clean-up by >= 0.10 and hard clean-up <= 0.60, on both encoders (informed by
  the smoke in item 4 above).
- P9 (if A runs) arena A, Phi-2: the best label-free arm is PARTIAL (0.25 <= R < 0.90).
- P10 (if B runs) arena B: every label-free family at or above the floor minus 0.03 on both encoders.
- P11 the second scorer agrees with the main runs on >= 0.99 of instances in every discrete cell.
- P12 total GPU-hours <= 10.

## Checks from other angles (in the verdict)

- **Recomputation:** `b42_verdict.py` recomputes every test accuracy from the saved scores and thresholds against labels
  regenerated by b31_struct and flags any difference from the logged value.
- **Second scorer:** `b42_indep.py`, numpy float64, its own regex parse of the raw text (mention order, hop structure,
  gold answer by a dictionary walk). For every discrete final it rebuilds each mention's code from the saved
  assignments and codebook / values / table of the dev-selected epoch, recomputes the chain and applies the saved
  threshold; per-instance agreement with the main run. For every final it recomputes accuracy from the saved scores
  against the dictionary-walk labels.
- **Hand-reading:** `b42_handread.py`, at least 10 arena-C test instances per encoder where the label-free arm with the
  highest dev accuracy is wrong: raw text, gold chain, which hop's clean-up failed, the code each chain mention got
  (in particular the hex role and the decimal filler of the hop-2 symbol).
- **Literature:** results/b42_literature.txt (VQ-VAE and EMA codebooks, Gumbel-softmax / Concrete, straight-through,
  DKVB, codebook-collapse remedies, unsupervised discrete units vs supervised labels, anisotropy and centring /
  whitening of LM states, binding IDs, learning through HRR), each marked supports / conflicts / no precedent after
  the verdict.

## Compute

A800 box only (`b42_worker.sh`: one idle GPU at a time, < 500 MiB used and no compute process, re-checked after 20 s,
lock in the claim directory shared with the other experiments, never GPU 3, never another process). Runs take 0.5-3
minutes each; C tuning is 104 runs + 6 gate runs, finals 32 runs of 3 seeds; A and B about 2.5x C. Projection 3-6
GPU-hours. Cap 40 GPU-hours. No PACE, no paid API.

## Files

Code: b42_arms.py, b42_gradpath.py, b42_queues.py, b42_select.py, b42_verdict.py, b42_indep.py, b42_handread.py
(written after this file, reading only), b42_worker.sh; reused unchanged: b31_arms.py, b31_struct.py, b31_prep.py,
b27_common.py, b27_arena/. Box: /path/to/work/b42 (out/, logs/, q/). Results: results/b42_verdict.txt,
results/b42_scores.json, results/b42_literature.txt, per-run outputs in b42_out/ (gzipped).
