# B50 note after test: verdict, predictions, and what the paper may say

Written 2026-11-20, about 01:50 box time, after every registered B50 test run (results/b50_verdict.txt,
results/b50_scores.json, results/b50_check.txt, results/b50_handread.txt, results/b50_literature.txt) and after the
composed part of exploratory 1 (results/b50_exploratory1.txt).

## Registered result

- **B50-1 = SURVIVES.** Every registered OOD gate on DISC, with its threshold set on dev so that it opens on 797 of
  800 trained-type dev items (the VSA gate's own dev count), still opens on most addition items, which DISC reads as
  bitwise OR: MSP 94.5%, energy 98.9%, Mahalanobis 99.6%, Mahalanobis-LW 99.6% (VSA gate 2.9%). All close on integer
  division (0%). All four are eligible (TR 0.9275-0.9302 vs DISC-0.8's 0.9298). DISC-best = MSP, HO 0.534;
  HO(VSA) - HO(DISC-MSP) = +0.431 [+0.389, +0.475], per seed +0.430 / +0.436 / +0.428. Every DISC gate still HURTS on addition
  (-0.89 to -0.93 vs BASE); none hurts on division.
- Trained types: TR(VSA) - TR(DISC-MSP) = -0.030 [-0.044, -0.016] (the other reading of "within 0.02" fails, as
  stated in the prereg it must, independent of B50).
- Threshold-free: AUROC (trained vs held-out) VSA similarity 0.9997; MSP 0.975-0.978; Mahalanobis 0.977; energy
  0.949-0.962. At the literal "matched false-open rate" (S2b: DISC thresholds set on TEST so that its held-out open
  count equals VSA's, 1.47%), DISC opens on only 71-90% of trained-type items and its composed TR falls to 0.72-0.87,
  below VSA's 0.8975; so the VSA gate also wins at matched false-open rate.
- Sensitivity (S3): only by opening on far fewer trained items (dev open rate 0.95) do the DISC gates approach VSA on
  held-out items (MSP HO 0.859, TR 0.888; still -0.107 behind VSA on HO).
- COT2048: task N trained types 0.302 [0.282, 0.322] (B40 COT512 0.275), 136 of 2,000 still at 2,048 tokens (6.8%),
  upper bound 0.367; held-out addition 0.796, division 0.972; task R 0.8085 [0.791, 0.825] (B40 0.801), 41 of 2,000
  at 2,048 (2.1%), upper bound 0.820. R1 re-read: task N REPLICATED (VSA - COT2048 +0.596 [+0.573, +0.619]); task R
  PARTLY as before (vs COT2048 +0.191 [+0.174, +0.208] PASS; vs LORA +0.059 fails). Cost: N 1,048,934 generated
  tokens (mean 420, median 266) and 2.73 A800 GPU-hours (3.9 s per item, on a GPU shared with another user's job);
  R 468,361 tokens (mean 234) and 1.62 GPU-hours (2.9 s per item); module arms: 6 new tokens.

## Exploratory 1 (registered after the verdict was read; composed from cached states, then confirmed live)

A least-squares one-hot type readout on the same states, with no vector-symbolic structure, reproduces the VSA gate:
AUROC 0.9999; at DE25's 0.8 it opens on 99.95% of trained-type items, 1.6% of addition, 0% of division; DISC gated by
it: TR 0.9293, HO 0.9720, HO(VSA) - HO = -0.007 [-0.017, +0.003]; at the dev-matched threshold (0.890): TR 0.9258, HO
0.980, HO(VSA) - HO = -0.015 [-0.026, -0.006]. So the held-out rejection comes from fitting a linear readout to exact
targets by least squares (DE25's encoder objective), not from HRR binding or the vector format. Live check (the same
gate inside DISC's forward pass, 6 runs on A800 GPU 2, 19:27-19:31Z): identical numbers at both thresholds (TR 0.9293 /
0.9258, HO 0.9720 / 0.9800, HO(VSA) - HO -0.007 [-0.017, +0.003] / -0.015 [-0.026, -0.006]); live vs cached gate
decisions 0 of 7,500 differ, composition mismatches 0, second scorer 0 / 0 on the 6 files.

## Predictions

| | prediction | outcome |
|---|---|---|
| P1 | k_s >= 796 | held (797 in every seed) |
| P2 | MAHA open <= 5% of each held-out type; >= 99% of trained | FAILED (addition 99.6%; division 0%, trained 99.85%) |
| P3 | MSP opens on >= 30% of addition in >= 2 seeds | held (96.0 / 94.4 / 93.2%) |
| P4 | ENERGY opens on fewer held-out items than MSP | FAILED (energy 98.9% vs MSP 94.5% on addition) |
| P5 | B50-1 DOES NOT SURVIVE, DISC-best MAHA, Delta in [-0.03, +0.02] | FAILED (SURVIVES, +0.431; DISC-best MSP) |
| P6 | all four gates eligible | held |
| P7 | AUROC MAHA >= 0.99, VSA >= 0.99, MSP <= 0.95 | FAILED (MAHA 0.977; MSP 0.975-0.978; VSA 0.9997 held) |
| P8 | p08 reproduces B40 on >= 2,498 / 2,500 per seed | held (0 differences in 7,500) |
| P9 | composition mismatches <= 0.5% | held (0 in 7,500 per gate) |
| P10 | COT2048 N in [0.28, 0.36]; >= 40% of the 517 still at 2,048; R1 vs COT passes | partly: 0.302 held, R1 held; only 150 of 517 (29%) still truncated: FAILED |
| P11 | COT2048 R in [0.80, 0.85]; R1 vs COT passes | held (0.8085; +0.191) |
| P12 | >= 99% of items ending before 512 tokens identical to B40 | held (1,977 / 1,977 and 1,913 / 1,913; longer items 523 / 523 and 87 / 87 prefix-identical) |
| P13 | second scorer 0 gold mismatches; 0 / <= 1% correctness mismatches | held (0 and 0 on 23 files) |
| P14 | independent scores Spearman >= 0.999 | held (dev 0.9990-0.9997, test 0.9997-0.9999) |
| P15 | <= 6 GPU-hours | held (5.05 logged for the registered runs, of which COT2048 4.35; 5.17 with the exploratory live check) |
| E1 | least-squares readout closes on >= 90% of held-out, AUROC >= 0.99, Delta in [-0.03, +0.02] | held (composed and live) |

Six of my predictions failed, and the central one (P5) failed in the direction that favours the VSA module: I
expected Mahalanobis to catch addition; it does not, because addition looks like OR to every feature-level score.

## What the paper may say

- It may say that DE25's similarity gate rejected the unseen addition and division items (open on 2.9% / 0%) while a
  same-shape softmax pipeline did not, even when that pipeline is given the standard OOD gates (max-softmax, energy,
  max-logit, Mahalanobis on its features, Ledoit-Wolf Mahalanobis) at the same trained-type open rate: held-out
  accuracy 0.965 vs at most 0.534 (+0.431 [+0.389, +0.475]); and that at a matched held-out false-open rate those
  gates give up 10-29% of trained-type items.
- It must say, beside that, that the rejection comes from the least-squares readout, not from the vector form: a
  plain least-squares one-hot readout on the same states does as well (exploratory, confirmed live: -0.007 [-0.017, +0.003]). So the
  paper must NOT list held-out gating as something the vector-symbolic format adds; at most "DE25's training
  objective (regression to fixed target vectors) yields a usable rejection score for free, which a cross-entropy
  head does not". It must also state that the VSA module's trained-type accuracy is 0.030 below DISC's.
- It must not generalise beyond Llama-3.1-8B-Instruct, task N and these two held-out operations, and should name
  relative Mahalanobis (Ren et al. 2021) as an untested competitor for the near-OOD case.
- CoT: every table replaces B40's COT (0.275 / 0.801) with COT2048 (0.302 / 0.8085). Because 6.8% of task-N
  trained-type items still hit 2,048 tokens (more than the 5% registered), the text must say the CoT baseline remains
  a lower bound, upper bound 0.367 (task R: 2.1%, 0.820), and give the cost (420 / 234 generated tokens per item vs
  6). R1's readings do not change.
