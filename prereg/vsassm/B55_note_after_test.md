# B55 note after test: verdict, predictions, and what the paper may say

Written 2026-11-20, about 06:50Z, after all 60 registered test runs (A800 GPU 0, 05:52:40-06:32:36Z) and after
results/b55_verdict.txt, results/b55_scores.json, results/b55_check.txt, results/b55_handread.txt and
results/b55_literature.txt. The A800 plan ran (prereg/B55_compute_note1.md); comparison arms are B40's and B50's files.

## Registered result

Task N, trained types (2,000 items), N-ARG = B40's VSA seeds with argmax digit clean-up; N-THR = the published
thresholded read; seed-mean accuracy; Delta = N-ARG - N-DISC, paired 95% CI:

| rho | N-ARG | N-THR | N-DISC | B55-N1 Delta [CI] | label |
|---|---|---|---|---|---|
| 0 | 0.960 | 0.898 | 0.930 | +0.030 [+0.021, +0.040] | WIN (all seeds; B40 R2 bar: VSA WIN) |
| 0.25 | 0.671 | 0.602 | 0.909 | -0.238 [-0.260, -0.217] | LOSE (all seeds) |
| 0.5 | 0.199 | 0.196 | 0.811 | -0.613 [-0.634, -0.591] | LOSE |
| 1 | 0.121 | 0.121 | 0.475 | -0.354 [-0.378, -0.332] | LOSE |
| 2 | 0.119 | 0.120 | 0.141 | -0.022 [-0.037, -0.007] | LOSE (below the R2 magnitude bar) |
| 4 | 0.069 | 0.069 | 0.073 | -0.004 [-0.015, +0.006] | TIE |
| 8 | 0.048 | 0.048 | 0.078 | -0.029 [-0.039, -0.019] | LOSE |

N-ARG - N-THR: +0.063 [+0.052, +0.073] at rho 0, +0.068 [+0.057, +0.080] at 0.25, +0.003 [-0.002, +0.007] at 0.5, and
within +/-0.001 from rho 1 up. So the argmax read removes the threshold's zeroed digits on clean and lightly noisy reads,
but from rho 0.5 up both reads fail on the same items: the noise fragility is not an artefact of DE25's threshold.

Task N, held-out types (500 items), B55-N2 = HO(N-ARG) - HO(N-DISC-MSP) (DISC with B50's MSP gate):
rho 0: 0.965 vs 0.534, +0.431 [+0.387, +0.475] WIN; 0.25: 0.927 vs 0.523, +0.403 [+0.361, +0.445] WIN; 0.5: 0.849 vs
0.526, +0.323 [+0.278, +0.368] WIN; 1: 0.719 vs 0.540, +0.179 [+0.130, +0.228] WIN; 2: 0.373 vs 0.632, -0.259 [-0.312,
-0.207] LOSE; 4: 0.139 vs 0.735, -0.596 LOSE; 8: 0.143 vs 0.751, -0.607 LOSE. Mechanism: the VSA type gate (similarity
> 0.8) opens on noisy held-out reads (held-out open rate 1.5% / 5.9% / 13% / 27% / 66% / 97% / 99.7% from rho 0 to 8),
while the MSP gate on DISC closes more under noise (held-out open 47% -> 25-26%, trained-type open 99.5% -> 48% at rho
8), returning the frozen LLM's own answer. Against DISC with B40's 0.8 softmax gate the held-out difference is positive
up to rho 2 (+0.060 [+0.006, +0.112]) and negative at 4 and 8.

Task R (2,000 items), R-ARG = B40's task-R VSA with every mention read cleaned up by argmax before binding;
B55-R1 = R-ARG - R-DISC:

| rho | R-ARG | R-VSA (B40) | R-DISC | B55-R1 Delta [CI] | label |
|---|---|---|---|---|---|
| 0 | 0.9995 | 0.9995 | 0.993 | +0.006 [+0.004, +0.009] | WIN (CI rule); R2 bar: TIE |
| 0.25 | 0.9995 | 0.9993 | 0.993 | +0.006 [+0.004, +0.010] | WIN; R2 TIE |
| 0.5 | 0.9995 | 0.991 | 0.993 | +0.006 [+0.004, +0.010] | WIN; R2 TIE |
| 1 | 0.999 | 0.728 | 0.993 | +0.006 [+0.003, +0.009] | WIN; R2 TIE |
| 2 | 0.554 | 0.344 | 0.993 | -0.439 [-0.457, -0.420] | LOSE |
| 4 | 0.317 | 0.301 | 0.894 | -0.577 [-0.595, -0.559] | LOSE |
| 8 | 0.290 | 0.291 | 0.444 | -0.153 [-0.171, -0.136] | LOSE |

R-ARG - R-VSA: +0.271 [+0.255, +0.287] at rho 1 and +0.210 [+0.188, +0.232] at 2; ~0 at 4 and 8. B40's task-R
fragility at rho 1 (0.728 with every mention read correctly, reads-ok 0.9998) came from binding uncleaned reads into the
memory; from rho 2 up the per-mention VSA read itself fails (reads-ok 0.856 / 0.317 / 0.098 vs DISC 1.000 / 0.976 /
0.558), and DISC is far ahead.

## Checks

- Same noise (A800, same GPU type as B40): N-ARG vs N-THR type read and similarity identical on 7,500 / 7,500
  seed-items at every level; N-DISC-MSP vs N-DISC reads (op, op_p, x, y) identical on 7,500 / 7,500 and, where both
  gates open, predictions identical (e.g. 6,689 / 6,689 at rho 0.25); R-ARG vs R-VSA per-item read flag identical on
  6,000 / 6,000. MSP gate re-derived from the logged score and tau: 0 mismatches.
- Noise-0 N-ARG (digit_thr -inf) vs B40 exploratory 1 (digit_thr 0): predictions and module records identical on
  7,500 / 7,500.
- Second scorer (b40_check.py) on all 60 files: 0 gold mismatches, 0 correctness mismatches.
- Hand-read (results/b55_handread.txt): at rho 0, of 10 sampled items N-ARG gets right and DISC wrong, 5 are DISC misreads
  and 5 DISC write-backs the LLM did not follow; at rho 0.25 11 of 11 N-ARG losses are operand misreads; task R rho 2: 12 of 12
  R-ARG losses are mention misreads.

## Predictions

| | prediction | outcome |
|---|---|---|
| Q1 | noise 0 reproduces exploratory 1 (>= 7,490 / 7,500); N1 WIN | held (7,500 / 7,500; +0.030) |
| Q2 | N1 LOSE at 0.25, 0.5, 1; N-ARG 0.62-0.76 / 0.15-0.45 / <= 0.25 | held (0.671 / 0.199 / 0.121) |
| Q3 | rho 2, 4, 8: |N-ARG - N-DISC| <= 0.05 | held (0.022 / 0.004 / 0.029) |
| Q4 | N-ARG - N-THR >= +0.03 (CI > 0) at 0, 0.25; |.| <= 0.03 at rho >= 1 | held |
| Q5 | N2 WIN at 0-0.5; HO(N-ARG) <= 0.5 at rho >= 2; HO(N-DISC-MSP) <= 0.3 at rho >= 2 | partly: first two held (WIN also at rho 1); HO(DISC-MSP) 0.63-0.75 at rho >= 2: FAILED (the MSP gate closes under noise; I expected noise to inflate the logits and open it) |
| Q6 | R-ARG >= 0.995 at 0-0.5, >= 0.98 at 1 (+0.20 over R-VSA), 0.20-0.80 at 2, <= 0.45 at 4, 8 | held (0.9995, 0.999, +0.271, 0.554, 0.317, 0.290) |
| Q7 | R1 WIN at 0 (R2 TIE); TIE/WIN at 0.25-1; LOSE at 2, 4; LOSE/NR at 8 | held |
| Q8 | same-noise checks 0 differences | held |
| Q9 | second scorer 0 / 0 | held |
| Q10 | <= 3 GPU-hours | held (1.31 lane-hours logged, two lanes on one A800 for 0.67 h wall; 0 PACE) |

One prediction (part of Q5) failed.

## What the paper may say

- Table 1 must name the read of every VSA row: DE25's published thresholded read 0.898 and the argmax read 0.960 vs
  DISC 0.930. The paper may say the argmax-read VSA module beats the same-shape discrete module by 3.0 points on the
  trained types (+0.030 [+0.021, +0.040], all three seeds), and must say in the same place that the argmax read is our
  deviation from DE25's published module, introduced after B40's registered test and registered here as a re-run of a
  known result (not a blind test), and that half of the sampled disagreements are DISC write-backs the LLM did not
  follow, so the margin is not only a read effect.
- The noise sentence must say it holds for both reads: with the argmax read the vector module still loses to the
  discrete module at every read-noise level from 0.25 to 2 and at 8 (task N: 0.67 / 0.20 / 0.12 vs 0.91 / 0.81 / 0.47 at
  0.25 / 0.5 / 1.0; tie at 4, where both are below the frozen LLM). The argmax read helps only at 0.25 (+0.07 over the
  thresholded read) and not at all from 0.5 up. So the earlier fragility is NOT a threshold artefact; it is the read
  margin of the MSE-to-HRR linear encoder.
- Held-out types: the VSA gate's advantage over DISC's best B50 gate holds up to read noise 1 (+0.18 to +0.43) and
  reverses from 2 up (-0.26 to -0.61), because the similarity gate opens on noisy reads; with B50's caveat unchanged (a
  plain least-squares readout reproduces the clean rejection), held-out gating is still not something the vector format
  adds.
- Task R: the paper must say that B40's task-R module already read by argmax at every hop. Cleaning each mention read
  before binding (R-ARG) makes it as robust as DISC up to rho 1 (0.999 vs 0.993; B40's uncleaned module 0.728 at rho 1),
  so B40's task-R noise result at rho 1 came from binding uncleaned reads; from rho 2 up DISC is far ahead (0.99 / 0.89
  / 0.44 vs 0.55 / 0.32 / 0.29). The +0.006 at rho <= 1 passes the CI rule but not B40's 0.03 magnitude bar: call it a
  tie in the text.
- Must not: call the clean 3-point win a property of the vector format (it depends on a clean-up rule we chose after
  seeing B40's test, and partly on DISC's write-back); claim any noise-robustness advantage for the vector module on
  either task; generalise beyond Llama-3.1-8B-Instruct, these two task families and this isotropic read-only noise.
