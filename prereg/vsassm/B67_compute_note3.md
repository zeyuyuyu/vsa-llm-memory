# B67 compute note 3 (checks; after the verdict)

Machine clock 2026-11-23 15:48. The four language models' (a) theory finished 15:36-15:38 (12 cells each, rc 0); the
verdict was run once on all 84 cells and once on the 48 language-model cells (results/b67_verdict_llm.*). Checks:

- C1 (a) language models, b67_check_llm.py (b67_check's C2, H and L / L' checks in one process per encoder, sharing one
  DevData): H within 0.53 points on all 48 language-model cells' lin configurations (Mistral's one sgn cell 0.38);
  L / L' checked on 58 of 96 arm-cells (p <= 1,500), 56 within 0.06 points, 2 cells (Qwen3-8B LSFR and Granite LAMc at
  M = 256, V = 2, 4 bits per fact; p = 992 > M = 256) 3.7 to 4.0 points apart. Diagnosis (b67 scratch, same spectrum,
  identical in the two codes to 4e-14): on Qwen3-8B the main bisection's kappa = 2.8042 gives kappa (1 - df1 / M) - lam_e
  = 2.4e-6; the check's damped fixed point stops at 3.65 (+0.65) and, rerun, at 3.07 (+0.20): the fixed-point map does
  not converge when df2 > M. The main numbers stand; the registered bar is failed by the check code in those 2 cells.
- C1 (b) b67_check.py b: D-a, S-a, T-b identical at every condition; D-b (scipy quad per shard load) and S-b (explicit
  binomial loops) give the same grid N* as the main code at all 7 conditions (k32 finished 15:53).
- C2: all 7 encoders: 3,990 rows used, all from the dev split; the guard raises on a test store.
- The (b) verdict scorer was corrected before the (b) predictions were scored on file: measured N* are grid values
  printed rounded (1,722 = 2^(43/4) = 1,722.16), so the step distance 1722 vs 1448.15 read as 0.9997 < 1 and PB2 as
  FAILED in the first verdict print; snapping the measured values to the grid gives +1.00 (PB2 HELD). Disclosed.
