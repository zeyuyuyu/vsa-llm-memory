# B67 note after the test (written after the verdicts; results/b67_verdict.txt, b67_verdict_llm.txt, b67_b.json)

Date: 2026-11-23, about 15:50 (machine clock). The paper is not edited by B67. No cut in (a) or (b); the registered
secondary H-G (cross-talk from each dev store's own Gram matrix) was not run (time; cut as registered, written here
before it was attempted). Disclosures: compute notes 1-2 (header times corrected; the registered theory L failed its own
synthetic selftest at V = 16 before any real number; an exploratory L' with a cross term restored was added after the
sentence encoders' registered verdict had been read; on the four language models L and L' were computed in one pass
before either was read). (b) is not blind (every measured N* and B61's scratch model were known).

## (a) Hebbian matched filter against least squares, 84 R1 cells (ideal; points)

Theory inputs: dev split only (2,152 alias-label pairs, 3,990 rows for the second moment; check C2), each cell's
dev-chosen configuration. Measured: B66 / B60 test accuracies.

| encoder | store theory / measured | LSFR theory L / L' / measured | LAMc theory L / L' / measured | STORE - LSFR: L / L' / measured | STORE - LAMc: L / L' / measured |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 45.4 / 48.8 | 43.7 / 45.5 / 50.3 | 45.1 / 47.7 / 52.9 | +1.75 / -0.13 / -1.50 | +0.30 / -2.32 / -4.06 |
| Granite-3.3-8B | 44.5 / 47.5 | 42.8 / 45.1 / 49.4 | 45.7 / 48.0 / 53.1 | +1.70 / -0.61 / -1.87 | -1.15 / -3.50 / -5.55 |
| Llama-3.1-8B | 45.2 / 49.0 | 43.8 / 45.8 / 50.7 | 46.2 / 48.7 / 54.6 | +1.43 / -0.54 / -1.68 | -0.93 / -3.51 / -5.56 |
| Mistral-7B | 45.4 / 51.3 | 43.7 / 45.7 / 52.8 | 46.8 / 48.8 / 56.5 | +1.70 / -0.26 / -1.49 | -1.37 / -3.39 / -5.23 |
| e5-base-v2 | 43.7 / 46.4 | 42.4 / 44.6 / 48.1 | 44.9 / 47.5 / 52.9 | +1.27 / -0.93 / -1.71 | -1.19 / -3.82 / -6.53 |
| bge-small | 43.2 / 45.1 | 42.7 / 44.8 / 47.4 | 43.8 / 46.1 / 49.3 | +0.42 / -1.65 / -2.26 | -0.65 / -2.92 / -4.18 |
| MiniLM-L6 | 42.9 / 44.3 | 41.7 / 43.6 / 46.2 | 42.6 / 44.6 / 48.6 | +1.15 / -0.75 / -1.88 | +0.33 / -1.74 / -4.30 |

Agreement over the 84 cells (registered metrics; L' exploratory):

| pair | Spearman [95%] | sign agreement | within 3 points | predicted < 0 / measured < 0 | mean predicted / measured |
| --- | --- | --- | --- | --- | --- |
| STORE - LSFR, L (primary) | +0.54 [+0.36, +0.69] | 34.5% | 53.6% | 16 / 71 | +1.35 / -1.77 |
| STORE - LAMc, L | +0.78 [+0.65, +0.87] | 72.6% | 23.8% | 54 / 77 | -0.67 / -5.06 |
| STORE - LSFR, L' | +0.59 [+0.41, +0.73] | 76.2% | 89.3% | 61 / 71 | -0.70 / -1.77 |
| STORE - LAMc, L' | +0.84 [+0.74, +0.91] | 95.2% | 66.7% | 77 / 77 | -3.03 / -5.06 |
| language models only (48), STORE - LSFR, L' (blind) | +0.34 [+0.04, +0.59] | 62.5% | 87.5% | 30 / 40 | -0.39 / -1.64 |
| language models only (48), STORE - LAMc, L' (blind) | +0.61 [+0.32, +0.80] | 95.8% | 62.5% | 46 / 46 | -3.18 / -5.10 |

Accuracy levels (measured - theory; Spearman over 84 cells): store +3.18 (MAE 3.26, rho 0.966; V = 2 +3.40, V = 16
+2.95); LSFR L +6.29 (0.967), L' +4.25 (0.962); LAMc L +7.57 (0.979), L' +5.21 (0.977). Language models: store +4.02,
LSFR L' +5.27, LAMc L' +5.94; sentence encoders: store +2.05, LSFR L' +2.89, LAMc L' +4.23. Largest departures: Mistral
and the largest budgets (M = 1,024, 16 bits per fact), store up to +8.8, least squares up to +16.6.

**Where measurement departs from theory (the new knowledge).**
1. *Both memories read better than Gaussian cross-talk theory says, least squares more so.* The shortfall is not a
   dev / test shift: against the dev accuracies of the same configurations (the split the theory is computed on;
   results/b67_dev_vs_theory.txt, exploratory, selection-biased upward by about a point) the theory is short by 3.9
   points for the store and 5.2 (LSFR, L') to 7.1 (LAMc, L') for least squares, and test is 0.7-1.9 below dev. The
   second-order (Gaussian, spectrum-only) description of the keys misses structure that both reads use, and least
   squares uses more of it.
2. *Direction right, size short.* With the cross term restored, theory puts least squares ahead of the store at the
   store's own features in 61 of 84 cells (measured 71) and ahead with its own features in 77 of 84 (measured 77), but
   predicts only about 40% (same features: -0.70 vs -1.77) to 60% (own features: -3.03 vs -5.06) of the measured gap.
   The registered derivation, which dropped that term, predicted the wrong sign (store ahead by +1.35 at the same
   features; R1 on all 7 encoders).
3. *Sign-feature cells.* For the store's sgn / sgn3 maps (13 of 84 cells) the spectrum-only kernel moment and the
   empirical moment of the dev keys differ by up to 6.2 points of predicted accuracy (check C1): the sixth-moment
   kernel's cross-talk depends on the tail of the key similarities, which the covariance spectrum does not fix.

## (b) Cells per stored pair at 99.5% exact recall

Inputs: the 50 RULER test stores' structure only (N_v 9 to 5,993, median 1,478; 1.11 values per probed key, so 0.45%
per slot; shard loads by B38's hash; MPHF bits). (x r) = measured / predicted; (+s) = measured above by s grid steps
(one step = x 2^(1/4) = 19%).

| noise | cell error | dense measured | Methods 4.1 (D-a) | Gaussian race (D-b) | sparse block measured | Bloom bound (S-a) | binomial (S-b) | best table | Shannon with directory (T-b) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k1 | 0 | 76 | 42 (x 1.81) | 64 (+1) | 32 | 27.9 (x 1.15) | 27 (+1) | 19 | 17.2 (x 1.10) |
| k2 | 1.3e-4 | 76 | 42 (x 1.81) | 64 (+1) | 32 | - | 27 (+1) | 19 | 17.3 (x 1.10) |
| k4 | 0.021 | 76 | 46 (x 1.66) | 76 (0) | 45 | - | 38 (+1) | 27 | 20.2 (x 1.33) |
| e05 | 0.05 | 91 | 52 (x 1.74) | 91 (0) | 54 | - | 45 (+1) | 38 | 24.2 (x 1.58) |
| k8 | 0.123 | 128 | 74 (x 1.73) | 128 (0) | 91 | - | 76 (+1) | 54 | 37.2 (x 1.45) |
| k16 | 0.276 | 431 | 210 (x 2.05) | 362 (+1) | 304 | - | 256 (+1) | 215 | 114.9 (x 1.87) |
| k32 | 0.374 | 1,722 | 667 (x 2.58) | 1,448 (+1) | 1,448 | - | 861 (+3) | 724 | 374.9 (x 1.93) |

The Bloom bound is a no-error bound (k1). With the largest store's vocabulary (N_v = 5,993, the paper's "31") it is
30.8 and the Methods expression 46.5. Readings: the Methods leading-order expression k* = d / (2 ln(N / eps)), with sign
storage and cell errors, under-states the dense store's need by 1.7 to 2.6 times (it omits the noise on the true
value's own overlap and the finite-N correction); the finite-size Gaussian race with the stores' own vocabularies and
shard loads is within one grid step everywhere (exact at k4 to k8). The sparse-block store sits one step above its
binomial model up to 16 times the noise and three steps (x 1.68) above at 32 times: the one clear departure in (b). The
best tables need 1.1 to 1.9 times their Shannon limit with the directory counted (1.1 without cell errors, the
directory and rounding; up to 1.9 at the highest noise, finite BCH / LDPC codes).

## Predictions

(a): PA1 FAILED (registered L predicts the store ahead at the same features in 68 of 84 cells and on all 7 encoders;
STORE - LAMc predicted < 0 in 54, not >= 70). PA2 HELD (Spearman +0.54 / +0.78; on the language models alone +0.47 /
+0.58). PA3 FAILED (all Spearman >= 0.96 and store MAE 3.3, LSFR 6.3 held; LAMc MAE 7.6 > 7). PA4 FAILED (53.6% in the
3-point band). PA5 FAILED both ways (store MAE is larger at V = 2, 3.5 vs 3.1, and the theory under-predicts rather than
over-predicts; the prereg's "mean bias > 0" was read as its words, over-prediction). PA6 FAILED (0 of 7 within 2
points). (b): PB1-PB4 HELD (not blind).

## Checks

- C1 (a), b67_check.py, own code (own dev second moment, Monte Carlo kernel moments, quantizer constants and races;
  fixed-point deterministic equivalent): quantizer constants identical to 4 decimals (q = 1 / 2 / 4: 0.5708 / 0.1349 /
  0.0117); theory H agrees within 0.53 points on all 71 lin cells but differs by up to 6.2 points on the 13 sgn / sgn3
  cells (registered bar 1.5 points: FAILS there, 6 of the 13 beyond it; item 3 above). Theory L and L': 112 of 168
  arm-cells checked (feature dimension <= 1,500); 110 agree within 0.07 points; 2 (Qwen3-8B and Granite, M = 256, V = 2,
  4 bits per fact, p = 992 > M) differ by 3.7-4.0 points because the check's fixed-point solver did not converge there
  (verified on Qwen3-8B: the check's kappa 3.65 leaves the defining equation at +0.65, the main bisection's 2.804
  solves it to 2e-6, with the same spectrum, which the two codes reproduce to 4e-14; Granite's cell has the same
  configuration type, kappa 0.00026 vs 0.00034, not separately verified): the bar fails there on the check's side
  (results/b67_check.txt, compute note 3).
- C1 (b): D-a, S-a, T-b identical; D-b by quadrature and S-b by explicit binomial loops give the same grid N* at all 7
  conditions.
- C2: every row entering a theory number is from the dev split (3,990 rows per encoder: 1,838 labels, 2,152 aliases);
  the make_store guard raises on 'test' (results/b67_check_c2_*.json).
- C3 (synthetic): H within 0.6 points of simulation; registered L failed (5.3 points at V = 16); L' within 0.2 points in
  3 of 4 cases, 2.1 in the interpolation case.

## Literature (from memory, not re-checked online in this session)

The Gaussian race and Hebbian capacity (Plate 2003; Frady, Kleyko & Sommer 2018) are derived for random i.i.d. codes and
match simulations there, as our C3 does; for encoder keys both memories beat it, consistent with the known failure of
Gaussian-equivalence arguments for structured, heavy-tailed data (random-features theory: Mei & Montanari 2022; Goldt et
al. 2022 state Gaussian equivalence for Gaussian-like inputs). The pseudo-inverse / ridge read as decorrelation is
Kohonen (1972) and Personnaz, Guyon & Dreyfus (1986); its deterministic equivalent Dobriban & Wager (2018), Hastie,
Montanari, Rosset & Tibshirani (2022). Capacity bounds of the form d >= C k ln(N / eps) are stated with loose constants
(Thomas, Dasgupta & Rosing 2021; Clarkson, Ubaru & Yang 2023), consistent with the 1.7-2.6 factor here; Bloom (1970) for
the OR bound. Reading: supports (b), departs (a) in a direction the literature allows.

## For the paper (plain sentences, numbers)

Gaussian cross-talk theory computed from the development keys' covariance tracked both memories' accuracies across the
84 conditions (rank correlations 0.96 to 0.98) but fell short of them, by 3.2 points for the store and 4.3 for least squares on
the same features; it predicted the direction of the write-rule gap (least squares ahead in 61 conditions, 71
measured) but only about 40% of its size (-0.7 against -1.8 points; -3.0 against -5.1 when least squares used its own
features), and our pre-specified derivation, which had dropped a cross term, predicted the wrong sign. Where the
measurement departs, it is in the direction of structure the second-order theory ignores: the real keys' cross-talk
helps least squares more than the store. For storage cells, the Methods' leading-order capacity expression
under-states the dense store's need by 1.7 to 2.6 times, a finite-size Gaussian race with the stores' own vocabularies
lands within one 19% grid step at every noise level, the sparse-block store sits one step above its Bloom-filter /
binomial bound up to 16 times the noise and three steps (1,448 against 861 cells per pair) above it at 32 times, and the
best tables need 1.1 to 1.9 times their Shannon limit.
