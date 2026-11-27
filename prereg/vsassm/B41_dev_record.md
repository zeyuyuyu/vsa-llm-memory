# B41 dev record (dev split only; before the test bank exists and before any test run)

Pre-registration: prereg/B41_superposed_uncertain_multihop_prereg.md (commit 2765541, 2026-11-18 14:56 UTC; the
file's "about 15:30 UTC" was an estimate, the commit time is the registration time).

## Naming head (b41_head.py fit; train bank only for fitting, dev bank for choices)

Pooled dev NLL at T = 1 per layer (epoch chosen by pooled dev NLL): L9 0.783 (epoch 11), L18 1.552, L27 0.961,
L36 1.387 -> layer 9 (typo'd names are read best early, as B31 found intermediate layers better for Qwen3-8B).
Temperature T = 0.927 (pooled dev NLL). Dev per-mention statistics at T:

  rho    argmax err  NLL    errors inside the true cluster  true in top-4  mass outside top-16  ECE
  0.03   0.036       0.128  0.995                           1.000          0.001                0.013
  0.06   0.071       0.239  0.974                           0.999          0.003                0.026
  0.10   0.115       0.393  0.967                           0.995          0.009                0.037
  0.15   0.187       0.668  0.911                           0.983          0.029                0.066
  0.20   0.262       0.994  0.857                           0.957          0.062                0.084
  0.25   0.329       1.315  0.786                           0.921          0.108                0.090
  0.30   0.402       1.708  0.710                           0.872          0.164                0.105

Registered rule (nearest dev error to 0.02 / 0.08 / 0.25): low rho = 0.03 (0.036), mid rho = 0.06 (0.071),
high rho = 0.20 (0.262). Low and mid are close because the head's error at the smallest typo rate is already 0.036.
Note: an earlier fit was started while the train features were still being written (partial train matrix); it was
killed and discarded before it finished (log kept as logs/head_fit_INVALID_partial_train.log on the box); the fit
above was run on the complete features.

## Dev engine runs (seed 101, 500 queries, the three rho levels, every N; results/b41_dev_verdict.txt)

Dev verdict (registered bars applied to dev): TABLE in all 45 cells under ops, traffic and state. Every VSA
configuration costs more per query (2.7e8-1.6e9 ops at D = 65,536; >= 1e6 at D = 256) than every table
configuration (ARGMAX / BEAM about 4.1e3-7e3, dominated by reading the start distribution; EXACT 5.7e4-1.5e8), so
the equal-cost table set is every table and v* (max dev A-gap) coincides with the most accurate VSA configuration in
all 45 cells. v* per cell is in results/b41_dev_choice.json (always D = 65,536; HOP16/HOP4 with kb in {4, 16} at
N <= 3,200, HOP1 at N >= 32,000; END only at low noise, h = 2, N = 32). C-ratio 2.5e-6 to 7.7e-6 in every cell.
A few dev numbers (accuracy at h = 6, high noise): ARGMAX 0.110 / 0.118 / 0.102 / 0.078 / 0.054 for
N = 32 / 320 / 3,200 / 32,000 / 100,000; BEAM(4,16) 0.504 / 0.568 / 0.530 / 0.408 / 0.138; EXACT 0.552 / 0.584 /
0.562 / 0.402 / 0.136; v* 0.420 / 0.390 / 0.214 / 0.048 / 0.014.

Gate G1 (noise-free readings, results/b41_g1.json): PASS (VSA-HOP1 at D = 4,096, N = 320: 1.000 at h = 2/4/6).
Noise-free capacity seen there: at N = 3,200 (about 100 facts per memory) HOP1 needs D >= 4,096 (0.99/0.96/0.95)
and D = 16,384 for 1.000; END fails at h >= 4 at every D tried (noise compounds without clean-up).

## Dev observation (diagnosis, not a design change)

The VSA falls below the noise-free table arms by more than crosstalk theory for a peaked query would predict. A
per-query trace (N = 320, high noise, D = 65,536; scratch scripts, not part of the pipeline) shows why: the VSA's
signal for an answer is proportional to the UNNORMALISED mass that survives the hop (the product of the start
reading and the fact reading), while the crosstalk is proportional to all of the query's mass, including mass on
candidates whose chain breaks. A table renormalises over surviving paths for free; in the VSA the mass on dead
candidates becomes noise (one traced query: normalised surviving mass 0.988 on the true entity, VSA score 1.6 noise
standard deviations). Signal scores matched theory (sqrt(D) x mass / |p|) on queries with large surviving mass.
The mechanics are otherwise exact (G1; second implementation on mock data: 0 disagreements).

## Implementation changes before the test (no design change)

- b41_engine.py: the per-hop re-bundle is computed as a sparse-weight matrix product (zeros(nq, E) with the top-c
  weights scattered in, times PHI) instead of gathering PHI rows (same sums; less GPU memory at large D); flag
  --no_tables for VSA-only runs. Dev runs used the earlier form. G0 compares the test runs with the second
  implementation as registered.
- b41_test_pipeline.sh, b41_indep_jobs.py: the test run order (registered seeds 1-3, 1,000 queries).

## Exploratory addition registered here (before the test; not in the bar, no verdict depends on it)

E1: VSA at D = 262,144 (2^18; every kb and mode) on the test runs with N >= 320 (N = 32 is skipped: its 1,000
contexts x ~25 memories at this D do not fit in GPU memory), to locate where the VSA's accuracy would meet the
tables'. Reported separately as "exploratory D = 2^18"; its cost is 4x that of D = 65,536.

## Predictions: dev status (for transparency; the test decides)

On dev, P2 (ARGMAX <= 0.10 at high noise h = 6, all N) would fail narrowly at N = 32 / 320 (0.110 / 0.118);
P3 (EXACT - ARGMAX >= 0.30 at high noise h in {4, 6}, all N) would fail at N >= 32,000; P10 would fail because
EXACT itself is only 0.136 at N = 100,000; P6 would hold at N <= 320 and fail at N = 3,200. No prediction is changed.
