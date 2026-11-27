# B72 pre-registration: the write rule in the store's own format, symmetric side-information accounting, store-level inference with few clusters, and nonlinear random features for least squares (R1, equal bits)

Date: 2026-11-23 (machine clock Asia/Shanghai, about 19:45; US Eastern 2026-11-23 06:45). Written and committed, then
OpenTimestamps-stamped, before any B72 development or test run on a development or test store. The paper is not
edited by B72.

## Why (referee round 23, four points on the equal-bit comparison)

(A) "Least squares on the store's own features" (B66's LSFR / B71's LSFM) is not a pure write-rule test at V = 16: least
squares stores p x c weights (p about d / c features, free value columns), the store keeps d numbers with fixed Hadamard
value codes. Isolate the write rule at the store's exact format and bits. (B) B71 (D2) charged side information only
beyond what the compared store design uses: reference-dependent. Charge every memory's training-derived side information
symmetrically, amortised over K stores sharing one encoder. (C) At 1,024 facts there are five test stores per condition:
few clusters for a percentile bootstrap. Add a t interval over per-store differences and an exact sign-flip permutation
test over test stores. (D) LAMc's grid has only linear random features while the store may use sign maps; add sign and
product-of-three-signs random features to LAMc, and describe where (which conditions) the store leads LAMc.

## What has been seen before this file (disclosed)

- Every verdict and note of B49-B71 (in particular B60's STORE - LAMc, B62's STORE - NLMSc, B66's STORE - LSFR / LSFR -
  LAMc per region, B71's tables, D1 / D2 and side-information table). The point estimates of (C)'s four contrasts are
  B60 / B62 / B71's; their t intervals and permutation p-values have not been computed. B60's per-cell STORE and LAMc
  accuracies exist in results/b60_part_{enc}.json and B71's in results/b71_part_{enc}.json; for (D)'s heterogeneity
  table they are read only after this file is committed (I have not looked at cell-level STORE - LAMc values; B60's
  note says RH-V2 store leads on the sentence encoders and MiniLM R1-V2 is level).
- Dev choice files read for the design (configuration fields only): B53's (prep, fmap, q, share) per R1 cell, B56's
  STAR family and its q, VSAPc's (prep, pfmap, q) and PSTAR family, B60's LAMc configuration and dev accuracy field
  name. E.g. B53 chose q = 1 in most language-model R1 cells; VSAS's q differs from B53's in some cells.
- B72 code runs before this file, TRAINING split / toy data only (b72_out/logs/selftest.log, scratchpad smoke): the
  closed-form solve equals a brute-force ridge minimiser of the matched-filter score error (relative error < 1e-8,
  V = 2 / 16, two prefixes); alpha -> infinity gives the Hebbian bundle (cosine > 1 - 1e-6); the charged size rule is
  within budget and maximal; on TRAINING-split stores (M = 256, 8 bits per fact, q = 1, whiten / lin) the optimal-format
  write read 0.637 vs the Hebbian bundle's 0.574 (MiniLM, V = 2), 0.295 vs 0.277 (MiniLM, V = 16), and no better than
  the Hebbian bundle on Qwen3-8B (0.596 / 0.134 at alpha = 1e4, i.e. the Hebbian limit); a smoke run of every arm's job
  code on one TRAINING-split store per arm (timing; e.g. OPT kind R, Qwen3-8B, M = 1,024, V = 16, three budgets: 73 s).
  No development or test store has been answered by any B72 arm.

## Fixed (B49 / B53 / B60 / B62 / B66 / B71's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (md5s as
B60-B71), splits, B32's make_store stores (dev 6 + 3, test 20 + 5 per (M, V)), queries (unseen aliases), **R1** = V in
{2, 16} x 4, 8, 16 bits per fact x M in {256, 1,024} (12 cells per encoder), reported as R1-V2, R1-V16 and R1; **ideal
accounting**. Existing arms are not re-run; their test predictions are read as b71_verdict.load reads them: STORE (B53),
VSAR (B49), VSAS (B56), VSAPC / VSAPCX (B60), LAMC (B60), NLMSC (B62), LSFR (B66), SDMT (B64, b66_storeboot.load's logic).

## (A) The optimal write in the store's own format (b72_run.py arm OPT)

**Format kept** (per cell, as the store's dev choice fixed it): features phi, value codes c (B53's antipodal / Sylvester-
Hadamard codes; row-0 signs r), d = B32's size rule at the format's q (bits ideal: d H_q + 32 per block of 8,192), the
store's quantizer (b32_common.quantize per block of 8,192: q = 1 sign; q >= 2 Max's step on the block's std), the stored
vector S in R^d, and the matched-filter read score(v) = sum_j S_j phi_j(q) c_j(v), argmax.

**Write replaced.** The Hebbian S = sum_i phi(k_i) * c(v_i) is replaced by
S* = argmin_S sum_i sum_v (score_i(v) - t_i(v))^2 + lam ||S||^2, t_i = e_{v_i} - 1/V (centred one-hot), over the written
facts. With x = phi * r and residue classes c = j mod 16 (V = 16; one class at V = 2), score_i = H16 m_i, m_ic = <x_ic,
S_c>; since H16 H16 = 16 I the objective splits exactly into one ridge regression per class (targets H16[v_i, c] for
c >= 1, 0 for c = 0, so S_0 = 0, which costs nothing at the argmax; +-1 at V = 2): S_c = X_c^T (X_c X_c^T + lam I)^-1
y_c, closed form in float64 (eigendecomposition of the per-class Gram matrix), lam = alpha x mean over classes of mean_i
||x_ic||^2 (one lam for every class). As alpha -> infinity, S* -> the Hebbian bundle (classes 1..15; selftest), so the
grid contains the store's own write in the limit. Target scaling is immaterial (S* is linear in the targets; the
quantizer is scale-equivariant), so no target-scale grid. S* is quantized once by the store's quantizer at the format's q
(seeds ('B72Q', store seed, kind, d, q, n_written, alpha)) and read by the matched filter (b56_run.quant_np, mf_read).

**Kinds (formats).** OPTR = the final design's format: B53's chosen (prep, fmap, q) of the cell (b71_run.store_feats_d
kind R). OPTS = the same random features and codes at VSAS's chosen q (B56's STAR family's q). OPTP = VSAPc's format:
VSAPc's chosen (prep, pfmap, q), d = min(B32's rule, n) (kind P). **OPTX** = per cell the format of VSAPcX's source
(OPTP where VSAPcX = VSAPc, OPTS where VSAPcX = VSAS, by the registered VSAPcX rule): the strongest variant's format.
VSAS's decorrelating (LMMSE) read is not kept: after a least-squares write a linear decorrelating read spans the same read
functions (redundant before quantization), so the matched filter is the read in every kind (disclosed).
**OPTQ** (secondary) = kind R with q re-chosen in {1, 2, 4} (bits per number freed; tie order q > f > alpha).

**Grid** (the write's own free parameters; order = tie order): share f in {1, 1/2, 1/4, 1/8} (the store's share grid;
written facts = the store order's first floor(f M)) > alpha in {1e-4, 1e-2, 0.1, 1, 10, 100, 1e4} = 28 per cell and
kind. **B60's dev rule** (highest mean dev accuracy over the cell's dev stores; within 0.1 points tied, first in grid
order). Test: the chosen configuration, run once.

**Contrasts** (per encoder; R1-V2, R1-V16 and R1 SEPARATELY; primary, 10,000 draws): OPTR - STORE (what the Hebbian write
costs at the final design's format and bits), OPTR - LAMC, OPTR - NLMSC; OPTX - VSAPCX, OPTX - LAMC, OPTX - NLMSC.
Secondary (2,000 draws): OPTP - VSAPC, OPTQ - OPTR, OPTR - LSFR (the format cost at identical random features and the
least-squares write: fixed Hadamard classes vs free value columns), and the exact decomposition STORE - LAMC = [STORE -
OPTR] (write rule at the store's format) + [OPTR - LSFR] (format) + [LSFR - LAMC] (features); per encoder over R1 the
largest term in absolute value is named. Dev audit: OPTR's quantized vs unquantized dev accuracy at its chosen
configuration (quantisation cost of the optimal write) beside the Hebbian store's dev accuracy.

## (B) Symmetric side-information accounting (four language models; arms BV, BL, BS)

**Scenario.** K stores share one encoder. Every training-split-derived component of EVERY memory is charged at 8 bits
per number, amortised over K, and deducted identically from the memory's budget before its own size rule:
mean n (both preps); whitening (ZCA) matrix n (n + 1) / 2 (whiten); principal basis n per feature (centre-prepped
principal features of LAMc / NLMSc; with whitening the basis is derivable from the whitening matrix, B71's minimal rule,
applied to every memory alike); SDM hard locations n per location (the prepped training codes); random projections and
value codes one 64-bit seed each. K in {1e3, 1e4, 1e5, infinity}; infinity = the paper's current comparison (no new run).
At n = 4,096 the whitening matrix + mean cost 67,142 / 6,714 / 671 bits per store at K = 1e3 / 1e4 / 1e5 against R1
budgets of 1,024-16,384 bits; the mean alone 33 / 3.3 / 0.33.

**Re-choice.** A configuration is feasible if its charged stored part has size >= 1 (d, p or N). Configurations whose
side information does not fit drop out automatically; the grids already contain the configurations without each
component (centring instead of whitening, random instead of principal features, fewer SDM locations). Each memory's
size parameter is re-derived at the reduced budget (d = B32's rule at B - charge / K; p = the largest p with stored bits
+ basis charge <= B - fixed charge / K, maximal; N = B64's largest grid point with counter bits + default-value bits +
location charge <= B), so every configuration is re-run on dev; each memory re-chosen by its own registered rule over its
feasible configurations (STORE / VSAR: B32's rule, prep > fmap > q > share; LAMC: B60's rule over its 288; NLMSC: B62's
rule over its 480; SDMT: B36's rule over B64's 504); test once for the chosen configurations. Whiten jobs are skipped
where no R1 budget of that M can hold the whitening matrix (K = 1e3: everywhere; K = 1e4: M = 256). A cell with no
feasible configuration answers the store's majority value.

**Memories and engines.** STORE (B53's VSA, b32_run.vsa_grid with Feat53, d explicit), VSAR (B49's first design,
b32_run.vsa_grid with b32_run.Feat), LAMC (b60_run.run_lamc, its job seed, with b60_run.p_of replaced by the charged
rule), NLMSC (b62_run.run_online likewise), SDMT (b64_run.run_sdmt with the location charge in its N rule; the only
change). Contrasts per K (R1-V2, R1-V16, R1; one-level 2,000 draws, two-level beside): **STORE - LAMC**, **VSAR - SDMT**,
**STORE - NLMSC**; secondary STORE(K) - STORE, LAMC(K) - LAMC. **VSAPcX - LAMc is cut now (before any run)**: VSAPcX
needs B56's LMMSE / retraining reads (training codes as side information) and VSAPc's principal store re-sized under the
charge, two engines not re-implemented within the time box; registered priority puts (B) last. The sentence encoders are
cut now (the brief's scope: language models first).

## (C) Store-level inference with few clusters (analysis of saved outputs only)

Contrasts (headline, over R1; also R1-V2 and R1-V16, and every (A) / (D) primary contrast): STORE - LAMC, VSAPCX - LAMC,
STORE - NLMSC, VSAPCX - NLMSC. Stores are nested in strata g = (M, V) (R1: 4 strata, 20 + 20 + 5 + 5 = 50 test stores;
R1-V2 / R1-V16: 25 each). For store u in stratum g with Q_u queries, D_u = the mean over the stratum's three budget
cells of the per-query difference within store u (a store's queries are the same at every budget). The registered
region statistic is EXACTLY T = sum_u w_u D_u with w_u = Q_u / (G Q_g), G = number of strata, Q_g = sum of Q_u in g.
- **t interval** (stratified, design-based; stores = clusters): per stratum the ratio estimate T_g = sum Q_u D_u / Q_g with
  variance s_g^2 = sum_u (Q_u / Qbar_g)^2 (D_u - T_g)^2 / (n_g (n_g - 1)); Var(T) = sum_g s_g^2 / G^2; df by
  Welch-Satterthwaite, (sum_g a_g)^2 / sum_g (a_g^2 / (n_g - 1)), a_g = s_g^2 / G^2; 95% interval T +- t_{0.975, df}
  sqrt(Var(T)); label by B60's rule.
- **Exact sign-flip permutation test** over test stores (H0: within every store the two arms' outputs are exchangeable,
  so each D_u is symmetric about 0): T* = sum_u w_u s_u D_u over ALL 2^S sign vectors (S = 50 or 25), enumerated exactly
  by meet-in-the-middle (two halves of 25 / 13 + 12 stores, sorted sums, counts by binary search); two-sided p = #{|T*| >=
  |T| - 1e-12} / 2^S. "Reject" iff p < 0.05.
- Reported beside the one-level and two-level bootstrap labels (b66_storeboot.analyse). A label is "store-robust" iff the
  one-level label, the t-interval label and (for leads) the permutation rejection agree.

## (D) Nonlinear random features for least squares (arm LAMP)

LAMc's options extended by sign (sgn: sign of a Gaussian projection, >= 0 -> +1) and product-of-three-signs (sgn3) random
features: b66_lsfc.run_job UNCHANGED (b60_run.run_lamc on b32_run.Feat's sign maps, the store's projection seeds; the 72
LAMc options per feature map: scale (col, flat) > target (V-1, V) > alpha (1e-4 ... 100) > q (1, 2, 4); p maximal, at
most 65,536), (prep, fmap) in {center, whiten} x {sgn, sgn3} = 288 new configurations per cell, every R1 cell, all 7
encoders, dev stores. **LAMc+** = per cell the best new configuration by B60's rule (tie order scale > target > (prep,
fmap) center-sgn, center-sgn3, whiten-sgn, whiten-sgn3 > alpha > q) iff its dev accuracy exceeds LAMc's chosen dev
accuracy by more than 0.1 points; otherwise LAMc's choice (so LAMc+ - LAMc = 0 there). Test: the new configurations
chosen, once. Contrasts (primary, 10,000 draws): **LAMC+ - LAMC**, **STORE - LAMC+** (R1-V2, R1-V16, R1).
**Heterogeneity of STORE - LAMC** (existing outputs): per encoder and R1 cell the point difference with its one-level
cell interval (b36_verdict.boot_region on one cell, 2,000 draws); the cells where the store leads (lower bound > 0) and
where its point estimate is above 0; a short reading of why from existing outputs (both arms' dev-chosen configurations:
LAMc's features / p / q, the store's prep / fmap / q / share; cell accuracies).

## Statistics and decision rule

Region statistic = mean over cells of the cell mean of the per-query difference; **one-level** 95% cluster bootstrap
(entity within store, within each cell; b36_verdict.boot_region) = the registered label; **two-level** = B66's store-level
scheme (b66_storeboot.analyse), reported beside; (C)'s t interval and permutation test as above. **B60's rule**: "x leads"
iff the lower bound > 0, "y leads" iff the upper bound < 0, else "level". Counts over the 4 language models and the 3
sentence encoders.

## Predictions (stated so that they can be wrong)

- QA1. OPTR - STORE over R1: OPTR leads (one-level) on at least 4 of 7; |OPTR - STORE| <= 2.5 points on all 7; OPTR -
  STORE smaller than B66's LSFR - STORE (+1.5 to +2.3) on at least 5 of 7 (the fixed q and the fixed codes bind).
- QA2. OPTR - LAMC and OPTX - LAMC: LAMc leads on 7 of 7 over R1 (and at R1-V16 on 7 of 7).
- QA3. OPTR - NLMSC: NLMSc leads on at least 4 of 7 over R1; OPTX - NLMSC: NLMSc leads on at least 3 of 7.
- QA4. OPTX - VSAPCX over R1: OPTX leads on at least 4 of 7.
- QA5. Decomposition of STORE - LAMC over R1: the features term LSFR - LAMC is the largest in absolute value on at least
  5 of 7; the format term OPTR - LSFR is <= 0 (free columns help) on at least 5 of 7.
- QA6. OPTQ - OPTR over R1 between -0.5 and +1.0 points on at least 5 of 7.
- QC1. t-interval labels equal the one-level labels in at least 26 of the 28 headline (contrast x encoder) R1 cases; the
  permutation test rejects in every case whose one-level label is a lead with |difference| >= 1 point.
- QC2. The t interval is wider than the one-level interval on average over the 28 cases (ratio > 1.0), narrower than the
  two-level one (ratio < 1.0 against it).
- QD1. LAMC+ - LAMC over R1: within +-0.5 points on at least 5 of 7; LAMc+ takes a sign map in at most 25% of the 84 R1
  cells.
- QD2. STORE - LAMC+ over R1: LAMc+ leads on 7 of 7.
- QD3. The store's point estimate is above LAMc's in at most 8 of the 84 R1 cells, all at V = 2, and the store leads
  (cell interval above 0) in at most 3.
- QB1. STORE - LAMC: LAMc leads over R1 on at least 3 of 4 language models at every K.
- QB2. VSAR - SDMT: the first design leads over R1 on at least 3 of 4 at K = 1e3 and 1e4; SDMt's lead (as run) is lost
  ("level" or "first design leads") on at least 2 of 4 at K = 1e5.
- QB3. STORE - NLMSC: NLMSc leads over R1 on at least 3 of 4 at K = 1e5.
- QX. Checks pass (below).

## Readings for the paper (fixed now)

- (A) If OPTR leads STORE on an encoder: "at the store's own format and bits (same features, value codes, dimension,
  quantizer and read), the least-squares write of the same vector answered x more queries"; the rest of least squares'
  lead is then named by the decomposition (format: free value columns; features). If OPTR - STORE is "level": the paper
  says that at the store's own format and bits the Hebbian write costs no measurable accuracy and least squares' lead is
  its format and features. The same for the strongest variant with OPTX - VSAPCX. OPTR / OPTX are named as diagnostic
  variants, not as the store's design.
- (B) The paper states, per K, the labels of STORE - LAMC, VSAR - SDMT and STORE - NLMSC with everything charged
  symmetrically, and replaces B71's D2 reading ("charged beyond what the compared store uses") by them; VSAPcX - LAMc
  under (B) is not measured and must not be claimed.
- (C) Methods may state the t-interval and permutation results; any headline label that is not store-robust must be
  stated as such.
- (D) "Least squares' features were linear only" is answered by LAMC+; if LAMC+ - LAMC is level everywhere the paper
  says sign features did not help least squares; any encoder where LAMc+ leads LAMc must be named. The conditions where
  the store leads LAMc are named.

## Checks (b72_check.py; second angle)

- **Plain-loop scorer**: every R1 cell accuracy of every new arm and every registered point statistic recomputed with
  plain Python loops from the job files and merged records; |difference| < 1e-9; (C)'s T recomputed by plain loops over
  stores equals the region statistic to 1e-12, and the t interval recomputed by an independent plain-loop formula.
- **Bits audit**: every dev and test row of every new arm: OPT d by an own re-implementation of B32's size rule at the
  row's q (bits <= budget, d maximal; kind P d = min(rule, n)); LAMP p by an own re-implementation of LAMc's bit formula
  (<= budget, maximal unless at its cap); B arms: stored bits + charge <= budget and maximal by own formulas.
- **Float64 re-derivation (independent code) of the optimal-format solve on a sample**: one test store per (encoder, M,
  V) at 4 and 16 bits per fact, OPTR (and OPTP on the same stores): own float64 features (kind R: b66_lsfc.own_feats_R;
  kind P: own eigh basis), own codes (b71_check.own_codes), own PRIMAL ridge per class (numpy.linalg.solve of (F^T F +
  lam I) S = F^T y; independent of the dual / eigendecomposition path), own quantizer (b71_check.own_q_store), own read.
  Bar: agreement >= 0.99 with the logged predictions on every re-derived store.
- **Permutation check**: for every exact p-value a Monte Carlo estimate from 200,000 random sign vectors agrees within
  4 Monte Carlo standard errors (+ 1e-5).
- **Sample reading by the AI-assisted process**: Qwen3-8B and e5-base-v2, the R1 cell with the largest |OPTR - STORE|:
  10 queries only OPTR answers right and 10 only the store answers right (alias, label, gold, both answers), with notes.

## Compute and cut rule

CPU only. The A800 box (ssh cavin, 128 cores; fresh directory ~/b72_run, code by md5 from this repository, panel codes
md5-checked against ~/b71_run's copies; one process per job, OMP / BLAS threads 1, nice 5, at most 48 workers; another
agent (B73) uses the box's GPUs and some CPU) for dev and test jobs; local (at most 12 workers) for choices, verdict,
(C) and checks. Every ssh command in the background; no retry loops on state-changing remote commands; processes
stopped only by exact PID; nothing of other users touched. **Time box: 3 hours from 19:21 (machine clock).** If time is
short, cut in this order, in a dated compute note written before any test output is read: (B) first (within it: NLMSC,
then VSAR - SDMT, then K = 1e3), then (D)'s sentence encoders, then (C)'s secondary regions; (A) last.

## Files

b72_run.py (arms OPT, LAMP, BV, BL, BS; plans, dev choice, selftest), b72_box.sh, b72_verdict.py ((A), (B), (C), (D)),
b72_check.py; results/b72_*; job outputs under b72_out/ (not committed).
