# B60 pre-registration: the corrected least-squares associative memory (LAMc) against the store, on the full panel

Date: 2026-11-21 (US Eastern 2026-11-20 afternoon; machine clock Asia/Shanghai, about 03:45). Written and committed,
then OpenTimestamps-stamped, before any B60 development or test run. The paper is not edited by B60.

## What has been seen before this file (disclosed)

- **The diagnosis (exploratory, not pre-registered; scratchpad lsdiag/step3-step6).** On Qwen3-8B only, R1 cells
  only, ideal accounting, B49's own dev and test stores, a clean-room re-implementation found three defects in B49's
  LAM and estimated their effect: R1 store (B53) - LAM moved from +4.29 [+3.26, +5.26] to -3.61 [-4.65, -2.63] at
  V = 16, from -4.21 to -3.86 [-4.76, -3.04] at V = 2, and from +0.04 to -3.73 over R1, when all three were fixed
  (grid: B49's 4 alphas, prep x feat x q, V-1 / V targets, per-column / flat scales; tie rule 0.1 points, grid
  order). LAM's test gain from the fix: +7.89 points at V = 16, -0.30 at V = 2 (Qwen3-8B, R1). **Qwen3-8B's test
  outcome is therefore known in advance; the other six encoders are blind.** No other encoder, region, accounting or
  map has been looked at with any corrected LAM.
- **B60 runs before this file (training split only, no dev or test store):** b60_run.py's selftest on random toy data
  (bit rules within budget and maximal on every (V, target, scale, q, acct, M, bpf, pmax); toy reads >= 0.95; the
  'col' + 'V-1' configuration reproduces b49_common.run_lam on rp features, agreement >= 0.995); the basis-overlap
  selftest (results/b60_overlap.json, below), which uses only the training split's codes; timing runs of the engine on
  one TRAINING-split store (results/b60_timing_train.txt).

## The defect (why)

B49's LAM (b49_common.LamFeat / run_lam), the paper's key equal-bit baseline, has three problems:
1. **Wrong 'pca' basis for prep = whiten (a bug).** LamFeat computes its principal directions on the already
   preprocessed codes; for whiten these are whitened (and row-normalised) codes, whose spectrum is nearly flat, so the
   "principal" basis is not principal. Overlap of its top-p directions with the true top-p eigenvectors of the raw
   codes' training covariance (b60_run.py overlap, training split): Qwen3-8B 0.077 / 0.127 / 0.227 / 0.485 / 0.903 at
   p = 8 / 32 / 128 / 512 / 2,048; Granite 0.086 / 0.130 / 0.234 / 0.467 / 0.889 (other encoders in
   results/b60_overlap.json).
2. **V - 1 target columns.** Value 0's score is minus the sum of the other V - 1 stored columns, so it collects every
   column's quantization noise.
3. **Per-column scales.** Each column gets its own 32-bit scale per 8,192-row block; at V = 16 (15 columns) these
   scales take 23-47% of the R1 budgets.
The paper currently says that with 16 values the store leads least squares on language-model codes (+2.3 to +4.7);
the diagnosis suggests that this is an artefact of (1)-(3). B60 tests that on all seven encoders, every region and both
accountings, with the store arms' existing test predictions.

## Fixed (B49 / B53 / B56 / B57 / B59's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (run from
b49_out/ whose b32_data holds every panel code; md5s as B49), splits, B32's make_store stores (dev 6 + 3, test 20 + 5
per (M, V) at M = 256 / 1,024), queries (unseen aliases), budgets 1, 2, 4, 8, 16, 32, 64 bits per fact, accountings
(ideal primary; real = q bits per number, secondary), regions as B49 / B56: **R1** = V in {2, 16} x 4, 8, 16 bits per
fact x both M (12 cells), R1-V2 and R1-V16 its halves; **R0** = 1, 2 bits; **RH** = 32, 64 bits; **R2** = V = 1,024 x
all seven budgets. The store arms are NOT re-run (except VSAPc below); their test predictions come from the merged
files b59_out/fam/{enc}/all59_* (V 2 / 16; they contain B53's VSA, B49's VSAR, B56's VSAS, B57's VSAP, the per-map
VSAm{m} / VSASm{m}) and b49_out/fam/{enc}/test/all49_*_V1024 (R2: B49's VSA). The other nine families' dev
accuracies and test predictions (for BESTc) are B49's (results/b49_choice_{enc}.json 'dev_by_fam'; merged files).

## LAMc (b60_run.py)

B49's LAM algebra (ridge W = F^T (F F^T + lam I)^-1 T, lam = alpha x mean diag(F F^T); all M facts; read
score(v) = f(q)^T W_hat t(v), argmax, first index on exact ties; p = 0 -> the store's majority value) with:
- **Feature maps.** 'rp': B49's Gaussian projection blocks, unchanged (seed ('LAMG', enc, prep, j0)), p <= 65,536.
  'pca', prep = whiten (CORRECTED): phi(x) = P_w(x) @ U0[:, :p], where U0 = the eigenvectors of the centred training
  covariance of the RAW codes (numpy.linalg.eigh of np.cov of B32's training matrix minus its mean, exactly the
  decomposition b32_common.Prep('whiten') whitens with), ordered by descending eigenvalue, and P_w = Prep('whiten')
  (ZCA with eps = 0.1 x the mean eigenvalue, then the row normalisation of the full n-dimensional code). This equals the
  prep's scaling on the principal axes, ((x - mu) U0 / sqrt(w + eps)) / norm, i.e. divide by sqrt(w + 0.1 mean(w)),
  then normalise the full row, then keep the first p coordinates. p <= n. 'pca', prep = center: P_c(x) @ U0[:, :p]
  (the centred, row-normalised code on the same principal axes; no second centring). Disclosed: the first draft of
  this design kept B49's LamFeat basis for center (the principal directions of the centre-prepped, row-normalised
  training codes, which the diagnosis kept); the overlap selftest on the training split, run before this file, gave
  that basis 0.948 on Mistral-7B at p = 32 (0.951 Granite at p = 512; >= 0.955 elsewhere;
  results/b60_overlap_pre_lamfeat_center.txt), under the 0.95 bar, so both preps use U0. The center 'pca' map
  therefore differs slightly from B49's and the diagnosis's (overlap 0.95-0.99).
- **Basis-overlap selftest (bar).** For every encoder and prep, at p in {8, 32, 128, 512, 2,048} (p <= n): overlap =
  ||U_ref[:, :p]^T Q_p||_F^2 / p, U_ref = an independent float64 eigh of the raw codes' centred training covariance,
  Q_p = an orthonormal basis of the pca map's first p directions. Bar: >= 0.95 everywhere (1.000 by construction
  for both preps; the check is that the code uses the right order and decomposition, against an independently
  computed float64 basis). B49's LamFeat overlaps (center and whiten) are reported beside it.
- **Targets** (both in the grid, each charged its own columns): 'V-1' (B49: centred one-hot, columns of values
  1..V-1; value 0 = minus their sum) and 'V' (all V columns e_v - 1/V stored and read). R2 (V = 1,024) also B49's
  'code32' and 'code128' (random +-1 value codes, read by correlation).
- **Scales** (both in the grid): 'col' (B49: one 32-bit scale per column and per block of 8,192 rows) and 'flat' (the
  store's rule: the p x c matrix flattened column by column, one 32-bit scale per block of 8,192 stored numbers).
  Quantizer per scale group: q = 1 sign x the group's mean |w|; q >= 2 b32_common.quantize (Max's uniform step on the
  group's std; the store's quantizer). Ties at exactly 0 for q = 1: random signs from seed ('B60Q', ...).
- **Bits.** ideal: c p H_q + 32 x (number of scale groups) (H_q = b32_common.H_Q); real: q instead of H_q. p = the
  largest that fits (and maximal: bits(p + 1) > budget unless p is at its cap).
- **Grid** (order = tie order): scale (col, flat) > target (V-1, V [, code32, code128]) > (prep, feat) (center rp,
  center pca, whiten rp, whiten pca) > alpha (1e-4, 1e-2, 0.1, 1, 10, 100) > q (1, 2, 4). 288 configurations at
  V in {2, 16}; 576 at V = 1,024. (B49's LAM: 48.)
- **Dev choice (B49's rule).** Per cell (M, V, bpf, acct): the highest mean dev accuracy over the cell's dev stores;
  within 0.1 points tied, first in the grid order above. Test: the chosen configuration only, run once, after the
  choices are committed.
- **BESTc.** B49's ten-family choice (IDX, SF, CM, BF, FBF, SDM, LAM, LHI, LHC, LHS; highest dev accuracy of each
  family's chosen configuration; exact ties in that order) with LAM's dev accuracy replaced by LAMc's; test predictions
  = LAMc's when LAMc is chosen, else the family's B49 predictions.

## Store arms compared (existing test predictions; no store re-run except VSAPc)

- **STORE = B53's VSA** (antipodal / orthogonal value codes; the final design's matched filter): the primary arm.
- VSAR = B49's store (first design, random value codes); R2: B49's VSA.
- VSAS = B56's store (dev-selected LMMSE read or retrained bundle).
- VSAP = B57's store on principal components and VSAPX = per cell the dev-best of VSAS and VSAP (B57's rule). B57's
  VSAP used b49_common.LamFeat's basis, so its whiten configurations (chosen in most R1 cells) are on the same wrong
  basis. **VSAPc** (secondary, compute permitting): B57's VSAP procedure re-run with the corrected whiten basis
  (FeatP with U = U0 and mu = 0 for whiten; center parts = B57's dev rows, unchanged code and seeds): B57's 48
  matched-filter configurations, B32's rule, B56's LMMSE / retraining reads on the chosen bundle, B56's STAR rule;
  VSAPcX = per cell the dev-best of VSAS and VSAPc (ties VSAS). If VSAPc is not run, B57's VSAP / VSAPX are marked
  "superseded (wrong whiten basis)".

## Primary bar (ideal accounting; 95% cluster bootstrap, entity within store, resampled within each cell; 10,000 draws)

Per encoder, **STORE - LAMc** (accuracy points; region statistic = mean over cells of the cell mean; b36_verdict's
boot_region), reported SEPARATELY on **R1-V2**, **R1-V16** and **R1**. Decision rule, fixed now, for each of the three:
- "store leads" iff the 95% lower bound > 0;
- "LAMc leads" iff the 95% upper bound < 0;
- otherwise "level".
Counts: store leads / level / LAMc leads, over the 4 language models and over the 3 sentence encoders, for each of
R1-V2, R1-V16 and R1. Also reported (continuity with B56 / B59): "not behind" (lower bound >= -2), "within the margin"
(95% interval inside (-2, +2)).

## Secondary (all with the same statistic; 2,000 draws unless stated)

1. STORE - BESTc (R1, R1-V2, R1-V16; 10,000 draws), and B49's H1 reading (>= +2 with the interval above 0 = store
   wins); count of conditions in which BESTc picks LAMc (paper: least squares the best of ten in 51 of 84).
2. VSAS - LAMc, VSAP - LAMc, VSAPX - LAMc, VSAR - LAMc (and VSAPc - LAMc, VSAPcX - LAMc, VSAPc - VSAP if run).
3. Chance-rescaled versions of STORE - LAMc, STORE - BESTc, VSAR - LAMc: per cell (acc_a - acc_b) / (1 - 1/V), then
   the region mean.
4. **LAMc - LAM** (B49's LAM, its own dev choice) per encoder: R1, R1-V2, R1-V16, R0, RH, R2; and the dev choices'
   composition (scale, target, prep, feat) over the R1 cells.
5. R0, RH (V 2 / 16) for STORE - LAMc and STORE - BESTc; R2 (V = 1,024): B49's store - LAMc and - BESTc.
6. Real accounting: every primary and secondary pair.

## Map variance (absorbs B59 part (1), stopped because it used B49's LAM)

Maps m = 1, 2, 3: seed tag `{enc}#b57m{m}` replaces the encoder name in LAMc's rp projection seeds (B57's tag); the
store's per-map test predictions are B57 / B59's (VSAm{m}: the final design's matched filter, the map's own B32 dev
choice; VSASm{m}: B56's read rules on that bundle). Per map, LAMc is re-chosen on dev by the rule above over the full
grid: the rp configurations re-run with the map's projection on the dev stores, the pca configurations' dev rows from
map 0 (no random component). Test: the map's chosen configuration. All seven encoders, V in {2, 16} (R0, R1, RH from
the same runs), ideal and real. Statistics (B52 / B57 / B59's two-level bootstrap, b59_verdict.twolevel: maps drawn
with replacement from the four, entity clusters resampled once per draw and shared across maps; 10,000 draws): for
STORE (VSAm{m}) - LAMc(m) and VSAS(m) - LAMc(m), per map the region difference, the map mean, the between-map SD, the
two-level 95% interval, the registered map's rank; R1, R1-V2, R1-V16. The primary decision rule is applied to the map
mean with the two-level interval (reported beside the registered-map label; the primary bar itself is the registered
map's, as registered above).

## Predictions (stated so that they can be wrong)

P1. R1-V16: LAMc leads on at least 3 of the 4 language models; the store leads at V = 16 on none of the 7 encoders.
P2. R1-V2: LAMc leads on all 4 language models.
P3. R1: LAMc leads on at least 6 of 7 encoders; the store leads on none.
P4. LAMc - LAM: R1-V16 between +3 and +10 points on each of the 7 encoders, interval above 0 on all 7; R1-V2
    |LAMc - LAM| <= 1 point on at least 6 of 7.
P5. STORE - BESTc over R1: BESTc ahead (upper bound < 0) on all 7; at R1-V16 BESTc ahead on at least 6 of 7.
P6. VSAS - LAMc and VSAPX - LAMc over R1: LAMc ahead (upper bound < 0) on at least 6 of 7 for each.
P7. Dev choices over the R1 V = 16 cells (ideal, pooled over the 7 encoders, 6 cells each): 'flat' scales in at least
    two thirds, 'V' targets in at least half; on the 4 language models whiten 'pca' chosen in at least half of all
    R1 cells.
P8. VSAPc (if run): |R1 VSAPc - VSAP| <= 1.5 points on at least 5 of 7.
P9. Maps: between-map SD of R1 STORE - LAMc <= 0.8 points on each of the 7 encoders; the map-mean labels (two-level
    interval) equal the registered-map labels in at least 19 of the 21 (encoder x {R1-V2, R1-V16, R1}) cases.
P10. R2 (if run): |LAMc - LAM| <= 2 points on at least 5 of 7.
P11. Checks pass: plain-loop scorer |difference| < 1e-9; 0 configurations over budget and 0 not maximal; overlap
    >= 0.95 everywhere; step5 agreement >= 0.99 on every common configuration with identical maps; float64
    re-derivation >= 0.99.

## Checks (b60_check.py; second angle)

- **Plain-loop scorer**: every cell accuracy (STORE, LAMc, BESTc, LAM) and the R1 / R1-V2 / R1-V16 point statistics
  recomputed with plain Python loops from the job files and the merged files; |difference| from the verdict < 1e-9.
- **Bits audit**: every logged test row and every dev row: bits <= budget, and maximal (bits(p + 1) > budget unless p
  is at its cap), by own re-implementation of the bit formula.
- **Basis-overlap selftest** (above).
- **Agreement with the diagnosis's step5 implementation** (Qwen3-8B): step5 saved dev and test predictions of its
  192 fixed configurations (B49's 4 alphas) at M = 256 / 1,024, V = 2 / 16, 4 / 8 / 16 bits per fact, ideal. On the
  dev stores (where B60 has every configuration): per configuration, the fraction of identical predictions, pooled over
  the stores; bar >= 0.99 for every configuration whose maps are identical in both codes (rp both preps, whiten pca);
  center pca reported without a bar (step5 used B49's LamFeat center basis, B60 uses U0). Expected differences: q >= 2
  'flat' groups use the std (B60, the store's quantizer) vs the root mean square (step5). On the test stores: the same
  for the configurations B60 chose; and the R1 cell accuracies of B60's LAMc against step5's selection, reported.
- **Float64 re-derivation (independent code)**: on one test store per (encoder, M, V) at 4 / 16 / 64 bits per fact,
  ideal, the chosen LAMc configuration recomputed by a separately written primal / dual ridge solve in float64
  (numpy.linalg.solve), own quantizer, own read; agreement with the logged predictions >= 0.99.
- **Sample reading by the AI-assisted process**: Qwen3-8B and e5-base-v2, the R1-V16 cell with the lowest
  STORE - LAMc: 10 queries only LAMc answers right and 10 only the store answers right (alias, label, gold, both
  answers), with notes.

## Readings (fixed now) for the paper's current equal-bit claims

- "With sixteen values the store led least squares by 2.3 to 4.7 on language-model codes": SURVIVES only if R1-V16
  "store leads" on all 4 LLMs; FLIPS if "LAMc leads" on at least 3 of 4 (the paper must then say least squares led
  with sixteen values too); otherwise it must be restated per encoder (leads / level as labelled).
- "Least squares led with two values by 3.5 to 4.2": survives (numbers updated) iff R1-V2 "LAMc leads" on all 4 LLMs.
- "Averaged over the region the store was within the +-2 margin of least squares on the four language models"
  (from B53 / B56): restated from R1 STORE - LAMc; "within the margin" survives only for encoders whose 95% interval
  lies inside (-2, +2); if LAMc leads over R1 on an encoder, the paper must say least squares led over the region.
- "Against the best of ten it was below on all seven, by 0.9 to 4.2": recomputed with BESTc.
- "The first design ... trailed least squares by 6.6 to 9.3 with two values ... and led it by 1.3 to 2.9 with
  sixteen": recomputed with VSAR - LAMc; "led with sixteen" survives only with "store leads" on all 4 LLMs.
- Introduction / Discussion "whereas with sixteen values the store led": as the first bullet.
- "A store choosing per condition between random and principal-component features ... led least squares by 0.7 to
  1.6 points on the four language models": recomputed with VSAPX - LAMc (and VSAPcX if run); MUST GO if LAMc leads.
- "Least squares was the development-chosen best of the ten in 51 of the 84 conditions": recounted with BESTc.
- Map-variance sentences (SDs, map means against least squares): replaced by B60's map numbers against LAMc.
- RH: "neither store reached 90% of an exact index ... best competitor did in 19": recounted with LAMc in the best of
  ten (B49's rate rule, b36_verdict.rate).
- Numbers that B60 does not re-run and that use B49's LAM structure: LAMRF (B56: no basis defect; targets / scales
  defects bind only at V = 16) and NLMS / RLS (B57: LAM's features, targets and quantizer, incl. the wrong whiten
  basis). Their V = 2 statements stand as V = 2 statements with problem (1) disclosed for NLMS / RLS; their V = 16 and
  R1 numbers are marked "uncorrected baseline".
- Whatever B60 shows, the paper must not present the corrected comparison as a new design of the store; the store's
  numbers are unchanged, only its baseline is corrected.

## Compute and time rule

CPU only. The A800 box (ssh cavin): at most 48 cores, fresh directory ~/b60_run (code by md5 from this repository,
codes by md5 against B49's), jobs one process each (OMP_NUM_THREADS = 1), with the total machine load kept reasonable
next to B59 part 2; local: VSAPc (B57's code path; at most 4 processes), the agreement check and the verdicts. Results
pulled back into b60_out/ and results/b60_*. Time budget: 3 hours from about 03:06 (machine clock). Priority if time
is short: (1) R1 for all 7 encoders, ideal, primary bar (one pass of every V 2 / 16 budget gives R0 / RH too);
(2) secondary arms (BESTc, VSAS / VSAP / VSAPX / VSAR, VSAPc); (3) map variance; (4) R2. Any cut is written into a
dated compute note before any test output is read.

## Files

b60_run.py (engine, plans, dev choice), b60_vsapc.py (VSAPc wrapper of B57's code), b60_verdict.py, b60_check.py,
b60_box.sh; results/b60_*; job outputs under b60_out/ (not committed).
