# B71 pre-registration: the write rule against the features, rotated principal features, the store written by a delta rule, and side information, at equal bits over R1

Date: 2026-11-23 (machine clock Asia/Shanghai, about 18:10; US Eastern 2026-11-23 05:10). Written and committed, then
OpenTimestamps-stamped, before any B71 development or test run on a development or test store. The paper is not
edited by B71.

## Why (referee M2, M3)

The paper says least squares (LAMc, B60) and its one-pass online form (NLMSc, B62) beat the Hebbian store at equal bits
on every encoder over R1. The referee asks: (A) per encoder, with intervals, how the strongest Hebbian variant (VSAPcX:
random or principal features chosen per condition, corrected basis) and B56's retrained / decorrelating-read store
(VSAS) stand against LAMc and NLMSc, with the write-rule effect separated from the feature effect; (B) whether a
Hebbian store on rotated principal features, which removes the coupling between the Hadamard value codes and the
principal coordinates (Supplementary Note 7), closes the gap; (C) whether the store written by an error-driven delta
rule on its own bundle and value codes (same vector, same read, same bits) closes it, and why B56's retraining gained
at most 0.8 points while NLMSc led the store by 1.5 to 4.8; (D) what uncharged side information each arm uses, and
whether the labels change when it is charged.

## What has been seen before this file (disclosed)

- Every verdict and note of B49-B70. In particular the point estimates of (A)'s contrasts between EXISTING arms are
  differences of published region means over the same cells and queries, so they are known up to arithmetic. Over R1,
  VSAPcX - NLMSc = (VSAPcX - LAMc) - (NLMSc - LAMc): Qwen3-8B -0.06, Granite -0.80, Llama +0.03, Mistral -3.46, e5
  -2.26, bge -1.32, MiniLM -1.51; VSAS - NLMSc: -1.78, -2.33, -1.09, -4.94, -3.29, -1.32, -1.83; VSAS - LSFR (B66's least
  squares on the store's own random features): -1.18, -1.33, -1.26, -1.64, -1.52, -1.55, -1.55. Their intervals (one-
  and two-level) and every quantity that involves an arm new in B71 (NLMSFR, NLMSFP, VSAPR, VSAD, the audit) are not
  known.
- B56's committed dev choices (results/b56_choice_*.json; read for this design): its retraining arm chose the smallest
  step (eta = 0.1) in most sixteen-value cells and eta = 1 with up to 10 epochs (the grid's edge) in several two-value
  cells; on dev, retraining was about level with the Hebbian bundle at V = 16 and up to +6.5 points above it at V = 2
  (Mistral-7B, M = 1,024).
- B71 code runs before this file: `b71_run.py selftest` (toy data and TRAINING-split stores only: one pass of the
  'nlms' delta rule equals per-class dual NLMS (forward substitution) to < 1e-8; retrain_count equals b56_run.retrain
  bit for bit; FeatPR's value codes equal B53's Feat53 codes; rotation blocks orthonormal; VSAPR within budget and
  d maximal) and `b71_run.py timing` on TRAINING-split stores (b71_out/logs/timing_train_*.log). No development or test
  store has been answered by any B71 arm.

## Fixed (B49 / B53 / B60 / B62 / B66's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (b49_out/
b32_data, md5s as B60 / B62 / B66), splits, B32's make_store stores (dev 6 + 3, test 20 + 5 per (M, V)), queries (unseen
aliases), **R1 only** = V in {2, 16} x 4, 8, 16 bits per fact x M in {256, 1,024} (12 cells per encoder), reported as
R1-V2, R1-V16 and R1; **ideal accounting only** (B62's NLMSc exists only there). Existing arms are not re-run; their test
predictions are read as B60 / B62 / B66's verdicts read them: STORE = B53's final design (VSA), VSAS = B56's dev-selected
LMMSE read or retrained bundle, VSAPC = B60's store on the corrected principal basis, VSAPCX = per cell the dev-best of
VSAS and VSAPC (b60_verdict.add_vsapx), LAMC = B60, NLMSC = B62 (b62_verdict.add_b62), LSFR / LSFP = B66's least squares
(LAMc's options) on the store's random / principal features (b66_lsfc.add_lsf).

## New arms (b71_run.py; seeds registered in the code)

1. **NLMSFR / NLMSFP (A): the one-pass delta rule on the store's own features.** B62's NLMSc engine
   (b62_run.run_online, unchanged: W_t = W_{t-1} + mu f_t e_t^T / ||f_t||^2 in the store's write order, output last or
   Polyak average, LAMc's targets V-1 / V, scales col / flat, quantizer, bits, read) with the feature map replaced by the
   store's: R = the final design's random map of the cell (B53's chosen prep / fmap, the store's projection blocks;
   b66_lsfc.FeatStoreR, p <= 65,536), P = VSAPc's principal map of the cell (b66_lsfc.FeatStoreP, p <= n). Grid (tie
   order): scale > target > mu (1, 0.5, 0.25, 0.1, 0.03) > output (last, avg) > q (1, 2, 4) = 120 per kind. B60's dev rule.
   **NLMSFM** = per cell the kind that matches VSAPcX's source (R where VSAPcX = VSAS, P where VSAPcX = VSAPC);
   **LSFM** likewise from B66's LSFR / LSFP. (A test of these two per-cell compositions needs no new choice.)
2. **VSAPR (B): the Hebbian store on rotated principal features.** phi(y) = fmap(((P(y) - mu) U_p) Rot): U_p = the
   store's principal basis (b66_lsfc.basis: centre = B57's basis of the centre-prepped codes; whiten = the corrected
   U0, mu = 0), the top p directions; Rot = a p x d matrix whose columns are those of independent Haar-random orthogonal
   p x p matrices placed side by side (seed ('B71ROT', enc, prep, fmap, p, m, k)), so that for d <= p the map is a
   random rotation of the top-p principal coordinates cut to d, and for d > p a union of independent rotations (every
   residue class then reads the whole principal subspace); fmap lin / sgn / sgn3 (sgn3: three independent rotations,
   product of signs). B53's value codes (the random store's, key (enc, prep, fmap)), quantizer and seeds, stored-subset
   order and matched-filter read through b32_run.vsa_grid unchanged; d = B32's size rule (not capped at n). **Grid** =
   the store's 72 (prep {center, whiten} x fmap {lin, sgn, sgn3} x q {1, 2, 4} x share {1, 1/2, 1/4, 1/8}) x p in P(n) =
   {64, 256, 1,024} below n, plus n (language models 4 values = 288 configurations; e5, bge, MiniLM 3 = 216). Dev rule:
   B32's (highest mean dev accuracy; within 0.1 points tied, first in prep > fmap > p > q > share order). p = n with
   whiten is, up to the orthogonal-vs-Gaussian projection, the random store, so the grid contains a near-copy of the
   store. **VSAPRX** (secondary) = per cell the dev-best of VSAPcX and VSAPR (within 0.1 points: VSAPcX).
3. **VSAD (C): the store written by a delta rule on its own bundle.** With x = phi(k) * r (r = the value codes' row-0
   signs) and the coordinates grouped by residue class c = j mod 16 at V = 16 (one class at V = 2), the store's
   matched-filter score of value u is sum_c H16[u, c] m_c with m_c = <x_c, S_c>. Per fact in store order: e_c = y_c - m_c,
   y_c = H16[v, c] (V = 16) or +1 / -1 (V = 2), S_c += mu e_c x_c / ||x_c||^2; rule 'nlms' = every fact, rule 'err' =
   only facts that the current bundle misreads (argmax of the matched-filter scores != v; perceptron-style). S_0 = 0;
   float64 accumulator; E passes over the stored facts in store order; quantized ONCE at the end by the store's quantizer
   (b56_run.quant_np, seeds ('B71Q', ...)) to q bits with d = B32's size rule; read by the matched filter (b56_run.mf_read).
   Same vector, value codes, read and bits as the store. **Kind R** = on B53's chosen (prep, fmap, share) of the cell
   (the bundle B56 retrained); **kind P** = on VSAPc's chosen (prep, pfmap, share), d <= n (as VSAPc). Grid: E = 1: rule
   (nlms, err) x mu (1, 0.5, 0.25, 0.1, 0.03) x output (last, Polyak average) x q = 60; E in {3, 10}: rule x mu x q, last
   = 60; tie order E > rule > mu > output > q. Arms: **VSADR1** (kind R, E = 1 only: the one-pass online write; primary
   for (C)), **VSADRE** (kind R, all 120: few passes, full-precision accumulation, quantized at the end), VSADP1 / VSADPE
   (kind P), **VSADX** = per cell the dev-best of VSADR1 and VSADP1 (within 0.1 points: VSADR1). B60's dev rule.
4. **AUDIT of B56's retraining (dev only, kind R, B53's configuration of each R1 cell).** For q in {1, 2, 4}: the
   Hebbian bundle quantized by B53's quantizer stream (b56_run.bundle_q) and unquantized; B56's retraining (b56_run.
   retrain's arithmetic, its quantizer seeds) on B56's grid (E 1, 3, 10 x eta 1, 0.1) and an extended grid (E 1, 3, 10,
   30 x eta 1, 0.1, 0.01), each read quantized and unquantized; for every arm the accuracy on the stored keys themselves
   (labels) and the number of corrected facts per epoch. B56's settings, from its code (b56_run.retrain, run_store_group):
   float64 accumulator (the read during training in float32), Hebbian initialisation, mini-batches of 32 in store order,
   update S += eta phi(k) * (c(v) - c(v_hat)) only on misread facts, no margin, no step normalisation, eta in {1, 0.1},
   E in {1, 3, 10}, quantized after training to the re-chosen q.

Test: the dev-chosen configuration of every new arm, run once, after the choices are committed.

## Statistics and decision rule

Region statistic = mean over cells of the cell mean of the per-query difference; **one-level** 95% cluster bootstrap
(entity within store, resampled within each cell; b36_verdict.boot_region), 10,000 draws for primaries, 2,000 otherwise;
**two-level** = B66's registered store-level bootstrap (b66_storeboot.cell_draws: test stores drawn with replacement, then
each drawn store's entity clusters; conservative by construction, about 1.3-1.4 times as wide with no store effect),
10,000 / 2,000 draws; where map replicates exist (STORE - LAMc, VSAS - LAMc), B60's map-level numbers are cited beside.
**B60's decision rule**: "x leads" iff the 95% lower bound > 0, "y leads" iff the upper bound < 0, else "level"; the
one-level label is the registered label, the two-level label is reported beside it. Counts over the 4 language models and
the 3 sentence encoders.

**Closing the gap (B and C), fixed now:** a remedy X_new closes the gap to Y in {LAMc, NLMSc} on an encoder iff R1
(X_new - Y) is "level" or "X_new leads" (one-level). Also reported: the share of the gap closed, (X_new - STORE) /
(Y - STORE) over R1 (descriptive).

**Write rule vs features (A), fixed now.** Exact per-query decompositions over the same cells and queries:
- batch: VSAPCX - LAMC = [VSAPCX - LSFM] (write rule at VSAPcX's own features) + [LSFM - LAMC] (features at the
  least-squares write); VSAS - LAMC = [VSAS - LSFR] + [LSFR - LAMC];
- online: VSAPCX - NLMSC = [VSAPCX - NLMSFM] + [NLMSFM - NLMSC]; VSAS - NLMSC = [VSAS - NLMSFR] + [NLMSFR - NLMSC].
Per encoder over R1: "write rule dominant" if |write term| >= 2 |feature term|, "features dominant" if |feature term|
>= 2 |write term|, else "both"; each term with its one-level interval and label.

**Cause of B56's small gain (C), fixed now.** Exact decomposition on test over R1:
NLMSC - VSAS = [NLMSC - NLMSFR] (features: LAMc's dev-chosen map vs the store's random map, at the one-pass delta write)
+ [NLMSFR - VSADR1] (layout: least squares' shared features and c columns vs the store's residue classes and Hadamard
codes, at the same features and rule family) + [VSADR1 - VSAS] (write rule on the store's own bundle: one delta pass vs
B56's read rules). The term with the largest mean over the 7 encoders is named the main cause; any term >= 1/3 of the
total on at least 4 encoders is named as contributing. The dev audit reports, per encoder over R1 (dev means, B53's
configuration): quantisation loss of B56's retraining (unquantized - quantized read at its dev-chosen (q, E, eta)), the
extended grid's gain (extended dev-best - B56-grid dev-best), the delta rule's 'err' dev-best vs 'nlms' dev-best (the
cost of updating only misread facts), the stored-key accuracy of the Hebbian bundle and the number of corrected facts per
epoch. A dev factor is named as contributing as follows: quantisation if its dev loss is >= 0.5 points on at least 4
encoders; the grid if the extended grid adds >= 0.5 points on at least 4; 'misread-only updates' if the 'nlms' rule's
dev-best beats the 'err' rule's by >= 0.5 points on at least 4.

## Contrasts (all per encoder over R1, R1-V2, R1-V16; ideal)

Primary (10,000 draws, one- and two-level):
- (A) VSAPCX - LAMC, VSAPCX - NLMSC, VSAS - LAMC, VSAS - NLMSC; the eight decomposition terms above.
- (B) VSAPR - LAMC, VSAPR - NLMSC.
- (C) VSADR1 - LAMC, VSADR1 - NLMSC; the three cause terms.
Secondary (2,000 draws, one-level unless stated): VSAPR - VSAPC, VSAPR - STORE, VSAPR - VSAPCX, VSAPRX - LAMC / NLMSC;
VSADR1 - STORE, VSADR1 - VSAS, VSADRE - VSADR1 (several passes), VSADX - LAMC / NLMSC, VSADP1 - VSAPC; NLMSFR - LSFR and
NLMSFP - LSFP (online vs batch at the store's features); dev-choice composition of every new arm.

## (D) Side information and operations (analysis of saved configurations; no new run unless stated)

**Table.** For every arm (STORE, VSAR, VSAS, VSAPC, VSAPCX, VSAPR, VSAD, LAMC, NLMSC, SDMT, the learned hashing map of
LHI / LHC / LHS, BEST7's and BESTc's chosen families), at its chosen configuration per R1 cell, the uncharged fixed
components in entries and in bits at 8 bits per entry (and at 32): random projections / value codes / rotations /
location order = one 64-bit seed each; mean vector n; whitening (ZCA) matrix n (n + 1) / 2 (symmetric); principal basis
n x p (the first p eigenvectors; derivable from the whitening matrix, which has the same eigenvectors, so not counted
again when the arm also whitens); the LMMSE read's training statistics (T training codes, T x n); SDMt's N hard
locations N x n (N = the chosen N, at most 2,818 / 2,803 / 5,621); the learned map n x r + n (LearnedPrep, r = its chosen
rank) and its index's codebooks; median and range over the 12 R1 cells, and the ratio to the cell's budget. Also per
arm the write and read operations per fact (multiply-adds, B62's cost model extended to the new arms: VSAPR features
n p + K p d, VSAD E passes x about 3 d per fact, B56's retraining E x (V d + d) per fact; read: features + V d for the
store family, n p + p c for least squares).

**Charged re-scoring (two rules, fixed now).**
- (D1) full charge at 8 bits per entry, per store: an arm whose side information alone exceeds the budget cannot be
  stored and answers the default value. Reported as the count of R1 cells in which each arm's side information exceeds
  the budget, and the number of stores K* over which it would have to be amortised to fall to 10% of the budget.
  Labels of LAMc - STORE and SDMT - VSAR are recomputed under D1 only if at least one arm of the pair is storable in at
  least one cell; otherwise D1 is reported as "no arm storable".
- (D2) bounded to what the store itself uses: each competitor may use, free, the fixed objects that the compared store
  design uses in that cell (the same mean and whitening matrix, and everything derivable from them) plus further side
  information up to the store's own total side information in bits (8 bits per entry); anything beyond is charged
  against its budget. For LAMc - STORE (STORE's chosen prep per cell): LAMc's chosen configuration is kept if its excess
  is 0; otherwise LAMc is re-chosen on dev, by B60's rule, among its 288 configurations whose stored bits plus excess fit
  the budget (dev rows exist), and the re-chosen configurations are run on test once. For SDMT - VSAR (VSAR's prep per
  cell): SDMt's locations are capped at N_max = floor(allowance / (8 n)) (allowance = the store's total side information
  minus what SDMt shares with it); SDMt is re-run on dev with N <= N_max (B64's code, its N grid restricted) and
  re-chosen by B36's rule, then run on test once. Labels (B60's rule) and B36's bar for SDMT - VSAR, before and after.

## Predictions (stated so that they can be wrong)

- Q1 (A). VSAPCX - NLMSC over R1 (one-level): level on Qwen3-8B and Llama-3.1-8B; NLMSc leads on Mistral, e5, bge and
  MiniLM; level on Granite. VSAS - NLMSC: NLMSc leads on all 7.
- Q2 (A). Two-level labels differ from the one-level labels in at most 6 of the 84 (4 contrasts x 7 encoders x 3 regions).
- Q3 (A). Batch decomposition over R1: VSAPCX - LSFM below 0 (LSFM leads) on all 7; the feature term LSFM - LAMC is
  smaller in absolute value than the write term on at least 4 of 7 (VSAPcX already uses principal features where they
  help); for VSAS the feature term LSFR - LAMC is the larger on at least 5 of 7 ("features dominant" or "both").
- Q4 (A). Online: NLMSFR - LSFR over R1 below 0 on at least 5 of 7 (online loses to batch at the same features);
  NLMSFM - NLMSC "NLMSc leads" on at least 4 of 7.
- Q5 (B). VSAPR - VSAPC over R1-V16 > +3 points on each of the 3 sentence encoders and above 0 (interval) on at least 3 of
  the 4 language models.
- Q6 (B). VSAPR - STORE over R1: VSAPR leads on at least 4 of 7.
- Q7 (B). VSAPR - LAMC: LAMc leads on at least 6 of 7 (the gap to LAMc is not closed); VSAPR - NLMSC: closes the gap
  (level or VSAPR leads) on at least 3 of 7.
- Q8 (C). VSADR1 - STORE over R1: VSADR1 leads on at least 5 of 7; VSADR1 - VSAS: VSADR1 leads on at least 4 of 7.
- Q9 (C). VSADR1 - LAMC: LAMc leads on 7 of 7; VSADR1 - NLMSC: closes the gap on at most 2 of 7.
- Q10 (C). Cause: the feature term NLMSC - NLMSFR is the largest of the three on at least 4 of 7; the write-rule term
  VSADR1 - VSAS is > 0 on at least 5 of 7; |NLMSFR - VSADR1| <= 1 point on at least 5 of 7. Dev audit: quantisation loss
  of B56's retraining < 0.5 points on at least 5 of 7; extended grid adds < 0.5 on at least 5 of 7; 'nlms' beats 'err' by
  >= 0.5 points on at least 4 of 7.
- Q11 (C). Several passes: VSADRE - VSADR1 over R1 within +-0.5 points on at least 5 of 7.
- Q12 (D). D1: every arm's side information exceeds the whole budget in every R1 cell on the 4 language models (even the
  store's centring mean, 32,768 bits at 8 bits per entry against at most 16,384), so D1 is "no arm storable" there. D2:
  LAMc - STORE labels unchanged on at least 6 of 7; SDMT - VSAR: SDMt's lead lost ("level" or "first design leads") on at
  least 5 of 7.
- Q13. Checks pass (below).

## Readings for the paper (fixed now)

- "Least squares beat every Hebbian store variant on every encoder" stands for the batch solve only if VSAPCX - LAMC and
  VSAPR - LAMC are "LAMc leads" on all 7; any exception is named.
- "Also when written one fact at a time" may be said against the strongest Hebbian variant only on the encoders where
  VSAPCX - NLMSC (and VSAPR - NLMSC) is "NLMSc leads"; elsewhere the paper says the one-pass online write was level with
  the strongest store variant.
- (B): if VSAPR closes the gap to NLMSc or LAMc on an encoder, the paper says so as a result for the store; if it beats
  VSAPC at V = 16 by the predicted margin, the paper says the sixteen-value loss on principal features came from the
  coupling of Hadamard classes and principal coordinates (and the d <= n cap), and that a rotation removes it.
- (C): if VSADR1 or VSADX closes the gap to NLMSc or LAMc on an encoder, it is reported as a positive, VSA-native result
  (same vector, same read, same bits, one pass): the gap was the write rule, and a VSA memory written by a delta rule
  recovers it. If not, the paper states how much of the gap the delta-written store closes and the registered cause of
  B56's small gain.
- (D): the paper states the side-information table, that the comparison charges none of it (as the field does), and
  whether D2 changes a label; it may not say that the comparisons are side-information-neutral.
- The paper must not present B71 as changing the store's registered design; the new arms are named as variants.

## Checks (b71_check.py; second angle)

- **Plain-loop scorer**: every R1 cell accuracy of every new arm and every registered point statistic (contrasts and
  decomposition terms) recomputed with plain Python loops from the job files and merged records; |difference| < 1e-9.
- **Bits audit**: every dev and test row of every new arm: VSAPR / VSAD d by an own re-implementation of B32's size rule
  (bits <= budget, d maximal; kind P d = min(rule, n)); NLMSF p by an own re-implementation of LAMc's bit formula (<=
  budget, maximal unless at its cap).
- **Reproduction**: the audit's Hebbian bundle at B53's chosen q reproduces B53's dev accuracy in every R1 cell, and the
  audit's retraining on B56's grid reproduces B56's RT dev choice and dev accuracy in every R1 cell (|difference| <= 1e-12).
- **Float64 re-derivation (independent code) on a sample**: one test store per (encoder, M, V) at 4 and 16 bits per fact:
  NLMSFR by own float64 features (b66_lsfc.own_feats_R, the store's seeds), a genuine one-fact-at-a-time NLMS loop, own
  quantizer and read; VSAPR by own float64 basis (eigh of the training covariance), the same Haar seeds, own features,
  own Hebbian sum, own quantizer and read; VSADR1 by the per-class dual form (forward substitution) where rule = nlms and
  an own float64 loop where rule = err, own quantizer and read. Bar: agreement >= 0.99 on every re-derived store.
- **Sample reading by the AI-assisted process**: Qwen3-8B and e5-base-v2, the R1 cell with the largest VSADR1 - STORE: 10
  queries only VSADR1 answers right and 10 only the store answers right (alias, label, gold, both answers), with notes;
  and for VSAPR the R1-V16 cell with the largest VSAPR - VSAPC.

## Compute and cut rule

CPU only. The A800 box (ssh cavin, 128 cores; fresh directory ~/b71_run, code by md5 from this repository, panel codes
md5-checked against ~/b66_run's copies; one process per job, OMP / BLAS threads 1, nice 5, at most 48 workers, fewer if
other jobs need the cores) for the dev and test jobs; local for the choices, verdict and checks. On the box: every ssh
command in the background; no retry loops on state-changing remote commands; processes stopped only by exact PID;
nothing of other users touched. **Time box: 3 hours from 17:10 (machine clock).** If time is short, cut in this order,
in a dated compute note written before any test output is read: (0) kind P of VSAD and NLMSFP's secondary uses (VSADP*,
VSADX; NLMSFP stays as long as LSFM's online twin NLMSFM is needed), (1) the sentence encoders' new runs, (2) (D)'s
charged re-scoring D2 (the table stays), (3) VSAD's several passes (VSADRE).

## Files

b71_run.py (arms, plans, dev choice, selftest, timing), b71_box.sh, b71_verdict.py, b71_side.py ((D)), b71_check.py;
results/b71_*; job outputs under b71_out/ (not committed).
