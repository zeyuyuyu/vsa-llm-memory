# B74 pre-registration: write rule x features at the store's fixed format (2 x 3 factorial with the interaction), exact-label against alias queries, and the one-pass delta rule on a matched grid (R1, equal bits)

Date: 2026-11-23 (machine clock Asia/Shanghai, about 22:40; US Eastern 2026-11-23 09:40). Written and committed with
the run code, then OpenTimestamps-stamped; no B74 development or test run on a development or test store before the
stamp has a Bitcoin block attestation (or two hours have passed, stated in a compute note). The paper is not edited by
B74.

## Why (referee round 24, three points)

(1) B72's decomposition of least squares' lead, "write 1.0-2.4 (STORE -> OPTR) + format ~0 (OPTR -> LSFR) + features
1.9-4.8 (LSFR -> LAMc)", follows one path. Along the other path (Hebbian random -> Hebbian principal -> least squares
principal) the write term could be larger; without the interaction the attribution is not identified. (2) Alias queries
mix storage error (cross-talk among the stored facts) with key-noise generalisation (an alias's code is not the written
label's code). (3) The one-pass delta rule (NLMSc, B62) was tuned over 480 configurations, the store over 72.

## What has been seen before this file (disclosed)

- Every verdict and note of B49-B73. In particular the corner HR -> LR of (A) is B72's OPTR - STORE (+1.0 to +2.4 over R1,
  all 7 encoders), B71's VSAPR - STORE (+0.5 to +1.9; VSAPR at its OWN dev-chosen q and share, not the store's), B66's
  VSAPc - LSFP (-4.0 to -11.2; principal features, free columns, VSAPc's own q), B62's full-grid STORE - NLMSc
  (-1.5 to -4.8) and B71's VSAPcX - NLMSc (Qwen3-8B -0.06, Granite -0.80, Llama +0.03, Mistral -3.47, e5 -2.26,
  bge -1.32, MiniLM -1.52). No arm of (A) at the levels HP, LP, HT, LT below has been run on a development or test store;
  no label-query prediction of any arm exists; no 72-configuration NLMSc choice has been computed.
- Dev choice files read for this design (configuration fields only): B53's (prep, fmap, q, share) per R1 cell, VSAPc's
  matched-filter (prep, pfmap, q, share) (results/b60_choicepc_*.json '|VSA'), B71's VSAPR (prep, fmap, p, q, share),
  B72's OPT choices. Counted from them: the store's q differs from VSAPc's matched-filter q in 4-11 of the 12 cells per
  encoder and from VSAPR's in 2-5; the store's dimension d exceeds the code width n in 4 of 12 cells on every language
  model (M = 1,024, 8 / 16 bits per fact) and in 12 of 12 on the three sentence encoders.
- B74 code run before this file only by `b74_run.py selftest` and `b74_verdict.py selftest` (results/b74_selftest.log; its "LAMc(pca) label" print shows one budget only, a print bug fixed before this commit):
  toy data and TRAINING-split stores only, plus a recomputation of B62's committed full-grid NLMSc choices from B62's
  existing dev rows (84 / 84 cells on all 7 encoders reproduced; no sub-grid choice computed). Seen there: the label
  wrapper leaves every alias prediction of the STORE, LAMc and NLMSc engines identical (MiniLM, Qwen3-8B, V = 2 / 16);
  on TRAINING stores (M = 256, whiten / lin, q = 1) the store's label recall was 0.84-0.86 (MiniLM) and 0.93-1.00
  (Qwen3-8B) at V = 2 against alias recall 0.57-0.64, and 0.39-0.90 at V = 16 against 0.13-0.36; FX's P arms at the
  store's q on MiniLM with V = 16 read 0.13 (d = n = 384 numbers in 16 classes), its T arms 0.22-0.45; Qwen3-8B P at
  V = 2 0.62-0.69. No development or test store has been answered by any B74 arm.

## Fixed (B49 / B53 / B60 / B62 / B66 / B71 / B72's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (md5s as
B60-B72), splits, B32's make_store stores (dev 6 + 3, test 20 + 5 per (M, V)), queries (unseen aliases), **R1** = V in
{2, 16} x 4, 8, 16 bits per fact x M in {256, 1,024} (12 cells per encoder), reported as R1-V2, R1-V16 and R1; **ideal
accounting**. Existing arms are not re-run for their registered numbers; their test predictions are read as
b72_verdict.load reads them: STORE (B53), VSAS (B56), VSAPC / VSAPCX (B60), LAMC (B60), NLMSC (B62), OPTR (B72).

## (A) 2 x 3 factorial at the store's fixed format (arm FX, b74_run.py)

**Format (one per cell)** = the final design's (STORE, B53's dev choice): its bits per number q_R, its dimension d_R =
B32's size rule at q_R (bits ideal: d H_q + 32 per block of 8,192 <= budget, maximal), B53's value codes
(Sylvester-Hadamard / antipodal, row-0 signs), B32's quantizer per block of 8,192 with its block scale, and the
matched-filter read score(v) = sum_j S_j phi_j(q) c_j(v), argmax. q_R and d_R are the same for all six arms of a cell.

**Features (3 levels; parameters fixed per cell from the existing dev choices, identical for both write rules; all
fitted on the training split as in the paper).**
- R (random): B53's (prep, fmap) of the cell, the store's own projection blocks (R53.Feat53).
- P (principal, as the strongest variant uses them): VSAPc's matched-filter dev-chosen (prep, pfmap) of the cell:
  the prepped code's principal coordinates (whiten: the corrected basis U0, mu = 0; centre: B57's basis), linear (plin)
  or signs (psgn), the first d coordinates with **d = min(d_R, n)** (principal coordinates exist only up to n: where
  d_R > n the P arms store n numbers at q_R and leave the rest of the budget unused, as VSAPc does at its own q).
  b57_run.FeatP with b60_vsapc's corrected basis (b57_run.make_feat).
- T (rotated principal, B71's VSAPR): VSAPR's dev-chosen (prep, fmap, p) of the cell: the top-p principal
  coordinates rotated by independent Haar matrices placed side by side (B71's seeds), fmap lin / sgn / sgn3, d = d_R
  (no cap). b71_run.FeatPR.

**Write rules (2 levels).**
- H (Hebbian sum, the store's write): S = sum over the stored facts of phi(k) * c(v), quantized by the store's
  quantizer stream (b32_run.vsa_grid unchanged, seeds R.store_seed / store_order). Free parameter: the stored share f in
  {1, 1/2, 1/4, 1/8}, chosen on dev by B32's rule (highest mean dev accuracy; within 0.1 points tied, the first in the
  order 1 > 1/2 > 1/4 > 1/8) at the cell's fixed features and q_R. **HR := STORE** (B53's test predictions, its own
  share; B53 chose the share jointly with prep / fmap / q).
- L (least squares in the store's format, B72's OPT write): S* = argmin sum_i sum_v (score_i(v) - t_i(v))^2 + lam ||S||^2
  over the written facts (closed form per Hadamard residue class, b72_run.opt_solve unchanged), quantized once by the
  store's quantizer at q_R (seeds ('B74Q', store seed, kind, d, q, n_written, alpha)), matched-filter read. Grid = share f
  (1, 1/2, 1/4, 1/8) x alpha (1e-4, 1e-2, 0.1, 1, 10, 100, 1e4) = 28, B60's rule, tie order f > alpha (B72's).
  **LR := OPTR** (B72's test predictions; exactly this arm at the store's format).

**New arms:** HP, LP, HT, LT (dev: every grid point on every dev store; test: the dev-chosen configuration, once).
Reproduction checks (not used for any choice): HP's dev rows at (prep, pfmap, q_R, f) against B60's VSAPc dev rows
(b60_out/vsapc/*/devp_*), HT's at (prep, fmap, p, q_R, f) against B71's VSAPR dev rows where available, and, wherever
the chosen configuration coincides with VSAPR's chosen one, HT's test predictions against B71's.

**Contrasts** (per encoder; R1, R1-V2, R1-V16; per-query linear combinations of the six arms on the same queries):
- write effect at each feature level: W_R = LR - HR, W_P = LP - HP, W_T = LT - HT;
- feature effect at each write rule: F_P|H = HP - HR, F_P|L = LP - LR, F_T|H = HT - HR, F_T|L = LT - LR;
- interactions: I_P = W_P - W_R = (LP - HP) - (LR - HR), I_T = W_T - W_R;
- main effects of the two 2 x 2 sub-designs (R, X), X in {P, T}: write (W_R + W_X) / 2, features (F_X|H + F_X|L) / 2
  (the averages of the two paths, i.e. Shapley shares); and of the 2 x 3 design: write (W_R + W_P + W_T) / 3;
- corner-to-corner gaps G_X = LX - HR, and **both path decompositions**: path 1 (write first) G_X = W_R + F_X|L;
  path 2 (features first) G_X = F_X|H + W_X; they differ by exactly I_X.
- secondary: LP - LAMC and LT - LAMC (the store's format and these feature definitions against LAMc's free columns,
  own q, p and features); the path-2 decomposition of STORE - LAMC = (HR - HT) + (HT - LT) + (LT - LAMC) (and with P).
Primary: W_R, W_P, W_T, F_P|H, F_P|L, F_T|H, F_T|L, I_P, I_T, G_P, G_T (10,000 draws); the rest secondary (2,000).

**Attribution rule (fixed now).** Per encoder over R1 and per corner X in {P, T}: "features larger (identified)" iff
|F_X|L| > |W_R| AND |F_X|H| > |W_X|; "write larger (identified)" iff |W_R| > |F_X|L| AND |W_X| > |F_X|H|; otherwise
"path-dependent". "Interaction material" iff I_X's one-level interval excludes 0 and |I_X| >= 1 point.
**The statement "the larger part of least squares' lead is its features, not its write rule" (B72's sentence) is
SUPPORTED only if "features larger (identified)" holds on at least 6 of 7 encoders at corner T AND at corner P.** It is
UNSUPPORTED (the paper must not say it) if at either corner "features larger (identified)" holds on 5 or fewer of 7
encoders; then the paper states, per corner, both path decompositions and the interaction, and that the attribution
depends on the path. The same rule is applied at R1-V2 and R1-V16 separately (reported; the R1 result decides).

## (B) Exact-label queries (arm LAB)

For STORE, VSAS, VSAPC (hence VSAPCX by B60's per-cell rule), LAMC and NLMSC at their committed test configurations,
the same test stores are queried with every stored entity's own label code (the key that was written; one query per
entity, gold = its value), in addition to the unseen aliases. Each arm's ORIGINAL test code is called unchanged with
only b32_common.make_store wrapped so that the store's query list is [aliases, labels] (labels appended; qent extended
with the entity index) and with its output directory redirected: STORE = b53_run.test's vsa_grid call (B53's chosen
(prep, fmap, q, share), Feat53, store seed and order); VSAS = b56_run.dec('test'); VSAPC = b57_run.engine through
b60_vsapc's corrected basis, the committed PLM / PRT selection; LAMC = b60_run.run_job on the spec of B60's own test
job; NLMSC = b62_run.run_job on the spec of B62's own test job. The stored memory does not depend on the queries, so
the alias part must reproduce the existing test predictions (check; bar: identical in >= 99.9% of alias queries of
every arm and encoder, and every differing query reported).

Reported (R1, R1-V2, R1-V16): each arm's label recall and alias recall (all M stored entities; where a store writes a
share f < 1 the unwritten entities count as stored-but-lost, as in the alias statistic); contrasts **STORE - LAMC,
VSAPCX - LAMC, STORE - NLMSC, VSAPCX - NLMSC on labels and on aliases** (one- and two-level, 10,000 draws for labels;
alias = existing), and per contrast the **storage-versus-generalisation split**: on labels the contrast measures
storage error only (cross-talk on the written keys); the change from labels to aliases is the generalisation part,
Delta = contrast(aliases) - contrast(labels), with a paired entity bootstrap (entity clusters drawn within store within
cell, the entity's label and alias queries together; 2,000 draws, one-level). Also per arm the generalisation loss
label recall - alias recall (point, same paired interval).

## (C) One-pass delta rule on a matched grid (arm NL72)

As B66 (C): for d = 0..19, 72 configurations drawn without replacement from NLMSc's 480 (b62_run.grid('NLMSC'), its
tie order) by numpy default_rng(b32_common.seed_of('B74C', 'NLMSC', d)), kept in grid order, the same sub-grid for every
cell and encoder. Dev choice inside each draw: B62's rule (highest mean dev accuracy over the cell's dev stores, B62's
existing dev rows in b62_out/jobs; within 0.1 points tied, the first in grid order). Test: the draw-chosen
configurations whose test predictions are not saved (the full-grid choice's are, b62_out/jobs), by b62_run.run_online
unchanged (its seeds; output directory b74_out), once. Reported per encoder: per draw R1 (and R1-V2, R1-V16) **STORE -
NLMSC72** and **VSAPCX - NLMSC72** (min, quartiles, median, max; number of draws with the difference < 0; number with the
full-grid sign), NLMSc72's R1 accuracy against the full-grid choice's, and per draw the one-level label (2,000 draws;
the count of "NLMSc leads"). No interval over draws is claimed. Four language models primary; the three sentence
encoders as well (first to be cut).
**Reading (fixed now, B66's):** the unequal-grid limitation is "not material" for a contrast if its sign over R1 equals
the full-grid sign in at least 18 of 20 draws on every language model and the median is within 1 point of the
full-grid value; otherwise the paper must give the matched-grid range. For VSAPcX - NLMSc, whose full-grid value is
within 0.1 points of 0 on Qwen3-8B and Llama, the count of draws in which NLMSc72 leads (one-level) is what the paper
may cite.

## Statistics and decision rule

Region statistic = mean over cells of the cell mean of the per-query value; **one-level** 95% cluster bootstrap (entity
within store, within each cell; b36_verdict.boot_region) = the registered label; **two-level** = B66's store-level
scheme (b66_storeboot.cell_draws), reported beside. B60's rule: "x leads" iff the lower bound > 0, "y leads" iff the
upper bound < 0, else "level". For (A)'s combinations of more than two arms the per-query value is the registered linear
combination of the arms' correctness indicators.

## Predictions (stated so that they can be wrong)

- PA1. The interaction is positive: W_T > W_R over R1 on at least 5 of 7 encoders, I_T's one-level interval above 0 on at
  least 4; W_P > W_R on at least 5 of 7, I_P above 0 on at least 4 (the least-squares write gains more on principal than
  on random features, most at V = 16).
- PA2. F_P|H (HP - HR) < 0 over R1 on at least 5 of 7 (at the store's q, d <= n and Hadamard classes on principal
  coordinates hurt the Hebbian store); F_T|H (HT - HR) >= 0 on at least 4 of 7.
- PA3. Attribution over R1: at corner P "features larger (identified)" on at most 2 of 7; at corner T on at most 4 of 7;
  hence the statement "features are the larger part" is UNSUPPORTED by the registered rule.
- PA4. Shapley shares over R1 at corner T: the write share >= the features share on at least 4 of 7.
- PA5. LT - LAMC over R1: LAMc leads on at least 4 of 7, |LT - LAMC| <= 2.5 points on at least 5 of 7.
- PA6. HT's dev rows reproduce B71's VSAPR rows and HP's reproduce B60's VSAPc rows exactly where compared.
- PB1. Label recall exceeds alias recall over R1 for every arm on every encoder.
- PB2. On labels LAMc leads the store by more than on aliases (Delta = alias contrast - label contrast > 0 for STORE -
  LAMC, i.e. less negative on aliases) on at least 5 of 7; the same for STORE - NLMSC on at least 5 of 7.
- PB3. The generalisation loss (label - alias recall) is larger for LAMc than for the store on at least 5 of 7.
- PC1. STORE - NLMSC72 < 0 over R1 in at least 18 of 20 draws on each of the 4 language models; median within 1 point of
  the full-grid value on at least 3 of 4.
- PC2. NLMSc72's median R1 accuracy is 0.2-1.5 points below the full-grid choice's on each language model.
- PC3. VSAPCX - NLMSC72: on Mistral-7B NLMSc72 leads (one-level) in at least 18 of 20 draws; on Qwen3-8B and Llama it
  leads in at most 5 of 20.
- PX. Checks pass (below).

## Readings for the paper (fixed now)

- (A) The paper may name the write rule or the features as "the larger part" only by the registered attribution rule.
  If UNSUPPORTED: "At the store's own format, the least-squares write gained W_R points on random and W_T (W_P) on
  principal features; the order of the two steps changes the split (both decompositions given) because the write
  rule and the features interact (I_T points)". The six arms are diagnostic variants, not the store's design.
- (B) The paper may say how much of each contrast is present on the written labels themselves (storage) and how it
  changes on aliases (generalisation); if the label contrasts carry the gap, "least squares' lead is a storage
  advantage (less cross-talk), present on the written keys".
- (C) The matched-grid range of STORE - NLMSc and VSAPcX - NLMSc is stated beside the full-grid values; "not material"
  only under the reading above.

## Checks (b74_check.py; second angle)

- **Plain-loop scorer**: every R1 cell accuracy of every new arm (HP, LP, HT, LT; label recalls; NLMSC72 per draw) and
  every registered point statistic (including (A)'s linear combinations and both path sums) recomputed with plain
  Python loops from the job files and merged records; |difference| < 1e-9; path 1 - path 2 = I_X to 1e-12.
- **Bits audit**: every dev and test row of FX: d by an own re-implementation of B32's size rule at q_R (bits <= budget,
  d maximal; P: d = min(rule, n)); NLMSC72 rows by an own re-implementation of LAMc's bit formula (<= budget, maximal
  unless at the cap).
- **Float64 re-derivation (independent code) on a sample**: one test store per (encoder, M, V) at 4 and 16 bits per
  fact: LP and LT by own float64 features (P: own eigh of the training covariance; T: own eigh basis and the registered
  Haar seeds), own codes (b71_check.own_codes), own PRIMAL ridge per class, own quantizer (b71_check.own_q_store), own
  read; HP and HT by own float64 Hebbian sum, B32's quantizer stream re-implemented (own_q_store with the store's
  seeds) and own read. Bar: agreement >= 0.99 with the logged predictions on every re-derived store. (B): the
  alias-reproduction check above, and an own float64 re-derivation of STORE's and LAMc's label predictions on the same
  sample (bar >= 0.99).
- **Determinism re-run**: a fixed sample of test jobs (the first test job in sorted order of every (encoder, job type))
  re-run in a fresh output directory with a different worker count: predictions identical.
- **Sample reading by the AI-assisted process**: Qwen3-8B and e5-base-v2, the R1 cell with the largest |I_T|: 10
  queries the interaction comes from (right only under LT, or only under HR, etc.) with alias, label, gold and the four
  answers; and 10 label queries the store misses but LAMc answers (storage errors), with notes.

## Compute and cut rule

CPU only. The A800 box (ssh cavin, 128 cores; fresh directory ~/b74_run made from ~/b72_run's code and codes, new files
by md5 from this repository; one process per job, OMP / BLAS threads 1, nice 5, at most 48 workers) for the language
models; local (at most 6 workers; about 14 GB free) for the sentence encoders, choices, verdict and checks. Every ssh
command in the background; no retry loops on state-changing remote commands; processes stopped only by exact PID;
nothing of other users touched; small files only over the link, md5-verified. **Time box: 4 hours from 22:19 (machine
clock), including the block wait.** If time is short, cut in this order, in a dated compute note written before any test
output is read: (C) first (its sentence encoders first), then (B) (VSAS / VSAPC first, i.e. VSAPcX on labels), then
(A)'s secondary contrasts; (A)'s primaries last.

## Files

b74_run.py (arms FX, LAB, NL72; plans, dev choice, draws, selftest), b74_verdict.py, b74_check.py; results/b74_*; job
outputs under b74_out/ (not committed).
