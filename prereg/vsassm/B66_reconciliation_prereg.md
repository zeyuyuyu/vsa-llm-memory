# B66 pre-registration: SDM reconciliation (two codes x two grids) and map replicates, least squares on the store's own features with the corrected options (LSfc), matched tuning grids, store-level intervals

Date: 2026-11-23 (machine clock Asia/Shanghai, about 13:45; US Eastern 2026-11-23 00:45). Written and committed, then
OpenTimestamps-stamped, before any B66 development or test run. The paper is not edited by B66.

## Why (the last referee's open questions)

1. **SDM reconciliation.** B64's SDM with training-split hard locations (SDMt) beat the store's first design over R1 on
   all 7 encoders (-0.9 to -4.7 points). The clean room's own SDMt was weaker: there the first design led its best of
   six by +2.3 (Qwen3-8B) and +2.1 (Llama-3.1-8B), its SDM family by +2.4 / +2.2. B64 attributed this to the grids
   (B64's dev choice lies outside the clean room's grid in 82 of 84 conditions) but never ran one grid in both codes,
   and B64 has no random-map replicates.
2. **LSf.** Least squares on the store's own features (B56's LAMRF) was never re-run with LAMc's corrected options, so it
   is open whether least squares wins by its write rule or by its principal-component features.
3. **Matched grids.** Tuning budgets differ: store 72 configurations, LAMc 288, SDMt 504, the six key-free families 468.
4. **Store-level variance.** The intervals resample entities within store only; at M = 1,024 there are 5 test stores
   per condition, and several "on every encoder" claims rest on small effects.

## What has been seen before this file (disclosed)

- B64's, B60's, B62's, B56's verdicts, notes and every number in them (the pipeline side of the 2 x 2 in (A) is known:
  B64 code, B64 grid: first design - SDMt over R1 = -1.89 Qwen3-8B, -1.20 Llama). The clean room's outputs on its own
  stores (cleanroom4_keyfree/out_qwen8, out_llama, compare_*.txt, my_cells_*.json): its first design 46.17 / 47.74, its
  SDM family (Gaussian or data addresses, dev-chosen) 43.73 / 45.51, best of six 43.83 / 45.61 over R1 (Qwen3-8B /
  Llama). So the (clean-room code, clean-room grid) cell is known in advance, up to separating its data-address
  configurations from its Gaussian ones.
- **Code self-tests before this file, training split / toy data only:**
  - b66_cr.py selftest (Qwen3-8B training stores): the added SDMt path with (LA, exact counters) equals cr.py's own
    sdm_eval with data addresses on 48 of 48 (configuration, store) pairs; the label / alias column split is correct.
  - b66_run.py selftest: location orders for map 0 equal B64's, maps 1-3 are permutations of the same sets; the
    clean room's grid (32 configurations) lies inside B64's grid; the first design with map 1 differs from map 0 on a
    training store (agreement 0.129).
  - b66_lsfc.py selftest (toy data and Qwen3-8B / e5 training stores): budgets respected and p maximal; LSfc's
    random features rebuilt with B53's codes reproduce b32_run.vsa_grid's store read (agreement 0.997-1.000); col /
    V - 1 with B49's alphas reproduces b49_common.run_lam on the same features (>= 0.995 on 36 configurations);
    principal features equal b57_run.FeatP under b60_vsapc's corrected basis; B53's chosen (prep, fmap) equals the
    merged records' cfg field in all 168 R1 cells (only that field read); timing 1-15 s per training store and kind.
  - b66_storeboot.py selftest (fabricated records only): with no store effect the registered two-level interval is
    1.33-1.39 times as wide as the one-level interval (see (D): the two-level scheme counts within-store sampling
    variance twice, Davison & Hinkley 1997 section 3.8, so it is conservative by construction); with a store effect
    (SD 0.05) 1.87 times; point statistics equal boot_region's to 1e-12. Not run on any real record.
  - b66_grids.py selftest: draws deterministic, 72 distinct configurations; the full-grid choices recomputed from the
    existing DEV rows reproduce the committed choices of B53 (store), B60 (LAMc), B64 (SDMt), B36's six families, B49's
    BEST36 and B64's BEST7 in 12 of 12 R1 cells on all 4 language models; a configuration run alone on a training store
    equals its run inside a larger selection (0 differences on 70 rows). Its `draws` / `plan` commands were run once
    into the scratchpad on existing DEV rows only, to size the compute (needed test runs: SDMt 483 predictions in 200
    jobs, LAMc 341 in 345 jobs, key-free six 366 configurations, of which IDX needs faiss, only on the box). No test
    accuracy of any non-chosen configuration has been computed or read.

## Fixed (B36 / B49 / B53 / B60 / B64's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (md5s as
B49 / B60 / B62; box copies checked), splits, B32's make_store stores (dev 6 + 3, test 20 + 5 per (M, V)), queries
(unseen aliases), V in {2, 16}, **R1** = 4, 8, 16 bits per fact x M 256 / 1,024 x V 2 / 16 (12 cells), R1-V2 and R1-V16
its halves; accountings ideal (primary) and real (secondary). Existing arms are not re-run: STORE = B53's VSA (final
design), VSAR = B49's store (first design), VSAm1-3 = B57 / B59's per-map final design, LAMc / VSAPc / VSAPcX = B60's,
NLMSc = B62's, SDMT / BEST7 = B64's, the six key-free families and BEST36 = B36 / B49's (merged records
b59_out/fam/{enc}/all59_*, b60_out, b62_out, b64_out). Statistic (B36 / B60): region statistic = mean over cells of the
cell mean of the per-query difference; 95% cluster bootstrap, entity within store, resampled within each cell
(b36_verdict.boot_region; 10,000 draws for primaries, 2,000 otherwise). **Decision rule (B60):** "first leads" iff the
95% lower bound > 0, "second leads" iff the upper bound < 0, else "level". **B36's bar** (key-free comparisons): >= +2
points with the interval above 0 = passes; <= -2 with the interval below 0 = loss; else tie.

## (A) Reconciliation

### (A1) Two codes x two grids (Qwen3-8B, Llama-3.1-8B; R1; ideal)

- **Clean-room code = b66_cr.py**: cr.py copied verbatim (md5 1c1b9c1e...; attribution in its header), changed only by
  (i) the grid: a family 'sdmt' with B64's exact configuration space (prep x location set LA / L / A x a in 1, 2, 4,
  8, 16, 32, 64 x onehot / bits counters x counter type cnt / clip3 / tern x share 1, 1/2 = 504, B64's tie order),
  evaluated by cr.py's own SDM code (its data-address similarities, its geometric N grid of 64 points from max(a, 8),
  its bisection size search, its cost = counters x empirical entropy with no default-value bits, its random tie
  breaking, its own stores, store and families); the two additions are the location-set columns (cr.py orders the
  training rows labels first, then aliases: 'L' = the first 2,818 columns, 'A' = the rest) and B64's counter
  transform before the cost and the read; and (ii) the harness: paths from the environment, a dev phase that writes
  every dev accuracy and the choices, and a test phase that reads the committed choices (same seeds and order of
  computation, so the predictions equal a one-pass run). The original grid comes first and is unchanged, so every
  original configuration sees the same random stream.
- **Arms in the clean-room code:** first / final design (cr.py's store, unchanged); its six families (unchanged;
  'sdm' = Gaussian or data addresses, dev-chosen, as cr.py); **sdmG** (its SDM restricted to Gaussian addresses, 32
  configurations); **sdmTcr** (its SDM restricted to data addresses = SDMt on the clean room's grid: prep x a in 1, 4,
  16, 64 x onehot / bits x share 1, 1/2 = 32); **sdmT64** (SDMt on B64's grid, 504); **best7cr** / **best7_64** =
  cr.py's family rule (best dev accuracy, ties within 0.1 to the first) over index, static, cms, bloom, fly, sdmG and
  sdmTcr / sdmT64; best6 = cr.py's original best of six (reproduction).
- **Pipeline code = B64's b64_run.py**, unchanged: B64 grid = B64's arm (existing). **Clean-room grid** in B64's code:
  configurations (prep, 'LA', a in {1, 4, 16, 64}, onehot / bits, 'cnt', share 1, 1/2) = 32, chosen by B36's rule on
  B64's existing dev rows restricted to them (tie order: the clean room's prep > a > counters > share), arm **SDMTCR**;
  test = b64_run.run_job on the chosen configurations; **BEST7CR** = B36's BEST rule over the six (B49's dev
  accuracies) + SDMTCR.
- **Reported, all four cells**, per encoder over R1 (and R1-V2, R1-V16): first design - SDMt and first design - best of
  seven, final design - SDMt and - best of seven; each cell with its own code's statistic: pipeline cells
  b36_verdict.boot_region (10,000 draws); clean-room cells the clean room's own bootstrap (agg.py's: entity
  multinomial weights shared across its 12 conditions, 2,000 draws, seed 20261123). Point = mean over the 12 cells in
  both codes.
- **Attribution (fixed now).** With D(code, grid) = first design - SDMt over R1: grid effect = mean over codes of
  D(code, clean grid) - D(code, B64 grid); code effect = mean over grids of D(clean code, grid) - D(pipeline code,
  grid); interaction = the remainder. The sign difference is attributed **to the grid** if the grid effect is at
  least twice the absolute code effect on both encoders and D(clean code, B64 grid) <= +0.5 (the clean room also finds
  SDMt level with or ahead of the first design on B64's grid); **to the code** if the code effect is at least twice the
  absolute grid effect on both encoders; otherwise **"both"**, with the shares stated. Reproduction check: the
  clean-room code's re-run reproduces its original test accuracies (first, final, the six families, best6) exactly
  (seeds unchanged).

### (A2) Random-map replicates (all 7 encoders)

Maps m = 1, 2, 3 with B57 / B60's tag '{enc}#b57m{m}' in place of the encoder name in every random component's seed:
- **VSARm** (first design per map): b32_run.Feat with the map's tag (projections and random value codes), the map's
  store seeds (b32_run.store_seed(tag, st)), B32's 72-configuration grid (prep x fmap lin / sgn / sgn3 x q 1 / 2 / 4 x
  stored share 1 / 1/2 / 1/4 / 1/8) on the dev stores at the R1 budgets, both accountings, B32's rule (highest mean dev
  accuracy, within 0.1 points tied first in prep > fmap > q > share order; b57_run.b32_choose's algorithm); test = the
  chosen configuration. Map 0 run through the same code must reproduce B49's VSAR dev choice and dev accuracy in every
  R1 cell (a dev-phase check).
- **VSAm** (final design per map): B57 / B59's existing predictions VSAm1-3.
- **SDMTm**: SDMt's only random component is its fixed location order; map m re-draws it with seed
  C.seed_of('B64LOC', loc, '{enc}#b57m{m}') (map 0 = B64's C.seed_of('B64LOC', loc)); B64's full 504 grid on the dev
  stores, B36's rule, test = the chosen configuration (b64_run.run_job unchanged; b66_run.py patches only the order
  seed). Cells where the chosen N equals the whole location set do not depend on the order.
- **BEST7m** = B36's BEST rule over the six key-free families (B49's dev accuracies and predictions, map 0: their random
  components are not re-drawn, as B60 did for BESTc's other families) + SDMTm.
- Contrasts: **VSARm - SDMTm** and **VSAm - BEST7m** (also VSARm - BEST7m), maps 0-3 (map 0 = B64's registered
  numbers), R1, R1-V2, R1-V16, ideal (real secondary). Statistic: B59 / B60's two-level bootstrap over maps
  (b59_verdict.twolevel: maps drawn with replacement from the four, entity clusters resampled once per draw and shared
  across maps; 10,000 draws): per map the region difference, the map mean, the between-map SD, the two-level 95%
  interval, the registered map's rank; the decision rule applied to the map mean with the two-level interval, reported
  beside the registered map's label; B36's bar on the map mean.

## (B) LSfc: least squares on the store's own features with LAMc's corrected options

- **Algebra:** B60's LAMc exactly (b60_run.run_lamc called unchanged: ridge W = F^T (F F^T + lam I)^-1 T, lam = alpha x
  mean diag(F F^T); all M facts; targets V - 1 or V columns; scales per column or flat; quantizer per scale group;
  bits ideal c p H_q + 32 x groups, real q per number; p maximal), with the feature map replaced by the store's own:
  - **LSFR (random features):** per cell, the prep and feature map that B53's dev choice picked for the store, with the
    store's own projection blocks (b32_run.Feat, same seeds: lin, sgn, or the 3-fold sign product sgn3); p <= 65,536.
  - **LSFP (principal features, corrected basis):** per cell, the prep and map (plin / psgn) of VSAPc's dev choice, the
    store's own principal features (B57's FeatP; whiten on b60_vsapc's corrected basis U0, mu = 0; center on B57's
    basis); p <= n.
- **Grid** per kind (order = tie order): scale (col, flat) > target (V-1, V) > alpha (1e-4, 1e-2, 0.1, 1, 10, 100) > q
  (1, 2, 4) = 72 configurations (the store's grid size). Dev: every configuration on every dev store at the R1 budgets,
  both accountings; **B60's rule** (highest mean dev accuracy, within 0.1 points tied first in grid order). **LSFX** =
  per cell the dev-best of LSFR's and LSFP's choices (within 0.1 points: LSFR). Test: the chosen configurations, once.
  Seeds C.seed_of('b66lsf', enc, kind, split, M, V, rep).
- **Comparisons** (all 7 encoders; R1, R1-V2, R1-V16; ideal primary 10,000 draws, real 2,000; B60's decision rule):
  **STORE - LSFR** (write rule at identical random features: Hebbian vs least squares), **LSFR - LAMC** (features at the
  same least-squares write: the store's random map vs LAMc's dev-chosen rp / principal features), **VSAPC - LSFP**
  (write rule at identical principal features), **LSFP - LAMC**, **STORE - LSFX**, **LSFX - LAMC**. Also per-cell
  accuracies and the dev choice composition.
- **Readings (fixed now).** "Least squares wins by its write rule" if STORE - LSFR has "LSFR leads" over R1 on at least
  6 of 7 encoders (and VSAPC - LSFP "LSFP leads" on at least 6 of 7). "By its features" if STORE - LSFR is "store leads"
  or "level" on at least 4 of 7 while LSFR - LAMC is "LAMC leads" on at least 4 of 7. "Both" if LSFR leads the store
  and LAMc leads LSFR, each on at least 4 of 7: the paper then states the two parts with their sizes. The sentence
  "least squares beat every Hebbian store variant on every encoder" is checked against STORE - LSFR / VSAPC - LSFP here
  and against (D).

## (C) Matched tuning grids (the 4 language models, R1, ideal)

- Families and their registered full grids and dev rules: STORE (B53, 72, B32's rule), LAMC (B60, 288, B60's rule),
  SDMT (B64, 504, B36's rule), KF6 (B36's six families IDX 48, SF 176, CM 132, BF 60, FBF 12, SDM 40 = 468; B36's
  per-family rule, then B36's BEST rule, exact ties in the order IDX, SF, CM, BF, FBF, SDM).
- **Draws:** for each family F and d = 0..19, 72 configurations without replacement from F's full grid
  (default_rng(C.seed_of('B66C', F, d)), kept in grid order), the same sub-grid for every cell and encoder. KF6: drawn
  from the union of the six grids; families with no drawn configuration are absent. STORE has exactly 72, so every draw
  is its full grid (no runs). BEST7_72 (draw d) = B36's BEST rule over KF6's and SDMT's draw-d choices.
- **Test:** the draw-chosen configurations whose test predictions are not already saved, run once by each family's own
  code (b64_run.run_job, b60_run.run_job, b36_run / b32_run for KF6; registered now). KF6's index family needs faiss
  (box only); see the compute rule.
- **Reported:** per family and encoder, the full-grid choice's R1 test accuracy and the 20 draws' R1 test accuracies
  (min, quartiles, median, max, mean); derived contrasts with matched grids as 20-draw distributions beside the
  full-grid value: STORE - LAMC72, VSAR - SDMT72, VSA - SDMT72, VSA - BEST7_72, VSAR - BEST7_72. No interval is claimed
  for the draws (the distribution is over grid draws only).
- **Reading (fixed now):** the unequal-grid limitation is "not material" for a contrast if its sign over R1 is the same
  as the full-grid contrast's in at least 18 of 20 draws on every language model, and the median matched-grid contrast is
  within 1 point of the full-grid one; otherwise the paper must name the contrast and give the matched-grid range.

## (D) Store-level intervals (analysis of saved outputs; no new test run)

- Contrasts (ideal primary, 10,000 draws; real secondary, 2,000, where the arm exists): B60 STORE - LAMC, VSAPCX - LAMC,
  VSAR - LAMC, VSAS - LAMC, VSAPC - LAMC; B62 STORE - NLMSC; B64 VSAR - SDMT, VSAR - BEST7, VSA - BEST7; regions R1,
  R1-V2, R1-V16; every encoder.
- **Two-level bootstrap (primary for (D), as the referee asked):** per draw and per cell, draw the cell's S test stores
  with replacement (S = 20 at M = 256, 5 at M = 1,024), then within each drawn store draw its entity clusters with
  replacement; cell draw = drawn difference sum / drawn query count; region draw = mean over cells; cells independent.
  Disclosed property: this scheme counts within-store sampling variance twice, so with no store effect it is about
  1.3-1.4 times as wide as the one-level interval (conservative). Also reported: the one-level interval (entity within
  store, boot_region) and a store-only diagnostic (stores resampled, each drawn store's queries kept as they are).
- **Claims checked** (label under the two-level interval; "holds" only if the label is unchanged on every encoder
  named): (1) LAMc leads the store over R1 on all 7; (2) at R1-V16 on all 7; (3) at R1-V2 on the 4 language models, e5
  and bge (MiniLM level); (4) LAMc leads VSAPcX over R1 on all 7; (5) "least squares beat every Hebbian store variant on
  every encoder": LAMc leads each of STORE, VSAR, VSAS, VSAPC, VSAPCX over R1 on all 7; (6) NLMSc leads the store over R1
  on all 7; (7) SDMt leads the first design over R1 on all 7; (8) BEST7 leads the first design over R1 on all 7; (9) the
  final design - BEST7 labels per encoder and the bar count (0 of 7). Reading: a claim that fails under the two-level
  interval must be restated per encoder ("on k of 7 encoders; level on ... when test-store variation is included").

## Predictions (stated so that they can be wrong)

- PA1. Grid effect on D (first - SDMt over R1) >= +2.0 points on both Qwen3-8B and Llama; |code effect| <= 2.0 on both;
  the registered attribution is "grid" on both.
- PA2. D(pipeline code, clean grid) between -1.0 and +2.5 on both; D(clean code, B64 grid) between -2.0 and +1.0.
- PA3. The clean-room code's re-run reproduces its original test accuracies exactly (every arm, every cell).
- PA4. Maps: between-map SD of R1 VSAR - SDMT <= 0.8 points on each of the 7 encoders; the map-mean label of R1
  VSAR - SDMT is "SDMt leads" on at least 6 of 7; map 0 of VSARm reproduces B49's VSAR dev choice in every R1 cell.
- PA5. Maps: R1 VSA - BEST7 map-mean labels equal the registered map's on at least 5 of 7; the bar passes on 0 of 7.
- PB1. STORE - LSFR over R1: LSFR leads on at least 5 of 7 encoders; at R1-V2 LSFR leads on at least 5 of 7.
- PB2. LSFR - LAMC over R1: LAMc leads on at least 4 of 7.
- PB3. VSAPC - LSFP over R1: LSFP leads on at least 6 of 7; |LSFX - LAMC| <= 1.5 points over R1 on at least 5 of 7.
- PB4. The registered reading is "both" or "write rule" (not "features").
- PC1. For LAMC and SDMT the median 72-draw R1 test accuracy is within 1.5 points of the full-grid choice's on each of
  the 4 language models; for KF6 within 1.0.
- PC2. STORE - LAMC72 has the full-grid sign in at least 18 of 20 draws on all 4; VSAR - SDMT72 has it on at least 3 of 4.
- PD1. The two-level interval is 1.3-2.0 times as wide as the one-level interval (average over contrasts, each encoder).
- PD2. Claims (1), (2), (4), (5) hold under the two-level interval; claim (6) fails on at least one encoder; claim (7)
  fails on at least one encoder.
- PX. Checks pass (below).

## Checks (second angle)

- **Plain-loop scorer:** every R1 cell accuracy of every new arm (SDMTCR, BEST7CR, SDMTm, BEST7m, VSARm, LSFR, LSFP,
  LSFX, the draws' choices) and every registered point statistic recomputed with plain Python loops from the job files
  and merged records; |difference| < 1e-9. For (D): every point statistic by plain loops.
- **Bits audit:** every new dev and test row within budget (SDMt: B64's rule incl. the default value; VSARm: B32's
  vsa_bits(d, q, acct) <= budget and d maximal; LSfc: own re-implementation of LAMc's bit formula, p maximal). The
  clean-room code's SDMt rows: its own cost rule (re-computed on a sample of chosen configurations).
- **Float64 re-derivation (independent code) on a sample:** SDMTCR and SDMTm: b64_check.rederive_one (own float64
  preprocessing, full stable sort, loop counters, own entropy, own N scan, own read; the map's order seed) on one test
  store per (encoder, M, V) at 4 and 16 bits per fact; LSFR: own float64 features from the map's seeds, own dual
  ridge (numpy.linalg.solve), own quantizer and read, rep 0 per (encoder, M, V) at 4 / 16 bits per fact; VSARm: an own
  float64 store write / read (numpy, b32_common.features' seeds) on one test store per (encoder, M, V) and map at 8 bits
  per fact. Bar: agreement >= 0.99 on every re-derived store.
- **Sample reading by the AI-assisted process:** (A1) the Qwen3-8B cell where the two codes' SDMt on B64's grid differ
  most: the chosen configurations, N, and 10 queries that only one code's SDMt answers right; (B) Qwen3-8B and e5, the
  R1 cell with the lowest STORE - LSFR: 10 queries only LSFR answers right and 10 only the store answers right.

## Readings for the paper (fixed now)

- **SDM sentence** ("an SDM addressed by training codes beat the first design on every encoder"): survives if the
  attribution is "grid" (the clean room agrees on B64's grid) and the map-mean labels keep "SDMt leads" on all 7 with
  (D)'s two-level interval; otherwise restated per encoder. If the attribution is "code", the paper must name the
  clean-room disagreement and say which implementation detail carries it (from (A1)'s cells).
- **"Least squares beat every Hebbian store variant on every encoder"**: survives if (D) claim (5) holds and STORE -
  LSFR is "LSFR leads" on all 7; otherwise restated with the exceptions; and (B)'s reading decides whether "the write
  rule" or "the features" (or both, with sizes) is said.
- **Per-ingredient superposition verdict:** B64's restatement (no margin over the best key-free memory) stands unless
  (A2) or (C) changes a label; (B) adds whether least squares' lead is the write rule at the store's own features.
- **Unequal-grid limitation:** stated with (C)'s numbers; "not material" only under (C)'s reading.
- **Every "on every encoder" claim:** (D)'s table decides; failing claims are restated per encoder.

## Compute and cut rule

Local CPU (16 cores, shared; another agent runs B65 on the A800's GPUs): b66_cr.py (A1) with at most 8 processes x 2
threads; LSfc (B) at most 4 workers, one thread each; checks and verdicts. The A800 box (ssh cavin; 128 cores; fresh
directory ~/b66_run; code copied by md5; panel codes copied from ~/b62_run's B49 copies after the md5 check): (A2)
SDMTm and VSARm (at most 48 workers, OMP 1, nice 5), (C)'s test jobs including KF6 (IDX with the box's faiss env, after
its index code reproduces a sample of B32's saved IDX dev rows). Every ssh command in the background; no retry loops
on state-changing remote commands. **Time box: 3 hours from 12:50 (machine clock).** If time runs short, cut in this
order of priority (keep the first): (A), (B), (D), (C); within (C) KF6 is cut first. Any cut is written into a dated
compute note before any test output is read.

## Files

b66_cr.py (A1 clean-room code), b66_run.py (A1 pipeline side, A2 maps), b66_lsfc.py (B), b66_grids.py (C),
b66_storeboot.py (D), b66_verdict.py and b66_check.py (A verdict / checks; B, C, D's verdict and check commands live in
their own files), b66_box.sh; results/b66_*; job outputs under b66_out/ (not committed).
