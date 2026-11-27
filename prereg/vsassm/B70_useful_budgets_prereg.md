# B70 pre-registration: the equal-bit comparison at the budgets where the best memory is useful

Date: 2026-11-23 (machine clock Asia/Shanghai, about 17:00; US Eastern 2026-11-23 04:00). Written and committed, then
OpenTimestamps-stamped, before any B70 development or test run of a memory on a dev or test store. The paper is not
edited by B70.

## Why (referee, M2)

The equal-bit region R1 (V = 2 or 16, 4-16 bits per fact) is one where every memory is far from useful: region means are
44-57% raw, neither store design reaches 90% of an exact index within 64 bits per fact in any of 28 settings, and the best
competitor does in 19. Request: add or report results at budgets where the best memory is useful (e.g. where it reaches
>= 90% of the exact index). B70 defines that region from development data by a rule fixed here, and reports the same
contrasts there.

## What was already run above 16 bits per fact (the request's first question; from the code and merged records)

- **32 and 64 bits per fact, V = 2 / 16 (and 128 / 1,024), M = 256 / 1,024, all 7 encoders, dev and test:** B36 / B49's
  six classical families (IDX per-fact index, SF static function, CM count-min, BF Bloom, FBF, SDM with random
  locations) and their best (BEST36); B49's LAM (defective; superseded), learned hashing LHI / LHC / LHS, the first design
  (VSAR = B49's VSA), VSA-L; B53's final design (STORE); B56's VSAS (LMMSE read / retrained bundle); B57's VSAP and its
  NLMS / RLS (B49's defective features; superseded); B56's LAMRF; B59's map replicates of the store (VSAm1-3, VSASm1-3);
  B60's corrected least squares LAMc (and at V = 1,024) and VSAPc (store on the corrected principal basis), hence VSAPcX.
  Exact index (EXACT) and the unbudgeted infinite-width reads VSA-inf / LAM-inf (B49) are budget-free references.
- **Only at R1 budgets (4, 8, 16):** B62's NLMSc / RLSc / RLScS; B64's SDMt (training-split hard locations) and BEST7;
  B66's LSFR / LSFP / LSFX, SDMTm, VSARm, the 72-configuration draws; B60's LAMc map replicates.
- **Above 64 bits per fact:** nothing, for any memory.

## What has been seen before this file (disclosed)

- Every verdict and note of B49-B69, in particular: B60's RH (32-64 bits per fact) TEST region statistics (STORE - LAMc
  RH-V2: store leads on e5 +1.66, bge +3.98, MiniLM +4.73, LAMc leads on the 4 language models by 1.3-3.5; RH-V16 LAMc
  leads on all 7; STORE - BESTc over RH -6.1 to -14.1), and B49's useful-accuracy TEST table (U = 0.9 x EXACT, the best
  of ten's bits to reach U, VSA-inf and LAM-inf: both below U in every M = 1,024 setting and in most M = 256 settings).
  So for STORE, VSAR, VSAPcX, LAMc, BEST36 and BESTc the R2 test numbers below are re-aggregations of test predictions
  that already exist and whose 32-64-bit aggregates are public; I have not opened their per-cell test accuracies at the
  R2 cells (results/b60_part_*.json 'cells').
- **Every dev accuracy in the committed dev choices** (results/b49_choice_*, b53_choice_*, b56_choice_*, b60_choice_*,
  b60_choicepc_*, b32_choice_* / b49_choice32_* / b46_choice_b32 for the exact index). R2 below is computed from them
  (b70_run.py region -> results/b70_region.json) and was read before this file was written; its table is part of the
  pre-registration.
- **Never run before B70 at 32-64 bits per fact: NLMSc and SDMt** (hence BEST7 there). These are the blind parts.
- b70_run.py selftest (training-split stores only; results in b70_out/logs/selftest.log): NLMSc budgets respected and p
  maximal at 32 / 64 bits per fact, the dual form equals a genuine one-fact-at-a-time loop, SDMt within budget, the
  wrapper's rows equal direct calls of b62_run.run_online / b64_run.run_sdmt.

## Region R2 (fixed now; B70's R2, not B49's R2 = V 1,024)

For each setting s = (encoder, M, V), V in {2, 16}, M in {256, 1,024} (28 settings):
- U_dev(s) = 0.9 x the exact index's dev accuracy (B32's dev record; its dev-chosen prep).
- best_dev(s, b) = the highest dev accuracy at budget b (ideal accounting) among the dev-chosen configurations of the 13
  memories that have dev outputs at every budget 1-64 before B70: B49's ten families with LAM replaced by LAMc (B60's
  BESTc record: IDX, SF, CM, BF, FBF, SDM, LAMc, LHI, LHC, LHS), STORE (B53), VSAR (B49), VSAPcX (the larger of B56's
  STAR and B60's PSTAR dev accuracies, B60's rule).
- b*(s) = the smallest b in {1, 2, 4, 8, 16, 32, 64} with best_dev(s, b) >= U_dev(s); **R2(s) = {b in the grid : b >=
  b*(s)}** (accuracy curves are not always monotone; every larger grid budget is included). Empty if no such b.
- NLMSc and SDMt are not in the definition (no dev outputs at 32-64 before B70). Their B70 dev accuracies are reported
  against the rule as a sensitivity line; they do not change R2.

Result of the rule (dev only; best_dev at b*, family):

| encoder | R2 cells (M, V, bits per fact) | settings without a useful budget within 64 |
| --- | --- | --- |
| Qwen3-8B | (256, 2, 64) LHI; (1,024, 2, 64) LHI | all V = 16 |
| Granite-3.3-8B | (256, 2, 8 / 16 / 32 / 64) LAMc at b*; (1,024, 2, 64) LHI | all V = 16 |
| Llama-3.1-8B | (256, 2, 64) LHI | (1,024, 2); all V = 16 |
| Mistral-7B | (256, 2, 16 / 32 / 64) LAMc at b*; (1,024, 2, 64) LHI | all V = 16 |
| e5-base-v2 | (256, 2, 64) LHI; (1,024, 2, 64) LHI | all V = 16 |
| bge-small | (256, 2, 32 / 64) LHI; (256, 16, 64) LHI; (1,024, 2, 32 / 64) LHI | (1,024, 16) |
| MiniLM-L6 | (256, 2, 32 / 64) IDX at b*; (256, 16, 64) LHI; (1,024, 2, 64) LHI | (1,024, 16) |

23 cells in 15 of the 28 settings; 21 of the 23 have V = 2. The other 13 settings (12 of the 14 with V = 16, and
Llama-3.1-8B at M = 1,024, V = 2) have no useful budget within 64 bits per fact on dev; they are reported descriptively
(below), not tested. The dev-best memory at b* is learned hashing (LHI, a per-fact index on a map trained on alias-label
pairs) in 12 of the 15 settings, the per-fact index IDX in 1 (MiniLM-L6, M = 256, V = 2), least squares in 2 (Granite and
Mistral at M = 256, V = 2).

## Arms (ideal accounting; existing arms are not re-run)

- **STORE** = B53's final design (Hebbian write, antipodal / orthogonal value codes, matched filter); **VSAPcX** = the
  strongest store variant (per cell the dev-best of B56's VSAS and B60's VSAPc; B60's rule); **VSAR** = the first design
  (for the SDM comparison, as B64 / B66).
- **LAMc** = B60's corrected least squares; **NLMSc** = B62's one-pass normalised delta rule with LAMc's options.
- **SDMt** = B64's SDM at training-split hard locations; **BEST36** = the best of B36's six classical families (B49's
  dev choice); **BEST7** = B36's BEST rule over the six and SDMt (b64_run.choose's rule).
- References: **BESTc** = the best of ten with LAMc (B60), **LHI** and **IDX** (named families), **EXACT**.
- Test predictions: merged records b59_out/fam/{enc}/all59_* (STORE, VSAR, six families, LH*), B60's jobs (LAMc) and
  vsapc files (VSAPcX), as b60_verdict.load; NLMSc and SDMt at 8 / 16 bits per fact from B62's and B64's committed
  choices and test jobs; at 32 / 64 from B70.

## New runs (NLMSc, SDMt) at the R2 budgets 32 and 64

- Code: b70_run.py, which calls b62_run.run_online (NLMSc; B62's 480-configuration grid, B62's seeds
  C.seed_of('b62', enc, split, M, V, rep), write order b62_run.store_order) and b64_run.run_sdmt (SDMt; B64's 504
  configurations, location order seed ('B64LOC', set), store order b64_run.store_order) unchanged.
- Dev: every configuration on every dev store (6 at M = 256, 3 at M = 1,024) of each setting with an R2 budget in
  {32, 64}, at those budgets and at 16 bits per fact. **Dev-phase reproduction check:** the 16-bit dev accuracies must
  equal B62's / B64's existing dev rows on every (configuration, store) (difference <= 1e-12); a mismatch stops the run
  before any choice is made.
- **Dev choice:** NLMSc B62's rule (= B60's), SDMt B36's rule (as b64_run.choose): per cell the highest mean dev
  accuracy over the cell's dev stores, within 0.1 points tied, first in grid order. BEST7 as b64_run.choose. Committed
  (results/b70_choice_{enc}.json) before any test run.
- Test: the chosen configurations on the test stores (20 at M = 256, 5 at M = 1,024), once.

## Statistics and decision rules

Per encoder over its R2 cells (region statistic = mean over cells of the cell mean of the per-query difference; 95%
cluster bootstrap, entity within store, resampled within each cell; b36_verdict.boot_region; 10,000 draws for the
primaries, 2,000 otherwise). Also over the R2 cells of the 4 language models together and of the 3 sentence encoders
together (cells of different encoders are independent; same statistic), descriptive.
- **B60's decision rule** (primaries P1-P3 and every two-arm contrast): "first leads" iff the 95% lower bound > 0,
  "second leads" iff the upper bound < 0, else "level".
- **B36's bar** (P4-P5, key-free comparisons): >= +2 points with the interval above 0 = passes; <= -2 with the interval
  below 0 = loss; else tie.

Primary:
- **P1 STORE - LAMc**, **P2 STORE - NLMSc**, **P3 VSAPcX - LAMc** (B60's rule);
- **P4 STORE - BEST36**, **P5 STORE - BEST7** (B36's bar; also B60's label).

Secondary: VSAR - SDMt and STORE - SDMt (B60's rule), VSAR - BEST7 (bar), STORE - BESTc, NLMSc - LAMc, LAMc - BEST36,
BESTc - LAMc; chance-rescaled versions of P1-P5 (per cell (a - b) / (1 - 1/V)).

Reported per encoder and R2 cell: every arm's raw accuracy; region mean raw; **rescaled for chance** ((a - 1/V) /
(1 - 1/V), cell then mean); **fraction of the exact index** (a / EXACT per cell, mean) and its chance-rescaled form
((a - 1/V) / (EXACT - 1/V)); the number of R2 cells in which each arm reaches U_test = 0.9 x EXACT's test accuracy in that
cell; the dev-best family per R2 cell. For the 13 settings without R2 cells: U_dev, the dev best at 64 bits per fact and
its family, and (existing B49 test numbers, labelled) VSA-inf and LAM-inf against U, so that the paper can say whether
any budget could make a superposed memory useful there.

## Predictions (stated so that they can be wrong; made from the dev table and the disclosed B60 RH statistics)

- Q1 (P1). STORE - LAMc over R2: LAMc leads on all 4 language models; on the 3 sentence encoders LAMc leads on none
  (store leads or level on each).
- Q2 (P2). STORE - NLMSc over R2: NLMSc leads on at least 3 of the 4 language models; store leads or level on all 3
  sentence encoders.
- Q3 (P3). VSAPcX - LAMc over R2: LAMc leads on at most 2 of the 4 language models; level or VSAPcX leads on all 3
  sentence encoders.
- Q4 (P4). STORE - BEST36: loss on all 3 sentence encoders; on the language models no loss, and passes on at least 2.
- Q5 (P5). STORE - BEST7: the same labels as Q4 on at least 6 of 7 encoders.
- Q6. STORE - BESTc: BESTc ahead (upper bound < 0) on all 7; by at least 4 points on each sentence encoder.
- Q7. VSAR - SDMt over R2: VSAR leads or level on at least 3 of the 4 language models.
- Q8. Fraction of the exact index (R2 means): STORE between 0.80 and 0.92 on every encoder; BESTc >= 0.88 on every
  encoder. Cells reaching U_test (of 23): STORE <= 1, VSAPcX <= 4, LAMc <= 8, BESTc >= 16.
- Q9. Sensitivity: adding B70's NLMSc and SDMt dev accuracies to best_dev would change b* in no setting.
- Q10. Checks pass (below), and the 16-bit dev reproduction is exact.

## Readings for the paper (fixed now)

- "Least squares beat the store at equal bits on every encoder" holds where memories are useful only if P1 is "LAMc
  leads" on all 7; otherwise the paper states P1 per encoder (and per group) with labels, e.g. "at the budgets where the
  best memory reached 90% of the exact index, least squares led the store on k of 7 encoders".
- "The one-pass delta rule also beat it" likewise from P2; "least squares beat every store variant" from P3 (with P1).
- "No margin over the best classical memory" (per-ingredient verdict) holds at R2 if P4 / P5 pass on no encoder; any
  pass is named.
- Wherever BESTc is ahead of every superposed arm by >= 2 points with its interval above 0 and its dev-best family is a
  per-fact index (LHI / IDX), the paper must say that at useful budgets the best memory stores per-fact codes, and that
  the superposition comparisons are between memories that are not the best there.
- The 13 settings without a useful budget within 64 bits per fact (almost all V = 16) are named; if VSA-inf and LAM-inf
  are below U there, the paper says that no budget makes those superposed memories useful in those settings.
- The paper must not present B70 as a new design of any memory; no arm, grid or rule is changed.

## Checks (b70_check.py; second angle)

- **Plain-loop scorer:** every R2 cell accuracy of every arm and every registered point statistic (region means, raw,
  rescaled, fractions, contrasts) recomputed with plain Python loops from the job files and merged records;
  |difference| < 1e-9.
- **Bits audit:** every B70 dev and test row: NLMSc bits <= budget and p maximal by own re-implementation of the bit
  formula; SDMt bits (B64's rule incl. the default value) <= budget; and the logged bits of every existing arm at the R2
  cells where logged (LAMc, SDMt, NLMSc's B62 rows, the store's d with B32's vsa_bits) <= budget.
- **Float64 re-derivation (independent code) on a sample:** NLMSc: one test store per R2 setting with a new budget, at
  its largest R2 budget, by an own float64 pipeline (own preps and eigenbasis, rp blocks regenerated from the registered
  seeds, a genuine one-fact-at-a-time NLMS loop, own quantizer and read; b62_check's functions); SDMt: the same sample by
  b64_check.rederive_one (own float64 preprocessing, full sort, loop counters, own entropy and N scan). Bar: agreement
  >= 0.99 on every re-derived store.
- **Sample reading by the AI-assisted process:** Qwen3-8B and e5-base-v2, the R2 cell with the largest |STORE - LAMc|:
  10 queries only the store answers right and 10 only LAMc answers right, and 10 only BESTc answers right against both
  (alias, label, gold, answers), with notes.

## Compute and cut rule

CPU only. Local (16 cores, shared) for the verdict and checks; the A800 box (ssh cavin, 128 cores) or local for the
NLMSc / SDMt jobs (fresh directory ~/b70_run, code by md5 from this repository, panel codes md5-checked against
~/b66_run's B49 copies; one process per job, OMP / BLAS threads 1, nice 5, at most 48 workers). On the box: every ssh
command in the background; no retry loops on state-changing remote commands; processes stopped only by exact PID;
nothing of other users touched. Time box: 2.5 hours from 16:43 (machine clock). If time is short, cut in this order,
in a dated compute note written before any test output is read: (1) the sentence encoders' new runs, (2) SDMt (and
BEST7) at 32-64; the remaining arms are still reported on every encoder's R2 cells (their predictions exist).

## Files

b70_run.py (region, plans, jobs, choice, selftest), b70_box.sh, b70_verdict.py, b70_check.py; results/b70_*; job
outputs under b70_out/ (not committed).
