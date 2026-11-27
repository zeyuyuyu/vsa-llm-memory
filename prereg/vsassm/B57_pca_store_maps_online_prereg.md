# B57 pre-registration: the store on principal-component features, map variance, online decorrelating writes, raw accounting

Date: 2026-11-20 (US Eastern; machine clock Asia/Shanghai). Written and committed (and OpenTimestamps-stamped) before
any B57 memory has answered a dev or test query. Runs so far: b57_run.py's selftest on random toy data only. No dev or
test store has been built by B57 code. Deadline-driven scope cuts are written here (section "Scope and cuts").

## Why (referee, 11th read)

M2b: B56 showed that LAM's lead depends on its principal-component features (LAM on the store's features, LAMRF, lost
0.8-3.3 points), and on matched random features the store led LAMRF over R1 (+0.5 to +1.6); the store was never given
principal-component features. M2c: the registered store used one random feature map; B52's three further maps moved
the qwen8 R1 result against the six key-free memories by +1.3 to +1.8 points, so entity-only intervals miss a map
component. M2e: LAM's write is a batch solve over all stored keys, the bundle's is one-shot and incremental; online
decorrelating writes should be compared. M2d: raw (bits stored) accounting next to ideal. Methods: the store's chosen
feature map, nonlinearity, dimension and bits per component must be stated.

## Fixed (B53 / B56's protocol)

Panel (7 encoders), splits, stores (dev 6 + 3, test 20 + 5 per V at M = 256 / 1,024), queries, budgets (1-64 bits per
fact), accountings (ideal primary, real secondary), B53's value codes (antipodal V = 2, Sylvester-Hadamard V = 16), B56's
read rules (matched filter, LMMSE with training-split statistics, perceptron retraining) and VSAS selection rule, B49's
competitors and their test predictions (LAM = B49's dev choice among 48; BEST = the dev-chosen family among ten; BEST36 =
B36's best of the six key-free memories IDX, SF, CM, BF, FBF, SDM). V in {2, 16} only; R2 is B49's.

## Part 1 (primary): VSAP, the store on principal-component features

- Feature map: phi(y) = U_d^T (P(y) - mu) ('plin') or sign(U_d^T (P(y) - mu)) ('psgn'), where U, mu are exactly
  b49_common.LamFeat's training-split principal directions (descending variance) and mean of the prep's codes, the
  basis B49's LAM may choose; U_d = the first d directions, d = min(B32's size rule at (bits, q, acct), n). No random
  projection (a random projection of all n principal coordinates is distributed as the existing random store).
- Store: B53's bundle (value codes seeded as the lin / sgn store's), quantizer, stored-subset order. MF grid: prep
  {center, whiten} x fmap {plin, psgn} x q {1, 2, 4} x stored share {1, 1/2, 1/4, 1/8} = 48 configurations, chosen per
  cell on dev by B32's rule (ties: grid order). Then B56's read rules on the chosen bundle: LMMSE (q x alpha, 15) and
  retraining (q x E x eta, 18), and B56's STAR rule (higher dev mean; ties LMMSE) -> **VSAP**. Grid: 48 + 33.
- Accounting: the stored vector is d <= n numbers at q bits + 32-bit block scales (B32's rule); the principal basis is
  a fixed training-split component (free, as for LAM).
- **Primary bar, per encoder: region R1 (V 2 / 16, 4-16 bits per fact, both M; 12 cells), VSAP - LAM** (B49's LAM, its
  own dev choice): not behind = 95% cluster-bootstrap lower bound >= -2 points; TOST (90% interval inside +-2); ahead
  (lower bound > 0); counts of 7 / 4 LLMs / 3 sentence encoders. Secondary: per V, R0, RH, chance-normalised
  differences, VSAP - BEST, VSAP - VSAS (B56), and VSAPX = per cell the dev-best of VSAS and VSAP (ties VSAS).

## Part 2: map variance

- Maps m = 1, 2, 3: seed tag `{enc}#b57m{m}` replaces the encoder name in every seed of the store's random components
  (projection blocks, value-code signs, quantizer tie signs, the LMMSE training sample) and of LAM's rp projection
  blocks. Map 0 = the registered map (B53 / B56 / B49 files).
- Per map, each with its own dev choice: the store VSAS (B53's 72-configuration MF grid, B32's rule; B56's LM / RT and
  STAR rule on top), LAMRF (B56's LAM-RF on the map's chosen store features), LAM (B49's 48 configurations: the 24 rp
  configurations re-run with the map's projection, the 24 pca ones B49's dev results; B49's rule), BEST (B49's
  ten-family dev choice with LAM's dev accuracy replaced by the map's LAM). BEST36 does not depend on the store's map
  (fixed). VSAP has no random projection and is not re-run per map.
- Encoders (cut, compute): qwen8, e5-base-v2, MiniLM-L6 (the paper's primary LLM and the two encoders where LAM led
  beyond the margin in B56).
- Statistics over R1 for store - LAM, store - BEST, store - BEST36 (store = VSAS): per map the B56 statistic; across
  maps the mean over the four maps with a **two-level bootstrap** (maps drawn with replacement from the four; entity
  clusters resampled once per draw, shared across maps; 10,000 draws; B52's method); the between-map SD; the
  entity-only interval of the map mean; whether the registered map is the lowest of the four and by how many
  between-map SDs.

## Part 3: online decorrelating writes

- **NLMS**: one pass of the normalised delta rule over the M facts in store order, W <- W + mu f (y - f^T W)^T / f^T f,
  on LAM's feature options (rp, pca) and preps, LAM's onehot targets, p by LAM's bit rule, LAM's quantizer and bits;
  mu in {1, 0.5, 0.25}; grid 2 x 2 x 3 x 3 = 36, dev choice per cell (ties: grid order prep, feature, mu, q).
- **RLS**: recursive least squares with P0 = I / lambda is, in exact arithmetic, the ridge solution with that lambda
  after the last update; its p x p working state limits it to pca features (p <= n <= 4,096; on rp p reaches 65,536 and
  the state 4.3e9 numbers). RLS's predictions are therefore computed by the batch ridge solve of B49's pca
  configurations (24; dev choice by B49's rule over them), and the equivalence is checked with a genuine recursion
  (b57_run.rls_recursive) on a sample (check section).
- Report R1 NLMS - LAM, NLMS - VSAS, RLS - LAM, RLS - VSAS.
- **Write cost** of every memory at its chosen configuration per R1 cell, from the code's shapes (multiply-adds and
  working memory in numbers; feature computation reported separately): Hebbian bundle (ns d), retraining (+ E ns d V
  reads and updates; ns d cached features), LAM batch (M^2 p Gram + M^3 / 3 solve + p M c), NLMS (2 M p c), RLS
  (~3 M p^2; p^2 state). Median and range over the 12 R1 cells per encoder.

## Part 4: accounting

Every B57 bar is also computed under real (raw) accounting; B53's and B56's raw R1 numbers are tabulated from their
verdict files beside the ideal ones.

## Methods table

Per encoder, over the 12 R1 cells (ideal): the distribution of the store's chosen prep, feature map (lin / sgn / sgn3 =
1 or 3 sign projections), d, q and stored share for B53's VSA and B56's VSAS (read rule, alpha, E / eta), from the
merged files.

## Predictions (stated so that they can be wrong)

Q1. R1-V2 VSAP - VSAS > 0 with the interval above 0 on at least 4 of 7.
Q2. R1-V16 VSAP - VSAS < 0 with the interval below 0 on at least 5 of 7 (the Hadamard classes split the principal
    coordinates into 16 disjoint subsets; each class reads 1/16 of them).
Q3. Primary: VSAP not behind LAM on at most 3 of 7; ahead on 0.
Q4. VSAPX not behind LAM on at least 5 of 7.
Q5. Maps: between-map SD of R1 VSAS - LAM <= 0.6 points on each map encoder; the two-level interval of the map mean
    contains the registered map's point estimate on each.
Q6. The registered map is not the lowest of the four maps by more than 1 between-map SD on R1 VSAS - BEST36 on any map
    encoder.
Q7. R1 NLMS - LAM below 0 (interval below 0) on at least 5 of 7; R1-V2 NLMS - VSAS > 0 on at least 4 of 7.
Q8. |R1 RLS - LAM| <= 1 point on at least 5 of 7.
Q9. Checks: scorer |difference| < 1e-9; 0 configurations over budget; float64 re-derivation agreement >= 0.99; RLS
    recursion = batch ridge (max relative difference of W < 1e-6) on the sample.

## Checks (b57_check.py)

Plain-loop scorer of every cell accuracy and the R1 regions; bits audit (VSAP d <= min(rule, n) and maximal under the
cap; NLMS / RLS / LAM-map p by LAM's rule; map-store d by B32's rule); float64 re-derivation of VSAP (own principal
basis from numpy.linalg.eigh of the training covariance, own codes, quantizer, matched filter / LMMSE / retraining) on
one test store per (encoder, M, V) at 4 / 16 / 64 bits per fact; RLS recursion vs batch ridge on one store per V for
two encoders; hand reading: qwen8 and e5, the R1 cell where VSAP - VSAS is largest in absolute value, 10 + 10 queries.

## Scope and cuts

Deadline: the report is due 5 hours after 17:09 (machine clock). Cut order if the test phase cannot start by 20:00
(machine clock): (i) maps reduced to qwen8 only (e5, MiniLM map runs dropped); (ii) NLMS reduced to M = 256. Any cut is
written into a dated compute note before the test runs.

## Compute

Local CPU only (16 cores, nice 10; A800 box not used), b57_local.sh: dev1 (MF grids, LAM rp per map, NLMS dev), dev2
(MF choices; LM / RT / LR engines), choose, commit, test, merge, verdict, checks; at most 4 jobs at once (memory: B56
reached 0.7 GB free with 6). No paid API.

## Files

b57_run.py, b57_local.sh (committed with this file); b57_verdict.py and b57_check.py are written during dev and committed with the dev choices, before any test store exists. Results: results/b57_*; dumps under b57_out/ (not committed).
