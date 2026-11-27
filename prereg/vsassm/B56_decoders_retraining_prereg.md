# B56 pre-registration: decorrelating read (LMMSE), retrained bundle, LAM on the store's own features, and a widened store grid with a sparsifying map

Date: 2026-11-20 (US Eastern; machine clock Asia/Shanghai). Written and committed (and OpenTimestamps-stamped) before
any B56 memory has answered a dev or test query. The only runs so far: b56_run.py's selftest on random toy data, and
timing runs of the engine on single stores drawn from the TRAINING split (qwen8 M = 1,024 V = 16 and V = 2; e5 M = 256
V = 2; logs b56_out/logs/timing_train*.log). Disclosed design change from those training-split runs: on the qwen8
V = 16 training store the retrained bundle with step 1 read worse than the Hebbian one, so a smaller step (0.1, the
usual HDC retraining rate range) was added to the retraining grid before this file was committed. No dev or test store
has been built by B56 code.

## Why

Referee (10th read), M2: B49 / B53's equal-bit comparison isolated the Hebbian (matched-filter) write and read rule,
not superposition. The least-squares associative memory (LAM) that beat the store is itself a superposition (W = F^T A
is a weighted sum of the stored keys' features) with a decorrelating write. Decorrelating decoders of superpositions
exist inside the VSA framework (Kleyko, Bybee, Huang et al. 2023, Neural Computation 35(7), "Efficient decoding of
compositional structure in holistic representations": matched filter vs LMMSE vs explaining-away), and
perceptron-style retraining of bundles is standard in HDC. Also: LAM was allowed principal-component features while the
store used random features (a confound), and the store's grid should be widened with a sparsifying nonlinearity (the
usual VSA remedy for correlated keys). B56 runs all four; the result goes into the paper whatever it shows.

## What is fixed (B53's protocol)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B, e5-base-v2, bge-small, MiniLM-L6; B49's codes and md5s),
splits, stores (B32's make_store; dev 6 + 3, test 20 + 5 stores per V at M = 256 / 1,024), queries (unseen aliases),
budgets (1, 2, 4, 8, 16, 32, 64 bits per fact), accountings (ideal primary, real secondary), the store's projections,
seeds, value codes (B53: antipodal at V = 2, Sylvester-Hadamard at V = 16), stored-subset order, quantizer and its
seeds, and the competitors' dev choices and test predictions (B49's merged files as carried in B53's
b53_out/fam/{enc}/all53_*: LAM = B49's dev choice among its 48 configurations, BEST = the dev-chosen family among the
ten). Only V in {2, 16} is re-run; V in {128, 1024} (incl. R2) is unchanged and reported from B49 / B53.

## Arms (b56_run.py)

Notation per cell (M, V, bpf, acct): B53's dev choice (prep, fmap, q53, f) for the matched-filter store; phi = the
store's feature map (same projection blocks); S = sum over the stored facts (order[:floor(f M)]) of phi(k) * c(v); Q =
the store's block quantizer at q bits; r = c(0) (B53's row-0 signs); H16 = Sylvester-Hadamard of order 16.

1. **VSA-LMMSE (VSAL).** The SAME bundle Q(S) (B53's prep, fmap, stored share; q in {1, 2, 4}, d = the largest that
   fits, B32's size rule), read by the linear minimum-mean-square-error decoder of the superposition. With z = Q(S) * r
   (elementwise) and, per residue class c of the coordinates (V = 16: j = c mod 16, the 16 classes that the Hadamard
   codes keep apart, so the cross-talk covariance in the unbound frame is block-diagonal over them; V = 2: one class):
   m_c(x) = phi_c(x)^T (Sigma_c + lam_c I)^-1 z_c, Sigma_c = F_c^T F_c / T, lam_c = alpha * mean diag(Sigma_c);
   score(v) = sum_c H16[v, c] m_c (V = 16), or m vs -m (V = 2); argmax. F = phi of T rows of the TRAINING matrix
   (b32_common.Pool.train_matrix, names and aliases of training entities; a seeded sample, T = 1,024 at V = 2 and 512
   at V = 16), computed in the dual (T x T) form. alpha in {inf, 10, 1, 0.1, 0.01}; alpha = inf is the matched filter
   (B53's read). This is Kleyko et al.'s LMMSE (W = E[s s^T]^-1 E[s a^T]) with the expectation over the population of
   keys (stored facts' values independent, zero-mean codes: E[z z^T] = n_stored Sigma_phi + quantization noise, the
   scale absorbed into alpha), the query's feature standing for the stored key's (as in the matched filter).
   **Accounting:** the stored object is the same d numbers at q bits (+ 32-bit block scales) as the store; the decoder's
   statistics come from the training split only (entity-disjoint from dev and test), fixed for every store, the same
   status as Prep's whitening matrix and LAM's principal directions, which B49 counts as free. Nothing per store is
   read beyond the bundle: 0 extra bits. (A decoder using the stored keys' own Gram would need per-store statistics
   that cost far more than the budget, e.g. n(n+1)/2 numbers; B56 does not use one.)
2. **VSA-retrained (VSAT).** The same configuration's Hebbian bundle S (unquantized), retrained on the stored facts
   with the HDC / perceptron rule: mini-batches of 32 facts in store order; for every fact of the batch whose read
   argmax_v <phi(k) * c(v), S> is not its value, S += eta phi(k) * (c(v) - c(v_hat)) (every fact of a batch read with
   the bundle at the batch start); E in {1, 3, 10} epochs, eta in {1, 0.1}; then
   quantized by the store's quantizer to q in {1, 2, 4} bits (d by B32's rule; its own seeds); matched-filter read.
   **Accounting:** identical to the store (the writer uses the stored facts, as every memory's writer, incl. LAM's,
   does; only the quantized bundle is kept).
3. **VSA\* (VSAS), the primary store arm:** per cell, VSAL's or VSAT's dev choice, whichever has the higher dev mean
   accuracy (within 0.1 points tied: VSAL).
4. **LAM-RF (LAMRF).** B49's LAM (onehot targets, c = V - 1 stored columns; ridge alphas {1e-4, 1e-2, 1, 100}; q in
   {1, 2, 4}; Max quantizer per column and block with 32-bit scales; p = the largest that fits, at most 65,536;
   b49_common's algebra and bit rule) on exactly the store's features of the cell: B53's chosen prep and fmap, the
   same projection matrices (no principal components). All M facts (as B49's LAM).
5. **Widened store grid (VSAW).** B53's 72 configurations plus 72 with a sparsifying ternary map on the sign store's
   projections: phi_j(y) = sign(p_j) [|p_j| >= theta], p = y G (G = the 'sgn' map's blocks), theta = the two-sided
   N(0, 1) quantile for an active fraction kappa in {1/4, 1/16, 1/64} (p_j ~ N(0, 1) exactly for unit-norm codes;
   'tern4', 'tern16', 'tern64'); x prep {center, whiten} x q {1, 2, 4} x stored share {1, 1/2, 1/4, 1/8}. Matched
   filter read, B53's codes. Chosen on dev by B32's rule over all 144 (ties: grid order prep, fmap lin, sgn, sgn3,
   tern4, tern16, tern64, q, share).
6. **VSA-ANY (VSAX), secondary:** per cell the dev-best of VSAL's, VSAT's and VSAW's choices (ties: VSAL, VSAT, VSAW).

Dev choice for VSAL / VSAT / LAMRF: per cell, the highest mean dev accuracy over the cell's dev stores; within 0.1
points tied, first in the order (q, alpha: inf first) / (q, E, eta: 1 first) / (alpha, q). Test once, after the choices are committed.

## Tuning-grid sizes (configurations searched on dev per cell)

| arm | configurations |
| --- | --- |
| VSA (B53) | 72 |
| VSAW | 144 (72 + 72 sparse) |
| VSAL | 15 (q x alpha) on B53's 72-configuration choice |
| VSAT | 18 (q x E x eta) on B53's choice |
| VSAS | the better of VSAL / VSAT (33 on B53's choice) |
| VSAX | the best of VSAL / VSAT / VSAW |
| LAMRF | 12 (alpha x q) on the store's chosen features (from B53's 72) |
| LAM / BEST | 48 / 690 (B49) |

## Bars (ideal accounting; 95% cluster bootstrap, entity within store; 10,000 resamples for VSAS - LAM and VSAS - BEST)

- **Primary, per encoder: region R1 (V 2 / 16, 4-16 bits per fact, both M; 12 cells), VSAS - LAM.** "Not behind" =
  the 95% interval's lower bound >= -2 points (equivalence margin 2 points). **Panel count: encoders with VSAS not
  behind** (of 7; of the 4 LLM families; of the 3 sentence encoders). Also reported: TOST at +-2 points (both
  one-sided 5% tests, i.e. the 90% interval inside [-2, +2]); "ahead" (95% lower bound > 0); "LAM ahead" (95% upper
  bound < 0).
- Secondary: R1 VSAS - BEST (B49's H1-B49 bar: >= +2 with the interval above 0 = pass; and the same not-behind rule);
  per V (R1-V2, R1-V16; R0, RH by V); R0, RH; real accounting; chance-normalised differences ((acc_a - acc_b) /
  (1 - 1/V) per cell, then the region mean) and chance-normalised accuracies; component effects VSAL - VSA,
  VSAT - VSA, VSAW - VSA, LAMRF - LAM, VSA - LAMRF, VSAS - LAMRF; VSAX - LAM, VSAX - BEST.
- R2 (V 1,024) unchanged (random codes; B49's numbers).

Readings (fixed now):
- VSAS not behind LAM on all 7: the paper may say that inside the VSA framework (a decorrelating read or an
  error-driven write of the same bundle) the store is within 2 points of, or ahead of, the least-squares associative
  memory at equal bits over R1 on every encoder, naming which rule dev chose and the per-V numbers.
- Not behind on k < 7: the encoders where LAM stays ahead by more than the margin are named; "LAM beats the store at
  equal bits" stands for them, stated as the write / read rule, not as superposition.
- If VSAT closes the gap and VSAL does not: the paper must say the gap was the write rule (Hebbian vs error-driven);
  a decoder with population statistics cannot remove the cross-talk of the particular stored keys.
- LAMRF far below LAM: the feature confound is named and its size given; VSA - LAMRF is the like-for-like comparison.
- Whatever B56 shows, the paper must not say that superposition per se loses to least squares (B49 / B53 compared
  rules, not superposition).
- vs BEST: as B49 (passes named; no "LLM codes" statement unless all four LLM families pass).

## Predictions (stated so that they can be wrong)

P1. VSAL adds little: |R1 VSAL - VSA| <= 1 point on at least 6 of 7 encoders (whiten prep, chosen in most B53 cells,
    already removes population correlation; what remains is the cross-talk of the particular stored keys).
P2. VSAT helps at V = 2: R1-V2 VSAT - VSA > 0 with the interval above 0 on at least 5 of 7 encoders.
P3. Primary: VSAS not behind LAM on at least 6 of 7 encoders; TOST-equivalent on at least 4 of 7; VSAS ahead of LAM
    (95% lower bound > 0) on at most 4 of 7.
P4. LAMRF - LAM over R1 below 0 with the interval below 0 on at least 4 of 7 encoders.
P5. Sparse maps: dev picks a sparse map in at most one third of the R1 cells (ideal) on every encoder, and
    |R1 VSAW - VSA| <= 1 point on at least 6 of 7.
P6. vs BEST: over R1 BEST ahead of VSAS (interval below 0) on at least 6 of 7; H1-B49 passes on 0.
P7. Per V: R1-V2 VSAS - LAM below 0 (interval below 0) on at least 4 of 7; R1-V16 VSAS ahead of LAM on all 4 LLMs.
P8. Checks: VSAL at alpha = inf reproduces B53's test predictions (agreement >= 0.995 on every encoder, differences
    only at exact ties); independent scorer |difference| < 1e-9; 0 configurations over budget; re-derivation
    agreement >= 0.99.

## Checks (second angle; b56_check.py)

- Reproduction: VSAL with alpha = inf (computed on every test store of the chosen cells) against B53's VSA test
  predictions where B53's q equals the chosen q; agreement per encoder, and whether the disagreements are ties.
- Independent scorer: plain loops over the merged files; every cell's VSAS - LAM, VSAS - BEST and the R1 regions
  against the verdict.
- Bits audit: for every test record and arm, the logged d (store arms) or p (LAMRF) against an own re-implementation
  of the size rules: bits <= budget and bits(d + 1) > budget (p maximal unless capped at 65,536).
- Re-derivation (float64, independent code): the LMMSE read by a direct primal solve per class (numpy.linalg.solve
  on Sigma_c + lam_c I) and the retraining by an online (fact-by-fact within the batch semantics) loop, on one test
  store per (encoder, M, V) and the chosen configurations; agreement with the logged predictions.
- Hand reading: Qwen3-8B and e5-base-v2, the R1 cell where VSAS - LAM is lowest: 10 queries only VSAS answers
  right and 10 only LAM answers right (alias, label, gold), notes by me.

## Compute

Local CPU only (16 cores, nice 10; the A800 box is not used). b56_local.sh: dev = the 42 sparse parts (3 at a time)
and the LM / RT / LR engine on the dev stores (per encoder and (M, V)); choose; commit the choices; test (chosen
configurations only); merge; verdict; checks. No paid API.

## Files

b56_run.py, b56_local.sh, b56_verdict.py, b56_check.py; results/b56_*; dumps under b56_out/ (not committed).
