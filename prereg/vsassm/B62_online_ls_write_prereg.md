# B62 pre-registration: online decorrelating writes (one-pass NLMS, RLS) with LAMc's corrected options, against the store at equal bits

Date: 2026-11-23 (machine clock Asia/Shanghai, about 02:30; US Eastern 2026-11-22 afternoon). Written and committed, then
OpenTimestamps-stamped, before any B62 development or test run. The paper is not edited by B62.

## What has been seen before this file (disclosed)

- B57's online arms (NLMS one pass, mu in {1, 0.5, 0.25}; RLS = ridge on B49's pca features) with B49's defective
  features (wrong whiten pca basis), V - 1 columns and per-column scales: NLMS behind B49's LAM over R1 on 6 of 7 by
  0.9-2.75, ahead of the store at V = 2 on 5 of 7, behind it at V = 16 on 6 of 7 (prereg/B57_note_after_test.md).
- B60's result (prereg/B60_note_after_test.md, results/b60_verdict.txt): store (B53) - LAMc over R1 -4.06 to -6.53
  (LAMc leads on 7 of 7), R1-V16 -4.17 to -10.10, R1-V2 -0.19 (MiniLM, level) to -5.47. LAMc's dev choices and test
  predictions exist (results/b60_choice_*.json, b60_out/jobs).
- No B62 code has been run on any dev or test store. b62_run.py's selftest (passed) uses random toy data only; the training-split
  feature-norm statistics (below) use training codes only.

## Why (referee)

LAMc is a batch ridge solve over all M stored facts. A memory beside a language model receives facts one at a time.
B57's online arms were run only on B49's defective configuration and were never re-run. The referee also asks for write
cost and working state. Question: with LAMc's corrected configuration, does an ONLINE decorrelating write (one pass of
the normalised delta rule; recursive least squares) still beat the store's Hebbian write at equal stored bits?

## Fixed (B60's protocol, unchanged)

Panel (Qwen3-8B, Granite-3.3-8B, Llama-3.1-8B, Mistral-7B; e5-base-v2, bge-small, MiniLM-L6), B49's codes (b49_out/
b32_data, md5 as B49 / B60), splits, B32's make_store stores (dev 6 + 3, test 20 + 5 per (M, V), M = 256 / 1,024), queries
(unseen aliases), **region R1 only**: V in {2, 16} x 4, 8, 16 bits per fact x both M (12 cells per encoder), reported as
R1-V2, R1-V16 and R1; **ideal accounting only** (c p H_q + 32 x scale groups, B60's bits()). LAMc's feature maps
(b60_run.FeatC: 'rp' = B49's Gaussian blocks seeded ('LAMG', enc, prep, j0); 'pca' = the prepped code on the raw codes'
principal axes U0), targets ('V-1', 'V'), scales ('col', 'flat'), quantizer (b60_run.quantW), p rule (the largest p that
fits; rp at most 65,536, pca at most n; maximal), read (score(v) = f(q)^T W_hat t(v), argmax, first index on ties; p = 0
-> the store's majority value). Write order: one fixed random order per store, b32_run.store_order(st) (seed ('subset',
split, M, V, rep); the order the store uses for its stored subset).

## Arms

**NLMSc (primary online arm): one pass of the normalised delta rule.** W_0 = 0; for t = 1..M in write order:
e_t = y_t - W_{t-1}^T f_t, W_t = W_{t-1} + mu f_t e_t^T / (f_t^T f_t + 1e-12); y_t = the target row of fact t (LAMc's
'V-1' or 'V' targets). Output W = W_M ('last') or the Polyak average of the iterates W_bar = (1/M) sum_t W_t ('avg';
one more p x c accumulator, still one pass). Then LAMc's quantizer, bits and read. Grid (order = tie order): scale (col,
flat) > target (V-1, V) > (prep, feat) (center rp, center pca, whiten rp, whiten pca) > mu (1, 0.5, 0.25, 0.1, 0.03) >
output (last, avg) > q (1, 2, 4): 480 configurations per (M, V, bpf). No other stopping rule (exactly one pass, every fact
written once).
- Computation (exact, not an approximation): with K = F F^T in write order and L its strictly lower triangle, the online
  recursion's innovations d_t = mu e_t / ||f_t||^2 solve (L + diag(K + 1e-12) / mu) D = Y (forward substitution: fact
  t's innovation depends only on facts written before it), and W_M = F^T D, W_bar = F^T (omega * D), omega_t =
  (M - t + 1) / M. B62 computes this dual form (scipy solve_triangular, float64); the selftest and the checks compare it
  with a genuine one-fact-at-a-time primal loop.

**RLSc: exponentially weighted recursive least squares.** P_0 = I / delta, W_0 = 0; per fact: k = P f / (beta + f^T P f),
W += k (y - W^T f)^T, P = (P - k f^T P) / beta. After the M-th fact W is, in exact arithmetic, the minimiser of
sum_t beta^(M - t) ||y_t - W^T f_t||^2 + beta^M delta ||W||^2, i.e. W = F^T (K + delta diag(beta^t))^{-1} Y (t = 1..M in
write order; beta = 1: ridge with lam = delta). B62 computes this closed form (Cholesky in float64; the primal p x p form
when p < M, the dual M x M form otherwise; the same W); the genuine recursion is run in the selftest and the checks.
- delta is fixed before any fact arrives: **delta = alpha x s_p**, s_p = the mean of ||f_p(x)||^2 over a fixed sample of
  2,000 training-split codes (seed ('B62TN', enc); the same prep and feature map; precomputed once per (encoder, prep,
  feat), results/b62_trainnorm_*.npz). (LAMc's lam = alpha x the store's own mean ||f||^2 is a batch statistic.)
- Grid: scale > target > (prep, feat) > alpha (1e-4, 1e-2, 0.1, 1, 10, 100) > beta (1, 0.999, 0.99) > q (1, 2, 4): 864
  configurations per (M, V, bpf).

**RLScS (RLS with its working state charged).** As RLSc, but p = the largest p with 32 x p (p + 1) / 2 (the symmetric
p x p working matrix at 32 bits per number) + bits(p, c, q, ideal, scale) <= the budget; same 864-configuration grid, same
dev rule. Stated in advance: at R1 budgets (1,024-16,384 bits) this leaves p <= 31 features, so RLS with its working
state charged is infeasible at the sizes RLSc chooses in every R1 cell; RLScS reports what is left.

**STORE** = B53's VSA (final design, matched filter), the existing test predictions in b59_out/fam/{enc}/all59_*
(the same cells as B60). **LAMc** = B60's dev-chosen configuration, its existing test predictions (b60_out/jobs), as the
batch reference. Neither is re-run.

**Dev choice (B60's rule, = B49's).** Per R1 cell (M, V, bpf; ideal): the highest mean dev accuracy over the cell's dev
stores; within 0.1 points tied, the first in the grid order above. Separately for NLMSc, RLSc and RLScS. Test: the
chosen configuration only, run once, after the choices are committed.

## Primary bar (ideal; 95% cluster bootstrap, entity within store, resampled within each cell; region statistic = mean over cells of the cell mean; b36_verdict.boot_region; 10,000 draws)

Per encoder, **STORE - NLMSc** and **STORE - RLSc**, SEPARATELY on R1-V2, R1-V16 and R1. Decision rule (B60's):
"store leads" iff the 95% lower bound > 0; "online LS leads" iff the upper bound < 0; otherwise "level". Counts over the
4 language models and the 3 sentence encoders. NLMSc is the primary online arm (its write is O(p c) per fact, like the
store's O(d)); RLSc is reported with its working state.

Secondary (same statistic): **NLMSc - LAMc** and **RLSc - LAMc** ("LAMc leads" iff upper < 0, "online leads" iff lower >
0; 10,000 draws); STORE - RLScS (2,000 draws); NLMSc - RLSc (2,000). Dev-choice composition (scale, target, prep:feat,
mu / output, alpha / beta, q) over the 84 R1 cells.

## Accounting, write cost, read cost, working state (stated in advance)

- Stored bits as in B60 (ideal): NLMSc and RLSc store W_hat only (c p numbers at H_q bits + 32-bit scales), the same as
  LAMc. Bits audit as below.
- **RLS working state**: the p x p matrix P (p (p + 1) / 2 numbers, symmetric), reported per R1 cell at the chosen p, NOT
  charged in RLSc; charged in RLScS (above). For every arm the working state during the write is reported in numbers
  and as 32-bit bits / budget (not charged for any arm except RLScS): STORE d (float accumulator) + d (one fact's
  features); NLMSc p c (W) + p c ('avg') + p; RLSc p (p + 1) / 2 + p c + p; LAMc (batch) M p (features) + M^2 (Gram) +
  p c. (A 32-bit float accumulator of d or p c numbers is also larger than the stored q-bit vector: stated so that the
  charged label is not read as a statement about RLS alone; RLS differs in being quadratic in p.)
- **Write cost** in multiply-adds, per fact and total for the M facts, from the chosen shapes per R1 cell (feature
  computation reported separately): STORE: ns facts x d (bind and accumulate) [features ns K n d, K = 3 for sgn3 else 1];
  NLMSc: M (2 p c + p) (+ M p c for 'avg') [features M n p]; RLSc: M (2 p^2 + 2 p c + 3 p) (+ M p^2 if beta < 1)
  [features M n p]; LAMc (batch, the cheaper exact form): min(dual M^2 p + M^3 / 3 + M^2 c + M p c, primal M p^2 + p^3 / 3
  + M p c + p^2 c) [features M n p]. Median and range over the 12 R1 cells per encoder; ratios to STORE.
- **Read cost per query** (multiply-adds): STORE K n d (features) + d + V d (scores against every value code; a fast
  Hadamard transform could lower V d, noted); LAMc / NLMSc / RLSc n p (features) + p c (+ V for the 'V-1' decode).

## Predictions (stated so that they can be wrong)

Q1 (primary). R1 STORE - NLMSc: NLMSc leads on at least 5 of 7; the store leads on none.
Q2. R1-V2 STORE - NLMSc: NLMSc leads on all 4 language models; on the 3 sentence encoders NLMSc leads on at most 2
    (MiniLM, level against LAMc in B60, is not "NLMSc leads").
Q3. R1-V16 STORE - NLMSc: NLMSc leads on at least 5 of 7.
Q4. NLMSc - LAMc: R1-V2 within [-1.5, +0.5] points on at least 6 of 7; R1-V16 LAMc leads on at least 5 of 7, by 1 to 5
    points on each of those.
Q5. RLSc - LAMc: |R1| <= 0.5 points on at least 6 of 7; the STORE - RLSc labels equal B60's STORE - LAMc labels in at
    least 19 of the 21 (encoder x region) cases.
Q6. RLSc's dev choice: beta = 1 in at least 75 of the 84 R1 cells.
Q7. NLMSc's dev choice: output 'last' in at least 70 of 84 R1 cells; mu = 1 in at least half of the 42 V = 2 cells; mu <=
    0.25 in at least half of the 42 V = 16 cells.
Q8. The dev-chosen RLSc's working state (32 x p (p + 1) / 2 bits) exceeds the budget in 84 of 84 R1 cells; R1 STORE -
    RLScS: store leads on at least 6 of 7.
Q9. Costs, median R1 cell per encoder: NLMSc's write (excluding features) per fact within a factor 4 of the store's;
    RLSc's at least 100 times the store's; LAMc's batch total at least 10 times NLMSc's total.
Q10. Checks pass: plain-loop scorer |difference| < 1e-9; 0 rows over budget and 0 not maximal; primal NLMS loop = dual
    form (max |dW| / max |W| < 1e-8) and genuine RLS recursion = closed form (< 1e-6) on the sample; float64
    re-derivation agreement >= 0.99.

## Readings (fixed now) for the paper's claims

- "a decorrelating least-squares write beat the Hebbian write on every encoder": SURVIVES WITH "online" (the paper may
  add "also when written one fact at a time, by one pass of the delta rule") iff R1 STORE - NLMSc is "online LS leads" on
  all 7. If on fewer: the sentence stands for the batch solve (B60) only, and the paper states the online result per
  encoder with labels. If the store leads NLMSc over R1 on any encoder: the paper must say that the online one-pass
  write did not beat the Hebbian write there.
- abstract "a least-squares memory, also a sum, beat it on every encoder": "also a sum" is structural (W = sum_t f_t
  a_t^T for LAMc; = sum_t f_t d_t^T for NLMSc, with d_t depending only on earlier facts). The sentence SURVIVES as written
  iff NLMSc leads over R1 on all 7 (an online sum also beats it); otherwise it MUST specify the batch solve ("a
  least-squares memory solved over all stored facts") and the online result goes beside it.
- RLS: if RLSc = LAMc (Q5), the paper may say "recursive least squares, which writes one fact at a time, gives the same
  memory, at a write cost of order p^2 per fact and a p x p working matrix that is larger than the stored budget in every
  condition"; it MUST NOT call RLS an online write at equal bits without that working-state statement.
- The paper must not present B62 as a new design of the store; the store's numbers are unchanged.

## Checks (b62_check.py; second angle)

- **Plain-loop scorer**: every R1 cell accuracy (STORE, LAMc, NLMSc, RLSc, RLScS) and the R1 / R1-V2 / R1-V16 point
  statistics of the primary and secondary pairs recomputed with plain Python loops from the job files and merged files;
  |difference| < 1e-9.
- **Bits audit**: every dev and test row: bits <= budget and maximal (bits(p + 1) > budget unless p is at its cap), by
  own re-implementation; RLScS rows also with the state term.
- **Recursion equivalence**: on one test store per (encoder, V) at M = 256 (and one per V at M = 1,024 for two encoders),
  the chosen NLMSc configuration at 4 and 16 bits per fact by a genuine primal one-fact-at-a-time loop (float64) against
  the dual form; the chosen RLSc configuration by a genuine RLS recursion where p <= 4,096 (larger p: not run, stated).
- **Float64 re-derivation (independent code)**: the same sample, own features (own eigh of the raw training covariance
  for pca; rp blocks from the registered seeds), own primal NLMS / RLS recursion, own quantizer, own read; agreement of
  the predictions with the logged ones >= 0.99.
- **Sample reading by the AI-assisted process**: Qwen3-8B and e5-base-v2, the R1 cell with the lowest STORE - NLMSc and
  the R1 cell with the highest: 10 queries only NLMSc answers right and 10 only the store answers right (alias, label,
  gold, both answers), with notes.

## Compute and time rule

CPU on the A800 box (ssh cavin): fresh directory ~/b62_run (code by md5 from this repository; codes md5-checked against
~/b60_run's B49 copies), one process per job (OMP / BLAS threads 1, nice 5), at most 48 workers; no GPU. Every ssh
command runs in the background; no retry loops on state-changing remote commands. Results pulled back into b62_out/ and
results/b62_*. Time budget: 2 hours from 02:19 (machine clock). Priority if time is short: (1) R1 on the 4 language
models, NLMSc (primary) and RLSc; (2) the sentence encoders; (3) RLScS; (4) the sample reading. Any cut is written into a
dated compute note before any test output is read.

## Files

b62_run.py (engine, plans, dev choice, costs), b62_box.sh, b62_verdict.py, b62_check.py; results/b62_*; job outputs
under b62_out/ (not committed).
