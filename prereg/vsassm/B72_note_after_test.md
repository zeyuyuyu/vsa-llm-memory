# B72 note after the test (test run once; results/b72_verdict.txt, b72_predictions.txt, b72_check.txt, b72_sample_reading.txt)

Pre-registration ec233f4 (19:40:33, OpenTimestamps d7d6186); dev choices 38ad411 (sentence encoders, 19:51), 6b4589d
((A) / (D) language models, 20:12), 4e96c0f ((B), 20:32); tests once: sentence encoders 19:51-19:55 (405 jobs), language
models (A) 20:13-20:30 (465 jobs, local), (B) 20:33-20:41 (3,970 jobs, box); 0 failures; compute notes 1-7. No cut beyond
the two made in the pre-registration ((B) without VSAPcX - LAMc and without the sentence encoders). Deviations, all
procedural and recorded before the test output was read: dev split between machines and re-ordered on the box (notes
1, 3, 5); (B) verdict computed on the box over the slow link, its cost-of-charge secondaries without interval (note 2);
two-step dev choice (note 3); a prep-index bug in the (B) choice / plan for LAMC and NLMSC fixed before any (B) choice or
test (note 6). R1 only (V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024), ideal accounting. Labels: B60's rule on the
one-level 95% interval (entity within store); two-level = test stores, then entities.

## (A) The least-squares write in the store's own format (same features, Hadamard codes, d, quantizer, bits per number, matched-filter read)

OPTR = the final design's format, OPTX = the strongest variant's format (VSAPcX's source per cell). Points, R1 (V = 2 | V = 16).

| encoder | OPTR - STORE | OPTR - LAMc | OPTR - NLMSc | OPTX - VSAPcX | OPTX - LAMc | OPTX - NLMSc |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | +1.61 [+1.21, +2.00] OPTR (+2.12 / +1.09) | -2.45 LAMc (-1.82 / -3.08) | -0.50 level | +0.67 [+0.35, +0.98] OPTX (+1.04 / +0.30 level) | -1.34 LAMc (+0.01 level / -2.69) | +0.61 OPTX (two-level: level) |
| Granite-3.3-8B | +1.66 OPTR (+0.97 / +2.35) | -3.89 LAMc | -1.21 NLMSc | +0.74 OPTX (+1.18 / +0.29 level) | -2.74 LAMc | -0.06 level |
| Llama-3.1-8B | +1.61 OPTR (+2.67 / +0.56, two-level level) | -3.95 LAMc | +0.10 level | +0.84 OPTX | -3.18 LAMc | +0.87 OPTX |
| Mistral-7B | +1.04 OPTR (+1.35 / +0.72) | -4.20 LAMc | -3.75 NLMSc | +1.69 OPTX | -2.22 LAMc | -1.77 NLMSc |
| e5-base-v2 | +1.47 OPTR (+1.30 / +1.64) | -5.06 LAMc | -2.01 NLMSc | +0.84 OPTX | -4.47 LAMc | -1.42 NLMSc |
| bge-small | +2.37 OPTR (+1.89 / +2.85) | -1.80 LAMc (+0.84 level / -4.45) | +0.35 level (V = 2 OPTR +2.94) | +1.76 OPTX | -1.72 LAMc | +0.44 level |
| MiniLM-L6 | +1.51 OPTR (+1.00 / +2.01) | -2.80 LAMc (+0.81 level / -6.40) | -0.65 level (V = 2 OPTR +2.08) | +1.17 OPTX | -2.49 LAMc | -0.35 level |

Replacing the Hebbian write by the ridge least-squares write of the SAME stored vector (closed form per Hadamard residue
class; the store's quantizer and bits) answers 1.0 to 2.4 more queries per hundred over R1 on every encoder, and also with
two values (+1.0 to +2.7) and with sixteen (+0.6 to +2.9) separately (all one-level intervals above 0; the two-level
interval includes 0 only for Llama-3.1-8B at V = 16). For the strongest variant's format the gain is +0.7 to +1.8 (7/7;
at V = 16 level on Qwen3-8B and Granite). The fixed format costs nothing measurable: OPTR - LSFR (least squares with free
value columns on the same random features) is level on 7/7 over R1 (-0.45 to +0.11); at V = 2 the two formats coincide
(one class, one column) and give identical predictions in most cells. Freeing the bits per number (OPTQ) changes at
most 0.3 points (-0.04 to +0.30; e5 +0.30 one-level, level two-level). LAMc still leads both optimal-format stores on 7/7 over R1 and at V = 16 (by 1.3 to 5.1 over
R1; at V = 2 the strongest-variant format is level with LAMc on 5 of 7: where it uses whitened principal features, a
one-class least-squares write in that format is essentially LAMc's memory, +0.00 / +0.01 on Mistral / Qwen3-8B). Against the one-pass online write (NLMSc) the optimal-format store is
level on 4 of 7 and behind on Granite, Mistral and e5 (OPTX: level 3, ahead 2, behind 2).

**Exact decomposition of STORE - LAMc over R1** (write at the store's format + format + features):

| encoder | total | write STORE - OPTR | format OPTR - LSFR | features LSFR - LAMc | largest |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | -4.06 | -1.61 | +0.10 | -2.55 | features |
| Granite-3.3-8B | -5.55 | -1.66 | -0.21 | -3.68 | features |
| Llama-3.1-8B | -5.56 | -1.61 | -0.07 | -3.88 | features |
| Mistral-7B | -5.23 | -1.04 | -0.45 | -3.75 | features |
| e5-base-v2 | -6.53 | -1.47 | -0.24 | -4.82 | features |
| bge-small | -4.18 | -2.37 | +0.11 | -1.92 | write |
| MiniLM-L6 | -4.30 | -1.51 | -0.37 | -2.43 | features |

At the store's own format and bits the Hebbian write costs 1.0 to 2.4 points; the fixed value codes cost about nothing;
the rest of least squares' lead (1.9 to 4.8) is its dev-chosen features (whitened principal coordinates). Dev audit: the
optimal write chose all facts (f = 1) in 82 of 84 cells and the Hebbian limit (alpha = 1e4) in 6; quantizing it to the
store's q cost 2.0-3.9 points on dev (unquantized 49.0-56.6% against quantized 47.0-52.8%; the Hebbian store 44.9-50.5%).

## (C) Store-level inference with few clusters (t interval over per-store differences; exact sign-flip test over the test stores)

For the four headline contrasts over R1 the stratified t interval (Welch-Satterthwaite df 7-43) gives the one-level label
in 28 of 28 cases; over all 140 (contrast x encoder x region) cases computed, in 135 (the 5 changes all become "level": bge
R1-V2 STORE - LAMc -1.05, STORE - NLMSc +1.04, VSAPcX - NLMSc +1.83; Qwen3-8B OPTX - NLMSc +0.61; MiniLM LAMc+ - LAMc
+0.65). The exact permutation test (2^50 sign vectors over R1's 50 test stores, by meet-in-the-middle) rejects every
headline lead on the four language models except Granite's VSAPcX - NLMSc (STORE - LAMc and VSAPcX - LAMc p < 2e-6;
STORE - NLMSc p <= 0.02) and fails to reject 8 one-level leads in all: Granite VSAPcX - NLMSc (-0.80, p = 0.18; also level two-level), bge and MiniLM
VSAPcX - NLMSc over R1 (-1.32 p = 0.17, -1.52 p = 0.075), the three bge R1-V2 cases above, Qwen3-8B OPTX - NLMSc
(+0.61, p = 0.16), and bge LAMc+ - LAMc (+1.07, p = 0.062 = 2 / 2^5: the difference lives in the five M = 1,024 stores
only, so no sign-flip test can go lower). The t interval is 1.36 times as wide as the one-level interval on average and
about as wide as the conservative two-level one (1.01). Every (A) primary over R1 keeps its label under the t interval
except Qwen3-8B OPTX - NLMSc.

## (D) Nonlinear random features for least squares, and where the store leads LAMc

LAMc+ (LAMc's 288 options + 288 with sign / sgn3 random features; kept LAMc's choice unless better by > 0.1 dev points)
took a sign map in 6 of 84 cells, all at V = 2 on bge (M = 1,024, 8 / 16 bits per fact) and MiniLM (four cells). LAMc+ -
LAMc over R1: 0.00 on the four language models and e5; +1.07 [+0.77, +1.36] bge, +0.65 [+0.24, +1.06] MiniLM (V = 2: +2.14,
+1.29). STORE - LAMc+ over R1: LAMc+ leads on 7 of 7 (-4.06 to -6.53; bge -5.24, MiniLM -4.95). The p-value floor of the
sign-flip test binds here: bge's LAMc+ - LAMc lives in five M = 1,024 stores only (p = 0.062).
Heterogeneity of STORE - LAMc (84 cells): the store's point estimate is ahead in 7, its cell interval above 0 in 4:
Qwen3-8B M = 1,024, V = 16, 4 bits per fact (+2.23 [+0.28, +4.29]), bge M = 1,024, V = 2, 16 and 8 bits per fact (+7.19,
+3.63), MiniLM M = 1,024, V = 2, 16 bits per fact (+3.67). Reading from the configurations: (i) at V = 2 with 8-16 bits per
fact on the 384-dimensional sentence codes LAMc's principal features stop at p = n = 384 (bge, MiniLM) while the store's
sign-product (sgn3) random features use the whole budget; adding sign features to least squares (LAMc+) recovers exactly
these cells (cell accuracies, store / LAMc / LAMc+: bge 8 bits 62.2 / 58.6 / 64.4, 16 bits 65.6 / 58.4 / 65.4; MiniLM 16 bits
65.5 / 61.8 / 66.3). (ii) At M = 1,024, V = 16, 4 bits per fact (4,096 bits) LAMc's sixteen columns leave it p = 254 (Qwen3-8B) /
117 (Mistral) features, while the store keeps about 4,000 numbers at q = 1; this is the one language-model cell with a
store lead.

## (B) Symmetric side information (every memory's training-derived components at 8 bits per number, amortised over K stores; language models)

| encoder | K | STORE - LAMc | first design - SDMt | STORE - NLMSc |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 1e3 / 1e4 / 1e5 / inf | -8.98 / -6.65 / -3.57 / -4.06 (LAMc 4/4) | +6.68 / +3.65 VSAR; +0.38 level; -1.89 SDMt | -4.97 / -3.56 / -2.68 / -2.11 (NLMSc) |
| Granite-3.3-8B | | -7.63 / -6.57 / -5.12 / -5.55 | +8.10 / +3.73 VSAR; +0.36 level; -2.25 SDMt | -4.45 / -3.02 / -2.94 / -2.87 |
| Llama-3.1-8B | | -8.76 / -6.47 / -4.71 / -5.56 | +7.71 / +4.49 VSAR; +0.90 level; -1.20 SDMt | -4.06 / -2.55 / -1.61 / -1.51 |
| Mistral-7B | | -13.04 / -9.86 / -4.69 / -5.23 | +4.66 VSAR; -0.50 level; -1.12 SDMt (two-level level); -3.56 SDMt | -7.39 / -4.65 / -3.97 / -4.79 |

(R1, points; every STORE - LAMc and STORE - NLMSc label is "LAMc / NLMSc leads" at every K.) At K = 1e3 no memory can
afford the whitening matrix (67,142 bits per store against 1,024-16,384); at 1e4 only the two largest budgets of M =
1,024 can. Every memory loses accuracy when charged (R1 point differences against K = infinity, no interval, note 2):
store -6.5 to -12.7 at 1e3 / -4.7 to -9.5 at 1e4 / -1.3 to -2.3 at 1e5; LAMc -3.8 to -4.9 / -3.7 to -4.9 / -1.7 to -2.8;
NLMSc -4.4 to -10.1 / -4.1 to -9.7 / -1.2 to -2.8; first design -3.9 to -10.4 / -3.0 to -8.1 / -0.5 to -1.8; SDMt -14.3 to
-18.6 / -9.0 to -11.2 / -3.1 to -4.2. The store depends on whitening more than least squares does, so symmetric charging
widens its gap to LAMc at small K (to 7.6-13.0 points at K = 1e3) and leaves it about unchanged at 1e5. The SDM's lead over
the first design is a K = infinity effect: it reverses at K = 1e3 on 4 of 4 and at 1e4 on 3 of 4 (Mistral level), and is
level at 1e5 on 3 of 4 (Mistral: SDMt still ahead, -1.12, level two-level).

## Predictions

Held: QA1, QA2, QA4, QA5, QA6, QD1, QD2, QB1, QB2, QB3. Failed: QA3 (NLMSc leads the optimal-format store on 3 of 7, not
>= 4; OPTX: 2, not >= 3 - the format-matched write is closer to the online write than I expected), QC1 in part (labels
agree 28 / 28 as predicted, but the permutation test does not reject 2 of 25 headline leads with |difference| >= 1:
bge and MiniLM VSAPcX - NLMSc), QC2 in part (t interval wider than one-level, held; narrower than two-level, failed:
1.01), QD3 in part (7 cells with the store ahead, <= 8 held; not all at V = 2, and 4 store leads, not <= 3), QX in part
(scorer, bits and permutation checks pass; the re-derivation bar is missed on 2 of 112 stores, explained below).

## Checks (results/b72_check.txt)

Plain-loop scorer: 1,092 cell accuracies, 414 region statistics and 185 (C) statistics with their t intervals (140 distinct)
recomputed by plain loops, max |difference| 2.8e-17; the box's (B) statistics by plain loops on the box, max 2.8e-17. Bits: 758,088 box rows and
67,095 local rows (every OPT, LAMP and (B) dev and test row; (B) with its charge), 0 over budget, 0 not maximal, 0 formula
or charge mismatches. Permutation: 140 exact p-values against 200,000 random sign vectors each, all within 4 Monte Carlo
standard errors. Float64 re-derivation (own features, codes, primal ridge, quantizer, read) of OPTR and OPTP on one test
store per (encoder, M, V) at 4 and 16 bits per fact: OPTR 56 / 56 >= 0.99 (min 0.994), OPTP 54 / 56 (all 64 language-model
rows 0.999-1.000); the bar is missed on 2 of 112. The two below (MiniLM OPTP, M = 1,024, V = 16, centre / principal sign features at d = n = 384, 0.879) were traced to the numerically
arbitrary eigenvectors of the three near-zero-variance directions of the centred MiniLM codes (std 1e-9 to 1e-17): two of
them rotate with the BLAS thread count (|cos| 0.998; the first 381 agree to 6e-8), so their sign features differ; recomputed single-threaded as the jobs ran, the pipeline reproduces its logged
predictions exactly (1.000; b72_diag_minilm.py), and its solve agrees with the independent primal solve to 2e-14. That cell
reads near chance (9.8%). Sample reading by the AI-assisted process: no scoring artefact; the optimal write's gain is the
difference of two large sets of near-copy aliases read differently.

## Literature (supporting / conflicting / new; recalled, not re-fetched)

Supporting: the pseudo-inverse / least-squares associative memory removes the cross-talk that the correlation (Hebbian)
memory keeps for correlated keys (Kohonen 1972; Personnaz, Guyon and Dreyfus 1986, projection rule); in hyperdimensional
classifiers, class vectors fitted by error-driven or least-squares updates beat plain bundling at the same dimension
(e.g. OnlineHD, Hernandez-Cano et al. 2021), consistent with OPTR > STORE in the store's own format. Few-cluster inference:
sign-change randomization tests over clusters (Canay, Romano and Shaikh 2017, Econometrica) have a floor of 2 / 2^q with q
clusters (our p = 0.062 at 5 stores), and small-sample t intervals with Satterthwaite degrees of freedom are the standard
remedy for few clusters (Bell and McCaffrey 2002; Cameron, Gelbach and Miller 2008 for the wild cluster bootstrap).
Nonlinear random features: least squares on sign random projections approximates kernel ridge with an arc-cosine kernel
(Cho and Saul 2009; Rahimi and Recht 2007 for random features), which matters exactly where linear features run out (p =
n). Shared codebooks / transforms are usually not charged per item in quantization and hashing (e.g. product quantization,
Jegou et al. 2011); our (B) shows the ranking of memories depends on that convention for the SDM, not for least squares.
Conflicting: none found. New here: at equal stored bits on frozen language-model codes, the write rule at the store's own
format costs 1.0-2.4 points and the fixed Hadamard format costs nothing, so the larger part of least squares' lead is its
features.

## What the paper may and must not say

**Write rule.** MAY say: written by least squares in the store's own format - the same features, value codes, dimension,
quantizer, bits per number and read - the store answered 1.0 to 2.4 more queries per hundred over the region on every
encoder (with two and with sixteen values separately), and its strongest variant 0.7 to 1.8 more; so at the store's own
format and bits the Hebbian write costs 1 to 2.4 points, the fixed Hadamard value codes cost nothing measurable (least
squares with free value columns on the same features was level on all seven), and the remaining 1.9 to 4.8 points of least
squares' lead come from its principal-component features. MUST NOT say: that the write rule explains most of least
squares' lead (it is the smaller part on six of seven encoders; the larger on bge-small), that the Hebbian write is optimal
for the store's format, that the optimal-format store matches least squares (LAMc leads it on all seven) or beats the
one-pass online write (level on four, behind on three), or that these optimal-format stores are the store's design (they
are diagnostic variants).

**Side information.** MAY say: when every memory's training-derived side information (mean, whitening matrix, principal
basis, SDM locations, seeds) is charged at 8 bits per number and shared by K stores, least squares and its online form
still led the store on all four language models at K = 1e3, 1e4 and 1e5 (by 3.6 to 13.0 and 1.6 to 7.4 points), and charging widened the
gap at small K because the store needs the whitening matrix more; the SDM's lead over the first design held only when side
information is free: the first design led at K = 1e3 on all four and at 1e4 on three, and the two were level at 1e5 on
three. MUST NOT say: that the comparisons are free of side information or that least squares wins by side information
the store does not use; anything about VSAPcX or the sentence encoders under this accounting (not run); intervals for
the charged-versus-free differences of one memory (point estimates only).

**Least squares' features.** MAY say: giving least squares sign and sign-product random features as well (the store's
nonlinear maps) changed nothing on the four language models and e5 and added 1.1 (bge) and 0.7 (MiniLM) points, all with
two values, where linear features stop at the code's 384 dimensions; least squares still led the store by 4.1 to 6.5
points on every encoder. The store led least squares in 4 of the 84 conditions: three with two values on bge / MiniLM at
1,024 facts and 8-16 bits per fact (closed by sign features), and one on Qwen3-8B with sixteen values at 1,024 facts and 4
bits per fact. MUST NOT say: that least squares' advantage depends on linear features only, or that the store never leads
it in any condition.

**Inference.** MAY say: a stratified t interval over test stores and an exact sign-flip test over the 50 test stores
reproduce the headline labels on the language models; on the sentence encoders the online-write comparison against the
strongest variant (bge, MiniLM) and the two-value bge contrasts are not significant at the store level. MUST NOT say that
every "leads" label is significant with test stores as the unit.

PLAIN SENTENCES FOR THE PAPER (proposals; the authors' wording decides):
1. Written by least squares in the store's own format (same features, value codes, dimension, quantizer, bits and read),
   the store answered 1.0 to 2.4 more queries per hundred over the region on every encoder; least squares with free value
   columns on the same features did no better, so the fixed Hadamard codes cost nothing, and the remaining 1.9 to 4.8
   points of least squares' lead come from its principal-component features.
2. Charging every memory's side information (whitening matrix, principal basis, SDM locations) at 8 bits per number,
   shared by 1,000 to 100,000 stores, least squares still led the store on every language model; the SDM's advantage over
   the first design held only when side information was free.
3. Sign and sign-product random features did not help least squares on the language models and helped only with two
   values on the two 384-dimensional sentence encoders, where the store had led least squares in three of the 84
   conditions.
4. With test stores as the unit (a stratified t interval and an exact sign-flip test over the 50 test stores) the headline
   labels on the language models stand; on the sentence encoders the comparison of the strongest store variant with the
   one-pass online write is not significant.

## Disclosed analysis additions after the test (no arm, choice or bar changed)

The count of label changes over all 140 (C) cases; the attained floor of the sign-flip test at five stores; the dev audit
summary of (A) (quantization cost, share and alpha composition); the MiniLM re-derivation diagnostic (b72_diag_minilm.py).
