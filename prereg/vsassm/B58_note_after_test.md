# B58 note after the test (test run once, on the subset of compute note 5; results/b58_verdict.txt / .json, results/b58_check.txt, results/b58_sample_reading.txt)

Scope actually run: part (a) on the 7 i.i.d. conditions only (damage conditions not run, compute note 4), budgets dev
N* +- 2 grid steps, both variants at k1-k4 and only the dev-better variant at e05 (P-BCH), k8 (P-BCH), k16 (P-LDPC),
k32 (P-LDPC), no extension passes (compute note 5); P-LDPC designs from dev pass 1 only (compute note 3). Part (c)
re-score: the 16 Fig. 5b conditions at windows fixed from pre-B58 results; i.i.d. E-LDPC taken from B38 (OWN = PKG on
i.i.d. inputs). Part (b) is a code audit plus computation. The B58 test, its re-score and every check ran on the local
CPU only; no paper file was edited.

## Part (a), primary: the store protected by the tables' channel codes

Cells per stored pair for >= 99.5% of the 4,776 facts exact (i.i.d. cell errors at the 30-day rates):

| k | uncoded store (B38) | protected store P-best (code) | best table (B38) | Delta H1-P (steps) | bootstrap Delta | bootstrap P-best |
|---|---|---|---|---|---|---|
| k1 (0) | 76 | 91 (P-BCH = P-LDPC) | 19 (BCH) | -9 | [-9, -9] | [91, 91] |
| k2 | 76 | 91 | 19 (BCH) | -9 | [-9, -9] | [91, 91] |
| k4 (0.021) | 76 | 91 | 27 (BCH) | -7 | [-7, -7] | [91, 91] |
| 5% control | 91 | 128 (P-BCH) | 38 (BCH) | -7 | [-7, -6] | [108, 128] |
| k8 (0.123) | 128 | 152 (P-BCH) | 54 (LDPC) | -6 | [-8, -6] | [152, 215] |
| k16 (0.276) | 431 | 1,024 (P-LDPC) | 215 (LDPC) | -9 | [-9, -8] | [861, 1,024] |
| k32 (0.374) | 1,722 | 3,444 (P-LDPC) | 724 (LDPC) | -9 | [-9, -9] | [3,444, 3,444] |

H1-P: **TABLE WIN** at every k; the best table needs 2.8 (k8) to 4.8 times fewer cells than the protected store.
Coding never helps the store (G): one grid step worse at k1, k2, k4 and k8 (the parity it must carry: 91 against 76;
152 against 128), two or more steps worse at 5% (128 vs 91), k16 (1,024 vs 431) and k32 (3,444 vs 1,722). The best
store form stays the uncoded store, so H1-Pany = B38's H1 (TABLE WIN, Delta -4 to -8). At k1-k4, 91 is the smallest
budget at which any code fits over the store's 76 information bits per pair, so it is exact, not a bound.

Reading (registered): protected by the same channel codes as the tables, the store needs no fewer cells than
unprotected, and the tables still need 2.8-4.8 times fewer. The gap is the store's cross-talk: with perfect cells it
already needs 76 cells (information bits) per stored pair, against 19 for the BCH table; a channel code can only add
parity on top of those 76 bits, and at high noise the store's own redundancy (similarity decoding of a random code) is
a cheaper protection than a code on its sign bits. The protected store fails gradually where its code does not
decode (it still reads most keys with a few percent of residual bit errors), so its best designs at k8-k32 are
heavily protected low-rate codes over few information bits.

Caveats, all against the store and stated: the comparison at e05 and k8 uses P-BCH only and at k16 / k32 P-LDPC only
(the other variant not run); P-LDPC designs are octave-resolution; no extension passes (none of the seven N* is
unbracketed: the top of every window passes and every N* above the window bottom has a failing budget below it, except
k1-k4 where nothing fits below 91). The damage conditions of part (a) were not run.

## Part (b): value-vocabulary accounting

Audit: the store's clean-up codes are generated from the vocabulary strings (b25_pcm.codes: Vc = bip(vocab, "value",
d)); every table stores vocabulary indices (b34_inmem.Store.vi, payload_bits; ceil(log2 N) bits) and maps them back
through the same vocabulary; T-vhdd regenerates its value codes from the same strings (hd_code("value:" + vocab[v]));
every memory returns vocabulary indices scored against the same index lists. **The tables' index-to-value dictionary
and the store's codebook source are treated identically: both are the shared, uncounted vocabulary ("both free").**
B38's V-last-vs-table comparison is symmetric. The one asymmetric arm is V-cb, which pays for a codebook that no table
has while the tables' dictionary stays free.

Both charged (vocabulary strings, UTF-8 x 8 bits, on the cells of every memory at the BSC capacity of the condition;
the store regenerates its codebook from the charged strings): pooled over the 50 stores (v = 288 bits per pair, set by
the 36-character UUID values of niah_multikey_3) every k becomes a **tie** by the grid-step rule (Delta' -0.6 to -1.0
steps), because the common vocabulary cost dwarfs both memories; the difference in cells per pair is unchanged (57 at
k1, 74 at k8, 998 at k32). Per family: niah_multikey_2 (v = 56) the table still needs fewer cells at every k (Delta'
-1.6 to -3.5); niah_multikey_3 (v = 288) ties; multivalue and vt in between. V-cb stays far behind in every variant
(Delta' -4.9 to -12.7).

## Part (c): the LDPC decoder discrepancy

**Cause.** B38's registered decoder (ldpc package BpDecoder, product-sum) has no numerical guard in its check update.
When the product of tanh(m / 2) over a check's other edges rounds to +-1, the message log((1 + p) / (1 - p)) is
infinite; the bit sums then meet inf - inf or inf x 0 and become NaN, and the decision "posterior <= 0 -> error" reads
NaN as "no error". It happens when confident, conflicting inputs meet: all copies of a coded bit on one failed line or
stuck cluster give a large wrong log-likelihood ratio (at the 34.5 clip) beside equally large correct ones. B38's
second decoder (b38_check._sp_decode) clips the tanh input at +-40 and the product at 1 - 1e-12. Evidence: on B38's
1,007 check codewords the package's posteriors are non-finite in 173 of the 175 codewords that the second decoder
decodes and the package does not; an emulation of the package's arithmetic, written from our reading of its algorithm
(its C++ source is not installed; disclosed addition), reproduces its output on 1,007 / 1,007 codewords and 3,600 /
3,600 stuck-copy frames, non-finite in the same 348 codewords; with the second decoder's two guards added, the
emulation decodes exactly the 874 codewords the second decoder decodes (identical output on 995 / 1,007; the rest are
non-converged words). The registered mechanism test (channel probabilities floored at 1e-6) restored agreement only
to 818 / 1,007, because the overflow arises inside the iterations, not in the inputs.

**Which is correct (exhaustive ML on K = 16 IRA codes, 72 settings x 400 frames).** With independent errors the two
decoders return the same words in every setting; both fall short of ML where belief propagation is weak (e.g. R =
1/2, eps 0.276, 4 copies: ML 235, both 88). With stuck copies and saturated inputs the guarded decoder is far closer to
ML at the rates the natural-layout tables use (R <= 1/2; R = 1/4, eps 0.123, 16 copies: ML 392, guarded 166, package
54), the package closer at R = 2/3 on the shortest code (N = 24). Neither is within 3 points of ML everywhere, so by
the registered rule the guarded decoder (OWN) is used. Plainly: both implement the same algorithm; the package's
implementation overflows on these inputs; the guarded one is the numerically safe form.

**Re-scored Fig. 5b conditions (E-LDPC with OWN; B38's package numbers in parentheses).** Natural layout, failed lines
k4 / e05 / k8 / k16: 128 / 304 / 3,444 / > 4,096 (all > 4,096); stuck clusters: 128 / 304 / 609 / 2,048 (512 / 724 /
1,024 / 2,435). Interleaved: lines 27 / 45 / 76 / 215, clusters 27 / 38 / 64 / 304, identical to B38. The store
(V-last) needs 76 / 91 / 152 / 512 (lines) and 76 / 91 / 152 / 724 (clusters) in the natural layout. Its lead over the
best non-HD table in the natural layout: lines +3 / +7 / +17 / +13 grid steps, clusters +3 / +7 / +8 / +6 (B38: +5 /
+9 / +17 / +13 and +6 / +10 / +11 / +7): **H4 VSA WIN stands, with the lead at 4% damaged cells cut to 3 grid steps
(1.7 times: 128 vs 76 cells per pair).** Interleaved, the tables keep their lead (H3 TABLE WIN); in the natural layout every
table needs at least two grid steps more than for scattered errors (position independence 0 / 8 for every table).

**The Fig. 5b sentence**, by its registered rule: part (1) holds (store lead >= 2 steps in 8 / 8 natural-layout
conditions) and part (3) holds (interleaved TABLE WIN), but part (2) does not: the interleaved LDPC table is within
one grid step of its scattered-error requirement in 6 of 8 conditions, not >= 7 (lines at k8: 76 vs 54; clusters at
k16: 304 vs 215; two steps each). This was already so with the package's numbers (the interleaved results are
identical). So "until an interleaver gave tables the same tolerance" is too strong as worded; what holds is "an
interleaver brought every table to within two grid steps of its requirement for scattered errors (the repetition, CAM
and random-code tables within one), and interleaved tables needed fewer cells than the store."

## Predictions

Q1 held (TABLE WIN, Delta <= -4 everywhere: -6 to -9). Q2 failed (coding never helps: held; but at k8 coding costs one
step, 152 vs 128, not >= 2). Q3 held (91 at k1, k2, k4). Q4 not evaluable at e05-k32 (one variant run, compute note
5); at k1-k4 the two variants tie (held). Q5, Q6 not evaluated (damage not run). Q7 held (audit symmetric; pooled
both-charged ties at every k; niah_multikey_2 table fewer at k1-k8). Q8 failed (non-finite posteriors in 173 / 175:
held; floor restores agreement: failed, 818 / 1,007). Q9 failed (independent errors: both decoders below ML by more
than 3 points at low copy counts and high error rates; stuck copies: OWN not within 3 points everywhere; PKG >= 10
points below ML at 16-30 copies: held). Q10 failed (lines-nat k4 128 <= 152: held; store lead >= 2 steps in >= 6 / 8:
held; Fig. 5b rule: failed on its interleaved part; i.i.d. E-LDPC not re-scored). Q11 failed in part (second scorer 0
differences, 0 over budget, BCH shortcut 893 / 893: held; numba port 1,581 / 1,988 identical: failed, so it was not
used; the batched decoder used is bit-identical to B38's second decoder, 1,988 / 1,988).

## Checks

Second scorer by plain loops on 10 test cells: 0 differences. Accounting: 2,041 evaluations, 0 over budget, 0
information bits over floor(b n), budgets filled exactly. N* recomputed by a second rule: 0 mismatches over 14
(variant, condition). PKG vs OWN on two non-converging P-LDPC i.i.d. cells: the hard decisions of failed words differ
(20 / 368 identical) but neither decodes any word there and the residual errors are equal (12,938 vs 13,016); on a
converging cell both decode 130 / 130. P-BCH: shortcut confirmed by galois on 893 / 893 words, overloaded words 400 /
400 identical. Sample reading by the AI-assisted process (not independent): results/b58_sample_reading.txt.

## What the paper can and must not say (from these results)

Can say: protected by the tables' own channel codes (BCH or LDPC, same lengths, rates, repetition, closed-form or
development selection), the store needs no fewer cells than unprotected (91 vs 76 per pair at low noise, more at high
noise), and the coded tables still need 2.8-4.8 times fewer; the gap comes from the store's cross-talk (76
information bits per pair even with perfect cells), not from a missing channel code. The vocabulary is shared and
uncounted for every memory; charging it to all of them dilutes the ratio but leaves the difference in cells per pair
unchanged. B38's package LDPC decoder overflows (non-finite messages) under confident conflicting inputs; with a
guarded decoder the natural-layout LDPC table needs 128 cells per pair at 4% failed lines (not > 4,096), and the store
still needs fewer cells than every non-interleaved table under failed lines and stuck clusters (by 1.7 times at 4%).

Must not say: that the store's disadvantage is due to lacking error correction; that the natural-layout LDPC table
"fails" under 4-10% failed lines (B38's > 4,096 was a decoder artefact); that an interleaver gives tables "the same"
tolerance as scattered errors (within two grid steps; within one for LDPC in 6 of 8); anything about the protected
store under localized damage or with multi-level cells (not run); that P-LDPC or P-BCH is the better protection at
e05-k32 (only one was run there).
