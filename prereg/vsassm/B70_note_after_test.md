# B70 note after the test (test run once; results/b70_verdict.txt, b70_predictions.txt, b70_check.txt, b70_sample_reading.txt)

Pre-registration d9f66ef (16:55, OpenTimestamps f8844cb); dev 16:58-17:10; dev choices 892f409 (17:13); test once
17:13-17:20 (NLMSc on the box, SDMt locally); compute notes 1-4. One registered cut: SDMt and BEST7 on the three
sentence encoders (their 16-bit dev rows did not reproduce B64's; notes 2-3). Ideal accounting throughout. The paper is
not edited by B70.

## What was already run above 16 bits per fact

At 32 and 64 bits per fact (V 2 / 16, both M, all 7 encoders) every memory of B36 / B49 / B53 / B56 / B57 / B60 had dev and
test runs: the six classical families, learned hashing, B49's LAM, the store's first and final designs and its read
variants (VSAS, VSAP, VSAPc, hence VSAPcX), LAMc, the exact index. Only at 4-16: B62's NLMSc / RLSc, B64's SDMt, B66's
arms. Above 64: nothing. B70 adds NLMSc and SDMt (language models) at 32 / 64 where R2 needs them.

## R2 (fixed before any run, from committed dev outputs)

Per setting (encoder, M, V), every grid budget from the first at which the dev-best of 13 memories reaches 0.9 x the exact
index's dev accuracy. 23 cells in 15 of 28 settings: 21 with V = 2 (M = 256 at 8-64 bits per fact, M = 1,024 at 32-64), 2
with V = 16 (bge-small and MiniLM-L6, M = 256, 64 bits per fact). The dev-best memory at the first useful budget was
learned hashing (a per-fact index on a map trained on alias-label pairs) in 12 settings, the unsupervised per-fact index in
1, least squares in 2. 12 of the 14 settings with sixteen values have no useful budget within 64 bits per fact.

## Primary results over R2 (per encoder; 95% cluster bootstrap; B60's rule / B36's bar)

| encoder | cells | store - least squares | store - delta rule | best store variant - least squares | store - best of six | store - best of seven |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 2 | -1.86 [-2.72, -1.01] LS leads | -1.01 [-1.82, -0.20] DR leads | -1.35 [-1.96, -0.74] LS leads | +2.67 [+1.15, +4.15] passes | +2.59 [+0.91, +4.31] passes |
| Granite-3.3-8B | 5 | -2.20 [-3.08, -1.34] LS leads | -1.72 [-2.56, -0.88] DR leads | -0.11 [-0.66, +0.44] level | +6.46 [+5.26, +7.68] passes | +0.69 [-0.45, +1.85] tie |
| Llama-3.1-8B | 1 | +0.43 [-1.65, +2.73] level | +0.23 [-1.69, +2.45] level | +0.16 [-1.78, +2.37] level | +3.56 [+1.63, +5.46] passes | +3.70 [+1.68, +5.67] passes |
| Mistral-7B | 4 | -4.18 [-5.31, -3.14] LS leads | -2.69 [-3.48, -1.90] DR leads | -2.20 [-3.17, -1.30] LS leads | +4.21 [+2.60, +5.79] passes | -1.84 [-3.25, -0.49] tie (best of seven ahead by the rule) |
| e5-base-v2 | 2 | +2.53 [+0.97, +4.13] store leads | +5.16 [+3.57, +6.84] store leads | +2.25 [+0.63, +3.93] store leads | -7.98 [-9.63, -6.40] loss | cut |
| bge-small | 5 | +1.79 [+0.66, +2.91] store leads | +3.37 [+2.29, +4.46] store leads | +1.77 [+0.63, +2.90] store leads | -7.50 [-8.51, -6.50] loss | cut |
| MiniLM-L6 | 4 | +0.94 [-0.21, +2.11] level | +3.33 [+2.24, +4.45] store leads | +1.57 [+0.40, +2.74] store leads | -9.60 [-10.63, -8.56] loss | cut |
| 4 language models | 12 | -2.59 [-3.14, -2.04] LS leads | -1.76 [-2.24, -1.26] DR leads | -0.99 [-1.43, -0.55] LS leads | +4.84 [+4.05, +5.62] passes | +0.41 [-0.33, +1.14] tie |
| 3 sentence encoders | 11 | +1.61 [+0.90, +2.34] store leads | +3.68 [+2.98, +4.38] store leads | +1.78 [+1.04, +2.52] store leads | -8.35 [-9.01, -7.71] loss | cut |

Store = B53's final design; least squares = B60's LAMc; delta rule = B62's NLMSc (one pass); best store variant = VSAPcX;
best of six = B36's classical families (mostly the per-fact index there); best of seven = six + SDM at training codes.
Rescaled for chance ((a - b) / (1 - 1/V) per cell), every B60-rule label is the same except MiniLM-L6's store - least
squares (+3.80 [+1.64, +6.07], store leads; level in raw points).

Against the best of ten (with least squares and learned hashing; mostly learned hashing at R2) the store trailed on 6 of 7
encoders (by 4.7 to 7.0 points on Qwen3-8B, Granite and Mistral, 10.0 to 11.0 on the sentence encoders) and was level on
Llama-3.1-8B (-1.62 [-3.70, +0.40], one cell). The best of ten led least squares by 2.5 to 4.3 on three language models
(Llama +2.06, level) and by 11.9 to 12.8 on the sentence encoders. On the language models the store's first design and SDMt were
level over R2 (-0.43 [-1.20, +0.35]; Qwen3-8B first design ahead, Mistral SDMt ahead).

## Region means (R2; raw / rescaled for chance / fraction of the exact index)

| encoder | exact | store | best store variant | least squares | delta rule | best of six | best of ten |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 88.4 / 76.8 / 1 | 76.1 / 52.1 / 0.86 | 76.6 / 53.2 / 0.87 | 77.9 / 55.9 / 0.88 | 77.1 / 54.2 / 0.87 | 73.4 / 46.8 / 0.83 | 82.2 / 64.4 / 0.93 |
| Granite-3.3-8B | 87.9 / 75.8 / 1 | 74.5 / 49.0 / 0.85 | 76.6 / 53.2 / 0.87 | 76.7 / 53.4 / 0.87 | 76.2 / 52.4 / 0.87 | 68.0 / 36.1 / 0.78 | 79.2 / 58.5 / 0.90 |
| Llama-3.1-8B | 89.4 / 78.8 / 1 | 80.4 / 60.7 / 0.90 | 80.1 / 60.2 / 0.90 | 79.9 / 59.9 / 0.89 | 80.1 / 60.3 / 0.90 | 76.8 / 53.6 / 0.86 | 82.0 / 64.0 / 0.92 |
| Mistral-7B | 90.8 / 81.7 / 1 | 78.1 / 56.1 / 0.86 | 80.0 / 60.1 / 0.88 | 82.2 / 64.5 / 0.91 | 80.7 / 61.5 / 0.89 | 73.9 / 47.7 / 0.81 | 85.1 / 70.2 / 0.94 |
| e5-base-v2 | 86.1 / 72.3 / 1 | 72.5 / 45.1 / 0.84 | 72.3 / 44.5 / 0.84 | 70.0 / 40.0 / 0.81 | 67.4 / 34.8 / 0.78 | 80.5 / 61.0 / 0.94 | 82.5 / 65.1 / 0.96 |
| bge-small | 82.6 / 70.0 / 1 | 66.3 / 41.3 / 0.80 | 66.3 / 41.8 / 0.80 | 64.5 / 36.5 / 0.78 | 63.0 / 33.5 / 0.76 | 73.8 / 54.6 / 0.89 | 77.3 / 60.5 / 0.94 |
| MiniLM-L6 | 80.1 / 66.9 / 1 | 65.3 / 42.5 / 0.81 | 65.9 / 43.2 / 0.82 | 64.3 / 38.7 / 0.80 | 62.0 / 34.3 / 0.77 | 74.9 / 58.1 / 0.93 | 76.3 / 60.4 / 0.95 |

Raw region means at R2 are 74-80 (language models) and 65-73 (sentence encoders) for the store, against 44-57 over R1:
these are budgets where memories are useful, but the store is still at 80-90% of the exact index. Cells reaching 90% of
the exact index on test (of 23): store 0, best store variant 4, least squares 4, delta rule 5, best of six 8, best of
ten 20.

## The 13 settings without a useful budget within 64 bits per fact

12 of the 14 settings with sixteen values and Llama-3.1-8B at M = 1,024 with two values. The dev-best memory at 64 bits
per fact (learned hashing in 11, least squares in 2) stood 1.0-9.5 points below 90% of the exact index. B49's
unbounded-width reads (test) of both the store and least squares are below that mark in 10 of the 13 (all M = 1,024
settings and Qwen3-8B / e5 at M = 256): there no budget makes either superposed memory useful. Only at M = 256 with
sixteen values on Granite, Llama and Mistral could a larger budget (beyond 64 bits per fact, never run) make them useful.

## Predictions

Held: Q2, Q3, Q4, Q7, Q8, Q9. Failed (in part): Q1 (least squares led on 3, not 4, language models; Llama level), Q5
(best-of-seven labels matched best-of-six on 2 of 4 language models: SDMt is the best of seven at M = 256 and closes the
gap on Granite and Mistral), Q6 (best of ten ahead on 6, not 7), Q10 (SDMt dev reproduction failed on the sentence
encoders). results/b70_predictions.txt.

## Checks

Plain-loop scorer: 719 cell accuracies and point statistics (new arms also re-read from the raw job files), max |difference|
1.4e-16 (pass). Bits audit: 168,694 B70 dev and test rows, 0 over budget, 0 NLMSc not maximal; 1,260 logged sizes of the
store, LAMc, NLMSc and SDMt at the R2 test records, 0 over budget (pass). Float64 re-derivation by independent code (own
preps and eigenbasis, rp blocks from the registered seeds, a genuine one-fact-at-a-time NLMS loop, own quantizer and read;
SDMt by b64_check's own float64 search, counters, entropy and N scan) on store 0 of every R2 setting with a new budget at
its largest R2 budget: NLMSc 15 stores, agreement 1.0000 on every one; SDMt 7 stores (language models), agreement
1.0000, N identical 7 of 7 (pass at 0.99). Dev-phase reproduction (registered): NLMSc 0 mismatches on all 7 encoders; SDMt 0 on
the 4 language models, failed on the 3 sentence encoders (cut; notes 2-3). Sample reading by the AI-assisted process
(results/b70_sample_reading.txt): no scoring artefact or leak; the queries only least squares reads are near-copies the
store's cross-talk flips; the queries only the best of ten reads are renamings and transliterations that a per-fact
index on a learned map finds and both superposed reads miss.

## Departures and disclosures

- SDMt's reproduction failure is a property of B64's implementation on float32 codes: its query-side nearest-location
  search depends on the set of budgets computed in one call and on the machine's BLAS (near-tie ranks); B64's own budget
  set reproduces its rows. SDMt / BEST7 on the sentence encoders therefore not reported (registered stop rule and cut).
- The R2 test numbers of the store, its variants, least squares, the six and the best of ten are re-aggregations of test
  predictions that existed before B70 (their 32-64-bit aggregates were public in B60's RH rows); only NLMSc and SDMt at
  32-64 bits per fact are new measurements. R2 itself was chosen from dev data only.
- Post hoc reading, not a registered statistic: on the sentence encoders at M = 1,024 least squares is a linear function
  of a 384- or 768-dimensional code (its dev-chosen maps are linear), close to or beyond the capacity of a linear read for
  1,024 binary values (Cover 1965: about 2n random dichotomies for n inputs), while the store's dev choice there is a
  nonlinear sign-product map with 65,280 coordinates. e5 at M = 1,024: store 72.0, least squares 64.0; bge 71.3 vs 58.4;
  MiniLM 71.5 vs 61.8. The reversal on the sentence encoders is concentrated there (at M = 256 least squares leads on
  e5 and bge cells). B66's least squares on the store's own features (LSFR) was not run at 32-64 bits per fact, so
  whether this is the write rule or the feature map is not separated at R2.

## Literature (recalled, not re-fetched)

Supporting: a linear threshold read of n inputs separates about 2n random dichotomies (Cover 1965), so a linear
associative memory over n-dimensional keys saturates once the number of stored binary facts approaches 2n (Kohonen
1972's optimal linear associative memory is bounded by the key dimension); the store's nonlinear feature maps are a
random-features expansion that lifts that bound (Rahimi and Recht 2007). Per-fact indexes over learned binary codes
outperforming unsupervised ones on similarity search is the standard result of supervised hashing (LDAHash, Strecha et
al. 2012; CCA-ITQ, Gong et al. 2013). That a superposed (holographic / VSA) memory needs dimension growing with the item
count and with the target accuracy while a table stores per item (Plate 2003; Frady, Kleyko and Sommer 2018) is
consistent with the per-fact index being best where accuracy is high. No conflicting result recalled. What is ours: the
comparison at equal stored bits restricted to the budgets where the best memory is useful, on frozen language-model and
sentence-encoder codes.

## What the paper may say (two or three plain sentences)

PAPER CAN SAY: "Where the best memory reached 90% of an exact index's accuracy (23 conditions, 21 of them with two values,
at 8 to 64 bits per fact), the best memory was a per-fact index, usually on a map learned from alias-label pairs, and the
VSA store reached 80 to 90% of the exact index and never 90%. There least squares still led the store on three of the four
language models (by 1.9 to 4.2 points; level on Llama-3.1-8B), but the store led least squares on two sentence encoders
(by 1.8 and 2.5; level on MiniLM-L6) and the one-pass delta rule on all three, and it beat the best unsupervised
classical memory on all four language models (by 2.7 to 6.5) while trailing it by 7.5 to 9.6 on the sentence encoders.
With sixteen values almost no setting had a useful budget within 64 bits per fact, and in most of them neither
superposed memory reaches that mark at any width."

So the equal-bit conclusions hold only in part where memories are useful: "least squares beat the store on every
encoder" becomes "on three of the four language models"; "no margin over the best classical memory" does not hold on
the language-model codes at these budgets (the store passes B36's bar against the best of six on all four; against the
best of seven, with the SDM at training codes, on two of four); and on the sentence encoders the store is ahead of both
least-squares memories but far behind a per-fact index.

PAPER MUST NOT SAY: that the R1 conclusions carry over to useful budgets unqualified; that least squares beats the store
"on every encoder" at useful budgets; that the store is useful (reaches 90% of an exact index) anywhere in the registered
grid; anything about sixteen values at useful budgets beyond the two sentence-encoder cells; any SDMt result on the
sentence encoders at 32-64 bits per fact.

## Limitations

R2 is mostly V = 2 and mostly 64 bits per fact; with V = 16 it has two cells (sentence encoders). Llama-3.1-8B has one R2
cell, Qwen3-8B and e5 two, so their intervals are wide. Budgets above 64 bits per fact were not run for any memory. SDMt
missing on the sentence encoders (cut). Registered maps only (no map replicates at R2). Ideal accounting only.
