# B74 note after the test (test run once; results/b74_verdict.txt, b74_predictions.txt, b74_check.txt, b74_sample_material.txt)

Pre-registration 71499dd (22:40:07) with run and verdict code, OpenTimestamps stamp c47ee65 (22:40), **Bitcoin block
977225** reported at 23:34:27 (upgraded proof 91144cb), 54 minutes after the stamp and before any development or test run.
Development 23:35-23:40 (box: language models; local: sentence encoders); NLMSc draw choices abacb7e (23:35), FX dev
choices d08695a (23:40); tests launched once at 23:41 (local) and 23:42 (box), all 2,434 test jobs written, 0 failures
in the final state (20 local label jobs failed on a missing training-norm input file and were run once after the
unchanged file was placed, compute note 4); compute notes 1-6, each written before the output it precedes was read.
No cut. R1 only (V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024), ideal accounting. Labels: B60's rule on the
one-level 95% interval (entity within store); two-level = test stores, then entities ("2l" where it differs).

## (A) Write rule x features at the store's fixed format (2 x 3 factorial)

Format of every arm of a cell = the final design's (STORE's): its q, its dimension d (principal kind: min(d, n)), B53's
Hadamard / antipodal value codes, B32's quantizer with block scales, matched-filter read. Writes: H = Hebbian sum
(share by B32's rule), L = B72's least-squares write in that format. Features: R = the store's random map (HR = STORE,
LR = OPTR, both existing), P = VSAPc's principal coordinates (d <= n), T = B71's rotated principal coordinates (VSAPR's
prep / map / p). W = write effect, F = feature effect, I = interaction (W_X - W_R), G = corner gap LX - STORE.

**Rotated principal features (corner T, the principal level that fits the store's format), R1, points:**

| encoder | W_R (random) | W_T (rotated principal) | F_T at H | F_T at L | I_T | G_T = LT - STORE | Shapley write / features | registered rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | +1.61 + | +1.30 + | +0.65 + (2l level) | +0.34 | -0.31 | +1.95 + | +1.45 / +0.49 | write larger |
| Granite-3.3-8B | +1.66 + | +1.64 + | +1.92 + | +1.90 + | -0.02 | +3.56 + | +1.65 / +1.91 | features larger |
| Llama-3.1-8B | +1.61 + | +1.47 + | +1.12 + | +0.98 + | -0.14 | +2.59 + | +1.54 / +1.05 | write larger |
| Mistral-7B | +1.04 + | +1.64 + | +1.10 + | +1.70 + | +0.61 | +2.74 + | +1.34 / +1.40 | path-dependent |
| e5-base-v2 | +1.47 + | +3.21 + | +0.55 | +2.29 + | +1.74 + | +3.76 + | +2.34 / +1.42 | path-dependent |
| bge-small | +2.37 + | +2.32 + | +0.34 | +0.29 | -0.05 | +2.66 + | +2.35 / +0.31 | write larger |
| MiniLM-L6 | +1.51 + | +1.34 + | +0.65 + (2l level) | +0.48 | -0.17 | +1.99 + | +1.42 / +0.56 | write larger |

At the store's own format the least-squares write gains 1.3 to 3.2 points on rotated principal features, about what it
gains on the store's random ones (1.0 to 2.4); the features gain 0.3 to 1.9 with either write. The interaction is level
on 6 of 7 encoders (|I_T| <= 0.6; e5 +1.74). So both orders give about the same split, and the write rule is the larger
part on 4 of 7 (Qwen3-8B, Llama, bge, MiniLM), the features on Granite only, path-dependent on Mistral and e5; Shapley
write share >= features share on 5 of 7. Halves: with two values the rule gives write 2 / features 1 / path-dependent
4; with sixteen 3 / 1 / 3.

**Unrotated principal features at the store's q (corner P):**

| encoder | W_P | F_P at H | F_P at L | I_P | G_P = LP - STORE | rule |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | +0.29 | -1.06 - | -2.37 - | -1.31 - | -0.77 - (2l level) | features larger |
| Granite-3.3-8B | +0.25 | -0.20 | -1.61 - | -1.41 - | +0.05 | write larger |
| Llama-3.1-8B | -0.05 | +0.16 | -1.50 - | -1.66 - | +0.11 | path-dependent |
| Mistral-7B | +1.49 + | -0.76 | -0.31 | +0.45 | +0.72 | write larger |
| e5-base-v2 | +0.14 | -6.00 - | -7.33 - | -1.33 - | -5.86 - | features larger |
| bge-small | +1.52 + | -8.29 - | -9.14 - | -0.85 - | -6.77 - | features larger |
| MiniLM-L6 | +0.37 | -7.49 - | -8.63 - | -1.14 - | -7.13 - | features larger |

In the store's format the unrotated principal coordinates HURT both writes over R1 (at sixteen values every encoder:
-2.4 to -16.7; Hadamard residue classes split the principal coordinates, and d <= n leaves budget unused where d > n,
12 of 12 cells on the sentence encoders), and they hurt the least-squares write more (I_P < 0 on 6 of 7). With two
values they help the Hebbian write on the four language models (+2.2 to +3.4) and the least-squares write
(+0.8 to +3.7). The corner gap G_P is about zero or negative:
at this corner there is no least-squares lead to attribute, and its four "features larger" labels are large NEGATIVE
feature terms. (The rule was applied mechanically, as registered.)

**Registered verdict: the statement "the larger part of least squares' lead is its features, not its write rule" is
UNSUPPORTED** ("features larger (identified)" on 1 of 7 at corner T and 4 of 7 at corner P, the registered bar being 6
of 7 at both; at R1-V2 and R1-V16 also below the bar).

**Path decompositions of STORE - LAMc through the rotated corner (R1; points):** path 2 = features at the Hebbian write
(HR - HT) + write at rotated principal features (HT - LT) + rest (LT - LAMc: LAMc's free value columns, own q, p and
whitened principal features against the store's format):

| encoder | STORE - LAMc | path 2: features / write / rest | path 1 (B72-like, in the store's format): write / features / rest |
| --- | --- | --- | --- |
| Qwen3-8B | -4.06 | -0.65 / -1.30 / -2.11 | -1.61 / -0.34 / -2.11 |
| Granite-3.3-8B | -5.55 | -1.92 / -1.64 / -1.99 | -1.66 / -1.90 / -1.99 |
| Llama-3.1-8B | -5.56 | -1.12 / -1.47 / -2.97 | -1.61 / -0.98 / -2.97 |
| Mistral-7B | -5.23 | -1.10 / -1.64 / -2.49 | -1.04 / -1.70 / -2.49 |
| e5-base-v2 | -6.53 | -0.55 / -3.21 / -2.78 | -1.47 / -2.29 / -2.78 |
| bge-small | -4.18 | -0.34 / -2.32 / -1.52 | -2.37 / -0.29 / -1.52 |
| MiniLM-L6 | -4.30 | -0.65 / -1.34 / -2.31 | -1.51 / -0.48 / -2.31 |

LAMc leads the best fixed-format arm LT on 7 of 7 by 1.5 to 3.0 points (also at V = 16 on 7 of 7, by 3.4 to 6.1; at
V = 2 LT is level with or ahead of LAMc on e5, bge and MiniLM). That remainder is the format and LAMc's own options
together; B74 does not split it. B72's "features 1.9 to 4.8" (LSFR -> LAMc) is therefore features AND the least-squares
format they need (free value columns): in the store's own format the principal features add only 0.3 to 1.9.
Secondary: LT - NLMSc over R1 level on Qwen3-8B, e5, bge, MiniLM, LT ahead on Granite (+0.69, 2l level) and Llama
(+1.08), behind on Mistral (-2.05); HT - VSAPcX level on 5 of 7 (VSAPcX ahead on Qwen3-8B -1.40, e5 -0.67).

## (B) Exact-label queries (each stored entity's own written key) against alias queries, same test stores

Recall over R1, label / alias (%):

| arm | Qwen3-8B | Granite | Llama | Mistral | e5 | bge | MiniLM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STORE | 77.6 / 48.8 | 75.9 / 47.5 | 77.3 / 49.0 | 78.4 / 51.3 | 71.2 / 46.4 | 68.5 / 45.1 | 72.6 / 44.3 |
| VSAPcX | 80.7 / 50.9 | 78.7 / 49.6 | 79.1 / 50.5 | 79.6 / 52.6 | 72.0 / 47.6 | 74.1 / 45.8 | 76.1 / 45.0 |
| LAMc | 82.2 / 52.9 | 80.6 / 53.1 | 82.8 / 54.6 | 82.0 / 56.5 | 74.5 / 52.9 | 73.2 / 49.3 | 74.2 / 48.6 |
| NLMSc | 78.5 / 50.9 | 76.4 / 50.4 | 75.6 / 50.5 | 78.7 / 56.1 | 72.9 / 49.9 | 69.7 / 47.1 | 68.2 / 46.5 |

Contrasts over R1 (points; one-level label; Delta = alias contrast - label contrast with its paired-entity interval):

| contrast | query | Qwen3-8B | Granite | Llama | Mistral | e5 | bge | MiniLM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STORE - LAMc | labels | -4.60 - | -4.70 - | -5.51 - | -3.67 - | -3.27 - | -4.65 - | -1.59 - |
| | aliases | -4.06 - | -5.55 - | -5.56 - | -5.23 - | -6.53 - | -4.18 - | -4.30 - |
| | Delta | +0.54 | -0.85 - | -0.06 | -1.57 - | -3.26 - | +0.47 | -2.71 - |
| VSAPcX - LAMc | labels | -1.52 - | -1.92 - | -3.64 - | -2.45 - | -2.53 - | +0.91 + | +1.84 + |
| | aliases | -2.01 - | -3.48 - | -4.02 - | -3.91 - | -5.31 - | -3.47 - | -3.66 - |
| STORE - NLMSc | labels | -0.93 - | -0.46 - (2l level) | +1.62 + | -0.37 level | -1.68 - | -1.12 - | +4.40 + |
| | aliases | -2.11 - | -2.87 - | -1.51 - | -4.79 - | -3.48 - | -2.02 - | -2.16 - |
| | Delta | -1.18 - | -2.41 - | -3.14 - | -4.42 - | -1.80 - | -0.90 - | -6.55 - |
| VSAPcX - NLMSc | labels | +2.15 + | +2.32 + | +3.49 + | +0.84 + | -0.94 - | +4.44 + | +7.82 + |
| | aliases | -0.06 level | -0.80 - (2l level) | +0.03 level | -3.47 - | -2.26 - | -1.32 - | -1.52 - |

The four compared memories recall their written labels far better than unseen aliases (by 21.6 to 31.1 points over R1;
VSAPc on the sentence encoders by 12 to 15, from a low label recall): most of every memory's loss on aliases is generalisation from the label's code to the alias's code, not storage. **Least squares'
(LAMc's) lead over the store is present on the written labels themselves** on all 7 encoders (1.6 to 5.5 points over
R1), on the language models about as large as on aliases (Delta -1.6 to +0.5): on the language models it is a storage
advantage (less cross-talk among stored keys). By halves it differs: with two values LAMc recalls 98.6-99.1% of the
labels on the language models against the store's 90.2-92.6% (label gap 6.0-9.0, larger than the alias gap 3.9-5.5),
while on the sentence encoders the store recalls labels BETTER than LAMc (+0.6, +4.3, +9.6) and LAMc's alias lead there
is generalisation; with sixteen values LAMc's label lead is smaller than its alias lead on the language models and
e5 (more than half of it generalisation) and larger on bge and MiniLM. **The one-pass delta rule's (NLMSc's) lead over
the store is mostly not storage:** on labels the store is ahead of NLMSc on Llama and MiniLM and level on Mistral (and
on Granite with test stores resampled), and NLMSc's alias lead exceeds its label lead on 7 of 7 (Delta -0.9 to -6.6, all intervals below 0); against
the strongest store variant NLMSc is behind on labels on 6 of 7 (+0.8 to +7.8) and level or ahead on aliases.
Generalisation loss (label - alias recall) is about the same for LAMc and the store (LAMc smaller on 5 of 7).

Alias reproduction (the label wrapper must leave every alias prediction unchanged): identical in 100% of the 55,359
alias predictions per arm and encoder, except e5 STORE (19 differ, one cell, M = 256, V = 2, 8 bits per fact, a sign
feature map), MiniLM STORE (6, M = 256, V = 2, 16 bits per fact) and **MiniLM VSAPc (1,365 = 2.5%, all in the three
M = 1,024, V = 16 cells: centre / principal sign features at d = n = 384, whose near-zero-variance eigenvectors are
numerically arbitrary, as diagnosed in B72)**. In those three MiniLM cells VSAPcX = VSAS, so no VSAPcX number is
affected; MiniLM's VSAPc label recall in those cells comes from a slightly different basis than B60's run. The other
differences are sign flips of near-zero projections when the query matrix has more rows (the stored vector does not
depend on the queries).

## (C) The one-pass delta rule re-chosen from 72-configuration sub-grids (20 draws of 72 of NLMSc's 480)

| encoder | STORE - NLMSc72: median [min, max]; < 0; NLMSc72 leads | full grid | VSAPcX - NLMSc72: median [min, max]; < 0; NLMSc72 leads | full grid | NLMSc72 - NLMSc (median) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | -1.76 [-2.47, +1.49]; 19/20; 17/20 | -2.11 | +0.29 [-0.43, +3.54]; 4/20; 0/20 | -0.06 | -0.35 |
| Granite-3.3-8B | -3.00 [-3.43, -0.16]; 20/20; 18/20 | -2.87 | -0.93 [-1.36, +1.91]; 14/20; 13/20 | -0.80 | +0.14 |
| Llama-3.1-8B | -1.23 [-2.44, +1.19]; 19/20; 13/20 | -1.51 | +0.32 [-0.89, +2.74]; 7/20; 3/20 | +0.03 | -0.29 |
| Mistral-7B | -3.84 [-5.10, -0.43]; 20/20; 19/20 | -4.79 | -2.52 [-3.78, +0.89]; 19/20; 19/20 | -3.47 | -0.95 |
| e5-base-v2 | -2.73 [-3.53, -1.91]; 20/20; 20/20 | -3.48 | -1.50 [-2.30, -0.69]; 20/20; 20/20 | -2.26 | -0.76 |
| bge-small | -1.39 [-2.29, +0.11]; 18/20; 15/20 | -2.02 | -0.69 [-1.58, +0.81]; 15/20; 10/20 | -1.32 | -0.63 |
| MiniLM-L6 | -1.58 [-2.46, -0.62]; 20/20; 19/20 | -2.16 | -0.94 [-1.82, +0.02]; 19/20; 14/20 | -1.52 | -0.57 |

(R1; "NLMSc72 leads" = one-level interval of the draw's contrast below 0.) Choosing NLMSc from 72 configurations, as
many as the store searches, costs it a median -0.95 to +0.14 points over R1. **STORE - NLMSc: by the registered reading
the unequal grid is "not material"** (the full-grid sign in 19, 20, 19, 20 of 20 draws on the four language models,
medians within 0.35, 0.13, 0.28, 0.95 points of the full-grid value); NLMSc72 leads with its interval in 13-19 of 20
draws on the language models and 15-20 on the sentence encoders. **VSAPcX - NLMSc is not "not material"**: the
strongest store variant and the matched-grid one-pass write stay level on Qwen3-8B and Llama (NLMSc72 leads in 0 and 3
of 20 draws), Granite splits (13 of 20), and NLMSc72 leads on Mistral (19), e5 (20), MiniLM (14) and bge (10).

## Predictions

Held: PA2, PA4, PA5, PB1, PC1, PC3. Failed: PA1 (I predicted the least-squares write would gain more on principal than
on random features; it gains about the same on rotated ones, W_T > W_R on 2 of 7, I_T above 0 on 1, and less on
unrotated ones, I_P below 0 on 6), PA3 in part (the statement is UNSUPPORTED as predicted, but corner P shows "features
larger" on 4 of 7, not <= 2, because the P features are large negative terms), PA6 (HP dev rows equal B60's in 1,479 of
1,512; the 33 others are all MiniLM M = 1,024, V = 16 centre / sign principal rows), PB2 (Delta > 0 for STORE - LAMc on
2 of 7 and for STORE - NLMSc on 0 of 7: I expected least squares to lead more on labels; for NLMSc it is the reverse),
PB3 (LAMc's generalisation loss larger on 2 of 7, not >= 5), PC2 (NLMSc72 - NLMSc median +0.14 on Granite, outside the
predicted -1.5 to -0.2), PX in part (see checks). Held: 6 of 13; failed or failed in part: 7.

## Checks (results/b74_check.txt)

Cut at 00:53 by the coordinator's new deadline (no new runs; unfinished check processes stopped by PID):
- **Plain-loop scorer** (all 7 encoders; every FX cell accuracy, every (A) combination incl. both path sums, every (B)
  label contrast and paired value, every (C) per-draw contrast): 1,722 values, max |difference| 2.1e-17; path checks
  (W_X - W_R = I_X, F_X|L - F_X|H = I_X, both paths = G_X) max 2.8e-17. PASS.
- **Bits audit**: completed for the three sentence encoders only (12,168 FX rows and 4,650 NL72 rows: 0 over budget, 0 not
  maximal, 0 rule / formula mismatches); the all-encoder run was stopped before it finished: **the language-model rows
  are NOT audited**.
- **Float64 re-derivation (independent code)**: sentence encoders complete (one test store per (M, V) at 4 and 16 bits
  per fact; HP, LP, HT, LT, STORE label and LAMc label predictions): 76 of 84 >= 0.99; the 8 below the bar are all
  sign-feature stores (sgn / sgn3 / psgn: e5 HT 0.988; MiniLM HT 0.966 - the same store B71 flagged; MiniLM LT 0.988 /
  0.990; MiniLM HP / LP at M = 1,024, V = 16, centre psgn 0.963-0.967, the near-null-eigenvector cells) with logged and
  re-derived accuracies within 0.6 points; for the MiniLM LT store the stored vector agrees to 1e-14 with 0 sign flips and
  1 of 4.2 million key sign features differs, so the remaining disagreement is in reading near-tied integer scores.
  Language models: PARTIAL - Qwen3-8B 32 and Granite 32 re-derived comparisons, Llama 4, Mistral 4, all agreement 1.000;
  the rest not done (cut).
- **Reproduction**: HP dev rows equal B60's VSAPc dev rows in 1,479 of 1,512 (the 33 others: MiniLM M = 1,024, V = 16,
  centre / sign principal, max 0.5 points); HT test predictions equal B71's VSAPR test predictions on 705 of 705 stores
  where the configurations coincide; label wrapper alias reproduction as in (B).
- **Determinism re-run**: 56 of 56 sample jobs (first test job of every (encoder, job type); sentence encoders locally with
  3 workers, language models on the box with 16) identical.
PX therefore FAILED in part (bits audit and re-derivation incomplete for the language models; 8 sentence-encoder
re-derivations below 0.99).

## Sample reading by the AI-assisted process (results/b74_sample_material.txt)

Cells with the largest |I_T| (as registered): Qwen3-8B M = 256, V = 16, 8 bits per fact (I_T = -2.42) and e5-base-v2
M = 256, V = 16, 8 bits per fact (+6.47). In both the cell's interaction is the small difference of two large sets of
queries whose answers change both ways (Qwen3-8B 599 queries with +1 and 718 with -1 of about 4,600; e5 1,023 and 735),
not a few queries or a scoring artefact: gold values and the four answers were checked on the 20 listed queries, and the
aliases are ordinary variants of the stored label (article dropped, "Republic of Kenya" / "Kenya", a German "Hermann
II." for the landgrave, initials, transliterations) plus a few hard ones ("ht" for Haiti, "Alexander III" for Alexander
the Great). The storage errors (label queries the store misses and LAMc answers; 1,418 and 1,219 in these cells) are
dominated by short, generic or common titles ("Spy", "Vacation", "Worst", "Thin Ice", "Cleaning Up", "The White Line"),
whose codes lie close to many other stored keys, so the Hebbian read suffers cross-talk that the least-squares write
removes; no leak (labels are the written keys by design, gold = the written value) and no answer-format artefact.

## Literature (supporting / conflicting / new; recalled, not re-fetched)

Supporting: that sequential decompositions of a difference depend on the order of the steps whenever the factors
interact, and that the average over orders (the Shapley value) is the order-free split, is the "index number problem"
of the Oaxaca-Blinder decomposition (Oaxaca 1973; Blinder 1973) and Shorrocks' Shapley decomposition (2013, Journal of
Economic Inequality); here the interaction at the rotated corner is small, so the two orders nearly agree. That least
squares (the pseudo-inverse / projection memory) removes cross-talk among the stored keys and so recalls the written
patterns better than the correlation (Hebbian) memory (Kohonen 1972; Personnaz, Guyon and Dreyfus 1986) is what the
label queries show on the language models; that recall of the stored patterns and generalisation to perturbed keys are
different properties of an associative memory is classical (Kohonen 1989; the basins of attraction literature after
Hopfield 1982). That tuning budgets must be matched before comparing methods (Melis, Dyer and Blunsom 2018; Dodge et al.
2019, "Show your work") is the premise of (C). Conflicting: none found. New here: at equal stored bits on frozen
language-model codes and within the store's exact format, the write rule (1.3-3.2 points on rotated principal
features) is at least as large as the principal-feature gain (0.3-1.9) and the two barely interact; the one-pass delta
rule's lead over the store is a generalisation effect, not a storage one.

## What the paper may and must not say

**Attribution of least squares' lead (write rule versus features).** MAY say: in the store's own format (its dimension,
value codes, quantizer, bits per number and read), the least-squares write answered 1.0 to 2.4 more queries per hundred
than the Hebbian write on the store's random features and 1.3 to 3.2 more on rotated principal-component features;
those features added 0.3 to 1.9 points under either write; the interaction was within +-0.6 points on six of seven
encoders (e5: +1.7), so the order of the two steps hardly matters and the write rule is the larger part on four of seven
encoders (Shapley shares: write 1.3 to 2.4, features 0.3 to 1.9); the rest of least squares' lead (1.5 to 3.0 points)
comes with its own format and options (free value columns, its q and p, whitened principal features) and is not split
further. Principal coordinates used without rotation in the store's format hurt both writes. MUST NOT say: that the
larger part of least squares' lead is its features (B72's sentence; unsupported by the registered rule), or the B72
decomposition "write 1.0-2.4 + features 1.9-4.8" without saying that its features term also contains least squares'
format; that the write-rule effect depends on the features (no material interaction except e5); that the fixed-format
least-squares store matches LAMc (LAMc leads it on all seven).

**Label versus alias queries.** MAY say: the store, its strongest variant and both least-squares memories recall their written labels 22 to 31
points better than unseen aliases, so most error on aliases is generalisation; least squares (batch) leads the store on the written labels on all
seven encoders (1.6 to 5.5 points), on the language models about as much as on aliases, i.e. a storage advantage (less
cross-talk); with two values on the sentence encoders the store recalls labels better than least squares and least
squares' alias advantage there is generalisation; the one-pass delta rule's lead over the store is mostly
generalisation: on the written labels the store is level with or ahead of it on three of seven encoders (four with
test stores resampled), and the strongest store variant is ahead of it on six of seven. MUST NOT say: that least squares wins only by generalisation
or only by storage on every encoder and value count (it differs by halves); that the one-pass online write stores the
facts better than the store.

**The one-pass write on a matched grid.** MAY say: re-chosen from 72 configurations, as many as the store searches (20
random sub-grids of its 480), the one-pass delta rule still led the store over the region in 19 to 20 of 20 draws on
every language model (medians 1.2 to 3.8 points; 18 to 20 of 20 on the sentence encoders), so the unequal grid is not
material for that comparison; against the strongest store variant the matched-grid one-pass write stayed level on
Qwen3-8B and Llama-3.1-8B (it led in 0 and 3 of 20 draws), led in 13 of 20 on Granite and in 19 of 20 on Mistral, and
in 10 to 20 of 20 on the sentence encoders. MUST NOT say: that the one-pass write beats the strongest store variant on
every encoder, or that its lead over the store depends on its larger grid.

PLAIN SENTENCES FOR THE PAPER (proposals; the authors' wording decides):
1. In the store's own format, the least-squares write gained 1.0 to 2.4 points on random and 1.3 to 3.2 points on
   rotated principal-component features, and those features gained 0.3 to 1.9 points under either write; because the
   two effects barely interact, the write rule is at least as large a part of least squares' lead as the features, and
   the remaining 1.5 to 3.0 points come with least squares' own format.
2. Queried with the written labels themselves, least squares still led the store on every encoder (by 1.6 to 5.5
   points), as much as on aliases for the language models: its advantage is mainly less cross-talk among stored facts.
   The one-pass online write's advantage is not: on the labels the store matched or beat it on three of seven encoders,
   and the strongest store variant beat it on six of seven.
3. Re-chosen from 72 configurations, as many as the store searches, the one-pass online write still led the store in at
   least 18 of 20 random sub-grids on every encoder.

## Disclosed after the test (no arm, choice or bar changed)

The cell-level diagnosis of the alias-reproduction differences and of the re-derivation misses; the path decomposition
of STORE - LAMc through the rotated corner (a sum of registered contrasts); the halves of (B) summarised in words;
b74_check.py: the path check rewritten as registered in compute note 1 and the LAMc label re-derivation's preps cached
per encoder (speed only); four oversubscribed check processes (16 BLAS threads each) stopped by PID at 00:48 and
restarted with 3 threads.
