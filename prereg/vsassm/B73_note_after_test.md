# B73 note after the test (confirmatory run on fresh data, every configuration frozen; results/b73_verdict.{txt,json}, b73_part_*.json, b73_check_*.json, b73_predictions.txt, b73_sample_reading*.txt; equal cells: prereg/B73_cells_note_after_test.md)

## What was run

Equal bits: pre-registered b8246b4 (20:21:58 machine clock; OpenTimestamps proof 02c0d07, 20:22:00; the header's
"about 20:25" was written ahead), before any fresh entity was fetched. Fetch 20:24-20:27 (after a transport change,
compute note 1), fresh entity file committed before any encoding (e5bc808, 20:28:18). Verdict and check code committed
before any test output existed (2b4c33e 20:33:57, 72e6d30, 1ab4015). Sentence encoders encoded and tested once on this
machine (20:30-20:36; compute note 2), language models encoded on four idle A800 GPUs and tested once on the box
(21:30-21:40; compute notes 4-5): every registered arm on every encoder, 0 failures, **no cut**. The best of the six
unsupervised classical memories was not run (registered beforehand). Equal cells: registered separately (8deea4e /
9c2e552), run once 19:30-19:59, note df77702. **External (OSF) registration: pending the human author.**

Deviations (all disclosed in compute notes 1-5; none changes a configuration, store, statistic or bar): the fetch's
transport (python urllib refused by the endpoint, curl used; same queries); the sentence encoders' test ran on this
machine instead of the box (link stalled); the entity file reached the box in verified pieces; the box chain's
post-test steps were run in parallel (same commands); compute note 5's header time is ahead of its commit (21:40:51).

## Data source and leakage check

Fresh set: one random-sample SPARQL query per class to the public Wikidata endpoint (people, films, literary works,
albums, TV series, singles, cities, football clubs, businesses), items with an English Wikipedia article, an English
label and at least one English alias; PopQA's subjects and every item or alias whose normalized form is a PopQA label
or alias removed; per-class quotas by hash order; B32's ambiguity filters and hash split unchanged. Result: 8,422
entities (train 2,500 / dev 1,736 / test 4,186), 12,744 aliases, 21,166 strings (PopQA: 9,278 entities, 9,014 aliases,
18,292 strings). **Leakage check: 0 of 8,422 URIs, 0 of 21,166 strings exactly and 0 after normalization occur in PopQA's
entity file or raw subjects** (12,244 PopQA URIs, 20,471 normalized PopQA strings). Values: random, V = 2 and 16, new
seed tag ('B73store'); test stores 20 (M = 256) and 5 (M = 1,024) per V, as the paper. The dev split was never used.

## Frozen-configuration rule

Every memory used, in every R1 cell, exactly the configuration chosen on PopQA dev in the paper's runs
(results/b73_frozen.json, committed with the pre-registration, with the choice files' MD5s): STORE = B53's choice,
VSAR = B49's, VSAS = B56's STAR arm, VSAPC = B60's PSTAR arm, VSAPCX = per cell the PopQA-dev-better of VSAS and VSAPC
(the source per cell is fixed), LAMC = B60's, NLMSC = B62's, SDMT = B64's. The fresh training split supplied the side
information (centring / whitening, principal bases, LMMSE statistics, SDM locations), as PopQA's did. Sizes (p, d, N)
follow each design's own budget rule, so no design is missing at any R1 cell. Audit: every recorded configuration
equals the frozen table (0 mismatches in 7,350 test rows).

## The confirmatory table, claim by claim (R1 = V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024; ideal accounting)

Points of accuracy; fresh value with the one-level 95% interval (entity within store, the paper's registered label) and
the two-level label (stores, then entities); paper = PopQA. Margin: within +-2.0 points of the paper's value or within a
factor of two (registered). Decision on the one-level R1 label.

**C1. Least squares (LAMc) ahead of the final design (STORE) on every encoder** — STORE - LAMc. **CONFIRMED** (label 7 / 7,
margin 7 / 7; two-level 7 / 7).

| encoder | paper | fresh R1 [one-level] | label (two-level) | fresh R1-V2 / R1-V16 |
| --- | --- | --- | --- | --- |
| Qwen3-8B | -4.06 | -4.38 [-4.78, -3.99] | LAMc (LAMc) | -4.60 / -4.17 |
| Granite-3.3-8B | -5.55 | -5.85 [-6.27, -5.44] | LAMc (LAMc) | -4.53 / -7.18 |
| Llama-3.1-8B | -5.56 | -6.26 [-6.69, -5.82] | LAMc (LAMc) | -5.55 / -6.97 |
| Mistral-7B | -5.23 | -6.37 [-6.77, -5.96] | LAMc (LAMc) | -6.76 / -5.99 |
| e5-base-v2 | -6.53 | -4.78 [-5.17, -4.39] | LAMc (LAMc) | -1.96 / -7.59 |
| bge-small | -4.18 | -4.65 [-5.06, -4.24] | LAMc (LAMc) | -0.89 / -8.41 |
| MiniLM-L6 | -4.30 | -3.46 [-3.87, -3.04] | LAMc (LAMc) | **+1.30** / -8.22 |

**C2. LAMc ahead of the strongest variant (VSAPcX) on every encoder** — VSAPcX - LAMc. **CONFIRMED** (7 / 7, margin 7 / 7,
two-level 7 / 7).

| encoder | paper | fresh R1 [one-level] | label (two-level) | fresh R1-V2 / R1-V16 |
| --- | --- | --- | --- | --- |
| Qwen3-8B | -2.01 | -2.41 [-2.77, -2.05] | LAMc (LAMc) | -1.68 / -3.13 |
| Granite-3.3-8B | -3.48 | -3.66 [-4.03, -3.29] | LAMc (LAMc) | -1.89 / -5.43 |
| Llama-3.1-8B | -4.02 | -4.50 [-4.90, -4.08] | LAMc (LAMc) | -2.81 / -6.18 |
| Mistral-7B | -3.91 | -4.92 [-5.30, -4.56] | LAMc (LAMc) | -2.86 / -6.98 |
| e5-base-v2 | -5.31 | -4.13 [-4.48, -3.77] | LAMc (LAMc) | -0.85 / -7.41 |
| bge-small | -3.47 | -3.41 [-3.83, -2.99] | LAMc (LAMc) | +0.25 / -7.07 |
| MiniLM-L6 | -3.66 | -2.69 [-3.11, -2.27] | LAMc (LAMc) | **+2.27** / -7.65 |

**C3. One-pass online least squares (NLMSc) ahead of the final design on every encoder** — STORE - NLMSc. **CONFIRMED**
(7 / 7, margin 7 / 7, two-level 7 / 7).

| encoder | paper | fresh R1 [one-level] | label (two-level) | fresh R1-V2 / R1-V16 |
| --- | --- | --- | --- | --- |
| Qwen3-8B | -2.11 | -2.21 [-2.60, -1.81] | NLMSc (NLMSc) | -2.62 / -1.80 |
| Granite-3.3-8B | -2.87 | -3.77 [-4.19, -3.35] | NLMSc (NLMSc) | -1.75 / -5.80 |
| Llama-3.1-8B | -1.51 | -2.37 [-2.78, -1.96] | NLMSc (NLMSc) | -3.69 / -1.05 |
| Mistral-7B | -4.79 | -5.11 [-5.51, -4.70] | NLMSc (NLMSc) | -4.58 / -5.64 |
| e5-base-v2 | -3.48 | -3.53 [-3.92, -3.13] | NLMSc (NLMSc) | -0.48 / -6.57 |
| bge-small | -2.02 | -3.48 [-3.90, -3.08] | NLMSc (NLMSc) | +0.03 / -6.99 |
| MiniLM-L6 | -2.16 | -1.89 [-2.32, -1.47] | NLMSc (NLMSc) | **+2.38** / -6.17 |

**C4. Strongest variant vs NLMSc** — VSAPcX - NLMSc. **CONFIRMED by the registered criterion** (VSAPcX leads on no
encoder; label equal to the paper's on 6 of 7; margin 7 / 7; two-level 6 / 7), **with one label change: Llama-3.1-8B**.

| encoder | paper | fresh R1 [one-level] | fresh label (two-level) | paper label |
| --- | --- | --- | --- | --- |
| Qwen3-8B | -0.06 | -0.24 [-0.60, +0.14] | level (level) | level |
| Granite-3.3-8B | -0.80 | -1.58 [-1.96, -1.20] | NLMSc (NLMSc) | NLMSc (two-level: level) |
| Llama-3.1-8B | +0.03 | -0.61 [-1.00, -0.23] | **NLMSc (NLMSc)** | level |
| Mistral-7B | -3.47 | -3.66 [-4.03, -3.28] | NLMSc (NLMSc) | NLMSc |
| e5-base-v2 | -2.26 | -2.88 [-3.25, -2.51] | NLMSc (NLMSc) | NLMSc |
| bge-small | -1.32 | -2.24 [-2.66, -1.82] | NLMSc (NLMSc) | NLMSc |
| MiniLM-L6 | -1.52 | -1.13 [-1.57, -0.69] | NLMSc (NLMSc) | NLMSc |

Read together: on fresh entities every headline equal-bit sign and label over R1 reappears, with magnitudes inside the
margin on every encoder (fresh ranges: LAMc over the final design 3.5 to 6.4 points, paper 4.1 to 6.5; over the
strongest variant 2.4 to 4.9, paper 2.0 to 5.3; NLMSc over the final design 1.9 to 5.1, paper 1.5 to 4.8). The one
label that moved is the weakest one: NLMSc against the strongest variant on Llama-3.1-8B, level on PopQA (+0.03), 0.6
points ahead on the fresh set (both intervals below zero). The sixteen-value halves carry every lead; with two values the
sentence encoders' gaps are small or reversed (MiniLM-L6: the store ahead of LAMc by 1.3, of NLMSc by 2.4, VSAPcX ahead
of LAMc by 2.3; on PopQA MiniLM's two-value half was level against LAMc, -0.19, and against NLMSc, +1.08).

**Secondary (no claim).** VSAS - LAMc: LAMc leads on 7 / 7 (-3.0 to -6.2). NLMSc - LAMc: LAMc leads on 7 / 7 (-1.2 to
-3.9). STORE - VSAR: the final design ahead of the first on 7 / 7 (+1.1 to +3.5). VSAPC - LAMc: -4.9 to -13.4 (the
principal-feature store's sixteen-value loss, -8 to -25 points at R1-V16, as on PopQA). **VSAR - SDMt (B64's claim that
the SDM at training codes leads the first design over R1 on all seven): NOT reproduced in full** — SDMt leads on 5 of 7
(Granite by 0.5, level two-level; Mistral 1.8; e5 2.3; bge 6.4; MiniLM 5.4), level on Llama-3.1-8B (-0.34 [-0.81,
+0.12]), and the first design LEADS on Qwen3-8B (+1.81 [+1.32, +2.29]; PopQA -1.89). Mean R1 accuracies on the fresh set
are 2 to 7 points above PopQA's for every arm with a PopQA value (STORE, LAMc, NLMSc), so the fresh aliases are somewhat
easier; the differences between memories are what replicate.

## Device table (equal cells; prereg/B73_cells_note_after_test.md, results/b73_cells_verdict.txt)

Cells per stored pair for >= 99.5% exact reads, 50 fresh stores from an independent generator (dictionary words,
capital-letter codes, hexadecimal and six-digit strings; 92,094 pairs, 4,779 probed keys; 0 strings shared with RULER),
frozen designs; fresh [paper], x1 / x8 / x16: dense store 76 [76] / 128 [128] / 431 [431]; sparse-block store 32 [32] /
91 [91] / **362 [304]**; best static table 19 [19] / 54 [54] / 215 [215]; best insertable table 32 [32] / 91 [91] / 304
[304]. Claims: static tables vs dense store 4.0 / 2.4 / 2.0 times fewer cells (TABLE WIN) **CONFIRMED**; vs sparse-block
store 1.68 at every level (TABLE WIN) **CONFIRMED**; insertable table vs sparse-block store 0 / 0 / -1 grid steps (TIE)
**CONFIRMED**; insertable vs dense 2.4 / 1.4 / 1.4 (TABLE WIN) **CONFIRMED**; ordering static < sparse-block < dense,
insertable < dense **CONFIRMED**. 20 of 21 requirements identical to the RULER stores'; the 21st (sparse-block at x16) is
one grid step higher.

## Predictions (results/b73_predictions.txt; cells: in its own note)

Held: Q1 (C1 confirmed), Q2 (C2 confirmed), Q3 (C3: 7 of 7, including Llama, my "likeliest exception"), Q5 (two-level =
one-level labels in 28 of 28 claim x encoder cases), Q6 (SDMt ahead of the first design on 5 of 7, the bar's floor), Q7
(fresh mean R1 accuracy within 10 points of PopQA for all 21 arm x encoder values; largest +6.8, Qwen3-8B LAMc), Q8
(checks below). **Failed: Q4** — I predicted C4 not confirmed (at least two label changes); one changed (Llama), and
the registered criterion allows one. Equal cells: all five predictions held.

## Checks

- Plain-loop scorer (from the units' raw files, plain Python loops; second code path): 231 region statistics (7
  encoders x 11 contrasts x 3 regions), max |difference| from the verdict 1.4e-17 (sentence encoders, local) and 1.0e-17
  (language models, box). After the panel step crashed on the box (compute note 6) the scorer compares against the
  per-encoder part files, which are the verdict's only input.
- Bits audit (own re-implementation of each size rule): 7,350 test rows (every arm, cell and store), 0 over budget,
  0 not maximal; frozen-configuration audit 0 mismatches.
- Leakage: 0 fresh URIs, 0 exact and 0 normalized strings in PopQA (results/b73_leak.json).
- Determinism: 17 units re-run into separate directories (9 sentence-encoder units locally, 8 language-model units on
  the box; every arm), 53 output files compared, predictions identical in 53.
- Sample reading by the AI-assisted process (results/b73_sample_reading.txt; Qwen3-8B and e5-base-v2, the R1-V16 cell
  with the lowest STORE - LAMc): no scoring artefact; the two memories differ by which near-copy aliases each reads,
  least squares reading more of them (1,735 vs 1,012 and 1,300 vs 531 queries one-sided right), as on PopQA.

## Compute

GPU: four encodings on idle A800 GPUs 1, 4, 5, 6 (1.95 + 1.77 + 1.87 + 2.25 minutes): **about 0.13 GPU-hours** (0.14 with
the 20 s idle re-checks). CPU: language-model test 4.8 CPU-hours (184 units, 48 workers on the box, 6.5 minutes wall);
sentence encoders' encoding about 0.2 and test 0.22 CPU-hours locally; verdict bootstraps and checks about 1 CPU-hour;
equal cells 3.9 CPU-hours (+0.5 for checks) locally. On the box at most 48 of my worker processes ran at a time (the
test runner); after it had finished, the determinism re-run (8), the four verdict parts, the bits audit and the sample
reading ran together (14).

## Literature (recalled, not re-fetched)

Supporting: that a least-squares (pseudo-inverse / projection) associative memory stores more than a correlation
(Hebbian) memory of the same size, and that one LMS pass recovers part of the gap, is the classical picture (Kohonen
1972; Personnaz, Guyon and Dreyfus 1986; Widrow-Hoff); B73 adds that the measured equal-bit gaps on frozen language-model
codes are stable to a change of entity set at fixed configurations. Confirmatory re-runs on fresh data with frozen
analysis choices are the remedy recommended against the garden of forking paths (Gelman and Loken 2013; Nosek et al.
2018, preregistration). Conflicting: none found for the equal-bit claims; the SDM claim's partial non-replication is
consistent with SDM performance depending on how well the hard locations cover the address distribution (Kanerva 1988),
which changes with the training set. New here: a frozen-configuration replication of the paper's ten-memory equal-bit
ranking on a second entity set.

## What the paper may and must not say (plain sentences; the authors' wording decides; the paper is not edited by B73)

May say:
1. In a confirmatory run on 8,422 fresh Wikidata entities that share no entity, label or alias with PopQA, with every
   configuration frozen at its PopQA development choice and new random values, least squares solved over all facts
   again answered more queries than the store's final design on all seven encoders, by 3.5 to 6.4 points over the
   region (PopQA: 4.1 to 6.5), and more than the strongest store variant on all seven, by 2.4 to 4.9 (PopQA: 2.0 to 5.3).
2. Written one fact at a time by a single pass of the normalised delta rule, least squares again led the final design on
   all seven encoders, by 1.9 to 5.1 points (PopQA: 1.5 to 4.8).
3. Against the strongest store variant the one-pass write was level on Qwen3-8B and ahead on the other six encoders on
   the fresh set, by 0.6 to 3.7 points; on PopQA it was level on Qwen3-8B and Llama-3.1-8B.
4. Every one of these signs and labels held with test stores resampled as well as entities.
5. On 50 fresh key-value stores from an independent generator, with every device design frozen, tables with a static
   directory again needed 4.0, 2.4 and 2.0 times fewer cells than the dense store at one, eight and sixteen times the
   30-day error rate, and 1.7 times fewer than the sparse-block store; the insertable table needed as many cells as the
   sparse-block store (one grid step fewer at sixteen times) — 20 of 21 requirements were identical to the RULER
   stores'.
6. These confirmatory runs were pre-registered internally (git commit and OpenTimestamps proof before any fresh data
   were fetched or generated); an external registration is pending.

Must not say:
1. That the one-pass delta rule is level with the strongest store variant on Llama-3.1-8B without adding that on fresh
   entities it led by 0.6 points (interval below zero).
2. That the SDM at training-split locations leads the first design on every encoder: on the fresh set it led on five of
   seven, was level on Llama-3.1-8B, and the first design led on Qwen3-8B (+1.8).
3. That least squares leads with two values on every encoder: on the fresh set the store's final design led LAMc by 1.3
   and NLMSc by 2.4 points on MiniLM-L6 with two values (the leads over the region come from sixteen values).
4. That the fresh numbers are on the same accuracy scale as PopQA's: every memory scored 2 to 7 points higher on the
   fresh aliases; only the differences replicate.
5. That B73 re-tested regions beyond R1 (R0, RH, R2), the real accounting, the map variance, the best of the six
   unsupervised classical memories (not run), or the device conditions other than x1 / x8 / x16 (see the cells note).
6. That B73 is externally registered (OSF is pending the author), or that the fresh set was chosen independently of the
   paper's authors: its selection rule was written by the same AI-assisted process, before the data were seen.

## Disclosed after the test

The box's panel step crashed when writing its JSON (a numpy integer in the claim counts) after the per-encoder part
files had been written; the counts were cast to int and the panel re-run locally from the same part files, and the
plain-loop scorer was pointed at the part files (compute note 6). No statistic, label or rule changed.
