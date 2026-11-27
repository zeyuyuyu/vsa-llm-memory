# B64 note after the test (written after the verdict; results/b64_verdict.txt, b64_check.txt, b64_predictions.txt)

Date: 2026-11-23, about 03:40 (machine clock). Test run once (03:27:46-03:32), no cut; verdict, checks and sample
reading after it. The paper is not edited by B64.

## Result (ideal accounting, primary; R1 = V 2 / 16 x 4-16 bits per fact x M 256 / 1,024; 95% cluster bootstrap)

| encoder | first - SDMt (R1) | first - best of 7 (R1) | V2 / V16 | final - best of 7 (R1) | V2 / V16 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | -1.89 [-2.78, -1.00] memory leads | -2.15 [-3.05, -1.28] memory leads, bar loss | -1.42 level / -2.89 memory | +1.25 [+0.39, +2.10] store leads, bar tie | +3.72 store / -1.22 memory |
| Granite-3.3-8B | -2.25 [-3.09, -1.42] memory leads | -2.25 [-3.05, -1.41] memory leads, bar loss | -1.00 level / -3.51 memory | -0.22 [-1.05, +0.62] level, bar tie | +2.06 store / -2.50 memory |
| Llama-3.1-8B | -1.20 [-2.12, -0.26] memory leads | -1.20 [-2.13, -0.27] memory leads, bar tie | -0.68 level / -1.72 memory | +0.91 [+0.04, +1.75] store leads, bar tie | +3.57 store / -1.75 memory |
| Mistral-7B | -3.56 [-4.50, -2.61] memory leads | -3.56 [-4.51, -2.61] memory leads, bar loss | -3.15 memory / -3.96 memory | -1.02 [-1.95, -0.12] memory leads, bar tie | -0.10 level / -1.93 memory |
| e5-base-v2 | -0.92 [-1.75, -0.11] memory leads | -1.79 [-2.63, -0.96] memory leads, bar tie | -1.51 memory / -2.07 memory | -1.51 [-2.34, -0.67] memory leads, bar tie | -1.30 memory / -1.72 memory |
| bge-small | -4.73 [-5.64, -3.83] memory leads | -5.01 [-5.90, -4.12] memory leads, bar loss | -4.69 memory / -5.33 memory | -3.73 [-4.66, -2.81] memory leads, bar loss | -2.36 memory / -5.10 memory |
| MiniLM-L6 | -3.80 [-4.68, -2.91] memory leads | -4.06 [-4.94, -3.19] memory leads, bar loss | -4.82 memory / -3.29 memory | -3.43 [-4.22, -2.63] memory leads, bar loss | -2.62 memory / -4.24 memory |

- SDMt (SDM whose hard locations are the training split's label / alias codes) minus B36's SDM (random locations):
  +11.8 to +13.9 on the language models, +8.5 to +10.4 on the sentence encoders. Region means: SDMt 47.3 / 47.8 /
  48.1 / 52.3 (language models) against the store's first design 45.4 / 45.5 / 46.9 / 48.8 and final design 48.8 /
  47.5 / 49.0 / 51.3; the exact index 81-86. (The clean room's SDMt reached 43.7 / 45.5 on Qwen3-8B / Llama; B64's
  larger grid, mainly a = 2 and labels-only locations, is 2.6-3.6 points stronger.)
- The best of seven is SDMt in 10-12 of the 12 R1 cells of every encoder; it is 8.3-10.7 points above B36's best of
  six on the language models.
- **B36's pre-specified bar (store - best >= +2 with the interval above 0) passes on 0 of 7 encoders for both designs.**
  First design: loss on 5 (Qwen3-8B, Granite, Mistral, bge, MiniLM), tie on 2 (Llama, e5); "memory leads" on all 7.
  Final design: tie on 5 (all four language models, e5), loss on 2 (bge, MiniLM); "store leads" on Qwen3-8B and Llama,
  "level" on Granite, "memory leads" on Mistral and the three sentence encoders. By V: with two values the final
  design leads on three language models (+2.1 to +3.7) and is level on Mistral; with sixteen values SDMt leads on all
  seven encoders (final design -1.2 to -5.1).
- Real accounting (secondary; fixed-width counters): the order reverses at V = 16, where SDMt's one-hot counters cost
  their full width (first design - best of seven +3.7 to +4.1 on the language models, bar passes on all 4; final
  design +6.0 to +7.6). B36's reading, fixed in advance, applies: a win that holds only under real accounting rests on
  how the competitor is stored, and no win is claimed from it. It may be reported as a sensitivity.
- Continuity: first design - best of six reproduces B49 exactly (+6.77, +6.06, +7.10, +7.10, +0.60, -0.71, -2.12).

## Predictions

Held: P1 (SDMt - SDM >= +5 on every language model: +11.8 to +13.9), P6 (sentence encoders: first design behind on all
3; final design passes on none), P7 (best of seven = SDMt in 11 / 12 / 12 / 12 cells), P9 (continuity exact), P10
(checks). Failed: P2 (first - SDMt was below the predicted -1 to +3 on all four language models, -1.2 to -3.6), P3
(the store led on none, not on at least 2), P4 (final design: store leads on 2, not 4; bar on 0, not 3), P5 (final
design at V = 2 below +4 on all four; at V = 16 Granite -2.5, outside -2 to +4), P8 (location set 'LA' chosen in 27 of 84
cells, labels-only in 49; already failed on the dev choices, compute note 2). I underestimated SDMt: the clean room's
smaller grid was my anchor.

## Checks (results/b64_check.txt)

Plain-loop scorer: 1,218 cell accuracies and region statistics, max |difference| 4.9e-17. Bits: 383,124 logged dev and
test rows, 0 over budget. Disjointness: 0 string overlaps, 0 locations with raw cosine >= 0.9999 to any dev / test code
on every encoder and set (max 0.956-0.999); 'Rio' / 'Río' (identical codes under accent-stripping sentence encoders)
was removed before the pre-registration. Float64 re-derivation by independent code on 56 test stores (one per encoder,
M, V at 4 and 16 bits per fact): agreement 1.0000 on every store, N identical on 56 of 56, bits identical on 55 (one
differs by 0.6 of 15,526 bits, a float32 / float64 near-tie in one activation; both within budget). Sample reading by
the AI-assisted process: results/b64_sample_reading.txt (SDMt at a = 1-2 is a count sketch whose cells are the
training labels' Voronoi cells; it wins when the alias falls into its label's cell, loses at cell boundaries; no leak).

## What the paper may say, must change, and must drop

1. **"The store beat unsupervised classical memories on language-model codes (+6.1 to +7.1)" (Results 2.2, Fig. 4a,
   Introduction "about 6 to 7 points more alias queries ... than unsupervised classical memories", abstract "the
   store beat key-free memories on language-model codes"): MUST GO.** The registered reading for "memory leads" on a
   language model applies on all four: the claim goes for the first design, and the paper must say that an SDM whose
   hard locations are the training split's codes (a classical variant: data-placed hard locations, Ratitch & Precup
   2004) beat the store's first design over the region on every encoder (by 1.2 to 3.6 points on the language
   models, 0.9 to 4.7 on the sentence encoders). The +6.1 to +7.1 may appear only as the margin over B36's six
   memories with random-location SDM, explicitly superseded. Sentence encoders: "-2.1 to +0.6" becomes -1.8 to -5.0
   against the best of seven.
2. **The final design** may be stated as: against the best of seven key-free memories it was +1.25 (Qwen3-8B) and
   +0.91 (Llama) ahead, level on Granite and 1.0 behind on Mistral, below the pre-specified +2-point bar on all seven
   encoders; ahead with two values on three language models (+2.1 to +3.7) and behind with sixteen values on all
   seven (-1.2 to -5.1). It must not be presented as rescuing the claim: it is a different store, and the bar fails.
3. **Table 1, superposition row:** "against key-free memories +6.1 to +7.1 (language models)" is replaced by "against
   the best of seven key-free memories (with an SDM addressed by training codes) first design -1.2 to -3.6, final
   design -1.0 to +1.3 (language models; pre-specified bar met on none of seven encoders)". The 'Unique to superposed
   memories?' cell stays "mostly" but must name SDMt beside least squares (SDMt sums the value counts of the facts
   that reach each location, so it is a superposed memory, though at the chosen a = 1-2 barely: each cell holds a
   handful of facts).
4. **Per-ingredient verdict for superposition** ("(ii) stored similar keys compactly at a few bits per fact against
   key-free memories, but a decorrelating least-squares write beat the Hebbian write on every encoder") **must
   change**: the first half no longer holds. Suggested content (the authors' wording): at a few bits per fact the
   Hebbian store did not beat the best key-free memory by the pre-specified margin on any encoder; an SDM with
   data-placed locations beat its first design everywhere and matched its final design, and least squares beat both.
   B36's pre-fixed reading: a superposed memory with a different feature map (here data-placed hard locations) does
   at least as well as the dense MAP store; "no bits on keys" stands as the mechanism of every winning memory, the
   VSA-specific claim does not. Section 2.2's closing "Superposition stored similar keys compactly, as capacity theory
   predicts" and Discussion's "a table spends bits on keys or on signatures that an alias does not reproduce" must be
   narrowed: random LSH / PQ signatures are not reproduced by aliases, but cells placed at training codes often are.
5. **Unchanged:** the store's own numbers; B60's least-squares comparisons (LAMc beat the store on every encoder);
   "no memory recalled most facts" (SDMt's region means 47-52% against the exact index's 76-86%). Not tested by B64:
   whether SDMt changes the best of ten (B49 / B60: least squares chosen in 68 of 84 cells), R0 / RH / R2 (V = 1,024),
   the 90%-of-exact budgets; these sentences must say "best of ten without SDMt" if kept.

## Literature (checked after the result)

Support: Ratitch & Precup (ECML 2004, "Sparse distributed memories for on-line value-based reinforcement learning")
allocate SDM hard locations from the observed data instead of Kanerva's uniform placement; SDMt is that classical
variant with a fixed training set, so it belongs among the "unsupervised classical memories" the paper compares
against. Also consistent: count sketches over data-dependent cells (vector quantization codebooks) are standard
key-free structures. No conflict found. (Search: https://www.aaai.org/Papers/Workshops/2004/WS-04-08/WS04-08-015.pdf,
https://www.stat.ubc.ca/~bouchard/pub/report_NSERC.pdf; the ECML paper itself was not opened.)

## Limitations

R1 only (4-16 bits per fact, V 2 / 16); one random map for the store (the store's map SDs 0.1-1.3 points in B60);
SDMt's grid has 504 configurations against the store's 72 (dev selection favours the larger grid; the test is clean);
in 34 of the 84 chosen ideal cells SDMt uses the whole location set below the budget, so a larger location set would
make it stronger, not weaker; the sentence-encoder 'Rio' / 'Río' location was removed by code equality.

## Compute

Local CPU only (16 cores; 8 workers, nice 10): dev 03:15-03:27 (126 jobs), test 03:27:46-03:32 (350 jobs),
verdict and checks to 03:36. The A800 box was not used (load 20-58, /data0 full). No GPU, no paid API.
