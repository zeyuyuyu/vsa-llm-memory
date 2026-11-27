# B62 note after the test (test run once; results/b62_verdict.txt, b62_predictions.txt, b62_check.txt, b62_sample_reading.txt)

Pre-registration ae3c72f (02:28:19, OpenTimestamps 1b73db8); dev 02:31-02:44; dev choices e5db971 (02:47:35); test once
02:48:23-03:08:03 on the box (640 / 640 jobs); no cut (compute notes 1-3). R1 only, ideal accounting, 7 encoders.
STORE = B53's final design (Hebbian write, matched filter; existing test predictions); LAMc = B60's batch ridge memory;
NLMSc = one pass of the normalised delta rule with LAMc's corrected features / targets / scales; RLSc = recursive least
squares with LAMc's options (delta fixed from training-split feature norms before the first fact; forgetting on dev).

## Primary: STORE - NLMSc (95% cluster bootstrap, 10,000 draws; "online LS leads" iff upper bound < 0)

| encoder | R1-V2 | R1-V16 | R1 |
| --- | --- | --- | --- |
| Qwen3-8B | -2.77 [-3.64, -1.91] online | -1.44 [-2.46, -0.43] online | -2.11 [-2.78, -1.45] online |
| Granite-3.3-8B | -1.73 [-2.66, -0.82] online | -4.01 [-5.04, -2.98] online | -2.87 [-3.56, -2.18] online |
| Llama-3.1-8B | -3.30 [-4.32, -2.25] online | +0.27 [-0.75, +1.28] level | -1.51 [-2.23, -0.80] online |
| Mistral-7B | -4.76 [-5.71, -3.84] online | -4.82 [-6.10, -3.54] online | -4.79 [-5.59, -4.01] online |
| e5-base-v2 | -1.55 [-2.33, -0.78] online | -5.41 [-6.39, -4.43] online | -3.48 [-4.11, -2.87] online |
| bge-small | +1.04 [+0.10, +1.98] STORE leads | -5.09 [-6.06, -4.14] online | -2.02 [-2.70, -1.35] online |
| MiniLM-L6 | +1.08 [-0.08, +2.22] level | -5.39 [-6.26, -4.51] online | -2.16 [-2.87, -1.43] online |

Over the region the one-pass online write beats the Hebbian write on all 7 encoders (by 1.5 to 4.8 points). Exceptions
inside the halves: with two values the store leads on bge-small and is level on MiniLM-L6; with sixteen values Llama
is level.

## Secondary

- **STORE - RLSc**: online LS leads over R1 on 7 of 7 (-3.60 to -6.47); R1-V2 6 of 7 (MiniLM level, -0.46); R1-V16 7 of 7.
  **RLSc - LAMc** over R1 -0.49 to +0.17 (LAMc ahead on Qwen3-8B, Granite, Llama by 0.44-0.49; level on the other four);
  the STORE - RLSc labels equal B60's STORE - LAMc labels in 21 of 21. RLS written one fact at a time gives the batch
  memory to within half a point.
- **RLS working state**: at the dev-chosen p, the p x p matrix (32-bit, symmetric) exceeds the whole budget in 84 of 84
  R1 cells (median 580 times the budget, range 61 to 65,000). **RLScS** (state charged, p cut to fit: p median 14,
  range 4-31): the store leads on 7 of 7 over R1, by 11.5 to 18.5 points. Charged for its working state, RLS is not a
  usable memory at these budgets.
- **NLMSc - LAMc** over R1: LAMc leads on 6 of 7 (-1.95 to -4.05), level on Mistral (-0.45 [-1.18, +0.25]); R1-V2 -0.71 to
  -2.36 (LAMc leads on 7); R1-V16 -0.18 (Mistral, level) to -6.99 (Llama). The online write recovers about half of the
  batch memory's lead over the store on most encoders (Mistral nearly all, Llama about a quarter).
- **NLMSc - RLSc**: RLSc leads over R1 on 6 of 7, level on Mistral.
- Dev choices: NLMSc whiten pca on 70 of 84 R1 cells; step 0.25-0.5 at V = 2, 0.03-0.25 at V = 16; last iterate 75 of 84
  (Polyak average 9). RLSc beta = 1 in 53 of 84 (0.999 in 28).

## Write cost, read cost, working state (median R1 cell per encoder; multiply-adds from the chosen shapes)

- Write, excluding feature maps, total for the M facts: STORE 1.6-2.6e6; NLMSc 1.2-5.9e6 (1.1-2.1 times the store's);
  LAMc batch (cheaper exact form) 5.4e7-2.1e8 (22-170 times; 79-170 on the language models); RLSc 1.1e8-1.9e9 (35-2,100
  times). Per fact: STORE 2.1-4.1e3, NLMSc 3.2e3-1.0e4, LAMc 1.2-4.3e5, RLSc 3.0e5-3.1e6.
- Feature maps: the store projects each code to d (about the budget in numbers) dimensions, the least-squares arms to
  p = d / c (and at most n for principal features). Including the feature maps, the store's write costs 9-15 times
  NLMSc's and its read per query 9-16 times (read: STORE 8.4e5-1.7e7, NLMSc / LAMc / RLSc 1.5e5-4.1e6).
- Working state during the write, in numbers: STORE about 2d (4.3e3-8.1e3); NLMSc p c + p (1.8e3-6.2e3); LAMc M p + M^2 +
  p c (7.4e5-1.2e6); RLSc p (p + 1) / 2 + p c + p (7.5e4-5.2e5, up to 8.4e6). At 32 bits per number all of them exceed the
  stored budget (STORE 62-64 times, NLMSc 21-34, LAMc 2,800-8,900, RLSc 230-8,300); only RLS's grows as p^2.

## Predictions

Held: Q1 (NLMSc leads over R1 on 7 of 7, store on 0), Q2 (V = 2: 4 of 4 LLMs; 1 of 3 sentence encoders), Q3 (V = 16: 6 of
7), Q5 (|RLSc - LAMc| <= 0.5 on 7 of 7; labels 21 of 21), Q8 (RLS state over budget 84 of 84; store leads RLScS on 7),
Q10 (checks). Failed: Q4 (NLMSc - LAMc at V = 2 inside [-1.5, +0.5] on 5 of 7, not 6: Granite -2.36, bge -2.09; at V =
16 Llama -6.99 is beyond the predicted 1-5); Q6 (beta = 1 in 53 of 84, not >= 75; known from the committed dev choices,
compute note 2); Q7 (mu = 1 in 6 of 42 V = 2 cells, not half; known from the dev choices; the other two parts held);
Q9 (RLSc's write >= 100 times the store's fails on the three sentence encoders, 35-70 times; the NLMSc and LAMc parts
held).

## Checks

Plain-loop scorer 567 values, max |difference| 5.6e-17. Bits: 837,774 dev and test rows, 0 over budget, 0 not maximal
(RLScS with its state term). Float64 re-derivation by independent code with genuine one-fact-at-a-time NLMS loops and
RLS recursions (own preps, eigenbasis, rp blocks from the registered seeds, training norms, quantizer, read) on 72 test
cells: agreement 1.0000 on all; the primal loop equals the dual form to 2.4e-15 and the recursion the closed form to
6.6e-13 (relative). Sample reading by the AI-assisted process: no scoring artefact; the store's losses are near-copies
read through cross-talk at V = 16; on Qwen3-8B 6 of 10 store-only successes are also read by the batch solve (losses of
the online write itself). Disclosed analysis only (b62_pos.py): on e5 at V = 16 the one-pass write reads late-written
facts better (29.5 / 32.8 / 35.6 by thirds of the write order), the recency of a constant-step online write with fewer
features than facts.

## Literature (supporting / conflicting / new; read before writing this note)

Supporting, not new: the delta rule (Widrow-Hoff LMS) and its normalised form are the classic online route to the
least-squares (Kohonen's optimal) linear associative memory, and iterative local rules reach the projection rule's
capacity where the Hebbian rule does not (Personnaz et al. 1986; Diederich and Opper 1987; recalled, not re-fetched).
In fast-weight memories, replacing the purely additive (Hebbian) outer-product write with a delta-rule write improves
retrieval capacity (Schlag, Irie and Schmidhuber, ICML 2021, arXiv 2102.11174), and DeltaNet computes the sequential
delta rule exactly in parallel through a compact representation of the update products (Yang et al., NeurIPS 2024,
arXiv 2406.06484), the same exactness that B62's forward-substitution form uses. One pass of NLMS with step 1 is a
Kaczmarz sweep. No conflicting result found. What is ours: the comparison at equal stored bits, with entropy-coded
quantized weights, on frozen language-model and sentence-encoder codes, against a tuned VSA store.

## The paper's claims (current text; the paper is not edited by B62)

- "a decorrelating least-squares write beat the Hebbian write on every encoder": **SURVIVES, and may now say "also
  when written one fact at a time"** (registered reading: NLMSc leads over R1 on all 7). Numbers: one pass of the delta
  rule beat the store by 1.5 to 4.8 points over the region (batch solve 4.1 to 6.5). Must name the exceptions if halves
  are quoted: with two values the store led the online write on bge-small (+1.0) and was level on MiniLM-L6; with
  sixteen values Llama was level.
- Abstract "an offline least-squares memory, also a sum, beat it on every encoder [B62 pending]": by the registered
  reading the sentence SURVIVES without the batch qualifier (an online sum also beats it over the region on every
  encoder); "offline" is no longer needed and is narrower than the evidence. Proposal for the authors (the abstract is
  theirs, not changed here): "... but a least-squares memory, also a sum, beat it on every encoder, even when written
  one fact at a time." The margin is smaller online (1.5-4.8 against 4.1-6.5 points).
- Results, "recursive least squares gives the same weights one fact at a time but keeps p^2 working numbers that the
  budget does not charge": **SURVIVES, quantified**: within 0.5 points of the batch memory on all seven; its working
  matrix exceeds the whole budget in every condition (median 580 times); charged for it, RLS trails the store by 11.5
  to 18.5 points.
- Results, "a one-pass delta rule on the corrected features [B62 pending]": fill with "beat the store over the region
  on all seven encoders by 1.5 to 4.8 points, at 1.1 to 2.1 times the Hebbian write's multiply-adds and a working state
  of its own weights, and trailed the batch solve by 0.5 to 4.1 (level on Mistral-7B)". Table row "online write [B62
  pending]": "one-pass delta rule -1.5 to -4.8 on all seven".
- Results, "a batch solve of about 100 times the Hebbian write's multiply-adds (2.7 to 4.5e8 against 1.6 to 2.6e6)":
  **MUST BE RESTATED** (those numbers are B57's cost model on B49's defective LAM shapes, dual form only). With LAMc's
  chosen shapes and the cheaper exact solve: 22 to 170 times (median 5.4e7 to 2.1e8 against 1.6 to 2.6e6; 79 to 170 on
  the language models).
- Supplementary (B57 text): "Written online by one pass of the normalised delta rule, the least-squares memory lost 0.9
  to 2.8 points against ..." and "the delta rule 1.4 to 5.4e6": **SUPERSEDED** by B62 (against the corrected batch
  memory it lost 0.5 to 4.1 over the region, LAMc ahead on six, level on Mistral; write 1.2 to 5.9e6). "only their
  two-value comparisons stand" no longer applies to the delta rule.
- Supplementary, "The Hebbian sum is thus the cheapest write but not the most accurate one, and a delta rule of similar
  cost read better with two values": **MUST CHANGE**. Excluding the feature maps the Hebbian sum is the cheapest write
  (the delta rule 1.1-2.1 times); including them the store's write costs 9-15 times the delta rule's and its read 9-16
  times, because it projects every code to about as many dimensions as the budget has numbers. "Read better with two
  values" becomes "read better over the region on all seven encoders (with two values on five, with sixteen on six)".

PAPER CAN SAY: at equal stored bits, a least-squares associative memory written one fact at a time by a single pass of
the normalised delta rule answered more alias queries than the VSA store over the pre-specified region on every
encoder, by 1.5 to 4.8 points, at about twice the Hebbian write's arithmetic and with a working state of its own
weights; the batch solve led by more (4.1 to 6.5); recursive least squares reproduces the batch memory to within half a
point but needs a working matrix hundreds of times larger than the budget; counted with its feature map, the store's
write and read are the more expensive ones.

PAPER MUST NOT SAY: that the online write matches the batch solve (it trails by up to 4.1 over the region and 7.0 with
sixteen values on Llama); that the delta rule beat the store with two values on every encoder (bge-small: store ahead;
MiniLM-L6: level); that RLS is an online memory at equal bits without the working-state statement; that the Hebbian
write is cheaper without "excluding the feature map"; that superposition per se loses (all four arms are superposed
memories; the comparison is between write rules); any B57 NLMS / RLS number without marking it superseded.

## Disclosed analysis additions after the test (no arm, choice or bar changed)

The write-position reading (b62_pos.py) and the per-encoder cost ratios including feature maps (computed from
results/b62_verdict.json's cost rows) were not registered as statistics; they are reported as analysis. Q10's line in
results/b62_predictions.txt was filled in after the checks ran.
