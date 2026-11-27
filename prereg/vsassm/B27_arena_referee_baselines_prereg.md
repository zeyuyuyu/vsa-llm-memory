# B27 pre-registration: a label-free anti-collapse projection and a tuned LoRA fine-tune in the three controlled arenas (referee baselines)

Date: 2026-11-17 (box clock and UTC; US Eastern 2026-11-17 morning). Written before any B27 run reads a test label or
scores a test instance. Done before this file (no test score involved; listed in "Done before registration"): code
written; the original arena modules copied verbatim (md5 below); generator self-tests; frozen Phi-2 states cached for
the train and dev splits and the second views; LoRA and label-free smoke runs on the training and dev splits only, for
timing and memory.

## Why

A referee of the npj Unconventional Computing manuscript (Results "To use a vector symbolic memory, a language model
must give each symbol its own code" and "What the vector symbolic memory adds"; Fig. 4a, Fig. 5b; Supplementary Notes
"Arenas and arms", "The cross-talk sweep", "What the naming head must deliver") raises two points.

1. The projection trained only through the binding loss (Fig. 4a pink: A 0.501, B 0.507, C 0.483) has no
   anti-collapse term, while the naming head receives identity labels from the generator. Missing baseline: a
   projection trained WITHOUT identity labels but WITH an anti-collapse regulariser (an orthogonality penalty among
   the codes of different mentions in an instance, VICReg-style variance/covariance terms, or a label-free
   contrastive objective).
2. The LoRA fine-tune of arenas A and C does not fit its own training split (final training accuracy 0.50-0.53), ran
   at one configuration, and in arena C on a smaller split (600/400 against 800/500), yet Fig. 5b uses its margins
   (+0.42 on C, +0.41 on A) to define when the memory helps. Needed: a fine-tune tuned on a dev split (learning rate,
   epochs, rank) until it fits its training split, on the same split sizes as the other arms, with several seeds.

B27 runs both baselines in all three arenas, three seeds each. Its result goes into the paper whatever it shows: if a
label-free projection matches the naming head, the paper must say that identity labels are not needed; if the tuned
fine-tune closes the gap, the paper must withdraw "the memory helps where each input binds dozens of new symbols" for
that arena.

## What is reused exactly

- **Arena code:** verbatim copies in `b27_arena/` (md5 in `b27_arena/MD5SUMS`, identical to the originals on the
  A800 box, checked 2026-11-17):
  - v13.py `996644f9…` (arena C, ngcls), v11.py `38dd9961…` (arena B, V-11/V-18 span fix),
  - v8.py `5d9e0122…` (binding-loss arm on A), v9.py `9f9efcba…` (naming head on A),
  - systematic_chain_dataset.py `dd30f7f9…`, systematic_role_binding_dataset.py `094256271…`,
  - explicit_chain_models.py `3dd5eb4f…`, vsa.py `0220c3e1…`, v15.py `763e3dcf…` (published B/C fine-tune, reference only).
  `b27_common.load_arena_modules` imports these copies by path, so no original file is imported or written.
- **Instances:** the published train and test splits, from the original classes and seeds.
  - A: SystematicChainDataset, train 2,000 (rng offset 42), test 1,000 ("eval", offset 123).
  - B: v11.Arena, train 1,000 ("train"), test 500 ("eval": every queried pair held out of training).
  - C: v13.Arena, train 800 (seed 42), test 500 (seed 4242).
  - The LoRA fine-tune uses these same instances and sizes (the published C fine-tune used 600/400).
- **Frozen states:** Phi-2 fp16 `last_hidden_state`, computed by the original functions with their original batch
  sizes (v8.cache_hidden, v11.cache, v13.cache), cached once to disk and checked against the rows' token ids at load.
- **Scoring:** as published. B and C: the training-accuracy-maximising threshold on the final cosine, fitted in closed
  form on the training instances, applied to dev/test. A: argmax of the two learned logits (scale and bias).
- **Comparators** (banked per-seed values behind Fig. 4a; figdata md5-identical to the box):
  - naming head: A 0.915 / 0.913 / 0.913 (mean 0.914), B 0.990 / 1.000 / 0.998 (0.996), C 0.932 / 0.932 / 0.938 (0.934);
  - probe floor: A 0.508 / 0.511 / 0.508 (0.509), B 0.794 / 0.818 / 0.832 (0.815), C 0.512 / 0.530 / 0.504 (0.515);
  - binding-loss projection: A 0.511 / 0.499 / 0.493, B 0.502 / 0.498 / 0.522, C 0.472 / 0.502 / 0.476;
  - published LoRA: A 0.493 / 0.508 / 0.515 (attention only, mean pool, lr 1e-4, 2 epochs);
    B 0.980 / 0.895 / 0.510 and C 0.515 / 0.518 / 0.515 (600/400 split, lr 2e-4, batch 2, 2 epochs).
  - Seeds 42, 123, 456 as published; in every arena the instances are the same for every seed.

## Dev split (all tuning and every selection; new in B27)

- A: 500 instances from the same generator code with rng offset 777 (`Random(1000003 * i + 777)`).
- B: 500 instances drawn like the test split (queried pair held out of training), rng offset 777.
- C: 500 instances, `v13.Arena(tok, 500, 777)`.
- The self-test reports exact token-level overlap of dev with train and test (expected 0).
- Tuning runs build only the train and dev splits; the test split is built only by final runs. Every selection
  (configuration, family, epoch) uses dev accuracy only.

## Part 1. Label-free projections with an anti-collapse term

### Arms

Each arm is the published binding-loss arm of its arena (v8.SpanVSA on A, v11.BindArm "spans" on B, v13.BindArm
"spans" on C: a linear map 2,560 -> 2,048 from the span-pooled frozen state, L2-normalised, trained end to end through
binding, bundling, unbinding and clean-up on the answer loss), unchanged, plus one regulariser R with weight lambda:

- **orth:** the mean squared cosine between the codes of every pair of different mentions in an instance (all
  mention positions; no label says which mentions are the same symbol).
- **vic:** VICReg variance hinge and covariance penalty over the codes of all mentions in the batch, on the unit codes
  scaled by sqrt(2048) (gamma = 1, covariance weight 0.04 = VICReg's 1:25). On unit vectors the variance hinge also
  forbids a shared mean direction, the collapse measured in V-165 (cross-talk 0.72 -> 0.99).
- **nce:** symmetric InfoNCE (temperature 0.1) between the code of each mention and the code of the same mention in a
  second rendering of the same training instance with the pair order reshuffled (same bindings, query and label;
  re-encoded by the frozen Phi-2). Positives come from the augmentation, as in SimCLR; every other mention in the
  batch is a negative, including other mentions of the same symbol, since no label is available.

Mentions are the spans whose projection enters the store: A, every R, X and Y marker (81 per instance); B, the 64
context fillers; C, the 128 context roles and fillers; plus, under qc = proj, the query and candidate spans.

**Option qc (B and C only).** In the published arms of B and C the query role and the candidate enter as codebook rows
by index (for every arm, including the naming head). Under qc = index this is kept. Under qc = proj the query (C) and
the candidate (B, C) are read through the same projection from their own token spans, so that no codebook row and
no identity reaches the store at all, and the projection need only be consistent and separated, not aligned with a
random codebook. In A the published arm already projects the query role, and the last hop uses codebook rows for the
fillers in every arm, so A has no qc option.

Nothing else changes: optimiser groups (AdamW; scale and bias at 0.1 on B and C), batch size (A 8, B 4, C 4),
gradient clipping 1.0, RNG order, scoring. With R off and qc = index the code path is the published arm (gate G0b).

### Tuning grid (seed 42, train + dev only; 10 epochs, dev scored after every epoch)

- lambda in {0.1, 1, 10, 100} for each of orth, vic, nce (100 added after the smoke runs below showed vic at lambda 1
  still collapsing in A; decided before any test run);
- learning rate in {3e-4 (published), 1e-3};
- qc in {index, proj} on B and C.
- That is 24 configurations on A and 48 each on B and C (120 runs).

### Final runs

For each arena and each family, the dev-selected configuration with seeds 42, 123 and 456, 10 epochs; test accuracy
is read at the epoch with the highest dev accuracy of that seed (primary), and also at the final epoch and the best
test epoch (the published protocol, which favours each arm; reported, never used for a bar). The cross-talk geometry
of the codes (V-165 protocol: first 120 test instances, up to 4 occurrences per symbol, first 40 symbols; mean cosine
between different symbols and between occurrences of the same symbol; labels group the vectors for this measurement
only) is logged after every epoch.

## Part 2. A tuned LoRA fine-tune

- **Model:** Phi-2 in bf16 with LoRA on every linear layer (q, k, v, dense, fc1, fc2), alpha = 2 x rank, dropout 0.05,
  fp32 adapters; a linear head on the final state of the last non-pad token (the readout of the published B/C
  fine-tune). Rank 16 gives about 23.6M trainable parameters.
- **Optimisation:** AdamW, weight decay 0, linear warm-up over 6% of the steps, cosine decay to 0 at the last epoch,
  batch 8, gradient clipping 1.0, cross-entropy on the two classes.
- **Inputs:** the arena token ids; each batch is cut to its longest unpadded row (right padding, causal model: no read
  state changes).
- **Every epoch** scores the whole training split (eval mode), the dev split, and in final runs the test split.

### Tuning (seed 42, train + dev only, 12 epochs)

- **Stage 1:** rank 16, learning rate in {3e-5, 1e-4, 3e-4, 1e-3}.
- **Stage 1b:** if the selected rate is at an edge of that grid, one more rate beyond it (1e-5 or 3e-3).
- **Stage 2:** ranks 4 and 64 at the selected rate.
- **Fit criterion:** training accuracy >= 0.95 at some epoch.
- **Selection** (b27_select.py): among the configurations that fit, the one with the highest dev accuracy at its best
  epoch; if none fits, the highest dev accuracy overall. Ties within 0.004 (2 of 500): rank 16, then the smaller rate.
- **Stage 3 (only if no configuration fits in 12 epochs):** the configuration closest to fitting (highest training
  accuracy at any epoch; ties: higher dev, rank 16, smaller rate) again with 30 epochs. If it fits, it is the selected
  configuration (30 epochs); otherwise the dev-selected 12-epoch configuration stays selected and the fine-tune is
  reported as "did not fit its training split after N configurations".

### Final runs

The selected configuration, seeds 42, 123, 456; test accuracy at each seed's dev-best epoch (primary), also the final
epoch and the best test epoch (published protocol). Training accuracy per epoch is reported for every run.

## Selection of the label-free configuration (b27_select.py)

- Per arena and family: the highest dev accuracy at its best epoch; ties within 0.002 go to the smaller lambda, then
  learning rate 3e-4, then qc = index.
- The **primary label-free arm** of an arena is the family with the highest selected dev accuracy (ties: orth, vic,
  nce). The bars below are judged on it; the other two families are reported beside it.

## Gates (run first, after this file is committed)

- **G0a (environment):** the ORIGINAL run functions (v13.run, v11.run, v8.run, v9.run_arm) on the published
  configuration reproduce the banked per-seed test accuracies within 0.004 (2 of 500): C parser, naming head
  (probe_argmax), binding loss (spans); B the same three; A naming head (extract_then_bind) and binding loss; seeds
  42, 123, 456.
- **G0b (new code path):** b27_lf.py with R off, qc = index, lr 3e-4, 2 epochs, best test epoch reproduces the banked
  binding-loss accuracy of each arena and seed within 0.004.
- If a gate fails, no Part 1 test is read until the cause is found and written in a dated amendment.

## Bars and how each outcome will be read (fixed now)

Test accuracy at the dev-selected epoch, per seed s. Floor F and naming head N are the banked means above.

### Point 1: does a label-free projection match the naming head?

Recovery R_s = (acc_s - F) / (N - F).

- **MATCH:** mean recovery >= 0.90 and R_s >= 0.80 in 3/3 seeds.
  - In accuracy: A mean >= 0.874 and each seed >= 0.833; B >= 0.978 and >= 0.960; C >= 0.892 and >= 0.850.
- **PARTIAL:** not MATCH, mean recovery >= 0.25.
- **FAIL:** mean recovery < 0.25.

Readings:
- Primary arm MATCH in an arena: identity labels are not needed there. A label-free anti-collapse objective gives
  codes good enough for the memory, and the paper must say so and recast "the naming head" as one way of supplying
  codes, not the necessary one.
- PARTIAL: anti-collapse recovers part of the gap; the paper reports the size.
- FAIL: the binding-loss failure is not only collapse; without an identity signal the frozen states do not yield
  usable codes in that arena. The cross-talk geometry says which property is missing (separation or consistency).
- A non-primary family that reaches MATCH on test is reported as such, with the note that it was not the dev choice.

### Point 2: does a tuned, fitted fine-tune close the gap?

Margin M = N - mean LoRA accuracy; per seed m_s = N_s - LoRA_s (naming head per seed, as banked).

- **STANDS:** M >= 0.10 and m_s >= 0.10 in 3/3 seeds (0.10 is the bold threshold of Fig. 5b).
- **CLOSED:** M <= 0.03.
- **NARROWED:** otherwise.
- **Validity:** the fine-tune counts as fitted in an arena if the selected configuration reached training accuracy
  >= 0.95 in tuning and in each final seed. If it did not, the arena's result is labelled "under-fitted baseline": the
  paper may report the margin but not as evidence that a fine-tune cannot learn the arena.

Readings:
- STANDS in A and C with a fitted fine-tune: the claim "the memory helps where each input binds dozens of new
  symbols" survives the referee's baseline, with the new margins replacing +0.41 / +0.42.
- CLOSED: the paper withdraws that claim for the arena and moves the arena row of Fig. 5b to "a fine-tune learns the
  task".
- NARROWED: the paper reports the smaller margin and drops "bold" where it falls under 0.10.
- Either way the Fig. 5b arena rows are redrawn with the tuned fine-tune, and its training accuracy is stated.

## Predictions (stated so they can be wrong)

- G0a and G0b pass in every cell.
- Label-free primary arm: A FAIL, B PARTIAL, C FAIL; no family reaches MATCH in any arena.
- In C the regularised arms separate the codes but do not make them consistent: on test, between-symbol cosine
  <= 0.25 and same-symbol cosine <= 0.6. (In C one symbol is written in decimal as a filler and in hex as a role, so
  consistency across the bridge needs the decimal-hex correspondence, which no label-free term supplies.)
- qc = proj is selected on dev in B and C.
- The tuned fine-tune fits its training split (>= 0.95) in all three arenas.
- Tuned fine-tune test means: B >= 0.95 with every seed >= 0.90; A <= 0.70; C <= 0.65.
- Point 2: STANDS on A and C, CLOSED on B.
- In C the fine-tune fits by memorising: training accuracy >= 0.95 with dev accuracy <= 0.65.

## Compute and budget

- A800 box (8 x A800 80 GB), idle GPUs only, never GPU 3, never another user's process. PACE was checked first
  (2026-11-17 09:54 UTC, `sbatch --test-only`, 1 GPU, 2 h): earliest starts H200 11-23, A100 11-22, H100 12-09, so the
  rule "if PACE queues, use the A800" applies.
- b27_worker.sh claims one idle GPU (< 500 MiB used, re-checked after 20 s; lock against other B27 workers) and runs a
  queue on it, several label-free processes per GPU; LoRA one run per GPU.
- GPU-hours are counted as GPU occupancy (worker start to end per GPU, logs/gpu_ledger.txt), failed runs included;
  process-hours are reported beside it.
- **Projection:** LoRA stage 1 about 6 GPU-hours, stages 1b-2 about 4, stage 3 (if needed) about 4, finals 5-11;
  label-free tuning about 4 (two GPUs, ten processes each), finals about 1.5; gates and caching about 1. Total about
  25-31 GPU-hours.
- **Cap:** 40 GPU-hours. If the projection passes the cap, cut in this order: LoRA stage 2 rank 4, then the
  non-primary label-free families' final runs.

## Checks from other angles (in the verdict)

- **Second implementation:** b27_verdict.py recomputes every test accuracy from the saved per-instance scores, labels
  and thresholds, independently of the logged numbers.
- **Intervals:** per-seed 95% Wilson intervals; for each arena mean, a bootstrap over test instances (paired across
  seeds, 10,000 resamples).
- **Geometry:** cross-talk (different-symbol and same-symbol cosine) of every label-free arm, set against the 0.25
  tolerance of the cross-talk sweep.
- **Fine-tune behaviour:** predicted-positive rate on test (a constant classifier is flagged); training versus dev
  curves.
- **Literature:** collapse and anti-collapse (SimCLR, BYOL, VICReg, Barlow Twins) and the binding problem, checked
  against the conclusion.

## Done before registration (no test score; train and dev only)

- **Self-test** (b27_prep.py selftest, out/selftest.json): the new split code with the original offsets reproduces
  the original generators exactly (A train 2,000 and test 1,000, B train 1,000 and test 500: token ids, labels,
  queries, candidates and spans identical); the second views hold the same bindings, query and label in a new order
  (A 2,000/2,000, B 1,000/1,000, C 800/800 reordered); the recorded query and candidate spans decode to the right
  symbols; tokenizer pads on the right; no input is truncated (longest rows: A 429 of 512, B 482 and C 491 of 768);
  dev, train and test share no instance (0 exact token-level duplicates between any two).
- **States cached** for train, dev and the second views of the training split (all arenas), with the original cache
  functions (fp16; C 36-42 s, B 43 s, A 51 s per training split on one A800).
- **LoRA smokes (training split only):**
  - 64 C instances, lr 3e-4, 10 epochs (80 steps): training accuracy stuck at the majority rate (0.56), the
    predicted class flipping between all-positive and all-negative.
  - 32 C instances, 40 epochs (160 steps): lr 1e-4 reached 0.81-0.94, lr 1e-3 reached 0.97 (it sat at 0.47 for 16
    epochs first).
  - So the pipeline can fit, but slowly; this is why stage 3 targets the configuration closest to fitting.
  - Rank 16 on all linear layers = 23,598,082 trainable parameters; peak memory 32 GB at batch 8; about 0.1 s per
    training instance per epoch including the training-split scoring.
- **Label-free smokes (one epoch, lambda 1, seed 42, train + dev):**

  | arm | train acc | dev acc | dev cross-talk (different / same symbol) |
  |---|---|---|---|
  | A nce | 0.602 | 0.534 | 0.764 / 0.874 |
  | A vic | 0.665 | 0.506 | 0.974 / 0.977 |
  | B nce proj | 0.818 | 0.668 | 0.279 / 0.936 |
  | B vic index | 0.947 | 0.500 | 0.947 / 0.984 |
  | C nce proj | 0.545 | 0.508 | 0.190 / 0.359 |
  | C orth index | 0.596 | 0.498 | 0.340 / 0.389 |

  - About 70-100 s per epoch with four processes on one GPU.
  - vic at lambda 1 does not stop the collapse in A or B (different-symbol cosine 0.97 and 0.95), so lambda 100 was
    added to the grid for every family.
- No model has seen a test instance. The self-test generated the test inputs only to compare their token ids and
  labels with the original generator's; nothing was scored on them, and no test state has been computed.

## Files

- Code: b27_common.py, b27_lf.py, b27_lora.py, b27_prep.py, b27_select.py, b27_verdict.py, b27_worker.sh, b27_arena/.
- Box: /path/to/work/b27 (feats/, out/, logs/), separate from the ngcls repository and its ledger.
- Results: results/b27_*.json, results/b27_verdict.txt.
