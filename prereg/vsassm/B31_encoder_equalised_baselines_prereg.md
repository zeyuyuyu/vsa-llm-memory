# B31 pre-registration: the coupling result on a larger, modern frozen encoder (Qwen3-8B), against equalised baselines

Date: 2026-11-18, about 05:45 UTC (13:45 on the box clock, Asia/Shanghai; US Eastern 2026-11-18 00:45). Written, with
all B31 code, gates and the verdict generator, before any B31 run reads a test label or scores a test instance. What
was done before this file (train and dev splits only, or no data at all) is listed under "Done before registration".

## Why

Four referees of the npj Unconventional Computing manuscript single out the coupling ("interface") result in the
synthetic arenas A, B and C over frozen Phi-2 states (Results "To use a vector symbolic memory, a language model must
give each symbol its own code"; Fig. 4; Supplementary Notes "Arenas and arms", "The cross-talk sweep") as the most
novel result, and object that it is shown on one 2.7B encoder and against tilted baselines:

1. one encoder only (Phi-2, 2.7B);
2. the memory arms receive the query and the candidate by index, while the probe, the learned map and the fine-tune
   must find them in the text;
3. the naming head receives identity labels that the fine-tune does not;
4. the probe floor is weak (a standardised probe reached 0.922 on arena-B dev against the reported 0.815; Phi-2
   last-token states have pairwise cosine 0.95-0.98);
5. no tuned LoRA fitted its training split, and rank was never tuned;
6. there is no dictionary arm;
7. arena A was not rerun with the order-keeping (permuted) binding that B28 showed removes the arena-C alias ceiling.

B31 reruns every arm of the coupling result on Qwen3-8B (and on Phi-2, for continuity), with equalised baselines, a
fine-tune given the same identity labels as the naming head, a dictionary arm, and the permuted binding in all three
arenas. Its result goes into the paper whatever it shows.

## A finding made before registration that changes arena A (train and dev only)

To predict question (d) I checked, on arena A's training and dev splits only, why exact symbols reach only 0.918 there
(scripts and outputs in `b31_diag/`):

- **The HRR chain is not the limit.** With the gold symbols parsed from the text by a character-level regex, the
  three-hop chain retrieves the right symbol at every hop on 2,000/2,000 training and 500/500 dev instances in every
  seed, under the published commutative binding and under the permuted one, and the closed-form threshold separates
  the classes perfectly (1.000; positives' cosine 0.15-0.25, negatives' at most 0.09). Arena A has no aliases: its
  roles, X, Y and fillers have four separate codebooks.
- **The published token scanner drops markers.** `SymbolicVSAChainClassifier._scan_markers` (used by the published
  exact-symbol arm, the naming head and the binding-loss map of arena A) expects the kind letter as its own token.
  Phi-2's tokenizer merges `Y` with a following hex letter (`[YA6]` -> `Ġ[`, `YA`, `6`, `]`), so the scanner misses those
  Y markers and pairs the remaining markers by adjacency. It drops 10,146 Y markers in the 2,000 training instances
  (1,893 instances lose at least one; dev 475 of 500, and, by the self-test's structural count, test 931 of 1,000); no
  R or X marker is lost. Run as published (scanner view, learned scale and bias, threshold starting at 0.20), the exact-
  symbol arm scores 0.9145 on train and 0.932 on dev at its initial read-out.
- So **the published 0.918 (exact) and 0.914 (naming head) in arena A are a tokenisation artefact of the scanner**, not a
  property of the binding, the encoding or the memory. B31 finds every mention by character offsets (below), for both
  encoders, and reproduces the published A numbers only inside gate G0, by emulating the scanner's view.

## What is reused exactly

- **Arena code:** the verbatim copies in `b27_arena/` (md5 in `b27_arena/MD5SUMS`; B27), loaded by
  `b27_common.load_arena_modules`. `b31_struct.py` re-runs the generator code of `b27_common` (A: `gen_chain`; B:
  `make_b_arena`) and of `v13.Arena` (C) with the same random streams and keeps the text. The self-test (below) shows
  that under the Phi-2 tokenizer the resulting token ids equal the B27 rows for every instance of every split, and that
  the token spans derived from character offsets equal the spans the published arms read (B and C: the rows' spans; A:
  the scanner's span for every marker it finds).
- **Splits:** as B27. A: train 2,000 (rng offset 42), dev 500 (offset 777), test 1,000 (offset 123). B: train 1,000,
  dev 500 and test 500 (dev and test draw only held-out queried pairs). C: train 800 (seed 42), dev 500 (777), test 500
  (4242). Second views for InfoNCE: the B27 reshuffles (offset 990001). The same instances (same text) go to both
  encoders.
- **Codebooks, chains, clean-up and read-out:** as published, per arena (A: vsa.py operators and four codebooks; B:
  v11 role and filler codebooks, one hop, no clean-up; C: v13 shared 64-symbol codebook, two hops with clean-up),
  d = 2048, seeds 42, 123, 456. Read-out: the closed-form training-accuracy-maximising threshold on the final cosine,
  as published for B and C and now also for A (the published A read-out, a scale and bias learned for two epochs, is
  kept inside G0 only).
- **Encodings (B28):** comm (published HRR) and perm (the value argument permuted by a fixed random permutation of the
  2,048 coordinates, private generator seed + 7001). For learned codes the value code is the code permuted.
- **Comparators banked for G0** (per seed, best test epoch of 2, published configuration): A exact 0.918 / 0.919 /
  0.918 (figdata v1_logs/vsa_parser.json), naming head 0.915 / 0.913 / 0.913, binding-loss map 0.511 / 0.499 / 0.493; B
  exact and naming head 0.990 / 1.000 / 0.998, binding-loss 0.502 / 0.498 / 0.522; C exact and naming head 0.932 /
  0.932 / 0.938, binding-loss 0.472 / 0.502 / 0.476 (b27_prep.BANKED).

## Encoders

- **Phi-2** (2.7B; continuity): the frozen fp16 `last_hidden_state` cached by B27 with the original cache functions;
  span means are taken on the GPU exactly as the published arms take them (`H[s:e].float().mean(0)`).
- **Qwen3-8B** (8.2B, base checkpoint `Qwen/Qwen3-8B`, snapshot b968826d), frozen, bf16, `AutoModel.last_hidden_state`
  (after the final RMSNorm), the arena text tokenised without truncation (longest inputs: A 486, B 576, C 520 tokens).
- **Mentions** are found by character offsets: A's markers `[R3A]` span the whole marker (the published scanner's
  span), every other mention spans its digits (B's and C's role hex digits, every filler's decimal digits). The token
  span is every token that overlaps the characters (`v13.token_span`). This is the "hand-written span finder" of the
  published arms, now applied to every symbol, including the query and the candidate.

## Arms

Memory arms put one code per mention into the arena's published chain; they differ only in where the codes come
from. "named" = every symbol, including the query and the candidate (and arena A's fillers), comes from its own span:
no codebook row is chosen by an index parsed from the string. "index" = the published convention (A: fillers and the
candidate by index; B: roles, the query role and the candidate by index; C: the query and the candidate by index).

| key | arm | codes | trained on | encoding / convention |
|---|---|---|---|---|
| dict | exact symbols + dictionary (upper bound) | gold symbols, Python dicts followed from the query | nothing | none |
| exact | exact symbols + VSA | codebook rows of the gold symbols | nothing | **perm** (comm reported) |
| name | naming head + VSA | arg-max of a linear head over the span-mean state picks the codebook row (snapped) | identity CE on **every** mention + the answer loss on scale/bias, as published; AdamW 3e-4 (A one group; B, C scale/bias at 0.1), batch 8/4/4, clip 1.0, 4 epochs | **perm / named** (perm/index, comm/named, comm/index read from the same head) |
| soft | naming head read without snapping | linear map fitted with the same labels to the codebook row (1 - cosine), frozen, read continuously (published "frozen continuous projection") | identity fit on every mention, AdamW 1e-3, data order, up to 6 fit epochs | perm / named |
| bind | binding-loss map | L2-normalised linear map of the span-mean state, trained only through the memory's answer | answer loss only; lr in {3e-4, 1e-3} chosen on dev; 10 epochs | perm / named |
| bindpub | binding-loss map, published form | as bind | lr 3e-4, 10 epochs | comm / index |
| nce | label-free InfoNCE map (B27's best family) | as bind + lambda x symmetric InfoNCE (tau 0.1) between each mention's code and the same mention in the reshuffled view; every other mention in the batch is a negative | lambda in {1, 10, 100} x lr in {3e-4, 1e-3} chosen on dev; 10 epochs | perm / named |
| probe | equalised probe | standardised read-out of the frozen states given the question's positions | see below | none |
| lora | equalised LoRA | fine-tune, head on the question's positions | see below | none |
| loraid | LoRA + identity auxiliary loss | as lora + an identity head on every mention, same labels as the naming head | see below | none |

- **Identity labels** are the generator's symbol ids. The naming head, the no-snapping head and LoRA + identity loss get
  exactly the same labels: every symbol mention in the input (A 98, B 130, C 130 per instance; R mentions are read over
  the 64 roles, other kinds over 256 in A and B; C's 64 symbols are shared by roles and fillers).
- **Equalised probe** (`b31_probe.py`). Features, each block standardised with training-split statistics: `last` (the
  last-token state; B27's standardised diagnostic), `eq` (query-span mean, candidate-span mean, last-token state, mean
  over the input), `eqx` (`eq` plus the elementwise product of the query and candidate span means). Models: L2 logistic
  regression (lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}; Adam 1e-2, full batch, 2,000 steps) and a one-hidden-layer MLP
  (512 ReLU, dropout 0.1, AdamW 1e-3, weight decay in {1e-4, 1e-2, 1e-1}, 1,000 steps). Dev accuracy every 50 steps; the
  best (feature set, model, hyper-parameter, step) on dev is selected (ties: logistic, smaller feature set, larger
  penalty, earlier step) and retrained with seeds 42, 123, 456; test read once at the selected step.
- **Equalised LoRA** (`b31_lora.py`). LoRA on every linear layer (Phi-2: q, k, v, dense, fc1, fc2; Qwen3-8B: q, k, v,
  o, gate, up, down), alpha = 2 x rank, dropout 0.05, bf16 base, fp32 adapters and heads. Head: one linear layer on
  [query-span mean, candidate-span mean, last-token state] of the final layer, each dimension standardised with fixed
  statistics of the initial model on the training split. AdamW, weight decay 0, warm-up 6% then cosine to 0, batch 8,
  clip 1.0, 8 epochs. Every epoch scores a fixed training subset (the first 500 training instances, eval mode) and dev;
  the dev-best state is kept and, in final runs, scored once on test.
- **LoRA + identity auxiliary loss**: the same plus an identity head (one linear layer) on the span-mean final state of
  every mention, cross-entropy on the identity labels, added to the task loss with weight 1 (the naming head's weight).

### LoRA tuning (seed 42, train + dev only)

- **Stage 1 (equalised LoRA):** rank in {8, 64} x lr in {3e-5, 1e-4, 3e-4} for Phi-2 and {1e-4, 3e-4} for Qwen3-8B
  (3e-5 is the rate B27 found best on Phi-2; 1e-4-3e-4 is the usual range for LoRA on 7-8B models).
- **Fit criterion:** training-subset accuracy >= 0.95 at some epoch.
- **Selection** (`b31_select.py`): among configurations that fit, the highest dev accuracy at its best epoch; if none
  fits, the highest dev accuracy overall; ties within 0.004: smaller rank, then smaller rate.
- **Stage 2 (LoRA + identity loss):** lr in {1e-4, 3e-4} at the rank selected in stage 1; same rule.
- **Stage 3 (only where no stage-1 configuration fits, arenas A and C):** the configuration closest to fitting (highest
  training-subset accuracy; ties: higher dev) again for 24 epochs. If it fits, it becomes the selected configuration
  (24 epochs); otherwise the stage-1 selection stays and the arena is reported as "did not fit its training split after
  N configurations (ranks 8 and 64, rates ..., up to 24 epochs)".
- **Final runs:** the selected configurations, seeds 42, 123, 456, test at each seed's dev-best epoch.

### Memory-arm tuning (seed 42, train + dev)

- bind: lr in {3e-4, 1e-3}; nce: lambda in {1, 10, 100} x lr in {3e-4, 1e-3}; highest best-dev accuracy (reading
  perm/named), ties within 0.002 to the smaller lambda, then lr 3e-4. name, soft, bindpub and exact are not tuned.
- Final runs: 3 seeds; test read at each seed's dev-best epoch (primary; the final epoch is reported beside it). The
  cross-talk geometry (B27/V-165 protocol: first 120 test instances, up to 4 occurrences per symbol, first 40 symbols;
  mean cosine between different symbols and between occurrences of one symbol; context mentions; labels group the
  vectors for this measurement only) is logged at every epoch.

## Gates (run first, after this file is committed)

- **G0-unit (train split, deterministic; run before registration, result below):** for every arena and the three
  published arm types (exact rows, naming head, binding-loss map) the new vectorised arm equals the original class on
  identical weights: same initial parameters from the same `torch.manual_seed`, final cosine (B, C) or positive logit
  (A) within 1e-4, the same decision on >= 99.5% of the first 200 training instances, at the initialisation and with
  identical perturbed weights. Arena A on both sides through the published scanner's view.
- **G0 (published configuration through the new code path, Phi-2):** comm, index, the published identity-label sets,
  2 epochs, best test epoch, arena A with the scanner's view and the learned scale/bias read-out; exact rows, naming
  head and binding-loss map reproduce the banked per-seed numbers within 0.004 (2 of 500, 4 of 1,000) in every arena.
- **Self-test (done):** see below. If a gate fails, no B31 comparison involving that arm is read until the cause is
  found and written in a dated amendment.

## Bars and readings (fixed now)

Notation, per encoder e and arena X: accuracies on test per seed; C = exact + VSA (perm) mean (the algebra's ceiling);
F = the equalised probe's mean on e (the floor); recovery R_s = (acc_s - F) / (C - F), undefined when C - F < 0.10.
Intervals: Wilson 95% per seed; means and differences by a bootstrap over test instances (paired across seeds and
arms, 10,000 resamples).

### (a) Does the binding-loss map still collapse, and are label-free codes still inconsistent, on Qwen3-8B?

- binding-loss map (bind and bindpub), geometry: **COLLAPSES** if the different-symbol cosine at the dev-selected epoch
  is >= 0.50 (twice the memory's 0.25 tolerance) in 3/3 seeds; **SEPARATED** if <= 0.25 in 3/3; PARTIAL otherwise.
- InfoNCE map, geometry: **INCONSISTENT** if the mean different-symbol cosine is <= 0.25 and the mean same-symbol
  cosine <= 0.80; CONSISTENT if separated with same-symbol > 0.80; NOT SEPARATED otherwise.
- accuracy (every map): **MATCH** mean R >= 0.90 and R_s >= 0.80 in 3/3; **PARTIAL** mean R >= 0.25; **FAIL** otherwise;
  where R is undefined, FAIL if the mean is <= F + 0.03.
- **Answer:** "the collapse persists on Qwen3-8B" iff the published-form map COLLAPSES and FAILS in all three arenas;
  "label-free codes stay inconsistent" iff InfoNCE is INCONSISTENT and short of MATCH in B and C.
- Readings: persists -> the paper states the cross-talk and the failure for both encoders; SEPARATED or MATCH on
  Qwen3-8B -> the paper must say the collapse (or the need for identity labels) is encoder-specific and say on which.

### (b) Does the naming head's lead over the strongest equalised baseline remain >= 0.10 in arenas A and C?

- N = naming head + VSA (perm, named). Strongest baseline Bmax = the equalised probe or the equalised LoRA, whichever
  has the higher test mean (the choice favours the baseline). M = mean(N) - mean(Bmax); m_s = N_s - Bmax_s.
- **STANDS:** M >= 0.10 and m_s >= 0.10 in 3/3; **CLOSED:** M <= 0.03; **NARROWED** otherwise. Reported for A, B and C;
  the bar is A and C on each encoder.
- Readings: STANDS in A and C on both encoders -> "the memory helps where each input binds dozens of new symbols"
  survives a second, larger encoder and equalised baselines, with the new margins replacing the old. NARROWED/CLOSED
  on an encoder -> the paper reports the smaller margin for that encoder and drops "bold" (or the claim) there.

### (c) With identical identity supervision, does the fine-tune catch up?

- Same rule with LoRA + identity loss in place of Bmax: **CLOSED** = the fine-tune catches up; STANDS = it does not.
- Effect of the identity loss on the fine-tune, LoRA+id - LoRA (paired by seed): **HELPS** if the mean is >= +0.05 and
  its bootstrap interval excludes 0.
- Readings: CLOSED -> the paper must say that identity labels, not the memory, carry the naming head's advantage where
  it closes; STANDS -> identity labels alone do not let a fine-tune compose the chain; the memory does the composing.

### (d) Does the permuted binding lift arena A's exact-code ceiling to the dictionary's 1.000?

- exact + VSA under perm in A: **LIFTED** if >= 0.98 in 3/3 seeds (B28's bar). If comm is also >= 0.98 in 3/3, the
  reading is **CEILING WAS NOT THE BINDING** (the published 0.918 is the scanner; the paper replaces it and says why);
  if only perm reaches it, PERM LIFTS THE CEILING; otherwise NOT LIFTED.

## Predictions (stated so they can be wrong)

- P1 G0-unit and G0 pass in every cell.
- P2 the dictionary scores 1.000 in every arena (fixed by construction; the self-test checked dictionary = label on
  every split).
- P3 exact + VSA (perm) >= 0.99 in every seed of A and C; in B (no aliases, noise-limited at 64 bindings) within 0.006
  of comm per seed.
- P4 (d) arena A: comm is also >= 0.98 in every seed: CEILING WAS NOT THE BINDING.
- P5 naming head (perm, named) mean >= 0.98 in every arena on Phi-2 and >= 0.95 on Qwen3-8B.
- P6 published-form binding-loss map: different-symbol cosine >= 0.90 in every arena and seed, accuracy FAIL, on
  each encoder.
- P7 B31-form binding-loss map (perm, named): COLLAPSES and FAIL in every arena, on each encoder.
- P8 InfoNCE: INCONSISTENT and short of MATCH in B and C, PARTIAL in A, on each encoder.
- P9 equalised probe: A and C means <= 0.60 on both encoders; B >= 0.90 on both.
- P10 Phi-2 equalised LoRA: A and C <= 0.65, B >= 0.95.
- P11 Qwen3-8B equalised LoRA: A <= 0.80, C <= 0.70, B >= 0.97.
- P12 (b) STANDS in A and C on each encoder.
- P13 (c) the lead over LoRA + identity loss STANDS in A and C on each encoder.
- P14 LoRA + identity loss within 0.05 of the equalised LoRA in A and C on each encoder.
- P15 snapping matters in A: naming head - no-snapping head >= 0.05 on each encoder.
- P16 the second implementation agrees with the main exact-symbol runs on >= 0.99 of instances in every cell.
- P17 total GPU-hours <= 60.

## Checks from other angles (in the verdict)

- **Recomputation:** `b31_verdict.py` recomputes every test accuracy from the saved per-instance outputs (memory arms:
  saved cosines against the saved training threshold at the dev-selected epoch; probe and LoRA: sign of the saved
  margin) and labels regenerated by `b31_struct`, and flags any difference from the logged number.
- **Second implementation:** `b31_indep.py`, numpy float64, its own regex parse of the raw text, the HRR chain written
  out per instance: dictionary and exact rows under comm and perm, per-instance agreement with the main runs, and 20
  fresh codebooks per arena and encoding.
- **Hand-reading:** `b31_handread.py`, at least 10 instances per arena on Qwen3-8B where the naming head is wrong (with
  the misnamed mentions) and 10 where it and the equalised LoRA disagree, read in the raw text.
- **Diagnostics:** naming accuracy per mention; the identity head of LoRA + identity loss on test mentions; the
  fine-tune's predicted-positive rate; training-subset versus dev curves.
- **Literature:** after the verdict, a search for support or conflict (anisotropy of LM states; in-context binding and
  binding IDs; fine-tuned transformers on multi-hop retrieval and pointer chasing; VSA front-ends for neural encoders),
  recorded as supports / conflicts / no precedent.

## Compute and budget

- A800 box only (PACE login timed out on 2026-11-18 05:15 UTC; the 11-15 survey had earliest PACE starts 1-4 days out),
  idle GPUs only, never GPU 3, never another user's process: `b31_worker.sh` claims a GPU with < 500 MiB used and no
  compute process, re-checked after 20 s, with a lock in the claim directory shared with the B29/B30b wrappers.
- GPU-hours = worker occupancy (logs/gpu_ledger.txt), smokes, gates and failed runs included.
- **Projection** (smokes: Qwen3-8B LoRA about 0.2 s per training instance and 0.04 s per scored instance, 61-67 GB at
  batch 8; Phi-2 about 0.43 x that, 30 GB): Qwen3-8B LoRA stage 1 8.4 h, stage 2 4.2 h, stage 3 up to 4.4 h, finals
  12.6 h; Phi-2 LoRA stage 1 5.4 h, stage 2 1.8 h, stage 3 up to 2 h, finals 5.4 h; memory arms, probes, features and
  gates about 4 h. Total about 48 GPU-hours.
- **Cap: 60 GPU-hours.** If the projection passes the cap, cut in this order: Phi-2 stage 3; Phi-2 stage 2 (LoRA + id then
  uses the stage-1 configuration); Qwen3-8B stage 3 for arena C; Qwen3-8B stage 2 (same fallback).

## Done before registration (no test score)

- **Diagnostics (arena A train + dev):** the exact-symbol chain with regex-parsed symbols (local CPU) and the published
  scanner (box CPU), as described above (`b31_diag/a_exact.py`, `b31_diag/a_scan.py` and their outputs).
- **Self-test** (`b31_prep.py selftest`, out/selftest_{phi2,qwen8}.json): Phi-2: token ids equal the B27 rows (A 2,000 /
  500 / 1,000 / 2,000 second views; B 1,000 / 500 / 500 / 1,000; C 800 / 500 / 500 / 800) and every span equals the
  published one; the A scanner sees every R and X marker and drops only Y markers. Qwen3-8B: no truncation (longest A
  486, B 576, C 520 tokens); every mention's tokens decode to text containing the mention. Every split: the dictionary
  answer equals the label (this fixes the dictionary arm at 1.000 by construction; it is the generator's definition).
- **Features** for the train, dev and second-view splits of both encoders (`b31_prep.py feats`); no test state has
  been computed.
- **LoRA smokes** (64 training instances, one epoch; the "dev" of a smoke is the same 64 training instances): Qwen3-8B
  A rank 64 + identity loss 20 s, 61.3 GB; B rank 8 22 s, 67.4 GB; Phi-2 A rank 64 + identity loss 9 s, 29.6 GB. The
  initial models' last-token states have mean pairwise cosine 0.999 (Qwen3-8B) and 0.968 (Phi-2), so the head inputs
  must be standardised.
- **G0-unit** (train split, first 200 instances; b31_out/g0_unit.json): PASS in all 18 cells. The same initial parameters
  from the same seed in every cell (head, projection, and arena A's role vectors); largest difference of the final
  cosine 1.2e-7 (C) and 3.8e-8 (B), of the positive logit 1.4e-6 (A; a cosine difference of 7e-8); the same decision on
  200 of 200 instances in every cell, at initialisation and with identical perturbed weights.
- **Arm and probe smokes** (train + dev, seed 42, one epoch; launched after this file's predictions were written, read
  before it was committed; nothing changed because of them): Phi-2 C naming head dev 1.000 (99.99% of dev mentions
  named correctly); A exact rows dev 1.000; A binding-loss map dev 0.532 with cross-talk 0.997 / 0.998; B no-snapping
  head (one fit epoch) dev 0.888; B InfoNCE (lambda 10) dev 0.928 (cross-talk 0.074 / 0.764); and the full Phi-2 arena-B
  probe grid, which selected eqx + logistic, lambda 1, step 200, dev 0.962 (training 0.995). The registered probe
  tuning reruns this grid (deterministic).

## Files

- Code: b31_struct.py, b31_prep.py, b31_arms.py, b31_unit.py, b31_probe.py, b31_lora.py, b31_select.py, b31_indep.py,
  b31_handread.py, b31_verdict.py, b31_worker.sh; b27_common.py and b27_arena/ (reused, verbatim).
- Box: /path/to/work/b31 (feats/, out/, logs/).
- Results: results/b31_verdict.txt (generated), results/b31_scores.json, per-instance outputs in b31_out/ (gzipped).
