# B35 pre-registration: can a language model answer from VSA bundles handed to it as a few soft input positions, and does the VSA structure beat learned memory tokens at an equal context budget? (pilot)

Date: 2026-11-18, about 09:20 UTC (Asia/Shanghai box clock +8 h; US Eastern -5 h). Written after a training/dev
pilot (recorded below) and before any B35 test run: no arm has been trained for the test, the test split has not
been scored by any arm, and only the CPU external decoding has been computed on the dev split. Priors written before
any GPU run: prereg/B35_priors_before_pilot.md (commit c0c579a).

## Why

A table hands its facts to a language model as text, so every fact costs context positions (here about 10 tokens per
fact). A VSA superposes many key-value bindings into one fixed-width vector: one 2,048-coordinate MAP bundle holds 64
facts at >= 0.999 single-hop retrieval by exact unbinding (theory and b35_ext). If a small trained projector maps
bundle chunks into the LM's input-embedding space as B "soft tokens", the LM might answer from many facts at a small,
fixed context budget. The question is whether the VSA structure is what helps: at an equal number of memory
positions, does the VSA memory let a LoRA-tuned LM answer more facts than equally engineered learned memory tokens
without VSA structure (gist / ICAE / xRAG-style compression), than text cut to the same budget, and how close does it
come to the full text and to exact look-up?

## Task (b35_data.py)

Synthetic entity-attribute facts in natural language. An instance holds N facts about E = N / 4 invented people
(names from one fixed pool of 600 syllable names shared by all splits; facts are fresh random draws per instance, so
nothing can be memorised across instances). Each person has three attributes drawn without replacement from six
(city, color, pet, sport, instrument, food; 32 single-word values each, uniform) and a manager (another person of the
instance). Facts are sentences in random order ("The city of Velor is Lisbon." / "The manager of Velor is Quenna.";
about 10 Qwen3 tokens each: 79 / 319 / 1,266 tokens of text at N = 8 / 32 / 128). Each instance carries one single-hop
question ("What is the city of Velor?") and one two-hop question ("What is the pet of the manager of Velor?"; the
manager always has that attribute). Chance for a reader that knows the attribute's 32 values: 1/32 = 0.031.
Splits: train = a fresh stream of instances (split name 'train', random indices); dev = 300 instances per N; test =
600 instances per N; N in {8, 32, 128}. Every arm is scored on the same test questions (paired).

## Arms (b35_run.py; Qwen3-1.7B, h = 2,048, bf16; LoRA rank 64, alpha 128, on q/k/v/o/gate/up/down)

The memory sits between "Memory:" and "\nQuestion: ...\nAnswer:"; the answer is generated greedily up to a newline
(6 tokens max) and scored by exact string match (one trailing period and spaces stripped).

Question-agnostic memory at an equal budget of B positions (B in {1, 2, 4, 8}):
- **VSA (V).** The parsed facts (exact, from the generator) as one MAP bundle m = sum_f C[e] * C[r] * P(C[v]) /
  sqrt(N), C[s] in {-1,+1}^(B*2048) (random codes), P a fixed random permutation inside each 2,048-coordinate chunk
  (the order-preserving binding of B28). Chunk b, through a shared projector (LayerNorm -> Linear(h, 2h) -> GELU ->
  Linear(2h, h) -> non-affine LayerNorm x learned gain + learned position vector), is soft position b. The projector
  cannot see the question; the LM has to unbind.
- **Learned memory tokens (L).** The same parsed facts; each fact (e, r, v) -> MLP of the three symbols' standardised
  mean LM input embeddings -> a Perceiver with B learned latents (2 blocks of cross-attention to the facts,
  self-attention, MLP; width h) -> the same output normalisation. No VSA structure; trained end to end with the LM.
- **Text cut to the budget (T-B).** The facts' text truncated to its first B tokens (B <= 8 < one fact).
References and ceiling:
- **Full text (T-full).** All N facts as text (79-1,266 tokens), the same LoRA.
- **Full text zero-shot.** The frozen LM, same prompt, 16 generated tokens, scored leniently (gold as a whole word in
  the first line) because the untrained model answers in sentences; no bar.
- **DICT.** Exact look-up: the question parsed by its two templates, followed through a dictionary of the facts
  (b35_data.dict_answer); 1.000 on dev and test by construction (checked).
Secondary:
- **VSA-q (Vq), single hop.** The bundle unbound OUTSIDE the LM with the parsed question key (P^-1(m * C[e] * C[r])),
  chunks through the same projector; the LM only names the retrieved value. What the LM can do when the VSA does the
  retrieval. Its non-VSA comparator is DICT.
- **VSA-ext (no LM).** Exact unbinding and clean-up of the same bundles (b35_ext.py; two hop: clean-up among the 600
  names in between) against the Gaussian theory p = int phi(t) Phi(t + sqrt(D/N))^(V-1) dt (Frady, Kleyko & Sommer
  2018), D = B * 2,048, V = 32 (600 for the middle hop).
- **From-scratch control (CPU, b35_scratch.py).** A 4-layer transformer (width 256) trained from scratch to answer
  from B = 4 memory vectors plus question tokens (192-way read-out), N in {8, 32}: vsa (bundle chunks, d = 256 per
  chunk), vsakey (bundle chunks plus the parsed first-hop key C[e] * C[r] as 4 more vectors), vsaq (unbound outside,
  positive control), lmt (Perceiver over learned symbol embeddings). Separates "a transformer reader cannot learn to
  unbind" from "a pretrained LM with LoRA does not".

Equal engineering. V and L get the same parsed facts, the same B output positions in the LM's embedding space with
the same output normalisation, the same LoRA, optimiser (AdamW, betas 0.9/0.99, no decay; 100-step warm-up, cosine to
10%), learning rates (LoRA 2e-4, memory module 2e-3), steps (8,000), batch (32), training stream (the same instances
per seed: half single-hop, half two-hop questions, N drawn uniformly from {8, 32, 128}), and the same number of
configurations tried in the dev pilot (below). L has 117.5 M trainable memory parameters, V 16.8 M (both + 69.7 M
LoRA); this favours the baseline.

## Dev pilot (train and dev only; done before this file)

All dev runs at B = 4 (memory arms), evaluated on dev subsets (100 instances per N) during training and on the full
dev split (300 per N) at the end. Box: /path/to/work/b35/out/dev_*; copies in b35_out/dev/.

Every number below is final-dev accuracy (300 instances per N) unless marked "subset" (100 per N, during training).
Single hop at N = 8 / 32 / 128, then two hop.

| run (B = 4) | steps | single hop | two hop | note |
|---|---|---|---|---|
| V, random codes, MLP projector, LoRA = memory lr 2e-4 | 2,000 | .023 / .033 / .040 | .023 / .037 / .033 | 14 distinct single-hop answers: one fixed value per attribute (the prior) |
| V, embedding codes, same | 2,000 | .017 / .033 / .033 | .030 / .033 / .040 | the same fixed answers |
| L, Perceiver 2 blocks, same | 2,000 | .017 / .020 / .030 | .033 / .030 / .047 | the same |
| L, Perceiver 4 blocks, same | stopped at 1,300 | subset .05 / .04 / .04 at 1,000 | | flat loss |
| Vq, memory lr 2e-4 | stopped at 850 | subset .02 / .03 / .05 at 500 | - | flat loss 1.43 |
| **Vq, memory lr 2e-3** | 2,000 | **1.000 / .840 / .333** | - | subset .07/.06/.06 (500), .16/.22/.08 (1,000), .99/.69/.29 (1,500) |
| Vq, memory lr 5e-3 | stopped at 1,000 | subset .03 / .04 / .02 | - | memory gradient 0.002: dead projector |
| T-B (text cut to 4 tokens) | 2,000 | .027 / .030 / .033 | .020 / .030 / .040 | |
| T-full, LoRA | 300 | 1.000 / 1.000 / 1.000 | 1.000 / .997 / .993 | 1.00 on the subsets from step 100 |
| T-full, LoRA, 2,000 steps | stopped at 50 | | | 7.7 s per step at four lanes; replaced by the 300-step run |
| T-full zero-shot (strict, 6 tokens) | 0 | .080 / .120 / .013 | .007 / .000 / .000 | answers in sentences ("The instrument of Kaitre ..."), hence 16 tokens and lenient scoring for the test |
| V, memory lr 2e-3, LoRA 2e-4 (**base**) | 8,000 | .037 / .037 / .060 | .030 / .047 / .027 | memory gradient norm ~5e-4 from step 1,000 |
| V, same, question before memory | 8,000 | .047 / .033 / .027 | .030 / .043 / .030 | |
| V, same, LoRA 5e-4 | 8,000 | .050 / .040 / .040 | .033 / .043 / .030 | |
| V, embedding codes | 8,000 | .050 / .037 / .023 | .033 / .037 / .027 | |
| L, memory lr 2e-3, LoRA 2e-4 (**base**) | 8,000 | running at registration; subset .04 / .04 / .00 at 7,000 | subset .00 / .03 / .02 | final appended below |
| L, same, question before memory | 8,000 | .057 / .043 / .050 | .040 / .040 / .020 | |
| L, same, LoRA 5e-4 | 8,000 | .053 / .037 / .040 | .033 / .043 / .030 | |
| L, 4 blocks | 8,000 | running at registration; subset .00 / .02 / .00 at 6,000 | subset .02 / .01 / .04 | final appended below |

Every subset evaluation of every question-agnostic run (V and L, 1,000 to 8,000 steps) lay between 0.00 and 0.09.

From-scratch control (CPU, B = 4, 6,000 steps, dev, single hop N = 8 / 32; two hop): vsa .040 / .030 (.037 / .027);
vsakey .037 / .043 (.027 / .037); vsaq 1.000 / .603 (two hop not addressed by its key); lmt .243 / .103
(.227 / .100), still rising. External decoding on dev (CPU): single hop 1.000 wherever N / B <= 64, .967 at N = 128,
B = 1 (theory .958); two hop .783 there (theory .770), >= .99 elsewhere; embedding codes within .03 of random codes.

Not run, decided after the 2,000-step results (all at the floor): linear projector, frozen LM (LoRA rank 0), one-block
Perceiver. An ICAE-style arm (the same LM encodes the facts' text into B memory slots; arm `icae`, CPU smoke passed)
was implemented but not run: one step costs as much as a full-text step (about 8 s at four lanes), 8,000 steps about
18 GPU-hours per run.

Selection rule (fixed before the 8,000-step runs ended): per arm, the configuration with the highest final dev
single-hop mean; if every configuration of the arm is at the floor (max <= 0.10 at every N), the base configuration.
Every configuration of both arms is at the floor, so both use the base configuration. The rule was applied, and this
file committed, while the last two L dev runs (base and 4 blocks) were still running; their last subset
evaluations were at chance; their finals are appended below whatever they show and do not change the choice.
The memory learning rate 2e-3 is used for every soft arm because it is the only one under which any soft arm (Vq)
learned at all.

## Registered configuration

- V: random codes, MLP projector, memory before the question, LoRA lr 2e-4, memory lr 2e-3, 8,000 steps.
- L: Perceiver with 2 blocks, the same optimiser settings and steps.
- Vq: as V, single-hop questions only in training and test.
- T-B: LoRA lr 2e-4, 2,000 steps. T-full: LoRA lr 2e-4, 300 steps. T-full zero-shot: 16 generated tokens.
- Commands: q/test.q on the box (`b35_run.py ... --eval_splits dev,test --out out/test_<arm>_B<B>_s<seed>`), the same
  lines in PACE list files. From-scratch control: `b35_scratch.py --arm {vsa,vsakey,vsaq,lmt} --B 4 --seed {0,1}
  --steps 6000 --splits dev,test`. External decoding: `b35_ext.py --split test`.

## Test protocol

- The test split (600 instances per N) is scored only by the registered test runs, once, at their last step. No
  checkpoint selection, no early stopping, no change after any test number is seen.
- Runs (all on the training stream of their seed): V and L at B in {1, 2, 4, 8}, seeds 0 and 1, 8,000 steps (16 runs);
  Vq at each B, seed 0, 8,000 steps (4); T-B at each B, seed 0, 2,000 steps (4; it sees less than one fact, its
  accuracy is fixed by that; dev 0.027-0.040); T-full, seed 0, 300 steps (it reaches 1.000 on dev by step 100; 300
  steps of 32 sequences of up to 1,300 tokens cost about as much GPU time as 2,000 soft-memory steps); T-full zero-shot.
  External decoding on test (CPU). From-scratch control on test: the four arms at B = 4, seeds 0 and 1, 6,000 steps.
- Hardware: idle A800 GPUs (never GPU 3) and PACE (L40S / A100 / H200, bf16). A run is the same command wherever it
  runs; the machine of every scored run is listed in the verdict; if both machines finish the same run, the first
  finished copy is scored and the other is reported as a reproducibility check.

## Bars and readings (b35_verdict.py)

Accuracy of a cell (arm, B, N, question type) = mean over the cell's 600 test questions and the arm's seeds. Paired
95% CIs: bootstrap over the 600 instances of N (10,000 resamples; an instance's score per arm = its mean over seeds).
Delta = V - L.

Per cell (single hop is primary; two hop is read the same way and reported, no headline):
- FLOOR: max(V, L) <= 0.10. REFERENCE: min(V, L) >= T-full - 0.02. Neither is eligible.
- VSA WINS: Delta >= +0.05, CI lower bound > 0, and V > L in both seeds.
- VSA LOSES: Delta <= -0.05, CI upper bound < 0, and V < L in both seeds.
- TIE: |Delta| <= 0.02 and the CI inside [-0.05, +0.05]. Otherwise UNRESOLVED.
Headline (single hop): **SUPPORTED** (the VSA structure helps at an equal budget) if >= 2 eligible cells are VSA WINS
and none is VSA LOSES; **REFUTED** (learned memory tokens do better) if >= 2 are VSA LOSES and none VSA WINS; **NULL**
if every eligible cell is a TIE; **NO READING** if no cell is eligible (both arms at the floor or at the reference
everywhere); MIXED otherwise.
Secondary readings:
- Soft memory vs text at an equal budget: X - T-B >= +0.10 with CI lower bound > 0 -> "beats text at the budget"
  (counted for X = V and X = L over the 12 single-hop cells).
- Capacity: VSA-ext within max(0.03, 3 SE) of the theory wherever the theory is in [0.05, 0.995] (single and two hop).
- LM read-out: in each cell, "the LM reads what the bundle holds" if V >= VSA-ext - 0.05, else "short of it"; the
  same for Vq ("the LM names what unbinding retrieves" if Vq >= VSA-ext - 0.05). Reported with N/B (facts per
  position), next to the theory.
- From-scratch control: "a from-scratch reader learns to read the bundle" if scratch-vsa single-hop >= 0.5 at N = 8
  (mean of two seeds); the same for vsakey and lmt.
Reading for the paper (decided now): SUPPORTED -> a candidate result for the VSA paper (a full experiment would be
needed: second model, real facts). REFUTED / NO READING / NULL -> the paper may at most say that, in a pilot,
superposed VSA memory handed to a LoRA-tuned 1.7B LM as soft tokens was not read at any budget (with the numbers),
while the same bundles decode exactly outside the LM and the LM names values once the unbinding is done for it; the
VSA's value stays on the side of exact external read-out, consistent with the paper's existing results.

## Predictions (stated so they can be wrong; written after the dev pilot)

- P1 DICT 1.000 on test (by construction).
- P2 External decoding within tolerance of the theory in every informative cell, single and two hop; single hop
  >= 0.99 everywhere except N = 128, B = 1 (0.93-0.98).
- P3 T-full >= 0.99 single hop and >= 0.98 two hop at every N.
- P4 T-B <= 0.08 in every cell.
- P5 V <= 0.10 in every cell, both question types (all at the floor).
- P6 L <= 0.10 in every cell, both question types, except possibly B = 8 with N = 8 (<= 0.25).
- P7 Single-hop headline: NO READING (at most one eligible cell).
- P8 Neither soft arm beats text cut to the budget in any cell (0/12 for V, <= 1/12 for L).
- P9 V is "short of" what its bundle holds in every cell (by >= 0.85 wherever exact decoding is >= 0.95).
- P10 Vq >= 0.95 at N = 8 for every B.
- P11 Vq at N = 32 >= 0.80 for B >= 4; at N = 128 Vq is short of exact decoding by >= 0.2 for B <= 4; Vq rises
  with B at N = 128.
- P12 From scratch: vsa and vsakey <= 0.10 single hop at N = 8; vsaq >= 0.95 at N = 8; lmt >= 0.15 at N = 8 and
  above vsa by >= 0.10.
- P13 Zero-shot full text (lenient) <= 0.6 single hop at every N.
- P14 Second scorer: 0 mismatches.
- P15 Total GPU-hours (dev + test) <= 25.

## Checks from other angles

- **Second scorer** (`b35_check.py`): regenerates the test questions from b35_data, recomputes each gold answer by the
  DICT parse of the question text against the instance's facts, re-normalises each saved prediction with its own
  rule, and compares question sets, per-question correctness and per-cell accuracy with every run's summary.
  Bar: 0 mismatches.
- **Hand-reading** (`b35_handread.py`): at least 10 wrong single-hop answers of V and 10 of L (B = 4, N = 8), 10 of Vq
  at N = 128, and 10 questions where V and L disagree if any; error kinds counted (a value of another person with the
  same attribute; another value in the memory; right type, not in the memory; other); a constant answer per
  attribute (the prior) is reported as such (distinct predictions per attribute).
- **Recomputation**: the verdict recomputes every accuracy from the saved per-question outputs, not from the
  summaries.
- **Literature** (after the verdict): supports / conflicts / no precedent, for: soft-token compression (gist tokens,
  ICAE, xRAG, AutoCompressor, 500xCompressor, RMT, "Cramming 1568 tokens into a single vector"), knowledge tokens
  (KBLaM, Larimar), HRR/VSA/TPR inside networks (Hrrformer, TP-Transformer, linear-transformer fast weights), and bundle
  capacity theory (Plate 1995; Frady, Kleyko & Sommer 2018; Thomas, Dasgupta & Rosing 2021).

## Compute

- A800 box: idle GPUs only (claim = < 500 MiB used, no compute process, re-checked after 20 s; lock in the claim
  directory shared with the B29-B31 wrappers; b35_worker.sh), never GPU 3, never another user's process. PACE: free
  account first, paid allowed, always sbatch (b35_pace.sbatch).
- GPU-hours = worker occupancy (logs/gpu_ledger.txt on the box, logs/gpu_ledger_pace.txt on PACE); smokes, dev pilot
  and stopped runs included. Dev pilot so far: about 5 A800 GPU-hours (GPU 2 from 06:48 UTC, GPU 0 07:40-09:10 UTC; the last two dev runs end about 09:40).
- Projection for the test: 20 soft-memory runs x 8,000 steps (about 0.35 GPU-hours each at four lanes per GPU), T-B
  and T-full about 1.5 GPU-hours, evaluation included: about 9 GPU-hours. Registered cap 25 GPU-hours in total (the
  coordinator allows up to 1.5 x if that shortens the wall clock; any use beyond 25 goes in the ledger with its
  reason). If the projection passes the cap: drop seed 1 at B = 1 and B = 2 first.

## Files

- Code: b35_data.py, b35_common.py, b35_run.py, b35_ext.py, b35_scratch.py, b35_check.py, b35_handread.py,
  b35_verdict.py, b35_worker.sh, b35_pace.sbatch.
- Box: /path/to/work/b35 (out/, logs/, q/). PACE: /path/to/pace-scratch/b35.
- Results: results/b35_verdict.txt (generated), results/b35_scores.json; per-question outputs in b35_out/ (gzipped).

## Appended after registration (no design change): the last two dev runs

Finished at 09:21 UTC, after the registration commit (186ed10), as announced in the dev record; final dev accuracy,
single hop N = 8 / 32 / 128, then two hop:
- L, memory lr 2e-3, LoRA 2e-4 (base), 8,000 steps: .037 / .023 / .040; .033 / .033 / .027.
- L, 4 blocks, same, 8,000 steps: .043 / .043 / .040; .037 / .040 / .017.
Both at the floor, like every other question-agnostic configuration; the registered choice (base configurations) stands.
Dev pilot GPU time (ledger): GPU 2 06:48-09:21 UTC (2.55 h), GPU 0 07:40-09:10 UTC (1.50 h): 4.05 A800 GPU-hours,
plus CPU-only smokes and controls.
