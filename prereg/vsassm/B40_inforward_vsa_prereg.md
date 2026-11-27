# B40 pre-registration: a VSA module inside the forward pass (Dhanraj & Eliasmith, EMNLP 2025) against prompting, CoT, LoRA and a discrete pipeline of the same shape

Date: 2026-11-18 (box clock Asia/Shanghai; US Eastern -13 h). Written, with the code, the second scorer and the
verdict generator, before any arm is scored on the dev or test split. What was done before this file: the literature
check (results/b40_literature.txt), the code, the data generator and its hashes, and a smoke run on TRAINING rows only
(pipeline mechanics; recorded in prereg/B40_smoke_record.md when it exists). No dev or test accuracy of any arm has been
seen.

## Why

The co-author's question is what a VSA gives a language model that a table cannot. DE25 (Dhanraj & Eliasmith,
"Improving Rule-based Reasoning in LLMs using Neurosymbolic Representations", EMNLP 2025, arXiv 2502.01657) is the
strongest published case for a VSA INSIDE the forward pass: a linear encoder maps the layer-17 hidden state of
Llama-3.1-8B-Instruct to an HRR vector encoding the problem, a symbolic algorithm runs in VSA space, a linear decoder
writes the HRR answer back into the residual stream (50/50 mix), and the LLM then states the answer itself. DE25
report 15.4x more problems solved and 88.6% lower cross-entropy than CoT prompting and a LoRA, with no non-VSA
pipeline of the same shape. Reading their code (github.com/vdhanraj/Neurosymbolic-LLM, llama/model.py and
vsa_engine.py) shows that the "symbolic algorithm" decodes each operand digit by unbinding and an argmax clean-up,
runs a Python function on the integers and re-encodes the integer answer as an HRR vector. So the VSA is the format of
the read and of the write; the computation is exact and discrete. The literature check found no comparison against a
discrete pipeline of the same shape; two non-VSA systems of that shape exist separately (IGC, arXiv 2501.00684, per-
digit classifiers on Llama-3.1-8B, 0.99 vs 0.70 for an equal-parameter adapter; tape layers, arXiv 2611.00783, gemma-3,
1.00 vs 0.35-0.38 for all-layer LoRA). The authors' own follow-up (thesis ch. 8) shows the read dominates end-to-end
accuracy. B40 therefore holds the layer, the read position, the training data and labels, the Python/table
computation, the mixing rule and the training budget fixed, and changes only VSA vs discrete in the read and the
write.

## Model

Llama-3.1-8B-Instruct, bf16, frozen: Meta's release as mirrored by ModelScope (LLM-Research/Meta-Llama-3.1-8B-Instruct;
the HF mirror stalled); the sha256 of the four safetensors shards are recorded in the smoke record and compared with
the LFS oids of meta-llama/Llama-3.1-8B-Instruct on the Hub (a mismatch, e.g. an earlier upload revision, is
disclosed, not hidden). Prompts use Meta's
Llama-3 chat format as in DE25's code (no date lines). Numbers 0-999, the entity names and the room words are single
tokens (checked by `b40_run.py pools`; any failure is fixed before registration by changing the pool, not the
protocol). "Layer L" is the state entering decoder layer L (HF hidden_states[L]); every module reads and writes there.

## Tasks and splits (b40_data.py; sha256 of the json rows, first 16 hex)

**Task N (DE25 replication).** DE25's Symbolic-Math construction at complexity 2: operands x, y uniform in 1..999 with
x >= y; types (train and test) multiplication (x*y mod 1000), modulo, GCD, LCM mod 1000, x^2 mod y, bitwise AND, XOR,
OR; held-out types (test only, never trained) addition and integer division. Prompt = DE25's code: system prompt, two
exemplars of the same type as user/assistant turns, the question. Splits: train 2,000 per type = 16,000
(679603129fbaa6bb); dev 100 per type = 800 (c7323a009f6a7cba); test 250 per type x 10 types = 2,500
(f70af3c45c8a740d). No (type, x, y) of dev is in train; none of test is in train or dev.

**Task R (relational, "where is it").** A story of facts: each of p people is in one of 4 rooms ("Mary is in the
kitchen."); each of c containers is held by a person ("John has the box.") or inside another container ("The key is in
the box." here 'key' is also a container word); pairings drawn fresh for every instance from fixed pools (20 people,
20 containers, 10 rooms). Question "Where is X?" (X a person: depth 1) or "Where is the X?" (X a container; depth =
hops to the room, 2..). Answer: one room word. Prompt: a system line, two exemplar stories (small, depth 1-3) with
answers, the target story. Sizes: small p = 4, c = 6 (10 facts), large p = 8, c = 12 (20 facts). Splits: train 4,000
per depth 1, 2, 3, small (12,000; add9d8fe8b7f1579); dev 200 per depth 1-3, small (600; b6110da880c854c4); test 200 per
cell for depth 1..5 x size {small, large} (2,000; 7e5c0826b7937ea7). Depth 4-5 (longer chains than trained) and the
large size (twice the facts) are out of the training range. Label-free sub-experiment: train_e2e = 12,000 depth-2
small (35740aea8932caf5); its dev = the 200 depth-2 dev rows; its test = the depth-2 test cells (small, large).

## Arms

All trained arms see the same training rows (and, where they use them, the same symbol labels), batch 16, and are
scored once on the test split with greedy generation (6 new tokens; CoT 512), the answer parsed from the text.

- **BASE**: frozen LLM, the 2-shot prompt.
- **COT**: DE25's CoT prompt (system: explain step by step, end with 'Final Answer: x'; no exemplars; user "Solve the
  following problem step by step: ..."); task R the analogue. 512 new tokens, greedy.
- **LORA**: standard LoRA, rank 6, alpha 12, dropout 0.05, on q, k, v, o, gate, up, down of all 32 layers (15,728,640
  parameters, matched to the VSA module's 16,777,216), cross-entropy of the answer's first token, AdamW, 50-step
  warm-up then cosine to 10%, 3,000 steps (3 epochs of task N, 4 of task R), gradient clip 1.
- **LORA-1L** (task N only; DE25's own "LoRA" baseline): encoder 4096->2048 and decoder 2048->4096 at layer L, last
  position, no symbolic step, always mixed 50/50 (16,777,216 parameters), trained by the same cross-entropy.
- **VSA** (the arm under test).
  - Task N = DE25's NS-LLM, re-implemented in HF from their code: the HRR engine (d = 2,048; unitary
    Gram-Schmidt-orthogonalised atoms for x, n1, n2, the ten problem types and the place base; place 10^k = base bound
    with itself 2^k times; digit i = x^i; 5 places; problem = n1 (*) num(x) + n2 (*) num(y) + type). Linear encoder
    4096->2048 (no bias) on the last-token state at layer L, trained with RMSE to the problem vector (DE25 schedule,
    Adam 1e-3, batch 512). Read: type = argmax dot product over the ten types, gate open iff max > 0.8 (DE25);
    digits = argmax over x^0..x^9 after unbinding the role and the place, kept iff similarity > 0.5. Python computes
    the answer of the decoded type; the answer is re-encoded as num(answer) and decoded by a linear 2048->4096 map
    (no bias), mixed 50/50 at layer L, last position, first pass only. Decoder initialised by DE25's reconstruction
    pre-training or zero (dev choice), then trained by the answer's cross-entropy. One deviation: DE25's code decodes
    the type of the FIRST batch item for the whole batch; B40 decodes per item.
  - Task R: the same block for relations. Linear encoder 4096->2048 applied to the state of every entity mention of
    the story (positions and roles from the template parse, given equally to the discrete arm) and of the question
    entity, trained with RMSE to fixed unitary random codes (50 entities). Memory M = sum over facts a (*) P(b) (P a
    fixed random permutation: located-at is asymmetric). Hop: r = P^-1(q^dagger (*) M), clean-up = argmax over the
    50 codes; repeat from the cleaned code until the clean-up is a room (at most 8 hops; gate closed otherwise). The
    room's code -> linear decoder 2048->4096 (zero init) -> 50/50 mix at layer L, last position.
- **DISC** (the strongest non-VSA arm of the same shape: same layer, same read positions, same labels, same Python
  function / table walk, same mixing and write-back training).
  - Task N: linear heads on the same state (8 trained types; 5 places x 10 digits per operand; 442,476 parameters),
    cross-entropy; argmax; gate open iff the type's softmax probability > 0.8 (the same number as DE25's
    threshold); the same Python function; write-back = sum over the answer's 5 places of a learned per-(place, digit)
    vector (204,800 parameters, zero init). A linear decoder over DE25's place-digit code computes exactly such a sum,
    so the write-back function classes are identical; only the parametrisation differs.
  - Task R: a linear 50-way entity classifier per mention; table a -> b from the argmax reads; the query is the key
    present in the table with the highest question-mention logit (closed world); walk the table until a room (at most 8
    hops; gate closed otherwise); write-back = learned room embedding (10 x 4,096, zero init).
- **Secondary arms (task R, label-free, depth-2 sub-experiment).** Trained by the answer's cross-entropy only, no
  symbol labels; two fixed hops; 50/50 mix at layer L.
  - **E2E-VSA**: encoder 4096->2048 (unit-norm codes) on every mention; M = sum a (*) P(b); two hops with SOFT
    clean-up (softmax(beta * similarity) over a fixed random unitary codebook of 64 codes, beta learnable from 20);
    decoder 2048->4096.
  - **E2E-TAB** (non-VSA, differentiable): soft one-hot reads over 64 slots, an exact outer-product table
    R = sum p_a p_b^T, y = p_q R, z = y R, decoder 64->4096 (524,352 parameters).
  - **E2E-CONST** (floor): one learned 4,096-vector mixed in the same way.
  - Module-use check: every trained E2E arm is also evaluated with the facts' b-codes permuted within the instance
    (pairings destroyed).

## Dev selection (dev split only; seed 0 unless stated; fixed before any dev number is seen)

- **Layer** (per task, shared by VSA, DISC and LORA-1L): candidates {8, 12, 14, 16, 17, 18, 20, 24}; VSA encoder,
  100 epochs of DE25's schedule on train (task R: a fixed 3,000-row subset); the layer with the highest dev read
  accuracy (task N: type and both operands exact; task R: entity clean-up exact over all mentions); ties within 0.005
  -> the layer nearest 17 (DE25's).
- **Read training length**: VSA epochs {100, 300, 1000} (DE25: 1000), DISC epochs {30, 100, 300}; highest dev read
  accuracy; ties within 0.002 -> fewer epochs. Seeds 1 and 2 retrain the reader with the chosen length.
- **Write-back** (VSA, DISC, LORA-1L): DE25's schedule (lr x0.5 at step 100 and 500), 1,000 steps (one epoch of task
  N), base lr {1e-3 (DE25), 3e-4}; task-N VSA also decoder init {DE25 reconstruction, zero}; dev metric = first-token
  argmax accuracy on the whole dev split, every 250 steps; the best (configuration, step) is kept. Seeds 1 and 2 use
  the chosen configuration and keep their own best dev step.
- **LORA**: lr {1e-4, 3e-4, 1e-3}, 3,000 steps, dev every 500 steps; best (lr, step); seeds 1, 2 at the chosen lr,
  own best dev step.
- **E2E arms**: lr {1e-3, 3e-3}, 3,000 steps, dev (depth-2 rows) every 500 steps; best (lr, step).
- **VSA-soft** (task R eval variant of the trained VSA arm): soft clean-up inverse temperature beta {10, 30, 100} on
  cosine similarity, chosen by clean dev accuracy.
- Ties in any dev choice -> the first listed option. The dev record (every dev number) is committed before any test run.

## Test protocol

Seeds 0, 1, 2 for every trained arm (VSA, DISC, LORA, LORA-1L, E2E-*); BASE and COT are deterministic (one run).
Each registered test evaluation runs once on the full test split; no checkpoint, layer or threshold is changed after
any test number is seen. Noise and soft variants are evaluation-only runs of the trained seeds (below). Hardware: idle
A800 GPUs (never GPU 3); PACE only if the box is full.

## Bars and readings (b40_verdict.py)

Accuracy of an arm = mean over items of the seed-mean correctness. Paired 95% CI of a difference: bootstrap over
items (10,000 resamples). "All seeds": the inequality holds for every seed's own accuracy (deterministic arms: one).

**Primary, per task** (task N: the 2,000 test items of the 8 trained types; task R: all 2,000 test items):
- **R1 (does the in-forward-pass VSA module replicate DE25's advantage?)**: for X in {BASE, COT, LORA}:
  VSA - X >= +0.10, CI lower bound > 0, all seeds. REPLICATED if it holds for all three; PARTLY if for one or two;
  NOT REPLICATED if for none.
- **R2 (VSA vs the same-shape discrete pipeline)**, Delta = VSA - DISC: VSA WIN if Delta >= +0.03, CI lower > 0 and
  VSA > DISC in all three seeds; DISC WIN if Delta <= -0.03, CI upper < 0, all seeds; TIE if |Delta| <= 0.02 and the
  CI inside [-0.03, +0.03]; otherwise UNRESOLVED.
- Context check: "the LLM alone fails" holds for a task if BASE <= 0.5 on its primary set.

**Secondary (reported, no headline):**
- S1 per-type (N) and per-cell (R: depth x size) accuracies of every arm, with Wilson intervals.
- S2 held-out types (N: addition, integer division): every arm vs BASE (an arm "hurts" if <= BASE - 0.05 with CI
  upper < 0); gate-open rates of VSA and DISC.
- S3 read accuracy on test (N: type and operands exact, digit error; R: entity reads) of VSA vs DISC; cross-entropy of
  the gold first token (DE25's loss metric) for BASE, LORA, LORA-1L, VSA, DISC.
- S4 noisy reads: Gaussian noise added to the READ only (the states the module reads; the LLM's own stream is
  untouched), relative norm rho in {0.25, 0.5, 1.0, 2.0} per position, fixed noise seed; VSA vs DISC (and task R
  VSA-soft) with R2's bars at each rho.
- S5 parameter counts (exact).
- S6 label-free (task R depth-2 cells): E2E-VSA, E2E-TAB, E2E-CONST; an arm "learns" if it exceeds E2E-CONST by
  >= 0.10 with CI lower > 0 in all seeds; "uses its module" if its accuracy falls by >= 0.10 when pairings are
  destroyed. E2E-VSA vs E2E-TAB read with R2's bars.
- S7 depth / size generalisation (R): per-cell VSA, DISC, LORA, COT.

## Predictions (stated so they can fail)

Task N (trained types, pooled):
- P1 BASE in [0.10, 0.25] (DE25's Standard LLM, same 8 types: 0.176).
- P2 COT in [0.15, 0.40] (DE25: 0.308).
- P3 LORA (all layers, dev-tuned) in [0.20, 0.55]; LORA-1L in [0.10, 0.35] (DE25's LoRA: 0.207).
- P4 VSA >= 0.85 (DE25: 0.904). R1 REPLICATED.
- P5 DISC >= 0.85; R2 = TIE.
- P6 DISC's test read accuracy (type and both operands exact) >= VSA's.
- P7 Held-out types: BASE >= 0.90 on addition; VSA's gate stays closed on >= 80% of held-out items; DISC's gate opens
  on >= 50% of them and DISC "hurts" on at least one held-out type.
- P8 VSA's mean first-token cross-entropy is >= 80% below BASE's.
Task R:
- P9 BASE pooled in [0.30, 0.65], falling with depth; depth 1 >= 0.80.
- P10 COT pooled in [0.50, 0.90].
- P11 LORA >= 0.90 on depth 1-3 small; <= 0.70 on depth 5; pooled in [0.60, 0.90].
- P12 VSA >= 0.95 and DISC >= 0.95 in every cell; R1 REPLICATED; R2 = TIE.
Both tasks:
- P13 Noise: at every rho, VSA - DISC <= +0.02 (no VSA robustness edge; most likely DISC >= VSA); task R VSA-soft
  within 0.02 of VSA-hard at every rho.
- P14 Label-free: E2E-VSA does not learn (fails S6's bar in >= 2 of 3 seeds); E2E-TAB learns in >= 2 of 3 seeds.
- P15 Second scorer: 0 gold mismatches and 0 correctness mismatches on non-CoT files; <= 1% on CoT files.
- P16 Total GPU-hours <= 40.

## Checks from other angles

- Second scorer (b40_check.py): rebuilds every gold answer from the prompt TEXT (regex + own arithmetic / own chain
  walk), re-normalises every saved output with its own rules, and compares; full test split, every file.
- Hand-read: 20 test errors per trained arm and task (seed 0), sampled with a fixed seed, classified (read error /
  computation path / write-back not followed / format); every VSA-vs-DISC disagreement on task N inspected for its
  source (read vs write).
- Full test sets only; per-cell numbers are reported beside the pooled ones, never instead.
- Module diagnostics logged on test: decoded type/operands/answer and gate per item (N); walked answer, hops and gate
  per item (R); so "the module computed the right answer but the LLM did not say it" is separable from read errors.

## What the paper may say (decided now)

- R1 REPLICATED and R2 TIE: an in-forward-pass module of DE25's shape gives a current 8B model large gains over
  prompting, CoT and an equal-parameter LoRA, but a discrete pipeline of the same shape (classifier heads, exact
  program, learned embedding) matches it with far fewer parameters; the gain comes from the exact computation and the
  learned write-back, not from the VSA format. Secondary results (noise, label-free, held-out gating) are then the
  only candidates for a VSA-specific property and are reported as measured.
- R2 VSA WIN: the VSA module beat the same-shape discrete pipeline by Delta [CI] on that task, with the attribution
  (read accuracy vs write-back) from S3 and the hand-read.
- R2 DISC WIN: the discrete pipeline beat the VSA module (Delta [CI]); the VSA format costs accuracy here.
- R1 NOT REPLICATED / PARTLY: the numbers, per baseline, and which baseline closed the gap.
- In every case the paper must not state that the VSA is necessary for the in-forward-pass gain unless R2 = VSA WIN on
  both tasks, and must not generalise beyond Llama-3.1-8B-Instruct and these two synthetic task families.

## Compute

A800 box, idle GPUs only, never GPU 3. Budget cap 60 GPU-hours (expected about 25-35). Per-run wall time is appended to
logs/gpu_hours.txt by b40_run.py; worker occupancy to logs/gpu_ledger.txt. No paid API.
