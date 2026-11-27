# B65 pre-registration: the in-forward-pass vector module against the same-shape discrete module on task N, with every write-back learning rate tuned on dev and at two read positions

Date: 2026-11-23 (machine clock Asia/Shanghai, about 13:05; US Eastern 2026-11-23 00:05). Written, with the code
(b65_run.py, b65_plan.py, b65_worker.sh, b65_verdict.py), before any B65 development or test run; committed, then
OpenTimestamps-stamped, before the first dev run. The paper is not edited by B65.

## What has been seen before this file (disclosed)

- **B40 (task N, last-token read at layer 20, write-back lr from {1e-3, 3e-4} only)**: test, trained types, 3 seeds:
  VSA (DE25's published thresholded read) 0.898, DISC 0.930, VSA - DISC -3.2 [-4.6, -1.9] (DISC WIN); dev curves and
  choices in prereg/B40_dev_record.md (DISC lr 1e-3 still rising at step 1,000: .552 .901 .922 .941; lr 3e-4 0.411).
  B40 test S3: VSA read exact 0.888, module answer 0.908; DISC read exact 0.952, module answer 0.964.
- **B40 exploratory 1 / B55 (best-match argmax read, the same trained VSA seeds, write-back trained on thresholded
  reads)**: test 0.960, read exact 0.970, VSA-argmax - DISC +3.0 [+2.1, +4.0]. That cell was not blind (registered
  after B40's verdict was read); B55's hand-read found half of the sampled disagreements were DISC write-backs the LLM
  did not follow.
- **The clean room (another agent, own re-implementation from the manuscript text, layer 17, its own 2,000 test rows,
  one seed; scratchpad cleanroom3_fp/: prereg_note.txt, my_scores.json, cr3_fp.py, which I have read)**: operand-
  position reads (VSA enc(h_x) + enc(h_y); DISC digits on [h_x; h_y], type on h_x + h_y): test VSA published 0.9935,
  best match 0.9985, DISC 0.984 (lr 1e-3), published - DISC +0.95 [+0.30, +1.60]; last-token reads: 0.930 / 0.984 /
  0.947, published - DISC -1.7 [-2.95, -0.45], best match - DISC +3.7. Dev-only write-back lr sensitivity (test not
  re-scored): DISC 0.544 / 0.986 / 0.9975 (ops) and 0.529 / 0.956 / 0.9675 (last) at 3e-4 / 1e-3 / 3e-3; the vector
  arms flat within 0.008.
- **B65 runs before this file**: none on dev or test. The self-test (below) runs on TRAINING rows only.
- The test split is B40's task-N test split (2,500 items: 2,000 of the 8 trained types, 250 addition, 250 integer
  division). These items have been scored by B40's / B55's modules; no B65 module has been trained or scored on them.
  So the direction of the last-token cells is partly known from B40 / B55 / the clean room, and the operand-position
  cells are known only from the clean room's one seed on other items with another implementation.

## Why (referee)

The referee calls the task-N comparison unsettled: (1) the discrete write-back lr was chosen from {1e-3, 3e-4} only,
and the clean room found 3e-3 much stronger for DISC on dev; (2) the sign of the published-read gap flips with the read
position (last token: vector behind, -3.2 here, -1.7 in the clean room; operand tokens: vector ahead +0.95, one seed);
(3) the best-match no-noise cell was not blind. Question: with both modules properly tuned, and at both read positions,
which module wins on task N under the published read and under best match?

## Fixed (B40's protocol, unchanged)

Llama-3.1-8B-Instruct (B40's ModelScope copy, /path/to/work/b40/llama31_8b_instruct), bf16, frozen; B40's task-N data
generator, splits and 2-shot prompts (b40_data.py, md5 67b29c80...; b40_core.py 835e9f4f...; b40_run.py fef8081d...,
imported unchanged); "layer L" = the state entering decoder layer L; DE25's HRR engine (b40_core.NumEngine: d = 2,048,
5 places); the Python function of the decoded type; the type gate (VSA: max dot product with the type atoms > 0.8;
DISC: softmax probability of the type > 0.8); write-back = 0.5 w + 0.5 h at layer L, LAST prompt token, first pass,
where the gate is open (b40_core.Injector unchanged); write-back training = B40's (AdamW, weight decay 0, batch 16,
clip 1, 1,000 steps, DE25's schedule x0.5 at steps 100 and 500, cross-entropy of the answer's first token, dev =
first-token argmax accuracy on the 800 dev rows every 250 steps, the best step kept); readers = B40's (VSA: linear
encoder 4096 -> 2048 without bias, RMSE to the problem vector; DISC: linear heads, cross-entropy; Adam, batch 512,
DE25's epoch schedule x0.5 at 50, x0.5 at 100, x0.1 at 250, x0.4 at 500); VSA decoder 2048 -> 4096 initialised by
DE25's reconstruction pre-training (B40's dev choice; tied with zero init on B40's dev; not re-gridded); test =
greedy, 6 new tokens, B40's parser, batch 32, longest prompt first; scoring = exact match of the parsed number.

## Arms

- **VSA-PUB**: B40's VSA module with DE25's published read (a digit is kept only if its best similarity > 0.5,
  otherwise 0). Its write-back is trained with the published read.
- **VSA-BM**: the same encoder architecture with best-match read (argmax over the ten digit atoms, no threshold; B55's
  N-ARG). Its reader epochs are chosen by its own dev read and its write-back is TRAINED WITH ARGMAX READS (B40 / B55
  reused the thresholded write-back), so this cell is new and blind.
- **DISC**: B40's discrete module of the same shape (linear type and per-place digit heads, argmax, the same Python
  function, learned per-(place, digit) write-back embedding, zero init).

## Read positions

- **(a) LAST** (B40's design): every module reads the state at the last prompt token at layer L = 20 (B40's dev scan,
  the same procedure; not re-scanned). The modules are b40_core.VSAN / DiscN unchanged.
- **(b) OPS** (the clean room's arm A, implemented in the pipeline: b65_run.VSANOps / DiscNOps): the module reads the
  states at the two operand tokens of the target question (x, y; each of 0-999 is one Llama-3 token; positions from
  the template, asserted to decode to the operand on every training row). VSA: v = enc(h_x) + enc(h_y), one shared
  linear encoder of B40's shape (summed after the encoder), the same RMSE target. DISC: type head on h_x + h_y (4096 ->
  8), digit heads on the concatenation [h_x; h_y] (8192 -> 2 x 5 x 10). Write-back unchanged (last prompt token).
  For the VSA decoder's reconstruction init the target is the state at the write position (last prompt token, B40's
  cache) and the input is the encoder's summed vector. Layer L_b chosen on dev (below).

## Dev selection (dev split only; seed 0; mechanical, b65_plan.py)

1. **Layer (OPS only)**: B40's rule: VSA encoder, lr 1e-3, 100 epochs, at each of {8, 12, 14, 16, 17, 18, 20, 24};
   highest dev read exact (type and both operands, published read); ties within 0.005 -> the layer nearest 17, then
   the lower. LAST keeps layer 20.
2. **Readers** (per arm and position): reader lr in {1e-3 (DE25 / B40), 1e-4 (the clean room's second value)} x epochs
   (VSA {100, 300, 1000}, DISC {30, 100, 300}; one run per lr with snapshots at those epochs, which equals separate
   runs because the schedule depends on the epoch only); metric = dev read exact under the arm's own read rule
   (VSA-PUB thresholded, VSA-BM argmax, DISC argmax); ties within 0.002 -> lr 1e-3 first, then fewer epochs. VSA-PUB
   and VSA-BM share the encoder runs but choose separately.
3. **Write-back learning rate** (EVERY module, both positions): grid {3e-4, 1e-3, 3e-3, 1e-2}; metric = best dev
   first-token accuracy over steps 250-1,000; ties (equal dev accuracy) -> the lr nearest 1e-3 in |log10|, then the
   smaller. **Edge rule**: if the chosen lr is the grid's smallest (largest) value, the next value of the ladder
   ... 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1 below (above) it is added and the choice repeated, until the
   chosen value is interior, at most 3 extensions per side (if still on the edge after 3, reported as "edge", not
   hidden). A tie between an edge value and an interior value is resolved by the tie rule (nearest 1e-3), so only a
   strict edge optimum triggers an extension.
4. **Seeds 1, 2**: the reader retrained with the chosen (lr, epochs) and the seed; the write-back with the chosen lr and
   the seed, keeping its own best dev step (B40's rule). Seed 0 = the chosen dev run.
5. The dev choices (out/choices.json and every train.json / reader.json) are copied into the repository and committed
   before any test run.

## Test protocol

18 registered test evaluations (3 arms x 2 positions x 3 seeds), each on the full test split (2,500 items), run once.
Primary items = the 2,000 trained-type items; held-out types reported as secondary. No checkpoint, layer, threshold or
learning rate is changed after any test number is seen. Determinism: the three seed-0 LAST evaluations are re-run once
into preds_rerun/ (a check, not a result).

## Decision rule (b65_verdict.py)

Primary, four cells: for each position in {LAST, OPS} and each read in {PUB, BM}: Delta = acc(VSA-read) - acc(DISC),
accuracy = mean over the 3 seeds and the 2,000 items. 95% interval = two-level bootstrap (10,000 draws): the seeds of
each arm drawn with replacement (independent trainings, not paired), then the items drawn with replacement once per
draw and shared by both arms; percentile interval. Labels (the paper's four-way rule): **AHEAD** if the interval is
above 0; **BEHIND** if below 0; **EQUIVALENT** if it contains 0 and lies within +/-2 points; otherwise
**INCONCLUSIVE**. Beside it (not the rule): B40's one-level paired item bootstrap on seed means, and whether the
inequality holds in every seed pair.

Answer to the question: the per-cell labels. "The vector module wins" under a read only if AHEAD at both positions
under that read; "the discrete module wins" under a read only if BEHIND at both; otherwise the answer names the
position.

Secondary (same interval, no headline): VSA-BM - VSA-PUB at each position; OPS - LAST for each arm; each arm at its
dev-best position (the position with the higher chosen dev accuracy; ties -> LAST) against DISC at its dev-best
position; per-type accuracies; held-out accuracy and gate-open rate; read exact, module answer right and gate open,
"followed" (correct given a right, open module answer); first-token cross-entropy.

## Predictions (stated so they can fail)

- Q1 (dev) DISC's chosen write-back lr is >= 3e-3 at LAST and at OPS.
- Q2 (dev) for each VSA arm and position, the best dev accuracy spans <= 0.010 over the four first-grid lrs (the
  vector write-back is insensitive to its lr), so the tie / nearest rule keeps the VSA choices interior; no VSA arm
  needs an extension.
- Q3 (dev) the OPS layer is 17 (several layers tie within 0.005 at the top), and the OPS dev read exact is >= 0.98 for
  all three reads; at LAST the lr-1e-3 readers reproduce B40's dev reads (VSA published 0.865 / 0.894 / 0.896 at 100 /
  300 / 1000 epochs, DISC 0.890 / 0.929 / 0.946 at 30 / 100 / 300) within 0.005, and lr 1e-4 does not beat lr 1e-3 by
  more than 0.002 for the VSA encoder.
- Q4 (test) LAST, PUB - DISC: BEHIND, point estimate in [-7, -3].
- Q5 (test) LAST, BM - DISC: point estimate in [-0.5, +2.5]; label not BEHIND.
- Q6 (test) OPS, PUB - DISC and BM - DISC: both EQUIVALENT, |point| <= 1.
- Q7 (test) every arm is at least as accurate at OPS as at LAST; VSA-PUB gains >= +5 points.
- Q8 (test) DISC at LAST >= 0.94 (B40 at lr 1e-3: 0.930).
- Q9 (checks) second scorer (b40_check.py) 0 gold and 0 correctness mismatches on all 18 files; plain-loop scorer
  agrees exactly; the three determinism re-runs reproduce their predictions on 2,500 / 2,500 items; every (arm,
  position) dev optimum is interior after at most 3 extensions.
- Q10 compute <= 12 GPU-hours.

## Checks from other angles

- Second scorer: b40_check.py (gold rebuilt from the prompt text with its own arithmetic, outputs re-normalised by its
  own rules) on every test file.
- Plain-loop scorer inside b65_verdict.py (pred == gold, independent of the 'correct' field).
- Dev-optimum-interior check: the chosen lr of every (arm, position) is not an endpoint of its final grid (reported per
  cell with the whole dev curve).
- Determinism: the three seed-0 LAST test evaluations re-run on the same GPU type; item-level agreement of predictions.
  Reader reproduction of B40's dev read numbers (Q3), and the LAST VSA-PUB / DISC write-back curves at lr 1e-3 / 3e-4
  against B40's dev record (same code path, same RNG order; differences reported, not hidden).
- Sample reading by the AI-assisted process (fixed seed, seed-0 files): for each primary cell, up to 12 test items
  where the two arms disagree, classified as read error, write-back not followed, or gate; written to
  results/b65_handread.txt.

## What the paper may / must say (decided now)

- The four labels with their two-level intervals are reported together, naming the read position and the read rule of
  every number. The paper must not state a task-N winner without naming the read position if the labels differ by
  position.
- If LAST PUB is BEHIND and OPS PUB is EQUIVALENT or AHEAD: the published-read deficit of the vector module is a
  property of reading the last-token state, not of the vector format; the paper must say so beside B40's -3.2.
- If BM is AHEAD at one position only: the best-match lead holds at that read position only; it may not be called an
  advantage of the vector format in general.
- If every cell is EQUIVALENT: with both modules tuned, the vector format adds nothing measurable over a same-shape
  discrete module inside the forward pass on task N; the paper must replace "DISC WIN" / "VSA WIN" for task N.
- In every case: no generalisation beyond Llama-3.1-8B-Instruct, task N, these two read positions and these two read
  rules; the earlier B40 / B55 numbers are reported as superseded where B65 differs, never silently replaced.

## Compute

A800 box, GPUs 1, 2, 4, 5, 6, 7 only and only if idle at claim (memory.used < 1 GB, no compute process; GPU 0 runs
another user's job and GPU 3 holds rm_server.py; neither is touched); the shared claim directory; no retry loops; logs
in /path/to/work/b65 (/data1). GPU-hours from logs/gpu_hours.txt (per process) and logs/gpu_ledger.txt (per GPU).
Expected about 6-8 GPU-hours (36+ write-back runs of about 0.18 h, readers and evaluations small). Time box 3 hours
wall: if time runs short, the cut is written in a compute note BEFORE any test output is read, in this order: first
drop seed 2 of OPS, then OPS entirely; LAST with the extended grid is never cut. No paid API.

## Self-test (training rows only, before the commit)

`b65_run.py selftest` (operand positions built and asserted on all 16,000 training rows; padded operand indices point
at the operand tokens; a forward of each of the six modules); `cache --split train` (operand states, 8 layers);
readers for 2 epochs with "dev" = 512 training rows; write-back for 20 steps with "dev" = 32 training rows; evaluation
on 32 training rows (queues q/S1-S3; outputs under smoke/, never dev or test).

Self-test result (box GPU 1, 05:01:57-05:07:40Z, 0.11 lane-hours): operand positions built and asserted on 16,000 /
16,000 training rows (0 failures); the six modules run (parameters: VSA 16,777,216 at either position; DISC 647,276 at
LAST = B40's, 1,056,876 at OPS); operand-state cache of the training split written (8 layers); readers, 20-step
write-backs and 32-row evaluations on training rows ran rc 0. One procedural slip in the smoke queue only: the
grep that split train from eval lines also matched the evals' "--split train", so 6 evaluations first ran before their
write-back existed (rc 1) and ran again after (rc 0); fixed in b65_orch.sh (the dev and test queues are homogeneous).
The verdict code was exercised on fabricated files built from B40's own test predictions (B40's VSA / DISC files placed
in the B65 layout): it reproduces B40's VSA - DISC = -3.23 (one-level [-4.60, -1.83] vs B40's [-4.6, -1.9]; two-level
[-4.80, -1.65]) and B40's S3 read-exact 0.8875 / 0.9518. No dev or test row has been touched by any B65 module.
