# B28 pre-registration: does an order-preserving VSA encoding remove the commutative-binding limit, and how do the bAbI task-2 routes compare on all 500 questions? (referee points)

Date: 2026-11-17, about 19:30 UTC (2026-11-18 on the local/box clock, Asia/Shanghai; US Eastern 2026-11-17
afternoon). Written, with all B28 code, before any B28 run of any kind: no smoke run, no gate, no test run. The only
numbers known are the published ones (listed below with their files). A toy check of the five encodings on random
vectors (two stored pairs, no task data) was run locally to test the code paths; it is described under "Done
before registration".

## Why

A referee of the npj Unconventional Computing manuscript (Results "What the vector symbolic memory adds", second
condition; Fig. 5b; Supplementary Note "Natural-text tasks"; Supplementary Note "Arenas and arms") raises two points.

1. **The commutative-binding limit.** The paper says that because HRR binding is commutative (tying a to b equals
   tying b to a), exact symbols reach only 0.691 on bAbI task 4 (directed relations) and 0.836 on task 15
   (deduction from a rule), where a LoRA fine-tune reaches 1.000 and 0.991, and that arena C's exact-symbol ceiling
   is 0.934 because some inputs alias (V-163: all 115 first-hop errors of the 1,500 seed-instances fall on the 228
   alias instances). The referee: this is a property of the encoding the authors chose, not of VSAs; standard
   remedies exist (a permutation applied to one argument, separate role and filler codebooks, role vectors such as
   relation (*) (subject (*) rho(object))).
2. **bAbI task 2 is compared on a biased subset.** The memory and the fine-tune were compared only on the 420
   questions that the hand parser plus the ordering rule answers correctly; they should be compared on all 500.

B28 tests point 1 with exact symbols and with the naming head on bAbI tasks 4 and 15 and in arena C, under the
published encoding and under standard order-preserving encodings, and answers point 2 by scoring both routes on
all 500 task-2 questions. Its result goes into the paper whatever it shows.

## What is reused exactly

- **bAbI code:** verbatim copies of the ngcls modules in `b28_ngcls/` (md5 in `b28_ngcls/MD5SUMS`, identical to the
  originals in /path/to/work/ngcls on the A800 box, checked 2026-11-17 19:15 UTC): v58.py `df951b33…` (task 4,
  V-58), v52.py `8a904e3e…` (task 15, V-52), v37b.py `229d8eec…` (task 2 memory, V-37b), v50.py `b96daeab…`
  (task 2 fine-tune, V-50; also run by V-131's sweep), v5.py `8ed74011…` (load_babi), v30.py `f932fcf8…`
  (tok_span), v13.py `996644f9…` (bind / unbind / cache). `b28_common.load_ngcls` imports the copies by path, so no
  original file is imported or written.
- **Arena C code:** the verbatim copies of B27 (`b27_arena/v13.py`, same md5 `996644f9…`), loaded by
  `b27_common.load_arena_modules`.
- **Instances:** as published. Tasks 4 and 15: `load_babi(task, 2000, 200, 500)`, rows built by the original
  `build` (900 train / 500 test). Task 2: `v37b.build` (444 of 500 test questions parsed, 420 expressible; 775
  expressible training stories). Arena C: `v13.Arena(tok, 800, 42)` train, `v13.Arena(tok, 500, 4242)` test.
- **Codebooks and clean-up:** as published. bAbI: `normalize(randn(TOT, 2048, generator seed + 11))`; arena C:
  `v13.unit(64, 2048, seed + 31)`. Clean-up = cosine argmax over the published candidate set (task 4: the rooms
  mentioned; task 15: the classes of the membership statements, then the properties of the class statements;
  task 2: the holders, then the locations of the bound movements; arena C: that level's fillers). d = 2048
  everywhere. Seeds 42, 123, 456.
- **Naming head:** as published (bAbI: linear head on span-mean frozen Phi-2 states, AdamW 1e-3, 40 epochs,
  batches of 512, after `torch.manual_seed(seed)`; arena C: v13's `probe_argmax` arm, 2 epochs, identity loss plus
  the answer loss on scale and bias). In bAbI the head's training does not depend on the encoding, so it is trained
  once per seed and read under every encoding. In arena C the arm is trained under each encoding (the answer loss
  shares the gradient clip with the head), exactly as v13.run.
- **Scoring:** as published. bAbI: accuracy over the 500 test questions, the prediction being the gold symbol of
  the winning mention. Arena C: the training-accuracy-maximising threshold on the final cosine, best test epoch of
  the two (the published protocol; the final epoch is reported beside it).
- **Comparators (banked):** task 4 parser and naming head .674/.702/.696, fine-tune 1.000 x3
  (ngcls v58_logs/v58.json); task 15 parser and naming head .824/.848/.836, fine-tune .992/.992/.990
  (v52_logs/v52.json); arena C parser and naming head .932/.932/.938 (figdata; reproduced by B27 gate G0a on
  2026-11-17); task 2 memory 1.000 x3 on the 420 (v37b.json), fine-tune first configuration (rank 8, 2 epochs,
  lr 1e-4) .8500/.6952/.8571 (v50_task2_BANKED.json) and .8500/.6952/.8405 (V-131 rerun of identical code,
  v131_logs/b_runs/r8_e2_lr0.0001.json), best of twelve (rank 16, 6 epochs, lr 1e-4) .9714/.9738/.9714
  (v131_logs/b_runs/r16_e6_lr0.0001.json). No per-question outputs of the fine-tunes were banked, so they are rerun.

## Part 1 and 2. The encodings

In every task a stored pair is (VALUE, KEY): the value is the item the query retrieves, the key is what the query
unbinds by. Each symbol i gets a key code K(i) and a value code V(i). A pair is stored as bind(V(value), KEY), a
query unbinds by KEY (circular correlation, as published), and the clean-up compares with V(candidates); at a hop
boundary the winner's KEY code becomes the next key. Task 4's key is bind(direction row, K(room)) as published;
direction rows stay codebook rows.

| encoding | K(i) | V(i) | role |
|---|---|---|---|
| comm | sym[i] | sym[i] | published HRR (bit for bit the published computation) |
| **perm** | sym[i] | sym[i] permuted by a fixed random permutation P of the 2,048 coordinates | **primary**: the standard order-preserving remedy, one argument permuted before binding (Plate; Kanerva 2009; Gayler's MAP). For task 4 this is exactly the referee's form: direction (*) (B (*) rho(A)) |
| roles | sym[i] | an independent random codebook (same size, same d) | secondary: separate role (key) and filler (value) codebooks; at a hop boundary the winner's index gives its key code |
| rtag | r_k (*) sym[i] | r_v (*) sym[i] (normalised); r_k, r_v random role vectors | control registered to FAIL: role vectors bound by convolution |
| shift | sym[i] | sym[i] rotated cyclically by one coordinate | control registered to FAIL: the familiar cyclic-shift permutation |

Why the controls must fail, derived now: circular convolution is commutative, associative and shift-equivariant,
so in rtag and shift V(x) = c (*) K(x) for one fixed vector c (c = r_v (*) r_k^-1, resp. a shifted delta). A pair
whose VALUE is the query q, bind(V(q), K(k')), unbound by K(q), then returns c (*) K(k') = V(k'): the same clean
spurious candidate as under comm. A random (non-cyclic) permutation and an independent codebook have no such c.
For shift the equivariance is exact, so shift must reproduce comm per question. The randomness of P, the second
codebook and the role vectors comes from private generators (seed + 7001, 7002, 7003, 7004), so the global torch
RNG, and with it every trained arm, is unchanged by the choice of encoding.

Settings: bAbI task 4, bAbI task 15, arena C. Arms: exact codebook rows (parser) and the naming head
(probe_argmax). Every encoding x arm x seed is scored on the full test split; per-question outputs are saved.

## Part 3. bAbI task 2 on all 500 questions

- **MEM (primary memory arm):** the naming head, comm, published parse patterns and ordering rule (V-37b), read on
  every test question that the parse reads (444). A question the parse cannot read (56; its patterns omit
  "grabbed" and "discarded") gets no answer and is scored wrong. Accuracy is over all 500.
- MEMx: the same with exact codebook rows. Both also under the four other encodings (reported; prediction below).
- **FT1:** the first published fine-tune configuration (rank 8, 2 epochs, lr 1e-4), **FT12:** the best of the twelve
  (rank 16, 6 epochs, lr 1e-4); both trained exactly as published on the 775 expressible training stories (v50's
  lora_direct arm), 3 seeds, and scored on all 500 test questions in dataset order (pass B) as well as on the 420
  in v50's order (pass A, reproducibility check). Per-question logits saved.
- Secondary, no bar: FT1-all and FT12-all, the same configurations trained on all 900 training stories (a
  fine-tune needs no parse, so restricting its training to parser-expressible stories handicaps it on the other
  80 questions); MEM-v, the memory with the two missing verbs added to the parse patterns ("grabbed" as a pick-up,
  "discarded" as a drop; a parser fix made after the paper disclosed the gap, labelled as such); the expected
  accuracy if MEM guessed uniformly among the six locations on the unread questions (sensitivity).
- Breakdown reported for every arm: the 420 expressible, the 24 parsed but not expressible, the 56 not parsed.

## Gates (run first, before any non-comm encoding and before any all-500 reading)

- **G0 (environment, original scripts):** v58.py and v52.py with `--arms parser probe_argmax`, v37b.py with
  `--arms floor parser probe_argmax` (verbatim copies, outputs redirected) reproduce the banked per-seed values
  within 0.002 (task 4, task 15; task 2 parser and naming head 1.000 x3 on the 420).
- **G1 (bAbI, new code path, comm):** b28_babi.py with encoding comm reproduces the banked per-seed parser and
  naming-head accuracies on tasks 4 and 15 within 0.002 (parser) and 0.004 (naming head).
- **G2 (arena C, new code path, comm):** parser and naming head .932/.932/.938 within 0.002 / 0.004 per seed.
- **G3 (task 2, new code path, comm):** the parse reads 444 test questions and 420 are expressible; parser and
  naming head 1.000 per seed on the 420.
- **G4 (task 2 fine-tune reruns; reported, not blocking):** pass-A accuracy per seed within 0.03 of a banked run
  (FT1: either of the two banked runs, which already differ by .017 at seed 456; FT12: the V-131 run). Training is
  bf16 on GPU and not bitwise reproducible. A miss is reported with its size, and the all-500 comparison then uses
  the rerun, with the rerun's 420-question accuracy printed beside the banked one.
- If G0, G1, G2 or G3 fails, no comparison in that setting is read until the cause is found and written in a dated
  amendment.

## Bars and readings (fixed now)

### Points 1 and 2: does an order-preserving encoding remove the limit?

Per setting and arm, accuracy per seed (500 test questions or instances), against comm on the same questions:

- **REMOVES THE LIMIT:** accuracy >= 0.98 in 3/3 seeds. On tasks 4 and 15 this is also within 0.02 of the banked
  fine-tune (1.000; 0.991). In arena C it lifts the exact-symbol ceiling from 0.934 to at least 0.98.
- **PARTIAL:** not REMOVES, mean gain over comm >= 0.05 with the paired bootstrap 95% interval excluding 0.
- **NO EFFECT:** |mean difference from comm| < 0.03.
- **GRADED:** otherwise.

The registered verdict is perm on exact symbols. **OVERALL:** if perm REMOVES in all three settings, the
commutative-binding limit is a property of the commutative encoding the paper chose, not of the memory or of VSAs.
roles is reported beside perm with the same bar; rtag and shift are controls.

**Memory against the fine-tune on tasks 4 and 15** (naming head under each encoding vs the banked fine-tune):
TIE if the mean difference is within 0.02; FINE-TUNE LEADS if the fine-tune is ahead by >= 0.05 in 3/3 seeds;
MEMORY LEADS if the memory is ahead by >= 0.10 in 3/3; GRADED otherwise.

Readings:
- perm REMOVES on tasks 4 and 15 and in arena C: the paper withdraws "these results mark a limit of the memory"
  and the second condition stated as a property of the memory. It says instead that the limit came from the
  commutative encoding we used: with one argument permuted before binding (the standard remedy) exact symbols
  reach the measured values on tasks 4 and 15 and in arena C, at the same dimension, codebook and clean-up, and
  the naming head reaches the measured values (a tie with the fine-tune where the bar says TIE). Fig. 5b's group
  "algebra not exact on the task" and the arena-C row are redrawn with the order-preserving encoding (the
  commutative values may stay as open markers). The controls are reported in one sentence: role vectors bound by
  convolution and a cyclic shift do not remove the limit, because both commute with circular convolution.
- PARTIAL in a setting: the paper reports the size of the reduction and keeps a limit, attributed to the encoding.
- NO EFFECT: the referee's remedy was tested and did not remove the limit; the paper says so, names the encodings
  tried, and keeps the limit.
- If the naming head under perm falls short of exact rows under perm, the remaining gap is the naming head's and
  is reported as such.

### Point 3: bAbI task 2 on all 500

Primary contrast: MEM against FT12 on all 500, paired per seed (memory seed s against fine-tune seed s) and as a
mean difference with a paired bootstrap 95% interval over questions (10,000 resamples). Also MEM against FT1.

- **MEMORY LEADS:** memory ahead by >= 0.10 in 3/3 seeds (the paper's ten-point bar).
- **FINE-TUNE LEADS:** fine-tune ahead by >= 0.02 in 3/3 seeds and the interval of the mean difference below 0.
- **NO CLEAR DIFFERENCE:** otherwise.

Readings: whatever the outcome, the paper replaces the 420-question comparison by the all-500 one, states how many
questions the memory leaves unanswered (parse) and gets wrong (ordering rule), and reports the 420 numbers only
as the subset they are. FINE-TUNE LEADS: the paper says that across all 500 questions the tuned fine-tune is ahead,
and that the memory's lead on the 420 (2.6-2.9 points) is confined to the questions its parse and ordering rule
can express. NO CLEAR DIFFERENCE / MEMORY LEADS: stated with the numbers.

## Predictions (stated so they can be wrong)

- P1: G0-G3 pass in every cell.
- P2: perm REMOVES on task 4, task 15 and arena C (exact symbols); I expect 1.000 or within one or two questions of
  it in every seed.
- P3: perm with the naming head REMOVES in all three settings and equals exact rows under perm within 0.004 per
  seed (the head names every symbol correctly on these tasks).
- P4: roles REMOVES in all three settings (exact symbols).
- P5: rtag does NOT remove the limit in any setting.
- P6: shift equals comm per question (agreement >= 0.99) in every setting and seed.
- P7 (mechanism): under comm, exact symbols, accuracy on non-alias instances >= 0.99 and on alias instances in
  [0.30, 0.70], every setting and seed (arena C: retrieval correctness, i.e. both hops return the true symbol,
  because a label-0 instance is scored correct even when retrieval fails).
- P8 (mechanism): under perm, exact symbols, >= 0.98 on alias and on non-alias instances, every setting and seed.
- P9: alias share (structural flag, computed from the gold symbols: the query also occurs as a value of a pair
  whose other symbol is a clean-up candidate and differs from the answer): task 4 in [0.50, 0.75], task 15 in
  [0.20, 0.45], arena C in [0.10, 0.25].
- P10: the naming head under perm TIES the banked fine-tune on tasks 4 and 15.
- P11: task 2: MEM on all 500 in [0.835, 0.850] in every seed; FT12 on all 500 with mean in [0.90, 0.96];
  MEM vs FT12 = FINE-TUNE LEADS.
- P12: task 2: MEM vs FT1 = NO CLEAR DIFFERENCE.
- P13: task 2: MEM 1.000 on the 420 and <= 0.05 on the other 80.
- P14: the second implementation agrees with the main runs on >= 0.99 of questions in every cell; on fresh
  codebooks perm reaches >= 0.98 in 20 of 20 for every setting, and comm's mean is within 0.03 of the published
  ceiling (.691, .836, .934).
- P15: MEM-v (completed verb patterns) on all 500 in [0.90, 0.97].
- P16: FT12-all on all 500 >= FT12 (775 expressible) on all 500 (mean).
- P17: total GPU-hours <= 8.
- Encodings other than comm on task 2: no change from comm (the three types in task 2 have disjoint codebook rows,
  so nothing aliases); reported without a bar.

## Checks from other angles (in the verdict)

- **Recomputation:** b28_verdict.py recomputes every accuracy from the saved per-question outputs (bAbI:
  predicted symbol against the gold label of the dumped structure; arena C: saved cosines against the saved
  threshold; fine-tune: argmax of the saved logits against the dataset labels) and flags any difference from the
  logged number.
- **Second implementation:** b28_indep.py, numpy float64, written separately: the exact-symbol arms under all five
  encodings from the dumped gold structures (bAbI) and from arena-C instances regenerated without the tokenizer
  (checked against the main run's dump); per-question agreement with the main runs; and 20 fresh codebooks per
  setting and encoding.
- **Mechanism:** accuracy split by the structural alias flag (P7-P9).
- **Hand-reading:** at least 10 discordant cases per bAbI setting (comm wrong, perm right; and every perm error),
  and at least 10 task-2 questions of each of the three subsets where the memory and FT12 disagree, read in the
  raw story and recorded in the verdict commit.
- **Intervals:** Wilson 95% per seed; bootstrap over questions (paired across seeds) for means and differences.
- **Literature:** after the verdict, a search for support or conflict (Plate's HRR work on order, permutation and
  role-filler frames; Kanerva 2009; Gayler's MAP; non-commutative bindings such as VTB and MBAT; VSA surveys),
  recorded as supports / conflicts / no precedent.

## Compute and budget

- A800 box, idle GPUs only, never GPU 3, never another user's process (`b28_worker.sh`: copy of b27_worker.sh, claims
  a GPU with < 500 MiB used, re-checked after 20 s). PACE was not queried for this run: the 11-15 survey and B27's
  11-17 check put the earliest H200/A100 starts days away, and the A800 had seven idle GPUs at 19:07 UTC.
- Projection: Phi-2 state caches and gates about 0.5 GPU-hours; arena C 30 runs about 1.5; bAbI arms about 0.5;
  fine-tune reruns (4 configurations x 3 seeds) about 2-3. Total about 5 GPU-hours, failed runs included, counted as
  GPU occupancy per worker (logs/gpu_ledger.txt). Cap 40 GPU-hours.

## Done before registration (no task data scored)

- Code written: b28_common.py, b28_babi.py, b28_arenaC.py, b28_lora2.py, b28_orig.py, b28_indep.py,
  b28_verdict.py, b28_worker.sh; verbatim copies in b28_ngcls/.
- Toy check (local CPU, random vectors, no task data): two stored pairs (value 1, key 0) and (value 0, key 2), query
  key 0, 20 random codebooks. comm, rtag and shift return the correct value and the alias with equal cosine
  (0.49 / 0.50; rtag 0.37 / 0.38); perm and roles return the correct value (0.57-0.58) and the alias at about 0
  in 20 of 20. comm's stored pair is bitwise the published bind; roll(x) (*) y = roll(x (*) y) to 2e-8.
- No smoke run on any task split has been made. After this file is committed, smoke runs score only training rows
  (tasks 2, 4, 15: the first 100 training rows; arena C: the B27 dev split v13.Arena(tok, 500, 777); fine-tune: 64
  training rows, one epoch); they are disclosed in an amendment if they change anything.

## Files

- Code: b28_common.py, b28_babi.py, b28_arenaC.py, b28_lora2.py, b28_orig.py, b28_indep.py, b28_verdict.py,
  b28_worker.sh, b28_ngcls/ (verbatim), b27_common.py and b27_arena/ (reused).
- Box: /path/to/work/b28 (out/, feats/, logs/).
- Results: results/b28_verdict.txt (generated by b28_verdict.py), results/b28_scores.json, results/b28_indep.json,
  results/b28_gpu_ledger.txt, per-question outputs in b28_out/ (gzipped).
