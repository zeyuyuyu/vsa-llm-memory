# B33 pre-registration: the corrected task-2 parse on bAbI questions that no part of the paper has used

Date: 2026-11-18, about 07:20 UTC (02:20 US Eastern; box clock Asia/Shanghai). Written, with all B33 code,
before any B33 run on any evaluation question: no smoke run, no gate run, no evaluation output. The fresh set does not
exist yet; it is generated after this file is committed, with the seed registered below. The only task-2 numbers
known are the published ones (listed below with their files) and the distribution statistics of the official
TRAINING split used to build and check the generator (results/b33_gencheck.json).

## Why

On bAbI task 2 (object -> holder -> location) the memory route (frozen Phi-2 + naming head + VSA, hand-written parse
and ordering rule) scored 1.000 on the 420 "expressible" test questions and, in B28 (commits 6500aa0 / ceb195c,
results/b28_verdict.txt), 0.840 on all 500, because the parse omits two of bAbI's verbs ("grabbed", "discarded").
With the two verbs added after the fact it answered 500 of 500 (B28 MEM-v; exploratory, post hoc: the gap was found
on these 500 questions), against 0.971 for the best of twelve LoRA fine-tunes (selected on the same test set) and
0.386 for prompted Phi-2. B33 turns the post-hoc fix into a confirmatory result: the corrected parse, nothing else
changed, on task-2 questions that no part of the paper has used, with every comparator run exactly as before.

## Which questions are fresh (audit, done before registration)

1. **The HF split has more than 500.** Muennighoff/babi (the paper's source; the official bAbI v1.2 en-valid 1k set)
   has 900 train / 100 validation / 1,000 test questions for task 2; the test questions form 200 stories of five
   questions each (a row's passage is every statement since its story began). The paper used the first 500
   (`load_babi(2, 2000, 200, 500)`: stories 1-100). Test rows 500-999 are stories 101-200: no story, and no
   (passage, question) row, is shared with the first 500 or with train/validation (tools/b33_count.py on the box).
2. **But rows 500-999 are not untouched.** ngcls v5.py (V-5, 2026-08-19/20; `--n_test` default 1000; v5_logs/v5.log:
   "bAbI task 2 ... n=900/100/1000") and v7.py (V-7; default 1000; v7_logs/v7.log: babi2 admitted, majority .1870 =
   V-5's 1,000-question value) featurized all 1,000 task-2 test questions and scored pooled-state probes, an MLP, an
   HD prototype read-out and the old VSA module on them. No route the paper reports (memory, fine-tune, prompting)
   and no parse ever read them, and the two-verb gap was found on rows 0-499; but by the rule "never touched by any
   banked run" they are not clean. Every other loader of this dataset (ngcls v30-v59, v131/v152 sweeps, memcheck,
   B28; grep of /path/to/work and of this repository) takes at most the first 500 task-2 test rows (v51's 20-task
   prompt survey: 300; v45/v47 use 2,000 but task 1 only). Rows 500-999 are therefore kept as a SECONDARY set
   (`hfrest`): same distribution as the paper's 500, never seen by the routes compared here.
3. **The official files with supporting facts and the 10k set** could not be fetched (dl.fbaipublicfiles.com returns
   AccessDenied; the HF facebook/babi_qa parquet export covers qa1 only), and BABILong, which the paper uses, is
   built from bAbI stories, so no other official task-2 story could be certified as unused.
4. **So fresh stories are generated (PRIMARY set).** The released generator (facebookarchive/bAbI-tasks, Lua,
   cloned 2026-11-18) is, in its README's words, "a rewrite of the original code that was used to generate the
   publicly available dataset ... not possible to produce exactly the same dataset". For task 2 (WhereIsObject) it
   differs from the public data where it matters: no "picked up", "went back to", "left", "discarded" or "there",
   an extra "let go of"; it never asks about a dropped object (its question filter needs the container's location),
   while 70.5% of public questions are about dropped objects; it asks after >= 3 single statements, while public
   stories advance in pairs. Running it would test a different task. b33_gen.py therefore keeps its world logic
   (hidden true world; a get needs actor and object co-located and the object free; only the holder drops; a move
   goes elsewhere; five questions, the story ending at the fifth; the answer read from the world) and takes the rest
   from the public TRAINING split (200 stories), as follows.

### What the official training split shows (all 200 stories; b33_gencheck.py)

- Templates (exact inventory): move "X moved/went/journeyed/travelled/went back to the L." (uniform, 546-563 each);
  get "X got/took/picked up/grabbed the O there." (always "there"); drop "X dropped/left/put down/discarded the O."
  with " there" in 21.2% of drops. Question "Where is the O?". 4 persons, 6 locations, 3 objects (as in the paper's
  codebook).
- Answer semantics: a text-only tracker (b33_track.py, written independently of the parse and of the generator: the
  holder's last stated place if the object is held, else the dropper's place at the drop) reproduces 1,000 of 1,000
  official answers. 70.5% of questions are about a dropped object.
- **The question rule is deterministic.** Stories advance in steps of two statements; after a step a question is
  asked exactly when some object is answerable, i.e. its last get/drop is among the last five statements, the text
  states its place, and (if held) the holder moved after the get; the object is uniform among answerable ones. With
  a window of 4 this explains 2,169 of 2,169 step ends (1,000 questions, 1,169 non-questions; windows 2/3/5/6
  explain 1,862/2,038/2,090/2,037), and no question follows an odd number of statements.
- Statement dynamics (no direct evidence of the original mechanism): gets also happen to people whose place was never
  stated (29%) and of never-mentioned objects (55%); people hold up to three objects; after a get the next statement
  is by the same person 39% of the time, after a drop 34%, after a move 25% (chance).

### Generator, fit and check (b33_gen.py, b33_gencheck.py)

Statement: a person uniform (after a get / drop the same person again with probability R_GET / R_DROP); if the person
holds objects, a drop (uniform among held) with probability P_DROP; else a get with probability P_GET (P_GET_HOLDER if
the person already holds something) if some free object can be where the person is (weight W_UNSEEN for a
never-mentioned object), else a move to a uniform other location. Hidden initial places INIT: random (uniform) or
objlazy (persons random, an unmentioned object can be where its first getter is) in the final fit. The structural
rules above were read off all 200 stories; the numeric parameters were fitted on the first 100 official training
stories only (random search of 320 settings, then a local search from the best 8, minimising the mean chi-square
distance per degree of freedom over every statistic; results/b33_gencheck_fit.json/.txt) and the generator was then
compared with the OTHER 100 stories on 26 statistics (story-level parametric bootstrap, B = 2,000; every p >= 0.05/26).

Statement model as registered (b33_gen.PARAMS, fitted on stories 1-100): INIT random, P_GET 0.62, P_GET_HOLDER 0.71
(a get when the person already holds something), P_DROP 0.55, R_GET 0.22, R_DROP 0.13, W_UNSEEN 1.05, P_THERE_DROP
0.228; WINDOW 4, steps of 2, five questions. INIT random because the places at which never-mentioned objects are
first picked up by people whose place is stated are spread over all six locations (football 11-24, milk 11-26, apple
15-20 per location), which excludes the released world file's fixed placement.

Registered check (results/b33_gencheck.json/.txt; frozen parameters, dev seed 1, 20,000 generated stories, the other
100 official stories): 24 of 26 statistics pass. Two fail, both about a person who already holds an object picking
up another: gets by such a person, official 13.9% vs generated 9.2% (p = 0.0005), and a holder's statement being a
get, 9.2% vs 6.8% (p = 0.0010). Nearest to the threshold among the passes: objects in play at question time (three
objects: official 27.1%, generated 38.8%; mean 2.04 vs 2.20 objects; p = 0.0045), dropper moved after the drop
(23.7% vs 18.3%; p = 0.008). Against all 200 stories (half of them used for the fit, so partly in-sample) three fall
below the threshold: the two above and objects in play. Everything the memory's parse and the questions depend on
matches: templates and verbs, the question rule (5,775 of 5,775 generated step ends), held vs dropped questions
(29.5% vs 30.8%), statements since the queried object's last event, answers, queried objects.

Disclosure of the fitting history: four statement models were tried (no persistence; persistence after gets and drops;
+ a weight for never-mentioned objects; + a separate get rate for holders), and other initial placements (lazy,
objlazy, the released world file, all objects at one place, clustered objects) were explored; the check half was
looked at for three of these fits (the third passed on its own fit-run seed, min p 0.0030, and failed on the
registered replication with seed 1, reported above). No model reproduced both the holders' second pickups and the
smaller number of objects in play; the remaining difference is that the fresh stories have slightly more objects in
play per question (more terms in the memory's object bundle, more objects for the fine-tune and the prompt to
track), which makes them, if anything, harder for every route.

Gate **G-DIST** (done before registration): **FAILED** (2 of 26 statistics below 0.05/26 on the check half). Rather
than keep fitting on the check half, the generator is frozen as it is and the design is changed before any
evaluation question is scored: every bar is read on the fresh set AND on hfrest, and is confirmed only if it holds on
both. The fresh set is never used but imperfectly matched; hfrest has the official distribution but was featurized
by the August pooled probes; each covers the other's weakness.

## The fresh set

`python b33_gen.py --n_stories 200 --seed 20261118 --out b33_data/b33_fresh.jsonl` (200 stories x 5 = 1,000 questions,
the size of an official test split), run once, after this commit. Seed 20261118 has never been used (the
generator fits and checks used dev seeds 1, 2, 3, 11, 12 and 101-103; a dry run of the verdict script on fake
outputs used 599 and 1099). Gate **G-FRESH** (b33_fresh_check.py, before any arm reads the
set): 1,000 questions in 200 stories; the second scorer (b33_track, independent of the generator's world and of the
memory's parse) agrees with every label; no (story, question) row identical to any official task-2 row
(train/validation/test); story grouping recovered from the text. Inputs longer than 768 tokens are reported, not
removed (scored as the published code scores them). If G-FRESH fails, nothing is scored until the cause is written
in a dated amendment; the set is not regenerated with another seed.

## Arms (exactly as before; nothing tuned on any evaluation set)

- **MEM (primary memory arm):** B28's MEM-v = b28_babi.py's task-2 route with `--patterns completed`: V-37b's parse with
  "grabbed" added to the pick-up verbs and "discarded" to the drop verbs (b28_babi.GET_COMPLETED / DROP_COMPLETED),
  nothing else changed; the ordering rule of V-37b; naming head trained as published on the parse's expressible
  training stories (900 under the corrected parse), seeds 42/123/456; published HRR encoding (comm). Code:
  b33_babi.py, which imports eval2/train_head/sf/index_rows from b28_babi.py and the verbatim ngcls copies.
  Exact codebook rows and B28's four other encodings are scored beside it (reported). A question the parse cannot
  read gets no answer and is scored wrong.
- MEM-pub (secondary): the same with the published patterns (head on 775 stories), to show what the fix is worth on
  new data.
- **FT12, FT1:** the published LoRA fine-tune (V-50 lora_direct, b28_lora2's training verbatim in b33_lora.py), best
  of twelve (rank 16, 6 epochs, lr 1e-4) and the first configuration (rank 8, 2 epochs, lr 1e-4), trained as
  published on the 775 expressible training stories, seeds 42/123/456; secondary: the same trained on all 900.
- **PROMPT:** V-49's zero-shot cell (verbatim copy b33_ngcls/v49.py, md5 a72de83e..., identical to the box original),
  one deterministic decoding.

Sets: `fresh` (primary), `hfrest` (secondary), `orig500` (the paper's 500; gates only). Per-question outputs saved.

## Gates

- G-DIST (generator; above): reported, not blocking (it failed before registration; see above). G-FRESH (fresh set):
  blocking.
- **G0:** the corrected-parse memory on orig500 reproduces B28 MEM-v per question (same parsed rows; agreement 1.000 in
  every arm x encoding x seed cell; naming head 1.000 x3). **G1:** the published-parse memory reproduces B28 MEM
  per question (0.840 x3). **G2:** prompting on orig500 within 0.004 of the banked 0.3860 (v49_logs/v49.json
  task2/zeroshot/all). Blocking: if G0, G1 or G2 fails, no fresh or hfrest number is read until the cause is
  written in a dated amendment.
- **G3 (reported, not blocking):** each fine-tune rerun on orig500 within 0.03 per seed of B28's rerun (FT12
  .968/.986/.958, FT1 .840/.832/.846); bf16 training is not bitwise reproducible. A miss is reported with its size.

## Bars (memory = MEM; fixed now). Each bar is read on the fresh set and on hfrest; it is CONFIRMED only if it holds
## on both, and a fine-tune reading is stated only if it is the same on both (otherwise SPLIT, reported per set)

- **B1** memory >= 0.98 in every seed.
- **B2** memory - prompting >= 0.40 in every seed, and the 95% CI of the mean difference has its lower bound > 0.30.
- **B3** memory vs FT12, paired per seed, CI by a bootstrap over stories: MEMORY LEADS if memory ahead by >= 0.10 in
  every seed (the paper's ten-point bar); FINE-TUNE LEADS if the fine-tune is ahead by >= 0.02 in every seed and the
  CI is below 0; MEMORY AHEAD, BELOW THE TEN-POINT BAR if memory is ahead by >= 0.02 in every seed and the CI is
  above 0; otherwise NO CLEAR DIFFERENCE. No advantage is claimed unless MEMORY LEADS.
- **B4** the same against FT1.
- **Shift safeguard:** if FT12's mean on fresh is more than 0.03 below its mean on the 1,000 official test questions
  (orig500 + hfrest), the generator is harder for the fine-tune than the official distribution; the hfrest
  contrasts are then reported first.

Intervals: Wilson 95% per seed (questions); means and differences: bootstrap over stories (10,000 resamples, paired
across arms and seeds), because a story's five questions share its text.

## Readings (what the paper may say)

- B1 and B2 CONFIRMED: the corrected parse is confirmed on questions the paper never used; the abstract may state
  both accuracies with CIs, the prompting gap, and the fine-tune contrast as its combined B3 reading.
- B1 fails on either set: the post-hoc 500/500 does not generalise there; the paper reports both numbers, says what
  failed (from the hand-reading), and keeps "with the two verbs added after the fact" as exploratory.
- B3: MEMORY LEADS is not expected; MEMORY AHEAD, BELOW THE TEN-POINT BAR or NO CLEAR DIFFERENCE keeps "we claim no
  advantage over a tuned fine-tune"; FINE-TUNE LEADS is stated as such.
- Both sets are reported in every case; the paper's 500 stay what they are (the set on which the fix was found).

## Predictions (stated so they can be wrong)

- P1 G-FRESH, G0, G1, G2 pass. P2 G3: both fine-tune reruns within 0.03 of B28 per seed.
- P3 memory on fresh >= 0.995 in every seed. P4 exact rows equal the naming head on fresh within 0.002 per seed.
- P5 published parse on fresh: mean in [0.78, 0.90]. P6 FT12 on fresh: mean in [0.93, 0.99]. P7 FT1: [0.72, 0.90].
  P8 prompting: [0.30, 0.46].
- P9 B1 and B2 CONFIRMED. P10 B3 = MEMORY AHEAD, BELOW THE TEN-POINT BAR on both sets. P11 B4 = MEMORY LEADS on both.
  P12 no shift flag.
- P13 hfrest: memory >= 0.98 every seed, memory - prompting >= 0.40 every seed, B3 reading as on fresh.
- P14 FT12 trained on all 900 >= FT12 trained on the 775 on fresh (mean). P15 total GPU-hours <= 4.

## Checks from other angles (in the verdict and the result record)

- Recomputation: b33_verdict.py recomputes every accuracy from the saved per-question outputs (memory: saved
  predicted location vs the label; fine-tune: argmax of the saved logits; prompting: saved prediction) and asserts
  equality with the logged numbers.
- Second scorer of the labels: b33_track (G-FRESH) on every fresh question; it also reproduced every official
  training answer.
- Second distribution: hfrest (official generator, never seen by these routes).
- Hand-reading: every memory error and unanswered question on fresh and hfrest; ten FT12 errors and ten prompting
  errors on fresh, read in the raw story (b33_handread.py), recorded in results/b33_handread_notes.txt.
- Literature, after the verdict: supports / conflicts / no precedent (LM parse + exact symbolic reasoner on bAbI,
  e.g. LLM + ASP; regenerated test sets, e.g. Recht et al. 2019; Dyna-bAbI), in results/b33_literature.txt.

## Compute

A800 box only, idle GPUs only (b33_worker.sh = b28_worker.sh with the B33 directory: claims a GPU with < 500 MiB used,
re-checked after 20 s; never GPU 3; never another user's process). PACE is allowed (user, 2026-11-18) but was not
reachable from this machine (ssh login-phoenix timed out at 06:15 UTC). At 06:15 UTC all seven usable A800 GPUs were
busy, so workers wait for a GPU to free. Projection: prompting 0.1, memory 0.3, fine-tunes (4 configurations x 3
seeds) about 1.6, smoke 0.1: about 2.5 GPU-hours. Cap 10 GPU-hours (the user allows up to 1.5x, 15, if parallel
runs need it; any use above 10 is recorded in the ledger).

## Done before registration (no evaluation question scored)

- The audit above (box: tools/b33_count.py, tools/b33_export_train.py; greps of the ngcls scripts and logs).
- Generator construction on the official training split (all reading of the official data was of train/validation
  rows; the generator was never run with the registered seed).
- Code written: b33_gen.py, b33_track.py, b33_gencheck.py, b33_common.py, b33_babi.py, b33_lora.py, b33_prompt.py,
  b33_fresh_check.py, b33_verdict.py, b33_handread.py, b33_worker.sh; b33_ngcls/v49.py (verbatim) + MD5SUMS.
- After this commit: smoke runs on TRAINING rows only (b33_babi/b33_prompt: 64 training rows; b33_lora: 64 training
  rows, one epoch), disclosed in a smoke record if they change anything; then the fresh set, G-FRESH, the gates
  and the arms.

## Files

- Code as listed; reused: b28_common.py, b28_babi.py, b28_ngcls/ (verbatim), b33_ngcls/v49.py (verbatim).
- Box: /path/to/work/b33 (out/, logs/). Results: results/b33_gencheck.json, results/b33_fresh_check.json,
  results/b33_verdict.txt, results/b33_scores.json, results/b33_gpu_ledger.txt, results/b33_handread_notes.txt,
  results/b33_literature.txt; per-question outputs in b33_out/ (gzipped); the fresh set in b33_data/.
