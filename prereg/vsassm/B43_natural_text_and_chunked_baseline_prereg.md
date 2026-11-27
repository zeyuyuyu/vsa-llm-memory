# B43 pre-registration: a natural-text discrete-state task (ProPara) and a chunked-reading LLM baseline (Chain-of-Agents) on RULER and BABILong

Date: 2026-11-19 (Asia/Shanghai). Written, with its code, before any B43 test run. Answers two referee concerns:
- **M4:** every positive result is on synthetic or templated text; add a natural-text task with discrete state that
  compares the store, a table and the LM.
- **M3:** add a chunked-reading LLM baseline without a symbolic store (map-reduce or Chain-of-Agents style, Zhang et
  al. 2024), on the same samples and n.

Scope (coordinator, 2026-11-19, about one day): Part B first (RULER's memory tasks at 128k, then 64k), BABILong only if
it fits; Part A with Qwen3-8B only (Qwen3-32B dropped). Any further cut is written here before the test run it affects.

## Part B (M3): Chain-of-Agents (CoA) with the paper's reader model

### The system being compared (banked, not re-run)
B21 (prereg/B21_ruler_vsa_prereg.md), the paper's RULER system: Qwen3-8B (thinking off, greedy) reads 250-character
sentence chunks (`ruler_rm.chunks`) with the query-independent extraction prompt `ruler_rm.FACT_PROMPT`; grounded pairs
go into the sharded Fourier HRR store (`ruler_vsa.Store`, d = 8192); B19's answer rules. Per-sample predictions:
`b21_dl/b21_test/*.jsonl` (`pred_vsa`; `pred_table` = the dict on the same reader output). Nine memory tasks
(niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery, vt), seed-42 test sets, 100 samples per task:
n = 900 per length. Paper: 9-task mean 99.5 (64k), 99.6 (128k).

Sample match (checked before this file, CPU): the RULER files the baseline reads (`/path/to/HOME/bench_dl/ruler`, the
same files as PACE `bench/ruler`, md5 equal for the files checked) contain every B21 test sample at 64k and 128k with
the same index and gold outputs (1,800 / 1,800). Pairing is by (task, index).

### The baseline (`b43_coa.py`)
- **Chunks:** exactly the reader's chunks (`ruler_rm.chunks(ctx)`), packed in order into windows of at most W = 8,192
  Qwen3 tokens (CoA's default 8k window, chosen a priori, not tuned), joined by newlines. Every chunk the reader reads
  is in exactly one window; nothing else is added. About 8 windows at 64k and 16 at 128k.
- **Workers:** worker i gets window i, the previous communication unit CU_{i-1} (empty for i = 1) and the RULER
  question (the input's last line), with CoA's query-based worker prompt verbatim (arXiv 2406.02818, Table 9;
  identical in the camera-ready, Table 11), and writes CU_i.
- **Manager:** CoA's manager prompt verbatim; {Task specific requirement} = RULER's own instruction (the input's first
  line); RULER's answer prefix starts the manager's turn (the RULER protocol, as in B19's direct arm).
- **Decoding:** Qwen3-8B (snapshot b968826d, the paper's reader), chat template, thinking off, greedy (CoA: temperature
  0). Workers at most 1,024 new tokens (CoA's setting); the manager 2 x RULER's tokens_to_generate (B19's direct arm).
  HF generate, bf16, sdpa; batches of samples move through their windows in lockstep (batch size by GPU memory:
  50 on an H200, 25 on an 80 GB card, 10 on a 48 GB card; dev used 16; a batch that runs out of memory is split).
  Thinking off matches the paper's reader; no thinking arm is run (scope).
- **No symbolic store, no retrieval, no tool:** the only memory is the text of the communication unit.
- **Logged per sample:** every CU, prompt and generated tokens of every call, window count, GPU.

### Scoring and comparisons
- Official `string_match_all` (as B19/B21), per sample; 9-task mean per length.
- d = store (B21 `pred_vsa`) - CoA, paired by sample; 95% stratified paired bootstrap (100 samples resampled within
  each task, 10,000 resamples, numpy seed 0; `b24_verdict.paired`'s method). Also reported: dict (`pred_table`) - CoA,
  per-task scores, Wilson intervals.
- Reading of each length, margin 3.0 points (B24's bar for the store against arm F):
  - **"store ahead"**: d >= 3.0 and the interval's lower end > 0;
  - **"store ahead, small"**: lower end > 0 and d < 3.0;
  - **"CoA ahead"**: upper end < 0;
  - **"no detectable difference"**: otherwise.

### Bars
- **C1 (primary): the reading at 128k.**
- **C2: the reading at 64k** (run after 128k; dropped only if compute runs out, which is then stated).
- **C3 (secondary, BABILong, only if it fits):** see below.
- **Cost (reported, not a bar):** tokens per sample (prompt and generated) for CoA and for the store's reader. The
  reader's tokens are recounted from B21's banked chunks and raw outputs with the same tokenizer and chat template
  (one question per sample; the reader's reading does not depend on the question, CoA's does), and with B21's cache
  (identical chunk prompts read once per run) as run.
- **Chain diagnostics (reported):** for each CoA sample, whether the final CU contains every gold value (evidence
  kept), whether some earlier CU contained a gold value the final CU lacks (chain loss, the CoA camera-ready's
  "catastrophic collapse"), and the manager's errors when the final CU holds every gold value.

### BABILong (C3, secondary; run only after RULER 128k and 64k are submitted and only if it fits)
- System (banked): B18 rules over the B30b re-run reader's events (Qwen3-1.7B + the distilled LoRA student_s0), QA1-5,
  official test release, 128k, 100 samples per task, n = 500: 99.8 (`b30b_out/bab/assembled_128k.jsonl`,
  `ok_official`; equal to banked B18 on 500/500).
- Baseline: the same CoA protocol over the sentences the B18 reader reads (`b18.sentences`, via `b18.load`), W = 8,192,
  with **Qwen3-8B** as worker and manager (not the 1.7B reader: the reader is fine-tuned to write events, not to
  answer, and its base model scored far below 8B on B30b's notepad arm; 8B is the stronger baseline). Manager
  {Task specific requirement} = the answer instruction of the paper's whole-text arm (`b18_llm.prep`); at most 64 new
  tokens.
- Scoring: `b18.correct_official` on the manager's answer; the official `compare_answers` reported alongside.
- C3: memory - CoA, paired by sample, 95% stratified paired bootstrap (within task) and exact McNemar; the same four
  readings with margin 3.0 points.

### Predictions (Part B)
- PB1: CoA 9-task mean at 128k between 80 and 97 (each window is within the length where Qwen3-8B direct is ~100% on
  these tasks; losses from summaries that drop needles, from multi-value accumulation over many windows, and VT chains).
- PB2: C1 reads "store ahead".
- PB3: CoA at 64k >= CoA at 128k.
- PB4: CoA's tokens per sample (one question) are FEWER than the store reader's without the cache (the reader repeats
  its ~250-token instruction for each of ~2,000 chunks at 128k); the reader's tokens are amortised over all questions.
- PB5: the weakest CoA tasks are niah_multivalue and niah_multiquery (several values, any one lost fails the sample)
  and vt.
- PB6 (BABILong): CoA (8B) at 128k between 40 and 85; C3 reads "store ahead".
- PB7: chain collapse is visible but not dominant: on RULER at 128k, the final CU of fewer than 20% of samples carries a
  rejection phrase ("not mentioned", "does not contain", ...), and most CoA failures are a gold value never written
  into any CU or lost along the chain, not a manager error.
- Context (results/b43_literature.txt): CoA's own camera-ready reports 97.8% on NIAH PLUS up to 128k (text-bison);
  a third-party CoA re-run on InfiniteBench (LLMxMapReduce, Llama3-70B) gives PassKey 9.3 / Number 15.6 / KV 1.8. We
  found no CoA or map-reduce result on RULER or BABILong. A weak CoA result here would be checked against collapse
  before it is used as evidence.

## Part A (M4): ProPara, a natural-text discrete-state task (`b43_propara.py`)

### Why ProPara
- Natural text: crowd-written paragraphs describing science processes (Mishra et al., NAACL 2018), not templates.
- Discrete state: for each participant, before the first sentence and after every sentence, "-" (does not exist), "?"
  (exists, location unknown) or a location span from the paragraph; the standard grids (allenai/propara,
  data/emnlp18/grids.v1.*.json) give exactly the official answer files (checked: 1,674 / 1,674 test rows equal).
- A public test set with an official scorer (the aristo-leaderboard ProPara evaluator; document-level Inputs / Outputs /
  Conversions / Moves precision, recall, F1; our run of its dummy predictions gives its published 0.326) and published
  LLM numbers to compare with (GPT-4o prompted 59.6-62.1; fine-tuned systems to ~73; see results/b43_literature.txt).
- Not chosen: the boxes task of Kim & Schuster (2023) is generated from templates (so it would not answer M4) and its
  data is shipped password-protected against contamination; OpenPI is open-vocabulary (no discrete state); 2Wiki over
  triples is already in the paper (its failure was extraction).
- Split sizes (EMNLP18 split): train 391, dev 43, test 54 paragraphs (test: 236 participants, 373 sentences).

### Arms (Qwen3-8B, snapshot b968826d, HF generate; one worked example, the same for every arm)
- Worked example: the first training paragraph (file order) with <= 6 sentences and <= 4 participants that has a
  CREATE, a DESTROY and a MOVE with a named destination (paragraph 35, rocks -> magma -> lava -> igneous rock); for the
  direct arms its gold grid, for the extractor its per-sentence events derived from the gold grid.
- **direct_think (primary direct arm, chain of thought):** the paragraph (numbered sentences), the participant list and
  the state definitions; the model writes the whole grid as one JSON object. Thinking on, Qwen3's recommended thinking
  sampling (T 0.6, top-p 0.95, top-k 20; seed fixed per batch), reasoning budget 8,192 tokens, then budget-forced with
  B30b's CLOSE phrase; answer at most 2,048 tokens.
- **direct_nothink (secondary):** the same prompt, thinking off, greedy.
- **extract -> store_vsa / store_dict (the paper's route):** the model reads ONE sentence at a time (the previous
  sentences as context, the participant list) and writes events: ["create", p, location], ["destroy", p],
  ["move", p, from, to]; thinking off, greedy, at most 400 tokens (the paper's readers). Names are matched to the
  listed participants (normalised exact match of the entry or one of its ';' alternatives, then without a leading
  article / plural s; unmatched events are dropped and counted). Read rules (the same for both memories): state 0 =
  "-" if the participant's first event is a create, the stated origin if it is a move with one, else "?"; then, sentence
  by sentence, create -> its location (or "?"), destroy -> "-", move -> its destination (or "?"); every participant is
  read back after every sentence.
  - store_vsa: one Fourier HRR vector per paragraph (d = 2,048; seeded-hash codes, separate key and value codebooks);
    write = delta rule (unbind and subtract the current value, add the new one); read = unbind + clean-up against the
    values written so far plus "-" and "?" (threshold 0.5).
  - store_dict: a Python dict with the same writes.

### Metrics
- **Primary: the official document-level F1** (points), from each arm's grid converted to the official action file
  (action from consecutive states; before / after locations from the grid), scored by the leaderboard evaluator's own
  modules (and its command-line program, which must agree to 0.001).
- Secondary: **state accuracy**, all states 0..N of every participant (2,0xx cells on test): "-" and "?" must match
  exactly, locations match under the evaluator's own normalisation (leading article removed, Porter stem).
- **Extraction (reported separately from end accuracy):** per (participant, sentence), the action implied by the
  extractor's events (create / destroy / move; NONE if no event) against the gold action: precision and recall over
  non-NONE actions (type must match), and destination accuracy (official location normalisation) on matched CREATE /
  MOVE. The same action-level numbers are reported for the direct arms' grids.
- Intervals: 95% bootstrap over the 54 test paragraphs (10,000 resamples, numpy seed 0), paired (the same resampled
  paragraphs for both arms; the official aggregation recomputed on every resample).

### Bars
- **A1 (primary):** d = store_vsa - direct_think (official F1). "Store route beats the direct LM" iff d >= 10 points and
  the paired interval's lower end > 0. Otherwise: "store ahead, < 10" (lower end > 0), "direct ahead" (upper end < 0),
  "no detectable difference".
- **A2:** store_vsa and store_dict give the same grid on 100% of cells (any difference is a VSA read error and is
  listed).
- A3 (reported): store_vsa - direct_nothink; direct_think - direct_nothink.

### Predictions (Part A)
- PA1: A1 does NOT reach "beats": paragraphs are short (3-10 sentences) and fit in context, and ProPara's gold encodes
  commonsense inferences about unstated locations that a sentence-by-sentence extractor will not write. Expected
  official F1: direct_think 45-62, store 40-58; most likely reading "no detectable difference" or "direct ahead".
- PA2: A2 holds (100% of cells).
- PA3: extraction recall of non-NONE actions < 0.70; precision > recall.
- PA4: direct_think >= direct_nothink.
- Prior evidence on the direction is mixed, so it is tested, not assumed: IC-DST (Findings EMNLP 2022) found that
  extracting per-turn state changes and accumulating them beats writing the full dialogue state (58.7 vs 47.9 JGA);
  Wang et al. (ACL 2024) found full-state prediction matched or beat difference prediction for GPT-4 on transitions
  that change. A training-free symbolic ProPara tracker (PROPOLIS, Findings EACL 2023) scores 58.7; GPT-4o prompted
  59.6-62.1 (IWCS 2025, per-participant prompts); fine-tuned systems 72-73. Nothing here is a matched baseline.

## Development (before any test run)
- Part B: CoA on RULER dev sets only (seed 7: 16k, 20 per task, B19's dev set; 64k, 10 per task, generated for B43
  with the official scripts, `/path/to/HOME/bench_dl/ruler_dev64`). Dev may change parsing and prompt-format plumbing, not
  W, the prompts' wording or decoding. Dev outputs and any change go into a dev record below, before the test run.
- Part A: ProPara dev split (43 paragraphs) for prompt wording, output format, name matching and read rules. Frozen in
  the dev record before any test call.
- The test sets are run once. A run that crashes is resumed (completed samples are kept); nothing is re-run because of
  its outcome.

## Checks
- Second scorer: RULER and BABILong CoA predictions re-scored by an independent implementation (`b43_check.py`, not
  importing `ruler_rm` or `b18`'s scorers); ProPara: the evaluator's command-line program against the in-process
  aggregation, and an independent re-implementation of state accuracy and of the grid -> action conversion.
- Hand reading: RULER 128k, every CoA failure up to 40 (sampled with seed 0 if more), classified as needle never in
  any CU / lost along the chain / in the final CU but the manager failed / format; ProPara, the 10 test paragraphs with
  the largest |store - direct| state-accuracy difference, both grids read against the text and gold, errors
  classified (extraction miss, read rule, gold ambiguity, direct-arm error).
- Full-set numbers only (900 per RULER length, 500 for BABILong, 54 ProPara paragraphs).

## Compute and safety
A800 (GPU 1 only is idle now; never GPU 3; `b43_run.sh` claims only GPUs with < 500 MiB used) and PACE Phoenix
(`b43_pace.sbatch`, free tier gts-nimam6 first; paid only if the free queue blocks the deadline, spend reported). HF
generate only. Cap 60 GPU-hours. No paid API.

## Dev record, part B (2026-11-19, before any B43 test run)
- CPU plumbing check (fake model, dev 64k, seed 7): windows of 7,966-8,014 tokens (9 windows at 64k), every reader
  chunk in exactly one window, every gold needle inside some window. Fix from it: VT's question line already starts with
  "Question:", which CoA's template adds again, so a leading "Question:" is stripped from the question (both prompts).
  Items now carry their position in RULER's file (`pos`): niah_multikey_3 repeats indices (93 / 96 distinct of 100 at
  64k / 128k), so pairing with B21 is by (task, position), with index and gold asserted equal.
- GPU dev (PACE H100, job 15836670, dev 16k seed 7, 20 per task; 2-3 windows per sample), first 48 of 180 samples
  (niah_single_1-3) read before this record: niah_single_1 20/20, niah_single_2 17/20, niah_single_3 5/8. Failures
  look like CoA's documented collapse ("not present" in the final CU) and drift (an essay number as the answer).
  Workers write 150-740 tokens (mean 346; none at the 1,024 cap); the manager's answer is bolded ("**9920785**."),
  which string_match_all scores correctly. No change made from these outputs.
- Throughput: ~290 s per batch of 16 at 2-3 windows on an H100 (decoding dominates), so a 128k task (100 samples,
  17 windows) is ~3 GPU-hours at batch 16. The PACE batch size now follows GPU memory (50 / 25 / 10); this changes no
  prompt or decoding setting.
- The rest of the 16k dev run (multikey, multivalue, multiquery, vt) finishes after this record; it may only lead to a
  crash or plumbing fix, recorded here before the affected test output is read.
- Compute plan given the cost: RULER 128k (all 9 tasks) first; 64k next if the 128k jobs leave room under the 60
  GPU-hour cap; BABILong only after both and only if it still fits (estimated ~15 GPU-hours, so likely dropped; stated
  in the verdict either way).

## Dev record, part A (2026-11-19, ProPara dev split, 43 paragraphs; before any ProPara test call)
- Run: PACE H100, job 15836667 (all three arms; 47 min, of which the thinking arm ~42 min). Thinking: 1 of 43 runs
  budget-forced; reasoning 2.5k-7.7k tokens otherwise.
- Dev scores (official document-level F1 / state accuracy): direct_think 61.7 / 56.6; direct_nothink 53.2 / 43.0;
  store_vsa 59.7 / 45.8; store_dict identical to store_vsa on 1,379 / 1,379 cells. store_vsa - direct_think F1 -2.0
  [-5.5, +1.4]. Extraction: action precision 0.41, recall 0.55 (414 predicted, 310 gold), destination accuracy 0.57.
  The CLI evaluator agrees with the in-process aggregation; the second scorer (nltk Porter) gives the same state
  accuracy and 0 grid -> action mismatches.
- Hand reading (3 paragraphs with the largest store/direct gap): the store's errors are extraction errors (a "create"
  for a participant that already exists, so state 0 becomes "-"; moves to non-locations such as "upwards"; events for a
  participant after it was destroyed), not read errors.
- One change, name matching only (applies to every arm): the gold participant list contains misspellings
  ("Enyzmes"), which the models write correctly; the matcher now falls back to the closest listed entry when it is
  unique and has a character-sequence similarity >= 0.85 (difflib). Dev after the change: direct_think 62.1, others
  unchanged.
- Not changed: prompts, worked example, decoding, read rules. No arm's prompt was tuned on dev (both routes keep their
  first wording), so no arm gets more prompt search than another.

## Dev record, part B addendum (2026-11-19, after the rest of the 16k dev run; before any test output is read)
- Full 16k dev run (180 samples, job 15836670, 50 min on an H100): CoA 9-task mean 81.1 (single_1 100, single_2 85,
  single_3 70, multikey_1 75, multikey_2 80, multikey_3 100, multivalue 77.5, multiquery 88.8, vt 54.0). VT answers
  are well-formed name lists; the misses are chain members absent from the final CU (partial credit under
  string_match_all). One VT answer ran to the manager's 60-token cap (RULER's 2 x 30) with a self-correction; kept as
  the protocol. No crash, no out-of-memory split, no plumbing change.

## Scope cut (2026-11-19 09:05 Asia/Shanghai; before any 64k output is read)
- C2 (RULER 64k) is not run. Measured 128k cost: 50-75 min per batch on H200/H100 for noise and needle haystacks,
  >2 h per batch for essay haystacks (single_3, multikey_1, multivalue, multiquery); projected 128k total ~43
  GPU-hours, so 128k + 64k (~22) + ProPara and dev (~3) would pass the 60 GPU-hour cap. The nine 64k jobs (three had
  started) were cancelled; their partial outputs are not read or reported.
- C3 (BABILong) is not run, for the same reason (estimated 15-25 GPU-hours).
- ProPara (part A) test ran and was scored before this cut; nothing in it changes.

## Compute note 1 (2026-11-19, during the 128k test run; no output read)
Essay-haystack tasks run at ~2.2 h per batch of 25 on an A100 and ~2.5 h per batch of 50 on an H200 (the workers'
summaries of essay text run to several hundred tokens, and a batch decodes until its longest summary ends), so some
128k jobs cannot finish inside their 6-hour limit. `b43_coa.py` gains `--start / --stop` (a slice of the samples by
file position, written to a separate output file); a job is cancelled right after a completed batch and the remaining
samples are split over new jobs. Prompts, decoding and the protocol are unchanged; outputs are keyed by (task,
position), and the verdict asserts that every one of the 900 samples is present exactly once.

## Compute note 2 (2026-11-19, during the 128k test run; no CoA test output read)
- niah_single_3 (A100): cancelled after 50 samples; positions 50-74 and 75-99 run as slices.
- niah_multivalue (A100): cancelled after 50 samples; positions 50-74 and 75-99 run as slices.
- niah_multikey_1 (H200, batch 50): no batch finished in 5.6 hours (the batch sat near the card's memory limit);
  positions 0-24, 25-49, 50-74 and 75-99 run as slices (25 per job). If the original job finishes its first batch
  before its time limit, the duplicate slices are cancelled or, if already done, the verdict keeps one record per
  (task, position) and reports the duplicates.
- Every slice uses the frozen code path; the only difference is the sample range and the batch size.

## Compute note 3 (2026-11-19 ~05:10 UTC)
- The original niah_multikey_1 job (H200, batch 50) reached its 6-hour limit without finishing a batch (0 samples
  written). Its 100 samples, and the remaining 50 of niah_single_3 and of niah_multivalue, run as eight 25-sample
  slices (jobs 15848754-15848763) on H200 / H100 / A100-80GB only (a 25-sample slice is one batch there; on 40-48 GB
  cards it would be three batches and overrun its limit). An earlier set of slice jobs was cancelled unrun after a
  constraint edit made them unschedulable; one slice briefly waited on the A800 (no GPU became free; stopped
  unrun). No slice ran twice.
- GPU-hours at this point: 41.8 on PACE (including 6.0 for the multikey_1 job that produced nothing and 1.3 for the
  cancelled 64k starts); the eight slices add ~18, so the total lands at about the 60-hour cap.
- If a slice cannot finish before the report is due, the registered 9-task bar is not computed; the verdict reports the
  complete tasks and the partial tasks separately, as written in b43_verdict.py.

## Compute note 4 (2026-11-20 01:40 Asia/Shanghai; before any remaining 128k output is read)
- The session was down for ~12 hours; PACE is unreachable (VPN), so the state of the eight PACE slice jobs
  (15848754-15848763) is unknown and they cannot be cancelled now.
- The same eight slices (niah_multikey_1 0-24 / 25-49 / 50-74 / 75-99, niah_single_3 50-74 / 75-99,
  niah_multivalue 50-74 / 75-99) run on the A800 (GPUs 0, 1, 2, 4, 5, 6, 7; never GPU 3; `b43_run.sh`, batch 25,
  same code, same model snapshot b968826d, RULER files md5-identical to PACE's), into `b43_out/a800/`.
- Rule fixed now, before either run's output is read: **the A800 outputs are the reported ones.** If PACE becomes
  reachable, pending twins are cancelled; twin outputs that already exist are not scored into any bar and are used
  only for a per-sample agreement check (hardware/batching numerics), reported as such. If an A800 slice does not
  finish before the report is due, its samples are reported as not run (no PACE substitution).

## Compute note 5 (2026-11-20 01:45 Asia/Shanghai; no remaining output read)
Only four A800 GPUs became free (other users took GPUs 0, 4, 6 within seconds): niah_multikey_1 25-49 (GPU 7),
50-74 (GPU 5), niah_single_3 75-99 (GPU 1) and 50-74 (GPU 2) run now; niah_multivalue 50-74 and 75-99 wait for the
next free GPU. A multikey_1 slice takes ~3-3.5 h on this class of GPU, so slices started later could not finish
before the report is due: the waiting multikey_1 0-24 and 75-99 claims were stopped unrun, and those 50 samples are
reported as not run. Consequence, stated before any of these outputs is read: the registered 9-task bar C1 cannot be
computed on the full set; the verdict reports the complete tasks (expected 8 of 9) and niah_multikey_1 on its finished
samples, separately, as b43_verdict.py does.

## Compute note 6 (2026-11-20 01:50 Asia/Shanghai; no remaining output read)
PACE came back. sacct shows the eight PACE slice jobs (15848754-15848763) never started (pending throughout, 0 s
elapsed, no output files). All eight were cancelled, so every remaining sample has exactly one runner (the A800).
niah_multivalue 50-74 started on A800 GPU 4; 75-99 waits for the next free A800 GPU. multikey_1 0-24 and 75-99 stay
not run (compute note 5); running them as well would also pass the 60 GPU-hour cap (41.8 on PACE so far + ~17 for the
six A800 slices).
