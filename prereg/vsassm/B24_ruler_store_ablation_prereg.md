# B24 pre-registration: does the VSA store matter beyond the reader's extraction? (RULER, two referee baselines)

Date: 2026-11-16 (US Eastern; 2026-11-17 on the box clock). Written before any B24 run on test data, and before any
B24 model run at all.

Done before this file:
- ruler_b24.py (both arms), b24_verdict.py (tables, bars, prediction checks) and pace/b24*.sbatch were written.
- A CPU self-test ran on seed-7 dev files at 4k (prompt splitting, BM25, retrieval).
- A dev retrieval check ran at 4k: how many gold strings fall inside the 20 retrieved passages. NIAH: 240 of 240.
  VT: 26 of 100 gold variable names.
- The saved B21 test outputs were read for sizing only: pair counts and the character length of each fact list. No
  model was run on them and no answer was produced. Arm F's fact lists are short (at most 30 pairs) on 7 of the 9
  tasks. The long ones are niah_multikey_2 (up to 5,996 pairs, about 151k characters at 128k) and niah_multikey_3
  (up to 1,724 pairs, about 129k characters at 128k).
- The verdict script was exercised on fake arm outputs (copies of B21's predictions). Its store and direct rows
  reproduce results/b21_verdict.txt and results/b19_verdict.txt exactly.

## Why

A referee: the RULER result may come from the reader's extraction rather than from the vector store. B21
(results/b21_verdict.txt) is a prompted Qwen3-8B reader. It reads each text in ~250-character chunks and lists
key-value pairs. The pairs go into a sharded Fourier HRR store, and NIAH / VT answers are read from the store. On the
9 tasks that use the memory, the mean is 99.5-99.9 at 4k-128k. The same Qwen3-8B reading the whole text (B19
direct) gets 84.5 at 128k.

Two questions:
1. Does an exact store matter, or would the same facts, given to the model as text, do as well? (arm F)
2. Does standard retrieval-augmented generation over the same chunks do as well? (arm R)

The result goes into the paper whatever it shows.

## Common to both arms (fixed now)

- **Model:** Qwen3-8B, the PACE HF snapshot used by B19 and B21. bf16, sdpa attention, nothing trained.
- **Samples:** the B19 / B21 test sets (official RULER scripts, Qwen3 tokenizer, seed 42), 100 samples per task.
  - Lengths: 4k, 8k, 16k, 32k, 64k, 128k.
  - Tasks: niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery, vt.
  - 5,400 samples per arm. Every sample is aligned with B21 and B19 by position in RULER's file, and the index and
    gold answer are asserted equal.
- **Prompt:** each RULER input is split into three parts, `prefix + text + "\n" + question` (asserted to rebuild the
  input exactly).
  - The prefix runs up to the last occurrence of the task instruction, which is the input's first line. For NIAH it
    is the instruction line. For VT it is RULER's in-context example plus the instruction.
  - Only the text is replaced, by a one-line note and the arm's content. The prefix, the question and RULER's answer
    prefix are kept verbatim.
- **Decoding:** as B19 direct (ruler_rm.run_direct).
  - Chat template, thinking off. RULER's answer prefix starts the model's turn.
  - Greedy, at most 2 x RULER's tokens_to_generate (NIAH 256, VT 60).
  - One sample at a time, with no batching and no padding.
- **Window rule:** a prompt longer than 32,768 tokens (template and answer prefix included) runs with YaRN factor 4
  (original_max_position_embeddings 32,768), as B19 direct ran 64k / 128k. Every other prompt runs natively. Each
  output row records the prompt length and whether YaRN was on. No input is truncated.
- **Scorer:** official string_match_all (ruler_rm.string_match_all), the scorer of B19 and B21.

## Arm F: facts in the prompt

- **Pairs:** the grounded pairs that B21's store received, read from the saved B21 output lines ("pairs"). Nothing is
  re-read.
  - The order is the one in which they were written to the store, which is reading order. Each (key, value) appears
    once, as B21 de-duplicated it.
  - The pairs include those read from VT's in-context example, as the store's did.
- **Case:** the store lower-cases keys, and its look-up lower-cases the query, so it is blind to key case. The fact
  list writes each key in the case the reader wrote it.
  - The case is recovered by replaying B21's grounding over the saved raw reader outputs and the same chunks, with no
    model.
  - The replay must give back the saved pairs exactly once keys are lower-cased. Otherwise the run stops.
- **Content:** one line per pair, `key: value`, in that order. The text is replaced by:

  `The text has been replaced by the key-value facts read from it, in reading order, one per line as "key: value"
  (an assignment VAR X = VAR Y is the line "X: Y").` (one line), then a newline, then the fact lines.

- **What it isolates:** the model gets the same facts as the store, with no text around them. It must find the
  queried key (NIAH) or follow the chain (VT) by reading. The store does it by exact look-up: NIAH reads the values
  of the queried keys, and VT repeats reads along the chain.

## Arm R: standard RAG (BM25)

- **Chunks:** ruler_rm.chunks (the B21 reader's chunker, ~250 characters at sentence / line boundaries) over the
  text. The B21 reader chunked prefix + text, so only the first chunk boundary can differ.
- **Query:** the question line.
- **Retriever:** Okapi BM25, k1 = 1.2, b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5)).
  - Tokens are lower-cased `\w+` runs, with no stemming and no stop list.
  - Each distinct query term is counted once. Documents are the chunks of one sample.
- **Top-k:** k = 20, ties broken by document order. The kept chunks are put back in document order, one per line.
- **Content:** the text is replaced by:

  `The text has been replaced by the {n} passages of it that best match the question, in their original order.`
  (n = number kept, 20 unless the text has fewer chunks), then a newline, then the passages.

- No query rewriting, no iteration, no reranking. This is the single-shot baseline the referee named.

## Comparators (already run, not re-run)

- **VSA store:** B21's pred_vsa (results/b21_verdict.txt), the system of the paper.
- **Direct:** B19 direct (b19_test/direct): the same Qwen3-8B over the whole text, YaRN x4 at 64k / 128k.

## Statistics

- Per task: n = 100.
- 9-task mean per length: the mean of the 900 sample scores. Tasks have equal n, so this equals the mean of the task
  means. Partial credit is kept: string_match_all gives fractions on multi-answer tasks.
- Intervals:
  - Wilson 95% interval on each 9-task mean, treating the 900 sample scores as a proportion with fractions counted
    as fractions.
  - Paired differences between systems: a 95% bootstrap interval with 10,000 resamples, seed 0. The bootstrap is
    stratified, resampling the 100 samples within each task, with the same draw for both systems.

## Bars

- **S1 (the store matters beyond extraction at 128k):** store minus arm F >= 3.0 points on the 9-task mean at 128k,
  and the paired 95% interval lies above 0.
- **S2:** the same at 64k.
- **R1 (the store beats standard RAG):** store minus arm R >= 3.0 points at every length, and each paired interval
  lies above 0.
- **Reported, not barred:**
  - every per-task and per-length cell;
  - arm F minus direct;
  - arm F minus arm R;
  - a reverse check: any length where arm F beats the store by >= 1.0 with the interval above 0;
  - prompt lengths and YaRN counts.

How the result may be used in the paper (fixed now):
- **If S1 and S2 pass:** the paper may say that the exact store adds X points over the same facts given to the model
  as text, at 64k / 128k.
- **If either fails:** the paper must say that at that length the advantage over direct reading comes from the
  reader's extraction (chunked reading), not from the store. The store's contribution is then exactness, CPU-only
  look-up and no long prompt at answer time, not accuracy.
- **If R1 fails at a length:** the paper says so for that length.

## Predictions (written before any run)

Arm F:
- **PF1:** 9-task mean >= 98.5 at 4k, 8k, 16k and 32k. The fact lists there are short, or native-window lists of
  repeated-format lines.
- **PF2:** below the store at 64k and 128k, with S1 and S2 PASS.
- **PF3:** 9-task mean at 128k between 80 and 95.
- **PF4:** the gap comes from the long fact lists.
  - niah_multikey_3 (about 860 uuid pairs at 64k, 1,700 at 128k, both likely beyond 32k tokens and so with YaRN):
    <= 80 at 64k and <= 60 at 128k.
  - niah_multikey_2: <= 90 at 128k.
- **PF5:** >= 97 on each of the other 7 tasks at every length, VT included. The chain is at most about 25 short lines.
- **PF6:** 9-task mean >= direct at 64k and 128k. These are the same facts with less around them.

Arm R:
- **PR1:** >= 97 on each of the 8 NIAH tasks at every length. Each queried key's chunk shares its rare key words with
  the question.
- **PR2:** VT <= 40 at every length. The question names only the final value, so BM25 finds the first assignment
  (VAR X = value) but not the hops (VAR Y = VAR X), which share no term with the question. Dev 4k: 26 of 100 gold
  names were inside the retrieved passages.
- **PR3:** 9-task mean between 88 and 93 at every length.
- **PR4:** R1 PASS at every length.

## Development (seed-7 dev set; pipeline check only)

- **Dev run:** B21's reader and store (ruler_vsa.py, unchanged) on the seed-7 dev set at 4k and 16k, 20 samples per
  task, 9 tasks. It writes B21-format rows (pairs and raw output). Arm F and arm R then run on those dev samples
  (pace/b24_dev.sbatch).
- **What may change:** the prompts, k and the BM25 settings are fixed above. Dev may change only the wording of the
  two notes or arm F's line format.
  - Condition: the dev outputs show the model misreading the format, for example answering with keys instead of
    values, or saying there is no text.
  - The change must be a dated amendment, written before any test run, with the dev numbers before and after.
- **What may not change:** retrieval settings. Arm R's expected VT failure is the design under test and is not a
  reason to change arm R.

## Compute

- **Where:** PACE, paid account (gts-nimam6-paid), one GPU per array task (partitions gpu-h200, gpu-h100, gpu-a100,
  gpu-l40s).
  - Arm F's long-list tasks (niah_multikey_2 / _3 at 64k and 128k) are restricted to 80 GB or larger cards (H200,
    H100, A100-80GB).
- **Fallback:** if PACE jobs wait more than about an hour, idle A800 GPUs may be used (one process per GPU; other
  users' processes are never touched). This changes only where a run goes. Greedy outputs can differ slightly across
  GPU types (B21 amendment 2), and each row records its GPU.
- **Budget:** at most 40 GPU-hours, expected under 10 (arm R prompts are about 1.5k tokens; arm F prompts are short
  except the long-list tasks).
- **Reporting:** GPU-hours from sacct, and US$ at about $1.17 per GPU-hour, in the verdict.
- **Changes:** any change after this file is a dated amendment written before the affected run.

## Dev record (2026-11-16 US Eastern, before any test run; no design change)

PACE job 15760849 (one H200, 27.5 min walltime, 0.46 GPU-hours). Seed-7 dev set, 20 samples per task.
- **Arm F vs the store:** arm F equals the store on all 18 dev cells.
  - 9-task mean 99.4 at 4k and 99.2 at 16k (store: 99.4 and 99.2).
  - Its three wrong or partial answers are samples where the reader missed the needle, which the store misses too.
  - The replay check (saved pairs = replayed pairs) passed on all 360 samples.
- **Arm R:** 100 on every NIAH dev cell. VT 26.0 (4k) and 20.0 (16k): it names the variable assigned the value
  directly and misses the hops. 9-task mean 91.8 and 91.1.
- **Format:** the model did not misread arm F's fact format anywhere, so the notes and line format stay as written.
- **Prompt sizes:** arm F's longest dev prompts were 3.5k tokens at 4k and 14.2k tokens at 16k (niah_multikey_3), all
  native. Scaling the dev ratios (about 67 tokens per uuid pair, about 15 per number pair) predicts about 115k tokens
  for niah_multikey_3 at 128k. That is inside 131,072 with YaRN x4, so no input needs truncation.

## Compute note (2026-11-17, before any test output existed)
The first submission of arm F's long-list array (PACE job 15761761) combined the gpu-l40s partition with an 80 GB
constraint; Slurm marked it BadConstraints and it never ran. It was cancelled and resubmitted without L40S (job
15761771: gpu-h200, gpu-h100, gpu-a100; H200, H100 or A100-80GB). Compute only; design, bars and predictions unchanged.
If PACE does not start the arrays, idle A800 GPUs may run the same code on the same inputs (user permission 11-17).
