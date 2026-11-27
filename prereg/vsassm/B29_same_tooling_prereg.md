# B29 pre-registration: does model size matter given the same tooling?

Date: 2026-11-18. Written, with its code and verdict generator, before any B29 test run. Requested after the referee's
r29 read ("the comparisons are not like-for-like: the small systems are engineered pipelines, the larger models read
the raw text"; the missing controls are larger and frontier models given the same tooling, "the only clean test of
model size", and the small reader with a plain dict).

## Question
The paper's headline is that a small language model coupled to an exact VSA memory matches or beats much larger
models. Those larger models read the whole text. B29 holds the tooling fixed (the extraction prompt or reader
instructions with their worked examples, the store, the read and answer rules, the engines) and changes only the
model that reads, so that the difference between readers is a difference of model size and nothing else. It also
fills a plain dict from each reader's output in place of the VSA store (RULER) or of the 256 x 256 matrix memory
(logs), to isolate the vector form from the exactness of the store.

## Systems

### RULER (B21's pipeline)
- Fixed: B19/B21 extraction prompt `ruler_rm.FACT_PROMPT` (four worked examples), 250-character sentence chunks, the
  grounding check, the dict (B21's "table": `ruler_rm.facts_of` / `answer_niah` / `answer_vt`, a Python dict of lists),
  B21's sharded Fourier HRR store and read (`ruler_vsa.Store`, d = 8192, 16 pairs per shard, clean-up threshold 0.5),
  the official `string_match_all`, the B19 seed-42 test sets (100 samples per task and length, 9 memory tasks:
  niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery, vt). Decoding as B21: chat template with
  thinking off, greedy, at most 400 new tokens, batches of up to 64 prompts sorted by length (`b29_read.py`, a copy of
  `b26_read.py` with the model as a parameter).
- Readers: Qwen3-8B (B21, banked: the paper's reader), **Qwen3-32B (primary new arm)**, Qwen3-1.7B (secondary, the
  bottom of the ladder). Same prompt, no training, no RULER data seen.
- **Compute shortcut, and why it is exact for the dict.** `answer_niah` returns `table[k.lower()]` for each queried
  key k, and `facts_of` writes `table[k'.lower()]` only for a pair whose key passes the grounding check
  `k'.lower() in chunk.lower()`. So the dict's answer can only depend on the chunks that contain a queried key, in
  reading order. Reading only those chunks (every other chunk's output replaced by "[]") gives the dict's exact
  answer, provided a chunk's output does not depend on the other prompts in its batch. VT scans every key, so VT reads
  every chunk (and there the store is exact too). This cuts the 32B reader's work at 128k from 289,992 distinct reads (302,342 summed over tasks)
  (284,216 of them the distractor-needle chunks of multikey_2/3) to about 2,500. Arm `rel` = this read.
  - **Gate G0 (CPU, run before this file was committed; uses only B21's banked outputs):** B21's own per-chunk outputs
    put through `rel` reproduce B21's pred_table on 5,400 of 5,400 samples (and pred_vsa on all VT samples)
    (`b29_out/ruler/g0_b21_replay.txt`).
  - **Gate G1 (GPU):** Qwen3-8B re-read through `rel` at all six lengths must reproduce B21: per read chunk, the
    output equals B21's banked raw output on >= 99% of chunks (VT: the grounded pairs equal B21's on >= 99% of
    samples) at every length; per sample, the dict's answer equals B21's pred_table on >= 99% of samples at every
    length; and the 9-task mean is within 0.25 points of B21's at every length. If G1 fails, no RULER arm counts; an
    amendment may be written only before any 32B or 1.7B test output is read.
- **Store arm.** The store's answer depends on every pair (shard count, cross-talk, clean-up vocabulary), so it needs
  every chunk read. Arm `full`: Qwen3-32B reads every chunk of all 900 samples at 128k, decoded with vLLM 0.11
  (`b29_vllm.py`; same prompts, chat template, greedy, 400 tokens; HF generate would need about 35 A800 GPU-hours
  here). Dict and store then answer from the same outputs, so the store-vs-dict comparison has no engine difference
  inside it; the engine difference to `rel` (HF) is measured (RS3).
- Not run, by design: Qwen3-32B reading the whole 128k text (KV cache does not fit one A800 next to the weights;
  vLLM tensor-parallel on 900 samples about 40 GPU-hours). No banked per-sample result exists; the RULER leaderboard
  row (Qwen3-32B, 13-task average 85.6 at 128k, 500 samples per task) is quoted as unpaired context only.

### Operational logs (the paper's three held-out one-day sets)
- where: 414 location questions, 40 test2 logs, pipeline apps/real at commit cd9de8d (copied verbatim to
  `b29_frozen/cd9de8d/`); banked 1.7B result 0.959.
- impact: 230 impact questions, 40 nsr1 logs, NSR1 arm 1 (FHRR rule memory, rules from one worked trace per question
  type); also arm 2a (count-table rules) and 2b (graph query); pipeline `nsr/vendor_707351b`; banked 0.930.
- window: 460 past-window questions, 40 nsr2 logs, NSR3 history engine; pipeline `nsr/vendor_707351b`; banked 0.874.
- Fixed, per set: the pinned pipeline imported read-only (`python3 -I -B`), `--verify` on: prompt C with its worked
  examples (`b14.PROMPTS[("C", domain)]`), the forced choice among named places, the one-word check, `b14.parse`, the
  registry, the connectors, `SiteMemory` and its 256 x 256 matrix memory, the engines and the registered scorers
  (`run_real.ok`; `nsr/nsr1_score.gold`; `nsr/nsr3_score` replay). `b29_logs.py` swaps only the reader object
  (`b14.Reader` replaced by a factory) and logs every call (prompt, text, tokens).
- Readers: Qwen3-1.7B + the distilled LoRA `b16_models/student_s0` (banked: the paper's reader; and its re-run through
  `b29_logs.py`, gate G2), **Qwen3-32B prompted, no adapter (primary new arm)**, Qwen3-8B prompted (secondary; prompt
  C on Qwen3-8B is B15's reader, the student's teacher), **GPT-6.1 Sol prompted (frontier)** through the api_zheng
  gateway (`apps/real/api_reader.zheng_reader`, reasoning effort low, the product's strong-tier setting; never an
  OpenAI key). Thinking off for the Qwen readers, as for the shipped reader.
- Frontier readers (registered subset and order). GPT-6.1 Sol (low) on all three sets, in the order impact, window,
  where (the dev pilot put a whole set at about USD 7-8 at list price). Claude Opus 5.5 through the user's endpoint
  (`~/nsengine/runs/claude_api.py`, effort low, one message per prompt; its relay caches the prompt, about USD 30 per
  set) on the impact set, then on the window set if the frontier total is then below USD 90. Spend stop: no new set
  starts once the list-price total of all B29 frontier calls exceeds USD 120 (cap 150). Each set runs as one job over
  its 40 logs (the pipeline reads all logs of a split line by line in lockstep). A call that fails after the client's
  retries returns "" (= no events); if more than 0.5% of a set's calls fail, the set is re-run once (completed calls
  replay from the cache at no cost); remaining failures are reported with a sensitivity analysis excluding the
  affected logs.
- **Gate G2:** the 1.7B re-run through `b29_logs.py` must give the banked answer on >= 97% of the questions of each
  set and an accuracy within 0.015 of the banked one. If G2 fails on a set, that set's comparisons use the re-run as
  the small reader (same code path as the new arms), and this is stated. (Dev check before registration: 8 dev logs,
  final boards 8/8, where answers 83/83, per-line events 1,839/1,840, impact and window dev accuracy identical.)
- **Dict arm:** each reader's logged calls are replayed on CPU with a plain dict in place of the matrix memory
  (`b29_logs.dict_memory`: the same name lists and codebook order, so the prompts are identical; the same
  canonicalisation; move = assignment, swap = exchange of the two places' contents). A replay that meets a prompt not in
  the log has diverged and is reported. For impact, arm 2a (count-table rules) is the dict for the rule memory.

## Comparisons, intervals and the registered reading
- RULER: paired by sample, 9-task mean, 95% stratified paired bootstrap (100 samples resampled within each task,
  10,000 resamples, numpy seed 0; as `b24_verdict.paired`). Margin 1.0 point.
- Logs: paired by question, 95% cluster bootstrap over logs (`cluster_bootstrap.cl_boot`: whole logs resampled with
  replacement, 10,000 resamples, numpy seed 0, pooled ratio); the stratified-by-scenario interval is also reported.
  Margin 0.05.
- For each set and larger reader, d = larger - small (same tooling), interval [lo, hi], margin m:
  - **"model size matters given the same tooling"**: d >= m and lo > 0;
  - **"a small reader suffices"**: hi < m (the larger reader cannot be shown to add m);
  - **"larger reader worse"**: hi < 0;
  - otherwise **inconclusive**.
  Why these margins: 1 point is four times B21's run-to-run spread on RULER (0.25); 0.05 is below every gap the
  paper interprets on the logs (0.42-0.67 against Qwen3-32B reading the log; the eight-day frontier gap 0.145).
- **Ceiling caveat (stated now):** on RULER the 8B reader is at 99.6 at 128k, so a larger reader can gain at most 0.4
  points: "size matters" cannot be reached there by construction. RULER can tell only whether the small reader already
  suffices (and whether a larger one is worse), and, with the 1.7B rung, whether 8B is needed. The logs (0.874-0.959,
  every miss of the 1.7B pipeline a misread fact) are where a larger reader has room to show an effect.

## Bars
- **RS1 (primary, RULER):** the reading of d = 32B - 8B (dict, 128k).
- **RS2:** with the 32B reader's full read, the store's answer equals the dict's on >= 99% of the 900 samples at 128k.
- **RS3:** the 32B dict answers of `full` (vLLM) and `rel` (HF) agree on >= 98% of the 900 samples at 128k.
- **RS4 (secondary):** the reading of d = 1.7B - 8B at 128k.
- **L-where / L-impact / L-window (primary, logs):** the reading of d = 32B prompted - 1.7B fine-tuned, per set.
- **LF (per set run):** the reading of d = frontier prompted - 1.7B fine-tuned, for GPT-6.1 Sol (impact, window, where)
  and Claude Opus 5.5 (impact; window if run).
- **L8 (secondary):** the reading for Qwen3-8B prompted.
- **LD:** for every reader and set, the dict replay gives the matrix memory's answer on >= 99% of the questions.
- Reported, not bars: the tooling effect of each model (the same model in the pipeline minus reading the whole log):
  Qwen3-32B on all three sets (banked whole-log, budget-forced: 0.543 / 0.278 / 0.202), GPT-6.1 Sol on impact and
  window (banked whole-log, OpenAI direct, effort xhigh, strict: 0.952 / 0.904; the pipeline reader runs at effort
  low, so this comparison favours the whole-log arm); every length 4k-64k on RULER.
- **What the paper may say** (decided now): if RS1 and the three L bars read "small reader suffices", and LF reads
  "small reader suffices" or "inconclusive" with d <= 0.02: given the same tooling, a 32B (and a frontier) reader does
  not beat the small reader, so the headline gains are gains of the tooling (exact external state + rules), not of
  the small model, and a small reader suffices for it. If any L or LF bar reads "size matters": model size still
  matters given the tooling on that set, and the headline must be restated as a tooling effect plus a size effect of
  the measured magnitude. Inconclusive sets are reported as such with their intervals.

## Predictions (stated so that they can be wrong)
- G1 passes (B26 saw 559/560 chunks reproduce across batches).
- RS1: d32 in [-0.5, +0.4] at 128k: "small reader suffices". RS2 holds (900/900). RS3 holds.
- RS4: the 1.7B reader is at least 1 point below the 8B at 128k (interval below 0).
- Logs, 32B prompted vs 1.7B fine-tuned: |d| <= 0.03 on each set; "small reader suffices" on at least two of the three.
- Frontier (GPT-6.1 Sol and Opus 5.5): d in [-0.02, +0.04] on every set run; "small reader suffices" on each.
- Qwen3-8B prompted: within 0.03 of the 1.7B student (its teacher on stories).
- LD holds for every reader (the matrix memory is exact below 256 names, so the dict changes nothing).
- Tooling effect of Qwen3-32B >= +0.30 on all three log sets; of GPT-6.1 Sol within +-0.05 on impact and window.

## Development record (before this file was committed; dev data only)
- RULER dev (seed-7 sets, 4k and 16k, 20 samples per task), Qwen3-32B through `rel`: 1,000 reads in 355 s on one
  A800 (HF generate, no OOM); dict accuracy 100% on 8 of 9 tasks, niah_single_1 95% at both lengths (the reader
  wrote the key "youngful-fratricide", failing the grounding check). vLLM 0.11 tensor-parallel 2 fails on these PCIe
  cards (custom all-reduce kernel); tensor-parallel 1 with the custom all-reduce disabled reads the 7,952 needle-dense
  16k dev prompts in 688 s. Cross-engine on the 509 dev prompts read by both engines: identical text on 469 (vLLM
  wraps the list in a json code fence where HF writes "JSON:"), identical parsed pairs on 509 of 509.
- Logs dev (8 logs, vendored pipeline): the 1.7B re-run reproduces the banked dev run (above). Qwen3-32B prompted:
  impact 0.913 (1.7B: 0.935), window 0.891 (0.902), location 0.952 (0.952); 760 s on one A800.
- Frontier dev: GPT-6.1 Sol (low) answers in the expected JSON (mean 12 completion tokens per call, about 400 prompt
  tokens); Claude Opus 5.5 (low) likewise, at about USD 0.006 per reader call (relay cache writes).
- G0 (CPU): 5,400 / 5,400 (above).

## Verification
- Second scorer `b29_check.py`: own string match and means on RULER; on the logs, gold from a different derivation
  (impact: `nsr1_catalog.answer` on the true state; window: `nsr3_questions.union_answer` over `truth_states`;
  location: own comparison with the stored answer); it must agree with the verdict on every accuracy.
- Every RULER sample on which the 32B and 8B dict answers differ is read by hand; on the logs, up to 20 questions per
  set on which the 32B and 1.7B pipelines differ (and the frontier's), classified by the reading error behind them.
- After the verdict: literature check (tool-augmented and program-aided LMs, LMs with symbolic memory such as ChatDB,
  extraction-then-reason pipelines and reader size), recorded as supports / conflicts / no precedent.

## Compute and spend
A800 box only (idle GPUs, never GPU 3, never another user's process; `b29_run.sh` claims GPUs and writes the ledger),
cap 60 GPU-hours (dev pilots included); PACE only if the A800 is full. Estimated: G1 0.5, 32B `rel` 1.5 (14,207
distinct prompts over the six lengths), 1.7B `rel` 0.3, 32B `full` (vLLM, 289,992 distinct prompts at 128k) about
5-7, logs 1.7B 0.5, 8B 1.5, 32B about 2 GPU-hours. Frontier: cap USD 150 at list price (GPT-6.1 Sol through api_zheng,
which has no balance meter: tokens from the client's meter x `api_reader.PRICE`; Opus 5.5 from `claude_api`'s
ledger). Per-sample and per-question outputs are saved under
`b29_out/`; the verdict is `results/b29_verdict.txt`, generated by `b29_verdict.py`.

## Code (committed with this file, before any test run)
`b29_read.py`, `b29_vllm.py`, `b29_ruler.py`, `b29_logs.py`, `b29_frozen/cd9de8d/` (verbatim), `b29_run.sh`,
`b29_verdict.py`, `b29_check.py`. Test prompt files (no model output; not committed, regenerated by
`b29_ruler.py prompts`): `b29_out/ruler/test_rel_prompts.jsonl` (14,207 distinct prompts, md5
7e3eb4c17e817b0d9cb0c554d2770bbe) and `test_full_prompts.jsonl` (289,992, md5 71ab22318fd9c1130ba415fbedefd23d).
