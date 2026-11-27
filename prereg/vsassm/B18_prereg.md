# B18 pre-registration: BABILong (public benchmark) with the readable memory (before any B18 code or run)

Date: 2026-10-20. BABILong (Kuratov et al., NeurIPS 2024) hides bAbI facts inside PG19 book text, from 0 to 1M+
tokens. Published finding: general LLMs use 10-20% of their context; the best entries are models fine-tuned on
BABILong (ARMT 137M ~97% and Mamba 130M ~92% on QA1-5 at 128k), which fall to near 0 at 512k and beyond.

## Scope (fixed now)

In scope, the state-tracking tasks the product is built for: **qa1, qa2, qa3, qa5, qa6, qa7, qa8, qa9, qa10, qa11,
qa12, qa13** (where is X, objects carried by people, where was X before Y, who gave what to whom, yes / no about
locations incl. negation and "either ... or", how many / what is X carrying, coreference, conjunction). Out of scope
(reasoning-system track): qa4 and qa14-qa20. Results are reported per task and as the mean over the 12 in-scope tasks,
and also over QA1-5 minus qa4 for comparison with the leaderboard's QA1-5 mean (qa4 is out of scope: reported as
not attempted, not as 0).

## System

- **Reader:** reads one sentence at a time (up to 3 earlier sentences as context) and returns events: go(person, place),
  take(person, object), drop(person, object), give(giver, object, receiver), not_at(person, place),
  either(person, place, place). Pronouns and "X and Y" are expanded by the reader. Book sentences give no events.
- **Memory:** the readable vector memory (delta-rule writes, codes per name): each object points to its holder or
  place, each person to their place (a chain read hop by hop); a drop sets the object to its holder's current place.
  Plain logs keep each entity's sequence of places (for "where was X before Y") and the gives (for qa5).
- **Answers** come from the memory and logs by question type (fixed rules, no model).
- **Development** uses only the BABILong *training* split (RMT-team/babilong-train-5k-samples), never the test data.
  Teacher: Qwen3-8B, prompt F (worked examples). Product reader: Qwen3-1.7B student distilled on teacher labels of
  training-split stories (bAbI sentences and book sentences). No test sample is used for development.

## Test

RMT-team/babilong-1k-samples (test): the first 100 samples of each in-scope task at 0k, 4k, 16k, 64k and 128k.
If longer test splits are available (512k, 1M), 20 samples per task. Scoring: the official answer (lower-cased,
stripped); lists as sets; "none" / "nothing" equivalent for empty answers.

## Claims and bars

- **P1** mean accuracy over the 12 in-scope tasks >= 0.90 at every length up to 128k.
- **P2** the drop from 0k to 128k is <= 0.05.
- **P3** (if longer splits exist) mean >= 0.85 at the longest length tested.
- Reported: per-task accuracy; Qwen3-8B (thinking) answering directly at 0k-16k on the same samples (context limit);
  the published leaderboard numbers for reference; generated tokens.

## Predictions (stated so they can be wrong)

P1 and P2 hold; the hardest tasks are qa3 (before) and qa10 (indefinite); accuracy does not depend on length because
book sentences produce no events, except where the reader invents events from book text (the main risk).

## Amendment 1 (development, training split only)

Teacher dev, 0k, 50 samples: qa1 0.94, qa2 0.96, qa3 0.96. All inspected errors: when a sentence repeats an earlier one,
the reader restates the context's events, re-applying old moves. Added a grounding check before the memory: an event
is kept only if its place (go / not_at / either) or object (take / drop / give, or a pronoun) occurs in the current
sentence. No test data seen.

## Amendment 2 (development, training split only)

Teacher dev with grounding: 0k qa1 1.00, qa2 .94, qa3 .92, qa5 .94, qa6 .98, qa7 .88, qa8 .86, qa9 .94, qa10 .96;
4k qa1 .967, qa2 .90, qa3 .867 (identical on A800 and PACE H200 at 0k). Error traces: (1) restated context events about
the same object pass the name check ("Mary grabbed the milk" -> take + a restated drop); (2) book sentences produce
events about book characters who share bAbI names ("Mary came into my room") or book places. Changes:
- **Cue check:** an event type needs a general English cue word in the sentence (go: went/moved/travelled/...;
  take: got/took/picked/...; drop: left/dropped/put down/...; give: gave/handed/passed/...; not_at: not / no longer;
  either: either). Word lists written from general English, not from test data.
- **Student fine-tuned on the BABILong training split** (this places B18 in the leaderboard's *fine-tuned* category,
  like ARMT and Mamba): Qwen3-1.7B + LoRA on teacher labels of training-split stories (qa1-qa10 at 0k / 4k / 16k) plus
  bAbI-original training stories for qa11-qa13 (facebook/babi_qa, en-10k); a sentence counts as a bAbI fact only if
  every word is in the vocabulary of the 0k training stories, and every other sentence is labelled "no events"; labels
  pass the grounding and cue checks. The test split is still unseen.

## Amendment 3 (before any test run of our system)

The 1k-samples test set has lengths above 0k only for qa1-qa5. The test set is therefore **RMT-team/babilong**
(100 samples per task and length, all tasks, 0k ... 1M): the in-scope tasks at 0k, 4k, 16k, 64k, 128k (P1, P2) and
512k, 1M (P3), n = 100 (fewer at 512k / 1M if compute requires, stated with the result). Qwen3-8B direct (thinking) is
run on the 1k-samples 0k set (all tasks, 50 each) and on this set at 4k (50) and 16k (25).
Correction to amendment 3: RMT-team/babilong has qa11-qa13 only at 0k. P1 / P2 at lengths above 0k are therefore over
the 9 tasks qa1, qa2, qa3, qa5-qa10; qa11-qa13 are reported at 0k only.

## Amendment 4 (development, training split, before any test run of our system)

qa4 (static directions) added to the scope so the QA1-QA5 mean can be compared with the leaderboard: reader event
["rel", place, direction, place], answers "what is D of P" / "what is P D of". The official BABILong metric
(babilong/metrics.py compare_answers; labels = answers seen in the training split) is now the primary score.
Student development (training split samples 300-349, never distilled; seeds 0 and 1): 0.98-1.00 on every task at 0k,
4k and 16k except qa4, 0.52, because questions use the opposite direction of the stated fact; a stated relation is now
also stored in the opposite direction (north/south, east/west).

## Amendment 5 (during the test run; disclosed)

- 128k qa7-qa10 ran out of GPU memory (very long book "sentences" in a batch of 256); rerun with length-budget batching
  (same model and decoding; outputs do not depend on batching).
- qa11-qa13 have no BABILong training data, so the metric's labels were empty and they first scored 0; their official
  labels are the room names (as qa1). Rescored from the saved predictions: 0.64 / 0.59 / 0.46.
- Cause of the low qa11-qa13 scores (found after the test): the bAbI-fact filter used the vocabulary of the qa1-qa10 0k
  training stories, which lacks "he", "she", "they", "afterwards", "following", so the qa11-qa13 training sentences
  were labelled as book text. **B18b** (follow-up, its own result): vocabulary + bAbI-original qa11-qa13 words,
  relabel, retrain; evaluated on all tasks at 0k and 4k plus qa11-qa13. The B18 numbers stay as measured.

## Amendment 6 (frontier API references; written before any API call)

User request 2026-10-21: compare with Claude Opus 5.5 and GPT 6.1. Budget approved by the user: about $410.
- **Models:**
  - `claude-opus-5-5` through the Anthropic Message Batches API. Thinking is always on; effort is set to `high`.
  - OpenAI's GPT-6.1 Sol (`gpt-6.1-sol`) cannot be reached: there is no OpenAI key, and the gateway does not list it.
    The previous model, `openai/gpt-6-sol` (1.05M context, $2/$10 per MTok), runs through CommonStack with
    reasoning effort `high`. It is reported under that name, never as "GPT-6.1".
  - GPT-4 is quoted from the leaderboard row (GPT-4-0125), not rerun.
- **Samples:** the first 25 official test samples (RMT-team/babilong) of qa1-qa5 at 0k, 4k, 16k, 64k and 128k, and the
  first 5 at 512k and 1M. Our system ran on exactly these samples (a subset of its 100 / 25), and its accuracy is also
  reported on this subset.
- **Prompt:** the direct-answer prompt of the Qwen3 references (b18_llm.py), unchanged. Zero-shot, max 16k output tokens.
- **Metric:** the official one (correct_official, labels from the training split), applied to the last line of the
  answer.
- **Failures:**
  - A request over a model's context window is reported as "exceeds context", not as wrong, and it is not sent. Token
    count comes from `count_tokens` for Opus; for GPT, the gateway's error is used.
  - API errors after retries are reported separately.
- **Category:** our student is fine-tuned on the BABILong training split; the frontier models are zero-shot. Tables keep
  the two apart.
- **Prediction (no tuning on these numbers):**
  - Both frontier models score >= 0.9 on the QA1-5 mean at 0k-16k.
  - Both fall below our system (on the same subset) at 128k and beyond.
  - Opus 5.5 may not fit 1M samples into its 1M window.
- **Routing change (2026-10-22, before any scored call).** The Anthropic direct account has no credit
  (count_tokens returned 400). Both models therefore run through OpenRouter, each pinned to its official provider
  with fallbacks off: `anthropic/claude-opus-5.5` on Anthropic and `openai/gpt-6.1-sol` on OpenAI. GPT-6.1 Sol is
  available there, so the substitute `gpt-6-sol` is dropped. Reasoning effort is `high`. The cost of each call is the
  amount OpenRouter billed for it.
- Caps: $190 (Opus) and $220 (GPT); total within the approved $410. Lengths run short to long. A length whose
  estimated cost would pass the cap is not started, and the result says so. Opus gets no `count_tokens` pre-check: a
  sample over its window is the provider's 400 "exceeds context" error, which is not billed.
- Sanity call before the run (not scored): one toy bAbI question per model; both answered "garden".
- **Scoring note (found during the run, by reading the Opus qa3 16k errors).**
  - The answer-line rule above ("last line") misreads answers written answer-first ("**bathroom**" followed by an
    explanation).
  - Frontier models are therefore scored with the most generous of three readings, so as not to undersell them: the
    official rule applied to the last line, to the first line, and to the whole output.
  - All three are reported. Opus qa3 16k is 0.68 / 0.76 / 0.76 under those readings, and 0.76 under the generous one.
  - The remaining Opus qa3 errors are real (wrong room).
- **Scoring rule, final.** This replaces the "most generous of three readings" note above, at the user's request
  (2026-10-22: accurate, no inflated numbers); the max over readings was chosen after seeing answers and inflates.
  - **Primary:** one fixed reading, used for every system: the official rule on the whole output. This is how the
    leaderboard scores GPT-4.
  - **Second scorer:** babilong/metrics.py compare_answers from the official repository, copied verbatim into
    `babilong_metrics_official.py`. Disagreements with the primary scorer are reported.
  - **First-line and last-line readings:** sensitivity only.
  - **What counts as wrong:** API errors (after one retry pass) and truncated outputs. Errors are never dropped from
    the denominator.
  - **Too long for the context window:** marked "exceeds context"; in means it counts as 0, as on the leaderboard.
  - **Reporting:** each cell gives n and a 95% Wilson interval. Each cell is 25 samples (5 at 512k / 1M), so a 1-2
    sample gap is not a difference.
  - **Audit:** a random sample of answers judged correct is read by hand, to look for false positives.
  - Checked before scoring: GPT-6.1 Sol uses reasoning at effort high (109 reasoning tokens on one 16k qa3 sample),
    and Opus thinks too (reasoning tokens > 0 in the sanity call). Both run on the pinned providers.

## Amendment 7 (2026-10-23, 10M tokens; before any 10M run)

The design has no context window: the reader streams sentences, and the memory has a fixed size. To test this beyond
the 1M already measured, the same B18 student (seed 0, unchanged), memory and answer rules are run at 10M tokens on
the official test set (RMT-team/babilong, data/{task}/10M.json).
- **Samples:** the first 5 samples of each of qa1-qa5 (n = 25), one PACE job per task. Compute is the limit, not the
  method.
- **Metric:** the official rule, as B18.
- **Reference:** ARMT's published 10M numbers (Rodkin et al. 2024, Table 1): QA1 89.4, QA2 84.4, QA3 37.0, QA4 91.5,
  QA5 80.9; average 76.6.
- **Bar P5:** QA1-5 mean at 10M >= 0.95.
- **Prediction:** P5 holds. Book sentences produce no events, so accuracy should not depend on length. The risk is
  more book sentences mistaken for facts as the text grows (10x the 1M exposure).
- **Cost estimate:** about 23 H200-hours on the paid account (about $27).

## Amendment 8 (2026-10-23, after the 10M jobs timed out; before any 10M result exists)

- **What happened.** PACE job 14306363 (one task per array element, n = 5 at 10M) hit its 10 h limit on all five
  tasks with no output: `run()` batches sentence i across samples, so with 5 samples each generate call carries 5
  prompts, and a 10M sample has about 340k sentences. Measured at 1M: 25 samples, ~34k sentences each, 8,100-8,450 s
  per task (about 0.24 s per batch); at 10M that is about 340k batches, about 23 h. No prediction was produced, so no
  10M number has been seen. Compute lost: 5 x 10 H200-hours.
- **Change.** `run_blocked()`: one sample at a time, sentences read in blocks of 2,048 (each block one call to the
  same length-budget reader, bs 256). A prompt depends only on the text (the sentence and the three before it), never
  on earlier outputs, so the prompts are the same; events are applied in sentence order. Same model, adapter,
  prompt, decoding, memory and answer rules. Per-sample checkpoint (a finished sample is never re-read) and progress
  lines.
- **Equivalence check before any 10M run:** 1M qa3, the first 5 test samples, with the new path: predictions must
  equal the banked B18 1M predictions on those samples (5/5). If not, the 10M run does not start and the difference
  is reported.
- **Unchanged:** samples (first 5 per task of qa1-qa5 at 10M), metric, bar P5 (QA1-5 mean >= 0.95), prediction.
- **Cost:** the check about 0.3 H200-hours; 10M about 25 samples x ~0.4 h, about 10 H200-hours on the paid account
  (about $12). Spend so far on the failed attempt: about 50 H200-hours.
