# B23 pre-registration: the same 1.7B model fine-tuned end to end on BABILong, with no memory (referee baseline)

Date: 2026-11-16 (US Eastern; 2026-11-17 on the box clock). Written before any B23 code reads test data and before any
B23 GPU run. Done before this file (no test data involved): the training split was downloaded; b23.py (prep, train,
eval, gate) was written and smoke-tested on CPU with a tiny random Qwen3 and fake data; the prep job built the training
and development examples from the training split and counted their tokens (numbers below).

## Why

A referee: "small beats large reflects task fine-tuning, not the memory". Our BABILong system (B18) is Qwen3-1.7B with
a LoRA reader distilled on BABILong / bAbI training sentences, writing events into a 256 x 256 matrix memory. QA1-5
mean (official rule, n = 500 per length): 99.6 at 0k, 97.8 at 4k, 99.2 at 16k, 99.6 at 64k, 99.8 at 128k (100 at
512k and 1M, n = 125; 100 at 10M, n = 25). The paper has no run of the same small model fine-tuned on the task
without the memory. B23 is that run. Its result goes into the paper whatever it shows.

## Model and fine-tuning (fixed now)

- **Base model:** Qwen3-1.7B, the snapshot of the B18 reader (PACE HF cache). bf16, sdpa attention, gradient
  checkpointing.
- **LoRA:** the B18 reader's configuration.
  - r 16, alpha 32, dropout 0, on q, k, v, o, gate, up and down projections.
  - AdamW, lr 2e-4, weight decay 0, 50 warm-up steps then linear decay to 0, gradient clipping at 1.0.
  - 16 samples per optimizer step (one sample per forward pass, no padding, gradient accumulation).
  - One epoch; seed 0.
- **Input:** chat template with thinking off. The user turn is the direct-answer prompt of the B18 Qwen3 and frontier
  references (b18_llm.py, verbatim): the whole text, then "Answer the question about the people and objects in the
  text above. Question: ... Give only the answer: a single word (...), or a comma-separated list ...".
- **Target and loss:** the official answer string plus the end-of-turn token. The loss is on the target tokens only.
- **Training data:** RMT-team/babilong-train-5k-samples, the training split, built from bAbI training stories and
  PG19 training books. Only qa1-qa5 are used. Nothing from the test release is used.
  - Index i holds the same bAbI story at every length. This was checked: 200 of the first 200 0k stories occur inside
    the 4k samples of the same index. Each length therefore takes its own index range, and no story is trained on twice.
  - The ranges per task are as follows.

    | length | indices | per task |
    |---|---|---|
    | 0k | 0-1999 | 2,000 |
    | 1k | 2000-2999 | 1,000 |
    | 2k | 3000-3499 | 500 |
    | 4k | 3500-3999 | 500 |
    | 8k | 4000-4299 | 300 |
    | 16k | 4300-4499 | 200 |
    | 32k | 4500-4599 | 100 |
    | 64k | 4600-4649 | 50 (arm Y only) |

  - qa3 and qa5 have no 1k file in the training split, and qa4's 1k file has only 999 samples. Every other file used
    has 5,000. The 1k slot of these three tasks (indices 2000-2999) is therefore read from the 2k file.
  - Arm N has 4,600 samples per task (23,000 in all, about 1,437 optimizer steps). Arm Y has 4,650 per task (23,250,
    about 1,453 steps). Samples are shuffled together with seed 0.
  - Token counts (Qwen3 tokenizer, with the template; first 40 per task and length; prep job 15758123): mean / max
    per sample.

    | length | train mean / max | dev mean / max |
    |---|---|---|
    | 0k | 203 / 1,284 | 211 / 893 |
    | 1k | 1,334 / 1,799 | - |
    | 2k | 1,737 / 1,898 | - |
    | 4k | 3,640 / 4,336 | 3,614 / 3,887 |
    | 8k | 7,474 / 8,602 | - |
    | 16k | 15,084 / 16,540 | 15,366 / 16,543 |
    | 32k | 30,691 / 33,113 | 30,641 / 31,792 |
    | 64k | 61,799 / 66,626 | 61,784 / 64,063 |

  - That is about 65M training tokens for arm N and 80M for arm Y. The longest 32k sample (33,113 tokens) is inside
    max_position_embeddings, so arm N drops nothing.
  - Label sets of our scorer (from the training split) equal the official TASK_LABELS for qa1-qa5.
- **Development split:** indices 4900-4919 of each task at 0k, 4k, 16k, 32k and 64k, from the training split. That is
  20 per task and 100 per length. These samples are never trained on, and 0 of the 100 dev stories at 0k occur among
  the 4,900 earlier indices. Dev is used only for the pipeline check, the validity gate and the choice of the primary
  arm.
- **Effective training data versus our reader:** the B18 reader was distilled on sentences from about 330 training
  stories per task. The baseline sees 4,600-4,650 whole samples per task. This is deliberate: the baseline gets more
  task data than our reader, not less.

## Two arms (handling of positions beyond the native window); the primary arm is chosen on dev

- **Arm N, the reference protocol.**
  - Trained with native RoPE on 0k-32k, the training lengths inside Qwen3's 32k native window. Training samples
    longer than 40,960 tokens (max_position_embeddings) are dropped; none are expected.
  - Test: native at 0k-32k. At 64k and 128k, YaRN with factor 4 and original_max_position_embeddings 32,768, exactly
    as the Qwen3 references were run (ruler_rm.py / lme_rm.py; B19, B20).
- **Arm Y, long-context fine-tuning.**
  - YaRN factor 4 is on during training and at every test length.
  - Trained on 0k-64k; 64k is the longest length in the training split.
- **Choice of the primary arm:** the arm with the higher dev QA1-5 mean, averaged over the five dev lengths with equal
  weight. An exact tie goes to arm Y. The choice uses dev numbers only. No test output is read before both dev results
  exist and the primary arm is recorded. Both arms are run on the test set and both are reported.
- **Seed replicate:** arm Y is also trained with seed 1 and tested (same data, new order), as a replicate. The bars are
  judged on the seed-0 primary arm, and the verdict says whether seed 1 agrees. If the spend projects past the cap,
  this replicate is the first thing cancelled.

## Validity gate (bar 0; on dev, per run, checked automatically before that run's test)

- **Rule:** dev QA1-5 mean at 0k >= 0.90. The script stops before the test if it fails.
- **Remedy (the only one allowed):** one more epoch over the same data, continuing from the adapter with a new
  shuffle and the same schedule. Then dev is run again, and the run's test uses whichever epoch has the higher
  five-length dev mean.
- **If it still fails:** the test runs anyway, and the result is labelled "under-trained baseline".
- Any other change needs a dated amendment written before the affected run.

## Test

- **Data:** RMT-team/babilong, the official test release used by B18, data/{qa1..qa5}/{length}.json. All 100 samples
  per task are used at 0k, 4k, 16k, 32k, 64k and 128k, which is 500 per length and 3,000 per run. These are the B18
  samples. B18 has no 32k run, so 32k is reported without a comparison.
- **Decoding:**
  - Greedy, thinking off, at most 32 new tokens.
  - One sample at a time, with no batching and no padding, because batched greedy decoding can change outputs.
- **Window rule:**
  - Before the first test run, the prompt tokens of the test inputs are counted (b23.py testlen: lengths only, no model,
    no labels).
  - The YaRN factor is 4. It is raised to ceil((longest prompt + 32) / 32,768) at any length where the longest prompt
    does not fit in 131,072 tokens.
  - Arm N stays native at 32k only if the longest 32k prompt plus 32 tokens fits in 40,960. Otherwise its 32k run uses
    YaRN factor 4.
  - No input is truncated.
- **Primary scorer:** our implementation of the official BABILong rule, b18.correct_official, applied to the whole
  output. This is the rule used for the frontier references (B18 amendment 6, final).
  - It reads the first sentence and matches whole words. Labels named in the question are ignored.
  - Labels are the answers seen in the training split (b18.labels; the 0k, 4k and 16k training files, as in B18).
- **Second scorer:** the verbatim official compare_answers (babilong_metrics_official.py, official TASK_LABELS), also
  on the whole output. Disagreements between the two scorers are reported.
- **Per cell:** k / n, accuracy and a 95% Wilson interval; outputs that contain no label; outputs that hit the
  32-token limit.
- **Comparison with our system** at 0k, 4k, 16k, 64k and 128k.
  - It is paired on identical questions, using the banked B18 per-question results
    (b18_test/student_s0/{L}_{task}_*/{L}_{task}.jsonl, field ok_official). Rows are matched by order and checked
    on question and target.
  - Reported: the difference in points, the discordant pairs, and the exact two-sided McNemar p.
- **Frontier subset:** the baseline on the first 25 per task (n = 125), the questions on which GPT-6.1 Sol and Claude
  Opus 5.5 were run (b18_api_verdict.txt). Reported without a bar.
- **Not run:** 512k, 1M and 10M. Those inputs exceed the 131k-token YaRN window. The verdict says that the baseline
  cannot read them, rather than scoring them as 0.

## Bars (seed 0, primary arm)

- **Bar 0:** the validity gate above.
- **Bar 1 (the memory adds beyond end-to-end fine-tuning at 128k):** ours minus baseline >= 5.0 points, and exact
  McNemar p < 0.05.
- **Bar 2:** the same at 64k.
- **Reporting rule:** no advantage of either system is claimed at a length where McNemar p >= 0.05; there the two
  are reported as not distinguishable at n = 500.

### How each outcome will be read (fixed now)

- **Bars 1 and 2 pass:** at long context the result comes from the memory, not from task fine-tuning alone. The
  paper gives the gap at each length.
- **Bar 1 or bar 2 fails:** at that length the same small model, fine-tuned end to end, matches our system. On
  BABILong up to 128k, then, the advantage over large zero-shot models is explained by task fine-tuning. What remains
  for the memory is reading inputs beyond the window (512k-10M), which the baseline cannot read at all. The paper says
  so.
- **Either way:** both arms and every length go into the paper.

## Predictions (stated so they can be wrong)

- **Bar 0 passes for every run,** with dev QA1-5 at 0k >= 0.95.
- **Dev:** arm Y >= arm N on the five-length mean, so arm Y is the primary arm.
- **Test, primary arm, QA1-5 mean in points:**
  - 0k: 97 (90-100).
  - 4k: 95.
  - 16k: 93.
  - 32k: 90.
  - 64k: 85 (65-97).
  - 128k: 72 (45-92).
- **Bars 1 and 2 both pass.**
- **qa3 (three supporting facts)** is the baseline's weakest task at every length from 4k on.
- **Arm N at 128k** is at least 5 points below arm Y.
- **Short lengths:** at 0k, 4k and 16k the baseline is within 3 points of ours. On the bare and short stories, then,
  fine-tuning alone suffices.
- **Large zero-shot models:** at 128k, on the frontier subset, the baseline is below GPT-6.1 Sol (96.8) and Opus 5.5
  (97.6).

## Compute and budget

- **Estimate, from FLOPs and the measured token counts:**
  - Training: about 1.3e18 FLOPs for arm N and about 2e18 for arm Y. That is about 5 and 8 h on an L40S, or about
    2 and 3 h on an H100 or H200.
  - Test: about 1.7e18 FLOPs per run, which is about 6 h on an L40S and 2 h on an H100.
  - Dev: about 0.5 h per run.
- **Total for three runs (N s0, Y s0, Y s1):** about 40 GPU-hours if every run lands on an L40S, and fewer on
  H-class GPUs.
- **Cap:** 60 GPU-hours, about US$70 at $1.17 per GPU-hour. If the projection passes the cap, the Y s1 replicate is
  cancelled first. Spend is counted from sacct elapsed time for every B23 GPU job, failed ones included.
- **Jobs:** PACE, account gts-nimam6-paid. One job per run does train, dev, gate and test in sequence; every step
  resumes from where it stopped.

## Files

- Code: b23.py, b23_score.py, pace/b23_*.sbatch.
- Data: b23_data/ (training and dev examples, labels.json, stats.json), b23_out/ (raw outputs).
- Results: results/b23_scores.json, results/b23_verdict.txt.
