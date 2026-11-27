# R2 — a trained reader for the CLUTRR rule memory, against a direct fine-tune (written 2026-10-24, before training)

## Question
R1 showed that the VSA rule memory composes exactly (gold relations: 0.975; 117/117 rules read back) and that the
end-to-end gap (0.717 with a prompted Qwen3-8B reader) is the reader's. R2 asks two things:
1. Does a small reader trained only on the training split (k = 2, 3) close that gap on the test split (k = 2-10)?
2. Is the length generalisation the memory's? A direct fine-tune of the same model, on the same stories with the
   same budget, answering without the memory, is the control.

## Data (no test data used for training or development)
- Split: CLUTRR gen_train23_test2to10 (r1_data). Train 9,074 stories (k = 2: 4,076; k = 3: 4,998); validation 2,020
  (k = 2, 3); test 1,146 (k = 2-10).
- Reader targets (r2_data.py): CLUTRR's own annotations. An edge (a, b) of type r means "node b is the r of node a";
  genders lists the nodes in order (checked on all 9,074 train rows: query endpoints match nodes 0 and last, 0 bad).
  Each edge is written in both directions in r1.PROMPT's JSON format, with the inverse chosen by the gender of the
  person named. Check before training: gold facts through the unchanged r1 pipeline (answer_vote + RuleMemory) give
  0.973 on test (2: 1.000, 3: 0.857, 4: 0.963, 5: 0.983, 6: 1.000, 7: 0.979, 8: 0.993, 9: 1.000, 10: 0.983). This is
  the ceiling for arm A.
- Direct-answer targets: the same stories, the question "How is B related to A?", the answer word (target_text).

## Arms (both Qwen3-1.7B + LoRA r16, b16_train.py unchanged: 2 epochs, bs 16, lr 2e-4, 300 held out, seed 0)
- **A, reader + memory:** the adapter writes the relations; r1.py --vote (rules learned from train only, FHRR memory
  at D = 2,048) answers. No thinking.
- **B, direct:** the adapter answers the question directly. Answer = the first relation word in the output (longest
  match first, so "father-in-law" before "father").

## Protocol
1. Train both; evaluate both on the full validation split (k = 2, 3) as a sanity check. Nothing is tuned on it; if
   it reveals a bug (format, parsing), the fix is disclosed here before the test run.
2. Test split, once, both arms.

## Bars (test split)
- **P1:** arm A mean accuracy >= 0.85.
- **P2:** arm A accuracy on k >= 7 >= 0.80.
- **P3:** arm A minus arm B on k >= 7 >= 0.10.
- Also reported: per-k accuracy; reader fact precision/recall against the gold edges (test); R1's prompted readers
  (Qwen3-8B thinking 0.717; Qwen3-1.7B thinking 0.344); DSR-LM 61.0 (best published from text we found).

## Predictions (fixed now)
- P1 holds (the reader's task is local: one sentence states one relation). Expected about 0.90.
- P2 holds, with some loss on long stories if the reader stops early on 20-fact outputs it never saw in training.
- P3 holds by a wide margin: the direct fine-tune falls with k (expected below 0.5 at k >= 7).

## Cost
Two LoRA trainings of about 1,100 steps each and three inference passes: about 1 H200-hour on PACE (paid account).
