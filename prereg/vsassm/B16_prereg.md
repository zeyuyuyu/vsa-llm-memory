# B16 pre-registration: distil the B15 reader into Qwen3-1.7B (before any B16 code, data or run)

Date: 2026-10-20. Goal: keep the product small. B15's reader (Qwen3-8B, prompt C, no name merging) passed a new
hand-written set (0.930 / 0.900); the same prompt family on Qwen3-1.7B scores 0.73 / 0.70.

## Data (teacher labels, no hand labels)

1. **Stories.** Qwen3-8B writes free stories (no event list) about objects being put, moved, swapped and nested, in
   varied styles (diary, chat messages, email, report, children's story, plain narrative), asked to use pronouns, state
   statements ("the keys were still in the car"), unrelated actions, plans that did not happen and several events per
   sentence. Two domains: locations and containers. About 3,000 stories.
2. **Labels.** The B15 reader labels every sentence of every story exactly as it runs in the product (context = up to 3
   earlier sentences, known names so far, prompt C, greedy). Each training example is (prompt, teacher output).
3. Nothing from the B13, B14 or B15 hand-written sets is used for training. Those sets are development data only.

## Student

Qwen3-1.7B + LoRA (r 16, all projection layers), supervised on the teacher's output tokens only, 2 epochs, 2 seeds
(0, 1). At inference the student gets the same prompt as the teacher and its output is parsed the same way.

## Test

A new hand-written set of 40 stories (`b16_accept.jsonl`), same specification as B13-B15 (A 8, B 8, C 7, D 7,
containers 10), written and frozen (sha256 recorded) before any student is run on it. The teacher and the references
(Qwen3-1.7B and Qwen3-8B thinking, direct) are run on it too.

## Claims and bars (each seed)

- **P1** student locations Q-acc >= 0.85; **P2** containers >= 0.80.
- **P3** student within 0.05 of the teacher on the same set (locations and containers).
- Reported: development scores on the B13 / B14 / B15 sets; generated tokens per sentence; teacher-student agreement on
  held-out teacher-labelled sentences.

## Prediction (stated so it can be wrong)

P1 and P2 hold for both seeds; P3 holds for locations but not for containers (fewer container training stories).

## Frozen test set

`b16_accept.jsonl`: 40 stories (A 8, B 8, C 7, D 7, containers 10), 90 questions, self-check 0 mismatches (two items
reworded before freezing: a hat "worn" by a person was out of scope; a possessive in a question), sha256 `12c95564d5ed87221ed79791ba3522f1a52b41e63b9d98363297d86fd062de57`.
Written and frozen while the B16 data were being generated, before any student existed.
