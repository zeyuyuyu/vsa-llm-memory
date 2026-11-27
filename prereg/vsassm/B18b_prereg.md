# B18b pre-registration: the qa11-qa13 vocabulary fix (before any B18b run)

Date: 2026-10-22. B18 found (amendment 5) that the fact filter's vocabulary (words of the qa1-qa10 0k training
stories) lacks "he", "she", "they", "afterwards" and "following". As a result, the bAbI-original qa11-qa13 training
sentences were never teacher-labelled, and the student scored 0.64 / 0.59 / 0.46 on them. The fix was identified from
the training data and the error traces; no test item was used to design it.

## Change

- Re-label the bAbI-original qa11-qa13 training stories (facebook/babi_qa en-10k, the same 300 per task). The
  vocabulary is extended with the words of those training stories (`b18_distill.py --vocab_babi 1`). Same teacher,
  prompt and checks.
- Training set = B18's train_all.jsonl, minus every example whose prompt also occurs in the re-labelled set, plus the
  re-labelled set.
- Student: Qwen3-1.7B + LoRA, seed 0, the same recipe as B18 (1 epoch, holdout 1000).
- Nothing else changes: memory, answer rules and metric are as in B18.

## Test (official RMT-team/babilong, as B18)

- qa11, qa12, qa13 at 0k, n = 100 each (the only length that exists for them).
- Regression: qa1-qa10 at 0k and 4k, n = 100 each.

## Bars

- **P1:** each of qa11, qa12, qa13 >= 0.90.
- **P2:** the qa1-qa10 mean at 0k and at 4k is within 0.01 of B18 (0.997 and 0.982), or higher.
- If P1 and P2 hold, the product reader is replaced by this student, and the B18 numbers at 16k-1M (qa1-qa10) are
  re-run with it before they are quoted for the new reader. Until then they are quoted for the B18 student.

## Prediction

P1 holds for qa11 and qa13 (pronouns). qa12 ("Mary and Daniel went") is at risk, because the event schema writes one
`go` per person, and the reader must expand the conjunction. P2 holds.
