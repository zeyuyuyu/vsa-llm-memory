# B18c — qa11-13 with corrected training labels (written 2026-10-24, before retraining)

## Why (diagnosis on data the student never saw; no test data read)
B18b's student on bAbI en-10k TRAINING stories 1000-1199 (distillation used the first 300): qa11 170/200, qa12
200/200, qa13 159/200 (b18c_diag.py, PACE 14351554). Every error read so far is a pronoun resolved to the wrong
person ("Daniel journeyed to the kitchen. Following that he moved to the bedroom." -> go(john, bedroom)). The Qwen3-8B
teacher's labels for these sentences are wrong at the same places: of the 600 unique pronoun sentences in
qa111213_v2.jsonl, 72/300 he/she and 194/300 they are labelled with the wrong person(s), judged by bAbI's rule
(a pronoun refers to the subject(s) of the previous sentence; b18c_coref.py, sequential, no unresolvable cases).
The rule itself is checked: events built from it give 200/200 on each of qa11, qa12, qa13 on the diagnosis stories.

## Change (only this)
b18c_data.py: the 5,400 qa11-13 training examples of B18b keep their prompts; their outputs are rewritten from the
rule (825 change). All other training data, the model (Qwen3-1.7B + LoRA, seed 0, b16_train.py, 1 epoch, 1,000
held out), the memory and answer rules are as B18b. The reader still resolves pronouns itself from its three
sentences of context at test time; the rule is used only to label training sentences.

## Test (the official BABILong test set, as B18 and B18b; this is the third use of the qa11-13 0k test samples,
disclosed)
- qa11-13 at 0k (n = 100 each), qa1-qa10 at 0k and 4k (n = 100 each); also the diagnosis stories.
- **P1:** each of qa11, qa12, qa13 >= 0.90.
- **P2:** qa1-qa10 within 0.01 of B18 at 0k and 4k (99.7 and 98.2).
- If both hold, the B18c student replaces the B18 student in the reported numbers only for qa11-13 and only with this
  disclosure; the 4k-10M numbers stay the B18 student's.

## Prediction
P1 holds for all three (expected >= 0.97), P2 holds.

## Cost
About 1 H200-hour on the paid account.
