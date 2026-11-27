# R2b — the R2 reader trained on longer stories built from the train split (written 2026-10-24, before training)

## Why
R2 (results/r2_verdict.txt) failed every bar: the 1.7B reader, fine-tuned on 2-3-sentence training stories, wrote
about as many facts as a training story has (12.3 of 19 at k = 10) and confused people in long stories; recall fell
from .96 (k = 2) to .38 (k = 10). The memory is not the limit (gold facts 0.973). The direct fine-tune failed in the
same way (0.358). R2b changes only the training distribution.

## Change (only this)
r2b_data.py: R2's 9,074 training examples plus 9,074 long stories, each 2-4 TRAIN stories with disjoint name sets
concatenated in random order (seed 0). Reader target = union of the component stories' gold facts (both directions,
as R2); 10.0 facts per example on average, at most 24 (1,362 characters, under the 600-token cap). Direct target =
one component story's question and answer. No validation or test story is used.
Model, LoRA, b16_train.py settings, r1.py --vote, the memory and r2_direct.py are R2's.

## Protocol
Train both arms, then evaluate on validation and test in the same job. Validation (k = 2, 3) is reported only; it
cannot show the length effect, so no decision is taken on it.

## Bars (test), unchanged from R2
- P1: arm A (reader + memory) mean >= 0.85.
- P2: arm A on k >= 7 >= 0.80.
- P3: arm A minus arm B (direct) on k >= 7 >= 0.10.
Also reported: reader fact recall/precision by k against the gold edges.

## Predictions
- Reader recall at k = 10 rises from .38 to at least .80.
- P1 at risk (expected 0.75-0.90); P2 at risk; P3 holds (direct answers stay below 0.5 at k >= 7, because composing
  ten relations in one pass is not what the long concatenated stories teach).

## Cost
About 1 H200-hour (paid account).
