# V-144 prereg — where this reader ALREADY collapses, does a fine-tune rescue it?

Registered 2026-09-17, before any V-144 number exists. Runner `v144.py` committed in the same commit.

## Why this run exists

The submitting author's direction is to keep raising the evidence count until the model cannot hold it
("N 需要逐渐调大，总有模型撑不住的时候"). On the V-143 task that point is not being reached: with real
2WikiMultihopQA entities on Qwen2.5-32B-Instruct, prompting is 0.9500 / 0.9550 / 0.9400 at pools of
0 / 64 / 256 and the LoRA arm is 1.0000 / 0.9850 / 0.9900. A pretrained reader has priors for real entity
names, so that task is the easier of the two we have.

We already know where this same reader does collapse. V-135 swept the pseudo-name chain task (nonce names
with no semantic handle, three decoy relations per chain node) at fixed depth 2 on the same 32B:

| K facts | prompt tokens | sharded store | plain prompting | truncation |
|---:|---:|---:|---:|---:|
| 120 | 1,404 | 1.0000 | 0.9250 | 0.000 |
| 250 | 2,775 | 0.9972 | 0.9417 | 0.000 |
| 500 | 5,415 | 1.0000 | 0.9000 | 0.000 |
| 1000 | 10,692 | 1.0000 | 0.7500 | 0.000 |
| 2000 | 21,246 | 0.9972 | **0.2500** | 0.000 |
| 4000 | 42,354 | 1.0000 | 0.0000 | **1.000 — excluded by the registered rule** |

The collapse sits between K=1000 and K=2000 with **zero truncation**, so it is a reading failure and not a
window failure. V-135 measured prompting only. The arm it did not have is the one that decides the question:
a fine-tune, given the same tables, trained in the same volume regime it is evaluated in.

## Design

- Reader, task and instances: identical to V-135. `v144.py` imports `make_instance`, `question` and
  `facts_block` from `v135.py` and records the sha256 of `v135.py`, `v133_task.py` and `v133a.py` in its
  output, so the instances are the same object V-135 scored. Same system prompt, same chat template.
- K in {120, 1000, 2000}: one cell where prompting is healthy, one where it has begun to fall, one where it
  has collapsed. K=4000 is NOT run — it does not fit the window, and a truncated cell is not scorable. That
  rule is V-135's and it is inherited unchanged.
- Eval instances are V-135's scored block (seeds 0..119); training instances use seeds 100000+, asserted
  disjoint. n_train 100, n_eval 120, 3 epochs, rank 16, lr 1e-4, seed 42, held constant across K so a
  decay cannot be read as "fewer gradient steps at large K".
- The fine-tune trains **in the volume regime it is evaluated in** — it sees K-fact tables during training.
  This is in the opponent's favour and is deliberate.
- A `plain` arm is re-measured inside this harness so the comparison is internally consistent; V-135's own
  plain and store values are quoted as banked context, not used as the comparator for the LoRA arm.
- `--max_ctx 30000`; the runner aborts rather than truncating.

## Bars, fixed now

Scored at K=2000, against the store's banked 0.9972 at the same K.

- **EVIDENCE that a fine-tune cannot do what the store does here:** LoRA <= store − 0.30 (i.e. <= 0.697).
- **REFUTED:** LoRA >= store − 0.10 (i.e. >= 0.897) — the fine-tune rescues the collapse.
- **PARTIAL** in between.

Per the V-142 amendment-2 rule, every bar is scored twice — over all seeds and over convergent seeds only
(a seed is non-convergent if its final-epoch loss exceeds half its first-epoch loss) — and a verdict stands
only if both readings agree. Non-convergence is reported either way.

## The negative reading, written before the numbers

If the fine-tune reaches the store at K=2000, then the store's advantage on the volume axis is not a
capability the weights lack, and the honest summary becomes: on this axis the algebra buys cost, not
capability, at every reader we have measured. The ledger will say that, and the manuscript gains nothing.

If instead the fine-tune stays at the prompt arm's floor while the store holds, that is the first evidence
in this line of work that the separation survives against a *tuned* opponent at volume — and it is still
bounded to readers up to 32B, because V-137 measured frontier readers holding 0.975 at 8000 facts and 84k
untruncated tokens. That bound is not optional and will be stated in the same sentence as the result.

## What this run cannot show

It cannot speak to frontier readers (V-137 already did, negatively), nor to real-entity tasks (V-143), nor
to depth (V-133). It answers one question: at the volume where this reader's prompting has already failed
with the evidence fully in its window, does training the weights on that regime recover it?
