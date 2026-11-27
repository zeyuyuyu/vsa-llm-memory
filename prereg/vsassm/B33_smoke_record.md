# B33 smoke record (no amendment)

Date: 2026-11-18, about 07:35 UTC. Nothing in the code, bars, predictions or readings changes; this file discloses
what the smoke runs showed and one procedural point.

## What ran (box /path/to/work/b33, logs/smoke.*; GPU 5, 07:27-07:29 UTC)

- b33_babi.py --smoke (corrected parse, seed 42, comm and perm; first 64 TRAINING rows; naming head on the full
  training split, 900 expressible stories): 64/64 parsed and expressible, exact rows and naming head 1.000 under both
  encodings.
- b33_prompt.py --smoke (64 training rows): 0.406; the verbatim v49.score and the logit-keeping pass agree on every row.
- b33_lora.py --smoke (rank 8, 64 training rows, one epoch, scored on 32 training rows): 0.3125, the same value as
  B28's LoRA smoke (b28_lora2 --smoke), i.e. the copied training loop behaves as before.

## Procedural point (disclosed)

To keep the claimed GPU (all seven usable A800 GPUs were held by other B31/B32 workers from 05:40 UTC; the first one
freed at 07:27), the main commands were appended to the smoke worker's queue right after the smoke commands, so the
evaluation runs started as soon as the smoke runs had exited with rc=0 (07:28-07:29), without a pause in between. The
run logs print each set's accuracy as it is computed, so the gate numbers (orig500) and the evaluation numbers
(fresh, hfrest) became visible together in the same log. This does not change anything that can still change: the
code, the fresh set (committed in a6e451f with its sha256) and every bar were fixed before the runs. The gates are
still applied as registered by b33_verdict.py, and if a blocking gate fails the evaluation numbers are not used.
