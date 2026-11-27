> **VOID, 2026-09-17.** This design cannot be run on phi-2 and was never validly executed. A 300-fact pool
> is ~3,352 tokens against phi-2's 2048-token window, and the runner truncated from the right, which removes
> the question at the end of the prompt; the N=300 run therefore printed EM 0.0000 with training loss stuck
> at 4.3 for a harness reason. Even N=150 (~1,702 tokens) would have been truncated under the old 1,400-token
> training cap. Nothing from this prereg is banked. `v140.py` now aborts rather than truncating, and the
> question it meant to ask is re-registered as **V-143** on a 32k-window reader (`v143_prereg.md`).

## V-141 prereg — push the distraction axis past the manuscript's largest pool

Registered before the run. V-140 sweeps 0/50/150 distractor facts, which is where the manuscript already has
prompting (0.950 -> 0.353) and the store (0.974 -> 0.953). The manuscript's own capacity work goes to 1,024
facts for the store, so the fine-tune can be pushed there too. Same runner, same 300 compositional evaluation
rows, both arms handed the SAME evidence, three seeds, rank 16 / 6 epochs / lr 1e-4.

New levels: N = 300 and N = 500 distractor triples.

Reference points at the store side, from the manuscript's sharding work at d=2048: a single store holds its
readout to about 128 facts and degrades past a few hundred, which is why the store is sharded at scale. The
honest comparison at N=300 and N=500 is therefore fine-tune vs SHARDED store, not vs a single store, and the
sharded reference (0.908--0.931 between 128 and 1,024 facts) is what the fine-tune is measured against.

BARS (fixed now):
  P1 fine-tune at N=500 vs its own N=0:
      drop >= 50 points -> the distraction axis defeats fine-tuning on real text at scale.
      drop <= 15 points -> it does not; fine-tuning tolerates pooled evidence and the axis is not the answer.
      otherwise -> partial.
  P2 monotonicity: the fine-tune's accuracy must not increase with N; a rise means the distractor construction
      is leaking the answer and the run is void.
  P3 the N=0 cell must reproduce V-140's N=0 within 5 points (same configuration, different process).

This is a scale extension, not a new claim. Whatever it shows goes in the ledger; nothing goes in the
manuscript, per the standing instruction.
