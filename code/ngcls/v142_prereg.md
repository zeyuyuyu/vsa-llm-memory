## V-142 prereg — natural-language prose whose pairings are arbitrary and instance-specific

Registered before the run. V-139 killed the easy version of this question: on bAbI task 2 a held-out
(carrier, location) combination costs the fine-tune nothing (1.000 vs 1.000), because the task is solved by a
combination-INDEPENDENT rule. Arena C, where the fine-tune sits at the floor, differs in a property that is
not difficulty: its pairings are arbitrary and hold only within the instance, so no rule shortcuts them.

This experiment puts that property into ENGLISH PROSE, which is the step arena C cannot make on its own, and
sweeps the one axis that should matter if the mechanism is right: how many arbitrary pairings the instance
carries.

Construction. Each instance is a short passage of ordinary sentences over nonce names drawn from a large
pool -- "The Zarnak protocol is handled by the Velm module. The Velm module reports to the Tessin office." --
followed by a two-hop question ("Which office handles the Zarnak protocol?"). Pairings are re-drawn at random
per instance, so no mapping recurs across training and evaluation; the surface grammar is fixed English.
Inventory size K in {8, 32, 128} pairs per instance; answer restricted to the K office names present.

Arms, both given the same passages, three seeds:
  lora        rank 16, 6 epochs, lr 1e-4, restricted classification over the present answers (the V-131 setup)
  exact_code  the algebra on exact codes read by a hand-written scan of the same passage (the parser arm,
              two trainable parameters) -- the algebra's ceiling, not a deployable method

BARS (fixed now):
  P1 (primary) at K=128: lora <= exact_code - 30 points  -> the property transfers to prose: arbitrary
        instance-specific pairing, not task difficulty, is what a low-rank update cannot absorb.
      lora within 10 points of exact_code               -> it does not transfer; arena C's result is an
        artifact of the synthetic format and should be described as such in the ledger.
      otherwise -> partial.
  P2 (trend) lora must degrade monotonically in K within sampling error; a flat curve means the task is
      rule-solvable after all and P1 is uninterpretable.
  P3 (gate) no pairing may recur between training and evaluation; checked, not assumed. At K=8 the fine-tune
      must reach >= 0.90, otherwise the format itself is the obstacle rather than the binding load.

Reported to the ledger whichever way it goes, like V-139. Nothing goes into the manuscript.

## AMENDMENT (before the run) — inventory ceiling set by the context window, not by choice
K=128 puts 256 sentences (about 2,500 tokens) in front of a model whose window is 2,048, so the top level
would measure truncation rather than binding load. The sweep is therefore K in {8, 32, 64}; K=64 is 128
sentences, roughly 1,300 tokens, comfortably inside the window. P1's threshold moves to K=64 unchanged in
size (lora <= exact_code - 30 points). Recorded here rather than silently re-scoped after seeing results.

## AMENDMENT 2 (2026-09-17), before any K=64 verdict exists — two corrections

### (a) The K=64 arm on phi-2 is VOID: the first amendment's token estimate was wrong

Amendment 1 moved the primary bar to K=64 on the ground that K=64 is "128 sentences, roughly 1,300 tokens,
comfortably inside the window". The runner's own instrumentation says otherwise:

| K | tokens/instance (measured by the runner) | inside phi-2's 2048 window | inside the 2000-token cap |
|---|---|---|---|
| 8 | 340 | yes | yes |
| 32 | 1,278 | yes | yes |
| 64 | **2,714** | **no** | **no** |

So the arm carrying the PRIMARY bar would have been scored on truncated passages, with the question at the
end of the prompt among what is cut. A low LoRA score there would have read as "the binding property
transfers to prose" when it would have measured a context limit — an error in our own favour, which is the
direction that most needs guarding. The K=64 run was killed 2.5 minutes in, before any evaluation; nothing
from it is banked. This is the second instance today of the same class of artifact (see V-141).

**Correction:** the reader becomes `Qwen/Qwen3-8B` (32k window) for the whole sweep, so all three K values
are comparable and none is truncated; `v142.py` now aborts rather than truncating, and records the reader in
its output. K=8 and K=32 on phi-2 were NOT truncated (340 and 1,278 tokens) and remain valid, but they were
measured on a different reader, so the re-run repeats all three K rather than splicing readers within one
curve. The phi-2 values are kept in the ledger as a superseded reading, not deleted.

### (b) How a seed that never trained is scored — fixed before the deciding data arrives

At K=32 on phi-2, seeds 42 and 123 reached 1.0000 while seed 456 scored 0.1000 against a chance level of
0.125, with training loss flat across all six epochs (2.54, 2.67, 2.18, 2.18, 2.15, 2.12) where seed 42 went
2.47 -> 0.04. That seed did not learn; the three-seed mean of 0.7000 is not a capability number, and quoting
"exact_code 0.855 beats lora 0.700" off it would be true and misleading.

The prereg did not say how to treat this. The rule, fixed now, before K=64 is re-run:

- A seed is **non-convergent** if its final-epoch training loss exceeds half its first-epoch loss.
- Every bar is scored **twice**: over all seeds, and over convergent seeds only.
- The verdict stands only if both readings agree. If they disagree, the result is **partial**, and the
  finding is reported as what it is — that a low-rank update on this task fails to optimise on some seeds
  while the algebra has no training and no seed variance. That is a real property and is reported as
  instability, never as a binding-capacity number.
- Non-convergence rates are reported per K either way.
