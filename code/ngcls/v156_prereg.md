# V-156 pre-registration — B2: extraction-easy, composition-hard. Does an external exact composer beat the frozen LM's own reasoning when bindings and hops pile up, with the LM doing the extraction?

Written and committed BEFORE the run. Companion to V-155 (B1, natural text, where the pilot showed the external
route losing badly). B2 tests the regime where an external composer should win: every fact trivially easy to
read, but many of them and many hops. Smoke test (3 instances/cell, small grid) passed end to end with the LM
copying bindings at recall 1.00; it is not a result.

## Task and design
`b2.py` (committed with this file). Each instance: H tables of N random bindings `KEY -> VALUE`, drawn fresh per
instance, per cell and per seed (no fixed data seeds); a chain from a key of table 1 through tables 1..H; the
question names the start and asks for the end. 30% of distractor values reuse chain symbols, the case
role-marked binding exists for. Grid N in {4, 16, 64, 128} x H in {1, 2, 4}; 50 instances per cell; seeds 0, 1,
2. Qwen2.5-7B-Instruct, greedy, one few-shot example for every arm.

Arms, same text, same instances, no filtering: **direct**; **cot**; **extract+dict** (the LM copies bindings out
by prompting, a table composes); **extract+vsa1** (same extraction, ALL bindings in ONE role-marked HRR vector,
d = 4096); **extract+vsa_shard** (one vector per table, the manuscript's sharding remedy); **regex_ceiling** (a
regex parse + table -- NOT a method, reported only to show the task is fully parseable); and extraction recall.

## Bars (instances pooled over the three seeds: 150 per cell; paired bootstrap, 20,000 resamples, seed 20261016)
- **P1 (B2's version of the title):** in the hardest cell, N=128 H=4, `EM(extract+vsa_shard) - EM(cot) >= 0.20`
  with the 95% CI excluding zero.
- **P2 (the LM degrades with load):** `EM(cot | N=4,H=1) - EM(cot | N=128,H=4) >= 0.30`.
- **P3 (sharded VSA is lossless against the table):** in every cell, `|EM(vsa_shard) - EM(dict)| <= 0.03`.
- **P4 (the capacity law predicts the single vector's failure):** in every cell with N*H <= 64,
  `|EM(vsa1) - EM(dict)| <= 0.03`, AND at N*H = 512, `EM(dict) - EM(vsa1) >= 0.30`.

## What I expect (separately)
P2 holds. P3 and P4 hold -- they are properties of the algebra, already measured in isolation (V-151 and the
manuscript's sharding result), here checked inside the pipeline. **P1 is the real question and genuinely
uncertain**: at N=128 H=4 the LM must copy 512 lines without dropping one; if its copying fails as its reasoning
does, the external arms fall with it and P1 fails. That would mean the frozen LM's extraction, not its
composition, is the bottleneck even on a task designed to make extraction easy -- itself a clear answer.

## Scope
Synthetic, single model, 7B. A held P1 says an external VSA composer helps a frozen 7B LM where composition,
not reading, is the bottleneck; it does not say VSA beats a table (P3 tests exactly that and is expected to
show they tie), and "external exact memory beats in-context reasoning on long chains" is broadly known from
tool-use work -- the VSA-specific content is the fixed-size vector, the sharding, and the capacity prediction.

Scorer `v156_score.py` frozen with this file, run verbatim.
