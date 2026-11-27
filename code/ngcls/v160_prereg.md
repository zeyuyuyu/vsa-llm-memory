# V-160 PREREG — B1 v2 replication on a second model (Qwen2.5-14B-Instruct), same 300 rows

Registered 2026-10-15, before any V-160 generation. Runner `b1v2.py` **unchanged** (amended version b458ef0, md5
4bba9242, reviewed by review C). Frozen scorer `v158_score.py` and second scorer `v158_indep.py` **are used verbatim**.
There is no determinism argument here, because the model differs from V-155. Launcher `run_v160.sh`.

## Why
The multi-angle rule. V-158 (Qwen2.5-7B) closed P1 FAILS | P2 FAILS | P3 HOLDS | P4 FAILS. Its failure analysis
(60% relation-name mismatch, 26% entity aliases) may be a 7B extraction or decomposition weakness. A 2x larger model
from the same family tests whether the verdict is a property of the route or of the model's size.

## Design
Qwen/Qwen2.5-14B-Instruct, bf16, greedy with its own generation_config; rows 50–349; tau .70; GPU 0 of the A800.

The smoke runs first, on PILOT rows 0–5 only, followed by a format-only gate (never accuracy):
- decomposition gives a start and ≥1 relation in ≥ 4/6 rows;
- mean triples per row ≥ 10;
- the CoT answer is parsed in ≥ 4/6 rows.

If the gate fails, the registered rows are not run.

## Bars (verbatim from V-158; counts of 300; EM registered, containment and F1 printed)
- P1: vsa_ident − direct ≥ +15, CI > 0.
- P2: vsa_ident − cot ≥ 0, CI > 0.
- P3: |vsa_ident − dict_ident| ≤ 9 and ≤ 9 disagreeing rows.
- P4: dict_ident − dict_exact ≥ +6, CI > 0.

## Predictions
- P1 FAILS and P2 FAILS: the grounding mismatch is a property of the open-extraction route, not of model size.
- P3 HOLDS.
- P4 FAILS.

I also expect the share of walks that complete to rise above 7B's 36/300, but it stays far below the LM arms.

## Reading
- Same verdict: B1's negative replicates across scale in the family.
- P1 or P2 HOLDS at 14B: the conclusion becomes size-dependent and is reported that way.
