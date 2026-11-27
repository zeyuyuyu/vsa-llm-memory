# NSR9 (2026-11-15): stopped part-way by the user ("不要花openai api key了"); partial result, no verdict on B3
Prereg: prereg/NSR9_prereg.md (commit before items). Items: data/nsr9_items.jsonl (75; gold sanity B4 PASS,
out/nsr9_B4_selftest.txt).
- The engine arm E ran level 8 first. It was stopped at 49 of 75 items: levels 8 and 7 are complete except one level 7
  vans item; level 6 was not run.
- Arm D (frontier direct) was never started, so B3 (engine vs direct) **cannot be evaluated**.
- No OpenAI calls are made after this point. Ledger at the stop: $105.34 of $150.

## What did run (49 items, levels 7-8)
| family | L7 E | L8 E |
|---|---|---|
| meetings (36 / 50 meetings) | 5/5 | 5/5 |
| roster (28x28 / 36x28) | 5/5 | 5/5 |
| project (60 / 80 tasks) | 5/5 | 5/5 |
| islanders (45 / 60 people) | 5/5 | 5/5 |
| vans (80 / 120 parcels) | 4/4 | 1/5, 4 abstain |

- E: 45 certified, **0 wrong**, 4 abstentions. All four are vans at 120 parcels: "primary optimum not proven" within
  300 s.
- P (first model alone) also had 0 wrong (44 correct).
- No formalisation needed a repair.
- B1 (E wrong <= 1) holds on the items run. B2 cannot be scored on 75. B3 is untested.

## Reading
At 2-8x the NSR8 sizes the general engine still certifies only right answers, and coverage stays high except where
the optimum proof is the bottleneck (bin packing at 120 parcels). Whether the frontier model alone falls at these
sizes stays open. It would take arm D, about $15-25 of OpenAI calls, which the user has stopped.
