# NSR9 verdict (2026-11-15): at 2-8x the NSR8 sizes the general engine beats the same frontier model answering directly, with 0 wrong
Prereg: prereg/NSR9_prereg.md. Amendment 1 (prereg/NSR9_amendment1.md, commit f6963c7, before any Claude call on an
NSR9 item) moved both arms to `claude-opus-5-5` on the user's endpoint after the user stopped OpenAI spending. Items:
data/nsr9_items.jsonl (75; sha256 495a866d; B4 PASS). Scorer: runs/nsr8_run.py --score --tag nsr9c (the family's
check()). Post-hoc lenient scorer (disclosed below): bench/nsr9_lenient.py.

## Outcome (75 items: 5 held-out families x levels 6-8 x 5)
| arm | correct | abstain | wrong | spend |
|---|---|---|---|---|
| E_c engine (Opus 5.5 writes 2 CPMpy models, certified) | **67** | 8 | **0** | $11.3 |
| P_c Opus 5.5 + code (E_c's first model alone) | 68 | 7 | 0 | (in E_c) |
| D_c Opus 5.5 direct, registered scoring | 27 | 32 | 16 | $34.7 |
| D_c, post-hoc lenient scoring (bare names accepted) | 42 | 32 | 1 | |

| family (sizes L6 / L7 / L8) | E_c | D_c strict | D_c lenient | why D_c falls |
|---|---|---|---|---|
| meetings (24 / 36 / 50 meetings) | 15/15 | 12 | 12 | 3 cut off at 32k (all L8) |
| roster (20x21 / 28x28 / 36x28) | 13, 2 abstain | 0 | 0 | 15/15 cut off at 32k |
| project (40 / 60 / 80 tasks) | 15/15 | 1 | 1 | 14 cut off |
| islanders (30 / 45 / 60 people) | 15/15 | 2 (13 "wrong") | 14, 1 wrong | the 13 were bare-name keys |
| vans (50 / 80 / 120 parcels) | 9, 6 abstain | 12 (3 "wrong") | 15/15 | the 3 were bare-name keys |

By level, E_c: L6 25/25, L7 24, L8 18. D_c lenient: L6 14, L7 16, L8 12.

## Bars
- B1 (E_c wrong <= 1): **PASS**, 0.
- B2 (E_c correct >= 45): **PASS**, 67.
- B3 (E_c - D_c >= 11, paired): **PASS**, +40 under the registered scorer. Under the post-hoc lenient scorer the gap
  is +25.
- B4 (gold sanity): **PASS**.

## Predictions against outcomes
- D falls with size: **held**. The cause is the 32k output limit (32 of 75 cut off), not a wrong reading. When D_c
  finished it was nearly always right: 42 of 43 under the lenient scorer.
- E wrong 0-1: **held** (0). E correct 0.60-0.85: **above** (0.89).
- P wrong 0-3: **held** (0).
- B3: **held**.

## What happened
1. **The engine keeps its contract at scale.**
   - 0 wrong of 67 certified, on unseen families up to 50 meetings, 36 nurses x 28 days, 80 tasks, 60 islanders and
     120 parcels.
   - The 8 abstentions:
     - 6 vans optima not proven within 300 s (vans 80-120 parcels);
     - 2 roster models that left one sentence unaccounted for (the coverage check).
   - The earlier gpt-6.1-sol run of the same arm (partial, 49 items) also had 0 wrong (results/NSR9_partial.md).
2. **The frontier model alone runs out of room.**
   - At these sizes a direct answer has to carry the whole search and the whole answer in its output: up to 1,008
     roster slots or a full 80-task plan.
   - Opus 5.5 at effort high stopped at the 32k output limit on 32 of 75 items, among them every roster.
   - The engine writes a model of a few thousand tokens, and the solver does the search.
   - The engine was 3x cheaper ($11.3 vs $34.7) and right on 25 more items under the most generous scoring.
3. **Direct answers also drift from the requested format.**
   - On 16 items Opus 5.5 keyed its answer by bare names ("Hiro": 0), not by the requested slots ("knight:Hiro").
   - The registered scorer counts these as wrong. Read leniently, 15 of them are right; 1 islanders answer is wrong
     on the merits.
   - The engine's answers come from the solver in the slot format, so it never had this problem.

## Checks from other angles
- Every D_c "wrong" answer was inspected: all 16 had none of the requested slot keys, which is how the format problem
  was found. Under the lenient scorer, 1 remains wrong.
- Every E_c and P_c answer was scored by the family's own check(). The gold is proved by complete search with z3
  (meetings, islanders), a planted roster (roster), or a lower bound met by the planted plan (project, vans).
- Exploratory, after the fact (not a registered result): 6 cut-off items (3 roster L6, 3 project L6) were re-run with
  a 64k output limit (out/nsr9_explore64k_D.jsonl).
  - Result: **5 of 6 right**. The sixth finished without a parsable answer.
  - Output: 31k-61k tokens per item, about $0.9 per item, against about $0.15 per item for the engine.
  - So at level 6, the direct arm's losses are mostly its token budget. Given about twice the budget, it solves them,
    at about 6x the engine's cost.
  - Levels 7-8 (up to 1,008 roster slots, 80 tasks) were not tried at larger budgets.

## Literature check
- **Supports:** He & Wang (arXiv 2512.03272), where solvers help when the search space is ample. ZebraLogic (Lin et al.
  2025), where direct accuracy falls with search-space size. Our own NSR6b, where gpt-6.1-sol direct fell to 0.30 at
  130-180 hosts and frontier + code had 1 silent miss.
- **Supports the cost side:** reasoning models spend output tokens in proportion to the search they do in text. Here a
  32k budget is exhausted by plans of 400-1,000 slots, while a solver call costs nothing in tokens.
- **Conflicts / limits:** a larger output budget recovers most of D_c's level-6 losses (5 of 6 at 64k). The accuracy
  claim therefore holds "at the same 32k output limit", not "at any budget". At larger budgets the engine's edge at
  level 6 is cost (about 6x) and the certificate, not accuracy. Levels 7-8 at larger budgets are untested.

## What this means
- First evidence, on families nobody tuned on, that the general engine beats a frontier model answering directly.
  - At the same output budget, it is more accurate: 67 vs 42 correct, under the most generous scoring.
  - At every budget we tried, it is cheaper (3-6x).
  - Every answer it gives is certified; the direct arm gives 1 wrong answer and 16 off-format ones.
- The advantage comes from scale: none at NSR8's sizes, large at 2-8x.
- What remains for the product:
  - optimum proofs on large bin-packing items;
  - a local translator: local models up to 32B give 0 coverage (docs/G1_devlog.md).

## Caveats
- One frontier model per run (Opus 5.5 here; gpt-6.1-sol in NSR8).
- The relay adds an unknown system prompt of about 130 tokens to every call.
- 75 items, five generated families.
- The direct arm's output limit is fixed at 32k (the registered setting; the same as gpt in NSR8).
