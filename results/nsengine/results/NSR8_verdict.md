# NSR8 verdict (2026-11-15): the general engine keeps its contract on five unseen families, but at these sizes the frontier model alone is just as right
Prereg: prereg/NSR8_prereg.md (commit b8efe65). Items: data/nsr8_items.jsonl (150), sha256 cb6326fb..., generated
once after the prereg. Questions given to the systems: data/nsr8_q.jsonl (sha256 77b09cdb...). Scorer:
runs/nsr8_run.py --score -> out/nsr8_score.json.

## Outcome (150 items: 5 families x 5 levels x 6)
| arm | correct | abstain | wrong |
|---|---|---|---|
| E engine (gpt-6.1-sol writes 2 CPMpy models, certified) | 142 | 8 | **0** |
| P frontier + code (E's first model alone, no checks) | 143 | 7 | 0 |
| D frontier direct (gpt-6.1-sol, effort high, 32k) | **150** | 0 | 0 |

| family | E | P | D |
|---|---|---|---|
| meetings (check) | 30/30 | 30/30 | 30/30 |
| roster (find) | 30/30 | 30/30 | 30/30 |
| project (optimize) | 30/30 | 30/30 | 30/30 |
| islanders (unique) | 30/30 | 30/30 | 30/30 |
| vans (optimize) | 22, 8 abstain | 23, 7 abstain | 30/30 |

By level, E: L1 30, L2 29, L3 30, L4 28, L5 25 (all abstentions are vans).

## Bars
- B1 (E wrong <= 1): **PASS**, 0.
- B2 (E correct >= 105): **PASS**, 142.
- B3 (levels 4-5, E - D >= 9): **FAIL**, 53 - 60 = -7.
- B4 (gold sanity): **PASS**. All 150 golds pass check(); z3 agrees on all 150 (out/nsr8_B4_selftest.txt).

## Predictions against outcomes
- E wrong 0-1: **held** (0). E correct 0.75-0.92: **above**, 0.95.
- P wrong 3-12: **missed**, 0. The single frontier model made no silent error on these families.
- D falling to 0.3-0.6 at level 5: **missed**. D is 30/30 at level 5, with about 2,100 completion tokens per item on
  average and $3.34 for all 150.
- The prereg's own fallback applies: "if D stays above 0.8 at level 5, the families are not hard enough for this
  model, and the verdict will say so rather than claim an edge."

## What happened
1. **The contract held on families nobody tuned on.**
   - The families were written by a separate agent that never saw the engine.
   - The prompt and checks were frozen before any family was read.
   - Result: 142 certified answers, 0 wrong.
   - Every abstention is in vans:
     - 7 are "optimum not proven": the solver hit its 60 s limit, because the models had no symmetry breaking;
     - 1 is a false disagreement: one model numbered vans in order, which rejects the other model's equally good answer
       (docs/G1_devlog.md has the fix to test next).
2. **The engine adds nothing over the frontier model at this scale.**
   - The largest instances are 18 islanders, 20 tasks, 24 parcels, a 13-nurse x 14-day roster, and 16 meetings over 5
     days.
   - gpt-6.1-sol solves all of them directly, cheaply, and never abstains.
   - Frontier + code (P) also made no error, so the certification layer had no silent error to catch here. Its safety
     value is shown on our own data only: NSR6b, large SRE, one silent miss by frontier + code in 40; and NSR7b's
     shared misreads.

## Checks from other angles
- D's 60 level 4-5 answers were fed into the engine's own CPMpy formalisations (both models per item). All 60 were
  accepted, and every optimisation answer reached that model's optimum. This check is independent of the family's
  check() (bench/nsr8_verify_D.py).
- Golds: the family's own exact solver plus z3, on every item (B4).
- Hand checks of one level-2 item each:
  - vans (bounds and every constraint), project (crew use day by day) and islanders (all seven statements): correct by
    hand;
  - meetings and roster: read, not recomputed.

## Literature check
- **Supports:** He & Wang (arXiv 2512.03272). Solvers help only when the search space is ample. At these sizes it is not
  ample for a 2026 frontier reasoning model.
- **Supports:** ZebraLogic (Lin et al. 2025). Accuracy falls with search-space size, not with the number of rules.
- **Consistent with our own NSR6b.** gpt-6.1-sol direct fell to 0.30 only at 130-180 hosts with a 3-wave plan, far
  larger than anything here.
- **No conflict found.** The finding narrows our claim; it does not contradict a published result.

## What this means
- **For the general engine:** one prompt and one certification layer carried five new families with 0 wrong. The
  contract generalises; this is the first evidence for generality.
- **For the value claim:** it holds only where problems are big. The next test has to scale the same held-out
  families until the frontier model's direct accuracy falls, and keep the gold exact. That means a proven optimum, or
  validity checks for find/check families. Next study: NSR9.
- **For the product:** at small and medium sizes the engine's offer is the certificate (the answer checked by two
  models, the reasons cited by sentence) and the cost. Accuracy is not part of the offer there.

## Cost
- E: $3.1 (300 formalisations, 4 repairs).
- D: $3.34.
- OpenAI ledger after NSR8: $100.39 of the $150 cap.
