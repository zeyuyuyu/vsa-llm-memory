# NSR9 preregistration (2026-11-15): the general engine against a frontier model where the problems are large
Registered before any NSR9 item exists. Engine, prompt, certification rules and runner are frozen at the commit of this
file.

## Why this study
NSR8 (results/NSR8_verdict.md) showed two things on five held-out families, levels 1-5:
- the engine kept its contract: 0 wrong of 142 certified;
- gpt-6.1-sol answering directly was 150/150, so there was no accuracy edge at those sizes.

The engine can only add accuracy where the search space outgrows direct reasoning, as in NSR6b (130-180 hosts:
direct 0.30). NSR9 asks whether that happens on the same unseen families when they are scaled up.

## Items (generated once, after this commit)
The same five families (bench/heldout/, written by the blind agent), now with levels 6-8. The same agent added these
levels on request without seeing any result, and its self-test passed (650 items, 0 failures).
`python3 bench/heldout/make.py --seeds 8000:8005 --levels 6,7,8 --out data/nsr9_items.jsonl --questions data/nsr9_q.jsonl`
= 5 families x 3 levels x 5 = 75 items.

Sizes at levels 6 / 7 / 8:

| family | sizes |
|---|---|
| meetings | 24 / 36 / 50 meetings, 14 / 20 / 26 people, over 6 / 8 / 10 days |
| roster | 20x21 / 28x28 / 36x28 nurses x days |
| project | 40 / 60 / 80 tasks |
| islanders | 30 / 45 / 60 people |
| vans | 50 / 80 / 120 parcels |

How the gold is proved:
- meetings: complete search, with z3 agreement;
- roster: a planted roster, and check() validates every rule;
- islanders: uniqueness by complete search and z3;
- project and vans: a planted solution that meets a provable lower bound. The agent notes the target value equals
  that bound, so finding a plan that reaches it is the hard part.

Gold sanity before any model call: `selftest.py` on these seeds. Any failure drops that family as a whole.

## Arms
- **E (engine v4):** commit bbe1137, design v1 with SYM-tagged symmetry breaking.
  - gpt-6.1-sol, effort medium, K = 2 formalisations, one repair round.
  - gcert certification with coverage "missing only".
  - The one setting fixed now for large models: the solver limit is 300 s per primitive (NSR8 used 60 s), and the
    process limit is 720 s.
- **P:** E's first formalisation alone, no checks. An optimisation answer counts only if proven optimal within the
  limit.
- **D:** gpt-6.1-sol direct, effort high, at most 32,000 completion tokens. Same prompt as NSR8.
- **Order and budget:** E first, then D, both with level 8 first. The shared ledger cap is $150, and the runner stops
  at $148 (ledger at $101.83 when this file was written). Any arm cut short is reported, and B3 uses the paired items.

## Bars
- B1 (safety): E gives at most 1 wrong answer in 75.
- B2 (coverage): E's correct answers >= 45 (0.60).
- B3 (value against frontier direct): E correct - D correct >= 11 (0.15 x 75), over paired items.
- B4 (gold sanity): the self-test passes on every test item.

Reported without a bar:
- P's wrong answers;
- per-family and per-level tables;
- D's finish reasons and token use;
- E's abstention reasons;
- spend.

## Predictions (written before any item exists)
- D falls with size: about 0.7 at level 6 and at most 0.4 at level 8.
  - Roster (1,000 shift slots at level 8) and vans (120 parcels to place exactly) fall most.
  - Answers cut off at 32k tokens are part of the fall.
- E: wrong 0-1, correct 0.60-0.85. Abstentions come from crashes and from optimum proofs that run out of time on
  project and vans.
- P wrong 0-3.
- B3 holds. If it does not, the families at level 8 are still within the frontier model's direct reach, and the verdict
  will say so.
