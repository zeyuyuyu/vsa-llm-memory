# NSR8 preregistration (2026-11-15): the general engine on problem families it has never seen
Registered before any NSR8 item exists and before I have read any held-out family's code or text. The engine,
translator prompt, certification rules and runner are frozen at the commit of this file. Design:
docs/DESIGN_v1_general.md. Development log: docs/G1_devlog.md.

## Why this study
The user's goal (11-15) is a general reasoning engine. Design v1 has one modelling language (CPMpy), one set of solver
primitives (engine/gm.py), one certification layer that does not know the problem type (engine/gcert.py) and one
translator prompt (runs/gtranslate.py). Development used three burned families only (SRE, large SRE, ZebraLogic):
51/51 certified correct, 0 wrong. Those families shaped the prompt and the checks. The question here is whether the
contract (certified answers are right; the rest are abstentions) holds on families nobody tuned on, and whether the
engine beats a frontier model answering directly where problems are large.

## Held-out families (bench/heldout/, committed here as written)
Five families were written by a separate agent that saw no part of the engine, its prompts or any result. Its brief:
search-heavy, read-light English problems, 5 levels, gold from its own exact solver (no CPMpy/OR-Tools), checker that
accepts every valid answer.
- meetings (check: plan or "impossible", balanced);
- roster (find);
- project (optimize: earliest finish);
- islanders (unique: knights and knaves);
- vans (optimize: fewest vans).

Its self-test, run on seeds 0-19 for every family and level (500 items), passed with 0 failures. In it, z3 (an
independent solver) agreed with every verdict, optimum and uniqueness claim, and every z3 answer passed check(). I
read the family code and some items only after this commit, to check the gold. Any defect found then is reported, and
an affected family is dropped as a whole, never repaired.

## Items (generated once, after this commit)
`python3 bench/heldout/make.py --seeds 7000:7006 --out data/nsr8_items.jsonl --questions data/nsr8_q.jsonl`
= 5 families x 5 levels x 6 = 150 items.
- The systems see only data/nsr8_q.jsonl (id, family, level, text, answer_keys, kind).
- Seeds 7000-7005 are outside the agent's self-test (0-19) and timing (20-219) ranges.
- Gold sanity before any model call: bench/heldout/selftest.py logic (gold passes check(); z3 agrees) on these seeds.

## Arms
- **E (engine):** gpt-6.1-sol, reasoning effort medium.
  - Two formalisations (K = 2) with the frozen prompt; one repair round for a program that crashes.
  - Certified by gcert: kinds check/find -> find, unique, optimize; coverage "missing only".
  - Anything not certified is an abstention.
- **P (frontier + code):** E's first formalisation alone, solved once, no checks. An optimisation answer counts only
  if the solver proves it optimal.
- **D (frontier direct):** gpt-6.1-sol, effort high, at most 32,000 completion tokens. It sees the same text and answer
  slots and gives a JSON answer. No answer, or unparsable output, = abstention.
- Order: E first, then D.
  - Budget: the shared OpenAI ledger cap is $150; the runner stops launching calls at $148.
  - If D cannot finish all 150 under the cap, it is run on levels 3-5 first. B3 then uses only the items both arms
    have.
  - Any shortfall is reported.

## Bars
- B1 (safety): E gives at most 1 wrong answer in 150.
- B2 (coverage): E's correct answers >= 0.70 x 150 (105).
- B3 (value against frontier direct, where it matters): on levels 4-5 (60 items), E correct - D correct >= 9 (0.15),
  over paired items.
- B4 (gold sanity): every item's gold passes its check(), and z3 agrees on every item.

Reported without a bar:
- P's wrong answers (the silent errors E is meant to remove);
- per-family and per-level tables;
- E's abstention reasons;
- spend.

## Predictions (written before any item exists)
- E wrong 0-1; E correct 0.75-0.92 (abstentions mostly from crashes and from models that disagree on how to read a
  rule).
- P wrong 3-12 (silent errors), more on roster and meetings, where any misread rule still gives a schedule.
- D correct falls with level: about 0.9 at level 1, 0.3-0.6 at level 5. Most at risk: project and vans (optimum
  proofs) and roster (many coupled rules); islanders at 18 people.
- B3 holds if D falls as predicted; if D stays above 0.8 at level 5, the families are not hard enough for this model,
  and the verdict will say so rather than claim an edge.

## What would change the plan
- If B1 fails, the certification layer is not a certificate on new families; the next step is to find which check let
  each wrong answer through, on these items, now burned.
- If B2 fails while B1 holds, the contract holds but coverage is the product problem.
