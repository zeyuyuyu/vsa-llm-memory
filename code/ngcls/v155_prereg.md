# V-155 pre-registration — B1: does a VSA memory improve a frozen LM's two-hop reasoning on natural text when the LM parses the text itself?

Written and committed BEFORE the run. This is the registered version of the B pilot (ledger, 2026-10-15),
on rows DISJOINT from the pilot's. It answers the manuscript's title question on natural text with every
defect named in AUDIT_2026-10.md removed: no regex, no gold spans or identities, no gold evidence order, no
oracle selection, no evaluation-set filtering; gold evidences used only to score extraction afterwards.

## Design
`b_pilot.py` verbatim (commit with this prereg), Qwen2.5-7B-Instruct, greedy. 2WikiMultihopQA compositional dev,
**rows 50-349 (n = 300)**, raw context paragraphs with distractors. Arms: direct, cot, extract+dict_exact,
extract+vsa_hash, extract+dict_ident, extract+vsa_ident, trivial. Identity clustering threshold **tau = 0.70**,
fixed here; it was the best dict_ident threshold on the pilot rows 0-49, which do not overlap these. HRR arms
role-mark the value with a fixed permutation. Metrics EM (primary) and token F1. Paired bootstrap over rows,
20,000 resamples, seed 20261015.

## Bars
- **P1 (the title's claim):** `EM(vsa_ident) - EM(direct) >= 0.05` with the 95% CI excluding zero.
- **P2:** `EM(vsa_ident) >= EM(cot)`, CI on the difference excluding zero in the positive direction.
- **P3 (does superposition cost anything?):** `|EM(vsa_ident) - EM(dict_ident)| <= 0.03`.
- **P4 (does LM-derived identity help over exact strings?):** `EM(dict_ident) - EM(dict_exact) >= 0.02`.
- Descriptive: extraction recall, start-entity and relation-path match rates, the both-facts-extracted subset.

## What I expect (separately from the bars)
P1 and P2 FAIL by a wide margin: the pilot had vsa_ident .06 against direct .24 and cot .32. P3 holds (the
pilot had vsa equal to the table). P4 is a coin flip (pilot .08 vs .06 at tau .70). If P1 fails as expected,
the manuscript's title is not supported on natural text, and the registered result says so.

## Protocol
Scorer `v155_score.py` frozen with this file, run verbatim.
