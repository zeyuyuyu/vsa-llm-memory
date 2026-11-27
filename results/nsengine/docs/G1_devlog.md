# G1/G2 development log (design v1, general engine)
Every variant of the generic translator and the certification layer is listed in the order it was tried, with its dev
result. Development uses burned items only: NSR7 T1 (SRE), NSR6b test (large SRE) and the ZebraLogic set used in NSR6.
None of these families may serve as a held-out family in NSR8.
Translator: gpt-6.1-sol, reasoning effort medium, K = 2 formalisations, generic prompt runs/gtranslate.py. Spend goes to
the shared ledger runs/spend_ledger.jsonl.

| # | change | items | frontier + code alone (primary) | certified (K = 2) | spend |
|---|---|---|---|---|---|
| v0 | first prompt | 10 SRE (NSR7 T1, random, seed 0) | 6 correct / 4 wrong | 5 / 3 abstain / **2 wrong** | $0.23 |
| v1 | fix: numbering dropped every block that started with a heading line (all rack sentences and the first service paragraph never reached the model) | same 10 | 10 / 0 | 10 / 0 / 0 | $0.26 |
| v2 | line-based numbering, list markers removed (ZebraLogic clues); `uses()` / `ignore()` and the sentence-coverage check (two variants: missing only; strict = also no sentence ignored by one and cited by another); ZebraLogic adapter | 51 dev: 21 SRE d3-d5 (7 per question type), 5 large SRE (NSR6b), 25 ZebraLogic (18 of size >= 5x4) (data/g1_dev_ids.txt) | SRE 20/21, 1 crash; large 3/5, 2 crashes; Zebra 25/25; **0 wrong** | SRE 15 / 6 abstain; large 2 / 3; Zebra 25; 0 wrong (all three coverage variants identical) | $1.56 |
| v2b | fix (harness bug, no new model calls): core tags were compared as raw strings ("S35,S36") instead of sentence ids, so the reason check for "impossible" failed on agreeing models | same 51, saved formalisations re-checked | same | SRE 19 / 2; large 3 / 2; Zebra 25; **0 wrong**; every remaining abstention is a program that crashed | $0 |
| v3 | one repair round: a program that raises an error is sent back with the error text only (never an answer); a timeout or "unsat" is not repaired | the 4 items with a crash (data/g1_dev_crash_ids.txt), re-formalised | 4/4 (3 repairs used) | 4/4 with no or missing-only coverage; strict coverage abstains on 1 (one model ignores the rack sentences, which do not bear on a maintenance plan, the other cites them) | $0.37 |

**Decision after v3:** coverage = missing only. Strict coverage adds false abstentions where models differ on facts that do not
bear on the answer. A relevant fact one model leaves out is caught anyway by the cross-checks (the other model rejects
the answer, or they disagree on uniqueness or optimality).
**Dev total (v2b + v3): 51/51 certified correct, 0 wrong, 0 abstentions; about $2.4.**

Notes
- v0's two certified wrong answers were both formalisations missing the same facts *because the harness never showed
  them*: a joint omission passes every cross-check when the answer keys are the same. This is the case the coverage
  check is meant for, when the omission is the model's.

## After NSR8's engine arm (held-out items now burned; for the NEXT version only, NSR8 itself is frozen)
- All 8 NSR8 abstentions were vans (bin packing):
  - 7 were "optimum not proven": the solver hit the 60 s limit; the models had no symmetry breaking.
  - 1 was a false disagreement: one model added symmetry breaking (vans used in number order), which rejects the
    other model's equally good answer with unordered van numbers.
- Design change to try (with a fresh test): the translator tags symmetry-breaking constraints (e.g.
  `add(c, "SYM")`). The harness uses them when it solves and proves optimality, and drops them when it checks
  another formalisation's answer.

| # | change | items | primary alone | certified | spend |
|---|---|---|---|---|---|
| v4 | prompt: symmetry-breaking constraints allowed but tagged "SYM"; harness: SYM constraints used by solve and the optimum proof, dropped by fix / other / enum | the 30 NSR8 vans items (burned) | 30/30 | **30/30, 0 abstain** (NSR8 frozen engine: 22 / 8 abstain) | ~$0.7 |
| v4 regression | same | 40 burned NSR8 items of the other four families (10 each) | | 40/40 (meetings, roster, project, islanders 10 each), 0 wrong | ~$0.7 |

## Local translators (A800; same generic prompt; 51 dev items; no API spend)
| translator | primary alone (single model, no checks) | certified (K = 2) |
|---|---|---|
| Qwen3-8B thinking, 8,192 tokens | every reply cut off in thought: no code | — |
| Qwen3-8B non-thinking | 3 correct / 9 wrong; most crash or write no code | 0 correct / 0 wrong |
| Qwen3-8B non-thinking + one repair round | 4 correct / 13 wrong | 0 / 0 |
| Qwen2.5-32B-Instruct non-thinking | 6 correct / 9 wrong; 33 crash | 0 / 0 |
Reading:
- Local models up to 32B cannot drive the general engine: coverage is 0. Most failures are crashes and models that
  skip the sentence accounting (uses / ignore).
- The certification layer still turns every one of their 22 wrong single-model answers into an abstention.
- The local tier needs a stronger local model or a fine-tuned translator (design v0's synthetic-pairs idea).
