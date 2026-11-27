# V-158 AMENDMENT 1 — before launch, before any registered-row generation

Committed after prereg 26b5404 and before V-158 has generated anything on rows 50–349. Its only output so far is
the smoke on pilot rows 0–5.

## Why

The second independent code review (`review_C.md`) found 0 result-breaking issues and 1 MATERIAL one, C1-1. The
prereg says "every generation … is saved", but b1v2.py kept only the parsed triples, not the 10 raw extraction
generations per row, and did not save the identity-cluster maps. Without them:
- "the LM did not extract the fact" cannot be told apart from "the parser dropped the line";
- vsa_ident (the arm P1/P2 test) and P4's entity merges cannot be re-derived.

## Change (logging only; no arm, prompt, parser, bar or row changes)

- `raw[i]["extract_raw"]`: the list of raw extraction generations for row i, in paragraph order.
- `raw[i]["clusters@0.70"]`: the entity and relation cluster maps (norm-class → cluster id) and the surface form
  embedded for each class.

The amended runner was re-run under the reviewer's mocked end-to-end harness (`revC/b1v2_mock.py`, stubbed LM, CPU):
it runs, and the new fields are present.

## Disclosures carried from review C (no code change)

- **C1-2.** P3 is close to decided in advance by load. At B1 store sizes (36–114 triples) table and HRR retrieval
  agree on 600/600 random queries, and with the low base rate the bar fails only if the HRR loses more than ~37% of
  retrievals. A P3 HOLD will be reported as "at this load and base rate", never as "superposition costs nothing on
  natural text".
- **Minor.** P3's disagreement count is on EM even in the containment and F1 blocks. The CoT determinism check
  compares EM only. "First triple wins" changes dict_exact relative to V-155, so dict_exact is not part of the
  determinism comparison.
- **C3-4 (applies to V-155, V-157, V-158).** Qwen2.5-7B-Instruct's generation_config sets repetition_penalty 1.05,
  which also applies with do_sample=False. "Greedy" in every B prereg means greedy with that penalty.
