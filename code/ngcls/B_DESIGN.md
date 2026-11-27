# B — the experiment the title needs (design, 2026-10-15; not yet a preregistration)

## Question
When a **frozen** language model must parse the input **itself**, does a vector symbolic memory that does the
composition improve its multi-hop accuracy over the same model's own single pass and chain of thought? And is
any gain due to the VSA specifically, or to externalised symbolic structure in general?

## What the audit (AUDIT_2026-10.md) rules out, and how this design avoids each
| Defect found | How B avoids it |
|---|---|
| Regex / generator supplies spans, roles, pairing | The LM extracts triples from raw passages by prompting; no regex on the input |
| Gold query / candidate identities | The LM decomposes the question into a start entity and a relation path |
| Gold evidence in gold traversal order (2Wiki 300/300 trivial) | Raw `context` paragraphs incl. distractors; gold `evidences` used ONLY to score extraction quality, never fed to any arm |
| Oracle fact selection; eval set filtered to rows the parser gets right | No selection; every arm on the same unfiltered test rows |
| Commutative binding (S5) | Value marked by a fixed permutation: store = sum bind(bind(s, r), P(o)) |
| Selection on test; data fixed across seeds | train / val / test; configurations and any threshold chosen on val; seeds vary sampling |
| Supervision asymmetry (identity head 9,171 labels vs LoRA 775) | Symbolic arms use NO trained component (prompted extraction + unsupervised identity); LoRA gets the full train split |
| Trivial solvability unknown | Trivial baselines reported first |

## Arms (one frozen LM, same raw input, same test rows)
1. **direct** — few-shot, passages + question -> answer.
2. **cot** — same shots with worked reasoning.
3. **extract + dict_exact** — LM extracts (s, r, o) per paragraph; LM decomposes question -> (start, [r1, r2]);
   compose by normalised exact-string lookup.
4. **extract + dict_ident** — as 3, but entity mentions and relation phrases are mapped to identities by the
   LM's own hidden-state similarity (cluster mentions; nearest identity for the query side). Table storage.
5. **extract + vsa_ident** — as 4, identities get random codebook rows, facts are superposed in one HRR vector
   with role-marked binding, queries unbind and clean up over stored identities.
6. **extract + vsa_hash** — as 3 but in HRR (exact-string codes): isolates the cost of superposition.
7. **LoRA** — same LM, trained on the train split (passages + question -> answer), selected on val,
   training accuracy logged. (Second phase; needs GPU training time.)
8. **trivial** — e.g. answer = an entity from the question's first paragraph; majority; reported first.

## What each comparison licenses
- 5 vs 1/2: does a VSA memory improve the frozen LM? (the title's claim)
- 5 vs 4: does HRR superposition add anything over a table with the same LM-derived identities? Expected: no,
  or slightly worse (cross-talk). If 5 ≈ 4 the honest claim is "externalised symbolic structure helps, and a
  fixed-size VSA vector implements it without loss" -- which is the neuromorphic-relevant statement.
- 4 vs 3: does LM-derived identity (handling "director of" vs "directed by", aliases) beat exact strings?
- 5 vs 7: against fine-tuning, on equal inputs.

## Data and model
2WikiMultihopQA compositional dev (two hops), raw `context` paragraphs; MuSiQue second. Pilot on rows 0-49,
labelled a pilot. The registered run uses a disjoint row range with bars set from the pilot's design, not its
numbers. Frozen LM: Qwen2.5-7B-Instruct (instruction-following needed for extraction); scale check later on
14B / 32B. Greedy decoding.

## Pilot exit criteria (before any preregistration)
- Extraction: report recall of the gold evidence triples among extracted triples (diagnostic only).
- Question decomposition: report how often the start entity and relations are recoverable.
- If extraction recall is very low, the symbolic arms cannot work on this data and that is itself the finding
  -- reported, not engineered around by adding gold structure back.
