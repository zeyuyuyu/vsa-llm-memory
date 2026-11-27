# B15 pre-registration: the prompted reader, configuration chosen on development data (before any run on the B15 set)

Date: 2026-10-19. **Disclosure: designed after B13 and B14.** B14's primary arms failed; its second prompt came close
(Qwen3-8B 0.84 / 0.90). A development sweep on the two spent hand-written sets (B13 + B14, 80 stories; reader size 1.7B /
8B, prompt B / new prompt C, name merging on / off) chose **Qwen3-8B, prompt C** (five worked examples: a distractor clause,
a pronoun after "took X out of Y", a state statement with two moves, an explicit swap with a specific place, a plan that
did not happen), **no name merging** (the B14 suffix rule merged "room" into "living room"). Development scores:
B14 set 0.914 / 0.900; B13 set 0.963 / 1.000 (inflated: prompt C's first example echoes a B13 story). The small-model arm
is Qwen3-1.7B, prompt B, no merging (development 0.778 / 0.80 and 0.863 / 0.70).

## Test set (new, hand-written, frozen before any run)

`b15_accept.jsonl`: 40 new stories, the same specification and categories (A 8, B 8, C 7, D 7, containers 10),
91 questions, written after the configuration was chosen; self-check 0 mismatches (one of my labels was corrected by the
self-check before freezing: "they took the jackets" was out of scope and was reworded); sha256 `5713109115147aab828d06b46a9c057d7f8b215e7641e184e1b734b7a5fac44c`.

## Claims and bars

- **P1** Qwen3-8B / prompt C / no merging: locations Q-acc >= 0.90. **P2** containers >= 0.85.
- Reported: the 1.7B arm; the reader's generated tokens per sentence; Qwen3-1.7B and Qwen3-8B (thinking) answering
  directly; S0 and the B13 writers; event precision / recall.
- **P3 (long stories)** for this configuration: the B13 amendment-5 long set, with B14's caveat (Qwen3-8B filtered those
  stories, so an 8B reader is graded on stories pre-filtered for it: reported as an upper bound, not a claim).

## Prediction (stated so it can be wrong)

P1 and P2 hold, with locations lower than on the B14 development set (0.88-0.93). The 1.7B arm fails both.
