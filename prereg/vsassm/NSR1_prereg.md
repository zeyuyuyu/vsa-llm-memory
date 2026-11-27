# NSR1 pre-registration: impact reasoning over the tracked state of the four application scenarios

Date: 2026-11-06. Brief: docs/neurosymbolic_brief.md. Written after NSR0 (results/nsr0_verdict.txt) and committed
before any NSR1 arm runs on held-out data and before the held-out split exists.

## Question

Readable Memory tracks where things are from a site's log (system feeds plus human messages); on test2 it answers
"where is X" at 0.959 against 0.396 for the system of record and 0.543 for Qwen3-32B thinking. NSR1 asks whether that
state supports *impact* questions, which compose the current state with fixed site facts, and where the advantage comes
from: the tracked state, or the composition step. NSR0 found that on relation symbols the VSA rule memory adds nothing
a table cannot do; NSR1 therefore carries a table and a hand-written graph query on the same inputs, and predicts that
they tie with the VSA rule memory exactly.

## Inputs (identical for every arm)

- The log (`text_lines`), the site asset list (registry IDs and aliases, as in apps/real `assets_block`), and a
  **site catalog** of static facts the logs never change (`nsr/nsr1_catalog.py`): rack -> aisle (warehouse), place ->
  room (home), cluster -> region and service -> database it needs (ops). Biobank needs no catalog (box, rack and
  freezer containment is all state). The LLM arms get the catalog as plain sentences.
- State facts are (thing, in, place); catalog facts are (place, part_of, area) and (service, needs, database primary).

## Impact questions (`nsr1_catalog.questions`, all asked about the state at the end of the log)

| Scenario | Type | Example | Composition |
| --- | --- | --- | --- |
| Biobank | freezer_down | "FRZ-07 has defrosted. Which specimens are affected?" | in o in o in (specimen, box, rack, freezer; all state) |
| Warehouse | aisle_closed | "aisle B is closed for a forklift repair. Which pallets are blocked in it?" | in (state) o part_of |
| Home | room_cleared | "The hallway is being repainted today. Which of Grandpa's things have to be moved out of it?" | in (state) o part_of |
| Ops | region_down | "If us-east-1 goes down, which services lose their running instance?" | in (state) o part_of |
| Ops | host_down | "If host db-11 fails, which services lose the database primary they need?" | needs o in (state) |

Every target of the type is asked (all freezers, aisles, rooms, regions, hosts in the log), so some answers are empty.
The answer is a set of registry IDs. **Primary metric: set-exact accuracy** pooled over all questions; also reported:
per scenario, the non-empty subset, and micro-F1 over answer items. Answer class (specimens, pallets, items, services) is
part of the question type and given to every arm.

## Arms

1. **RM + VSA rule memory (primary).** State = Readable Memory's final state (the shipped reader: Qwen3-1.7B +
   `b16_models/student_s0`, `--verify`, v4 configuration, as committed at 707351b; biobank chains flattened to
   one-hop parents). Rules = relation-type composition rules (e.g. (in, part_of) -> in, (needs, in) -> needs_host) and the
   asked relation per question type, learned from **n = 1 worked trace per question type** taken from dev truth (a
   trace lists the hops and each intermediate conclusion, as CLUTRR's proof_state does), stored in R1's FHRR rule memory
   (D = 2048, three codebooks). Inference: every directed fact path (<= 4 hops) from a candidate of the answer class to
   the target, folded with any bracketing; the candidate is in the answer if a path folds to the asked relation.
   Also reported: n = 3.
2. **Controls on the same RM state:** (2a) the same learner with a count table; (2b) a hand-written graph query
   (`nsr1_catalog.answer`, rules given).
3. **Qwen3-32B thinking over the whole log** with the asset list and the catalog (apps/real protocol: greedy, 8,192
   reasoning tokens, budget forcing for runs that hit the limit; strict and forced scores, forced is primary).
   Second model for robustness: Qwen3-8B thinking, same protocol.
3b. **Diagnostic: Qwen3-32B thinking over the RM state written as sentences** ("X is in Y") plus the catalog, no log.
   Compared with 2b on the same state, it separates the LLM's composition from its state tracking.
4. **System of record + graph query** (state from system feeds only, `log["sor"]`).
- **Sanity:** truth state + graph query must score 1.000 (checks questions and scorer).
- **Negative control:** RM state with places permuted among things (seeded) + graph query.

Every arm-1 answer item carries a **trace**: each hop is (x, rel, y, source), where source is the log line whose reading
last wrote that state fact into the memory, or "catalog".

## Splits

- Design: `dev` (2 logs per scenario) and the retired `test` (10 per scenario) are development data. Development
  numbers seen before this prereg (RM state + graph query, set-exact): dev 0.935 (46 questions), retired test 0.857
  (230); system of record 0.217 / 0.309. The retired test's follow-up labelling flaw (apps/real/README.md) counts against
  RM there.
- **Held-out: a fresh split `nsr1`, 10 logs per scenario, seeds 400+**, generated after this commit with apps/real's
  generator (`world.py`, `gen.py` as committed at 707351b; Qwen3-32B writes and checks the human lines) through a
  wrapper that writes to `nsr/data/`. ~230 questions. Scored once. `test2` is not used.

## Bars (held-out, pooled set-exact)

- **P1:** arm 1 >= 0.80.
- **P2:** arm 1 beats arm 3 (Qwen3-32B, forced) by >= 0.20; counts only if arm 1 >= 0.60 (usefulness floor).
- **P3:** arm 1 beats arm 4 (system of record) by >= 0.30.
- **P4:** arm 1 gives the same answer set as 2b on >= 99% of questions and as 2a on >= 99%. This is a bar *against* a
  VSA-specific claim: it states that the composition mechanism is interchangeable.
- **P5:** every arm-1 answer item has a trace, and an independent checker confirms each hop is in the RM state or the
  catalog and that each cited log line's reading wrote that fact (trace validity = 1.00). Trace correctness (every hop
  true in the truth state) is reported.
- **Must hold or the run is void:** sanity = 1.000; negative control <= arm 4 + 0.05.

## Verification

Independent scorer `nsr/nsr1_score.py` (committed before the held-out data exists), recomputing every number from the
per-question answer files and the logs' truth; independent trace checker; manual inspection of every arm-1 miss and 20
arm-3 misses (is a "reasoning" miss a parse or format failure?); second LLM (Qwen3-8B); literature check on the
conclusion (SUPPORTED / CONFLICTS / NO PRECEDENT), recorded in the verdict.

## Predictions

- P1 holds (about 0.88; all misses are state misses, none composition misses).
- P2 holds: Qwen3-32B about 0.35-0.45, because set answers compound its per-item tracking errors.
- P3 holds: system of record about 0.30.
- P4: identical answers on 100% of questions (three rules, read back exactly).
- P5: validity 1.00.
- 3b: Qwen3-32B over the RM state scores within 0.05 of 2b, i.e. the LLM composes well when given the state, and its
  gap in arm 3 is state tracking.
- Negative reading if P2 fails: the LLM tracks this state well enough over a ~10k-token log that impact questions add
  nothing to the where-question result.
- What a positive result would and would not support: "a continuously tracked, readable state composed with site facts
  answers impact questions that an LLM reading the log misses"; not "VSA reasons" (NSR0, P4).

## Compute

PACE first (gts-nimam6 free tier, paid account if the free queue stalls; spend reported): generation (Qwen3-32B),
reader (Qwen3-1.7B), LLM arms (Qwen3-32B, Qwen3-8B). The A800's free GPUs only if PACE is queueing; never GPU3.

## Amendment 1 (2026-11-06, on development data only; the held-out split does not exist yet)

1. **Negative-control bar recalibrated.** On set questions with many empty answers, chance is the share of questions
   whose true answer is empty ("always none"), and the system of record sits *below* it (dev 0.217 vs 0.326; retired
   test 0.309 vs 0.339), so "negative control <= arm 4 + 0.05" would void a sound run by construction. New bar: negative
   control <= always-none + 0.05. Dev / retired test: negative control 0.326 / 0.335, always-none 0.326 / 0.339.
2. **Always-none is reported next to every arm**, and P3 is also reported against it. P1-P5 are unchanged.
3. **Development numbers seen (independent scorer `nsr/nsr1_score.py`):** arm 1 = arm 2a = arm 2b on 46/46 (dev) and
   230/230 (retired test) questions: 0.935 / 0.857; arm 4 0.217 / 0.309; sanity 1.000; traces: validity 1.00 on 81 / 412
   answer items, correctness 0.988 / 0.930. Rules learned from one worked trace per question type (dev truth):
   (in, in) -> in, (in, part_of) -> in, (needs, in) -> needs_host.
4. **Literature found after the prereg (positioning, no change to the design):** MEME (arXiv 2605.12477) already poses
   impact questions over changing state (Cascade, Absence) and reports memory systems at <= 0.06; with an explicit
   dependency graph, plain traversal answers impact questions (Tacheny, arXiv 2601.07342). NSR1's claim can therefore
   only be about the tracked state that the traversal runs on, which is what P2-P4 test.

## Amendment 2 (2026-11-07, after all LLM runs and before any scoring)

1. **Decoding engine.** The LLM arms ran with vLLM 0.11 (greedy, thinking on, the same chat template, 8,192 new tokens,
   b6_qwen.py's record format; nsr/nsr1_vllm.py) on the A800 box, not HF generate on PACE: PACE estimated a week of
   queueing, and HF generate on an A800 needed about an hour per 8 problems (transformers falls back to materialised
   repeat_kv whenever a padding mask is present). 12 problems answered earlier with HF generate are kept only as a
   cross-engine check.
2. **Budget-forcing answer budget raised from 64 to 512 tokens** for every arm (same CLOSE phrase, greedy). Observed on
   the held-out outputs before any scoring: 40 of the 48 forced Qwen3-32B answers were cut off inside an explanation
   before reaching an "Answer:" line, because set answers ("which pallets ...") are preceded by a list; apps/real's 64
   tokens were sized for single-place answers. Raising the budget can only help the LLM baselines, i.e. it works
   against P2. The 64-token forced files are kept (forced_*.jsonl); the 512-token ones are forced512_*.jsonl.
   Nothing else changes; the answer parser is the registered one.

## Amendment 3 (2026-11-11): a frontier-model arm, added AFTER the NSR1 verdict (post hoc, at the user's request)

The NSR1 verdict named "a frontier long-context model" as the strongest baseline not run (a 2026 paper found long-context
GPT-5-mini beating a Mem0 fact memory on recall). This arm is added after the results are known and is labelled as such.
- **Gateway and model:** Sidrune (OpenAI-compatible; the only working gateway on 2026-11-11: the OpenRouter keys are
  expired or at their limit and the CommonStack key is inactive). Model from the OpenAI GPT-5.6 / 6.1 family, chosen by
  the measured cost of a 10-problem pilot (gpt-5.6-terra vs gpt-6.1-sol); the pilot answers are not scored before this
  amendment is committed, and the chosen model's 10 pilot answers count as part of its run.
- **Protocol:** the same prompts as arm 3 (nsr/out/held/nsr1_llm_problems_nsr1_arm3.jsonl), one call each, default
  reasoning, max_completion_tokens 32,000; no budget forcing (the hidden reasoning cannot be handed back); a call that
  ends with finish_reason "length" counts as no answer. Registered parser; the underscore-normalising parse and the
  exclusion of the ambiguous home questions are reported as sensitivity analyses, as for arm 3.
- **Bar P2-F (post hoc):** arm 1 - frontier >= 0.20 set-exact over all 230 questions.
- **Prediction:** the frontier model is far stronger than Qwen3-32B (0.28): about 0.55 (range 0.45-0.70), so P2-F holds by
  about 0.35. If it reaches 0.73 or more, P2-F fails, and the claim "an LLM reading the log misses impact questions" is
  restricted to the open models tested.
- If the spend allows, the same model on NSR3's 460 temporal questions (nsr/out/held/nsr3_llm_problems_nsr2.jsonl), bar
  T2-F: history engine - frontier >= 0.30; prediction about 0.40 for the frontier model.
- Model choice (2026-11-11, before any frontier answer is scored): the user asked not to economise ("不要怕花钱"), so the
  arm uses the strongest model the gateway offers, **gpt-6.1-sol**, not the cheaper pilot candidate.
- Decoding settings (2026-11-11, before any frontier answer is scored): **reasoning_effort "xhigh", streamed** (unstreamed
  xhigh calls hit Cloudflare's ~100 s cut-off, HTTP 524), max_completion_tokens 32,000. The gateway's default effort
  barely reasons (~130 output tokens per answer in the pilot), so the 10 pilot answers (default effort) are EXCLUDED and
  all 230 problems are run with xhigh. Pilot spend: $0.18 (gpt-5.6-terra) + $0.04 (gpt-6.1-sol); about $0.015 per xhigh
  call. Streamed usage does not report reasoning tokens, so the arm's reasoning length is recorded as wall time only.
