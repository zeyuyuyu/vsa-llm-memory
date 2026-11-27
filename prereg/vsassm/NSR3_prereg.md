# NSR3 pre-registration: temporal impact questions need reasoning over the state's history

Date: 2026-11-09. Written after development on dev, the retired test and nsr1; committed before any temporal question is
generated on the held-out logs.

## Question

NSR1's impact questions ask about the state at the end of the log. Real impact questions are often about a past window:
"FRZ-07 logged a defrost between 10:41 and 12:15: which specimens were inside it at any point in that window?" Things
moved out since still count, so the final state cannot answer them, and the engine must reason over the history of the
tracked state. Does a history engine over the read state answer them, and how do an LLM over the log, the final state and
the system of record do?

## Questions (nsr/nsr3_questions.py at this commit)

For every NSR1 impact question of a log (same targets), two windows [line i, line j] drawn deterministically from the
log's id, each spanning 15-35% of the log and starting after its first 20%; times are the log's own clock. Gold = the
union over the window of the truth states (the state when the window opens and after every line in it), replaying the
log's events. Wording: "{target} logged a defrost between {t1} and {t2}. Which specimens were inside it at any point in
that window?" and analogues (home asks for "tracked items", avoiding NSR1's "Grandpa's things" ambiguity).

## Arms

1. **History engine**: the same union over the read state's per-line snapshots (the shipped reader, as in NSR1).
2. **Final state only** (NSR1's engine): ignores time.
3. **System-of-record history**: union over the states built from system-feed lines only.
4. **Qwen3-32B thinking over the whole log** with the asset list, the catalog and the temporal question (NSR1 protocol as
   amended: vLLM greedy, 8,192 tokens, budget forcing with 512 tokens, registered parser; the underscore-normalising
   parse reported as a sensitivity analysis).
- **Sanity:** truth history + the same union = 1.000.

## Data

Held-out: the 40 nsr2 logs (seeds 500+), generated for NSR2 after the NSR2 prereg. NSR2 uses only their final states
(end-of-log impact questions); no temporal question on them is generated before this commit. 2 windows x 230 questions
= 460 temporal questions. Reader output: the same nsr2 reader run NSR2 uses.

## Bars (nsr2, set-exact, all questions)

- **T1:** history engine >= 0.75.
- **T2:** history engine - Qwen3-32B (forced) >= 0.30.
- **T3:** history engine - system-of-record history >= 0.40.
- **T4:** history engine - final state only >= 0.50.
- **Must hold:** sanity = 1.000.
Reported: per scenario, micro-F1, the LLM's strict score and hit-limit count, manual inspection of every history-engine
miss in 2 scenarios and 20 LLM misses.

## Development numbers (seen)

History engine 0.902 (dev, 92 q) / 0.822 (retired test, 460) / 0.874 (nsr1, 460); final state only 0.022 / 0.061 / 0.037;
system-of-record history 0.152 / 0.220 / 0.183. Warehouse is the weakest scenario (0.65-0.69 on test/nsr1): temporal
windows accumulate the reader's transient errors over time. The LLM arm was not run in development.

## Predictions

T1 holds (about 0.85). T2 holds: Qwen3-32B around 0.2-0.3 (its NSR1 end-state score was 0.28; windows are harder, but
some questions need only part of the history). T3 and T4 hold. Sanity holds.
What a pass would support: temporal impact questions need an engine over a tracked history, and the read history
supports them. What it would not: that the LLM cannot do temporal reasoning in general (one model, one protocol).
