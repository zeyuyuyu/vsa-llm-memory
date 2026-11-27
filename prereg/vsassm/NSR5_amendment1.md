# NSR5 amendment 1: a second frontier model (Claude Fable 5.1)

Date: 2026-11-11. Written before any Claude call on NSR1/NSR3/NSR5 questions, and before any NSR5 gpt-6.1-sol answer was
parsed or scored (the only Claude calls so far are two 17-token key probes).

Why: so far a single frontier model has been tested. NSR1 found that gpt-6.1-sol ties the pipeline, so whether
"frontier models tie at one day" holds for more than one model is now the question that matters. The user approved a
separate Anthropic budget (USD 180) for a Claude arm.

## Arm
claude-fable-5-1 (Anthropic's most capable widely released model), effort "xhigh" (the setting used for gpt-6.1-sol),
adaptive thinking (always on), max_tokens 64,000, streamed (nsr/nsr_claude.py). The prompt is sent as two text blocks,
split just before "\n\nQuestion:", with a cache breakpoint on the first block. The model sees exactly the prompt text the
other arms see. No refusal fallback: a declined call counts as no answer and is reported. Dropped or errored calls are
re-run (frontier amendment 1).

## Order (stop when the budget runs out; a hard check before each call)
1. NSR5 K = 8 (69 questions).
2. NSR1 K = 1 (230 questions, problems nsr/out/held/nsr1_llm_problems_nsr1_arm3.jsonl).
3. NSR5 K = 4 (161) if budget remains, then NSR3 (460).

## Measures
The same as for gpt-6.1-sol, scored with the same registered parsers and scorers. Z1, Z2 and Z3 are computed for Claude
as secondary bars (Z2 uses Claude's own K = 1 gap from step 2); gpt-6.1-sol remains the primary arm.

## Prediction (low confidence, made before any result)
Claude at K = 1: within 0.05 of gpt-6.1-sol (about 0.90). At K = 8: set-exact about 0.5, so Z1 holds for Claude too.
Same direction as the NSR5 prediction for gpt-6.1-sol.

## Amendment 2 (2026-11-11, before any Claude answer was parsed or scored): effort "high"
Calibration on the first two K = 8 questions (site 0, aisle A and B) at effort xhigh:
- Aisle A used all 64,000 output tokens on thinking and gave no answer (USD 4.32; the prompt is 89,651 Claude tokens).
- Aisle B was still running after 22 minutes when the gateway cut the stream (no usage returned; booked at an estimated
  USD 3.22).
At that rate K = 8 alone would cost more than the USD 180 budget. Anthropic's effort levels are not calibrated to
OpenAI's either, so matching "xhigh" by name buys no comparability.
Rule:
- Run at effort "high", keeping max_tokens at 64,000 and the same model.
- If any of the first site's questions exhausts the budget at "high", stop. Discard the "high" records and rerun the
  whole arm at "medium", so that one effort level answers the whole arm.
- The two xhigh calibration records stay in nsr5_K8_claude-fable-5-1_xhigh.jsonl and are not scored.
Runner fix: a stream cut mid-body (httpx.RemoteProtocolError) is now recorded as an error with an estimated cost and
re-run on resume, instead of killing the process.

## Amendment 3 (2026-11-11, before any Claude answer was parsed or scored): effort "medium"
At effort "high", site 0 aisle A again used all 64,000 tokens with no answer (USD 4.32). The three other site-0 calls were
stopped after 26 s (booked at an estimated USD 0.23 each). As amendment 2 requires, the arm now runs at "medium". If
aisle A also exhausts the budget at "medium", the Claude K = 8 arm is reported as infeasible within the budget (with
these calibration calls as the evidence), and the Claude budget goes to K = 1 (NSR1) at "medium".
Calibration spend so far: USD 12.56 (ledger).

## Amendment 4 (2026-11-11): K = 8 runs at "medium"; K = 1 order fixed for a possible budget cut
At "medium", site 0 aisle A finished: 50,241 output tokens, 505 s, USD 2.60, an answer was given. It was not parsed or
compared with gold. So the K = 8 arm runs at "medium" (all 69 questions; this record is its first).
K = 1 (NSR1, 230 questions) may not fit in the remaining budget. Its problems run in a fixed order: round-robin over the
four scenarios, one whole site at a time, with sites shuffled within each scenario (seed 20261111):
nsr/out/held/nsr1_llm_problems_nsr1_arm3_interleaved.jsonl. If the budget stops the run, the answered prefix is the
sample, and the pipeline and gpt-6.1-sol are re-scored on exactly those questions for the K = 1 gap that Z2 uses.

## Amendment 5 (2026-11-11, user instruction; before any Claude answer was parsed or scored): model claude-opus-5-5
The user asked for Opus 5.5 instead of Fable 5.1, which is too expensive. Fable at "medium" averaged about 45k output
tokens and USD 3 per K = 8 question, so K = 8 alone would have used the whole budget. The second frontier arm is now
claude-opus-5-5 (USD 4 / 20 per million tokens), at effort "medium" (its default), with max_tokens 64,000 and the
same prompts, prompt split and caching. The other questions of a site start 60 s after its first one, so that they
read its cache. The runner joins text blocks with line breaks (frontier amendment 2).
Order: K = 8 (69), then K = 1 in the interleaved order of amendment 4, under the same USD 180 ledger. Fable 5.1 spend so
far is USD 57.48 (conservative estimates included), which leaves about USD 122.
The 10 K = 8 answers Fable 5.1 gave at "medium" (nsr5_K8_claude-fable-5-1_medium.jsonl) are kept as an exploratory
sample. They are not used for any bar.
Prediction for Opus 5.5 (unchanged in direction): K = 1 about 0.90, K = 8 set-exact about 0.6.

## Amendment 6 (2026-11-11, before any Opus 5.5 answer was parsed or scored): queue for the remaining budget
Opus 5.5 at "medium" costs about USD 0.05 per K = 8 question once a site's cache is warm (about 1-9k output tokens).
The whole queue therefore fits in the budget:
1. K = 8 (running).
2. K = 1 (NSR1, interleaved order).
3. K = 4 (161).
4. NSR3 (460 temporal questions).
5. Effort sensitivity, not used for any bar: Opus 5.5 at "xhigh" on K = 8, then on K = 1. It answers whether the
   frontier-vs-pipeline comparison depends on how long Opus thinks.
Every step has the same parsers (with --repair, frontier amendment 2) and scorers. Opus is a secondary arm:
gpt-6.1-sol stays primary for Z1-Z3 and T2-F.

## Amendment 7 (2026-11-12, before any 64k answer exists): K = 4 64k sensitivity through OpenAI direct
Frontier amendment 1, rule 3 re-runs the budget-exhausted questions with a 64,000-token budget. At K = 4 these are the 21
hit_limit records of nsr/out/api/nsr5_K4_gpt-6.1-sol_xhigh.final.jsonl (home 8, ops 13). The Sidrune wallet holds USD 1.53,
so the user supplied an OpenAI API key (2026-11-12). The re-run uses the same model id (gpt-6.1-sol, listed by
api.openai.com/v1/models), the same prompts, reasoning_effort xhigh and max_completion_tokens 64,000, streamed through
nsr/nsr_api.py with --base https://api.openai.com/v1. Only the route differs (Sidrune resells the same model id; that
the weights served are identical cannot be verified from either side). Output: nsr/out/api/nsr5_K4_gpt-6.1-sol_xhigh_64k.jsonl.
Spend: no balance endpoint, so --max_calls 21 is the cap (plus at most 3 transport re-runs per question, rule 2); cost is
reported from usage. Scoring: the 21 records replace the hit-limit records of the answer of record, the merged file is
parsed with the registered parser and with --repair --normalize, and the K = 4 gap is reported. Not used for any bar.
Prediction: most of the 21 finish within 64k (NSR3: 2 of 2 did) and are right, as the other home/ops questions are
(home .73, ops .61 registered at K = 4, with the 21 counted wrong); the K = 4 gap shrinks from +0.124 to about +0.03.

## Amendment 8 (2026-11-12, before any answer exists): route replication of the gpt arm on OpenAI direct
Why: amendment 7 found that the 21 K = 4 questions that ran to 32k on Sidrune finish in under 4.2k tokens on OpenAI
direct (same model id). Two of the three operational failures that separate the frontier arm from the pipeline (budget
exhaustion, glued answers) may therefore belong to the Sidrune route, not to the model. Test: re-run the whole gpt arm
on api.openai.com at the REGISTERED settings: gpt-6.1-sol, reasoning_effort xhigh, max_completion_tokens 32,000, same
prompts, streamed (nsr/nsr_api.py --base https://api.openai.com/v1). Order: K = 4 (161), K = 8 (69), K = 1 (the 230 NSR1
prompts, for Z2). Outputs nsr/out/api/nsr5_K{4,8}_gpt-6.1-sol_xhigh_direct.jsonl, nsr1_gpt-6.1-sol_xhigh_direct.jsonl.
Rules as before: transport failures re-run (frontier amendment 1, rules 1-2), hit_limit scored as no answer (rule 3);
parsers registered, normalised, and repaired (frontier amendment 2). Scorers unchanged.
Status: a replication on a second route. It does NOT replace the registered Sidrune results or their bar outcomes. Both
are reported side by side, and Z1-Z3 are recomputed on it as "direct route".
Cost cap: --max_calls equal to each set's size; about 12M input tokens, expected USD 30-40 at USD 2 / 10 per million
(worst case, every call at 32k output, about USD 170).
Predictions: (1) hit_limit at most 2% at every K (Sidrune: K = 4 13%, K = 8 3%); (2) glued "Answer:" lines 0, so the
registered and repaired parses agree within 0.01; (3) K = 4 set-exact (registered) at least 0.78, K = 8 at least 0.70;
(4) Z1 fails again (pipeline - frontier at K = 8 below +0.10). If (1) fails, the runaway tail is the model's, not the
route's, and amendment 7's route reading is wrong.
