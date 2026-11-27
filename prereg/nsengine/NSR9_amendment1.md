# NSR9 amendment 1 (2026-11-15): finish NSR9 with Claude Opus 5.5 on the user's endpoint
Written before any Claude call on an NSR9 item.
- The user stopped OpenAI spending mid-run. Arm E (gpt-6.1-sol) has 49/75 items (results/NSR9_partial.md); arm D
  (gpt-6.1-sol direct) was never run.
- The user then gave an Anthropic-compatible endpoint for Claude Opus 5.5 and asked for the comparison to be made with
  it.

## What changes
The two arms are re-run in full on all 75 NSR9 items with one model, `claude-opus-5-5`:
- **E_c (engine).** Everything as registered for E: engine v4, the same generic prompt (runs/gtranslate.py), K = 2
  formalisations, one repair round, gcert certification with coverage "missing only", solver limit 300 s. The
  translator is Claude Opus 5.5 at effort medium.
- **D_c (direct).** The same prompt and answer format as D. Claude Opus 5.5 at effort high, at most 32,000 output
  tokens (thinking included), adaptive thinking (the model's only mode).
- **P_c:** E_c's first formalisation alone.

This makes the comparison cleaner than the registered one: the same model, with and without the engine.

## Unchanged
- Items: data/nsr9_items.jsonl (75; B4 already PASS).
- Bars B1-B3, applied to E_c and D_c: B1 E_c wrong <= 1; B2 E_c correct >= 45; B3 E_c - D_c >= 11 (paired).
- The predictions as written, read with D_c in place of D.
- The gpt-6.1-sol partial E results stay reported as they are.

## Disclosed before the run
- **The endpoint is a relay.** A 15-token probe was billed 148 input tokens, so a system prompt of about 130 tokens is
  added upstream. Its content is unknown, and it applies to both arms equally.
- **No server-side refusal fallbacks.** A refusal, or a reply cut off at the token limit, counts as an abstention for
  D_c, as a missing answer did for D.
- **Spend** goes to runs/spend_ledger_claude.jsonl at list prices ($4 in / $20 out per MTok). The runner stops
  launching calls at $80.
- **Order:** E_c first, then D_c, both with level 8 first.
- **Pipeline check:** E_c and D_c were run once on two burned NSR8 items (out/claude_smoke_*.jsonl) before this
  amendment was committed.
