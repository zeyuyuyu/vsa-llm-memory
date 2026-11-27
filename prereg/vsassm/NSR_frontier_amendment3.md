# Frontier amendment 3 (2026-11-12, before any answer exists): NSR3 gpt arm on the OpenAI direct route

Why: NSR5 amendment 8 re-ran the whole NSR5 gpt arm (K = 1, 4, 8) on api.openai.com at the registered settings. The
direct route had 0 budget exhaustions and 0 glued answers in 460 calls, and it scored 0.952 / 0.925 / 0.855 against
Sidrune's 0.922 / 0.677 / 0.739 (registered parse). For the same prompt, Sidrune's reported prompt_tokens differ from OpenAI's
by a factor of 0.72-2.71 (median 1.13). One model's tokenizer cannot give that spread on identical text, so the Sidrune
"gpt-6.1-sol" is either a different model or reports usage that is not the model's. Every gpt-arm result so far
(NSR1, NSR3, NSR5) came through Sidrune. NSR1 is covered by the K = 1 direct run (the same 230 prompts). NSR3 is not.

Run: the 460 NSR3 temporal prompts (nsr/out/held/nsr3_llm_problems_nsr2.jsonl), gpt-6.1-sol, reasoning_effort xhigh,
max_completion_tokens 32,000, streamed through nsr/nsr_api.py --base https://api.openai.com/v1. Output
nsr/out/api/nsr3_gpt-6.1-sol_xhigh_direct.jsonl. The same parsers (registered, normalised, repaired) and the same scorer
and paired comparison against the history engine (nsr/nsr3_compare.py).
Status: route replication, reported beside the registered Sidrune result. It does not replace the T2-F outcome as
registered.
Cost cap: --max_calls 460, expected about USD 10.
Predictions: hit_limit at most 2%; the parses agree within 0.01; set-exact about 0.90 (Sidrune repaired + normalised 0.867);
history engine (0.874) - direct frontier below 0, so T2-F (>= +0.30) fails again.
