# V-161 PREREG — composition capability GIVEN CORRECT EXTRACTION: VSA memory vs LMs (small local and frontier)

Registered 2026-10-15, before any registered-seed LM generation. Runners `b4.py` (local models and model-free arms) and
`or4.py` (OpenRouter, with `or4_config.json`). Frozen scorer `v161_score.py`. Independent second scorer
`v161_indep.py`. Code reviewed independently before launch: `review_D4.md`, 1 BREAKS + 2 MATERIAL, all fixed below,
then re-checked.

## Premise and question

The user's premise: extraction is assumed correct, and what is measured is the capability after it. The task is b3's
bijection pointer chase. Its generator was cleared by three reviews; at this scale, 13 shortcut heuristics sit at
chance (review D4). The text the model reads IS the correct fact list. The question: given the same correct facts,
does an exact VSA composer beat an LM composing in context, for small local LMs and for current frontier LMs, and how
does the gap move with hop count H and load N?

## Arms

- **Local LMs** (from the correct facts in context):
  - `direct`: "Answer with only the final symbol", 8 tokens;
  - `cot`: b3's step-by-step prompt, budget 256+96H;
  - `cot_alt`: a second wording.
- **API LMs** (`or4_config.json`; one pinned host per model, no fallbacks — review D4-3; reasoning tokens logged
  on every call — review D4-2):

  | model | host | direct / cot hidden reasoning | think |
  |---|---|---|---|
  | openai/gpt-6.1-sol | OpenAI | cannot be disabled (HTTP 400); adaptive, probe 18 / 0 tokens | effort high |
  | anthropic/claude-opus-5.5 | Anthropic | cannot be disabled; adaptive, probe 25 / 0 tokens | effort high |
  | google/gemini-4-argon | Google | effort minimal → 0 tokens | effort high |
  | deepseek/deepseek-v4-pro-0813 | Together, quantisation not disclosed | enabled:false → 0 tokens | effort high |

  - **Hosts and sampling (review D4 recheck N7, N8, N9).**
    - The first-party DeepSeek host was removed by this account's "paid model training" guardrail (HTTP 404), not
      for lack of the snapshot.
    - The pinned GPT and Claude endpoints do not support `temperature`, so their outputs are single samples at the
      provider's default sampling. Gemini and DeepSeek receive temperature 0.
    - Pinning by provider name can still span that provider's endpoint variants. The serving host is logged on every
      record.

  - `direct` uses b4.msg_direct (max_tokens 2048). `cot` uses b4.msg_cot (32k). `think` uses b4.msg_cot with
    reasoning effort high (64k): each model's strongest setting, the strongest baseline.
  - For GPT and Claude, "direct" therefore means "answer only, after the model's own minimal adaptive reasoning".
    The tokens are reported, and it is never described as reasoning-free.
- **One shared answer parser** (`b4.parse_answer`: the last "Answer" or "Final answer", any case, optional bold,
  backticks or parentheses; review D4 N5) for every step-by-step arm, local and API. Unparsed answers are counted
  and reported for local and API runs. `v161_indep.py` re-parses with a differently built parser.
- **Model-free arms** (given the correct bindings and start / table order):
  - `gold_dict`: exact table, 100% by construction;
  - **`gold_vsa_shard` = one HRR vector per table at D = 16384, the method;**
  - `gold_vsa1@D` (one vector, Gaussian codes) and `gold_vsa1_unitary@D` (one vector, unitary codes), for
    D ∈ {4096, 8192, 16384}, as capacity references;
  - `gold_vsa_shard@4096/@8192` reported beside the method arm.

### Why D = 16384 (review D4-1), stated plainly (review D4 recheck N6)

At D = 4096 (b3/b_pilot's width) the shard is not exact at this scale: 122/150 at N256_H1 and 39/150 at N256_H8. This
is superposition crosstalk with 256 bindings per vector, as theory predicts.
- **This was seen on the registered seeds before D was chosen.** I computed the D = 4096 arms myself. Review D4
  computed them too, and also reported 150/150 for the shard at D = 16384 on the registered N256_H8 instances.
- **The choice therefore cannot claim to be blind.** What makes it acceptable: D enters only the model-free arms, and
  C1 is a deterministic gate on them, not a hypothesis about any LM.
- The rule that motivates 16384 needs two caveats:
  - V-157 measured exact single-hop retrieval at 128 bindings per vector for D = 4096 up to H4; it was 144/150 at
    N128_H8.
  - V-151's own fitted exponent (k* ∝ d^0.85, consistent with the classical analysis of Frady et al. 2018 but
    fitted by us) gives D ≥ 4096·(256/128)^(1/0.85) ≈ 9,258 for 256 per vector.
- The next power of two is 16384.
- **The full D-sweep (4096, 8192, 16384) is reported for every VSA arm.**

## Instances

N ∈ {16, 64, 128, 256} × H ∈ {1, 2, 4, 8}; seeds 0–2 on a fresh stream (`9000011*seed + 1009*N + H + 31`); 50 per cell
per seed, so n = 150 per cell. The largest prompt is 16.7k Qwen tokens.

- **Local (A800):**
  - Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct and Granite-3.3-8B-Instruct at batch size 8;
  - Qwen2.5-32B-Instruct at batch size 2, so it does not run out of memory at N256_H8 (review D4 minor).
- **API:** the four models above on N ∈ {16, 64, 256} × H ∈ {1, 4, 8}, which contains both bar cells.

## Bars (counts of 150)

- **G0:** regex_ceiling = 150 everywhere, and freq_heuristic ≤ 15 at N ≥ 64. Fails → every comparison VOID
  (enforced in the scorer, review D4 N2).
- **C1 (GATE):** gold_vsa_shard (D = 16384) ≥ 147 in every cell. If C1 fails, no comparison against the VSA is
  scored.
- **C3[model, arm]:** at N256_H8, count(gold_vsa_shard) − count(arm) ≥ +30, and the paired bootstrap CI > 0.
  Scored for cot (primary), and for think, cot_alt and direct.
- **C4[model, arm]:** arm(N16_H1) − arm(N256_H8) ≥ 30.
- **Missing API calls:** a bar cell needs ≥ 98% completed calls, or that model and arm is VOID. Every bar is
  computed twice: with never-completed calls excluded, and with them counted wrong. If the two disagree, the result
  is UNRESOLVED, not either verdict. A truncated response counts as wrong.

## Seen before registration

- Review D4's own seed-99 probes (HTTP metadata, one refused request) and its D = 16384 check on registered N256_H8
  instances (150/150).

- The model-free arms on the registered seeds:
  - at D = 4096, by me (myverify run) and by review D4;
  - the D-sweep on the box CPU (b_logs/v161_gold_*), started before this commit.
- A 1-instance smoke per API model on UNREGISTERED seed 99, cells N16/N256 × H1/H8: **all four answered N256_H8
  correctly with 118–254 output tokens.**
- Reasoning probes on seed 99 (N64_H4): DeepSeek with reasoning disabled answered the direct arm wrong and the cot
  arm right.
- V-157 seed 0 and the light cells of seeds 1–2 (Qwen2.5-7B, same task family).
- No LM output on V-161's registered seeds has been generated.

## Predictions

- **G0 PASS, C1 HOLDS** (algebra, from the sweep).
- **Local 7B / 14B / Granite:** C3 HOLDS and C4 HOLDS for cot.
- **32B:** uncertain.
- **Frontier API models: C3 FAILS for cot and for think.** I expect them near 100% at 2,048 facts and 8 hops.
- **Frontier direct:** likely degrades at N256_H8, more for the models whose reasoning is truly off (Gemini,
  DeepSeek).

## How it will be read

- "Given correct facts, an exact VSA composer beats in-context composition" is claimed only for the named models
  and arms where C3 holds.
- Where frontier models match the VSA, the paper says so. The VSA's remaining role there (fixed width, no
  16k-token prompt per question) is a cost claim, which needs its own measurement and is not claimed here.
- Every VSA statement is "as exact as a table", never "more accurate than a table".
