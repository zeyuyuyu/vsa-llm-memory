# V-134 pre-registration — knowledge in weights against knowledge in an algebraic memory

Written and committed BEFORE any knowledge base is generated, any LoRA is trained, or any query is run.

## Why this comparison and not the one V-133 ran

V-133 asked whether chain depth widens the algebra's lead. Against a reader that is *given the facts and a
scratchpad*, it does not — the lead inverts, because the store pays a cleanup error at every hop while a
32B reader walks a 120-line table reliably. That closes a question the manuscript itself flags as open
("Whether the gain grows with hops is testable only where the comparator has real purchase").

The request now is to widen the gap against **fine-tuning**. V-133's task cannot answer that, and I want the
reason on the record before I build anything: in V-133 the facts sit in the context, so a fine-tune only has
to learn the *procedure* of walking a table, and it can read any unseen edge straight off the page. A
combination split there would be close to vacuous.

The comparison where a fine-tune and an algebraic memory genuinely differ is **where the knowledge lives**.
That is also the only comparison in which an external memory has a reason to exist: you build one precisely
so that facts need not be baked into weights or re-pasted into every prompt.

- `lora_weights` — the facts are trained into the model. At test time it is asked a *k*-hop question with
  **no facts in the context** and must compose internally.
- `store` — the same facts sit in an HRR memory and are composed by *k* unbind-and-cleanup hops.

Latent multi-hop composition over memorised facts is a documented weak spot of fine-tuning, so this
predicts a gap **with a mechanism** rather than hunting for a configuration that happens to flatter us.

## Setup

`M = 4` independent knowledge bases. Each: 40 pseudo-name entities, 8 relations, `B = 4` outgoing edges per
entity with distinct relations, `F = 160` facts. Deterministic, seeded, no bare `set` iteration,
`PYTHONHASHSEED=0` (standing rules).

Per KB, one LoRA is trained on the base model to memorise that KB: single-hop statements and single-hop
question/answer pairs in several surface forms. Training contains **single-hop material only** — no chains,
no multi-hop examples — so composition is what is being tested, not recalled.

Evaluation: `k ∈ {1, 2, 3, 4}` hop chains over the same KB, 80 queries per (KB, depth).

## Arms

| arm | knowledge from | facts in context | scratchpad |
| --- | --- | --- | --- |
| `lora_weights` | trained-in weights | no | **yes** |
| `store` | HRR memory | no | n/a (algebra) |
| `ctx_cot` | context | yes | yes |
| `base_nofacts` | nothing | no | yes |

`lora_weights` is given a scratchpad deliberately. V-133 showed what happens when the opponent is denied
one: it floors at two hops and stops being an opponent at all. A gap that only exists against a handicapped
comparator is not a gap worth reporting.

`store` runs at `d ∈ {2048, 4096, 8192}` × 3 codebook seeds; **primary `d = 4096`**.

Reader: `Qwen/Qwen2.5-32B-Instruct`, local, greedy, row-replicable.

## Bars

**P0 — validity gate. If any clause fails, no reading is taken.**
- (a) **Memorisation**: `lora_weights` at `k = 1` ≥ 0.80, averaged over the four KBs. A fine-tune that never
  learned the facts cannot be asked whether it can compose them.
- (b) **Not guessable**: `base_nofacts` ≤ 0.05 at every depth.
- (c) Scratchpad truncation ≤ 0.02 for every arm that uses one.

**P1 — the headline gap.** `store(k=2) − lora_weights(k=2)` ≥ **+0.30** at `d = 4096`, with the 95%
bootstrap CI (20,000 resamples, seed 20260915, resampled over queries clustered by KB) excluding zero.

**P2 — does depth amplify it.** With `gap(k) := store(k) − lora_weights(k)`:
`gap(4) − gap(2)` ≥ **+0.15**, CI excluding zero.

**P3 — scope, descriptive, no bar.** `ctx_cot` at every depth, reported beside the others.

## What I expect (recorded before the data, separately from the bars)

- **P0(a) is the technical risk of this run.** Memorising 160 facts into a 32B through LoRA may need more
  epochs, a higher rank, or more surface forms than my first attempt uses. **Any such tuning strengthens the
  opponent**, so it is a concession, not a thumb on the scale — the same logic under which V-131 let the
  fine-tune pick its configuration on the test set. If the gate still fails, the honest report is that I
  could not train a competent opponent and no gap may be quoted.
- **P1 HOLDS, and large.** Composing facts that exist only in weights, with no context to read, is a known
  failure mode.
- **P2 HOLDS.** Latent composition should decay faster in `k` than the store's per-hop cleanup.
- **P3 shows `ctx_cot` high — around or above 0.9 — so it will very likely BEAT the store**, exactly as
  V-133 found at every depth past two. I am recording that expectation now because it bounds what P1 and P2
  are allowed to mean: a held P1/P2 licenses **"an algebraic memory composes what a fine-tune of the same
  base model cannot"** and does **not** license "the algebra beats putting the evidence in the prompt".
  Those are different claims and only the first is being tested here.

## Scope of any reading

Synthetic knowledge bases, one base model, one adapter family. A held bar says something about
**where knowledge is stored**, not about natural text, and does not reopen V-131's withdrawal. The manuscript
already reports a held-out-combination gap on arenas B and C; this is a different and complementary axis and
must be reported as such, not merged with it.

## Protocol

Scorer `v134_score.py` is frozen and committed with this file before any data exist, and is run verbatim.
Arms are excluded value-blind on completeness. One tick is held before scoring. The ID V-134 is fresh.
