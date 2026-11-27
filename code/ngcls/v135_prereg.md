# V-135 pre-registration — evidence VOLUME, the axis that actually stresses a reader

Written and committed BEFORE any instance is generated or any model is called.

## Why this axis, after two that did not separate the arms

V-133 varied chain depth at a fixed 120 facts and found the algebra's lead over a prompted reader small and
shrinking: +0.082 at two hops, +0.056 at three, −0.004 at four against a plain prompt. V-134's pilot varied
where the knowledge lives and found a LoRA that memorised a 160-fact base composes two to four hops about as
well as an HRR store holding the same facts.

The common cause is visible in the design rather than in the algebra: **120 facts is a load at which a 32B
reader is comfortable.** Scanning a 120-line table and walking a chain is something it does well, so depth
never stressed it, and the comparison measured two arms that were both near ceiling.

The axis that is known to stress a reader is the amount of evidence, and two already-banked results predict
a widening gap from independent directions:

- **Prompting degrades with volume.** V-100 measured a reader falling from 0.94 at 8 facts to 0.59 at 512.
  V-128 named the mechanism: past a certain retrieval breadth the model has the evidence in context
  (recall 0.89–0.92) and still loses 15–20 points, because it cannot *use* buried evidence.
- **A sharded store is approximately flat in total volume.** V-102 established that readout is set by facts
  per shard, not by the total, and that cross-shard selection costs about a point.

Two independently published facts that jointly predict the gap opening. That is a mechanism, not a search
over configurations, and it is why this experiment is worth running rather than another sweep.

## Setup

The V-133 chain task, **depth held fixed**, fact count swept. Deterministic, seeded, no bare `set`
iteration, `PYTHONHASHSEED=0`.

- Depths: `k = 2` (primary) and `k = 4` (secondary).
- Fact counts: `K ∈ {120, 250, 500, 1000, 2000, 4000}`.
- `N = 120` instances per cell.
- Reader: `Qwen/Qwen2.5-32B-Instruct`, local, greedy, row-replicable, 32k context.

Every arm receives exactly the same facts in the same order, as in V-133.

## Arms

| arm | what it is |
| --- | --- |
| `store` | Sharded HRR, `S` fixed by the manuscript's ~64-facts-per-shard rule, so shard count grows with `K` and facts-per-shard stays constant. `d = 4096` primary, 3 codebook seeds. No model call. |
| `plain` | The same facts in the prompt, **no instruction about format or reasoning**. V-133 established this is what the reader does naturally: 100% of its outputs reasoned, and it scored 0.907 at two hops against 0.920 for an explicitly instructed arm. This is the honest prompting baseline and it is the primary comparator. |
| `cot` | Same facts, explicit step-by-step instruction. Reported so the instruction's own worth stays visible. |

Scoring uses the **same** answer rule for every arm: the `Answer:` line when present, otherwise the last
token that is an entity of that instance's own fact set. V-133 scored a plain-prompt arm at 0.0000 because
the strict rule could not see "So, we end at Garion."; that parser failure is fixed here and the rule is
applied identically everywhere.

**Truncation is a measured quantity, not an accident.** At roughly 10 tokens per fact, 4000 facts exceed the
reader's 32k window. Prompt tokens and truncation rate are recorded at every cell, and any cell whose
truncation rate exceeds 0.02 is reported as **window-limited** and excluded from the P1/P2 readings, which
are taken only over cells the reader can actually read. A win purchased by truncating the opponent is not a
win — V-131 already had to withdraw a 47-point separation that turned out to be exactly that.

## Bars

**P0 — validity gate.**
- (a) At `K = 120`, `k = 2`: `store` within 0.03 of V-133's banked 0.9889 and `plain` within 0.05 of its
  banked 0.9067. The harness must reproduce the run it extends.
- (b) Task well-formedness: 100% of instances re-derive their answer under the independent path-follower.
- (c) Every cell entering P1/P2 has truncation ≤ 0.02.

**P1 — the headline.** At the largest untruncated `K`, `k = 2`: `store − plain` ≥ **+0.30**, with the 95%
paired bootstrap CI (20,000 resamples, seed 20260915, over instances) excluding zero.

**P2 — it must be volume that does it.** `gap(K_max_untruncated) − gap(120)` ≥ **+0.25**, CI excluding zero.

**P3 — the store must hold.** `store` at the largest untruncated `K`, `k = 2`, `d = 4096` ≥ **0.70**.

**P4 — descriptive, no bar.** Prompt tokens and wall-clock per query for each arm at each `K`; the `k = 4`
sweep; the point where the window is exceeded.

## What I expect (recorded before the data, separately from the bars)

- P1 and P2 **HOLD for this reader**. Volume is the axis where a fixed-capacity sharded memory should beat a
  linearly growing context, and both halves of that are already measured separately.
- P3 holds, because sharding keeps facts-per-shard constant by construction; if it fails, the sharding
  remedy does not survive the scale at which it matters and that is the more important result.
- **The scope limit I expect to have to write:** V-130 found frontier readers tie or beat the store when
  given the same evidence, and frontier context windows are far longer than 32k. So a held P1 licenses
  **"against a 32B reader at volumes above its comfortable range"** and not "against readers in general".
  I am recording this now because it is the sentence that will bound the headline, and I do not want it to
  look like a concession invented after the fact.

## Scope

Synthetic chains, one local reader. Nothing here transfers to 2WikiMultihopQA or MuSiQue, and nothing
reopens V-131's withdrawal. The three axes now measured -- depth (V-133, no separation), knowledge location
(V-134), volume (here) -- are reported together, not selectively.

## Protocol

Scorer `v135_score.py` is frozen and committed with this file before any data exist, and run verbatim.
Value-blind completeness exclusion. One tick held before scoring. V-135 is a fresh ID.
