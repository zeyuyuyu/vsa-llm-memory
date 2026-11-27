# V-145 pre-registration — the cost frontier, the one axis three experiments did not kill

Written and committed BEFORE any timing is taken.

## Why this and not another accuracy experiment

V-133 (depth), V-134 (knowledge location) and V-135/V-137 (evidence volume) each set bars in advance and
each came back negative on accuracy: against a competent reader holding the same evidence in context, the
algebra has no usable advantage. V-137 is the sharpest — glm-5.3 and kimi-k3 read 8000 facts and 84k tokens
at 0.975 where the store reads 1.000, a gap of 2.5 points, flat across a 67-fold range of volume.

What survived every one of those runs is cost. The store answers with **no model call at all**. That is
structural, not empirical, and it is the same for every reader. The manuscript currently reports it only in
a source comment and one clause; nothing in the Results section claims it.

## A correction I am making before the design, not after the data

I have described the sharded store as "flat in total volume". That is true of its **readout accuracy** and
false of its **cost**: a query searches every shard, and the shard count grows linearly with the fact count
under the ~64-facts-per-shard rule. So store cost is linear in K, as prompting's prefill is. The honest
claim is not "flat against growing" but **same order, very different constant, and no context ceiling**.
Registering that distinction here so the bars test the real thing.

## What is measured

Per query, on the V-135 instances (same generator, same seeds, depth 2), at
`K ∈ {120, 500, 1000, 2000, 4000, 8000}`, `N = 60`:

- **store**: wall-clock for the *whole* route --- building the store from the K facts **and** the two
  unbind-and-cleanup hops. Build time is included because a comparison that hides it is not a comparison.
- **local reader** (`Qwen2.5-32B-Instruct`, the V-135 arm): wall-clock, prompt tokens, completion tokens.
- **frontier reader** (`kimi-k3`, the V-137 arm that held at 8000 facts): prompt and completion tokens.

**Tokens are the primary cost unit, not wall-clock.** Tokens are deterministic and billable; API wall-clock
carries network and queueing that say nothing about the method. Local wall-clock is reported on fixed
hardware and is the only timing used in a bar.

**Two regimes, both reported.** *Per-query*: the store is built for every question. *Amortised*: built once
and queried many times. The store wins the amortised regime trivially, so **the bars are taken on the
per-query regime only** and the amortised numbers are descriptive.

## Bars

**P0 — validity gate.** At `K = 120` and `K = 2000`, the store's accuracy reproduces V-135's banked column
within 0.03. Same instances, same code; a miss means the harness moved and nothing is read.

**P1 — the cost claim, per-query regime.** At the largest `K` the local reader can read untruncated, the
store's total wall-clock per query (build + two hops) is at least **10x** lower than the local reader's,
*and* the store's accuracy is within **0.05** of that reader's. Cheaper while losing is not a result.

**P2 — does the advantage grow with volume.** cost-ratio at `K_max` divided by cost-ratio at `K = 120`
is at least **3**.

**P3 — the store's own cost must stay linear, not worse.** Fitting `log(store cost)` against `log(K)` over
the six volumes, the exponent is at most **1.3**. If sharding makes the store super-linear, the cost story
degrades at exactly the scale where it is being claimed.

**P4 — descriptive, no bar.** Frontier token counts and a dollar estimate at published rates; the amortised
regime; the volume at which the local reader's context window ends while the store continues.

## What I expect (recorded before the data, separately from the bars)

- P0 passes.
- **P1 holds by a wide margin.** V-135's incidental timings already show 27.5 s against 0.30 s at K=2000,
  a factor of 92. I expect the bar to be cleared by an order of magnitude more than it asks.
- **P2 holds.** The reader pays prefill over a growing context with attention that is not linear in it; the
  store pays a linear shard scan.
- **P3 is the one I am unsure of**, and it is the one that could embarrass the claim: cleanup over more
  candidates as shards multiply may push the exponent above 1. I would rather find that here than have a
  reviewer find it.

## Scope

Synthetic chains, one local reader on fixed hardware, one frontier reader for tokens only. A held bar
licenses a claim about **cost at matched accuracy on this task**, not about accuracy, and not about
deployments where the evidence must first be extracted --- which the manuscript already reports as the
route's binding limit (recall 0.34--0.44, end-to-end 0.018--0.031, no better at 32B than at 7B).

## Protocol

Scorer `v138_score.py` frozen and committed with this file before any timing exists, run verbatim.
Value-blind completeness. `PYTHONHASHSEED=0`. V-145 is a fresh ID.
