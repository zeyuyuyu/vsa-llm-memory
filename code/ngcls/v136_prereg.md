# V-136 pre-registration — a POWERED replication of V-134. Bars carried over verbatim; only N(KB) changes.

Written and committed BEFORE any new knowledge base is generated.

## Why

V-134 closed with both bars failing, but its own ledger entry records a defect I put there: the cluster
bootstrap treats the knowledge base as the independent unit and there were only **four** of them, while the
between-KB spread at k=2 ran +.288 to −.159 — wider than the effect being measured. P1's CI came out ±.16
wide and could not have excluded zero at any plausible effect size. That is an underpowered design, not a
finding.

One result inside V-134 is worth resolving properly. The gap against knowledge-in-weights **grows
monotonically with depth** (−.115, +.075, +.128, +.172 at k=1..4) and P2's CI [+.0021, +.1917] excluded zero
while its point estimate +.0969 sat under the +.15 bar. With four clusters that is the most one can say.
With sixteen it becomes a number.

## What changes, and what does not

**Changes: `n_kb` 4 → 16.** Nothing else. Same generator, same 160-fact knowledge bases (new KB ids 100–115,
disjoint from V-134's 0–3), same LoRA recipe (r=32, 12 epochs, lr 1e-4, all attention and MLP projections),
same single-hop-only training material, same sharded store at the published ~64-facts-per-shard rule, same
four arms, same 80 queries per (KB, depth), same scratchpad for every language-model arm.

**Does not change: every bar, verbatim from `v134_prereg.md`.**
- P0 (a) `lora_weights` at k=1 ≥ .80 averaged over KBs; (b) `base_nofacts` ≤ .05 at every depth;
  (c) scratchpad truncation ≤ .02.
- **P1** `store(k=2) − lora_weights(k=2)` ≥ **+.30**, 95% cluster bootstrap CI excluding zero.
- **P2** `gap(4) − gap(2)` ≥ **+.15**, CI excluding zero.
- P3 descriptive `ctx_cot`.

The frozen scorer `v134_score.py` is reused **unmodified** — it reads `n_kb` from the artifact, so no edit is
needed and none is made. This is deliberate: reusing the identical scorer is the strongest available
guarantee that no bar moved.

I am **not** setting a new bar at the observed +.17. Choosing a threshold after seeing the effect is fitting
the judgement to the data, and it is exactly what the prereg discipline exists to prevent. If the honest
answer is "the growth is real but smaller than we registered as interesting", that is the answer.

## What I expect (recorded before the data)

- P0 passes, as it did at n_kb=4.
- **P1 FAILS again.** The k=2 point estimate was +.075 against a +.30 bar; power does not move a point
  estimate. I expect the CI to tighten around a small positive value and the bar to fail cleanly.
- **P2 is the open question.** Point estimate +.0969 against a +.15 bar. I expect it to **FAIL on magnitude
  while its CI excludes zero more decisively** — i.e. the growth is real and smaller than registered. I
  record this rather than hoping, because a narrow CI around +.10 is a more useful result than the wide one
  V-134 produced, whichever side of .15 it lands.
- P3 continues to show `ctx_cot` above the store at every depth, keeping the binding scope sentence in force.

## Scope

Unchanged from V-134: synthetic knowledge bases, one base model, one adapter family. A held or failed bar
speaks to **where knowledge is stored** and licenses nothing about natural text, and does not reopen V-131's
withdrawal.

## Protocol

No new scorer is written. `v134_score.py` is run verbatim on the V-136 artifact. Value-blind completeness
exclusion applies. V-136 is a fresh ID; V-134's artifact and verdict stand as recorded and are not revised
by this run — the two are reported side by side.
