# V-133 AMENDMENT — P1 is declared not testable, and why that costs me the result I wanted

Written after the P0(b) calibration and before the registered run. Labelled as an after-the-fact amendment,
per the V-132 precedent.

## What happened

The n=4 smoke test returned `direct` = 0.000 at every depth. Inspection of the raw generations showed the
reader emitting clean single-entity names — the format and the extractor were both fine. It was simply
wrong. So I ran the P0(b) anchor as a calibration over six fact loads (n=40 per cell, on a disjoint seed
block that never touches the registered instances):

| K facts | direct @ k=2 | store @ k=2 | direct @ k=4 | store @ k=4 |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.675 | 1.000 | 0.400 | 0.950 |
| 32 | 0.600 | 1.000 | 0.225 | 0.925 |
| 48 | 0.450 | 1.000 | 0.150 | 0.975 |
| 72 | 0.425 | 0.950 | 0.125 | 0.950 |
| 96 | 0.125 | 0.950 | 0.050 | 0.950 |
| 120 | 0.200 | 0.950 | 0.075 | 0.925 |

**No load satisfies the anchor.** `direct` never reaches 0.80 at two hops — not even at 24 facts, where the
entire knowledge base fits in a short paragraph. The store clears its half of the anchor everywhere.

## What that means, and what it does NOT mean

P0(b) exists to catch one specific failure: "a `direct` arm already broken at two hops would make any depth
trend a fact about my prompt, not about depth." The calibration settles which of those it is. It is not the
prompt — six loads, clean output format, verified extraction. **Qwen2.5-32B-Instruct cannot compose two
relation-typed lookups in a single forward pass without a scratchpad, at any load tried.**

That is a real measurement, and it is worth reporting. But it disqualifies `direct` as an *opponent* for the
depth question. An arm pinned near the floor from k=2 onward has no dynamic range, so "does the gap widen
with depth" cannot be asked of it. Worse, the arithmetic runs backwards: with `direct` low everywhere and
the store falling from 0.960 (k=2) to 0.727 (k=6), `gap(6) − gap(2)` is *negative* for reasons that have
nothing to do with the hypothesis being tested.

## The amendment

**P1 is declared VOID — not testable on this task — rather than scored.** Scoring it would record a failure
caused by the opponent's floor rather than by depth, which is not what P1 was written to measure.

**Everything else is carried over byte-identical.** P0(a), P0(c), P2, P3 and P4 are unchanged in wording and
in threshold. P0(b)'s store clause (≥0.95 at k=2, d=4096) still applies and passes at 0.960. P0(b)'s direct
clause is retained as a *reported diagnostic*, not as a gate on P2/P3.

**The `direct` arm is still run at full N and reported.** I am not dropping an arm after seeing it fail;
its numbers go in the artifact and the ledger as a descriptive result about one-shot composition.

**K stays at 120**, chosen by the design criterion written in the prereg rather than by any outcome: the
constant-K design only keeps depth unconfounded while background distractors dominate, and at K=120 the
non-chain facts run 112 (k=2) to 96 (k=6), a 14% swing, against a factor of two at K=40.

## Why this amendment cannot be outcome-motivated in my favour

It removes the one reading I preregistered an expectation of **holding**, and leaves the run resting
entirely on P2 — the reading I preregistered an expectation of **failing**. The amendment therefore moves
the experiment away from the result the hypothesis needs, not toward it. If P2 now fails as I predicted, the
honest conclusion is that chain depth does not rescue the algebra's accuracy case, and the user's proposal
is refuted rather than confirmed.

There is also a second cost I should state plainly: the store arm is already measured, and it falls from
0.960 at two hops to 0.727 at six (0.969 → 0.758 at d=8192). **The algebra is not depth-free.** If the
chain-of-thought arm holds near ceiling, the store does not merely fail to pull ahead with depth — it falls
behind. That possibility is visible to me now, before the reader run, and I am recording it here rather than
discovering it later.

## Planned descriptive follow-up, declared now so it cannot be dressed up later

If P2 fails because chain-of-thought sits at ceiling, the obvious next question is what load or depth breaks
it. Any such sweep is **descriptive only**, carries no bar, is run after the registered scoring, and may not
be quoted as a registered reading.
