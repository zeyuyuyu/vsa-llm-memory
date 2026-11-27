# V-137 AMENDMENT — a harder ladder against the strongest readers reachable. Written before any API call.

V-137's first launch made **one** call and died on the old key's exhausted cost cap, so **no V-137 data
exists**. This amendment is therefore written before data, not after it.

## The request, and what I will and will not do with it

The user asks for evidence that the algebra beats frontier readers, and asks me to challenge them harder.

**I will not search configurations until one flatters the store.** That is the failure mode the whole
prereg discipline exists to prevent, and in this session I have already had to retract two conclusions
caused by an instrument choice of my own; a third, outcome-directed, would make the whole batch unusable.

What makes the question well posed is a mechanism, not a hope. A sharded store's readout is set by
facts **per shard**, which is held constant, so it is flat in total volume by construction. Every reader's
context is finite. Therefore some volume exists at which the store wins for any given reader. The real
question, and the only one worth registering, is:

> **Is that volume inside the window the reader can actually read, or only past it?**

A win bought by overflowing the context is not a win. V-131 already withdrew a 47-point separation that
turned out to be exactly that. So every bar below is taken **only on cells whose prompts fit, untruncated,
in the reader's own window**, and cells that overflow are reported separately as a window boundary and
excluded from the reading.

## What changes from the original V-137 prereg

- **Readers: the four strongest reachable**, not two — `zai-org/glm-5.3`, `moonshotai/kimi-k3`,
  `deepseek/deepseek-v4-pro`, `openai/gpt-5.4-2026-03-05`. V-130 measured the first three at 1024 facts
  with **zero** degradation, so they are the honest opponents.
- **A ladder instead of one cell**: `K ∈ {120, 2000, 4000, 8000}`, run in ascending order and **stopped
  early per reader** once that reader falls below 0.60, so budget is spent where the answer is.
- **A second axis at the decisive volume**: depth `k = 4` at `K = 2000`, since V-133 found depth alone
  does not separate the arms and volume alone does; whether they compound has not been measured.
- `n = 40` per cell (from 60) to pay for the wider ladder.
- `max_tokens = 3000`, because the smoke showed glm-5.3 spending its whole visible budget on reasoning
  tokens the gateway does not expose (the V-127 instrument rule).

## What does NOT change

The store column is still copied from the V-135 artifact for the same instances and is not re-measured.
The `plain` arm is still the comparator, scored by the identical rule. The instances, facts and ordering
are still V-135's.

## Bars (fixed here, before any data)

**P0 — validity gate, per reader.**
- (a) At `K = 120`, the reader scores ≥ 0.85, else it is not a competent reader on this task and no reading
  is taken from it.
- (b) Per cell, API error rate ≤ 0.05 and truncation rate 0; a cell failing either is excluded value-blind.

**P1 — the headline.** At the largest `K` that is **untruncated for that reader**, averaged over the
admitted readers: `store − plain` ≥ **+0.30** with the 95% bootstrap CI over instances excluding zero.

**P2 — is there a within-window cliff at all?** At least one admitted frontier reader scores ≤ **0.60** at
some untruncated `K ≤ 8000`.

**P3 — the store must hold.** Store ≥ **0.70** at that same `K`.

**P4 — descriptive, no bar.** Per-reader curves; prompt tokens and latency; the `k = 4, K = 2000` cell; and
the `K` at which each reader's window is exceeded.

## What I expect (recorded before the data, separately from the bars)

V-130's numbers are the reason I have been sceptical and they are strong: from 8 facts to 1024, glm-5.3 and
kimi-k3 lost **0.0000** and deepseek 0.025, against the 8B's 0.408. Two arguments cut the other way and are
the user's, not mine: this task uses pseudo-names with no semantic handle and three decoy relations per
chain node, which V-130's real Wikipedia entities did not; and 2000 is beyond anything V-130 tested, where
my local 32B fell 0.900 → 0.250 between 500 and 2000, a cliff rather than a slope.

Weighing those, **I expect P2 to hold at K = 4000 or 8000 and P1 to be marginal** — that is, I expect the
frontier cliff to exist but to sit two to four times further out than the 32B's, and I am no longer
confident enough to predict a clean failure as I did in the original prereg. If the readers hold at 8000
untruncated, the honest report is that the volume advantage is **not** available against frontier readers
at any practically relevant scale, and the paper may not claim it.

## Spend

The previous key died mid-experiment with a permanent 429 and no warning. Hard in-process caps: **1,200
calls** and **25,000,000 prompt tokens**, with the ascending ladder and per-reader early stop so the
expensive cells are only reached if the cheap ones say they are needed. Spend remains unobservable on this
gateway.
