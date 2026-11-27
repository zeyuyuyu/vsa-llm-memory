# V-137 pre-registration — does a FRONTIER reader also collapse at volume, or only the small ones?

Written and committed BEFORE any API call.

## Why this is the decisive missing cell

V-135 (in flight) has the local 32B reader collapsing from 0.925 at 120 facts to **0.250 at 2000 facts**
with **zero truncation** — it reads all 21,246 tokens and cannot use them — while the sharded store holds
0.997. That extends the manuscript's existing volume result, which is measured on an 8B reader and leads by
37 points at 1024 facts.

But the manuscript's own sentence beside that result reads: *"Frontier readers handed the same 1024 facts
read them at 0.95–1.00 in the companion study, so the margin too is the small model's."* (V-130.) So the
existing evidence says the volume advantage **closes** against frontier readers at 1024 facts. Nobody has
measured 2000.

This is the first question a reviewer will ask, and it decides which sentence the paper may write:
- if frontier readers also collapse at 2000 facts, the volume claim generalises and is the paper's
  strongest result;
- if they hold, the claim is bounded at "readers up to 32B" and the manuscript's existing scope sentence
  must be extended rather than relaxed.

I do not know which, and the bars below are set without reference to which I would prefer.

## Setup

Identical instances, facts and ordering to V-135 (same generator, same seeds, depth `k = 2`), so the cells
are directly comparable to the local reader's.

- Fact counts: `K ∈ {120, 2000}` — the anchor and the decisive cell.
- `N = 60` instances per cell.
- Readers: `zai-org/glm-5.3` and `deepseek/deepseek-v4-pro` over CommonStack.
- Arm: `plain` only (no instruction about format or reasoning), scored by the same rule V-135 uses.
  V-135 showed `cot` tracks `plain` closely at volume (0.350 vs 0.250 at K=2000), so the extra arm buys
  little and doubles the spend.
- The store column is copied from V-135's artifact for the same instances; it involves no model and is not
  re-measured.

## Bars

**P0 — validity gate.**
- (a) At `K = 120`, each frontier reader scores ≥ 0.85. A reader that cannot do the easy cell is not a
  frontier reader on this task and no reading is taken from it.
- (b) Truncation / API-error rate ≤ 0.05 per cell; a cell above that is excluded value-blind.

**P1 — the question.** At `K = 2000`, averaged over the admitted frontier readers:
`store − plain` ≥ **+0.30**, with the 95% bootstrap CI over instances excluding zero.

**P2 — is it the same collapse?** Frontier `plain` at `K = 2000` ≤ **0.60**, i.e. the readers lose at least
a quarter of the accuracy they hold at `K = 120`.

**P3 — descriptive, no bar.** Per-model numbers, prompt tokens, and the comparison against the local 32B's
0.250.

## What I expect (recorded before the data)

**I expect P1 and P2 to FAIL** — that is, I expect frontier readers to hold up at 2000 facts where the 32B
did not. V-130 already found they read 1024 facts at 0.95–1.00, their context windows are far longer than
32k, and long-context retrieval is exactly what the last generation of frontier training has targeted. If
that is what happens, the honest outcome is that **V-135's headline is bounded to readers up to 32B**, and
the paper may not claim the volume advantage generally.

I am recording this expectation because V-135 currently looks like the best result of this whole line of
work, and I want it on the record that I expected the check to cut it down before I ran it.

## Deviations, recorded in advance

CommonStack exposes no balance meter, no `provider` field and no `reasoning` field, so three standing API
rules cannot be honoured (see the V-125/V-133 records). Substituted: a hard cap of **400 calls** and
**4,000,000 prompt tokens**, enforced in-process, since the gateway returns HTTP 429 permanently once the
key's own cap is hit and gives no warning. Per V-123, temp-0 API inference is not row-replicable, so this
supports arm-level accuracy with CIs over instances and **not** per-row claims.

## Scope

Synthetic chains. Nothing here transfers to 2WikiMultihopQA or MuSiQue and nothing reopens V-131's
withdrawal. V-137 bounds V-135; it does not replace it.
