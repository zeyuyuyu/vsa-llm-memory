# V-143 prereg — does a fine-tune survive pool growth, on a reader whose window holds the pool?

Registered 2026-09-17, before any V-143 number exists. Runner `v143.py` committed in the same commit.

## Why this run exists

The manuscript's Results now carry the evidence-scale sweep (V-106): given the same pooled gold triples from
8 to 1024 facts, few-shot prompting on Qwen3-8B decays 0.950 -> 0.542 while a store at d=2048 sharded at 64
facts per shard holds 0.936 -> 0.908, a lead of 37 points at 1024 answered in 0.06 s against 11,612 tokens
and 0.83 s. The comparator there is prompting. The obvious referee question, and the user's question, is what
a **fine-tune** does on the same axis.

V-140/V-141 tried to ask this on phi-2 and could not: phi-2's window is 2048 tokens, a 300-fact pool is
~3,352 tokens, and the runner truncated the prompt from the right, which removes the question at the end of
the prompt. The N=300 run of 2026-09-17 therefore scored EM 0.0000 with training loss stuck at 4.3 for a
harness reason, not a task reason. **That run is VOID** (see ledger). `v140.py` now aborts instead of
truncating. V-143 replaces it on a reader whose window actually holds the pool.

## Design

- Reader: `Qwen/Qwen2.5-32B-Instruct`, bf16, 32k window, sharded over two free GPUs. At N=1024 the prompt is
  ~11k tokens, so **nothing is truncated at any pool size**; the runner aborts if it would be.
- Task: the 2WikiMultihopQA compositional rows and the `ft_goldev` pooling of V-138/V-140 — each row's gold
  evidence triples pooled with N distractor triples drawn from other rows, shuffled. Train and eval rows are
  disjoint (asserted); the scored slice is the covered slice, as in V-138/V-140.
- Pools: N in {0, 64, 256, 1024}. The training budget is held CONSTANT across N (n_train=150, 3 epochs,
  rank 16, lr 1e-4, bs 1) so that a decay cannot be read as "fewer gradient steps at large N".
- The fine-tune is trained **in the distractor regime it is evaluated in** — it sees pooled prompts during
  training. This is in the opponent's favour and is deliberate.
- Two arms on the SAME reader, identical input, differing only in whether LoRA was trained:
  `prompt` (the instruct model, zero-shot on that same prompt) and `lora`.
  This is NOT a replication of V-106's few-shot prompt route on Qwen3-8B; it is a matched control on this
  reader, and the ledger will not quote it as a V-106 number.
- The store is not re-run. Its V-106 values (0.936 at 64 facts -> 0.908 at 1024, a drop of 2.8 points) are
  quoted as banked.
- Seeds: 42 first at all four pools (shape probe). Seeds 123 and 456 are added at N=256 and N=1024 if and
  only if the primary bar lands in the ambiguous band.
- `PYTHONHASHSEED=0`. The sha256 of the imported `v140.py` helpers is recorded in the output JSON.

## Bars, fixed now

Let A(N) be the LoRA arm's covered-EM at pool N, on seed 42 (or the seed mean where three were run).

- **EVIDENCE that a fine-tune cannot do what the store does on this axis:** A(1024) <= A(0) − 0.30.
- **REFUTED:** A(1024) >= A(0) − 0.10.
- **AMBIGUOUS** in between -> run seeds 123/456 at N=256 and N=1024 and score on the three-seed mean.

## The negative reading, written before the numbers

If the fine-tune holds within 10 points out to a 1024-fact pool, then on this axis fine-tuning is as robust
to pool growth as the store, the hypothesis "a low-rank update cannot locate evidence at scale" is REFUTED
for pooled gold facts on a 32B reader, and the store's remaining advantage here is **cost** — 0.06 s and no
training against an 11k-token prompt per query — not capability. The ledger will say exactly that, and the
manuscript will not gain a capability sentence from this run.

If instead both arms decay together, the finding is about the reader and the pool, not about fine-tuning
specifically, and must be reported that way.

## What this run cannot show

It cannot speak to raw passages (V-138 already found the fine-tune at 0.2722 there, one seed below the
untrained baseline), and it cannot speak to held-out symbol combinations (arena C already has that). It
answers one question: pool size.

## Amendment, 2026-09-17, before the main run and before any banked number

The smoke run put the zero-shot prompt arm at 0.000 at N=0. That is a format artifact, not a capability
reading: given two gold facts the instruct reader restates the question as a sentence ("The mother of the
director of the film ... is Ma") and exact match cannot score it, and 16 new tokens cut it off mid-answer.

Two changes, applied **identically to both arms** so the comparison stays matched:

1. A one-line format instruction is prepended to every prompt: "Answer the question from the facts. Reply
   with the answer only -- a name, a place or a date -- and nothing else."
2. `max_new_tokens` 16 -> 32.

**Disclosure:** the probe that established this was run on the first 20 rows of the eval split, so I have
seen eval predictions before the main run. It fixed an output format and touched no bar. Its value (19/20)
is recorded here rather than banked, and it is not a V-143 result. The bars in this prereg are unchanged.

## AMENDMENT 3 (2026-09-17), registered while N=256 was still training, before N=1024 or any larger pool existed

**Extension, at the submitting author's direction ("N 需要逐渐调大，总有模型撑不住的时候").** Pools
1536 and 2048 are added to the sweep, run with the identical reader, training budget, seed and scorer.

**The ceiling is the window, and it is stated before the run.** The runner's own printed lengths give
14.2 tokens per pooled fact (81 at N=0, 984 at N=64, 3,711 at N=256). Against Qwen2.5-32B-Instruct's
32,768-token window:

| N | prompt tokens | status |
|---|---|---|
| 1024 | ~14,600 | inside |
| 1536 | ~21,900 | inside |
| 2048 | ~29,100 | inside, the last usable pool |
| 2200 | ~31,300 | too close to the limit to trust |
| 4096 | ~58,200 | exceeds the window; NOT runnable on this reader |

**A failure at a pool whose prompt does not fit is not evidence** — it is the artifact that voided V-141 and
the phi-2 K=64 arm of V-142 on this same day. `--max_ctx` is set to 32000 and the runner aborts rather than
truncating, so such a cell cannot be scored by accident. Pushing past N=2048 requires a longer-context
reader and is a separate registration, not an extension of this one.

**The primary bar is unchanged** and is still read at N=1024 against N=0, so this extension cannot move the
registered verdict. Pools 1536 and 2048 are reported as an additional descriptive curve; if the LoRA arm
does break between 1024 and 2048 while the prompt still fits, that is stated as a new finding at a stated
pool size, with its own comparison to the store's banked curve, and NOT read back into the primary bar.

**What is already known about this axis, recorded here so the extension is not read as new ground.**
V-130 measured frontier readers losing 0.0000 from 8 to 1024 facts; V-137 extended that eight-fold, to 8000
facts and 84k untruncated tokens, where kimi-k3 and glm-5.3 both held at 0.975. The local 32B does collapse
on the harder pseudo-name task of V-135 (0.925 to 0.250 between 120 and 2000 facts, zero truncation). So
"the reader eventually cannot hold it" is already established as a property of readers up to 32B, not of
volume. The present task uses real 2WikiMultihopQA entities, which a pretrained reader has priors for, and
is therefore the easier of the two; that is why its prompting arm is still at 0.940 at N=256. The sharper
experiment — the fine-tune arm on the pseudo-name task at the volumes where this same 32B already collapses
under prompting — is registered separately as V-144.

## AMENDMENT 4 (2026-09-17) — the distractor pool was silently capped; pools above 256 are VOID and re-run

**Found before the N=1024 cell finished, and before the 1536/2048 extension started.** The N=1024 prompt
printed 5,770 tokens where 14.2 tokens per fact predicts ~14,600. Cause: `v140.body_for`, which this runner
imported, samples `min(n_dist, len(pool))` **rows** from the arm's own row list and flattens their evidence.
Every 2WikiMultihopQA row carries exactly two evidence triples, so:

| requested N | eval delivered (200 rows) | train delivered (150 rows) |
|---:|---:|---:|
| 64 | 64 | 64 |
| 256 | 256 | 256 |
| 1024 | **400** | **300** |
| 1536 | **400** | **300** |
| 2048 | **400** | **300** |

Three consequences, all of which would have produced a confidently wrong statement:

1. The cell labelled N=1024 carried ~400 distractors, not 1024 — mislabelled by a factor of 2.5.
2. Training saw 300 and evaluation saw 400, so the arm was not evaluated in the regime it was trained in,
   contradicting this prereg's own design clause.
3. The registered extension to 1536 and 2048 would have produced cells **identical** to 1024. Three
   coincident points would have been reported as a sweep showing the fine-tune holding out to 2048 facts,
   when it never saw more than 400. That reads against our own hypothesis, but it is wrong either way.

**Status of the data.** N=0, 64 and 256 delivered exactly the requested counts on both sides and are sound
as measurements. The N=1024 cell was killed mid-training and nothing from it is banked; the log is preserved
as `v143_logs/run_capped.log`.

**Fix, committed before the re-run.** The runner no longer calls `v140.body_for`. It harvests a dedicated
distractor pool (`--n_dist_rows`, default 1400 rows = 2,800 triples) from the train split, disjoint from
both the training and evaluation rows (asserted), shared by both arms so the regimes match, and **aborts**
if the pool cannot deliver the requested count. The delivered fact count is written into the output JSON.

**Re-run scope.** Because the distractor *source* changed — previously each arm drew from its own row list,
now both draw from the dedicated pool — the already-measured 0/64/256 cells are not composed the same way
as the new ones. To avoid a seam inside one curve, the whole sweep is re-run: pools 0, 64, 256, 1024, 1536,
2048, same reader, seed, training budget and scorer. The old cells are kept in the ledger as a superseded
reading, not deleted.

**Bars unchanged.** The primary bar is still A(1024) against A(0), with the same thresholds. It could not
have been read before this fix, because A(1024) did not exist.
