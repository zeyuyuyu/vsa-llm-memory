
## 2026-09-10 — V-128 PREREG: the retrieval baseline pushed to where it can BEAT me. A budget sweep and a dense retriever, because V-126's headline invites one obvious rebuttal and I would rather answer it than survive it. (BLIND on P1–P4)

**Why.** V-126 closed with a strong finding: at the oracle's own per-row budget a lexical retriever is
*worse than no retrieval at all* (2Wiki $-0.290$, HotpotQA $-0.130$, both CIs excluding zero). The paper
states it, and states the limitation. But the rebuttal writes itself: **"you crippled the retriever"** —
$k$ was matched to the gold count (2.00 and 2.48 sentences), and BM25 is lexical. A deployed system
retrieves five or ten *paragraphs* with a dense encoder. If that closes the gap, my newest and strongest
claim is a budget artifact, and I want to know that from an experiment rather than from a referee.

**Design.** Identical reader, rows and renderer to V-126 — the retrieval functions and dataset loaders are
**imported from `v126.py`** so the code path is the same object, and the launch record checksums it. Per
dataset ($n=300$, 2WikiMultihopQA compositional and HotpotQA-distractor bridge):

| arm family | configurations |
|---|---|
| controls | `single_pass` (all ten paragraphs), `oracle_sentences` |
| sentence retrieval | {BM25, dense} $\times$ $k \in \{$gold, 5, 10, 20$\}$ |
| paragraph retrieval | {BM25, dense} $\times$ $k \in \{2, 5\}$ |

$14$ arm-runs per dataset, $28$ in all, $8400$ greedy generations. **Paragraph-level retrieval is included
because that is what deployed retrieval-augmented systems actually do**, and its absence is the honest
weakness of V-126. Dense retriever: `BAAI/bge-small-en-v1.5`, CLS pooling, L2-normalised, cosine, query
prefixed per its model card. BM25 is V-126's Okapi ($k_1{=}1.5$, $b{=}0.75$) unchanged.

**Predictions.**
- **P0 (validity gate):** the BM25 sentence arm at $k=$gold reproduces V-126's banked accuracy within
  $\pm0.05$ on **both** datasets ($0.0433$ and $0.3667$). Same rows, same reader, same renderer, imported
  retriever — a miss means the harness moved and **nothing is read**.
- **P1 (PRIMARY):** **no** configuration reaches the oracle: for every (retriever, unit, $k$), the paired
  bootstrap CI of `oracle - config` excludes zero. Registered as: P1 holds iff that is true for all $24$
  retrieval configurations.
- **P2:** the **best** configuration beats `single_pass`, i.e. `best - single_pass` has a CI excluding
  zero.
- **P3 (descriptive, no bar):** for each retriever and unit, the smallest $k$ at which it stops hurting
  relative to `single_pass`.
- **P4 (descriptive, no bar):** recall of the gold supporting sentences against $k$, per retriever and
  unit — the mechanism behind P1 and P3.

**My expected directions, stated separately from the bars, and one of them costs me.** I expect **P1 to
hold**: the second hop's sentence is often not similar to the question at *any* budget, and as $k$ grows
the context re-approaches the unfiltered one, so retrieval should approach `single_pass` from below rather
than reach the oracle. I expect **P2 to HOLD**, and that **weakens my own current claim**: if a larger
budget lets retrieval beat `single_pass`, then "BM25 is worse than no retrieval" is true only *at oracle
budget*, and the paper's Sec. 3.2 must be narrowed to say exactly that. I am registering that in advance
so the narrowing cannot be presented as a nuance I always intended.

**Readings, fixed now.**
- **P1 fails ⇒ REPORTED FIRST.** Some ordinary retriever reaches oracle evidence at a realistic budget.
  Then the paper's strongest argument for an exact index on this task is gone: what the store supplies is
  reachable by retrieval, and the contribution must be restated as being about *capacity and cost*
  (Sec. 3.1, Appendix on the envelope) rather than about evidence quality. I would rather publish that
  than have it found.
- **P1 holds ⇒** the oracle-versus-retrieval gap survives a budget sweep and a dense encoder, so
  "evidence *selection* is the hard part" gets a curve behind it instead of a single harsh point.
- **P2 holds ⇒** Sec. 3.2's claim is narrowed to the oracle-budget regime in the same commit that reports
  it, and the crossing $k$ (P3) is reported as the honest boundary.
- **P2 fails ⇒** retrieval never beats the unfiltered context at any tested budget, which strengthens
  V-126 rather than qualifying it.
- **P0 fails ⇒** nothing is read; the harness divergence is investigated first.

**Discipline.** Prereg + `v128.py` + the frozen `v128_score.py` committed before the first generation.
`PYTHONHASHSEED=0` enforced by abort. Every arm's prompt set carries a printed sha256 and token total.
Per-row dumps for all $28$ arms. Value-blind completeness: an arm with `n_scored != n_rows` is excluded on
counts alone. Intervals are paired bootstraps over rows (same rows, same reader), $20\,000$ resamples,
seeded. One tick held before scoring; the scorer is run verbatim.

**Scoping.** GPU only, **zero API spend**. The store is never re-measured — no model participates in it.
Nothing here bears on acquisition (V-97/V-98) or on the npj submission. GPU 0 is in use by another user,
so this run pins a free device.
