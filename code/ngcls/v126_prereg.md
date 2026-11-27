
## 2026-09-10 — V-126 PREREG: the retrieval baseline the ICLR draft lacks, on a SECOND benchmark, with a local reader (user-directed: "去提升一下论文质量吧，需要的话，可以继续跑实验"; BLIND on P1–P4)

**Why now.** I reviewed the ICLR draft adversarially and put it at 5 (marginally below acceptance). Two of
the four blocking objections are experiments, not prose: (a) the paper never compares against an ordinary
retriever — its `evidence_in_prompt` arm is the retrieval-error-zero *ideal*; (b) every number is
2WikiMultihopQA, so both the 28.9% evidence mismatch (V-125) and the surviving claim that *structuring* the
evidence is the lever (V-125 step 5) could be artifacts of one benchmark. **OpenRouter is at $0.27, below
the $3 floor, so no API arm is possible.** This design closes both objections with GPU only.

**Design.** Three arms share ONE reader (local Qwen3-8B, bf16, greedy, `do_sample=False`) and ONE prompt
renderer, run over two datasets:

| arm | evidence the reader sees |
|-----|--------------------------|
| `single_pass` | all ten paragraphs under a 1200-token budget (the paper's arm) |
| `bm25_topk` | BM25 over the row's sentences, top-k, rendered in document order |
| `oracle_sentences` | exactly the annotated supporting sentences, same rendering |

**k is matched per row** to that row's number of annotated supporting sentences, so `bm25_topk` and
`oracle_sentences` differ only in *which* sentences were chosen and never in how many. BM25 is Okapi with
k1=1.5, b=0.75 over lowercased alphanumeric tokens, implemented inline so its constants are visible.

**Datasets.** 2WikiMultihopQA compositional validation (the paper's slice, via the banked `v85_data`
loader) and HotpotQA-distractor validation, `type == "bridge"`, rows whose supporting pointers all
resolve. n = 300 each. The two differ in exactly the way that matters: **2Wiki's hop facts are drawn from
a knowledge base, HotpotQA's supporting sentences are human-selected from the same prose the reader sees.**

**A fourth arm on 2Wiki only, as the validity anchor:** `single_pass_banked_builder`, which uses
`v85_data.prompt_for` — the exact builder behind the served ladder — so the local reader can be placed
against the banked served value. The three registered arms use the new shared renderer, because a
cross-dataset comparison must not be confounded by two different builders.

**Predictions.**
- **P0 (validity gate):** 2Wiki `single_pass_banked_builder` lands within **±.10** of the banked served
  `.47` (V-108, qwen3-8b). **A miss does NOT void the run** — the three arms share one reader and one
  renderer and stay internally comparable. What a miss forbids is cross-referencing these numbers to the
  served ladder, and the scorer does not do that.
- **P1:** on **both** datasets, `oracle_sentences − single_pass` has a paired bootstrap 95% CI excluding
  zero.
- **P2:** on **both** datasets, the fraction of the oracle gain that ordinary retrieval recovers,
  `(bm25 − sp) / (oracle − sp)`, is **< 0.50**.
- **P3 (descriptive, no bar):** the oracle gain on 2Wiki minus the oracle gain on HotpotQA — the
  annotation-provenance contrast.
- **P4 (descriptive, no bar):** BM25 sentence-level recall of the annotated supporting set, per dataset,
  which is the mechanism behind whatever P2 does.

**My expected directions, stated separately from the bars.** I expect **P1 to hold on both**. I expect
**P2 to FAIL on HotpotQA and hold on 2Wiki**: HotpotQA bridge questions share surface tokens with the
first hop, so a lexical retriever should recover much of the gain there, whereas on 2Wiki the second-hop
fact is often not stated in the prose at all (V-125: 28.9%), and no retriever can select a sentence that
does not exist. I have **no expectation** on P3's magnitude. If P2 fails on both, that is a result against
the paper's general framing and I have written the reading for it below before seeing any number.

**Readings, fixed now.**
- **P1 fails on either dataset ⇒ reported first.** On that benchmark, handing the reader the exact
  supporting sentences does not beat handing it all ten paragraphs, so "structuring the evidence is the
  lever" does not replicate and the paper gets a dataset-scope statement naming where it holds.
- **P2 fails on both ⇒** an ordinary lexical retriever already captures most of the available gain, the
  store's contribution over a plain retriever is small on these tasks, and the defensible argument for an
  exact index moves to the axes where retrieval does not help: evidence beyond the context window (V-106)
  and capacity (V-101/V-102). This would be a real weakening and it goes in the paper as one.
- **P2 fails on one ⇒** the claim becomes benchmark-conditional and is stated that way, with the
  BM25-recall mechanism (P4) as the explanation.
- **P2 holds on both ⇒** evidence *selection* is a real unsolved problem here rather than a straw man,
  which is the strongest available argument for the exact index, and it goes in the body.
- **P0 fails ⇒** no cross-reference to the served ladder anywhere; the arms are still read.

**Discipline.** Prereg + `v126.py` + the frozen `v126_score.py` committed **before the first
generation**. `PYTHONHASHSEED=0` enforced by an abort (defect 024d9e0). Every prompt set carries a
printed sha256 prefix and token total. Per-row dumps for all arms. The completeness filter is
value-blind: an arm with `n_scored != n_rows` is excluded on row counts alone, and a dataset missing any
registered arm is not read. Intervals are paired bootstraps (same rows, same reader) at 20000 resamples,
seeded. One tick held before scoring; the scorer is run verbatim.

**Scoping.** **No API call, zero spend.** The store is not re-measured — no model participates in it, and
nothing here changes its .974. Nothing here bears on the npj submission. This run cannot speak to the
*acquisition* question (V-97/V-98: recall .6589, end-to-end .4678), because every arm here is handed
either all the paragraphs or an annotated sentence set.

**Cost.** GPU only: 2 datasets × 3 arms + 1 anchor = 7 arm-runs × 300 rows ≈ 2100 greedy generations on
one A800 with an 8B model. Estimated well under two hours. Host: the A800 box, `dino_wm` env
(transformers 4.57.6, torch 2.4.1), one GPU, `HF_ENDPOINT=https://hf-mirror.com`.
