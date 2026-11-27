
## 2026-09-10 — V-127 PREREG: is the 28.9% evidence mismatch specific to KNOWLEDGE-BASE-DERIVED annotation? The V-125 audit, replicated on MuSiQue with the same judge (user-directed: "去提升一下论文质量吧，需要的话，可以继续跑实验"; BLIND on P1–P3)

**The question, and why it is the paper's biggest open hole.** V-125 measured that on 2WikiMultihopQA
**83 of 287 decidable instances (28.9%)** have the second-hop fact neither stated nor entailed by the
supporting sentences the single-pass arm reads. My adversarial review of the ICLR draft put the paper at
5/10, and the first of the four blocking objections was that **every number in it is 2Wiki**, so the most
novel finding in the paper might be an artifact of one benchmark.

**The mechanism hypothesis, stated so it can be wrong.** 2Wiki pairs Wikipedia *prose* with *Wikidata
triples*, so a triple can be true in the knowledge base and absent from the passage. MuSiQue is built the
other way round: it **composes verified single-hop questions** whose answers are, by construction,
answerable from their paired paragraph. If the mismatch is an artifact of KB-derived annotation, MuSiQue's
rate should be far lower. If MuSiQue is just as bad, the mismatch is a general property of multi-hop
benchmark construction — a stronger claim, not a weaker one. **Either outcome improves the paper**, which
is why this is the right experiment rather than a fishing expedition.

**Protocol, held as close to V-125 as the datasets allow.**
- **Data:** MuSiQue answerable, validation, **2-hop rows only** (`id` starts `2hop`), first n=300 whose
  `question_decomposition` has exactly two hops and whose `paragraph_support_idx` resolves for both. Rows
  that fail to resolve are declared **undecidable in advance**, counted, reported, and never judged — the
  V-125 rule.
- **What is judged:** for each row, whether the supporting text **states or entails** that hop's fact.
  The fact form matches V-125 exactly, which I confirmed from the data before writing this: MuSiQue's
  decomposition gives hop 1 as `"Green >> performer" -> "Steve Hillage"` and hop 2 as
  `"#1 >> spouse" -> "Miquette Giraudy"`, so hop 2's fact is `(hop1 answer, relation, hop2 answer)` =
  `(Steve Hillage, spouse, Miquette Giraudy)` — the same `(s, r, o)` triple V-125 judged. The judge
  prompt is the V-125 template with the triple rendered the same way; labels unchanged:
  `STATED` / `ENTAILED` / `ABSENT`.

- **A CONFOUND I am registering AGAINST MY OWN PREDICTION.** V-125 judged supporting **sentences**;
  MuSiQue annotates support at **paragraph** granularity (`paragraph_support_idx`, `is_supporting`).
  A paragraph is strictly more text than a sentence, so paragraph-level judging biases the absent rate
  **downward** — the same direction as my expected result. Left unaddressed, a P1 hold would be
  partly manufactured. So the run measures **both** granularities on the same rows:
  **(i) paragraph-level**, the dataset's own annotation, which is the primary; and
  **(ii) sentence-level**, the single sentence inside that paragraph with the highest lexical overlap
  with the fact, which is the closest analogue of what V-125 judged. Both are judged by the same judge in
  the same run.
- **Judges:** `deepseek/deepseek-v4-flash` — **the same model that was the usable judge in V-125** — plus
  `moonshotai/kimi-k3` as a second judge so consensus is possible. Both via **CommonStack**
  (`https://api.commonstack.ai/v1`), temperature 0.
- **Calibration:** 40 rows drawn deterministically, annotated **before any judge output exists**, and
  committed as its own artifact. As in V-125 the annotator is the **AGENT**, disclosed as such and not
  claimed to be human annotation; a human pass by the user or the co-author remains available and would
  supersede it.
- **Gate, unchanged from V-125:** a judge is usable iff it agrees with the 80 calibration
  fact-judgements on **≥ 85%**, pooled `PRESENT (STATED|ENTAILED)` vs `ABSENT`. A judge that fails the
  gate, or that does not return a parseable verdict for every decidable row (value-blind completeness),
  is excluded **before its answers are consulted**. If only one judge survives, the single-judge path is
  a **pre-registered degraded mode** and is reported as such.

**Predictions.**
- **P0 (validity gate):** at least one judge clears the 85% pooled calibration gate. If none does,
  **nothing is read** and the failure is the result.
- **P1 (PRIMARY, paragraph-level):** MuSiQue's hop-2 not-answerable rate is **< 0.15** — below half of
  2Wiki's 0.289 and below the bar V-125 itself used. Registered as: P1 holds iff `rate < 0.15`.
- **P1b (REGISTERED SECONDARY, sentence-level):** the same rate over the best-overlap sentence.
  **Registered reading: the provenance claim is licensed only if BOTH are < 0.15.** If P1 holds while
  P1b does not, the contrast is at least partly annotation *granularity* rather than annotation
  *provenance*, and the paper must say that instead.
- **P2 (descriptive, no bar):** the per-hop `STATED`/`ENTAILED`/`ABSENT` distribution, and the hop-1
  versus hop-2 asymmetry (2Wiki: 6 against 83).
- **P3 (descriptive, no bar):** a bootstrap 95% CI on the MuSiQue rate, and on the 2Wiki − MuSiQue
  difference, using V-125's banked 83/287.

**My expected direction, stated separately from the bars.** I expect **P1 to HOLD** — MuSiQue's hops are
verified single-hop QA pairs, so the prose should contain them. I am on record that this is a prediction I
can lose: if MuSiQue's rate is also near 29%, my mechanism story is wrong and the finding is broader than
I claimed.

**Readings, fixed now.**
- **P1 holds ⇒** the mismatch is **annotation-provenance-specific**. The paper states it that way, names
  the three constructions (2Wiki = KB triples over prose; HotpotQA = human sentence pointers into the
  prose; MuSiQue = composed verified single-hop questions), and the 28.9% becomes a measured claim about
  KB-derived benchmarks **with a contrast case**, which is exactly what the single-benchmark objection
  asked for.
- **P1 fails ⇒ reported first, as the stronger finding:** multi-hop benchmarks are broadly not
  evidence-matched, the audit generalises beyond 2Wiki, and the paper's caution is raised from one
  dataset to a class. My mechanism hypothesis is then recorded as refuted.
- **P0 fails ⇒** nothing is read; the judge failure is reported and the experiment is re-registered only
  with a different judge, never with a relaxed gate.

**Deviations from the standing rules, recorded BEFORE launch because the gateway forces them.**
1. **No provider pinning.** CommonStack returns no `provider` field, so V-123's standing rule ("pin the
   serving provider on every API prereg, record its quantization") **cannot be honoured**: which weights
   served a call is not observable. Consequence accepted — per-row claims were already forbidden by
   V-123, and this run makes only aggregate claims.
2. **No spend meter.** `/credits`, `/usage`, `/billing/*` all 404 and the `usage` object carries no
   `cost` field, so V-108's rule (cap against the account meter, because per-call cost fields understate
   by up to 8×) is unenforceable. Substituted: a **hard cap of 1400 calls** for the whole run, counted in
   the harness, and the user is told that spend is not observable from my side.
3. **No reasoning field**, so the four-category hop split is not computable on this gateway. Not needed
   here — judge verdicts do not use it.

**Scoping.** The store is not re-measured; no model participates in it. This run says nothing about
*acquisition* (V-97/V-98) and nothing about the npj submission. It bears on exactly one sentence of the
ICLR draft — the scope of the audit — and on the "one benchmark" objection.

**Cost.** ≤ 1400 calls: 300 rows × 2 granularities × 2 judges + retries, plus the calibration draw. `PYTHONHASHSEED=0`
enforced by abort. Prereg + harness + **frozen scorer** committed before the first call. One tick held
before scoring; the scorer is run verbatim.
