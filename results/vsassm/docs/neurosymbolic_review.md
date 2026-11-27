# Neuro-symbolic reasoning on Readable Memory: literature review

2026-11-06, Part III added 2026-11-07. Part A of docs/neurosymbolic_brief.md. Parts I-III below were compiled from primary sources (each number
carries its table or figure; anything not opened is marked UNVERIFIED). Section 0 is the synthesis, and it folds in our
own composition-table control (NSR0 to NSR0e, results/nsr0_verdict.txt), because that control changes what the
literature means for us.

## 0. Synthesis

### 0.1 What the literature and NSR0 say together

1. **A composition table is the published state of the art for learned kinship composition, and it is what our VSA
   rule memory amounts to.** R5 (ICLR 2022) stores rules in two hash tables, (r_B0, r_B1) : r_head and
   (r_B0, r_B1) : score, and reaches .97 +- .03 at 10 hops on CLUTRR trained on k = 2, 3 (R5 Table 2; re-checked from the
   PDF). DSR-LM without its language model learns a weighted composite(r1, r2, r3) table and reaches 98.81 averaged over
   k = 4..10 (DSR-LM Table 7). DSR-LM's learned table and its 92 hand-written rules score the same on text (60.98 vs 61.34,
   Table 6). Our gold-facts result (0.975 overall, 1.000 at k = 10) sits at this level, not above it.
2. **NSR0 confirms it for our own store.** On the same training chains, a symbolic count table beats every VSA store
   we tested in every cell (sample size, coverage, label noise, fact noise, size, confidence), confirmed on fresh data
   draws with parameters fixed in advance (NSR0e). VSA's apparent wins under noise (two registered bars passed on the
   way) came from how weakly supported rules are handled; a table with staged support thresholds and derivation voting
   does it better. R1's VSA memory is also less exact than R1 reported: over ten
   codebooks it averages 0.962, not 0.975, and a packed table of the same rules takes 219 bytes against 16 KB for the
   vector. There is a structural reason (Part I, RotatE): FHRR binding is commutative and kinship composition is not, so
   binding alone cannot be the composition operator, and a VSA rule store reduces to an associative lookup.
3. **Our text-input comparison point was wrong.** DSR-LM's 61.0 is not the best published text-input number on
   gen_train23_test2to10: SatLM reports 68.3 (greedy) and 80.1 (self-consistency, 40 samples) with code-davinci-002, and
   Faithful CoT (ProgLM column) 58.9 / 71.9 (SatLM Table 1; re-checked from the PDF). Both run on a 1,042-story version of
   the test set with a hand-written kinship table. Our Qwen3-8B reader's 0.717 is not like-for-like with either.
4. **Impact questions over changing state are an established task, and memory systems fail it.** MEME (arXiv
   2605.12477, May 2026) defines Cascade (an upstream update plus an if-then rule changes a dependent value) and Absence
   (no rule applies, so "uncertain"); its abstract reports average accuracy of 3% on Cascade and 1% on Absence across the
   systems tested (re-checked from the abstract; per-system numbers in Part II). When the dependency graph is explicit,
   traversal solves impact questions (Tacheny, arXiv 2601.07342: 100% for a tool-using model on 10 synthetic cases).
   So the hard part is having the right graph, i.e. the state, which is what Readable Memory tracks.
5. **The VSA-specific advantages that have published evidence are elsewhere:** reusing a rule on unseen
   (attribute, rule) pairs (Learn-VRF: 98.8-100% vs 14.8-75.0% for an MLP, Table 3), scoring many rule candidates at once
   (NVSA), calibrated abstention from similarity (ConformalHDC), and low-power hardware (Loihi resonator, 171x
   energy-delay product). None of these is a property of a relation-symbol lookup table, and none has been shown for
   relational composition over operational state.
6. **Fast Weight Memory (Schlag et al., ICLR 2021) is the closest architectural ancestor of Readable Memory** (fixed-size
   tensor-product fast weights, delta-rule overwrite, chained reads; 96.75% on catbAbI) and must be cited.

### 0.2 The open space for us (candidate claims a-d)

| Claim | Closest prior | Verdict | What would make it ours |
| --- | --- | --- | --- |
| (a) rules learned from few examples into a fixed-size readable store | R5, NCRL, CTP (tables, CLUTRR); Learn-VRF / ARLC (VSA rules, Raven) | Pre-empted as stated. NSR0: a table does it as well. | Only a property a table lacks: rule reuse on unseen relation pairs, which needs non-commutative, structured codes (Learn-VRF is the precedent) |
| (b) reasoning over a continuously updated state (impact, what-if) | MEME Cascade/Absence; MQuAKE; Tacheny; AlMahri; ChatDB; Log2Graph; MM-3DSGU (Part III) | Task framing not novel; our exact combination not found | Reading changes the system of record never logged, from human channels mixed with system feeds, into a continuously updated symbolic state, answering exact impact queries with log-line provenance (NSR1). "The interface is the hard part" is already established for static text (Part III.3), so the novelty is the stream version: overwrites, unlogged changes, cross-channel reconciliation, provenance |
| (e) a neuro-symbolic engine for impact queries | Scallop, DeepProbLog, DSR-LM; clingo event calculus; NVSA (Part III.6) | Pieces exist; the combination was not found | Probabilistic, time-aware impact queries over the read state (per-fact read confidences, defrost-window history including things moved out since, what-if overlays that do not mutate the state, calibrated abstention). The VSA angle (many hypothetical states in superposition) is open but must beat Scallop |
| (c) confidence, abstention, human correction | MEME Absence; ConformalHDC; HtT and R5 rule scores; ELLEN, TeachMe | Each piece has precedent; the combination was not found | Calibrated abstention on rule chains plus editable state with traced answers; edit locality must be measured, since deleting a table row is exact |
| (d) edge and robot settings | Loihi resonator; PCM factorizer; HDReason | Hardware precedent is perception and KG completion only | A measured on-device run against an embedded key-value or graph store, not against an LLM |

No "first" claim survives for (a) or (b) as stated. (c), (d) and (e) survive only as combinations, and need measurements. The specific stream setting of (b) is the strongest candidate (Part III.5, III Verdict).

### 0.3 Baselines each claim must beat

| Claim | Strongest baseline | Second | Note |
| --- | --- | --- | --- |
| Composition from few examples (CLUTRR, gold facts) | Count table with staged support thresholds and derivation voting, same examples (NSR0e) | R5 at equal data; DSR-without-LM | A VSA store must beat the table, not tie it |
| Composition from text (CLUTRR) | SatLM-style "LLM + hand table" with our own reader LLM | Reason Popper-ly (ILP table from 200 2-hop examples + LLM extraction) | Published SatLM/Faithful CoT numbers use Codex and a 1,042-story test |
| Impact over tracked state | Same state + graph query (NSR1 arm 2b); a frontier long-context LLM reading the full log + site facts | ChatDB-style pipeline with a frontier reader (isolates distillation); a fine-tuned 8B end-to-end model; Graphiti/Zep or Mem0 fed the same streams; Qwen3-32B over the log | Risk: long-context GPT-5-mini beat a Mem0 fact memory on recall (arXiv 2603.04814, abstract opened); in our favour, end-to-end LLMs collapse under overwrites (PI-LLM) and on indirect effects (ActionReasoningBench: GPT-4o state tracking 73-76%, ramification 0%, Tables 3-4, re-checked) |
| Neuro-symbolic engine | Scallop (top-k proofs) for uncertain facts; clingo event-calculus encoding with a self-correcting LLM+ASP for temporal/what-if; DSR-LM for learning from answers | Exact ProbLog on small cases; Dolphin for scale | Answer-only training can give right answers from wrong facts (rsbench: DeepProbLog 0.94 answer F1, 0.06 fact F1), so report fact-level accuracy and use answers as fine-tuning on top of fact labels |
| Abstention / confidence | Rule table with support counts that abstains when no rule fires | Conformal prediction on the LLM's answer | NSR0 A4: table confidence AUROC >= VSA's |
| Editing | Table row delete or replace | ELLEN, TeachMe | VSA edits must show locality |
| Edge | Dict or SQLite + BFS on a Jetson or microcontroller | Small on-device LM | Energy against an LLM is a strawman |

External benchmarks worth running (Part III): ActionReasoningBench State Tracking + Ramification subsets (ASP ground truth; indirect effects = our cascades) and PI-LLM (overwrites); MEME Cascade/Absence (Part II).

### 0.4 Corrections to our own records

- "Best published CLUTRR text-input number is DSR-LM 61.0" (readable_memory_lit_check, paper_rm): false; SatLM 68.3 /
  80.1 and Faithful CoT 71.9 (SC), on a 1,042-story test version with a hand-written table.
- "Gold-facts 0.975 / 1.000 at k = 10" is at the level of R5 and DSR-without-LM, not a distinguishing result; and our
  VSA version of it averages 0.962 over codebooks.
- Any sentence that credits the reasoning to VSA is unsupported (NSR0 to NSR0e).

### 0.5 Verification of this review

Three independent subagents compiled Parts I-III from primary sources. For Part III I re-opened ActionReasoningBench
(Tables 3-4: GPT-4o 73.33/74.42/75.51, ramification 0.00, o1-preview 18.4) and arXiv 2603.04814 (abstract); both
matched. For Parts I-II I re-opened the three load-bearing sources
myself: SatLM Table 1 (CLUTRR 68.3 greedy / 80.1 SC; ProgLM 58.9 / 71.9; code-davinci-002), R5 Table 2 (.97 +- .03 at 10
hops) and its hash-table rule memory (Section 3), and the MEME abstract (Cascade 3%, Absence 1% on average). All three
matched. Items the subagents could not open are marked UNVERIFIED in place (e.g. Emruli & Sandin 2014; AMIE).

---

# Part I. LLM + solver, learned rule induction, compositional KG reasoning

Draft, 2026-11-06. Scope: the three families in the brief, with most depth on CLUTRR. Every number below comes from a primary source I opened (arXiv PDF or HTML, the HF dataset card, or the authors' PDF), and each gives its table or figure. Anything I could not open is marked **UNVERIFIED**.

---

### I.0. Bottom line (decision-relevant)

1. **Our text-input comparison point is out of date.** DSR-LM's 60.98 is not the best published text-input number on our split. On the same CLUTRR split (`data_089907f8` = `gen_train23_test2to10`, using a 1,042-example test version), two papers report higher numbers:
   - **SatLM: 68.3 greedy, 80.1 with self-consistency** (40 samples, code-davinci-002; SatLM Table 1).
   - **Faithful CoT: 58.9 greedy, 71.9 with self-consistency** (Faithful CoT Table 1).

   Both pipelines are an LLM translator plus a **hand-written kinship composition table**. Our Qwen3-8B reader scores 0.717 overall. That is about the same as Faithful CoT-SC and below SatLM-SC, but neither comparison is like-for-like: different LLM, different test subset (1,042 vs 1,146), and their table is given, not learned.
2. **The closest published precedent for our idea is a composition table.**
   - **R5** (ICLR 2022) stores rules in an explicit hash table `(r_B0, r_B1) -> r_head` and reduces a path pair by pair. Trained on k=2,3 graphs, it scores **.97±.03 at k=10** (R5 Table 2).
   - **DSR-LM without the LM** learns a weighted table `composite(r1,r2,r3)` and reaches **98.81** on gold KBs (train k=2,3, test k=4..10; DSR-LM Table 7). CTP scores 95.57 and GAT 39.05 in the same table.

   So gold-graph composition-table learners are already at our gold-facts level: we have 0.975 overall and 1.000 at k=10.
3. **Learned table vs hand table: no difference.** DSR-LM with learned rules scores 60.98±2.64; with 92 hand-written rules it scores 61.34±1.56 (DSR-LM Table 6). Of its top-92 learned rules, 70 match the hand-written ones (DSR-LM §4.3). This matches our own "plain table does as well" worry.
4. **Table vs VSA store on CLUTRR: no precedent found.** No paper I found puts a learned composition table head-to-head with a VSA, HRR or FHRR store on CLUTRR (queries in §5.2). The nearest head-to-heads are table-like rule learners against neural models:
   - R5 vs CTP, GNTP and GNNs (R5 Tables 2–3).
   - DSR-without-LM vs CTP and GAT (DSR-LM Table 7).
   - EpiGNN vs R5 (EpiGNN Table 13). The EpiGNN authors state that rule learners win in the k=2,3 regime because the training set is small.
5. **The newest neighbour (July 2026): "Reason Popper-ly"** (Chen & Ma, NeSy 2026). It learns a `compose(r1,r2,r3)` table with Popper ILP from **200 2-hop instances**, supplements it with gold entries, and uses it behind LLM fact extraction ("Exogenous" pipeline).
   - At 8–10 hops it scores **69.53 with GPT-5.4** and **46.72 with Qwen-3.5:4B** (Table 1).
   - The authors attribute the long-chain decline to extraction and graph construction, not to the table. This is the same pattern as our reader (0.645 at k≥7).
6. **RotatE's relation rotation is FHRR binding, and it composes commutatively.** RotatE models a relation as an element-wise unit-modulus complex rotation, `t = h ∘ r, |r_i| = 1`, with composition `r3 = r1 ∘ r2` (θ3 = θ1 + θ2); see RotatE Eq. 1 and §1, Lemma 3 in §3. That operation is mathematically FHRR binding, and it is commutative; DensE criticises rotation models for exactly that.
   - Kinship composition is not commutative (father∘mother ≠ mother∘father), so binding alone cannot implement the rule store. A VSA system needs an associative lookup, which is functionally a table. Reviewers will make this point.
7. **What the literature says a table cannot do:**
   - **Disjunctive compositions** (RCC-8 and Allen intervals). R5 and CTP time out at 5+ hops (EpiGNN Fig. 4 caption).
   - **Non-deterministic CLUTRR compositions.** "A grandmother's daughter can be either a mother or an aunt" (HtT §4). The Faithful CoT annotators flagged the same ambiguity.
   - **Noisy or mislabelled data** (∂ILP Table 4; R5 on GraphLog).

   None of these has been tested with a VSA store on CLUTRR. They are the most plausible places for our store to beat a table.

---

### I.1. Which split is which (needed to compare numbers)

| Name used in papers | HF config | Train k | Test k | Sizes | Source |
|---|---|---|---|---|---|
| `data_089907f8` | `gen_train23_test2to10` | 2,3 | 2..10 | 9074 / 2020 / 1146 (train/val/test) | HF CLUTRR/v1 card; CLUTRR Table 6 |
| same, graph version "CLUTRR_G(k=2,3)" | — | 2,3 | 4..10 reported | 10,094 train / 900 test | CTP §5.3; EpiGNN Table 7 |
| same, Faithful CoT/SatLM text version | — | prompt exemplars k=2,3 | 2..10 | **1,042 test** | Faithful CoT Table 9 & App. E; SatLM App. A |
| `data_db9b8f04` | `gen_train234_test2to10` | 2,3,4 | 2..10 (5..10 reported) | 12064/3019/1048 (HF); 15,083/823 (NCRL Table 12) | HF card; NCRL |
| `data_9b2173cf` | — (ET regenerated) | 2,3 | 4..10 | 35,394 / 39,825 | EpiGNN Table 7; ET §4.1 |
| `data_7c5b0e70 / 06b8f2a1 / 523348e6 / d83ecc3e` | `rob_train_{clean,sup,irr,disc}_23_test_all_23` | 2,3 | **2,3 only** | ~8.1k / ~2.0k / ~445 | HF card; CLUTRR Table 6 |

**Caveats:**
- CTP §5.1 describes its CLUTRR_G graphs with **nine gender-neutral relation types** (child, grand, in-law, inv-child, inv-grand, inv-in-law, inv-un, sibling, un). EpiGNN Table 7 lists **22 relations** for the same hash. Check which vocabulary R5 and CTP actually used before claiming parity with our gold-facts numbers.
- The noise ("robust") splits contain only k=2,3. **No published system reports long-chain (k>3) accuracy under supporting, irrelevant or disconnected noise.**

---

### I.2. Family 1: LLM + symbolic solver (2023–2026)

| System (venue) | Benchmark / input | Headline numbers (exact, with source) | Rules given/learned | Failures reported | Composition table in effect? |
|---|---|---|---|---|---|
| **SatLM** (Ye et al., NeurIPS 2023) | CLUTRR `data_089907f8`, 1,042 test, text; prompts k≤3, test to 10 | code-davinci-002: greedy **68.3**, self-consistency (40 samples) **80.1**. CoT 40.8 / 45.7, ProgLM 58.9 / 71.9 (Table 1). gpt-3.5-turbo 50.6, text-davinci-003 58.2 (Table 6). Answers 76.0% of items, selective accuracy 89.9, UNSAT 15.5% (Table 5) | **Given:** transitivity rules from Lyu et al. / Zhang et al. 2022a (App. H) | LLM writes wrong constraints that still yield an unambiguous wrong answer (Fig. 9). No per-k breakdown | **Yes.** The solver applies a fixed composition table to LLM-written constraints |
| **Faithful CoT** (Lyu et al., IJCNLP-AACL 2023) | Same split, 1,042 test | Codex greedy **58.9**, SC **71.9** (Table 1; Table 3 "optimal" 71.9). Removing the solver: 40.9 (Table 4) | **Given:** "transitivity rules provided by Zhang et al. (2022)… son@daughter=grandson" (§3.4) | Annotators flagged CLUTRR's ambiguity, e.g. a grandmother of one's child may be a parent-in-law (§6.3) | **Yes**, literally: LLM translation, then a table fold |
| **LM+ASP** (Yang, Ishay & Lee, Findings ACL 2023) | CLUTRR 1.0 robust splits; CLUTRR 1.3 (400 self-generated, k=3); CLUTRR-S; StepGame | GPT-3(d3)+ASP, CLUTRR 1.0: clean 68.5 / supp 82.8 / irr 74.8 / disc 67.4. CLUTRR 1.3: 97.0 / 84.0 / 92.0 / 90.0 (Table 3). CLUTRR-S d3: 100 in all cells (Table 8). StepGame k=10: 88.3 (d2, Table 2) | **Given:** hand-written ASP modules (limitations: "knowledge modules be written by hand") | **Did not run systematic generalisation**: "we focus on the robust reasoning task" (§4.3). Low CLUTRR 1.0 scores blamed on dataset errors; on 1.3, 16/400 wrong labels and 21/400 parse errors | **Yes** (hand rules) |
| **DSR-LM** (Zhang et al., Findings ACL 2023) | CLUTRR, train 10K (5K k=2, 5K k=3), test 1.1K k∈{2..10}, text | **60.98±2.64** (Table 5/6). Manual 92 rules: 61.34±1.56. No integrity constraints: 51.48±0.57 (Table 6). Gold-KB variant: **98.81** vs CTP 95.57, GAT 39.05 (train 2–3, test 4–10; Table 7). GPT-3 zero-shot 28.6, ZS-CoT 25.6 (Fig. 3) | **Learned:** weights over all `composite(r1,r2,r3)` (20 relations → 8K facts), initialised by LM prompting; top-150 used at test (§4.2, App.) | Errors come from relation extraction (RoBERTa extractor 84.69% single-relation accuracy) | **Yes:** a probabilistic composition table |
| **HtT, "LLMs can Learn Rules"** (Zhu et al., arXiv 2310.07064; venue not listed) | CLUTRR symbolic and textual; induction on **2,000** 2–3-hop samples, test **200** samples of 2–10 hops | GPT-4 symbolic: 5-shot CoT 50.6 → **+HtT 64.3** avg, 36.0 at 10 hops. EdgeTransformer reproduced in the same setting: 71.8 avg, 36.0 at 10 (Table 1). Textual: CoT 48.7 → 59.1; 48.0 at 10 (Tables 5, 8). CLUTRR "require[s] 500 examples to obtain a significant gain" (§4.5) | **Learned**, as a natural-language rule library | Rules "not deterministic, e.g. a grandmother's daughter can be either a mother or an aunt"; retrieval errors; library limited by context length (Limitations) | **Yes:** a textual composition table retrieved by the LLM |
| **Reason Popper-ly** (Chen & Ma, NeSy 2026; arXiv 2607.23019) | 3,073 CLUTRR instances, 2–10 hops (split unnamed) | Exogenous = LLM extraction + table. Bins 2-3 / 4-5 / 6-7 / 8-10: GPT-5.4 92.90 / 80.44 / 71.75 / **69.53**; Claude-4.6 Sonnet 88.87 / 75.91 / 71.31 / 66.79; Qwen-3.5:4B 75.07 / 70.94 / 56.67 / **46.72**. Their CoT-patching method: GPT-5.4 75.09 at 8-10 (Table 1) | **Learned:** Popper ILP over ground facts `compose(r1,r2,r3)` from **200 2-hop instances**, "supplemented with any missing gold entries" | Failures "primarily reflect brittleness in fact extraction and graph construction"; kinship only | **Yes** (plus gold patching) |
| **Symbolic Working Memory** (Wang et al., arXiv 2408.13654) | 235 CLUTRR test items, 2–6 steps | GPT-4 **92.34**, GPT-3.5 78.72 (Table 1); Llama-3-8B 63.40 (Table 3) | **Given** in the input context | No k>6 | Rules given; not comparable to our split |
| **NeSy prompting, "pitfalls"** (Stein et al., PMLR 2025 draft; arXiv 2505.24874 under another title) | CLUTRR (setup in App. B) | Figures only. Larger foundation models plus a Scallop program overtake trained Scallop (Fig. 2). Program-based prompting stays high as reasoning steps grow, while CoT drops (Fig. 6a) | Given program | "Program pitfall": the trained NeSy model hallucinates facts yet reaches the right answer (Fig. 5) | Yes (Scallop program) |
| **LINC** (Olausson et al., EMNLP 2023) | FOLIO, ProofWriter (no CLUTRR) | GPT-4: ProofWriter LINC 98.3 vs CoT 72.2; FOLIO 72.5 vs 75.3 (Fig. 3) | FOL premises translated, then Prover9 | FOL syntax errors in 13% of GPT-4 outputs (§5) | N/A (FOL proving) |
| **Logic-LM** (Pan et al., Findings EMNLP 2023) | PrOntoQA, ProofWriter, FOLIO, LogicalDeduction, AR-LSAT | GPT-4: 83.20 / 79.66 / 78.92 / 87.63 / 43.04; on PrOntoQA, CoT is higher (98.79) (Table 2). FOLIO executable rate 79.9% (Table 3) | Translated, then solver | Non-executable formulations | N/A |
| **Logic-LM++** (Kirtania et al., arXiv 2407.02514) | FOLIO, ProofWriter, AR-LSAT | GPT-4: 84.80 / 79.66 / 46.32 (Table 1) | Same | — | N/A |
| **SymbCoT** (Xu et al., ACL 2024) | Same family | GPT-4: ProntoQA 99.60, ProofWriter 82.50, FOLIO 83.33 (Table 1); LogicalDeduction 93.00, AR-LSAT 43.91 (Table 2). No external solver | LLM does the symbolic steps itself | — | N/A |
| **Scallop** (Li, Huang & Naik; arXiv 2304.04812; PLDI 2023 venue **UNVERIFIED**) | CLUTRR train k∈{2,3}, test 2..10 | Bars only (Fig. 17/18): "25% improvement over baselines" including GPT-3-FT (§6.3) | CLUTRR-G: learnable weights on `composition(FATHER,MOTHER,GRANDMOTHER)` facts | RoBERTa extractor "correctly extracts only 84.69%" (§6.5) | **Yes** (the precursor of DSR-LM) |

**Takeaways for family 1:**
- Every CLUTRR pipeline with a solver uses a composition table, and every one except DSR-LM, HtT and Popper-ly is given that table.
- Best text-input numbers on our split:
  - **SatLM-SC 80.1** (hand table, Codex, 40 samples).
  - **SatLM greedy 68.3**.
  - **Faithful CoT-SC 71.9**.
  - Learned-table text pipeline: **DSR-LM 60.98**.

---

### I.3. Family 2: learned rule induction

| System (venue) | Benchmark | Headline numbers | Rules given/learned; data | What fails | Composition table in effect? |
|---|---|---|---|---|---|
| **R5** (Lu et al., ICLR 2022) | CLUTRR graph `data_089907f8` (k=2,3 → 4..10); `db9b8f04`; GraphLog | k=2,3 → 4..10: **.98 .99 .98 .96 .97 .98 .97** (Table 2). k=2,3,4 → 5..10: .99 .99 .99 1.0 .99 .98 (Table 3). GraphLog World 2: 0.755 accuracy, rule recall 1.0, vs CTP 0.685, E-GAT 0.412 (Table 4) | **Learned:** the rule memory is "two hash tables… `(rB0, rB1) : rh`" plus invented relations; MCTS chooses which adjacent pair to reduce (§2.1) | Not applicable to KB completion (§4). On RCC-8/IA, R5 needs >30 min of inference at 5+ hops (EpiGNN Fig. 4) | **Yes, literally.** Closest conceptual precedent to our rule memory and pair-wise fold |
| **CTP** (Minervini et al., ICML 2020) | CLUTRR_G `089907f8` / `db9b8f04`, 10 seeds | k=2,3 → 4..10, CTP_L: **.98 .98 .97 .96 .94 .89 .89**; GNTP .49 → .31; GAT .91 → .45; LSTM .98 → .75 (Fig. 3, tuned on short stories). k=2,3,4: CTP_L .99 → .95 at 10 (Fig. 4) | **Learned:** rule templates with goal-conditioned predicate selection (soft unification) | Accuracy falls with k. Graph input only | **Soft table:** rules over learned predicate embeddings |
| **GNTP** (Minervini et al., AAAI 2020) | KBs + text | CLUTRR via CTP Fig. 3: .49±.18 (4 hops) → .31±.22 (10 hops) | Templates | NTPs "can only be used with small-scale symbolic KBs" (abstract) | Soft table |
| **NTP** (Rocktäschel & Riedel, NeurIPS 2017) | Countries, Kinship, Nations, UMLS | Countries S3 AUC-PR: NTPλ 77.26±17.0 vs ComplEx 48.44±6.3. Kinship MRR: NTPλ 0.80 vs ComplEx 0.81 (Table 1) | Templates; learned embeddings | Complexity (per GNTP abstract) | Soft table |
| **DSR-LM / DSR-without-LM** | See §2 | Gold KB **98.81** vs CTP 95.57, GAT 39.05 (Table 7) | Learned weighted table, randomly initialised in this variant | — | **Yes** |
| **NCRL** (Cheng et al., ICLR 2023) | CLUTRR **k=2,3,4 only** (22 relations, 15,083 train / 823 test, Table 12); GraphLog; KG completion | 5..10 hops: **1.0 .99 .98 .98 .98 .97**; R5 .99 .99 .99 1.0 .99 .98; RLogic .99 → .94 (Table 3). GraphLog World 2: 0.774 vs R5 0.755 (Table 4) | Learned: "recursively combining predicates into a composition and reducing the composition into a single predicate" (§1) via attention over sliding windows | Notes that an improper hierarchy (bracketing) hurts (Fig. 2). No k=2,3 result | **Neural soft table** applied recursively. A precedent for bracketing-dependent folding |
| **Edge Transformer** (Bergen et al., NeurIPS 2021) | CLUTRR graph, noiseless; k=2,3 regenerated to 35k examples | k=2,3, hops 6..10: **82.0, 74.3, 66.9, 61.8, 58.1** (Table 5, base). k=2,3,4: 98.1 → 92.1 | Neural (triangular attention) | Steep k=2,3 decline. With 2,000 training samples, 36.0 at 10 hops (HtT Table 1) | No (neural) |
| **EpiGNN** (Khalid & Schockaert, ICLR 2025) | CLUTRR, GraphLog, RCC-8, IA, inductive KGC | k=2,3,4: EpiGNN-mul 0.98±.02 at 10 (Table 1). k=2,3 (`9b2173cf`): EpiGNN-mul .96 → **.85±.13** at 10, ET .55±.08 (Table 13) | Neural composition ψ over "epistemic" embeddings | Authors: "R5… consistently outperforms our method… due to the inevitably small size of the training set… Rule learners can still perform well in such cases" (App. G) | No. Handles **disjunctive** compositions (RCC-8/IA), which a deterministic table cannot |
| **CLUTRR baselines** (Sinha et al., EMNLP 2019) | CLUTRR text and graph | Robustness (k=2,3), **GAT on graphs**, train=test noise: clean 1.0, supporting 0.98, irrelevant 0.93, disconnected 0.96. Train clean, test supporting: 0.24. Best text model average 0.61 (Table 2). Systematic generalisation shown only in Fig. 5 (no table) | Neural | Text models far below GAT; paraphrase variation drives much of the gap | No |
| **∂ILP** (Evans & Grefenstette, JAIR 2018) | 20 ILP tasks incl. family tree | Share of runs reaching MSE <1e-4: Grandparent 96.5, Uncle 70.0, Son 100 (**Table 3**). Mislabelling: Son MSE 0.065 at 10% noise; "At 10% mislabelled data, ∂ilp still finds a perfect answer in 5 out of 6 of the tasks" (Table 4, §5.4) | Learned from **tiny** example sets with rule templates | Cannot scale (NLM Table 1: "∂ILP cannot scale up in 2-OutDegree") | Rule set (table-like for 2-body rules) |
| **Neural Logic Machines** (Dong et al., ICLR 2019) | Family-tree concepts, graphs, blocks world | 100% on HasFather, HasSister, IsGrandparent, IsUncle, IsMGUncle at m=20 and m=100. MemNN IsGrandparent 96.5% / 84.7% (Table 1) | Learned tensor operators; generalise to larger domains | Single-concept tasks, not path-length generalisation | No (neural logic operators) |
| **NLIL** (Yang & Song, ICLR 2020) | KB completion, Visual Genome | FB15K-237 MRR 0.25 vs RotatE 0.34. WN18 0.95. 82 min vs 342 (Table 1) | Learned chain-like rules | Lower MRR than embeddings | Rule set |
| **RNNLogic** (Qu et al., ICLR 2021) | FB15k-237, WN18RR, Kinship, UMLS | Kinship MRR: RNNLogic+emb **.722**, w/o emb .639, RotatE .651, NeuralLP .302, DRUM .334, CTP .335 (Table 3) | Learned rule generator (EM) | — | Rule set (KG completion, not CLUTRR) |
| **Neural LP / DRUM** (NeurIPS 2017 / NeurIPS 2019) | KG completion | Via RNNLogic Table 3 (above) | Learned | Memory issues on large KGs (NCRL Table 2 text) | Soft rule sets |
| **GraphLog** (Sinha et al., arXiv 2003.06560; venue not listed) | 57 synthetic rule worlds | Via R5 Table 4 (above) | — | Benchmark finding: generalisation depends on rule diversity (abstract) | — |
| **LTN** (Badreddine et al., AIJ 2022) | Framework; no CLUTRR or path-length experiment found (grep) | — | Rules given as FOL axioms | — | N/A |
| **Gontier et al.** (NeurIPS 2020) | CLUTRR proof generation, trained on levels 2,4,6 | Figures only (Figs. 2–9) | Transformer LMs | "TLMs suffer from length-generalization issues" (Conclusion) | No |
| **AMIE** | — | **UNVERIFIED** (not opened) | — | — | — |

**Takeaways for family 2:**
- **Learned from k=2,3 and generalising to k=10** on the gold graph `089907f8`:
  - R5: .97±.03 at k=10.
  - CTP_L: .89±.07.
  - DSR-without-LM: 98.81 averaged over k=4..10.
  - Neural models drop: ET 58.1 (35k training examples), GAT .45.
- **Training data:** about 10k graphs (EpiGNN Table 7). The only few-example table learners are HtT (2,000 examples; 500 needed for a gain) and Popper-ly (200 2-hop instances, gold-patched).

---

### I.4. Family 3: compositional KG reasoning and query embeddings

| System (venue) | Composition operator (from the paper's equations) | Headline numbers | Composition table in effect? |
|---|---|---|---|
| **Guu, Miller & Liang** (EMNLP 2015) | A path query applies relation operators recursively. Bilinear = matrix product (order-sensitive); Bilinear-Diag and TransE are commutative (§3) | Path-query mean quantile, single-edge → compositional training: WordNet TransE 83.7 → 93.3; Freebase Bilinear 58.0 → 83.5 (Table 2). Naive recursive application "suffer[s] from cascading errors" (abstract) | No. Continuous operators, trained on paths |
| **GQE** (Hamilton et al., NeurIPS 2018) | Projection `P(q,τ) = R_τ q` (Eq. 4), a matrix | Bio AUC 91.0 (GQE training) vs 79.2 (edge training) (Table 1) | No |
| **Query2Box** (Ren et al., ICLR 2020) | Projection is box translation: "p + r, where we sum the centers and sum the offsets" (§3). Commutative | H@3 average, FB15k: Q2B .484 vs GQE .386 (Table 2) | No |
| **BetaE** (Ren & Leskovec, NeurIPS 2020) | Beta distributions; MLP projection | MRR average FB15k-237: 20.9 vs Q2B 20.1, GQE 16.3; 3p 10.0 (Table 1) | No |
| **CQD** (Arakelyan et al., ICLR 2021) | **Trains only on 1-hop (atomic) queries**; multi-hop answered with t-norms plus beam search | FB15k-237 H@3: CQD-Beam average .290 vs Q2B .268; 3p .221 vs .186 (Table 2) | No, but it is the KG analogue of "train short, compose long" without retraining |
| **RotatE** (Sun et al., ICLR 2019) | **`t = h ∘ r`, h, r, t ∈ C^k, |r_i| = 1, ∘ = Hadamard product (Eq. 1). Composition: `r3 = r1 ∘ r2` (θ3 = θ1 + θ2) (§1; Lemma 3, §3).** The relation vector is a unit phasor vector and binding is element-wise phasor multiplication, so this is **FHRR binding**. The equations make composition **commutative**, and every relation is invertible (`r2 = conj(r1)` for inverses) | FB15k-237 MRR .338, WN18RR .476 (Table 5). "RotatE… is the only model that can model and infer all the three types of relation patterns" (§3.2) | No. It cannot represent father∘mother ≠ mother∘father (my derivation from Eq. 1; the commutativity criticism is in **DensE**, Neurocomputing 2022, abstract) |
| **ComplEx** (Trouillon et al., ICML 2016) | `Re(<r, h, conj(t)>)` | FB15k-237 MRR .247 (RotatE Table 5). Cannot infer composition (RotatE Table 2 / §3) | No |
| **HolE** (Nickel et al., AAAI 2016) | Circular correlation, "related to holographic models of associative memory". "Correlation, unlike convolution, is not commutative" | FB15k MRR .524, WN18 .938 (as reported in RotatE Table 4) | No. HRR-family and non-commutative |
| **Rotate3D** (CIKM 2020) | 3D rotations, non-commutative | **UNVERIFIED** (ACM page not opened) | — |
| **Learned RCC-8 / IA composition** | EpiGNN builds RCC-8 and Allen-IA benchmarks from the composition tables (App. Tables 5, 6) and learns them from paths (Fig. 4). STaR follow-up (Khalid et al., arXiv 2503.23487): **UNVERIFIED** (search summary only) | EpiGNN numbers are figures only (Fig. 4, Figs. 17–18: average 0.74 RCC-8, 0.77 IA at b=8, k=15) | Disjunctive tables, so a plain functional table fails |

**Takeaway for family 3:** none of these systems reports CLUTRR. Two points matter for us:
- **The RotatE/FHRR equivalence is exact for relation embeddings.** It is also commutative, which rules out pure-binding composition for kinship. Order-sensitive alternatives in the literature are matrix (Bilinear/GQE), correlation (HolE) and 3D-rotation or quaternion (DensE/Rotate3D) operators.
- **CQD is the KG precedent for training on short queries and evaluating on longer ones.**

---

### I.5. Answers to the two questions

#### I.5.1 Strongest baseline for "rules learned from few examples into a fixed-size readable store, answering long chains"

| Setting | Strongest published system | Exact numbers | Notes |
|---|---|---|---|
| Gold graph, our split (`089907f8`, train k=2,3) | **R5** (readable hash-table rule memory) | 4..10 hops: .98±.02 .99±.02 .98±.03 .96±.05 .97±.01 .98±.03 **.97±.03** (R5 Table 2) | Most direct conceptual competitor. Relation vocabulary should be checked (§1 caveat) |
| Gold graph, same split | **DSR-without-LM** (readable weighted table) | **98.81** averaged over k=4..10 (DSR-LM Table 7). CTP 95.57, GAT 39.05 | Averages only, no per-k numbers |
| Gold graph, larger training set (`9b2173cf`, 35k) | EpiGNN-mul | .85±.13 at k=10 (EpiGNN Table 13) | Neural; loses to R5 in this regime |
| Text input, learned table | **DSR-LM** | **60.98±2.64** (Table 5/6) | RoBERTa extractor |
| Text input, few examples, learned table | **Reason Popper-ly Exogenous** (200 2-hop examples, gold-patched) | 8–10 hops: GPT-5.4 69.53±3.45; Claude-4.6 Sonnet 66.79±6.70; Qwen-3.5:4B 46.72±4.35 (Table 1) | Split unnamed; gold supplementation weakens the "learned" claim |
| Text input, given table | **SatLM** | **80.1** SC (40 samples), **68.3** greedy, Codex; 1,042-ex test (Table 1) | Highest text number found on our split |
| Text input, given table | Faithful CoT | 71.9 SC, 58.9 greedy (Table 1) | Same 1,042 test |
| Noise variants, gold graph (k=2,3 only) | GAT | train=test: supporting 0.98±0.01, irrelevant 0.93±0.01, disconnected 0.96±0.01; clean→supporting 0.24±0.2 (CLUTRR Table 2) | No rule learner reports these splits |
| Noise variants, text (k=2,3 only) | LM+ASP (hand rules) | CLUTRR 1.0: 68.5 / 82.8 / 74.8 / 67.4 (clean/supp/irr/disc). CLUTRR 1.3 self-generated: 97.0 / 84.0 / 92.0 / 90.0 (Table 3); App. Table 7 has per-train/test-pair values | Authors blame CLUTRR 1.0 label errors |

**Recommended baselines to beat or match:**
- **R5** (gold graph).
- **SatLM-style "LLM + hand table"** run with *our* reader LLM (text). Their Codex numbers are not like-for-like.
- **A table learned from the same k=2,3 chains** (our own control). The literature (DSR-LM Table 6) predicts it ties.

#### I.5.2 Published head-to-head of learned table/rule set vs neural or VSA store on CLUTRR?

**Table vs neural:** yes, several (graph input):
- R5 Tables 2–3 (hash-table rules vs CTP, GNTP, GAT, GCN, LSTM and others).
- DSR-LM Table 7 (weighted table vs CTP vs GAT).
- EpiGNN Table 13 plus App. G text (R5 beats EpiGNN in the small-data k=2,3 regime).
- HtT Table 1 (LLM-induced textual rule library vs EdgeTransformer: ET better on average, 71.8 vs 64.3; equal at 10 hops, 36.0).
- DSR-LM Table 6 (learned vs hand table: 60.98 vs 61.34).

**Table vs VSA/HRR/FHRR store on CLUTRR: no precedent found.** Queries run:
- "vector symbolic architecture CLUTRR kinship relation composition"
- "holographic reduced representations OR hyperdimensional computing relational reasoning kinship rule learning compositional generalization"
- "tensor product representation CLUTRR systematic generalization"
- "learned composition table versus neural model CLUTRR rule-based baseline 'composition table'"
- "CLUTRR 'rule table' OR 'lookup table' baseline composition kinship LLM extraction symbolic 2025"
- "vector symbolic architecture knowledge graph multi-hop path reasoning relation composition FHRR binding"
- "CLUTRR neurosymbolic LLM rule learning 2024 2025 systematic generalization longer chains"
- "CLUTRR length generalization LLM symbolic solver 2025 arXiv kinship k=10 accuracy rule induction"

Hits were GHRR (arXiv 2405.09689), PathHD (KGQA with HDC) and TPR/AID papers. None evaluates on CLUTRR (search-level check only; I did not open those papers).

#### I.5.3 Where the literature contradicts or qualifies our numbers and claims

1. **"DSR-LM 61.0 is the best published text-input number" is false.** SatLM reports 80.1 (SC) and 68.3 (greedy); Faithful CoT reports 71.9 (SC). All three use our split, on a 1,042-example test version.
2. **Gold-facts 0.975 is at the level of existing table learners, not above it.** R5 reaches .97 at k=10 and DSR-without-LM 98.81. Report ours on k=4..10 and with the same relation vocabulary to compare.
3. **A ceiling below 1.0 on gold facts is expected.** CLUTRR has label and graph errors (LM+ASP: 16/400 wrong in CLUTRR 1.3) and non-deterministic compositions (HtT; Faithful CoT annotators).
4. **The reader, not the store, is the bottleneck.** Popper-ly finds the same for its LLM-plus-table pipeline. Their small-model numbers (Qwen-3.5:4B at 46.72 for 8–10 hops) are below our Qwen3-8B (0.645 at k≥7), but the setups differ.
5. **Bracketing-free pair-wise folding is not new.** R5 picks adjacent pairs to reduce with MCTS; NCRL "recursively combin[es] predicates"; Faithful CoT and SatLM fold with a table. The novelty has to be the store (capacity, readability, noise, disjunction), not the fold.

---

### I.6. References (all opened unless marked)

- Arakelyan, Daza, Minervini, Cochez (2021). Complex Query Answering with Neural Link Predictors. ICLR 2021. arXiv:2011.03459.
- Badreddine, d'Avila Garcez, Serafini, Spranger (2022). Logic Tensor Networks. Artificial Intelligence 303:103649. arXiv:2012.13635.
- Bergen, O'Donnell, Bahdanau (2021). Systematic Generalization with Edge Transformers. NeurIPS 2021. arXiv:2112.00578.
- Chen, Ma (2026). Reason Popper-ly: Patching In-Context Reasoning with Inductive Logic Programming. NeSy 2026 (per arXiv). arXiv:2607.23019 (HTML opened).
- Cheng, Ahmed, Sun (2023). Neural Compositional Rule Learning for Knowledge Graph Reasoning (NCRL). ICLR 2023. arXiv:2303.03581.
- CLUTRR/v1 dataset card, Hugging Face. https://huggingface.co/datasets/CLUTRR/v1
- Dong, Mao, Lin, Wang, Li, Zhou (2019). Neural Logic Machines. ICLR 2019. arXiv:1904.11694.
- Evans, Grefenstette (2018). Learning Explanatory Rules from Noisy Data (∂ILP). JAIR. arXiv:1711.04574.
- Gontier, Sinha, Reddy, Pal (2020). Measuring Systematic Generalization in Neural Proof Generation with Transformers. NeurIPS 2020. arXiv:2009.14786.
- Guu, Miller, Liang (2015). Traversing Knowledge Graphs in Vector Space. EMNLP 2015. arXiv:1506.01094.
- Hamilton, Bajaj, Zitnik, Jurafsky, Leskovec (2018). Embedding Logical Queries on Knowledge Graphs (GQE). NeurIPS 2018. arXiv:1806.01445.
- Khalid, Schockaert (2025). Systematic Relational Reasoning with Epistemic Graph Neural Networks (EpiGNN). ICLR 2025. arXiv:2407.17396.
- Khalid, Nourollah, Schockaert (2025). Benchmarking Systematic Relational Reasoning with Large Language and Reasoning Models. arXiv:2503.23487. **UNVERIFIED** (not opened).
- Kirtania, Gupta, Radhakrishna (2024). Logic-LM++. arXiv:2407.02514 (venue not listed).
- Li, Huang, Naik (2023). Scallop: A Language for Neurosymbolic Programming. arXiv:2304.04812 (PLDI 2023 venue **UNVERIFIED**).
- Lu, Hu, Lin (2022). DensE: An Enhanced Non-commutative Representation for Knowledge Graph Embedding with Adaptive Semantic Hierarchy. Neurocomputing 476. arXiv:2008.04548 (abstract opened).
- Lu, Liu, Mills, Jui, Niu (2022). R5: Rule Discovery with Reinforced and Recurrent Relational Reasoning. ICLR 2022. arXiv:2205.06454.
- Lyu, Havaldar, Stein, Zhang, Rao, Wong, Apidianaki, Callison-Burch (2023). Faithful Chain-of-Thought Reasoning. IJCNLP-AACL 2023. arXiv:2301.13379.
- Minervini, Bošnjak, Rocktäschel, Riedel, Grefenstette (2020). Differentiable Reasoning on Large Knowledge Bases and Natural Language (GNTP). AAAI 2020. arXiv:1912.10824.
- Minervini, Riedel, Stenetorp, Grefenstette, Rocktäschel (2020). Learning Reasoning Strategies in End-to-End Differentiable Proving (CTP). ICML 2020. arXiv:2007.06477.
- Nickel, Rosasco, Poggio (2016). Holographic Embeddings of Knowledge Graphs. AAAI 2016. arXiv:1510.04935.
- Olausson, Gu, Lipkin, Zhang, Solar-Lezama, Tenenbaum, Levy (2023). LINC. EMNLP 2023. arXiv:2310.15164.
- Pan, Albalak, Wang, Wang (2023). Logic-LM. Findings of EMNLP 2023. arXiv:2305.12295.
- Qu, Chen, Xhonneux, Bengio, Tang (2021). RNNLogic. ICLR 2021. arXiv:2010.04029.
- Ren, Hu, Leskovec (2020). Query2box. ICLR 2020. arXiv:2002.05969.
- Ren, Leskovec (2020). Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs (BetaE). NeurIPS 2020. arXiv:2010.11465.
- Rocktäschel, Riedel (2017). End-to-End Differentiable Proving (NTP). NeurIPS 2017. arXiv:1705.11040.
- Sadeghian, Armandpour, Ding, Wang (2019). DRUM. NeurIPS 2019. arXiv:1911.00055.
- Sinha, Sodhani, Dong, Pineau, Hamilton (2019). CLUTRR. EMNLP-IJCNLP 2019. arXiv:1908.06177.
- Sinha, Sodhani, Pineau, Hamilton (2020). Evaluating Logical Generalization in Graph Neural Networks (GraphLog). arXiv:2003.06560 (venue not listed).
- Stein, Naik, Velingker, Naik, Wong (2025). Neuro-Symbolic Programming in the Age of Foundation Models: Pitfalls and Opportunities. PMLR-format PDF (https://nvelingker.github.io/res/papers/neurosymbolic_pitfalls_and_opportunities.pdf). The arXiv version is 2505.24874 under a different title, not opened.
- Sun, Deng, Nie, Tang (2019). RotatE. ICLR 2019. arXiv:1902.10197.
- Trouillon, Welbl, Riedel, Gaussier, Bouchard (2016). Complex Embeddings for Simple Link Prediction. ICML 2016. arXiv:1606.06357.
- Wang, Wei, Choi, Ren (2024). Symbolic Working Memory Enhances Language Models for Complex Rule Application. arXiv:2408.13654 (HTML opened).
- Xu, Fei, Pan, Liu, Lee, Hsu (2024). Faithful Logical Reasoning via Symbolic Chain-of-Thought (SymbCoT). ACL 2024. arXiv:2405.18357.
- Yang, Ishay, Lee (2023). Coupling Large Language Models with Logic Programming for Robust and General Reasoning from Text. Findings of ACL 2023. arXiv:2307.07696.
- Yang, Song (2020). Learn to Explain Efficiently via Neural Logic Inductive Learning (NLIL). ICLR 2020. arXiv:1910.02481.
- Yang, Yang, Cohen (2017). Differentiable Learning of Logical Rules for Knowledge Base Reasoning (Neural LP). NeurIPS 2017. arXiv:1702.08367.
- Ye, Chen, Dillig, Durrett (2023). SatLM: Satisfiability-Aided Language Models Using Declarative Prompting. NeurIPS 2023. arXiv:2305.09656.
- Zhang, Huang, Li, Naik, Xing (2023). Improved Logical Reasoning of Language Models via Differentiable Symbolic Programming (DSR-LM). Findings of ACL 2023. arXiv:2305.03742.
- Zhu, Xue, Chen, Zhou, Tang, Schuurmans, Dai (2023). Large Language Models can Learn Rules (HtT). arXiv:2310.07064 (venue not listed on arXiv).
- Not opened: AMIE (Galárraga et al., WWW 2013); Rotate3D (Gao et al., CIKM 2020); PathHD; GHRR (arXiv 2405.09689). Cited only as search hits.

---

# Part II. VSA/HDC reasoning, world-state agents, novelty search

Draft, 2026-11-06. Scope: (1) VSA/HDC reasoning 2019–2026; (2) agents that keep a world-state memory and reason over it, including impact and what-if questions; (3) novelty search for candidate claims (a)–(d).

**Verification legend.** [V] means I opened the primary source (arXiv abs/html/pdf, proceedings PDF, or publisher metadata via Crossref) and took the numbers from it. [V-meta] means only the bibliographic record was confirmed (Crossref or publisher), not the content. UNVERIFIED means I could not open it. Every number below carries its table or figure number from the opened source. Vendor-authored papers are labelled VENDOR. Items listed as "already verified" in the brief (Kanerva 2009; Kleyko Part II §3.1.2; Statler; Host-Materialized State; PD-SSM; Kumar 2606.24948; VET-Bench) are cited, not re-derived.

---

### II.0. Decision-relevant findings first

1. **A composition table is already published prior art on CLUTRR.** R5 (ICLR 2022) keeps a "dynamic rule memory": a dictionary keyed by the rule body (r1, r2) that stores the head and a score. Trained on k∈{2,3}, it scores .97±.03 at 10 hops (Table 2). NCRL and CTP are close behind (§2). So "a VSA rule memory composes k=10 exactly" does not count as novel on its own. Exact composition from gold facts is the result a plain table is expected to give, and R5 has already published it.
2. **The closest VSA prior for learned, readable rule stores is IBM's Learn-VRF and ARLC on Raven's matrices, not relational QA.** ARLC uses 480 parameters and reaches 92.4–92.6% on I-RAVEN (Table 2, NeSy'24). Its rules can be hand-programmed and then trained, and the authors say training "extend[s]" the programmed knowledge rather than erasing it. Learn-VRF's rules are extractable as readable formulas (Table 4). Learn-VRF also reaches 98.8–100% on unseen attribute–rule pairs, against 14.8–75.0% for an MLP (Table 3). That OOD rule sharing is the one VSA-specific advantage with published evidence that a table keyed on symbols would not obviously reproduce.
3. **The impact/what-if benchmark already exists, and current memory systems fail on it.** MEME (arXiv 2605.12477, May 2026) has a "Cascade" task (upstream update plus if–then rule, so the dependent value changes) and an "Absence" task (no rule, so the answer should be "uncertain"). Among the six memory systems (Mem0 and Graphiti included) Cascade is ≤0.06 and Absence ≤0.05. In-context gpt-4.1-mini scores 0.03/0.04 and in-context Sonnet 4.6 0.05/0.35 (Table 2). MQuAKE (EMNLP'23) and MemoryAgentBench's multi-hop FactConsolidation (≤28%, Table 3) show the same failure. Claim (b) cannot be "first to ask impact questions over updated state". It can be "first fixed-size state that answers MEME-style Cascade/Absence", and MEME is the external benchmark to run.
4. **When the dependency graph is explicit, a plain graph query or tool-using LLM already solves impact questions.** Tacheny (arXiv 2601.07342) reports 100% impact accuracy for Claude Haiku 3.5 with graph tools on 10 synthetic infrastructure cases (Table 1). AlMahri et al. (2601.09680) use deterministic Neo4j/BFS traversal for supplier exposure. VSA therefore cannot win on traversal. Its case has to rest on (i) rules learned from a few examples instead of a hand-written dependency schema, (ii) robustness when an LLM writes the state from noisy logs, (iii) abstention, or (iv) hardware.
5. **The closest architectural ancestor of Readable Memory is Fast Weight Memory** (Schlag et al., ICLR 2021). It keeps fixed-size tensor-product fast weights, uses a delta-rule "remove old association, write new" update, and chains reads for multi-hop inference. It scores 96.75% QA accuracy on catbAbI with 694k weights (Table 1). It must be cited. The differences are that FWM is trained end-to-end on large data and is neither LLM-written nor human-readable.
6. **No VSA/HDC paper on CLUTRR, and none on bAbI, was found** (queries in §5). The bAbI precedent is TPR. Lee et al. (2016) reach near-100% with hand-designed encodings (Table 5). TPR-RNN reaches 1.34% mean error (Table 1).
7. **Edge and neuromorphic claims have precedent only for perception factorization and KG completion**: Loihi resonator with a 171× energy-delay-product advantage over CPU (Renner et al., NMI 2024; arXiv v4 Fig. 6C), in-memory PCM factorizer (Nat. Nano 2023), FPGA HDC knowledge-graph completion (HDReason). None covers LLM-written operational state with rule inference. The honest baseline for an edge claim is a key-value graph on a microcontroller, not an LLM.

---

### II.1. VSA/HDC reasoning, 2019–2026

| Paper | Problem / benchmark | Headline numbers (source) | Rules given or learned; from how many examples | What fails | What it does that a lookup table / graph query could not |
|---|---|---|---|---|---|
| Kleyko, Rachkovskij, Osipov, Rahimi, survey Parts I & II (ACM CSUR 2022/2023) [V] | Field survey | n/a. Part II §3.1.3.6 "general-purpose rule-based and logic-based inference with HVs" lists production systems, KB clauses with deductive inference, and HRR modus ponens/tollens, all as demonstrations without benchmark numbers | Rules hand-given in every cited inference demo | Part II §3.1.2 open problems (already verified): holistic mapping not shown to scale to large episodes; no competition among alternative mappings | Survey; no claim |
| NVSA, Hersche et al., Nat. Mach. Intell. 5:363 (2023) [V] | RAVEN, I-RAVEN, PGM | 87.7% RAVEN, 88.1% I-RAVEN avg. (Fig. 4; Ext. Tables II–III). Backend with ground-truth attributes, 3×3 grid: 96.89% in 63.2 min vs unrestricted PrAE 97.50% in 15,408.5 min, i.e. 244× faster (Table I). PGM 68.3% | **Given**: the four RPM rule types are hand-coded as VSA bind/unbind formulas (Learn-VRF calls them "hard-coded") | End-to-end 3×3 grid only 54.3% on I-RAVEN (Learn-VRF Table 2). Some position rules handled outside VSA (in the PMF domain) | **Search in superposition**: scores all n³ rule realizations with O(n) cleanup instead of enumerating them. This is a real advantage over table enumeration when rule realizations are combinatorial |
| Learn-VRF, Hersche et al., NeurIPS'23 MATH-AI workshop (arXiv 2401.16024) [V] | I-RAVEN | 84.1% avg. at 50 epochs and 81.0% after **one pass** through the data, with 5k params (Table 2). OOD unseen rule–attribute pairs: 98.8–100% vs GPT-3 65.8–94.7% and MLP 14.8–75.0% (Table 3) | **Learned** soft-selection of VSA rule formulas. Readable: dominant formulas extracted and simplified to "arithmetic plus", "progression", and so on (Table 4) | Fixed template can only express sums and differences, not logical rules on position bit-codes. 3×3 grid 50.9% vs a 73.3% upper bound (App. D) | One rule representation shared across attributes, so a rule learned on some attributes transfers to an unseen attribute (Table 3). A table keyed on (attribute, rule) has no entry for the unseen pair |
| ARLC, Camposampiero et al., NeSy 2024 (arXiv 2406.19121) [V] | I-RAVEN | 92.6±0.2 (programmed then trained), 92.4±1.5 (learned), 92.4 (programmed only), with **480 params**. Baselines: NVSA 88.1, MLP 87.1 (300k), GPT-3 86.5, Learn-VRF 79.5 (Table 2). OOD 98.6–100% (Table 4) | Learned (R=5 rule slots, 12-term template) **or hand-programmed then fine-tuned**. Trained on the 2×2 constellation only | Progression and arithmetic on position/number "currently impossible to learn". Assumes perfect perception. Rule selection "still not optimal" | Programmable plus trainable with knowledge retained: a precedent for editable learned rules (claim c) |
| Hersche, Camposampiero et al., arXiv 2412.05586 (2024) [V] | I-RAVEN, new I-RAVEN-X (3×10 grid, value range up to 1000) | Task accuracy at range 1000: ARLC programmed 99.7, ARLC learned 90.6, GPT-4 76.6, Llama-3-70B 74.2 (Table 5). Arithmetic-rule accuracy: ARLC programmed 99.5, ARLC learned 61.6, GPT-4 8.4, Llama-3 0.4 (Table 6) | Programmed (R=4) vs learned (25 epochs) | Learned rules degrade with range (arithmetic 99.9 → 61.6). LLMs given oracle perception and only 500 samples | Exact arithmetic over large value ranges via fractional-power encoding |
| Resonator networks 1 & 2, Frady/Kent/Olshausen/Sommer, Neural Comput. 32(12) (2020) [V; venue V-meta] | Factoring products of codevectors (decoding bound structures) | Operational capacity roughly two orders of magnitude above ALS and gradient methods (Fig. 3). **Capacity M_max grows quadratically in N** for F = 2..7 factors (Fig. 4). Renner et al. (2024, Methods) cite about 10⁸ combinations at N=12,000 | n/a (codebooks given) | Not guaranteed to converge. Less stable than Hopfield nets because of "percolated noise" (§6.1) | Factorizes a bound product without enumerating M = ∏Dᶠ combinations. Relevant to multi-hop decode capacity, consistent with Kumar 2606.24948 that capacity is the bottleneck |
| Ganesan et al., "Learning with HRR", NeurIPS 2021 [V] | Extreme multi-label classification | Projection step improves HRR retrieval >100× (abstract). Output-layer parameters cut by 59–99% (Table 2). Training speedup up to 6.3× (Table 3) | n/a | Amazon-670K: P@1 19.9 vs 34.6 for the FC baseline (Table 1); HRR runs out of discriminability | Makes HRR stably differentiable. Not a reasoning paper |
| Hrrformer, Alam et al., ICML 2023 (arXiv 2305.19534) [V, abstract] | Long Range Arena, malware | O(TH log H) time; up to 280× faster training on LRA; about 10× fewer epochs (abstract) | n/a | LRA accuracy only "near state-of-the-art" | Attention via HRR binding. Not reasoning |
| LARS-VSA, Mejri et al., arXiv 2405.14436 (2024) [V] | Relational-bottleneck tasks (order, SET, ABA), sorting, math | >80% on pairwise order with 200 samples. Up to 17× less memory than Abstractor and HD attention about 25× faster (Fig. 12). Exact accuracies only in figures | Learned from limited data (10–6,000 samples) | Possible catastrophic forgetting in online training, not studied (§8) | VSA binding as a relational bottleneck |
| HolE, Nickel/Rosasco/Poggio, AAAI-16 (arXiv 1510.04935) [V, abstract] | KG link prediction | "Outperforms state-of-the-art" (abstract; numbers not extracted) | Learned embeddings | n/a | Circular correlation (HRR) as KG scorer. The pre-2019 anchor for VSA-style KG embedding |
| HDReason, Chen et al., arXiv 2403.05763 (2024) [V, abstract] | KG completion on FPGA | 10.6× speedup and 65× energy efficiency vs RTX 4090; 4.2× / 3.4× vs FPGA GCN, with "similar accuracy" (abstract) | Learned | Accuracy not quantified in abstract | Hardware efficiency |
| PathHD, Liu et al. (Imani lab), arXiv 2512.09369 (2025/26) [V] | KGQA: WebQSP, CWQ, GrailQA | WebQSP Hits@1 86.2 vs RoG 85.7. CWQ 71.5 vs GoG 75.2. GrailQA F1 86.7 (Table 2). One LLM call vs about 12 for RoG. Claims 40–60% lower latency (abstract) | **No learning**: GHRR block-diagonal, non-commutative binding encodes relation paths. Paths enumerated by BFS | Depends on top-K path coverage; Freebase only; theory is a sketch | VSA as a cheap order-aware path scorer replacing per-candidate LLM scoring. **Closest VSA + LLM multi-hop QA prior** |
| Lee, He, Yih, Gao, Deng, Smolensky, "Reasoning in Vector Space", arXiv 1511.06426 (ICLR'16 per DBLP link; venue not confirmed) [V] | bAbI 20 tasks | Test accuracy 100% on all categories except C5 99.8% and C16 99.5% (Table 5) | **Given**: hand-designed TPR encodings plus a parser with 100% accuracy; "common-sense" rules built in | Errors attributed to dataset ambiguity | TPR algebra replaces the reasoner. The bAbI precedent for VSA-style reasoning |
| TPR-RNN, Schlag & Schmidhuber, NeurIPS 2018 (arXiv 1811.12143) [V] | bAbI-10k | All-task mean error 1.34±0.52, failures (>5%) 0.86 (Table 1) | Learned end-to-end from 10k examples per task | Task 19 unstable across seeds (Table 2 note) | Third-order TPR memory |
| **Fast Weight Memory**, Schlag, Munkhdalai, Schmidhuber, ICLR 2021 (arXiv 2011.07831) [V] | catbAbI (concatenated bAbI stories), meta-RL, LM | QA acc 96.75±0.05% vs TXL 87.66, MNM 88.97, LSTM 80.88. FWM has 694k weights (Table 1) | Learned end-to-end | n/a in this setting | **Fixed-size tensor fast weights with a delta-rule replace (retrieve old value, subtract, write new) and chained multi-hop reads.** Architecturally closest to Readable Memory |
| Emruli & Sandin, Cogn. Comput. 6:74–88 (2014) [V-meta; content UNVERIFIED, 403 on full text] | Analogical mapping | A search snippet says it generalizes after a few examples. UNVERIFIED | Mapping vectors learned from examples, stored in SDM | UNVERIFIED | Few-example mapping-vector learning. Extends Kanerva 2009 |

**Notes.**
- **Rule provenance.** In the VSA reasoning literature, rules are either hand-coded (NVSA, Lee 2016, the Kleyko §3.1.3.6 demos) or learned within a fixed formula template from thousands of examples (Learn-VRF, ARLC). Learn-VRF's single pass is the closest thing to sample efficiency. No VSA paper learns relational composition rules from a handful of chains.
- **CLUTRR and bAbI coverage.** None of these works evaluates on CLUTRR. On bAbI the evidence is TPR, not VSA.
- **Capacity.** The resonator result (capacity ∝ N², Fig. 4) gives the quantitative ceiling for decoding multi-factor bindings. Any "fixed-size" claim should state N and the size of the combination space it implies.

### II.2. Rule learning on CLUTRR (the baselines for claim a)

| Paper | Setup | Headline numbers | Rules: form; # examples | What fails |
|---|---|---|---|---|
| R5, Lu et al., ICLR 2022 (arXiv 2205.06454) [V] | CLUTRR train k∈{2,3} → test 4..10; GraphLog | Trained k∈{2,3}: .98 / .99 / .98 / .96 / .97 / .98 / .97 at 4–10 hops. CTP-L .89 and GAT .45 at 10 hops (Table 2). Trained k∈{2,3,4}: .98 at 10 hops (Table 3). Low-data GraphLog worlds: R5 accuracy .665 / .575 / .446 with rule recall 1.0 / 1.0 / .70, vs CTP .425 / .190 / .359 (Table 5) | **Dictionary rule memory**: body (r1, r2) → head with a score, pruned when the score falls below a threshold. Readable Horn clauses. Trained on the full CLUTRR training set (RL plus MCTS) | Needs an RL/MCTS training budget; the low-data regime is still hard (Re 2: .446) |
| NCRL, Cheng et al., ICLR 2023 (arXiv 2303.03581) [V] | CLUTRR train 2–4 → test 5–10; KG link prediction | 10 hops: NCRL .97±.02, R5 .98, CTP .90–.95, GAT .68, GCN .41 (Table 3). WN18RR MRR .67 vs RotatE .47 | Learned composition by recurrent attention. CLUTRR train set 15,083 examples (Table 12) | Interpretability only qualitative; not SOTA on FB15k-237 |
| HtT ("LLMs can learn rules"), Zhu et al., arXiv 2310.07064 [V] | CLUTRR (symbolic), arithmetic, List Functions | GPT-4 5-shot CoT 50.6 → 64.3 with HtT. GPT-3.5 21.2 → 37.6, or 48.4 with GPT-4-induced rules. Supervised EdgeTransformer 71.8 (CLUTRR table) | **Textual rule library** filtered by occurrence ≥k and confidence ≥p (CLUTRR: k=2, p=0.7 for GPT-4). Induced from 2,000 training examples; capped at about 500 rules by context; XML-tagged hierarchy | Needs a base model ≥ about 20% on training examples; library limited by context; about 500 examples needed for a significant CLUTRR gain (log-linear in examples) |

**Implication.** On gold facts, a composition table (R5's memory, or simply counting (r1, r2) → r3 over the k=2 examples) is the baseline to beat. R5 reports how its accuracy and rule recall degrade with less data (Table 5). That gives a measured low-data curve to compare against in a sample-efficiency study.

### II.3. Agents with a world-state memory that also reason over it

#### 3a. State memories and benchmarks

| Paper | Problem / benchmark | Headline numbers (source) | State representation; rules | What fails | Beyond a lookup / graph query? |
|---|---|---|---|---|---|
| Statler, Yoneda et al. (ICRA 2024 per brief; arXiv 2306.17840v4) [V] | Simulated pick-and-place, disinfection, relative weight; real UR5 | Episode success Statler vs CaP: 0.50 vs 0.00, 0.40 vs 0.00, 0.55 vs 0.00 (Table I). Temporal queries 83/65/55% vs 31/5/0% (Table II). Real robot 40% vs 20% | JSON-like state written by a writer LLM and read by a reader LLM. Rules (for example dirtiness propagation) in the prompt | Hallucinated conditions; misapplied dirtiness rule; writer errors propagate. 20 episodes per domain. **No what-if queries** | No: the state is text and the LLM does the reasoning |
| AriGraph, Anokhin et al., arXiv 2407.04363v3 [V] | TextWorld (treasure, cooking, cleaning), NetHack, MuSiQue/HotpotQA | Normalized score 1.0 / 1.0 / 0.79 vs Reflexion 0.93 / 1.0 / 0.27 and RAG 0.33 / 0.36 / 0.39 (Table 4). MuSiQue EM 45.0 (GPT-4) vs HOLMES 48.0 (Table 2) | Semantic KG plus episodic memory. LLM detects and replaces outdated triplets | Graph quality not measured; weaker on Cleaning, where filtering outdated info matters; static QA does not test updates | KG update; reasoning still done by the LLM |
| HippoRAG, Gutiérrez et al., NeurIPS 2024 (arXiv 2405.14831) [V, abstract] | Multi-hop QA | Up to 20% over SOTA; single-step retrieval 10–30× cheaper and 6–13× faster than IRCoT (abstract) | OpenIE KG plus Personalized PageRank | Static corpus | PPR spreads activation over the graph; no state updates |
| Zep / Graphiti, Rasmussen et al., arXiv 2501.13956 [V] **VENDOR** | DMR, LongMemEval_s | DMR 94.8 vs MemGPT 93.4 vs full context 94.4 (Table 1). LongMemEval 71.2 vs 60.2 full context with gpt-4o (Table 2). Knowledge-update 76.9 → 74.4 with gpt-4o-mini, 78.2 → 83.3 with gpt-4o (Table 3) | Bi-temporal KG. A contradicting edge sets t_invalid on the old one | Latency comparison not like-for-like (network); single-session-assistant drops (94.6 → 80.4) | Temporal invalidation; no rule inference |
| Mem0 / Mem0ᵍ, Chhikara et al., arXiv 2504.19413 [V] **VENDOR** | LOCOMO | J: Mem0 66.88, Mem0ᵍ 68.44, full context 72.90, Zep 65.99, OpenAI 52.90 (Table 2). Multi-hop: Mem0 51.15, Mem0ᵍ 47.19 (Table 1) | LLM ADD/UPDATE/DELETE/NOOP; graph variant marks relations invalid | The graph variant does not help multi-hop; full context still best; 10 conversations | No |
| SayPlan, Rana et al., CoRL 2023 (arXiv 2307.06135) [V] | 3D scene graphs, 3 floors / 36 rooms | Semantic search success with GPT-4: 86.7% simple, 73.3% complex; GPT-3.5 0–6.6% (Table 1). Long-horizon executability 86.6% vs LLM-as-Planner 13.3% (Table 3). Collapsed graph 878 vs 6,731 tokens (Table 2) | Hierarchical 3DSG explored by the LLM through expand/contract; scene-graph simulator verifies | Negation ("office without a cabinet"), counting and distance reasoning; 6.67% hallucinated nodes | The simulator verifier is symbolic; reasoning is the LLM's |
| ConceptGraphs, Gu et al. (arXiv 2309.16650; ICRA 2024 UNVERIFIED) [V] | Open-vocabulary 3D scene graphs | Node precision 0.71 avg., edge precision 0.88 (CG, Table I) | Objects plus spatial edges; GPT-4 picks the object | Caption errors, missed thin objects | No |
| Kim & Schuster, ACL 2023 (arXiv 2305.02363) [V] | Boxes entity tracking | Only text-davinci-003 beats the strong baseline; it gets all box contents right after 7 operations in >25% of cases (Fig. 2). Reanalysis of Li et al.: a probe scores 86.8% on trivial cases but 3.1% on non-trivial ones | In-context; T5 can be fine-tuned | Accuracy falls with the number of operations | Benchmark |
| **MEME**, Jung et al., arXiv 2605.12477 (May 2026) [V] | Six tasks including **Cascade** (if–then rule plus upstream update, so the dependent value changes) and **Absence** (no rule, so uncertain) | Table 2 (gpt-4.1-mini unless noted), Cascade / Absence: MD-flat 0.06/0.05; text-embedding-3-small 0.04/0.00; Mem0 0.03/0.00; BM25 0.02/0.00; Graphiti 0.02/0.01; Karpathy Wiki 0.01/0.02; in-context gpt-4.1-mini 0.03/0.04; in-context Sonnet 4.6 0.05/0.35. Only ingestion-time propagation by Claude Opus 4.7 reaches Cascade 0.32 / Absence 0.59 (20-episode subset) at about 70× cost | Rules given in text; nothing learned | Retrieval surfaces the pre-change value. Deeper retrieval, stronger LLMs and prompt tuning do not fix it | **This is the target task for claim (b) and (c)** |
| StateMemBench, Fan et al., arXiv 2608.19652 (Aug 2026) [V, abstract] | 234 multi-session evolving-state scenarios | Current-state accuracy 0.205 → 0.363 (DeepSeek-V4-Flash) with their StateMem. As a wrapper, +32 to +67 points over six backends (abstract) | State-first memory tracking supersession and dependencies | Baselines answer with superseded state | Benchmark plus method |
| MemoryAgentBench, Hu, Wang, McAuley, arXiv 2507.05257 [V] | FactConsolidation single-hop (SH) and multi-hop (MH) after conflicting updates | FC-MH: GPT-5-mini 28.0, GPT-4o 5.0, HippoRAG-v2 5.0, MemGPT 3.0, Mem0 2.0 (Table 3). "All methods fail on the multi-hop situation" | n/a | Multi-hop over updated facts | Benchmark |
| MQuAKE / MeLLo, Zhong et al., EMNLP 2023 (arXiv 2305.14795) [V] | Multi-hop questions after knowledge edits | GPT-J edited by ROME/MEMIT: edit-wise 88.3/96.2 but multi-hop 7.4/7.0 (Table 3). MeLLo multi-hop with GPT-3.5: 52.1 at k=1, 30.9 with all edits (Table 5) | Edits held in external memory; LLM decomposes the question | Accuracy drops as the number of edits grows | Update propagation through multi-hop chains; external memory beats parametric editing |
| WIQA, Tandon et al., EMNLP 2019 (arXiv 1909.04739) [V] | 40,695 "what if" questions over 379 procedures (Table 1) | BERT 73.80 vs human 96.33. In-paragraph 79.68 vs out-of-paragraph 56.13 (Table 2) | Crowdsourced influence graphs | Indirect (multi-hop) effects are hardest (Fig. 3) | Classic what-if QA, but over static text |

#### 3b. Impact / root-cause analysis over operational dependency state (AIOps, supply chain)

| Paper | Problem | Headline numbers | Mechanism | What fails | Relevance |
|---|---|---|---|---|---|
| OpenRCA, Xu et al., ICLR 2025 [V] | Root cause from 68 GB of telemetry, 335 failures | Best: Claude 3.5 with RCA-agent solves 11.34% (abstract) | LLM agent over raw telemetry | "Only the simplest cases" | Shows that RCA from raw logs is hard; the dependency structure is implicit |
| Praxis, Cui et al., DSN (accepted; arXiv 2512.22113v2) [V] | Code/config-caused cloud incidents; 30 scenarios (Table II) | RCR Pass@1 61.5% and RCI 73.9% (gpt-5-codex) vs SRE-Agent 0.00 / 11.43 (Table III). Tokens per diagnosis 166k vs 885k (Table V) | LLM traverses a service dependency graph plus program dependence graphs | Missing graph edges stop traversal; infra faults near 0 | Graph-guided LLM; graph given by static analysis |
| Tacheny, arXiv 2601.07342 (Jan 2026) [V] | RCA plus "which services and parties are impacted if resource r fails" over a multi-layer telecom/DC topology | Impact accuracy: Claude Haiku 3.5 100%, GPT-OSS-120B 99%, Llama-3.1-8B 86.1% (Table 1); 10 synthetic cases × 10 runs | LLM with MCP graph tools (GetImpactedServices, …) | No time modelling; Llama hallucinates IDs (7.3%); no non-LLM baseline | **Strongest evidence that the explicit-graph baseline is near-perfect on impact questions** |
| AlMahri, Xu, Brintrup, arXiv 2601.09680 (Jan 2026) [V] | Supply-chain disruption exposure, Tier 1–4 | KG-query F1 0.980; disruption monitoring 0.991 (30 synthetic scenarios). Network-impact narrative accuracy only 0.317 | Neo4j (6,596 nodes) plus deterministic BFS tools; GPT-4o writes the Cypher | Errors cascade from entity extraction; parts not derived from graph | Deterministic traversal plus LLM front-end is the industrial baseline |
| Heus, Bookstaber, Sharma, ICAIF 2025 workshop (arXiv 2510.01115) [V, abstract] | Supply-chain risk narratives | No numbers in abstract | Centrality-ranked graph paths fed to an LLM | n/a | Context only |

### II.4. Novelty check for candidate claims (a)–(d)

#### (a) Rules learned from a few examples into a fixed-size, readable store
- **Closest prior:**
  - Learn-VRF and ARLC: a VSA rule store learned from data, with readable formulas (Learn-VRF Table 4), 480 parameters, and programmable-then-trainable rules (ARLC). Domain is Raven's matrices, trained on a full constellation; Learn-VRF needs only one pass (81.0%).
  - R5: readable dictionary rule memory on CLUTRR, .97 at 10 hops from k∈{2,3} (Table 2).
  - HtT: textual rule library with confidence filtering, induced from 2,000 examples.
  - Kanerva 2009 (already verified): a one-shot mapping vector. Emruli & Sandin 2014: mapping vectors learned from examples, content UNVERIFIED.
  - WorldCoder (NeurIPS 2024): a readable code world model learned in ≤100 actions on MiniGrid UnlockPickup, where PPO failed after 3×10⁸ (App. A).
- **Verdict:** partly pre-empted. "Fixed-size, readable, learned VSA rules" exists (Learn-VRF/ARLC). "Learned composition rules generalizing k=2,3 → 10" exists (R5, NCRL, CTP). **No precedent was found for learning relational composition or containment rules from a handful (≤10) of chains into a fixed-size VSA store and testing them over an LLM-written operational state.**
- **Strongest baseline to beat:** a count-based composition table. Collect (r1, r2) → r3 from the same few examples, with support and confidence (R5-style dictionary or HtT-style thresholds). Then R5/NCRL at equal data. Then HtT with a frontier LLM as the "no new machinery" baseline.
- **Where VSA could earn its place:** OOD rule sharing across relation types (Learn-VRF Table 3 is the precedent). A table keyed on relation symbols cannot fill an unseen (relation, rule) pair; a typed rule language could, so that baseline is needed too.

#### (b) Reasoning over a continuously updated state (impact / what-if)
- **Closest prior:**
  - MEME Cascade/Absence (2026): the exact task. No system exceeds 0.06 on Cascade (Table 2).
  - MQuAKE/MeLLo (2023): multi-hop after edits.
  - MemoryAgentBench FC-MH (≤28%) and StateMemBench (2026).
  - Statler: state updates, but no what-if queries.
  - Tacheny (2026) and AlMahri (2026): impact queries over an explicit dependency graph, solved by tools/BFS.
  - WIQA (2019): what-if over static text.
- **Verdict:** the task framing is not novel. **No precedent was found for a fixed-size (non-graph, non-text) state that answers Cascade-type impact questions by query-time rule application**: MEME's only working configuration propagates at ingestion with a frontier LLM. A precise, defensible claim would be "query-time update propagation in a fixed-size state, measured on MEME Cascade/Absence."
- **Strongest baseline to beat:** the explicit graph plus deterministic traversal (Neo4j/BFS as in AlMahri; tool-LLM as in Tacheny). It is near-perfect when the graph is right, so the comparison must include the step where the LLM builds the graph from logs. Next come MEME's MD-flat with ingestion-time propagation, MeLLo, and Zep/Graphiti.

#### (c) Confidence, abstention and human correction of the learned rules or state
- **Closest prior:**
  - Abstention: MEME Absence task, where memory systems score ≤0.05 and in-context Sonnet 4.6 scores 0.35 (Table 2). ConformalHDC (arXiv 2602.21446): conformal coverage and abstention from HDC prototype similarity, e.g. MNIST α=0.05 coverage 0.950 and OOD AUC 0.815–0.873 (Table 1).
  - Confidence-filtered rules: HtT (confidence ≥p). R5 rule scores with pruning.
  - Human editing of rules: ARLC programmed-then-trained. ELLEN (arXiv 2403.03305): editable rules give +26% relative on one TACRED relation without retraining. TeachMe (EMNLP 2022): a memory of user corrections reaches within 1% of the upper bound with feedback on 25% of examples. TellOnce (arXiv 2606.13174): revisable rules for coding agents.
- **Verdict:** each component has precedent. **No precedent was found for calibrated abstention from cleanup similarity on multi-hop rule chains, combined with direct editing of a learned VSA rule and a traceable answer chain.**
- **Risk:** a symbolic table is the most editable store there is, since deleting a row is exact. VSA editing (unbind and subtract) leaves crosstalk, so edit locality must be measured.
- **Strongest baseline to beat:** a rule table with support/confidence counts that abstains when no rule fires. On the abstention side, conformal prediction on the LLM's own answer.

#### (d) Edge and robot settings (on-device, neuromorphic, low-power)
- **Closest prior:**
  - Renner et al. (NMI 2024): resonator on Loihi with a 171× energy-delay-product advantage over CPU (arXiv v4 Fig. 6C); synthetic 2-D scenes only; dense cleanup is the bottleneck.
  - Langenegger et al. (Nat. Nano 2023): in-memory PCM factorizer handling problems "at least five orders of magnitude larger" (abstract).
  - HDReason (FPGA KG completion): 65× energy vs RTX 4090.
  - VSA-OGM: occupancy mapping with 45× lower latency and 400× less memory (abstract).
  - Dold et al. (ICONS 2022): spiking KG embeddings.
  - Neubert et al. (KI 2019): low-power robustness for robots argued, not measured.
  - Statler, SayPlan and ConceptGraphs all rely on cloud GPT-class LLMs.
- **Verdict:** **No precedent was found for VSA state tracking plus rule inference over LLM-written operational events on neuromorphic or edge hardware.** Hardware precedent covers only factorization (perception) and KG completion.
- **Strongest baseline to beat:** an embedded key-value or graph store (a dict or SQLite plus BFS on a microcontroller or Jetson) for the reasoning step, and a small on-device LM for the writer. An energy claim against an LLM baseline would be a strawman. It has to beat the embedded table.

### II.5. Search log (queries actually run, standard mode)

1. `Abductive Rule Learner with Context-awareness ARLC vector-symbolic Raven arXiv`
2. `Probabilistic Abduction for Visual Abstract Reasoning via Learning Rules in Vector-symbolic Architectures Learn-VRF arXiv`
3. `Resonator networks 2 factorization performance and capacity compared to optimization-based methods ...`
4. `Towards Learning to Reason: Comparing LLMs with Neuro-Symbolic on Arithmetic Relations in Abstract Reasoning I-RAVEN-X ARLC`
5. `hyperdimensional computing knowledge graph embedding link prediction FB15k-237 HDC` (no HDC hit)
6. `vector symbolic architecture bAbI question answering holographic reduced representations reasoning` (no VSA-on-bAbI hit)
7. `hyperdimensional computing CLUTRR kinship reasoning vector symbolic` (no hit)
8. `hyperdimensional computing rule learning inductive logic programming vector symbolic first-order logic 2024` (no VSA-ILP hit)
9. `HDReason hyperdimensional knowledge graph reasoning algorithm-hardware codesign`
10. `"vector symbolic" multi-hop reasoning knowledge graph query hypervector 2025` (found PathHD)
11. `bAbI hyperdimensional computing OR "vector symbolic" OR "tensor product representation" question answering accuracy`
12. `Nickel holographic embeddings of knowledge graphs HolE ...`
13. `R5 rule discovery ... CLUTRR`; `Neural Compositional Rule Learning ... NCRL`
14. `LLM cloud incident impact analysis service dependency graph blast radius benchmark 2025`
15. `OpenRCA benchmark root cause analysis LLM ICLR 2025 accuracy`
16. `supply chain disruption question answering knowledge graph LLM what-if impact which suppliers affected`
17. `WIQA "what if" reasoning over procedural text ...`
18. (a) `few-shot rule learning interpretable compact rule memory editable human correction neuro-symbolic 2025`
19. (a) `learn relational composition rules from few examples vector symbolic architecture kinship one-shot mapping vector generalization`
20. (a) `vector symbolic architecture learn rules few examples systematic generalization compositional benchmark 2025 IBM abductive in-context`
21. (a) `vector symbolic architecture large language model reasoning memory hypervector 2025 2026 arXiv neuro-symbolic LLM` (only probes/self-published reports)
22. (b) `LLM agent reasoning over continuously updated world state counterfactual "what if" impact questions benchmark dynamic environment`
23. (b) `multi-hop reasoning over evolving facts state updates LLM memory benchmark 2026 "state tracking" fixed-size memory` (found MEME, StateMemBench)
24. (c) `hyperdimensional computing confidence calibration abstention uncertainty similarity threshold reject option` (found ConformalHDC)
25. (c) `Towards Teachable Reasoning Systems dynamic memory of user feedback TeachMe`; `ELLEN neuro-symbolic few-shot relation classification ...`
26. (c) `human-in-the-loop correction of learned rules agent world model editable memory abstain uncertain inference LLM 2025`
27. (d) `neuromorphic visual scene understanding resonator networks Loihi ...`; `In-memory factorization of holographic perceptual representations ...`
28. (d) `spiking neural network knowledge graph reasoning neuromorphic hardware relational embeddings Loihi multi-hop`
29. (d) `vector symbolic architecture robot memory reasoning on-device low power scene semantic memory hypervector robotics`

**Not opened / UNVERIFIED:**
- Emruli & Sandin 2014: full text returned 403; only the Crossref record was checked.
- Venues for ConceptGraphs (ICRA 2024), Lee et al. (ICLR 2016), HtT, AriGraph, ELLEN (Findings NAACL 2024) and arXiv 2412.05586 (journal version) are not stated on the pages I opened, so they are left unconfirmed.
- Not opened at all: OpenRCA 2.0 (2606.27154), WhatIfBench (2608.27953), Hyperdimensional Probe (2509.25045), RESOLVE (2411.08290), EverMemBench, OAKS, MINTEval, GRSNN.

### II.6. References

- Alam, M. M., Raff, E., Biderman, S., Oates, T., Holt, J. (2023). Recasting Self-Attention with Holographic Reduced Representations. ICML 2023. arXiv:2305.19534.
- AlMahri, S., Xu, L., Brintrup, A. (2026). Automating Supply Chain Disruption Monitoring via an Agentic AI Approach. arXiv:2601.09680.
- Anokhin, P., Semenov, N., Sorokin, A., Evseev, D., Kravchenko, A., Burtsev, M., Burnaev, E. (2024/2025). AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents. arXiv:2407.04363.
- Camposampiero, G., Hersche, M., Terzić, A., Wattenhofer, R., Sebastian, A., Rahimi, A. (2024). Towards Learning Abductive Reasoning using VSA Distributed Representations. NeSy 2024. arXiv:2406.19121.
- Chen, H. et al. (2024). HDReason: Algorithm-Hardware Codesign for Hyperdimensional Knowledge Graph Reasoning. arXiv:2403.05763.
- Cheng, K., Ahmed, N. K., Sun, Y. (2023). Neural Compositional Rule Learning for Knowledge Graph Reasoning. ICLR 2023. arXiv:2303.03581.
- Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413. VENDOR.
- Cui, S., Krishna, R., Jha, S., Iyer, R. K. (2025/26). Praxis: Integrating Program Analysis with Observability for Root-Cause Analysis. DSN (accepted). arXiv:2512.22113.
- Dalvi Mishra, B., Tafjord, O., Clark, P. (2022). Towards Teachable Reasoning Systems. EMNLP 2022. arXiv:2204.13074.
- Dold, D., Soler Garrido, J., Caceres Chian, V., Hildebrandt, M., Runkler, T. (2022). Neuro-symbolic computing with spiking neural networks. ICONS 2022. arXiv:2208.02576.
- Emruli, B., Sandin, F. (2014). Analogical Mapping with Sparse Distributed Memory. Cognitive Computation 6:74–88. doi:10.1007/s12559-013-9206-3. [V-meta only]
- Fan, X., Liu, M., Yang, R., Ouyang, S., Han, J. (2026). Can Agent Memory Systems Track Evolving State? arXiv:2608.19652.
- Frady, E. P., Kent, S. J., Olshausen, B. A., Sommer, F. T. (2020). Resonator Networks, 1. Neural Computation 32(12):2311–2331. arXiv:2007.03748.
- Ganesan, A. et al. (2021). Learning with Holographic Reduced Representations. NeurIPS 2021. arXiv:2109.02157.
- Gu, Q. et al. (2023). ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. arXiv:2309.16650.
- Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. (2024). HippoRAG. NeurIPS 2024. arXiv:2405.14831.
- Hersche, M., Zeqiri, M., Benini, L., Sebastian, A., Rahimi, A. (2023). A neuro-vector-symbolic architecture for solving Raven's progressive matrices. Nature Machine Intelligence 5(4):363–375. arXiv:2203.04571.
- Hersche, M., di Stefano, F., Hofmann, T., Sebastian, A., Rahimi, A. (2023). Probabilistic Abduction for Visual Abstract Reasoning via Learning Rules in VSAs. NeurIPS 2023 Workshop MATH-AI. arXiv:2401.16024.
- Hersche, M., Camposampiero, G., Wattenhofer, R., Sebastian, A., Rahimi, A. (2024). Towards Learning to Reason: Comparing LLMs with Neuro-Symbolic on Arithmetic Relations in Abstract Reasoning. arXiv:2412.05586.
- Heus, E., Bookstaber, R., Sharma, D. (2025). Exploring Network-Knowledge Graph Duality. ICAIF 2025 Workshop. arXiv:2510.01115.
- Hu, Y., Wang, Y., McAuley, J. (2025). Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench). arXiv:2507.05257.
- Jung, S., Rubinstein, A., Uselis, A., Yun, S., Oh, S. J. (2026). MEME: Multi-entity & Evolving Memory Evaluation. arXiv:2605.12477.
- Kanerva, P. (2009). Hyperdimensional Computing. Cognitive Computation 1:139–159. (already verified)
- Kent, S. J., Frady, E. P., Sommer, F. T., Olshausen, B. A. (2020). Resonator Networks, 2. Neural Computation 32(12):2332–2388. arXiv:1906.11684.
- Kim, N., Schuster, S. (2023). Entity Tracking in Language Models. ACL 2023. arXiv:2305.02363.
- Kleyko, D., Rachkovskij, D. A., Osipov, E., Rahimi, A. (2022/2023). A Survey on HDC/VSA, Part I (ACM CSUR 55(6)) and Part II (ACM CSUR 55(9)). arXiv:2111.06077, arXiv:2112.15424.
- Kumar (2026). arXiv:2606.24948. (already verified)
- Langenegger, J. et al. (2023). In-memory factorization of holographic perceptual representations. Nature Nanotechnology 18(5):479–485. arXiv:2211.05052.
- Lee, M., He, X., Yih, W., Gao, J., Deng, L., Smolensky, P. (2016). Reasoning in Vector Space: An Exploratory Study of Question Answering. arXiv:1511.06426.
- Liang, Z. et al. (2026). ConformalHDC: Uncertainty-Aware Hyperdimensional Computing. arXiv:2602.21446.
- Liu, Y., Chung, W. Y., Chen, H., Yeung, C., Imani, M. (2025/26). Are Hypervectors Enough? Single-Call LLM Reasoning over Knowledge Graphs (PathHD). arXiv:2512.09369.
- Lu, S., Liu, B., Mills, K. G., Jui, S., Niu, D. (2022). R5: Rule Discovery with Reinforced and Recurrent Relational Reasoning. ICLR 2022. arXiv:2205.06454.
- Mejri, M., Amarnath, C., Chatterjee, A. (2024). LARS-VSA. arXiv:2405.14436.
- Neubert, P., Schubert, S., Protzel, P. (2019). An Introduction to Hyperdimensional Computing for Robotics. KI 33:319–330. doi:10.1007/s13218-019-00623-z.
- Nickel, M., Rosasco, L., Poggio, T. (2016). Holographic Embeddings of Knowledge Graphs. AAAI-16. arXiv:1510.04935.
- Rana, K. et al. (2023). SayPlan. CoRL 2023. arXiv:2307.06135.
- Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., Chalef, D. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956. VENDOR.
- Renner, A. et al. (2024). Neuromorphic visual scene understanding with resonator networks. Nature Machine Intelligence 6(6):641–652. arXiv:2208.12880.
- Schlag, I., Schmidhuber, J. (2018). Learning to Reason with Third-Order Tensor Products. NeurIPS 2018. arXiv:1811.12143.
- Schlag, I., Munkhdalai, T., Schmidhuber, J. (2021). Learning Associative Inference Using Fast Weight Memory. ICLR 2021. arXiv:2011.07831.
- Snyder, S. et al. (2024/2025). Brain Inspired Probabilistic Occupancy Grid Mapping with VSAs. arXiv:2408.09066.
- Tacheny, N. (2026). Agentic Diagnostic Reasoning over Telecom and Datacenter Infrastructure. arXiv:2601.07342.
- Tandon, N., Dalvi Mishra, B., Sakaguchi, K., Bosselut, A., Clark, P. (2019). WIQA: A dataset for "What if..." reasoning over procedural text. EMNLP 2019. arXiv:1909.04739.
- Tang, H., Key, D., Ellis, K. (2024). WorldCoder. NeurIPS 2024. arXiv:2402.12275.
- Vacareanu, R., Alam, F., Islam, M. A., Riaz, H., Surdeanu, M. (2024). Best of Both Worlds: A Pliable and Generalizable Neuro-Symbolic Approach for Relation Classification (ELLEN). arXiv:2403.03305.
- Xu, J. et al. (2025). OpenRCA: Can LLMs Locate the Root Cause of Software Failures? ICLR 2025.
- Yoneda, T. et al. (2024). Statler. ICRA 2024. arXiv:2306.17840. (already verified; extended here)
- Zhong, Z., Wu, Z., Manning, C. D., Potts, C., Chen, D. (2023). MQuAKE. EMNLP 2023. arXiv:2305.14795.
- Zhou, Y. et al. (2026). TellOnce: Correcting Coding Agents Once, Improving Their Future Actions. arXiv:2606.13174.
- Zhu, Z. et al. (2023/2024). Large Language Models can Learn Rules (HtT). arXiv:2310.07064.

---

# Part III. Neuro-symbolic state tracking over streams, and reasoning engines beyond lookup

Draft, 2026-11-07. Scope 2023–2026, weighted to 2025–2026.

**How to read this.** Every number has a table, figure or section number from a primary source that was opened during this review (arXiv abs/html, ACL/ICLR/IJACSA PDFs). "UNVERIFIED" means only a search snippet or secondary page was seen. "Covered" means the work is on the caller's already-researched list: it is cited, not re-researched. Vendor or industry studies are labelled as such.

---

### III.0. Bottom line (expanded in the Verdict)

- Neuro-symbolic AI as a whole is large. The LLM-to-solver translation niche (text to FOL/ASP/PDDL/SQL, then a solver) is crowded, and at least six independent 2024–2026 studies agree that the residual error is semantic translation, not the solver. So "the interface is the hard part" is an established finding. It is not novel on its own.
- No precedent was found for the specific combination: reading state changes that the system of record never logged, from human channels interleaved with system feeds, into a continuously updated symbolic state, then exact impact queries with log-line provenance, evaluated set-exact against the system of record.
- The pieces exist separately:
  - ChatDB: an LLM writes SQL per record.
  - Log2Graph: an LLM turns telemetry into a dynamic KG and answers "which services are impacted if this DB fails" (weak evidence).
  - MM-3DSGU: human utterances update a robot scene graph.
  - DST, ATOM and Graphiti-style temporal KGs.
  - Medication reconciliation: clinic notes vs the structured EHR.
- One 2026 result is a warning for the pending "LLM over log" comparison. On LongMemEval/LoCoMo, a long-context GPT-5-mini beat a Mem0 fact-memory on recall. Our win has to come from overwrites, unlogged changes and multi-hop composition. PI-LLM and ActionReasoningBench support that this is where long-context models break.

---

### III.1. Field maps

| Survey | Scope | Counts / taxonomy (from the survey itself) | Crowded vs sparse |
|---|---|---|---|
| Colelough & Regli 2025, arXiv 2501.05435 (PRISMA, 2020–2024) | Neuro-symbolic AI broadly | 1,428 initial → 392 candidates (§2.3/§3) → 167 read in detail → 158 included (§3). The abstract says "167 met inclusion criteria", which is internally inconsistent. Area shares on the 158 (§3): Learning & Inference 63% (n=99), Knowledge Representation 44% (n=70), Logic & Reasoning 35% (n=55), Explainability & Trustworthiness 28% (n=44), Meta-cognition 5% (n=8) | L&I and KR are crowded. E&T∩L&R has only 9 papers (3.33%), and only AlphaGeometry sits at the intersection of four areas. State tracking is not a category; world models appear only in reference titles |
| Yang et al. 2025, IJCAI-25 Survey Track, arXiv 2508.13678 | Neuro-symbolic for LLM reasoning | Three directions: Symbolic→LLM, LLM→Symbolic, LLM+Symbolic. No counts on the abs page (a GitHub list is linked) | Our work is "LLM→Symbolic" (LLM as reader), the translate-then-solve branch |
| Liu et al. 2025, arXiv 2502.09100 | Logical reasoning in LLMs | Paradigms: deductive / inductive / abductive / analogical. Enhancement strategies: data-centric tuning, RL, decoding, neuro-symbolic hybrid | Neuro-symbolic is one of four enhancement families |
| Delvecchio, Molfetta & Moro, IJCAI-25 (arXiv 2603.03177) | Task-directed NeSy survey | Organised by task. No counts on the abs page | — |

Note: none of these maps has a category for continuously updated symbolic state read from text streams. The crowded sub-area adjacent to us is one-shot text-to-logic for logic QA (FOLIO, ProofWriter, ProntoQA, etc.).

---

### III.2. LLM → symbolic state / world model, updated over time

Column "Stream?" asks whether the symbolic state is updated incrementally from a stream (S), or built once from a static text (O).

| Work | Problem / benchmark | Headline numbers | Stream? | Reading / translation error | What fails | Relation to our claim |
|---|---|---|---|---|---|---|
| ActionReasoningBench, Handa et al. (arXiv 2406.04046; ICLR 2025 per mlanthology listing, not opened) | Reasoning about actions and change (RAC) over 8 IPC domains, action sequences of 1–19 steps, 3,498 test questions, 6 categories including State Tracking | State Tracking free-answer at lengths 1/10/19 (Table 3): GPT-4o 73.33/74.42/75.51; Llama-3.1-8B 28.89/18.60/16.33; Llama-3.1-70B 64.44/60.47/51.02; fine-tuned Llama-3.1-8B 75.56/66.67/65.31. Ramification free-answer: GPT-4o 0% at every length (Table 4); o1-preview 18.4% overall (abstract) | O (the action sequence is given as text). The LLMs used no solver, deliberately; ASP was used only to build ground truth | n/a (end-to-end) | Ramifications (indirect effects), negated fluents (−12.16% average), and longer sequences | Strongest evidence that end-to-end LLMs plateau near 75% on full-state tracking even with clean action text, and collapse on indirect effects, which are our cascades. A fine-tuned 8B model matches GPT-4o |
| ACPBench, Kokel et al., AAAI 2025 (arXiv 2410.05669) | 7 tasks × 13 PDDL domains; "Progression" = effect of one action on the state | Progression best LLM, 2-shot CoT (Table 2): Bool 95.38 (GPT-4o-mini), MCQ 93.85 (Llama-3.1-405B). Fine-tuned Granite-code-8B on unseen domains: 80.0/82.0 vs best LLM 94.0/96.0 (Table 4) | O, single step | n/a | Other tasks (reachability, landmarks) not examined here | Single-step state progression is nearly solved by large LLMs. The difficulty sits in multi-step accumulation and cascades |
| ByteSized32-State-Prediction, Wang et al., ACL 2024 (arXiv 2406.06485) | Can an LLM act as a text-world simulator? 76,369 transitions from 31 games (Table 1); 2,954 sampled for GPT-4 | GPT-4 full-state accuracy, dynamic transitions with LLM-written rules (Table 2): overall 59.0, action-driven 76.1, environment-driven 44.1. With no rules, environment-driven drops to 24.4 | O per transition | n/a | Errors concentrate on properties needing arithmetic, commonsense or science. Unaltered-value errors appear when the model predicts both transition types at once | Even one-step state updates written in JSON are unreliable for GPT-4. Supports separating reading from a deterministic update rule |
| ChatDB, Hu et al. 2023 (arXiv 2306.03901) | Synthetic fruit-shop log: 70 chronological records (~3.3k tokens), 50 questions | ChatDB 41/50 (82%) vs ChatGPT 11/50 (22%); hard questions 28/35 vs 1/35 (results table) | **S**: the LLM writes INSERT/UPDATE/DELETE per record, then answers by SQL | Not analysed | Tiny, synthetic, single run; ChatDB's own errors are not analysed | **Closest structural precedent** for "LLM writes events into a symbolic store, then exact query". Everything is logged, there is one clean channel, and there is no static-fact composition |
| LLM-State, Chen, Xiao & Hsu (arXiv 2311.17406) | VirtualHome long-horizon planning (5 maps × 15 tasks) plus a Fetch robot | Hard tasks success rate (Table I): 77.14% vs 8.71% naive LLM, 0% ProgPrompt. Removing the summary gives 14.29%; removing object entries gives 64.86% (Table II). Real robot 3/3 vs 0/3 (Table III) | Partly S: updated after failures and plan completion, re-derived from the full action history | Not measured. Assumes a perfect detector | One real-robot task; perception errors not evaluated | The LLM maintains an object→attribute state. Its source is the agent's own actions, not third-party messages |
| MM-3DSGU, Olivastri et al. 2024 (arXiv 2411.02938) | Robot 3D scene graph updated from human text, perception, time and robot actions | Table 1, one scripted mission × 10 runs: Add 100%, Remove 66.67%, Move 66.67%. Failures were attributed to RGB-D | S, but only one scenario | The LLM parses operation type, object and rooms. No parse-error rate | No baselines; the time module is unevaluated; no conflict policy between humans and perception | **Closest robotics precedent** for "human messages update a symbolic place state". No impact queries and no system-of-record comparison |
| LLM+P, Liu et al. 2023 (arXiv 2304.11477) | NL planning problem → PDDL problem → classical planner | Success % (Table 1), LLM+P vs LLM-as-planner: Blocksworld 90 vs 15, Grippers 95 vs 35, Storage 85 vs 0, Floortile 0 vs 0 | O | Most failures are mis-specified problem PDDL, such as a missing initial condition (§ failure analysis) | Without an in-context example PDDL, it fails | Classic "symbolic state beats end-to-end once translation is right" result. The failures are reading the initial state |
| Text2World, Hu et al. (arXiv 2502.13092; ACL Findings 2025 per ACL Anthology listing) | Generate a PDDL domain (the world model) from text, 103 domains | DeepSeek-R1, zero-shot → 3 correction rounds (Table 1): Exec 72.3→89.1, F1 preconditions 57.6→65.0, F1 effects 58.8→67.3 | O | Syntax errors fall with correction. "The majority" of errors are missing preconditions or effects (Intro) | Incomplete modelling | Learning the dynamics is hard. We sidestep this: our dynamics (move/contain) are fixed, and only event reading is learned |
| WorldCoder, Tang, Key & Ellis 2024 (arXiv 2402.12275); PoE-World, Piriyakulkij et al. 2025 (arXiv 2505.10819; NeurIPS 2025 per proceedings listing, not opened) | World model as an LLM-written program, learned from interaction (gridworlds; Pong and Montezuma's Revenge) | No numbers on the abs pages. Claims: more sample-efficient than deep RL (WorldCoder); generalises to unseen Montezuma's Revenge levels (PoE-World) | S (interaction), but the learned object is the transition function | — | — | Orthogonal: they learn *rules*, we read *events* |
| TG-LLM, Xiong et al., ACL 2024 (arXiv 2401.06853) | Text → temporal graph → CoT over the graph (TGQA, TimeQA, TempReason) | EM (main table): TGQA 0.797 vs GPT-4 CoT 0.821; TimeQA-Hard 0.631 vs 0.618; TempReason-L2 0.424 vs GPT-4 SP 0.454 | O | No translation accuracy. Temporal-graph errors in 100 human-checked CoTs: 0.10 → 0.05 | Graph reasoning is LLM CoT, not an engine | Text-to-structured-time with a learned reader; no exact engine |
| ATOM, Lairgi et al. 2025 (arXiv 2510.22590) | Incremental temporal KG from a news stream (1,076 NYT COVID articles), dual time (observed vs valid) | Entity-resolution F1 0.994 vs Graphiti 0.959; relation F1 1.0 vs 0.902 (Table 3); latency −93.8% vs Graphiti (Fig. 3) | **S** | Extraction hallucination rate 0.428; temporal omission 0.366 (Table 2) | No downstream QA or reasoning evaluation | Shows the stream-to-temporal-KG reader is far from solved: about 43% hallucinated atomic facts |
| FnCTOD, Li et al., ACL 2024 (arXiv 2402.10466) | Zero-shot dialogue state tracking via function calling (MultiWOZ 2.1) | +5.6% average JGA over the previous SOTA with ChatGPT; GPT-4 +14%; a fine-tuned 13B ≈ ChatGPT (abstract) | **S** (turn by turn) | JGA is the error metric | — | DST is the oldest "read conversation into slot state" field. It covers one dialogue and one schema: no system feeds, no unlogged-vs-logged reconciliation, no composition |
| Belief-R, Wilie et al. 2024 (arXiv 2406.19764) | Belief revision given new premises, about 30 LMs | Abstract: LMs "generally struggle"; models that update well under-perform when no update is needed | S (premise sequence) | — | Trade-off between updating and retaining | Motivates explicit overwrite semantics in a symbolic store |
| Symbolic Working Memory, Wang et al. 2024 (arXiv 2408.13654) | Multi-step rule application with an external symbolic memory | No numbers on the abs page | S within a problem | — | — | Adjacent: the memory serves rule grounding, not world state |
| OpenPI2.0, Zhang et al., EACL 2024 (arXiv 2305.14603); TRAC, He et al., ACL 2023 | Entity-state tracking in procedures; textual RAC | Abstract only: SOTA LMs "far from competent" (OpenPI2.0). TRAC numbers **UNVERIFIED** (not opened) | O | — | — | Background benchmarks |

**How symbolic-state approaches compare with end-to-end LLMs (§2 synthesis).**
- Where translation is right, symbolic state wins by large margins: LLM+P 90 vs 15 (Table 1); ChatDB 82% vs 22%; LLM-State 77% vs 9% (Table I).
- End-to-end LLMs cap around 75% on full-state tracking (ActionReasoningBench Table 3) and about 59% on simulator transitions (ByteSized32 Table 2). They fail on indirect effects (ActionReasoningBench Table 4).
- A fine-tuned 8B end-to-end model can match GPT-4o (ActionReasoningBench Table 3). So the fair comparison for us is a fine-tuned small *end-to-end* model, not only frontier zero-shot models.

---

### III.3. Where the errors come from in LLM+symbolic pipelines

| Study | Setting | Error split (with source) | Conclusion |
|---|---|---|---|
| Ishay & Lee 2026, "LLMs as ASP Programmers" (arXiv 2604.27960) | NL → ASP on 6 benchmark sets (ZebraLogic, Sudoku, Mystery Blocksworld, MultiLogicNMR, BoardgameQA) | Accuracy 54.6% without revision → 78.0% with solver-feedback self-correction (Table 3). Failure counts, no revisions (Table 5): syntax 360 (59.0%), satisfiable-but-wrong 117 (19.2%), unsatisfiable 116 (19.0%), timeout 17 (2.8%). With revisions: syntax 7.0%, sat-wrong 49.2%, unsat 40.3% | Solver feedback removes syntax errors. What is left is semantic: wrong or over-constrained programs. Solver timeouts are rare |
| QUITE, Schrader et al., EMNLP 2024 (arXiv 2410.10449) | NL Bayesian premises → ProbLog vs end-to-end GPT-4 | Numeric premises (Table 4): ProbLog-FT (Mistral-7B) 54.5 correct / 28.6 error vs GPT-4-Turbo zero-shot 36.7 / 57.6 wrong. ProbLog-Prompt (GPT-4) 76.1% invalid programs. With premises given (oracle): 87.3% correct | Authors: "parsing the lengthy premises is the main source of errors". A small fine-tuned translator beats GPT-4 end-to-end |
| Text2World (arXiv 2502.13092) | Text → PDDL domain | Executability 72.3→89.1 after correction, but F1 for effects stays at 67.3 (Table 1) | Correction fixes syntax; semantic omissions persist |
| Signal-Coverage Matrix, Dai et al. 2026 (arXiv 2606.28013) | Lean autoformalization (ProofNet#, MiniF2F) | Elaborator feedback recovers about 64% of the type stratum. The semantic-only stratum is flat on net: 87.5% of errors rescued, 8 new ones created (abstract) | Same pattern in maths: feedback fixes form, not meaning |
| Thatikonda et al. 2026 (arXiv 2601.09446) | Small (≤4B) models NL→FOL | Gemma-3-4B SFT+ on FOLIO: 66.00% executable but 36.95% accurate (Table 2). Error catalogue: parsing, NoneType, token, arity (Tables 1, 6) | For small readers, "executes" ≠ "correct". Arity and format errors dominate before fine-tuning |
| FOLIO/MALLS audit, Brunello et al., EMNLP 2026 (arXiv 2606.02837) | Gold FOL annotations | 42.5% of FOLIO and 42% of MALLS gold formalizations are wrong; ambiguous NL in 17.8% / 51%; corrected labels give +11 to +23 pp (abstract) | Measured "translation error" is itself noisy. Our set-exact gold labels from a simulator avoid this, which is a methodological point in our favour |
| McGinness & Baumgartner, LPAR 2024 (arXiv 2408.03492) | PRONTOQA steamroller → FOL + ATP | Defines syntactic vs semantic error classes; the ATP reduces semantic errors "significantly" (no numbers on the abs page) | Same split |
| LLM+P (arXiv 2304.11477) | NL → PDDL problem | Failures are mostly mis-specified initial state or goal (text) | Reading the *state* is the failure point. This is the closest analogue to ours |
| ATOM (arXiv 2510.22590) | Stream → temporal KG | Hallucination 0.428 (Table 2) | Stream readers hallucinate inferred facts |
| PI-LLM, Wang & Sun, ICML 2025 LCFM workshop (arXiv 2506.08184) | *End-to-end* LLM reading a key-value update stream (46 keys, 3–400 updates per key) | Accuracy declines log-linearly toward zero with interference (abstract). Llama-4-Maverick goes from ~100% at 2 keys to <5% at 46 keys (Fig. 9). A "forget" instruction changes accuracy by <10 pp (Fig. 12). CoT does not help (Fig. 14). Larger models do better (t=3.03, p=0.005); context length does not predict it (p=0.886) | The end-to-end failure mode for logs with overwrites is retrieving superseded values. This is exactly our "thing moved again" case |

**Takeaway for "the interface is the hard part".** The finding is consistent across text→ASP, text→ProbLog, text→PDDL, text→FOL and text→Lean. Once solver or compiler feedback fixes syntax, the residual error is semantic reading, almost never the solver. That supports our framing, but it is prior art at the claim level. Our novelty has to be the stream-state version: overwrites, unlogged changes, multi-channel reconciliation and provenance. That version is not measured in any study found here.

---

### III.4. Applied neuro-symbolic state tracking in operations

| Work | Domain / data | Graph or state source | Numbers | Stream? | What fails | Relation |
|---|---|---|---|---|---|---|
| OpenRCA, Xu et al., ICLR 2025 | 335 failures, 3 enterprise systems, >68 GB of logs, metrics and traces (abstract) | None (the LLM reads telemetry) | Best: Claude 3.5 + RCA-agent 11.34% (abstract). Hard queries (3 root-cause elements): 0.00 for every model and method (Table 3) | n/a | Long, heterogeneous telemetry | End-to-end LLM over operational logs is very weak. Different task (root cause, not state) |
| PRAXIS, Cui et al. 2025 (arXiv 2512.22113; IBM/UIUC) | 30 injected code/config incidents in a K8s microservice app | Service dependency graph from a topology monitor (snapshot at alert time) plus a static program-dependence graph | Root-cause reasoning Pass@1 61.54 vs SRE-Agent 19.71; identification 73.85 vs 35.77 (Table III); 3.8× fewer tokens (Table V) | Snapshot | Outdated or incomplete graphs hurt; near zero on manifest and resource faults | The graph is given by monitors, not read from text |
| SynergyRCA, Xiang et al. 2025 (arXiv 2506.02490) | Two production K8s clusters | StateGraph (spatial and temporal) plus MetaGraph | Precision ~0.90, ~2 min (abstract) | unclear | — | Same family |
| Log2Graph, Amanmadov & Abdullayev, IJACSA 16(10) 2025 | Multi-cloud testbed, >150 GB/day of telemetry | LLM parses logs, traces and config into a Neo4j KG over 10-minute sliding windows | Entity F1 0.91, relation F1 0.88 on 50k annotated entries; template baselines ~0.72 (§IV-C). NL queries 87% vs 65% LLM-only vs 58% traditional tools, expert-judged (§IV-E). Explicitly answers "what services will be impacted if this database fails?" (abstract) | **S** | Evidence quality is low: lower-tier venue, an unnamed "LLM similar in scale to GPT-4" fine-tuned on 100M log lines, expert-judged accuracy, no released data | **Closest framing to our impact queries.** Machine telemetry only: no human messages, no unlogged changes, no set-exact metric |
| KG data layer for AssetOps, Mandarapu & Kunkunuru, AG2026@VLDB (arXiv 2605.26874) | AssetOpsBench, 139 industrial maintenance scenarios | Typed KG (12,647 nodes) | GPT-4 65% → 82–83% with LLM-written Cypher; native graph primitives 99% on graph-answerable scenarios (abstract) | No (schema-built) | — | Supports "the data layer is the lever, not the LLM". The KG is not read from text |
| Causely, Dalal et al. 2026 (arXiv 2605.18327), **vendor study** | 24-microservice OpenTelemetry demo | "Live, queryable model" built from telemetry | RCA accuracy 75% → 100%; time-to-diagnosis −63% (abstract) | S | Vendor self-evaluation | Commercial existence proof of causal topology plus agents |
| Tacheny (arXiv 2601.07342), covered | Telecom and datacenter infrastructure | Typed directed infrastructure graph for RCA and impact analysis | — | — | — | Closest on impact analysis over dependencies (covered) |
| Medication evidence across notes vs EHR, Jiang et al. 2026 (arXiv 2608.01570; submitted to IEEE BIBM) | Clinic notes vs structured medication history | LLM-assisted reference construction plus deterministic normalisation | Only 16.44% of note-derived rows exactly match same-visit structured history; 55.17% match semantically; 90.34% match within ±30 days (abstract) | — | Mismatch comes from normalisation, terminology and timing | **Closest "system of record vs human notes" measurement**, in clinical data. It measures the gap but does not maintain a state |
| Robot object tracking (MM-3DSGU, LLM-State; Statler, SayPlan and ConceptGraphs covered) | Homes | Scene graph or object-attribute dict | See §2 | S | — | Same pattern, physical domain |
| Inventory/WMS reconciliation; lab chain-of-custody | — | — | No arXiv or peer-reviewed LLM work found. Search hits were vendor marketing pages (agentui.ai, sysgenpro) and LIMS vendor documentation (lablynx, LIMSWiki) | — | — | Open application niche (vendor pages only, unevaluated) |

---

### III.5. Novelty check for the reframed claim

| Sub-claim | Queries run (abridged; full log at end) | Closest prior work | Verdict |
|---|---|---|---|
| (a) Reading state changes from multi-channel human messages plus system logs into a symbolic state, where the system of record misses 25–45% of changes | "LLM read human messages plus machine logs infer unlogged state changes system of record"; "LLM extract state changes from chat messages incident channel…"; "LLM inventory reconciliation warehouse…"; "LLM laboratory sample chain of custody LIMS…"; "LLM maintenance logs shift handover notes…"; "LLM extract events from Slack chat and system logs reconcile operational state…"; medication-reconciliation query | ChatDB (one clean channel, all logged); Log2Graph (telemetry only); MM-3DSGU (human text plus perception, one mission); FnCTOD/DST (one dialogue); ATOM (news stream); Jiang 2026 (measures the notes-vs-record gap, no state). Products: Harness AI SRE Scribe and ProdRescue AI build incident *timelines* from Slack plus logs (vendor pages, unevaluated) | **No precedent found** for the combination, or for measuring a reader against a stale system of record |
| (b) Impact / what-if queries over a continuously updated state, with provenance | "'what-if' impact analysis LLM knowledge graph dependency propagation blast radius"; "counterfactual what-if reasoning knowledge graph…" | Log2Graph ("which services impacted if this DB fails", weak evidence); Tacheny (covered); PRAXIS/SynergyRCA (dependency graphs from monitors); CFKGR/COULDD (counterfactual KG reasoning, §6b); Causely (vendor) | **Partly taken as a framing in AIOps.** In those systems the dependency state comes from monitors or telemetry, not human text. None has a set-exact evaluation of affected sets or per-fact log-line provenance |
| (c) Small distilled reader vs a large LLM reading the whole log | "small fine-tuned model extract structured state … vs long-context LLM"; "small language model distilled … agent memory outperforms long-context…"; PI-LLM; LongMemEval | ActionReasoningBench (FT 8B ≈ GPT-4o end-to-end, Table 3); UniversalNER (ICLR 2024: distilled models beat ChatGPT by 7–9 F1 on 43 NER datasets); MemReader-0.6B (2604.07877, distilled extractor, no numbers on abs); **Pollertlam & Kornsuwannawit 2026 (2603.04814): long-context GPT-5-mini has higher factual recall than Mem0 fact-memory on LongMemEval and LoCoMo; memory is cheaper after about 10 turns at 100k tokens**; PI-LLM (long-context collapse under overwrites) | **No direct head-to-head for operational state.** Prior evidence cuts both ways: plan the LLM-over-log baseline carefully (§Verdict) |

---

### III.6. Reasoning engines over a tracked state, where lookup or BFS is not enough

#### III.6a Uncertain state (probabilistic logic with neural inputs; read confidences; abstention)

| Work | What the engine does beyond lookup | Benchmark / numbers | What fails |
|---|---|---|---|
| Scallop, Li, Huang & Naik, PLDI 2023 (arXiv 2304.04812) | Datalog with recursion, negation and aggregation, plus provenance semirings (e.g. diff-top-k-proofs). Gives P(answer) from per-fact probabilities and keeps proof structure, so correlated derivations are not double-counted | PacMan-Maze 99.4% success after 50 episodes vs DQN 84.9% after 50K (Table 3); CLUTRR +25% over baselines incl. GPT-3-FT (§6.3); DeepProbLog took 40 days for 4K samples on sum4 while Scallop finishes a 15K epoch in minutes (§6.4, Table 4) | Cost blows up as k grows in top-k proofs (§6.4). Approximate, not exact, WMC |
| Dolphin, Naik et al. 2024/25 (arXiv 2410.03348) | Python-level NeSy: symbolic on CPU, vectorised probabilities on GPU | 1.71×–62× faster than baselines (abstract). Scallop exceeds 10 h on HWF-15/19 and Path-128/256 (Table 2). ISED fails to converge on larger Sum-N and HWF | — |
| QUITE (see §3) | ProbLog over LLM-parsed Bayesian premises | ProbLog-FT 54.5 vs GPT-4 36.7 (Table 4). All models except ProbLog-FT fail on evidential and explaining-away questions (text) | Parsing errors (28.6%) |
| OrLog, Hoveyda et al., ECIR 2026 (arXiv 2601.23085) | LLM gives predicate plausibilities in one forward pass; a probabilistic engine computes query posteriors | Better top-rank precision than LLM-as-reasoner, larger gains on disjunctive queries; ~90% fewer tokens (abstract, no table numbers) | — |
| Lobster (arXiv 2503.21937) | GPU Datalog with provenance semirings | 5.3× average over Scallop (search snippet) — **UNVERIFIED** | — |
| Calibrated abstention over read facts | — | **No NeSy precedent found.** Only generic LLM conformal abstention turned up (e.g. arXiv 2604.27914, not opened) | — |

#### III.6b Temporal / history-dependent and counterfactual (non-mutating) queries

| Work | What it needs beyond current-state lookup | Numbers | What fails |
|---|---|---|---|
| Test of Time, Fatemi et al., ICLR 2025 (arXiv 2406.09170) | Interval and timeline reasoning over temporal facts | ToT-Semantic by question type (Table 5), Claude-3-Sonnet / GPT-4 / Gemini 1.5 Pro: Timeline 24.00 / 28.29 / 36.29; NumberOfEventsInTimeInterval 29.71 / 61.43 / 59.14. Complete graphs 34.00 / 40.25 / 52.50 (Table 3) | Timeline and interval-count questions are the hardest. These are exactly "which specimens were in FRZ-07 during the window" |
| TG-LLM (see §2) | Learned text→temporal graph | TimeQA-Hard 0.631 | Reasoning is CoT, not exact |
| ATOM (see §2) | Dual time: observation vs validity | No QA evaluation | — |
| CFKGR / COULDD, Zellinger, Stephan & Roth, EACL 2024 (arXiv 2403.06936) | Answer under a hypothetical without corrupting the base KG | F1 on *unchanged* facts (retention), Table 3, S/M/L datasets: gpt-3.5-turbo 40.22 / 42.25 / 40.95 vs COULDD-ComplEx 69.95 / 69.21 / 54.25. ChatGPT is better at detecting *changed* facts on S/M (68.90, 52.12) | LLMs "mutate" unrelated facts under what-ifs. Embedding methods miss changes that do not follow learned rules |
| Event calculus with LLMs | — | Only LM+ASP (Yang 2023, covered; event-calculus-style axioms) and LLM2LAS (Borroto et al. 2025, arXiv 2509.16590, under review at TPLP, no numbers on abs) found | **No 2024–2026 evaluation of LLM→event-calculus translation found** |

#### III.6c Cascades, defaults/exceptions, non-monotonic reasoning with LLM-extracted facts

| Work | Numbers | Note |
|---|---|---|
| ActionReasoningBench ramifications (see §2) | GPT-4o 0% free-answer (Table 4); o1-preview 18.4% | Indirect effects are cascades; end-to-end LLMs fail |
| Ishay & Lee 2026 LLM+ASP (see §3) | MultiLogicNMR skeptical 80.7→90.5; BoardgameQA 57.6→82.4 with self-correction (Table 3) | Abstract claims stable-model semantics beats SMT on default and exception tasks. Strongest current "LLM-extracted facts + non-monotonic engine" baseline |
| ChatDB, LLM+P | See §2 | Exact engines remove the arithmetic and propagation errors of end-to-end LLMs |

#### III.6d Learning rules from few examples

| Work | Numbers | Note |
|---|---|---|
| Yang, Wu & Yue 2025 (arXiv 2505.21486) | LLM-generated language bias plus the MAXSYNTH ILP solver: 82.3–84.8 accuracy vs best HypoGeniC 71.65 (Table 1). ZENDO (Claude-3.7) 81.3 vs 68.3 / 60.0. IHR is better on SHOES | Two synthetic tasks, about 60 test items; no Popper or Aleph baseline |
| DSR-LM (covered; read from primary) | Rule learning on ground-truth KBs, train k∈[2,3] → test k∈[4,10]: DSR-without-LM 98.81 vs CTP 95.57 vs GAT 39.05 (Table 7) | Learns weighted kinship-composition rules (Table 4) |
| Reason Popper-ly, HtT, NCRL (covered); FOLD-R++ (arXiv 2110.07843, default rules with exceptions) | FOLD-R++ **UNVERIFIED** (not opened) | Exceptions are a natural fit for our site-rule overrides |

#### III.6e Where VSA or neural engines have a measured advantage

| Work | Measured advantage | Caveat |
|---|---|---|
| NVSA, Hersche et al. (covered; numbers from the primary text) | Search in superposition: 63.2 min vs 15,408 min for unrestricted PrAE on the 3×3 grid, **244× faster**, with memory <10 GB vs >53 GB; accuracy 96.89 vs 97.50 (Table I and §Results) | Baseline is exhaustive probabilistic abduction. VSA is slightly *less* accurate. The advantage is speed and memory on combinatorial search, not exactness |
| In-memory factorizer, Langenegger et al., Nature Nanotechnology 2023 (arXiv 2211.05052) | Resonator-network factorisation on phase-change memristive chips: MVMs at O(1); solves problems "at least five orders of magnitude larger" (abstract) | Hardware; perceptual factorisation, not relational queries |
| HDReason, Chen et al. 2024 (arXiv 2403.05763) | HDC KG completion on FPGA: 10.6× speedup and 65× energy vs RTX 4090; 4.2× / 3.4× vs an FPGA GCN at similar accuracy (abstract) | Baseline is a GCN, not a symbolic engine |
| GHRR, Yeung et al. (arXiv 2405.09689; IEEE TAI DOI listed) | Non-commutative binding: "enhanced decoding accuracy" for trees vs FHRR (abstract) | No numbers on the abs page |
| Fast Weight Memory, Schlag et al., ICLR 2021 (covered) | catbAbI QA accuracy 96.75% with 694k weights vs Transformer-XL 87.66% with 10.5M (Table 1) | A neural state memory trained end-to-end; not exact |
| Robustness to noisy facts vs a symbolic engine | **No measured head-to-head found** | — |

Net: the measured VSA advantages are speed, energy and memory on search-heavy problems (abduction, factorisation, KG completion), plus differentiability. No study found shows VSA beating an exact symbolic engine on *accuracy* for relational or impact queries. This matches our own control result that the engine is trivial for exact lookups.

#### III.6f Training the reader end-to-end through the engine (answer-only supervision)

| Work | Data / supervision | Answer-only vs fact-label reader | What breaks |
|---|---|---|---|
| DSR-LM, Zhang et al. (covered; primary text) | CLUTRR, answers only; RoBERTa reader (~127M params) plus Scallop and a semantic loss | Isolated answer-only relation extractor: **84.69%** vs **85.32%** when trained on intermediate relation labels (§4 ablation, near Table 6). End-to-end CLUTRR 60.98 ± 2.64 (Fig. 3, Table 5) | End-to-end accuracy is far from the 98.81 achieved on ground-truth KBs (Table 7), so reading is the bottleneck |
| NVSA (covered; primary text) | RAVEN / I-RAVEN | End-to-end **87.7** vs attribute-label-trained **98.5** average on RAVEN (Table II); I-RAVEN end-to-end 88.1 (Table III) | A 10.8-point gap: answer-only perception is clearly worse |
| rsbench, Bortolotti et al. (arXiv 2406.10368; NeurIPS 2024 D&B per proceedings listing) | DeepProbLog and others with label-only supervision | MNAdd-EvenOdd: DPL label F1 **0.94** but concept F1 **0.06** (Table 2). Kand-Logic: DPL 0.87 / 0.25 (Table 4). The only concept-supervised model (CBM) reaches at most 0.61 concept F1 (SDD-OIA, Table 5) | **Reasoning shortcuts**: right answers from wrong facts. Fatal for a trace that must cite the right log line |
| Scallop / Dolphin (above) | Output labels only | HWF data efficiency degrades slowly with less data (Scallop Table 5) | Scaling (Dolphin Table 2) |
| Stein et al. 2025 position (arXiv 2505.24874) | — | End-to-end NeSy training "limited to simplistic problems"; three pitfalls: compute, data, programs (abstract) | Argues for foundation-model readers plus symbolic programs instead |
| NeSyGPT, Cunnington et al. 2024 (arXiv 2402.01889) | Fine-tuned VLM extractor plus learned ASP, trained sequentially | **UNVERIFIED** (not opened) | — |
| LLM-scale readers trained end-to-end through a probabilistic engine (2024–2026) | — | **None found.** Dolphin's CLUTRR uses RoBERTa-base | Open |

Implication: answer-only training reached parity with fact labels once (DSR-LM, a single-relation extractor), lost 10.8 points another time (NVSA), and is prone to shortcuts (rsbench). For a provenance-citing reader, use it as a fine-tuning signal on top of a fact-label-trained reader (DSR-LM-style semantic loss), not as a replacement.

---

### III.Verdict

**How crowded.** Neuro-symbolic AI overall is large: 158–167 included papers in one 2020–2024 PRISMA review alone (Colelough & Regli §3). The LLM→solver translation niche is crowded, with at least ten 2024–2026 error-analysis papers in §3 plus the covered Logic-LM, LINC, SatLM and LM+ASP. Our specific niche is not taken in the searches run here. The niche: a stream-updated symbolic state read from multi-channel operational text with unlogged changes, exact impact queries over its composition with static site facts, per-fact log provenance, and a set-exact evaluation against the system of record. Adjacent framings (AIOps impact graphs, DST, temporal-KG agent memory) are busy, so the positioning must be precise.

**Five closest prior works, and how each differs**
1. **ChatDB** (Hu et al. 2023, 2306.03901). The LLM writes SQL per record into a DB, then answers exactly (82% vs 22%). It differs in several ways:
   - one clean synthetic channel, where every change is logged;
   - no stale system of record and no human-message reconciliation;
   - no containment chains or static-fact composition;
   - no provenance;
   - a GPT-3.5 reader, 50 questions.
2. **Log2Graph** (Amanmadov & Abdullayev, IJACSA 2025). An LLM turns telemetry into a dynamic Neo4j KG and answers "what services will be impacted if this database fails". It differs in four ways:
   - machine telemetry only: no human channels and no unlogged changes;
   - expert-judged accuracy (87%), not set-exact;
   - an unspecified model;
   - weak venue and no data.
3. **MM-3DSGU** (Olivastri et al. 2024, 2411.02938). An LLM parses human utterances into Move/Add/Remove on a robot scene graph alongside perception. It differs in four ways:
   - one simulated mission, 66.67% move success, no baselines;
   - no impact queries;
   - no long multi-source stream;
   - no LLM-over-log comparison.
4. **Tacheny (2601.07342, covered) / PRAXIS (2512.22113)**. LLM agents over typed dependency graphs for RCA and impact (PRAXIS 61.54 vs 19.71, Table III). The dependency and state graph comes from monitors or static analysis, not read from text. The question is root cause, not "which items are affected given the current location state".
5. **LLM-State (2311.17406) / Statler (covered)**. An LLM maintains object→attribute state for planning (77% vs 9% on hard tasks). The state is fed by the agent's own actions and perception, not third-party human channels plus system feeds. There is no system-of-record comparison and no provenance.

Runners-up: FnCTOD/DST (dialogue slot state); ATOM, Zep and Graphiti (temporal KG from streams, 0.428 hallucination); Jiang et al. 2026 (notes vs EHR gap, 16% exact match); Kim & Schuster and Host-Materialized State (covered).

**Must cite.**
- ChatDB, Statler, LLM-State, MM-3DSGU.
- ActionReasoningBench and ByteSized32-SP (end-to-end state-tracking ceilings).
- PI-LLM and LongMemEval (long-context failure under updates).
- Pollertlam & Kornsuwannawit 2026 (fact-memory loses to long context on recall: a counterpoint we must address).
- OpenRCA, Log2Graph, Tacheny, PRAXIS (ops).
- LLM+P, QUITE, Ishay & Lee 2026, Signal-Coverage Matrix, FOLIO audit (interface error).
- FnCTOD (DST).
- ATOM and Zep/Graphiti (temporal KG).
- Jiang et al. 2026 (system of record vs notes).
- Kim & Schuster; Host-Materialized State.
- Engine side: Scallop, DSR-LM, rsbench, CFKGR, Test of Time, NVSA.

**Strongest baselines**
1. Frontier long-context LLM reading the full log plus static site facts in context. Use the strongest available model, and report the update count per entity: PI-LLM predicts collapse with overwrites, and 2603.04814 predicts long context wins on plain recall.
2. The same pipeline with a frontier model as the reader (ChatDB-style LLM→SQL/state writer). This isolates "distilled small reader" from "symbolic state".
3. A fine-tuned 8B end-to-end model (no symbolic state). ActionReasoningBench shows this matches GPT-4o.
4. Temporal-KG agent memory (Graphiti/Zep or Mem0) fed the same streams.
5. System of record plus rules: the 0.22–0.31 baseline we already have.

**External benchmarks to run**
- **ActionReasoningBench**, State Tracking plus Ramification subsets. Clean RAC text with ASP ground truth, published GPT-4o and fine-tuned-8B numbers (Table 3/4), and ramifications are our cascades. Our reader→state→closure pipeline maps directly onto it.
- **PI-LLM** (key-value update stream; public code and data). Directly tests the overwrite failure that our symbolic state should eliminate. Alternatives: Michelangelo Latent List, or the LongMemEval knowledge-update subset.

**Reasoning engine: most promising open problem in our setting.** Probabilistic, valid-time impact queries with non-mutating what-ifs over a *read* state. Example: "which specimens were, with probability ≥ p, in FRZ-07 during the defrost window, including ones moved out since; abstain otherwise; and what if rack R3 had been moved to FRZ-09".

Each needed piece is shown to break end-to-end or to be missing:
- timeline and interval questions: ≤36% for frontier LLMs (ToT Table 5);
- retention under what-ifs: ~40 F1 for ChatGPT (CFKGR Table 3);
- indirect effects: 0% for GPT-4o (ActionReasoningBench Table 4);
- no NeSy work found combining per-fact read confidences with valid-time history, counterfactual overlays and calibrated abstention.

A VSA-specific angle is evaluating many hypothetical states in superposition. Measured VSA wins exist only for search-heavy problems (NVSA 244×, HDReason 10.6×/65×), so it would need a head-to-head against Scallop with one copy per hypothesis, at equal exactness. No such comparison exists. That makes the angle open but risky.

**Strongest existing neuro-symbolic engine baseline, per capability**
- **Uncertain facts:** Scallop with diff-top-k-proofs provenance (PLDI 2023). Use exact ProbLog as the reference on small instances. Dolphin if scale is an issue.
- **Temporal / what-if:** an ASP (clingo) event-calculus encoding driven by a self-correcting LLM+ASP translator (Ishay & Lee 2026; LM+ASP covered). Scallop with time as an attribute is the differentiable alternative. Note: no LLM→event-calculus benchmark exists.
- **End-to-end learning from answers:** DSR-LM (Scallop plus semantic loss; 84.69 vs 85.32 answer-only vs fact-label extractor), with Dolphin for scaling. Report concept-level (fact) accuracy alongside answer accuracy, because rsbench shows answer F1 0.94 with concept F1 0.06.

---

### III.References (primary sources opened unless marked)

1. Colelough, B. C., Regli, W. 2025. Neuro-Symbolic AI in 2024: A Systematic Review. arXiv:2501.05435.
2. Yang, X.-W., Shao, J.-J., Guo, L.-Z., et al. 2025. Neuro-Symbolic AI: Towards Improving the Reasoning Abilities of LLMs. IJCAI-25 Survey Track. arXiv:2508.13678.
3. Liu, H., Fu, Z., Ding, M., et al. 2025. Logical Reasoning in Large Language Models: A Survey. arXiv:2502.09100.
4. Delvecchio, G. P., Molfetta, L., Moro, G. 2025. Neuro-Symbolic AI: A Task-Directed Survey in the Black-Box Models Era. IJCAI-25. arXiv:2603.03177.
5. Handa, D., Dolin, P., Kumbhar, S., Son, T. C., Baral, C. ActionReasoningBench. arXiv:2406.04046 (ICLR 2025 per mlanthology listing; venue not opened).
6. Kokel, H., Katz, M., Srinivas, K., Sohrabi, S. 2025. ACPBench. AAAI 2025. arXiv:2410.05669.
7. Wang, R., Todd, G., Xiao, Z., et al. 2024. Can Language Models Serve as Text-Based World Simulators? ACL 2024. arXiv:2406.06485.
8. Hu, C., Fu, J., et al. 2023. ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory. arXiv:2306.03901.
9. Chen, S., Xiao, A., Hsu, D. 2023/24. LLM-State. arXiv:2311.17406.
10. Olivastri, E., Francis, J., Pretto, A., Sünderhauf, N., Rana, K. 2024. Multi-Modal 3D Scene Graph Updater for Shared and Dynamic Environments. arXiv:2411.02938.
11. Liu, B., Jiang, Y., Zhang, X., et al. 2023. LLM+P. arXiv:2304.11477.
12. Hu, M., Chen, T., et al. 2025. Text2World. arXiv:2502.13092 (ACL Findings 2025 per Anthology listing).
13. Tang, H., Key, D., Ellis, K. 2024. WorldCoder. arXiv:2402.12275.
14. Piriyakulkij, W. T., Liang, Y., Tang, H., Weller, A., Kryven, M., Ellis, K. 2025. PoE-World. arXiv:2505.10819 (NeurIPS 2025 per proceedings listing; not opened).
15. Xiong, S., Payani, A., Kompella, R., Fekri, F. 2024. Large Language Models Can Learn Temporal Reasoning. ACL 2024. arXiv:2401.06853.
16. Lairgi, Y., Moncla, L., Benabdeslem, K., Cazabet, R., Cléau, P. 2025. ATOM. arXiv:2510.22590.
17. Li, Z., Chen, Z. Z., Ross, M., et al. 2024. LLMs as Zero-shot Dialogue State Tracker through Function Calling (FnCTOD). ACL 2024. arXiv:2402.10466.
18. Wilie, B., Cahyawijaya, S., Ishii, E., He, J., Fung, P. 2024. Belief Revision (Belief-R). arXiv:2406.19764.
19. Wang, S., Wei, Z., Choi, Y., Ren, X. 2024. Symbolic Working Memory Enhances Language Models for Complex Rule Application. arXiv:2408.13654.
20. Zhang, L., Xu, H., Kommula, A., Callison-Burch, C., Tandon, N. 2024. OpenPI2.0. EACL 2024. arXiv:2305.14603 (abs only).
21. He, W., Huang, C., Xiao, Z., Liu, Y. 2023. Exploring the Capacity of PLMs for RAC (TRAC). ACL 2023. **UNVERIFIED** (not opened).
22. Ishay, A., Lee, J. 2026. LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning. arXiv:2604.27960.
23. Schrader, T. P., Lange, L., Razniewski, S., Friedrich, A. 2024. QUITE. EMNLP 2024. arXiv:2410.10449.
24. Dai, C., Yan, Z., Lin, Z. 2026. The Signal-Coverage Matrix. arXiv:2606.28013.
25. Thatikonda, R. K., Han, J., Buntine, W., Shareghi, E. 2026. Improving Symbolic Translation of LMs for Logical Reasoning. arXiv:2601.09446. Also 2024, arXiv:2409.16461 (abs).
26. Brunello, A., Curaba, C., Geatti, L., et al. 2026. Fixing FOLIO and MALLS. EMNLP 2026. arXiv:2606.02837.
27. McGinness, L., Baumgartner, P. 2024. Automated Theorem Provers Help Improve LLM Reasoning. LPAR 2024. arXiv:2408.03492.
28. Xu, L., Beckmann, P., Valentino, M., Freitas, A. 2025. Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition. arXiv:2510.06774 (abs only).
29. Wang, C., Sun, J. V. 2025. Unable to Forget: Proactive Interference … (PI-LLM). ICML 2025 LCFM Workshop. arXiv:2506.08184.
30. Wu, D., Wang, H., Yu, W., et al. 2025. LongMemEval. ICLR 2025. arXiv:2410.10813.
31. Vodrahalli, K., et al. 2024. Michelangelo: Long Context Evaluations Beyond Haystacks. arXiv:2409.12640 (opened; no numeric scores in text).
32. Xu, J., Zhang, Q., Zhong, Z., et al. 2025. OpenRCA. ICLR 2025 (proceedings PDF).
33. Cui, S., Krishna, R., Jha, S., Iyer, R. K. 2025. Agentic Structured Graph Traversal for RCA (PRAXIS). arXiv:2512.22113.
34. Xiang, Y., Chen, C. P., Zeng, L., et al. 2025. SynergyRCA / StateGraph. arXiv:2506.02490.
35. Amanmadov, N., Abdullayev, T. 2025. From Logs to Knowledge: LLM-Powered Dynamic Knowledge Graphs for Real-Time Cloud Observability (Log2Graph). IJACSA 16(10).
36. Mandarapu, M., Kunkunuru, S. 2026. Knowledge Graphs as the Missing Data Layer for LLM-Based Industrial Asset Operations. AG2026@VLDB. arXiv:2605.26874.
37. Dalal, D., Sara, E., Yemini, B., Miller, C., Kliger, S. 2026. Causely … Benchmark Study (vendor). arXiv:2605.18327.
38. Jiang, M., Ni, C., Liu, W., Yin, Z. 2026. Characterizing Treatment-Context Medication Evidence Across Clinic Notes and Structured EHR. arXiv:2608.01570.
39. Zhou, W., Zhang, S., Gu, Y., Chen, M., Poon, H. 2024. UniversalNER. ICLR 2024. arXiv:2308.03279.
40. Kang, J., Li, C., Chen, D., et al. 2026. MemReader. arXiv:2604.07877 (abs; no numbers).
41. Pollertlam, N., Kornsuwannawit, W. 2026. Beyond the Context Window: Fact-Based Memory vs Long-Context LLMs. arXiv:2603.04814.
42. Li, Z., Huang, J., Naik, M. 2023. Scallop: A Language for Neurosymbolic Programming. PLDI 2023. arXiv:2304.04812 (full text).
43. Naik, A., Liu, J., Wang, C., et al. 2024/25. Dolphin. arXiv:2410.03348.
44. Lobster (GPU neurosymbolic). arXiv:2503.21937. **UNVERIFIED** (not opened).
45. Hoveyda, M., Piepenbrock, J., de Vries, A. P., de Rijke, M., Hasibi, F. 2026. OrLog. ECIR 2026. arXiv:2601.23085.
46. Fatemi, B., et al. 2025. Test of Time. ICLR 2025. arXiv:2406.09170.
47. Zellinger, L., Stephan, A., Roth, B. 2024. Counterfactual Reasoning with Knowledge Graph Embeddings (CFKGR/COULDD). EACL 2024. arXiv:2403.06936.
48. Borroto, M., Gallagher, K., Ielo, A., Kareem, I., Ricca, F., Russo, A. 2025. Question Answering with LLMs and Learning from Answer Sets (LLM2LAS). arXiv:2509.16590.
49. Yang, Y., Wu, J., Yue, Y. 2025. Robust Hypothesis Generation: LLM-Automated Language Bias for ILP. arXiv:2505.21486.
50. Wang, H., Gupta, G. FOLD-R++. arXiv:2110.07843. **UNVERIFIED** (not opened).
51. Langenegger, J., Karunaratne, G., Hersche, M., Benini, L., Sebastian, A., Rahimi, A. 2023. In-memory factorization of holographic perceptual representations. Nature Nanotechnology. arXiv:2211.05052.
52. Chen, H., Ni, Y., Zakeri, A., et al. 2024. HDReason. arXiv:2403.05763.
53. Yeung, C., Zou, Z., Jeong, S., Huang, W., Bastian, N. D., Imani, M. GHRR. arXiv:2405.09689.
54. Bortolotti, S., Marconato, E., Carraro, T., et al. 2024. rsbench. arXiv:2406.10368 (NeurIPS 2024 D&B per proceedings listing).
55. Stein, A., Naik, A., Velingker, N., Naik, M., Wong, E. 2025. The Road to Generalizable NeSy Learning Should be Paved with Foundation Models. arXiv:2505.24874.
56. Cunnington, D., Law, M., Lobo, J., Russo, A. 2024. NeSyGPT. arXiv:2402.01889. **UNVERIFIED** (not opened).
57. Covered (cited, not re-researched; numbers taken from primary text already on disk where stated):
    - DSR-LM (Zhang et al. 2023): numbers from primary text.
    - NVSA (Hersche et al.): numbers from primary text.
    - Fast Weight Memory (Schlag et al., ICLR 2021): numbers from primary text.
    - Logic-LM, LINC, SatLM, LM+ASP (Yang et al. 2023), Faithful CoT, R5, NCRL, CTP, HtT, Reason Popper-ly (2607.23019), Learn-VRF, ARLC, PathHD, HolE, TPR.
    - Statler, AriGraph, HippoRAG, Zep/Graphiti, Mem0, SayPlan, ConceptGraphs, MEME (2605.12477), MQuAKE/MeLLo, MemoryAgentBench, Tacheny (2601.07342), AlMahri (2601.09680), Host-Materialized State (NeurIPS 2026), Kim & Schuster 2023, VET-Bench (2603.08436).

---

### III.Search log

WebSearch, standard mode, 2026-11-07:

1. neuro-symbolic AI systematic review 2024 2025 arXiv number of papers taxonomy
2. survey neuro-symbolic large language models reasoning 2025 arXiv taxonomy symbolic solver LLM
3. 2026 survey neurosymbolic AI large language models systematic literature review papers counted trends
4. ActionReasoningBench reasoning about actions and change LLM benchmark state tracking
5. TRAC benchmark textual reasoning about actions and change LLM He 2023
6. "Can Language Models Serve as Text-Based World Simulators" ByteSized32 state prediction GPT-4 accuracy
7. LLM event calculus narrative translation reasoning about actions answer set programming 2024 2025 arXiv
8. large language models translate narrative into event calculus axioms reasoning temporal fluents evaluation arXiv
9. "event calculus" "large language models" 2025 arXiv translation narratives fluents inertia evaluation → no direct precedent
10. WorldCoder model-based LLM agent world model as code successors 2025 PoE-World code world model
11. Text2World benchmark LLM symbolic world model generation PDDL 2025
12. ChatDB augmenting LLMs with databases as symbolic memory
13. OpenPI2.0 entity state tracking procedures LLM benchmark Zhang 2023 salience
14. LLM-State open world state representation long-horizon task planning LLM 2023 arXiv 2311.17406
15. incremental temporal knowledge graph construction from streaming text with LLMs updates conflicting facts 2025 arXiv
16. LLM natural language to first-order logic translation errors analysis neurosymbolic solver error vs parsing error FOLIO 2024 2025
17. autoformalization errors LLM logic solver pipeline failure analysis "semantic errors" "syntax errors" percentage neuro-symbolic reasoning
18. "translation" bottleneck neuro-symbolic LLM logical reasoning most errors arise from formalization not solver study 2025
19. LLM root cause analysis knowledge graph dependency graph incidents microservices 2024 2025 arXiv OpenRCA
20. OpenRCA benchmark LLM root cause analysis ICLR 2025 telemetry Claude accuracy
21. LLM update 3D scene graph object relocation household robot dynamic environment human utterance object moved 2024 2025
22. Olivastri "Multi-Modal 3D Scene Graph Updater for Shared and Dynamic Environments" human language
23. Michelangelo long-context latent structure queries Latent List state tracking benchmark
24. LLM inventory reconciliation warehouse records discrepancies chat messages system of record extraction arXiv → vendor pages only
25. LLM extract state changes from chat messages incident channel maintain world state knowledge base continuously updated stream "state tracking" operations → no precedent
26. LLM laboratory sample chain of custody LIMS audit trail natural language notes specimen location tracking → no LLM precedent
27. LLM medication reconciliation clinical notes versus structured EHR medication list discrepancies 2024 2025 study
28. LLM maintenance logs shift handover notes knowledge graph extraction asset state industrial operations 2024 2025 arXiv
29. "what-if" impact analysis LLM knowledge graph dependency propagation "blast radius" service failure reasoning arXiv
30. LLM extract events from Slack chat and system logs reconcile operational state multi-source incident timeline construction arXiv 2025 → products only
31. LLM read human messages plus machine logs to infer unlogged state changes "system of record" stale knowledge base update from conversation agents arXiv → no precedent
32. proactive interference LLM working memory key value updates retrieve latest value accuracy declines number of updates 2025
33. small fine-tuned model extract structured state from text then query vs long-context LLM reading full history comparison state tracking 2025 arXiv
34. small language model distilled from GPT-4 structured state extraction stream events agent memory outperforms long-context LLM reading full transcript 2025 2026
35. LongMemEval knowledge update category accuracy temporal reasoning long-term memory chat assistants
36. LLM dialogue state tracking zero-shot function calling MultiWOZ joint goal accuracy 2024 FnCTOD
37. TG-LLM temporal graph translate text to temporal graph LLM learn temporal reasoning TGQA ACL 2024
38. LLM extracted facts with confidence probabilistic logic programming ProbLog Scallop question answering uncertainty 2024 2025
39. calibrated abstention neurosymbolic LLM extracted facts probabilistic reasoning selective prediction confidence of extracted facts 2025 → no NeSy precedent
40. reasoning shortcuts neuro-symbolic learning from labels concepts wrong high label accuracy rsbench benchmark Marconato
41. Lobster GPU neurosymbolic Scallop 2025 OR Dolphin scalable neurosymbolic learning OR ISED data-efficient learning with neural programs
42. neuro-symbolic learning with foundation models answer set programming end-to-end weak supervision LLM perception 2024 2025 NeSyGPT
43. Test of Time benchmark LLM temporal reasoning synthetic graphs Fatemi 2024
44. LLM inductive logic programming Popper rule learning with exceptions FOLD-R++ LLM 2024 2025 arXiv noisy facts
45. counterfactual what-if reasoning knowledge graph hypothetical facts without changing base KG benchmark LLM COULDD
46. Generalized holographic reduced representations non-commutative binding 2024 arXiv compositional
47. in-memory factorizer holographic perceptual representations resonator network Nature Nanotechnology 2023 speedup
48. hyperdimensional computing knowledge graph reasoning hardware accelerator energy efficiency vs GNN symbolic 2024 2025 arXiv

Primary sources read from disk: Scallop, DSR-LM, NVSA and FWM full texts, already present at /path/to/scratch OpenRCA and Log2Graph PDFs were extracted with pdftotext.
