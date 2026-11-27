# R1 pre-registration (draft, to be finalised before any R1 code): learned relation composition on CLUTRR

Date: 2026-10-20. Reasoning-system track (docs/reasoning_track.md), asked for by the user's boss. Architecture free
(VSA, SSM, transformer).

## Benchmark

CLUTRR v1 (Sinha et al., EMNLP 2019; HF `CLUTRR/v1`, licence CC-BY-NC 4.0: research comparison only), split
`gen_train23_test2to10`: kinship stories whose answer needs chaining k relations; training stories have k = 2, 3;
tests have k = 2 ... 10. The question "how is X related to Y" asks for one of ~20 kinship relations. The point is
systematic generalisation: models trained on short chains must answer long ones.

## System (Readable Reasoner)

1. **Reader** (prompted Qwen3; later distilled to 1.7B): each sentence -> facts (person, relation, person) with the
   persons' genders when stated.
2. **Rule memory (VSA):** composition rules r1 o r2 -> r3 learned only from training stories (k = 2: the two facts on the
   path and the gold answer), stored as delta-rule writes key = bind(code(r1), permute(code(r2))) -> value = code(r3).
   Inverses (e.g. mother -> son / daughter by gender) are learned the same way from training facts.
3. **Inference:** find the path between the two persons in the fact graph, orient its edges with the learned inverses,
   and fold the path left to right through the rule memory with cleanup at each step. Every intermediate relation is
   readable; the fold works for any k by construction.
4. **Abstain:** if a needed rule was never seen in training, answer "unknown" (counted as wrong) and report how often.

## Claims and bars (tentative; fixed in the final version)

- **P1** mean accuracy over k = 2-10 >= 0.90 on the test split.
- **P2** accuracy at k = 10 >= 0.85 (length generalisation).
- References: Qwen3-1.7B and Qwen3-8B thinking answering directly (same stories); published results (e.g. DSR-LM, Edge
  Transformer) quoted from their papers with their settings.

## Prediction

P1 holds; P2 holds if the reader extracts facts correctly, since the fold itself is exact; the reader (genders,
"in-law" relations) is the main risk.

## Development note 1 (2026-10-20): the composition engine with oracle facts

`r1_oracle.py` folds CLUTRR's gold relation sequences (`f_comb`) with rules learned only from the training split
(62 from k = 2 stories, 55 more from k = 3 via known rules; no conflicting votes). **Disclosure: this diagnostic was run
on the test split** (it involves no reader or tuned component; from now on development uses the validation split).
- Left-to-right folding: 0.520 overall, 0.218 at k = 10 (47% of chains need a pair never seen in training).
- Folding with any bracketing (CYK over sub-chains; kinship composition is associative): **0.975 overall, 1.000 at
  k = 10**, the same with the rules held in the VSA memory (117/117 rules read back exactly at D = 2048).
- Bug found and fixed: one codebook for keys and values gave exact crosstalk (a rule (x, b) -> a reads back code(x)
  when (a, b) is queried); three independent codebooks (first slot, second slot, value) remove it.
So the engine is not the bottleneck; the reader (relations and genders from text) is.

## Development note 2 (validation split)

First reader run: 0.083 (8B), 0.047 (1.7B) at k = 2. Cause: the reader consistently writes ["Jason", "grandfather",
"Guillermina"] as "Jason is Guillermina's grandfather", the opposite of the convention the prompt asked for. The prompt
now asks for that natural reading ("A is B's relation") and the code converts it to graph edges.

## Development note 3 (validation split)

After the direction fix: 0.750 (8B reader), 0.380 (1.7B) at k = 2. Errors: the reader adds relations the story does not
state ("Joe's grandfather Jason" also gives "Jason is Lewis's father"; "her father" attached to the wrong person), and
the shortest path then uses the invented fact. Changes: two worked examples in the reader prompt showing what not to
infer; the answer is a vote over all simple paths between the two people (ties: the shortest path).

## Development note 4 (validation split)

Path voting + examples: 0.793 (8B), 0.410 (1.7B). Remaining errors: relations still written the wrong way round
(["Joe", "mother", "Lisa"] for "Lisa is Joe's mother") and pronouns attached to the wrong person. Changes: the reader
writes {"who": A, "is_the": relation, "of": B} ("A is the relation of B"), which names the direction; and a reader in
thinking mode is tried (it thinks once per story; its tokens are reported). Also noted: some CLUTRR validation labels
contradict their stories (crowd-written noise, known for this benchmark).

## Development note 5 and final configuration (written before any R1 test run)

Validation (k = 2, 300 stories), explicit-field facts + path voting: 8B reader 0.757, **8B reader thinking 0.910**,
1.7B 0.440, 1.7B thinking 0.727. The reader thinks once per story; the VSA engine then composes exactly.

**Final (test split, k = 2-10, all 1,146 stories):**
- Primary system: Qwen3-8B reader in thinking mode, explicit-field prompt, rules learned from the training split in the
  VSA rule memory, vote over paths. Also reported: 8B no thinking, 1.7B thinking, 1.7B no thinking.
- References on the same stories: Qwen3-8B and Qwen3-1.7B answering directly in thinking mode.
- **P1** primary mean accuracy over k = 2-10 >= 0.80. **P2** primary accuracy over k >= 7 >= 0.75.
  **P3** primary beats Qwen3-8B thinking-direct over k >= 7 by >= 0.10.
- Reported: per-k accuracy; generated tokens per story (reader) vs per question (direct).
- **Prediction:** P1 and P3 hold; P2 is at risk because longer stories give the reader more chances to misread.
