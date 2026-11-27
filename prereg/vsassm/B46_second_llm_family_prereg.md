# B46 pre-registration: B36 on the hidden-state codes of a second LLM family

Date: 2026-11-19 (US Eastern; box clock Asia/Shanghai). Written and committed before any candidate model has encoded a
string, before the encoder choice, and before any memory (B32's VSA store and index, B36's key-free families) has
answered a dev or test query on the new codes.

Done before this file: reading B32 / B36 (prereg/B32_*.md, prereg/B36_*.md, results/b36_verdict.txt); the B46 code
(b46_embed.py, b46_choose_encoder.py, b46_run.py, b46_verdict.py, b46_check.py, b46_handread.py, b46_box/*.sh); a smoke
test of b46_run.py in a scratch directory on e5 codes (B32 and B36 selftests pass from the working directory; one e5 dev
store, one family); a tokenizer-only probe of the four tokenizers on the box (no weights loaded); the literature check
(results/b46_literature.txt); the Mistral weights' download on the A800 box (ModelScope, mistralai/Mistral-7B-Instruct-
v0.3; sha256 of every shard checked against ModelScope's listing). No string has been encoded by any candidate model.

## Why

B36 found that on frozen Qwen3-8B hidden-state codes (mean of layer-9 states over the name's tokens) the superposed VSA
store beats the best dev-chosen competitor (the per-fact index or a key-free classical memory: static functions, LSH/PQ
bucket sketches, Bloom filter, fly Bloom filter, SDM) in region R1 (V in {2, 16}, 4-16 bits per fact) by +6.77
[+5.82, +7.74] points, but its registered replication on e5-base-v2 sentence-encoder codes failed (+0.60 [-0.27,
+1.43]). A referee: "The 'LLM codes' claim rests on one layer of one model. Replicate on a second LLM family." B46 runs
the B36 protocol unchanged on one second, non-Qwen LLM family, chosen by B32's neutral encoder criterion. It goes into
the paper whatever it shows.

## Encoder: candidates and choice rule (fixed now)

Candidates: three non-Qwen open-weight instruction-tuned decoder LLMs of about the size of Qwen3-8B (Qwen3-8B, B32's
reader, is itself the post-trained release):
1. **Llama-3.1-8B-Instruct** (Meta; 32 layers, 4,096 numbers). File set already on the box from B40
   (/path/to/work/b40/llama31_8b_instruct; ModelScope LLM-Research/Meta-Llama-3.1-8B-Instruct; sha256 in
   results/b40_model_sha256_modelscope_etag.txt).
2. **Mistral-7B-Instruct-v0.3** (Mistral AI; 32 layers, 4,096 numbers). ModelScope mistralai/Mistral-7B-Instruct-v0.3
   (the official repository mirrored; the HF original is gated); shard sha256 recorded in the B46 compute record.
   If its download does not verify, it is dropped and the drop is recorded; the choice runs on the others.
3. **Granite-3.3-8B-Instruct** (IBM; 40 layers, 4,096 numbers), in the box's HF cache (ibm-granite/granite-3.3-8b-instruct).

Variants per model (B32's Qwen3-8B set, mapped by depth): two poolings x four layers at 1/4, 1/2, 3/4 and 4/4 of the
depth (Qwen3-8B: 9 / 18 / 27 / 36 of 36; Llama and Mistral: 8 / 16 / 24 / 32; Granite: 10 / 20 / 30 / 40).
- "mean": mean of hidden_states[l] over the name's tokens in "Name: {x}";
- "eol": hidden_states[l] of the last token of 'This name: "{x}" means in one word:"' (PromptEOL).

Recipe: b32_embed.lm (B32's reader code), bfloat16 weights, float32 pooling, rows L2-normalized, with two changes that
only matter for tokenizers unlike Qwen3's, both fixed now (b46_embed.py):
1. The prompt is tokenized as one string with the tokenizer's default special tokens. Llama 3.1 and Mistral prepend a
   BOS token by default and were trained with it; b32_embed.lm passed add_special_tokens=False, which for Qwen3 (no BOS
   token) is the default anyway. Without BOS the first token of these models acts as an untrained attention sink.
2. The name's tokens are found by character offsets (tokens starting at or after the colon of "Name:" that cover at
   least one character of the name), not by tokenizing " " + name separately (which, with a BOS-adding tokenizer,
   would need the BOS handled separately). Tokenizer-only probe (no weights): for all four tokenizers (Qwen3,
   Llama 3.1, Mistral v0.3, Granite 3.3) the selected tokens equal B32's separate tokenization for 98.99-99.09% of
   the 18,292 strings; the rest differ only by a lone space token before a name that starts with a digit or a
   symbol (now left out of the mean); no string has an empty span. The model input itself (for Qwen3: identical
   token ids) is unchanged.
Tokenizer loading: AutoTokenizer, as transformers loads it by default. Mistral v0.3's tokenizer_config makes
AutoTokenizer rebuild the fast tokenizer from the SentencePiece model, which needs the sentencepiece package that the
box's environment lacks (found by a tokenizer-only probe, no weights loaded); then the tokenizer.json shipped with the
model is loaded directly (PreTrainedTokenizerFast with special_tokens_map.json; its post-processor adds "<s>": probe
"Name: Paris" -> <s> _Name : _Paris). The loader used is recorded per model (embinfo_{tag}.json).
BOS by default: Llama 3.1 yes (<|begin_of_text|>), Mistral yes (<s>), Granite 3.3 no, Qwen3 no (no BOS token).
Recipe check (not a gate): b46_embed.py also encodes the strings with Qwen3-8B at layer 9 (tag qwen8chk); its codes are
compared with B32's (row cosine) and scored by the criterion; agreement shows the two changes are neutral for Qwen3.

**Criterion (B32's, unchanged; b32_lm_select.py):** exact-cosine entity retrieval over the whole dev pool (every dev
label stored, every dev alias a query, codes centred on the training mean). No value, no memory, no test string.
**Choice:** the (model, pooling, layer) with the highest criterion over all candidate variants; ties within 0.001 go to
the earlier model in the list above, then mean before eol, then the shallower layer. This choice is the B46 encoder.
It is stored as float16 (as B32's Qwen3-8B codes; its criterion is re-computed on the float16 file) and gets the
encoder tag **b46** (the tag seeds every fixed random component, as 'qwen8' and 'e5' did; fixed now, independent of
which model wins). Every variant's score is reported.
Diagnostic (reported, changes nothing): the literature check done before encoding (results/b46_literature.txt; Sun et
al. 2024) warns that Mistral-7B has massive activations on some ordinary tokens, which centring cannot remove from
mean-pooled codes. For the chosen codes the largest |value| per dimension (raw, centred, whitened) is reported.

## Protocol (B32 + B36, unchanged)

Everything is B32's and B36's code, run unchanged in a separate working directory (b46_out/: its b32_data holds B32's
entities.json and emb_strings.json, identical files, plus the new codes; b32_out/, b36_out/ and results/ there receive
the outputs that B32 / B36 wrote at the repository root). b46_run.py only registers the tag (b32_run.ENCS['b46'],
b36_run.ENCS['b46'] = the codes' file) and calls B32's / B36's functions; it changes no grid, rule or seed.
- Data, splits, stores: B32's (PopQA entities, random values, every alias of a stored entity a query; dev stores 6 at
  M = 256 and 3 at M = 1,024 per V; test stores 20 / 5 per V, the same entities, values and queries as in B32 / B36).
- Memories and grids: B32's VSA (72 configurations) and per-fact index (48, OPQ with 4 iterations, compute note 1) at
  4-64 bits per fact (b32_run.py dev) and at 1-2 (b36_run.py dev32); B36's SF (176), CM (132 / 44), BF (60), FBF (12),
  SDM (40) at 1-64 (b36_run.py dev), with amendment 1 (random plurality ties). Budgets, preprocessing (centre;
  centre + ZCA whitening fitted on the training split), projection pool, PQ codebooks (b36_run.py pqfit), accounting
  (ideal primary, real secondary) and limits as in B36.
- Choice: B32's rule for the VSA and index (b32_run.py choose, then b36_run.py choose for 1-2 bits per fact); B36's rule
  for the new families and BEST (the competitor family with the highest dev accuracy per cell). Written to
  b46_out/results/ and committed (copied to results/b46_choice_*.json) before any test run.
- Test: once (b32_run.py test, b36_run.py test / test32 / merge).

## Metric, statistics, regions (B36's, unchanged)

Accuracy on alias queries; cell difference VSA - BEST paired on the same queries; 95% cluster bootstrap (entity within a
store; 10,000 resamples); Bonferroni over 56 cells; exact sign test. Regions R1 = V in {2, 16} x bpf {4, 8, 16} x both
M (12 cells); R0 = V in {2, 16} x bpf {1, 2} (8); RH = V in {2, 16} x bpf {32, 64} (8); R2 = V = 1,024 (14). Per-family
differences (VSA minus each family's own dev choice), ORACLE (per cell the competitor with the highest TEST accuracy),
rate at equal accuracy (bits per fact to reach chance + lift x (EXACT - chance), lift 0.25 / 0.5; ratio BEST / VSA with
a 1,000-resample bootstrap). Computed by b36_verdict.py's own functions (b46_verdict.py calls them with encoder list
['b46', 'e5']; the e5 rows are B36's e5 test files, so they must reproduce B36's e5 numbers exactly: a check of the
wrapper) and written to results/b46_verdict.{json,txt} with the bars below.

## Bars (primary: the B46 codes, ideal accounting)

- **H1-B46 (primary):** over R1, VSA - BEST >= +2.0 points and the 95% interval lies above 0. This is B36's H1, unchanged.
- Secondary: the same bar under real accounting; R0 difference reported against the same bar; ORACLE over R1.
- Cell map (B36's): VSA WIN if the cell difference >= +2 and its Bonferroni interval excludes 0; BEST WIN if <= -2 and
  the Bonferroni interval excludes 0; otherwise tie / inconclusive; a VSA win is useful only if the VSA is >= 10
  points above chance.
- Reporting rule: no advantage of either side is claimed in a cell or region whose interval includes 0.

## How each outcome will be read (fixed now)

The paper's current evidence: R1 win on Qwen3-8B (+6.77), no win on e5 (+0.60, tie). B46 adds the second LLM family.
- **H1-B46 passes:** "At equal bits, in the few-values low-budget region, the VSA store beats the best dev-chosen
  classical memory (key-free or per-fact index) on hidden-state codes of two LLM families (Qwen3-8B layer 9 and the
  B46 encoder), but not on a contrastive sentence encoder (e5)." The paper may say the R1 win holds for frozen
  hidden-state codes of two LLM families, with both margins and intervals, accuracies (and distance from EXACT and
  chance), and where the store loses (V = 1,024; high budgets). It must not say the win holds for LLM codes in general,
  for every reader, or for neural codes as such, must keep the e5 tie in the same sentence or paragraph, and must name
  the layer and pooling of each LLM (chosen by a retrieval criterion, not by the memory comparison). If the ORACLE R1
  interval includes 0, it must add that against a competitor selected on the test set the win is not established.
- **H1-B46 fails as a tie** (difference < +2 or interval includes 0, and not a loss): "the R1 win is specific to Qwen3-8B
  layer 9": the paper must name Qwen3-8B as the only reader with a win (1 of 3 readers: Qwen3-8B win; B46 encoder and e5
  ties), and must drop any wording that attributes the win to LLM codes.
- **H1-B46 fails with VSA - BEST <= -2 and the interval below 0:** a classical memory beats the store in R1 on the second
  family; the paper reports a win on 1 of 3 readers and a loss on 1, and drops the bit-efficiency claim as a property of
  LLM codes.
- **H1-B46 passes only under real accounting:** no win is claimed for the B46 encoder (as in B36).
- If BEST in R1 is mostly a superposition-type sketch (CM count, SDM) and it ties or wins, the paper says so, as in B36.

## Predictions (stated so they can be wrong)

Accuracies and differences in points; B46 codes, ideal accounting unless stated.
P1. Encoder choice: the chosen variant is mean pooling at the 1/4-depth layer (as for Qwen3-8B), of Llama-3.1-8B-Instruct;
    its criterion lies between 0.62 and 0.74 (Qwen3-8B 0.7035, e5 0.6980).
P2. H1-B46 passes: R1 VSA - BEST between +2 and +9 with the 95% interval above 0.
P3. R1 VSA - IDX (B32's comparison) between +4 and +11.
P4. On dev, a key-free family is BEST in at least 4 of the 12 R1 cells (B36: 8 of 12 on Qwen3-8B, 7 on e5).
P5. R0 VSA - BEST between -3 and +4.
P6. R2 (V = 1,024) VSA - BEST <= -2 with the interval below 0; BEST is the index in at least 10 of the 14 cells.
P7. RH (32-64 bits per fact, V <= 16) VSA - BEST between -3 and +7.
P8. ORACLE over R1: VSA - ORACLE >= +1 with the interval above 0.
P9. Rate at equal accuracy (lift 0.25, ideal): BEST needs at least 1.5 times the VSA's bits per fact in at least 3 of
    the 4 settings (M, V in {2, 16}).
P10. EXACT on the B46 codes is within 6 points of Qwen3-8B's EXACT in each of the 8 (M, V) settings.
P11. Checks: independent scorer |difference| < 1e-9; 0 configurations over budget; re-derivation >= 99% (SF, CM, BF,
    FBF) and >= 98% (SDM); the e5 rows of b46_verdict.py equal B36's e5 rows; recipe check: qwen8chk codes have row
    cosine >= 0.9999 with B32's for every string whose name tokens are unchanged (99.09%) and criterion within 0.003
    of 0.7035.

## Checks (second angle; B36's, reused)

- b36_check.py score / bits / rederive (b46_check.py runs b36_check.py's code with only the encoder list and the code
  file table substituted; each substitution is asserted to occur): independent scorer (every cell from the dumps, |diff|
  < 1e-9, store-level bootstrap for R1); bits audit (every new-family prediction on every test store within budget;
  VSA dimension and index key bits from the written rules); independent re-derivation of the new families on 10% of the
  test stores (>= 1 per (M, V)), >= 99% agreement required, lower agreement investigated before the verdict is reported.
- b32_check.py rederive for the B46 VSA and index choices on 10% of test stores (same substitution), reported with the
  per-family agreement and accuracy difference (B32: VSA lin and LSH 100%; iterative-training families not bit-exact).
- Hand reading (b36_handread.py through b46_handread.py): 10 queries only the VSA answers right and 10 only BEST answers
  right, in the R1 cell with the largest difference and in the cell where BEST leads most; read by me, notes in
  results/b46_handread_notes.txt.
- Literature: B36's (results/b36_literature.txt) plus results/b46_literature.txt (done before encoding by a literature
  subagent; 30 verified entries): no published comparison of a VSA / HDC store or of key-free memories at equal bits
  across LLM families' hidden states was found; layer profiles and outlier geometry differ by family, so the size of
  the margin is not predicted by the literature. Updated after the result (support / conflict / gap).

## Compute

A800 box: GPU only for encoding (one idle GPU claimed through the shared lock directory; never GPU 3; no other process
touched; expected < 0.5 GPU-hours; cap 5 GPU-hours), and the encoder criterion on the box's CPU. Everything else on the
local CPU (16 cores, shared), as B36. No paid API. Ledger: b46_out/compute_ledger.txt (GPU wall-clock from the box's
logs/gpu_ledger.txt).

## Files

Code: b46_embed.py, b46_box/b46_gpu.sh, b46_box/b46_dl_mistral.sh, b46_choose_encoder.py, b46_run.py, b46_verdict.py,
b46_check.py, b46_handread.py. Working directory b46_out/ (codes not committed; regenerable with b46_embed.py).
Results: results/b46_encoder_choice.json, results/b46_choice_{b32,b36}.json, results/b46_devsum.txt,
results/b46_verdict.{json,txt}, results/b46_check.txt, results/b46_rederive.json, results/b46_handread.txt,
results/b46_predictions.txt, results/b46_literature.txt.
