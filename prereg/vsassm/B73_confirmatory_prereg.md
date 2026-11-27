# B73 (equal bits) pre-registration: a confirmatory run of the headline equal-bit comparisons on fresh Wikidata entities, every configuration frozen

Date: 2026-11-23, about 20:25 machine clock (Asia/Shanghai; US Eastern 07:25). Written and committed, then
OpenTimestamps-stamped and the proof committed, BEFORE any fresh entity is fetched or any fresh code is encoded. This
file registers the equal-bit half of B73, the confirmatory run requested by the round-23 referee ("a final confirmatory
run with every configuration frozen, on fresh data (entities and aliases outside PopQA or a different knowledge base;
new key-value stores with an independent value format)"). The equal-cell half is registered in
prereg/B73_cells_prereg.md (8deea4e, proof 9c2e552) and has already been run (df77702). **An external registration
(OSF) needs the human author and is pending; this internal registration (git commit + OpenTimestamps proof) does not
replace it.** The paper is not edited by B73.

## What has been seen before this file (disclosed)

- Every PopQA result of the paper's equal-bit runs (B49-B71): the dev choices, the test numbers and labels quoted
  below. The claims and margins below were written from them.
- The equal-cell half of B73 (its result and note after the test were committed at 20:04, before this file).
- **One count-only probe of the SPARQL endpoint** (about 20:10; run before this file was begun): of a random sample of 40,000 items with P31 = human,
  2,514 had an English Wikipedia article and an English alias. Only that number was returned (COUNT); no item, label or
  alias was seen. It set the human sample size below.
- **Self-test on synthetic data only** (b73_data.py synth: 3,000 random syllable names, 0-3 aliases each, Gaussian codes
  of the paper's dimensions; scratchpad, not committed): every arm (STORE, VSAR, VSAS, VSAPC, LAMC, NLMSC, SDMT) ran
  through b73_run.py on qwen8-sized (4,096) and bge-sized (384) codes at (M 256, V 16) and (M 1,024, V 2), and
  b73_run.py assemble built records with every arm's predictions. Seen: synthetic accuracies only (0.09-0.64); nothing
  was chosen from them. Timing: a LAMc process spends about 1-2 minutes fitting its side information on 4,096-d codes,
  so the per-store arms run in chunks of 5 test stores per process (this changes no number).
- No fresh entity has been fetched; no fresh code exists.

## Fresh entity set (b73_data.py; frozen)

Selection rule (b73_data.py module docstring, committed with this file): for each of nine classes of PopQA's subject
kinds (people Q5, films Q11424, literary works Q7725634, albums Q482994, television series Q5398426, singles Q134556,
cities Q515, association football clubs Q476028, businesses Q4830453), ONE query to the public Wikidata SPARQL endpoint
(generic User-Agent, no personal information): a random sample (Blazegraph bd:sample, RANDOM) of 40,000 / 25,000 /
40,000 / 30,000 / 15,000 / 15,000 / 15,000 / 20,000 / 30,000 items with that P31, kept if the item has an English
Wikipedia article (PopQA's subjects are English-Wikipedia entities; its popularity is page views), an English label and
at least one English alias. A failed query is re-sent once with half the sample (read-only; recorded). The query text,
start time, row count and response MD5 of every query go to fetch_log.json; raw responses are kept (gzipped).
Then, in this order:
1. **Leakage filter.** PopQA = b49_out/b32_data/entities.json plus the raw subjects and aliases of
   b32_data/raw/popqa_test.tsv. An item is dropped if its URI is a PopQA subject or its normalized label equals the
   normalized form (b32_data.norm) of any PopQA label or alias; an alias is dropped if its normalized form equals that of
   any PopQA label or alias, is empty, or equals the item's own normalized label; an item left without an alias is
   dropped.
2. **Quotas.** Per class, survivors ordered by sha256('b73order|' + uri); the first 2,600 / 2,000 / 800 / 700 / 600 /
   400 / 700 / 600 / 600 kept (9,000 in all; a class with fewer survivors keeps all). An item in two classes counts
   under the first in this list.
3. **B32's filters, unchanged logic** (b32_data.build): entities whose normalized label is shared are dropped; an alias
   whose normalized form belongs to a different entity, or repeats one of the entity's own, is dropped.
4. **Split**: b32_data.split_of(uri) unchanged (train 30 %, dev 20 %, test 50 %, by sha256('b32|' + uri)). The dev
   split is never used (no dev store is built: nothing is re-chosen).
5. The entity file and string list are committed (with the fetch log) before any code is encoded.
If the fresh test split had fewer than 1,024 entities, make_store would take all of them (as B32's rule); the quotas
make about 4,500 test entities expected. Expected differences from PopQA, stated now: every fresh entity has at least
one alias at selection (PopQA: about half), so stores have more queries per stored entity; the classes' mix differs.

**Values and stores.** b32_common.make_store, unchanged except its seed tag 'store' -> 'B73store' (a new draw of the
stored entities and new random values; V = 2 and 16). Test stores per (M, V): 20 at M = 256, 5 at M = 1,024 (the
paper's counts), from the fresh test split; queries = every alias of a stored entity (unseen at write time).

**Codes: the same seven encoders, layers, pooling, recipes and model files as the paper.**
| enc | model (box path as in the paper's runs) | recipe | code |
| --- | --- | --- | --- |
| qwen8 | Qwen3-8B (HF snapshot b968826d...) | b32_embed.py lm, mean over the name's tokens in 'Name: {x}', layer 9 | float16, emb_qwen8_mean_L9_f16 |
| b46 | Granite-3.3-8B-Instruct (snapshot 51dd4bc2...) | b46_embed.py, mean, layer 30 | float16, emb_b46_granite33i_mean_L30_f16 |
| llama | Llama-3.1-8B-Instruct (/path/to/work/b40/llama31_8b_instruct) | b46_embed.py, mean, layer 24 | float16, emb_llama31i_mean_L24_f16 |
| mistral | Mistral-7B-Instruct-v0.3 (/path/to/work/b46/mistral7b_instruct_v03) | b46_embed.py, PromptEOL, layer 24 | float16, emb_mistral7i_eol_L24_f16 |
| e5 | intfloat/e5-base-v2 ('query: ') | b32_embed.py st | float32, emb_e5_base |
| bge | BAAI/bge-small-en-v1.5 | b32_embed.py st | float32, emb_bge_small |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | b32_embed.py st | float32, emb_minilm |
The layer and pooling are the paper's (chosen on PopQA dev in B32 / B46 / B49); they are not re-chosen.

**Side information from the fresh training split** (as PopQA's training split supplied it): the centring mean and ZCA
whitening (b32_common.Prep), the principal bases (b60_run.FeatC's U0, b60_vsapc's corrected basis, B57's LamFeat basis
for centred VSAPC), the LMMSE training statistics of VSAS / VSAPC (b56_run.train_feats), the SDMt hard locations (the
fresh training labels / aliases; N from B64's geometric grid over the fresh set's size). Nothing else is fitted.

## Frozen configurations (results/b73_frozen.json, committed with this file)

For every encoder and R1 cell (V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024; ideal accounting) each memory uses
EXACTLY the configuration chosen on PopQA dev in the paper's runs (the choice files' MD5s are in the JSON):
- **STORE** (final design): B53's '|VSA' choice (prep, feature map, q, stored share); b53_run.test's loop.
- **VSAR** (first design): B49's '|VSA' choice; the same loop with b32_run.Feat's random value codes.
- **VSAS**: B56's STAR choice (LM or RT) and that arm's configuration, on B53's bundle.
- **VSAPC**: B60's corrected-basis store, its PSTAR choice (PLM or PRT) and configuration on its matched-filter bundle.
- **VSAPCX** (the strongest variant): per cell VSAPC if its PopQA dev accuracy exceeded VSAS's by more than 0.001, else
  VSAS (b60_verdict.add_vsapx's rule, applied to the PopQA dev numbers, so the per-cell source is frozen: VSAPC in
  9 / 8 / 8 / 9 / 4 / 0 / 2 of the 12 cells for qwen8 / b46 / llama / mistral / e5 / bge / minilm).
- **LAMC**: B60's '|LAMC' choice (scale, target, prep, feature map, alpha, q).
- **NLMSC**: B62's '|NLMSC' choice (scale, target, prep, feature map, mu, output, q).
- **SDMT**: B64's '|SDMT' choice (prep, location set, a, counters, q, share).
- **Best of the six unsupervised classical memories (B36): NOT RUN** (declared now). Their test paths need per-encoder
  PQ / OPQ / ITQ fits and B36's six code paths; not feasible in the time box. No claim below uses them.
Mapping rule: every design is defined at every R1 budget by its own size rule (p or d = the largest that fits the
budget, capped at the code dimension where the design caps it; SDMt's N = the largest grid point that fits), so no frozen
design is missing at any R1 cell; the size rule is applied to the fresh stores, the configuration is not changed.
Code: each memory is run by its own module (b53_run, b32_run, b56_run, b57_run + b60_vsapc, b60_run, b62_run,
b64_run), called unchanged through b73_run.py, which only (i) points the modules' data roots at the fresh set, (ii)
changes make_store's seed tag, (iii) restricts the budgets to R1 and the accounting to ideal, (iv) runs only the chosen
read arm of VSAS / VSAPC (B56 / B57 computed every arm's chosen configuration; only the STAR / PSTAR one is used).

## Statistics and decision rule (the paper's)

Per encoder and contrast x - y (accuracy points): region statistic = mean over the region's cells of the cell mean of
the per-query difference, regions R1 (12 cells), R1-V2, R1-V16. One-level 95% interval = b36_verdict.boot_region
(entity clusters within store resampled within each cell, 10,000 draws); two-level = B66's scheme (test stores drawn
with replacement, then their entities; b66_storeboot.analyse, 10,000 draws). Label (B60's rule): "x leads" iff the
lower bound > 0, "y leads" iff the upper bound < 0, else "level". **Claims are decided on the one-level label over R1**
(the paper's registered label); two-level labels are reported beside them, and a claim that holds one-level but not
two-level is reported as such.

## Pre-specified confirmatory criteria (headline equal-bit claims)

Magnitude margin (all claims): on an encoder, the fresh R1 point estimate is "within the margin" if it lies within
+-2.0 points of the paper's PopQA R1 value, or within a factor of two of it with the same sign.

| claim | contrast (R1) | paper (PopQA) R1 per encoder: qwen8 / b46 / llama / mistral / e5 / bge / minilm | expected label | confirmed iff |
| --- | --- | --- | --- | --- |
| C1 least squares ahead of the final design on every encoder | STORE - LAMC | -4.06 / -5.55 / -5.56 / -5.23 / -6.53 / -4.18 / -4.30 (B60) | LAMC leads, 7 / 7 | label on 7 of 7 AND magnitude within the margin on >= 6 of 7 |
| C2 least squares ahead of the strongest variant on every encoder | VSAPCX - LAMC | -2.01 / -3.48 / -4.02 / -3.91 / -5.31 / -3.47 / -3.66 (B71) | LAMC leads, 7 / 7 | label on 7 of 7 AND margin on >= 6 of 7 |
| C3 one-pass online least squares ahead of the final design on every encoder | STORE - NLMSC | -2.11 / -2.87 / -1.51 / -4.79 / -3.48 / -2.02 / -2.16 (B62) | NLMSC leads, 7 / 7 | label on 7 of 7 AND margin on >= 6 of 7 |
| C4 strongest variant vs one-pass online least squares | VSAPCX - NLMSC | -0.06 level / -0.80 NLMSC / +0.03 level / -3.47 NLMSC / -2.26 NLMSC / -1.32 NLMSC / -1.52 NLMSC (B71) | the paper's label per encoder | VSAPCX leads on no encoder AND label equal to the paper's on >= 6 of 7 AND margin on >= 6 of 7 |

A claim is counted **confirmed** or **not confirmed** by these criteria only; the components (label count, margin
count, two-level labels) are reported for every claim. C1-C3 read as the paper words them ("on every encoder"); for C4
the paper's sentence is "level with the strongest variant on Qwen3-8B and Llama-3.1-8B and ahead by 0.8 to 3.5 points on
the other five".

**Secondary (reported, no claim decided on them):** the same contrasts on R1-V2 and R1-V16; VSAS - LAMC, VSAPC - LAMC,
VSAR - LAMC, STORE - VSAR, VSAS - NLMSC, NLMSC - LAMC; VSAR - SDMT (B64 / B66: SDMt ahead of the first design over R1 on
7 of 7: -1.89 / -2.25 / -1.20 / -3.56 / -0.92 / -4.73 / -3.80); every arm's mean R1 accuracy on the fresh set beside
PopQA's.

## Predictions (stated so that they can be wrong)

- Q1. C1 confirmed (LAMC leads on 7 of 7; margin on >= 6).
- Q2. C2 confirmed.
- Q3. C3: NLMSC leads on at least 6 of 7; I expect Llama-3.1-8B (PopQA -1.51, the smallest) to be the likeliest
  exception, and give C3 as a whole (7 of 7) about even odds; my point prediction is "confirmed".
- Q4. C4 NOT confirmed: the two "level" encoders on PopQA (intervals about +-0.6 around zero) and Granite (-0.80, level
  two-level) make at least two label changes likely; I predict VSAPCX leads on no encoder.
- Q5. Two-level labels equal one-level labels in at least 90 % of the 28 (claim x encoder) R1 cases.
- Q6. VSAR - SDMT: SDMt ahead over R1 on at least 5 of 7.
- Q7. Every arm's mean R1 accuracy on the fresh set lies within 10 points of its PopQA value on every encoder.
- Q8. Checks pass (plain-loop |difference| < 1e-9; 0 rows over budget, 0 not maximal; 0 leakage hits; determinism
  re-run identical; frozen-configuration audit 0 mismatches).

## Checks (b73_check.py; second angle)

- **Plain-loop scorer**: every cell accuracy and R1 / R1-V2 / R1-V16 point statistic recomputed with plain Python loops
  from the units' raw output files (not from the assembled records); |difference| < 1e-9.
- **Bits audit**: every LAMC / NLMSC / SDMT test row: bits <= budget and maximal (own re-implementation of each size
  rule); every STORE / VSAR / VSAS / VSAPC test configuration: its d by b32_common.vsa_dim re-derived and its bits
  (b32_common.vsa_bits) <= budget.
- **Frozen-configuration audit**: every configuration recorded in the outputs equals results/b73_frozen.json.
- **Leakage check** (b73_data.py leak): number of fresh URIs, exact strings and normalized strings found in PopQA's
  entity file and raw subjects (bar: 0 / 0 / 0).
- **Determinism**: a sample of units (one per arm, two encoders) re-run into a separate output directory; predictions
  identical.
- **Sample reading by the AI-assisted process**: Qwen3-8B and e5-base-v2, the R1-V16 cell with the lowest STORE - LAMC:
  10 queries only LAMC answers right and 10 only the store answers right (alias, label, gold, both answers), with notes.

## Compute and time rule

Fetch and build on this machine. Encoding on the A800 box (ssh cavin, in the background): GPUs only if idle at launch
by nvidia-smi (memory.used < 1 GB, no compute app), only GPUs 1, 2, 4, 5, 6, 7 (never 0 or 3), the paper's recipes and
model files; GPU-hours logged. Test runs on the box CPU (fresh directory ~/b73_run, code by MD5 from this repository),
at most 48 worker processes, one process per unit (OMP / BLAS threads 1, nice 5), next to B72's jobs; verdict and
checks locally (at most 12 workers). Each memory and cell is run once on the test stores; a compute note is written
before any test output is read. Time box: 4 hours from about 19:17 (ends about 23:15). If time runs short, cut in a
compute note BEFORE reading any test output, in this order: (1) SDMt on the sentence encoders (the best of six is
already not run), (2) SDMt on the language models and VSAR (no claim uses them), (3) Granite-3.3-8B. The language-model
core (STORE, VSAPCX = VSAS + VSAPC, LAMC, NLMSC on Qwen3-8B, Llama-3.1-8B, Mistral-7B) runs first, then the sentence
encoders, then Granite (b73_run.py plan's order). A claim whose encoders are cut is reported "not decided" on the
encoders run, not "confirmed".

## Files

b73_data.py (fetch, build, leakage check, synthetic self-test data), b73_run.py (frozen table, units, plan, assembly),
b73_box.sh, results/b73_frozen.json; after this file: b73_verdict.py, b73_check.py, results/b73_*; data and job
outputs under b73_out/ (the fresh entity file and fetch log committed; codes and job files not committed).
