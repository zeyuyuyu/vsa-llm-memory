# B48 pre-registration: arena C with one surface form per symbol (A), and naming where identity is not in the surface string (B)

Date: 2026-11-19, about 06:30 UTC (14:30 on the box clock, Asia/Shanghai; US Eastern 2026-11-19 01:30). Written, with
all B48 code, before any B48 run reads a test label or scores a test instance. What was run before this file (train and
dev splits only; test features without labels) is listed under "Done before registration". The file is committed and
time-stamped with OpenTimestamps before any test run.

## Why (referee, 7th read, M4)

The paper says a frozen LM can use a VSA memory only once it names each symbol. The referee: (1) in arena C this is
partly built in, since "a symbol is written in decimal as a filler and in hexadecimal as a role and nothing without
labels links the two forms" (SI Note 10), so no label-free method could discover the identity from 800 binary
answers; (2) the naming head is effectively a learned tokenizer look-up over bracketed tokens, and naming is never
tested where identity is not in the surface string (aliases, coreference). Requested: (A) an arena C variant with ONE
surface form per symbol, reported for all label-free interfaces; (B) a naming experiment where identity cannot be
recovered from the surface string, comparing the naming head, string hashing and LM-written canonical strings. B48's
result goes into the paper whatever it shows.

## Part A: arena C1 (one surface form per symbol)

### Arena

`b48_struct.build_C1`: arena C's generator (v13.Arena, line by line as b31_struct.build_C) with the same random streams,
chains, labels, candidates, pair orders, sizes (train 800 / dev 500 / test 500, seeds 42 / 777 / 4242) and question
position. Only the role digits change: every symbol is written in decimal everywhere, as a role (`[R63]`, `[F63]`), as
a filler (`{63}`) and in the question (`Q [R63] {50} ?`); arena C writes roles in hexadecimal (`[R3F]`). Decimal (not
hexadecimal) is used because digits are always their own pre-tokens in both tokenizers (Phi-2 / GPT-2 BPE: a digit run;
Qwen3: one digit per token), so each symbol has the same token ids in every position; with hexadecimal, letters merge
with the level letter (`[RAB` vs `{AB`). The self-test (`b48_prep.py selftest`) checks, per split and tokenizer: chain,
label and structure equal arena C's instance by instance; dictionary answer = label; the number of symbols written with
more than one token form (C1 and, for comparison, C); no truncation.

### Encoders and features

Phi-2 (fp16 `last_hidden_state`, rows padded to 768 as the published rows, batch 4, span means in fp32 as the published
arms take them) and Qwen3-8B (`Qwen/Qwen3-8B` snapshot b968826d, bf16, after the final norm; b31_prep.feats unchanged),
computed for C1 by `b48_prep.py featsC1` on PACE (arena C's Phi-2 features came from the B27 caches on the A800 box; the
C1 features are new for both encoders, and every C1 arm reads the same C1 features).

### Arms (all as B31 / B42, run unchanged through `b48_run.py`, which only swaps arena C for C1)

- **References on C1:** equalised probe F (B31's grid and rule, b31_probe.py, dev-selected, 3 seeds); naming head N
  (B31: identity labels on every mention, 4 epochs, perm / named); exact symbols + VSA (perm; in C1 this is also
  **string hashing** of the written symbol over the hand-written spans, since one written form = one symbol);
  binding-loss map (B31 form, lr in {3e-4, 1e-3} chosen on C1 dev, B31's rule; read also with post-hoc centring and
  ZCA as B42's bind+pc / bind+pz).
- **The seven B42 label-free interfaces:** vq, dkvb, gumbel, st, cw, bind+pc, bind+pz, with B42's registered grids
  (b42_queues.grid; cw at B42's rate, Phi-2 3e-4, Qwen3-8B 1e-3), B42's selection rule (highest best-epoch dev
  accuracy, perm/named, ties within 0.002 to the first listed), seeds 42 / 123 / 456, test at each seed's dev-best epoch,
  training clean-up hard or straight-through as in the grid; evaluation through the published hard chain.
- **Gates:** G0: B42's code path (b42_arms bind) reproduces B31's (b31_arms bind) on C1 dev at the same rate, epoch by
  epoch (max |dev difference| <= 0.002); G1: gumbel and st with the identity loss added (labelled positive control)
  reach the C1 naming head's best dev within 0.02. A failed gate is reported; the arm's result is then not attributed to
  missing labels alone.
- **String hashing in arena C (two forms), for contrast:** codes = fixed random rows per written form (kind, symbol)
  with no learning, the published chain, threshold 0.5; computed by the second scorer (numpy) on C's test split.

### Measurements

As B42: accuracy per seed (Wilson 95%), mean with a bootstrap over test instances (10,000 resamples, paired across seeds
and arms), recovery R = (mean - F) / (N - F) with a paired bootstrap interval (F, N = the C1 probe and naming head on the
same encoder), cross-talk geometry (B31 / V-165 protocol), discrete statistics (mention consistency, code purity, codes
used; the hex / decimal alias rate is not defined in C1), clean-up correctness at hop 1 and hop 2.

### Bar and readings (fixed now)

- A label-free arm **MATCHES** in C1 if its mean recovery is >= 0.90 on **both** encoders (per-seed R_s >= 0.80 in 3/3
  reported as a flag); PARTIAL R >= 0.25 on an encoder; FAIL R < 0.25. If N - F < 0.10 on an encoder, recovery is
  undefined there and the arm is compared with F and N directly.
- **Answer A:** "with one surface form per symbol, a label-free interface on the frozen states reaches the naming head"
  = YES iff at least one of the seven arms MATCHES.
- Readings:
  - YES: the paper must say that arena C's two written forms (decimal filler, hexadecimal role) were what defeated the
    label-free interfaces there (the referee's point), name the interface that suffices with one form, and restrict
    "only once it names each symbol" to symbols with more than one written form.
  - NO, every arm FAIL: the referee's mechanism is not what defeats the label-free interfaces; with one written form
    per symbol none of the seven recovers 25% of the floor-to-naming-head gap on either encoder (best recovery and
    interval given). The paper must still add that in C1 hashing the written symbol over the hand-written spans reaches
    the exact-symbol ceiling with no labels and no LM state, so the claim is about codes read from the LM's states,
    and that where identity is in the surface string a parser suffices (naming adds nothing there).
  - PARTIAL somewhere: the recovery and its interval are reported and "label-free codes fail" is not said for C1.

## Part B: naming where identity is not in the surface string (alias stories)

### Data (`b48_struct.build_B`)

Two datasets of stories over K = 16 entities, listed at the top ("Entities: a; b; ..." / "Characters: ..."), then 16
level-0 statements (every entity once as subject, object another entity, uniform), then 12 level-1 statements (12 of the
16 entities as subjects, always including x1, objects another entity), then a yes/no question about a two-hop chain
(q -> x1 at level 0, x1 -> x2 at level 1; candidate = x2 with probability 0.5, otherwise another entity). The memory has
to bind the level-1 subject to x1 across two different written forms, as arena C's chain does across decimal and
hexadecimal.

- **W (Wikidata aliases):** entities from b32_data/entities.json (PopQA subjects, B32's entity split by URI hash; train
  462 / dev 270 / test 717 usable entities). Level-0 mentions use the listed name; every level-1 subject is a Wikidata
  alias of the entity that shares **no normalised word** with the listed name (e.g. "DPRK" for North Korea, "Pleasant
  Island" for Nauru, "Hollandia" for Jayapura), chosen at random per mention. Template: "A is linked to B." / "<alias of
  B> is paired with C." / "Question: is the entity that A is linked to paired with C?". Identity is not in the surface
  string; it can come only from the LM's world knowledge (the alias is never declared in the story).
- **P (pronouns, bAbI task-11 style):** 104 female and 104 male first names (hard-coded), split by a hash of the name
  (train 52 / dev 55 / test 101 names); 8 female and 8 male names per story. Level-1 unit: "B went to the <place>. Then
  <she|he> visited C." (the pronoun refers to the subject of the sentence just before it; the binding is pronoun ->
  C). Template level 0: "A called B."; question: "did the person whom A called visit C?". Identity comes from the
  context only.
- Splits: train 800, dev 500, test 500 stories per dataset (generator seeds 4801 / 4802 / 4803, +100 for P). **Dev and
  test stories contain only entities (names) never seen in training.**
- Mention types: roster (the list), can (listed-name mentions in statements and the question), alias (W level-1
  subjects), pron (P level-1 subjects), ante (P antecedents, not in the chain).

### Memory and scoring

Every arm gives each mention a key; a key's code is a fixed random unit vector (d = 2048; generator seeded by the codebook
seed and md5 of the key). The codes go into arena C's two-hop HRR chain (perm encoding with B31's permutation seed + 7001;
clean-up by arg-max over the level's object codes after each hop; score = cosine of the hop-2 winner's code with the
candidate's code); answer "yes" iff score >= 0.5 (all codes are snapped unit vectors: a matching code gives 1, others
|cos| < 0.1). Codebook seeds 42, 123, 456.

### Arms

| key | arm | key of a mention | trained on |
|---|---|---|---|
| oracle | exact identity | the gold entity | nothing |
| hash | exact string hashing | norm(written string) (b32_data.norm); an alias or pronoun is a different key **by design** | nothing |
| lm | LM-written canonical names | Qwen3-8B (same snapshot, chat template, thinking off, greedy, max 1,600 new tokens) reads the story with its 28 statements numbered and writes, per statement, the two participants by their listed names (`b48_lmcanon.py`, prompt fixed in the code); key = norm(written name); question mentions: norm(written string); an unparsed or missing name gets a key of its own | nothing (prompted) |
| head | naming head on frozen Qwen3-8B states (open-vocabulary form) | arg-max over the 16 listed names of a bilinear pointer between the mention's span-mean state and the span-mean states of the listed names (each dimension standardised by training statistics; rank 256); key = the chosen entity | identity labels (gold slot of every mention) on training stories, cross-entropy, AdamW, batch 4 stories, clip 1.0, 10 epochs |

The paper's naming head is a classifier over a fixed inventory of 64 symbols, which cannot name an entity it has never
seen; the pointer form is the same idea (a head on the frozen span states that picks a discrete symbol, trained with
identity labels) made open-vocabulary, so that it can be tested on unseen entities. **What can generalise:** the head
learns one mapping on training entities and is applied to stories whose 16 entities were all absent from training; it
can succeed only if the frozen states of an alias or pronoun carry the identity of its referent in a form shared across
entities.

Head tuning (seed 42, train + dev): layer of the states in {9, 18, 27, 36 (= final norm output)} x lr in {3e-4, 1e-3}
(8 per dataset). Selection: highest best-epoch dev chain-naming accuracy (share of the chain's mentions -- level-0 and
level-1 subjects and objects, question, candidate -- named right); ties within 0.002 to the earlier grid entry (layers
in the order above, then 3e-4). Finals: seeds 42 / 123 / 456 (head seed s with codebook seed s), test read at each
seed's dev-best epoch (same criterion).

### Measurements

- **Naming accuracy per mention type** (roster, can, alias / pron, ante, q, cand) on test, per arm: share of mentions
  that get the key the arm gives the entity's listed name (oracle, head: the gold entity; hash, lm: norm(listed name)).
  Intervals: bootstrap over stories (10,000 resamples) and Wilson over mentions.
- **End accuracy** per codebook seed (Wilson 95%) and mean (bootstrap over stories, paired across arms and seeds); hop-1
  and hop-2 clean-up correctness.
- Recovery R_B = (mean - hash) / (oracle - hash) per arm and dataset, paired bootstrap interval.
- LM: parsed-line rate (lines of the 28 found), share of written names that are exactly a listed name.

### Bars and readings (fixed now)

- Per arm (head, lm) and dataset: **MATCH** if mean R_B >= 0.90; PARTIAL R_B >= 0.25; FAIL otherwise.
- **Names the hard mentions:** naming accuracy on the dataset's hard mentions (W alias, P pron) >= 0.90.
- Comparison head - lm: paired bootstrap of the end-accuracy difference and of the hard-mention naming difference.
- Readings:
  - head MATCH on a dataset: the paper may say that a naming head trained with identity labels names unseen entities'
    aliases (W) / pronouns (P) from the frozen states, with the numbers, and must say that this is the pointer form,
    not the 64-symbol classifier of the arenas.
  - head FAIL / PARTIAL on W but MATCH on P: naming from the context (coreference) works from frozen states, naming
    from world knowledge (undeclared aliases) does not, at this training size.
  - head FAIL on both: the paper must say that the naming head was shown to work only where identity is in the surface
    string, i.e. the referee's "learned tokenizer look-up" reading is not refuted.
  - lm MATCH: an LM that writes the listed name for each mention, followed by string hashing, is a training-free way
    to name symbols whose identity is not in the surface string; the paper says so. lm beating head (interval excludes
    0) means the paper must not present the naming head as the best available route.
  - hash ~ chance on both datasets is a check of the design (it fails by construction) and is reported as such.

## Predictions (stated so they can be wrong)

Part A
- PA1 Gates: G0 passes on both encoders; G1 passes for gumbel and st on both encoders (as in B42's arena C).
- PA2 C1 naming head (perm / named) mean >= 0.98 on Phi-2 and >= 0.95 on Qwen3-8B; exact symbols >= 0.99 per seed.
- PA3 C1 probe mean <= 0.60 on both encoders.
- PA4 Answer A = NO: no label-free arm MATCHES on both encoders; none MATCHES on either encoder.
- PA5 every label-free arm's C1 recovery is < 0.25 on both encoders, except at most one (arm, encoder) cell.
- PA6 the C1 binding-loss map COLLAPSES (different-symbol cosine >= 0.50 in 3/3 seeds) and FAILS on both encoders.
- PA7 string hashing: arena C (two forms) <= 0.60 per codebook seed; arena C1 (= exact) >= 0.99.
- PA8 discrete arms in C1: mention consistency <= 0.60 in every seed for vq and dkvb (as in C, where it was 0.19-0.50).

Part B
- PB1 oracle >= 0.99 per seed on W and P; hash <= 0.62 per seed on W and P.
- PB2 lm naming: P pronouns >= 0.95; W aliases in [0.40, 0.85]; listed-name mentions >= 0.98 on both.
- PB3 head naming: P pronouns >= 0.90; W aliases <= 0.60; listed-name mentions >= 0.97 on both.
- PB4 P: head MATCH and lm MATCH. W: head FAIL or PARTIAL; lm PARTIAL.
- PB5 W: lm end accuracy > head end accuracy (paired interval excludes 0).
- PB6 the dev-selected head layer is an intermediate one (9, 18 or 27) on W.
- PB7 total GPU-hours (PACE + A800) <= 8.

## Checks from other angles (in the verdict)

- **Recomputation:** `b48_verdict.py` recomputes every part-A accuracy from the saved scores and thresholds against
  labels regenerated by b48_struct (flags differences, including threshold ties as B42 note 1) and every part-B
  accuracy from the saved scores.
- **Second scorer:** `b48_indep.py`, numpy float64, its own regex parse of the raw text (C1: decimal roles and fillers;
  B: the statements of each story from the text and the listed names, gold answers by a dictionary walk over the
  parsed identities), the HRR chain written out per instance. A: per-instance agreement for every discrete final
  (rebuilt from the saved assignments, as B42) and accuracy from saved scores for all finals; the C string-hash
  contrast. B: every arm's chain recomputed from the keys rebuilt from the raw text and the saved head predictions /
  LM outputs; per-story agreement with the main scorer.
- **Hand-reading:** `b48_handread.py`: A: >= 10 C1 test failures per encoder of the label-free arm with the highest
  selected dev accuracy (which hop failed; the codes of the query role and its context role). B: >= 20 W test alias
  mentions misnamed by the head and >= 20 by the lm, >= 10 P pronoun errors per arm if that many exist, and 10 random
  correct alias namings per arm, read against the raw story.
- **Literature** (after the verdict): entity linking / alias resolution with LM states, coreference probes of frozen
  LMs (e.g. binding IDs, entity tracking), LM-based canonicalisation for knowledge bases; marked supports / conflicts /
  no precedent.

## Compute

GPU work on PACE Phoenix (free tier gts-nimam6; jobs <= 1 h for backfill; no A800 GPU was idle at 05:15 UTC: GPUs 0, 2,
4, 6 held by another user's jobs and 1, 5, 7 by B44; GPU 3 is never used). If an A800 GPU becomes idle (shared claim
protocol), queued work may move there; every move is recorded in a compute note, no run is executed twice. Projection:
features 0.5 GPU-h, C1 tuning 2-3 GPU-h, C1 finals 1-2 GPU-h, B heads 0.3 GPU-h, LM generation 1-1.5 GPU-h. Cap 40
GPU-hours. No paid API.

## Done before registration (no test score, no test feature, no test output of any arm)

- **Code and local smokes (local CPU, no LM state):** story generation for W and P, all splits (dictionary answer =
  label for every story; every listed-name mention equals its entity's name; every alias differs from it after
  normalisation); the oracle and hash arms on the **dev** split (oracle 1.000 on W and P; hash W 0.534 / 0.558 / 0.550,
  P 0.576 / 0.562 / 0.562 per codebook seed); the part-B second scorer on these dev outputs (agreement 1.000, max score
  difference 1e-6); the LM prompt rendered for two dev stories (no LM run).
- **One design change during the smokes, before any LM or head output existed:** the first version had every entity
  as a level-1 subject once (16 of 16), which makes the 16 aliases a permutation of the list, so an LM could match them
  by elimination; it is now 12 of 16 (always including x1). The oracle / hash dev numbers above are of the final version.
- **PACE jobs submitted, none started at registration:** the feature job (self-tests; C1 features of both encoders and
  story features, all splits including test -- frozen states only, no label is read), the LM-written-names run on the
  dev split of W and P, and the tuning queues (part A: G0, G1, the naming head's dev run, the binding-loss lr, the probe
  grid and B42's seven grids on C1 dev; part B: the head grid on dev). The free tier (gts-nimam6) had no billing
  minutes left (AssocGrpBillingMinutes), so these jobs run on gts-nimam6-paid (Imam's approval of 2026-10-18; spend is
  reported in the verdict).
- **Not run in C1:** B42's labelled diagnostic dkvbor (keys = written form). In C1 the written form is the symbol, so
  those keys are identity labels; its role (what a perfect label-free clustering of the written forms gives) is taken
  by string hashing (= exact symbols in C1) and by the string-hash contrast in arena C.

## Files

Code: b48_struct.py, b48_prep.py, b48_run.py, b48_queues.py, b48_select.py, b48_alias.py, b48_lmcanon.py,
b48_pace.sbatch, b48_feats.sh, b48_verdict.py, b48_indep.py, b48_handread.py; reused unchanged: b31_arms.py,
b31_probe.py, b31_struct.py, b31_prep.py, b42_arms.py, b42_queues.py, b27_common.py, b27_arena/, b32_data/entities.json.
PACE: /path/to/pace-scratch/b48 (feats/, out/, logs/, q/). Results: results/b48_verdict.txt,
results/b48_scores.json, per-run outputs in b48_out/ (gzipped).
