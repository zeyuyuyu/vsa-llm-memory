# B26 pre-registration: does the vector form itself beat an equally exact table? (referee's central objection)

Date: 2026-11-17 (US Eastern; 2026-11-18 on the box clock). Written before any B26 arm has produced an answer on dev or
test data, and before any B26 GPU run.

Done before this file (no arm output on dev or test data):
- b26_fuzzy.py (variants, the seven arms, dev grid, write-side prompt builder, splicer), b26_read.py (reader replica)
  and b26b_pcm_table.py (part b) were written. Self-tests ran on hand-made and synthetic pairs only.
- Pipeline checks on the dev rows (b24_dev/reader): the RULER dev inputs align with the rows (index and gold), the
  B21 chunker reproduces the number of banked reader outputs on all 360 rows, and every queried key sits in exactly one
  chunk (four for niah_multivalue).
- Sizing on dev and test inputs (no model, no answer): the write-side prompt sets were built (dev 560 chunks per
  condition, 2,222 unique prompts; test about 8,400 chunks per condition).
- Timing only (no accuracy printed) on one synthetic stress store of 1,500 and one of 6,000 keys.
- The part-b self-test ran on a synthetic 1,500-pair store; where it informed a part-b prediction, the prediction says so.

## Why

A referee of the npj Unconventional Computing manuscript ("Vector symbolic memory lets small language models match much
larger ones"): on every benchmark an equally exact table gives identical answers (BABILong: orthogonalised codes are a
lookup; RULER: VSA store == table on 5,400 / 5,400 samples; CLUTRR and the logs: a plain table gives the same answers).
"Show, on at least one headline benchmark, that the vector form itself (superposition plus similarity clean-up) beats an
equally exact table", for example with "similarity-preserving codes with paraphrased or misspelled entity names, or a
fixed memory budget". B26a tests the first, B26b the second. Both go into the paper whatever they show.

## Part a (primary): entity names that vary in surface form

### Benchmark and why

- **RULER, the eight needle tasks** (niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery) at 4k, 8k,
  16k, 32k, 64k and 128k tokens, 100 samples per task and length: 4,800 samples, exactly B21's.
- **Why RULER.**
  - It is the benchmark of the referee's sharpest example (B21: the vector store and the table agree on 5,400 / 5,400).
  - Its memory is a key -> value store whose keys are entity names (adjective-noun names such as "jittery-hospital",
    or uuids), and the question names the key, so a surface variant of a name is a variant of the store's look-up key.
  - The pipeline is the easiest to reuse: B21 banked the reader's raw output per chunk, so every store can be rebuilt on
    CPU (query side), and a variant in the text needs only the chunks that hold the varied needle re-read (write side).
  - Its stores range from 1 to about 6,000 keys per text. RULER draws adjective-noun keys from 912 adjectives and
    6,782 nouns, so at 128k each adjective recurs about six times among niah_multikey_2's keys and many nouns twice: a
    crowded vocabulary in which a fuzzy match can pick a wrong but similar key.
- **Why not the others.** CLUTRR's vector holds composition rules over about 20 relation types; people's names go to a
  graph, not to the vector, so a varied name would test the graph. BABILong's matrix holds a handful of bAbI names per
  story with orthogonalised codes, and re-reading varied texts would need the fine-tuned reader over up to 1M tokens.
  The operational-log pipeline lives in read-only directories (nsr/, apps/real/) and its messages are model-written.
- **VT is excluded:** its question names a value, not an entity, and its answer is a list of variable names, so a
  varied name would change the gold answer.
- **Limitation, stated now:** the keys and the variants are synthetic. A typo, a format change and a short alias of a
  name are realistic kinds of variation; their rates are ours.

### Variation conditions (fixed now)

Applied to every queried key (niah_multiquery: each of its four keys independently). Each variant is drawn from a seed
that hashes (condition, split, length, task, position in RULER's file, key position), so every arm sees the same variant
and the query and write sides see the same variant string.
- **C0:** unchanged (control).
- **T1, T2, T3 (typos at three rates):** n = max(1, floor(rho x len(key) + 0.5)) edits with rho = 0.05, 0.10, 0.20
  (an adjective-noun key of 15 characters: 1, 2, 3 edits; a 36-character uuid: 2, 4, 7). Each edit is, with equal
  probability, a substitution by a keyboard neighbour (a random other hex digit in a uuid), a deletion, an insertion of
  a keyboard neighbour next to the character, or a transposition of two adjacent characters, at a random alphanumeric
  position. Separators are not edited (they are condition P). A variant equal to the key is redrawn.
- **P (format):** adjective-noun keys: the adjective-noun hyphen becomes one of " ", "_", "" or " - ", and the case
  one of lower, Title or UPPER. Uuids: dashes removed or replaced by spaces, lower or upper case.
- **A (short alias):** adjective-noun keys: the noun alone ("hospital" for "jittery-hospital"); uuids: the first eight hex
  digits (a short id, as for commit hashes). Adjective and noun are split with RULER's word lists (either may contain a
  hyphen). Aliases can be ambiguous in crowded stores; the gold answer stays that of the full key.
- **Query side (primary):** the question names the variant; the store holds what the reader read from the unchanged text
  (B21's banked pairs). CPU only. Conditions C0, T1, T2, T3, P, A.
- **Write side (secondary):** the text names the variant at every occurrence of the queried key (the same variant
  everywhere); the question is unchanged. B21's chunk boundaries are kept; only the chunks that hold a queried key are
  re-read by B21's reader (Qwen3-8B, prompt and decoding unchanged; b26_read.py is a standalone copy of its generation
  path); B21's grounding is replayed over the new chunks. Conditions C0 (the same chunks re-read unchanged: a control
  for re-reading), T2, P, A.

### Arms (all on the same reader output; keys are compared in lower case, as B21 stores them)

Every arm returns, for each queried key, the values of at most one stored key, so no arm can collect values from several
candidate keys. The candidate keys are the keys that the reader wrote for this text (the closed vocabulary), in
first-read order; ties go to the earliest. No arm has an acceptance threshold for keys: under RULER's metric
(string_match_all, which ignores extra values) a threshold could only lower accuracy, so every fuzzy arm takes its top
candidate.
- **E, the exact table:** B21's table and look-up, unchanged (ruler_rm.answer_niah).
- **Ev:** B21's random-code VSA store, unchanged (ruler_vsa.answer_niah_vsa).
- **En:** the exact table after normalization (lower case; every run of non-alphanumeric characters becomes one space).
- **L, the mandatory fair table:** the table plus the nearest key by edit distance over normalized strings (rapidfuzz
  3.14.5). Variant chosen on dev from: Levenshtein distance; Levenshtein distance divided by the longer length; optimal
  string alignment distance (adjacent transpositions cost 1).
- **W, a stronger standard fuzzy table:** the table plus the best key by rapidfuzz's WRatio with its default processor
  (lower case, non-alphanumerics to spaces). WRatio combines whole-string, best-substring and token-order-free
  similarities; it is fuzzywuzzy/rapidfuzz's general-purpose default. No tuning.
- **N, V's exact twin:** the table plus the key with the highest cosine between character n-gram count vectors, with the
  n-gram set chosen for V. It computes exactly the similarity that V's codes approximate, without vectors.
- **V, the vector form:** the code of a string is the bundle (sum) of the Fourier HRR codes (unit-modulus complex,
  d = 8,192, seeded per n-gram) of its character n-grams, taken over the normalized string padded with one space at each
  end, and projected back to unit modulus. The queried key's code is cleaned up against the codes of the stored keys
  (highest real inner product), and the values of that key are read by B21's read (unbind the key code and position j
  from its shard, clean up against the text's value codes, accept at >= 0.5, stop at the first j below). Chosen on dev:
  the n-gram set ({3}, {2,3}, {1,2,3} or {2,3,4}) and whether the store binds keys by their n-gram codes ("ng", one code
  per symbol everywhere) or by B21's random codes ("rand", the n-gram codes used only for the clean-up).

### Development (fixed now; the only tuning)

- **Dev data**, no test data:
  - "real": B21's reader and store on the seed-7 dev set at 4k and 16k (b24_dev/reader, 20 samples per task, the
    eight tasks: 320 samples), with the RULER dev inputs for the questions.
  - "stress": synthetic stores built with RULER's own key generators, at the sizes of the test set's largest stores and
    smaller: adjective-noun keys with 7-digit values (6,000, 1,500 and 200 keys) and uuid keys with uuid values (1,700
    and 400 keys), 40 stores per size, one queried key per store (200 stores). No reader.
- **Grid:** V over 4 n-gram sets x 2 binding codes; L over its 3 distances; N with each n-gram set. Query side only.
- **Dev score:** for each arm setting, the mean over the two dev parts of its mean accuracy over the five variant
  conditions (T1, T2, T3, P, A).
- **Choice:** the highest dev score; any setting within 0.1 points of the best counts as tied, and ties go to the first
  in the order listed above (smaller n-gram set first; "ng" before "rand"; Levenshtein, then normalized, then OSA). N
  takes V's n-gram set. The choice is written to results/b26_dev_choice.json and committed before any test run.
- **Write-side dev** (re-read of the dev chunks): a pipeline check only. If fewer than 95% of the dev chunks re-read in
  condition C0 reproduce B21's banked output, the reader replica is investigated before any test re-read is used. It
  tunes nothing.

### Metric and statistics

- **Primary metric:** RULER's official string_match_all (the B21 scorer), per sample in [0, 1]; cells are task x length
  x condition, means are over the eight tasks.
- **Second scorer:** key choice. For each queried key, whether the arm chose the gold key (the original key, lower case)
  among the stored keys. And a strict answer scorer: the predicted values equal the gold values as a set. Disagreements
  with the primary metric are reported.
- **Paired statistics** (all arms on the same samples): the mean difference in points with a 95% paired bootstrap
  interval (20,000 resamples of samples, seed 0; for pooled conditions, a sample's five conditions are resampled
  together), and the exact two-sided sign test on the samples whose scores differ (counts of each direction). By
  condition, by length (n = 800) and by task (n = 600).

### Bars (query side, the 4,800 test samples)

- **Bar 0 (validity, C0):** in C0, every arm's per-sample score equals E's on >= 99.9% of samples, and in every condition
  Ev's score equals E's on >= 99.9% of samples. A failure means an arm is implemented wrongly; it is fixed and the run
  repeated before any variant result is read.
- **Bar 1 (similarity codes rescue the exact table):** pooled over T1, T2, T3, P and A, V - E >= 50 points with the
  interval above 0.
- **Bar 2 (primary; the referee's test):** pooled over T1, T2, T3, P and A, V - L >= +1.0 point and the 95% interval
  excludes 0.
- **Bar 3:** the same against W: V - W >= +1.0 point and the interval excludes 0.
- **Reporting rule:** no advantage of either arm is claimed in a condition whose interval includes 0.

### How each outcome will be read (fixed now)

- **Bars 2 and 3 pass:** on this benchmark the vector form beats an equally exact table with either standard fuzzy
  matcher. The paper says so and names the conditions where the gain lies.
- **Bar 2 passes and bar 3 fails:** the VSA beats plain edit distance but not a stronger standard fuzzy matcher. The
  advantage is then the similarity built into its codes, which a table with a standard fuzzy matcher shares, and the
  paper may not claim that the vector form beats a fair table.
- **Bar 2 fails:** no advantage over the mandatory fair table. If the interval lies below 0, the fair table is better.
- **V against N (descriptive, registered now):** if |V - N| <= 0.5 points pooled, V behaves as character n-gram cosine,
  and the superposed codes add nothing beyond that similarity; the paper says so.
- **Write side:** the same bars are computed and reported as secondary. They cannot overturn the primary reading; a
  disagreement between the sides is reported.

### Predictions (part a; stated so they can be wrong)

- Bar 0 passes; every arm equals E on all 4,800 C0 samples, and Ev equals E in every condition.
- E (and Ev) score 0.0 in every variant condition (a varied key is never stored as such). En scores 65-80 in P
  (three of four separators normalize to the stored form; uuids with spaces do too; about 72 expected) and <= 0.5 in
  T1-T3 and A.
- V, query side, mean over tasks and lengths: T1 >= 99.0, T2 >= 98.5, T3 >= 96.0, P >= 99.3, A 93 (85-98).
- Typos (T1-T3): L is equal to V or better, by at most 1.0 point pooled over T1-T3 (0 <= L - V <= 1.0); the largest
  per-cell gap L - V in T1-T3 is on niah_multikey_2 at 64k or 128k, where similar keys crowd.
- P: V, L, W and N all within 0.3 points of their C0 accuracy.
- A: V beats L by >= 10 points (raw edit distance prefers short keys over the key that contains the alias); W is
  within 3 points of V.
- Bar 1 passes. Bar 2 passes, carried by A. Bar 3 fails (V and W tie: interval includes 0).
- |V - N| <= 0.3 points pooled.
- V's alias accuracy falls with length on niah_multikey_2: >= 95 at 4k, <= 80 at 128k (at 128k a noun is shared by
  another key in more than half of the stores).
- Dev choice: V with {2,3} or {3} and "ng" (binding by n-gram codes changes nothing once the key is cleaned up); L: OSA.
- Write side: the reader copies the varied key so that it is stored in >= 95% of T2 needles and >= 98% of P and A
  needles; the C0 re-read reproduces B21's banked chunk output on >= 95% of chunks; the bars read as on the query side.

## Part b (secondary): the store and a table on the same simulated device

### Design (fixed now)

- **Stores and keys:** B25's: the 50 B22 stores (b21_dl/b21_test), the same 4,776 probed keys, at 16 pairs per vector.
- **Device:** B25's model (Joshi et al. 2020) unchanged: programming noise, per-device drift exponent, 1/f read noise,
  8-bit ADC, every cell a differential pair (+1 -> (G_ON, RESET), -1 -> (RESET, G_ON)) read as a sign, ADC zero -> random.
  Global drift compensation is irrelevant to sign reads.
- **Arms:**
  - VSA: the bipolar store (B25's B1) with its store crossbar on the device and the clean-up digital (B25 stage S).
  - T1, the table at its native cell count: one row per (key, j) holding the index of the j-th value in the store's value
    vocabulary in ceil(log2 N_values) bits, one cell per bit.
  - Trep, the table with the VSA store's cell count: every bit repeated r = floor(VSA cells / native table cells) times
    (VSA cells = shards x 8,192), read by majority vote, ties at random. For the largest stores r is about 39.
  - Both tables keep the key -> row directory and each key's number of values digital (the VSA store keeps neither,
    so this favours the tables). The device model has no spatial correlation, so the placement of the copies does not
    matter.
- **Grid:** times 60 s, 1 h, 1 day, 30 days, 1 year (B25's) x device-noise multiplier k = 1 (the measured model), 2, 4,
  8, 16, 32 (B25's descriptive multipliers, applied to every noise source at once). Common random numbers: the same
  underlying device draws at every k.
- **Metric:** B25's: the share of the 4,776 probed keys read exactly (all values, in order).

### Bars (part b)

- **H0 (check):** at k = 1 every arm reads >= 99.5% of keys at every time.
- **H1 (primary):** at the same cell count the vector store is at least as robust as the table: VSA >= Trep at every
  (k, time) of the grid.
- **H2:** the vector store beats the unprotected table at k = 8, 30 days, by >= 5 points.

### Predictions (part b; informed by B25's verdict and by the synthetic self-test)

- H0 passes: no bit errors at k = 1 (B25: B1 store bit errors 0 at every time).
- B25's bit-error rates at 30 days (2.1% at k = 4, 12% at k = 8, 28% at k = 16, 38% at k = 32) set the tables. At 30
  days: T1 reads 60-90% at k = 4, 5-35% at k = 8 and <= 5% at k = 16; the VSA reads >= 99.5% at k = 4, 95-100% at
  k = 8 and < 1% at k = 16; Trep reads >= 99.5% up to k = 8, >= 95% at k = 16 and 30-80% at k = 32.
- H1 FAILS (Trep beats the VSA at k = 16 and 32, at every time). H2 passes.
- Reading if so: the store is robust because it is redundant; spent as a plain repetition code, the same cells protect a
  table better. The paper may not claim a robustness advantage of the vector form at a fixed cell budget.

## Hand checks and other angles (both parts)

- Read by hand, query side: 10 samples per variant condition where V and L disagree, and 10 where V and W disagree.
- Recompute E's C0 accuracy per length from the run and compare with B21's verdict (99.5-99.9 on the needle tasks).
- Write side: the share of re-read C0 chunks identical to B21's banked output; on the rows where they differ, whether
  the grounded pairs differ.
- Literature: published comparisons of string-similarity measures for name matching, and HD/VSA work that encodes
  strings by n-grams, are looked up after the result, and each is recorded as supporting, conflicting or unrelated.

## Compute

- Part a query side and part b: CPU, local (4 or fewer worker processes).
- Part a write side: about 2,200 dev and 27,000 test prompts of about 860 characters, at most 400 new tokens, on
  Qwen3-8B. PACE's estimated start on 2026-11-17 was 2026-11-19 (L40S) to 2026-11-23 (H200), so this runs on idle
  A800 GPUs (never GPU 3, never another user's process), at an estimated 1-2 GPU-hours.
- Cap: 40 GPU-hours. Ledger in the verdict (wall-clock x GPUs, every B26 GPU process including failed ones). No paid API.

## Files

- Code: b26_fuzzy.py, b26_read.py, b26b_pcm_table.py, b26_verdict.py (written before any test output exists).
- Data: b26_data/ (prompts, reader outputs), b26_out/ (arm outputs).
- Results: results/b26_dev.json, results/b26_dev_choice.json, results/b26_pcm_table.json, results/b26_verdict.txt.

## Dev record (2026-11-17 US Eastern, before any part-a test run; no design change)

- **Query-side dev grid** (b26_fuzzy.py dev; results/b26_dev.json; CPU, 2 workers, about 45 min). Mean accuracy over
  the five variant conditions, then over the two dev parts (real: 320 seed-7 dev samples at 4k and 16k; stress: 200
  synthetic stores):
  - V: {3} ng 98.40, {3} rand 98.40, {2,3} ng 98.43, {2,3} rand 98.43, {1,2,3} 98.42 (both), {2,3,4} 98.33 (both).
  - L: Levenshtein 90.33, normalized Levenshtein 95.87, OSA 90.33. W 97.95. N with {3} 98.40.
  - Where the settings differ: almost only in A (stress A: V 88.5-89.5, N 88.5-90.0, W 85.5, L 38.0 / 67.0 / 38.0;
    real A: V and N 98.3-98.9, W 98.3, L 68.4 / 94.8 / 68.4). Typos and P: every fuzzy arm 98.6-100 on both parts.
  - V's two binding codes gave the same score in every cell.
- **Choice by the registered rule** (best V 98.43; every V setting within 0.1 of it counts as tied, and the first
  listed wins): **V {3}, ng; L normalized Levenshtein; N {3}** (results/b26_dev_choice.json).
- **Write-side dev check** (A800 GPU 7, 254 s): in C0, 559 of 560 re-read chunks reproduce B21's banked output
  (bar 95%); the one difference is the reader writing "makehift-junior" for "makeshift-junior". 319 of 320 dev samples
  give the banked grounded pairs. The reader stored the varied key for 434 / 440 (T2), 426 / 440 (P) and 422 / 440 (A)
  queried keys. Nothing is tuned on this.
- **Corrections to the compute section:** the test write side has 8,400 chunks per condition and 32,839 unique prompts
  (not about 27,000). A first dev re-read was started on GPU 0, which had been idle at the check but was taken by
  another process at the moment of launch; it was killed after about 2.5 minutes, its partial output deleted, and the
  run restarted on idle GPU 7. The test re-read started on GPU 7 after the dev re-read (2026-11-17 10:36 UTC); no
  test write-side arm is run before this record is committed.
- **Part b** has no dev stage; it ran after the pre-registration commit (8005c14) and its output is not read before the
  verdict script scores it.
