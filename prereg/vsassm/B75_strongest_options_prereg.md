# B75 pre-registration: the strongest options on both sides of the equal-cell device comparison (a decorrelating write and full-store joint decoding for the stores; a near-optimal insertable table)

Date: 2026-11-23, about 22:50 machine clock (Asia/Shanghai; US Eastern 09:50). Written and committed with the code
(b75_strong.py), then OpenTimestamps-stamped and the proof committed, BEFORE any B75 development or test run. No
development or test store has been read by B75 code (the self-test used synthetic stores only; record at the end).
Development and test runs start only after `ots upgrade` reports a Bitcoin block attestation for this file (or after 2
hours without one, said in a compute note). **An external registration (OSF) needs the human author and is not made
here.** The paper is not edited by B75.

## Why (referee, round 24)

"The equal-cell comparison omits the strongest options on both sides." (1) VSA side: the tables get BCH
errors-and-erasures and LDPC belief-propagation decoders, the dense store only a matched-filter read; the sparse-block
store's explaining away (B61's H-SIC) covered only the queried keys; the decorrelating (least-squares) write that won at
equal bits (B72) was never applied to the device store. (2) Table side: the cuckoo design (B63) spends 11-15 bits per
pair beyond a static directory, so a near-optimal dynamic table could need fewer cells than the sparse-block store,
which the abstract says "matched a table accepting new keys".

## Fixed (B38 / B59 / B61 / B63 / B73's setting; unchanged)

- Stores: B38's 50 RULER test stores (b34_out/test_stores.json.gz, MD5 6c83632c...; 4,776 probed keys); development:
  B34's 12 synthetic dev stores (b34_out/dev_stores.json.gz, c3ee82da...; 1,254 probed keys). Secondary (D): B73's 50
  fresh stores (b73_out/cells/test_stores.json.gz; 4,779 probed keys). A key is exact when all its values come back in
  order with nothing extra.
- Cells: every stored bit is one binary cell (B38's differential PCM pair read as a sign); independent cell errors at
  the device model's 30-day bit-error rate x 1, 2, 4, 8, 16, 32 (B38's IID: k1 0, k2 1.3e-4, k4 0.0208, k8 0.1226, k16
  0.2760, k32 0.3744; B38's Channel, one uniform per cell seeded by store, arm tag and budget). e05 and damage are not
  run.
- Budget B = cells per stored pair; at most floor(B n) cells on a store of n pairs; every bit kept between write and
  read is on them. Grid 8 ... 4,096 in quarter octaves (one step = x 2^(1/4) = 19%). N* = the smallest grid budget with
  >= 99.5% of probed keys exact there and at every larger evaluated budget, pooled; a budget counts only where every
  store was evaluated (B38 / B58; b69_vocab.nstar_counts).
- Free, as in the paper: hashing the query, seeded codes, binding, decoders, directory logic, clean-up search, the value
  vocabulary (uncharged accounting, the paper's headline).
- Comparators, already run, not re-run (b63_verdict.all_counts on the RULER stores; b73_cells.counts on the fresh
  stores): U (B38 V-last + B59 T1: 76 / 76 / 76 / 128 / 431 / 1,722 at k1 / k2 / k4 / k8 / k16 / k32), H-SBC (B61: 32
  / 32 / 45 / 91 / 304 / 1,448), the static tables T-rep / E-BCH / E-LDPC / E-CAM (best 19 / 19 / 27 / 54 / 215 / 724),
  the cuckoo tables O-BCH / O-LDPC (B63; best 32 / 32 / 38 / 91 / 304 / 1,218). Fresh stores (B73, k1 / k8 / k16): U 76
  / 128 / 431, H-SBC 32 / 91 / 362, static 19 / 54 / 215, cuckoo 32 / 91 / 304.

## Memories (b75_strong.py; module docstring = the exact definitions)

**(A) U-LS: the dense store with a decorrelating write.** U's shards, dimension d, key / position / value / LAST codes,
binary cells and matched-filter V-last read are unchanged; only the stored vector changes. Per shard with stored bound
codes Z (m x d, rows z_p = key * position * value (* LAST)): w = Z^T (Z Z^T + alpha d I)^-1 y with y = 1 (ridge
least squares of the stored pairs' matched-filter scores, the projection / pseudo-inverse rule at alpha = 0), then
sign(w) on the cells. R refit rounds: s = Z sign(w) (the sign-domain scores of the stored pairs), y <- y + (mean(s) -
s) mean(y) / mean(s), re-solve (equalises the stored pairs' scores after quantisation). Designs (alpha, R): alpha in {inf
(the Hebbian limit), 1, 0.1, 0}, R in {0, 3}; (inf, 0) is U itself (built by b38_eq.vsa_build: its rows equal U's) and
comes first in the tie order, so a least-squares design is chosen only if it is strictly better on development. Channel
tag = U's ("V-last|null": common random numbers with U's rows). Quantisation is the sign (one cell per dimension, as
U); multi-level formats are not run. B72's full-candidate form (targets on every candidate's score) is not run: with
2 N_values ~ 12,000 random candidate codes, sum_c c c^T ~ 2 N_values I and its solution reduces to the Hebbian sum
(stated, not tested).

**(B) Full-store joint decoding.** Both decoders are given the stored slot list (every stored key and its number of
values) and the read cells; they are never given a stored value (check below).
- **U-J (dense store).** U's cells (same draw as U's rows in the uncharged accounting). Pass 1 decodes every stored
  (key, j) slot by U's matched filter, its LAST variant known from the slot list. Round t: W^ = the shard's sum of the
  decoded pair codes; per shard an error scale sigma is fitted so that the model W = W^ + N(0, sigma^2) reproduces the
  observed disagreement of sign(W^) with the read cells; each slot's candidates are scored by the log-likelihood of
  the read cells given the other slots' estimates R = W^ - (own estimate): P(b = +1 | x) = (1 - eps) Phi((R + x) / s) +
  eps Phi(-(R + x) / s), s = sqrt(sigma^2 + 1/4), eps = the condition's cell-error rate (a weighted matched filter
  with weights (log P(b | R + 1) - log P(b | R - 1)) / 2); every slot re-decoded. The probed keys are read after round T
  by U's V-last rule on these scores. Design T in {0, 1, 2, 3} (nested, one evaluation; T = 0 is U's own read, first in
  the tie order).
- **H-SBC-J (sparse-block store).** B61's H-SBC cells (same draw as B61 at the same lam, uncharged). Pass 1 decodes every
  stored slot (overlap score, LAST variant known); every decoded item's positions are explained (B61's H-SIC weights:
  log((1 - e) / q) for an unexplained active cell, -log((1 - q) / e) for an unexplained inactive one, explained cells
  carry no evidence, the slot's own item excluded, q per shard, e = max(eps, 1e-3)). T = 1: the probed keys read by
  B61's kernel with every other stored slot explained (H-SIC over the whole store instead of the probe batch); T = 2:
  every slot re-decoded once with explaining away before the probe read. Designs (lam, T): lam in {0.5, 0.7, 1.0, 1.4}
  (1.4 added: explaining away tolerates a denser load), T in {0, 1, 2} (T = 0 is H-SBC's own read, first in the tie
  order).
- **Charge rule (registered now).** Two accountings for (B), reported side by side: **free** (the slot list given at no
  cost) and **charged**: the slot list as h-bit key fingerprints from which the codes are seeded, h = ceil(log2 n_keys)
  + 10 (a probed key collides with another stored key with probability about 2^-10), at its information-theoretic
  minimum ceil(log2 C(2^h, n_keys)) bits for the set plus ceil(n_keys H(counts)) bits for the counts (zero-order
  empirical entropy); placed on cells at the binary symmetric channel's capacity bound ceil(L / (1 - h2(eps))) and then
  read without error (B69's 'cap': no real code reaches it, so this is the charge most favourable to the store); those
  cells are deducted from the store's budget (the store is built with floor(B n) - key cells); a probed key whose
  fingerprint equals another stored key's is counted as failed (the hash-seeded store would give both the same codes).
  The simulation seeds codes from the strings (statistically the same as from distinct fingerprints) and gives the
  decoder the true list. About 12 bits per key at k1; at k8 / k16 the capacity bound multiplies it by 2.2 / 6.7. For
  reference (closed form, not run): with the key list free a table needs no directory either (rows only: the value
  index and LAST flag, 14 bits per pair on the 6,000-pair stores).

**(C) O2: an insertable table with fewer bits per pair than the cuckoo table.** B63's bucketized cuckoo table (4 slots,
two candidate buckets, partial-key cuckoo hashing with B63's hashes, load <= 0.95, 500 kicks, every slot on the cells)
with SHORT g-bit fingerprints made unambiguous by a **conflict stash**: a new key whose fingerprint already occurs in
one of its two buckets goes to a stash entry [(ceil(log2 n) + 12)-bit key hash | last flag | value index]. With
partial-key cuckoo hashing two entries with equal fingerprints whose bucket pairs meet have the same pair, so this
check (made by the writer from the stored fingerprints only) keeps every (pair, fingerprint) unique, and cuckoo moves
never create a conflict: a lookup is ambiguous only through bit errors. The stash is provisioned for mu + 3 sqrt(mu) + 1
entries, mu = 4 x 0.95 x n / (2^g - 1) (a design constant from n, as the bucket count; overflow = the key is lost).
**ss = 1: semi-sorted buckets** (Fan et al. 2014): the four fingerprints' top 4 bits stored as one 12-bit index of the
sorted 4-tuple (3,876 values), saving one bit per slot. Read: the stash first (one exact hash match -> its row; two or
more -> fail), else B63's table rule (one exact fingerprint match -> read; two or more -> fail; none -> the unique nearest
slot in Hamming distance; a tie fails). Every slot, index, stash entry and empty slot is on the cells, protected by B38
/ B63's machinery unchanged: **O2-BCH** (B38's closed-form BCH per store, budget and condition; needs = the
fingerprint region of both buckets and the own payload, or the own stash entry) and **O2-LDPC** (IRA LDPC, rate R).
Insertion is simulated logically, one pair at a time in store order, each insertion seeing only the stored bits and the
new pair (the self-test checks that a half build continued online equals the full build). Designs: g in {6, 8, 10}, ss
in {0, 1}; O2-LDPC also R in a per-condition set (k1, k2: 3/4, 5/6, 9/10; k4: 1/2, 2/3, 3/4, 9/10; k8, k16: 1/4, 1/3,
1/2; k32: 1/4, 1/3 - the rates B63's development chose near its N*). Logical bits per pair on a synthetic 6,000-pair
store (self-test): 22.7-25.5 (g = 6-10), against the cuckoo table's 27.4 (f = 12) and the static MPHF table's 17.0.
**O2-best** = per k the smaller N* of O2-BCH and O2-LDPC (test-selected, favours the table; as B63's OT-best).
Literature: stash (Kirsch, Mitzenmacher and Wieder 2009), semi-sorting (Fan et al. 2014); the dynamic-retrieval lower
bound n v + Omega(n log log(u / n)) (Mortensen, Pagh and Patrascu 2005) - O2 pays g + empty slots + stash, about 6-9
bits per pair over the static table's rows, the cuckoo table 10-11.

## Development (dev stores only)

One pass per memory over a window of grid budgets around the paper's dev N* (every design at every budget in the
window): U-LS: U's dev N* -6 ... +3 steps (k1-k32); U-J: U's -10 ... +1 (k1, k8, k16; both accountings); H-SBC-J:
H-SBC's -8 ... +1 (k1, k8, k16; both accountings); O2-BCH / O2-LDPC: O-BCH's / O-LDPC's (B63) dev N* -5 ... +3 (k1-k32).
**Choice** per (memory, accounting, condition, budget) (B38 / B58 / B63's rule; b75_strong.choose): the design with the
highest pooled dev share (a design counts at a budget only if feasible on all 12 dev stores); ties: the higher share
at the next harder run condition at the same budget, then design-list order. Dev N* per (memory, accounting,
condition). The choices are committed before the test.

## Test (run once) and (D)

Per (memory, accounting, condition): every grid budget within 4 steps of the dev N* (the window's top 5 if dev never
reached the bar), each with the dev design there (the nearest dev budget's design, ties the larger). Then B38's
extension rule, one round (4 steps up if the top evaluated budget fails; 4 down if N* is the lowest evaluated
budget); an N* still not bracketed is censored ('> top' counts as not reached; '<= bottom' is reported as the lowest
evaluated budget, flagged). Conditions: (A) and (C) k1, k2, k4, k8, k16, k32; (B) k1, k8, k16. Every test cell runs
once. **(D) fresh stores (secondary, if time allows):** the same frozen dev choices and test plan restricted to k1, k8,
k16, run once on B73's 50 fresh stores, compared with B73's frozen rows; order (C), (A), (B).

## Measures, labels and readings (fixed now)

Steps = grid index; a memory that never reaches the bar counts as 37. Delta(X, Y) = steps(X) - steps(Y), so Delta < 0
means X needs fewer cells; per condition "X fewer" if Delta <= -2, "Y fewer" if >= +2, "tie" if |Delta| <= 1; label over
the run conditions by B58's h_label (X WIN / Y WIN / MIXED / TIE). Ratio = 2^(-Delta / 4).

- **(A)** N*(U-LS) at k1-k32; Delta(U-LS, U) [does the decorrelating write lower the dense store's requirement?],
  Delta(U-LS, H-SBC), Delta(U-LS, static), Delta(U-LS, cuckoo).
- **(B)** per accounting: Delta(U-J, U), Delta(H-SBC-J, H-SBC); Delta(store-best-J, static), Delta(store-best-J,
  cuckoo), Delta(store-best-J, O2-best), store-best-J = per k the smaller of U-J's and H-SBC-J's N* (test-selected,
  favours the store).
- **(C)** Delta(O2-best, H-SBC) (PRIMARY: does the sparse-block store still match the best insertable table?),
  Delta(O2-best, cuckoo), Delta(O2-best, static) (the online penalty), Delta(O2-best, U); O2's logical bits per pair
  beyond the static MPHF table's (per store, median and range).
- **Store-cluster bootstrap** (secondary; 2,000 resamples of the 50 stores within the three RULER families, seed
  'b75-boot'): share of resamples in which each primary label holds; 95% ranges of the primary Deltas.

**Readings for the paper (fixed now):**
- (A) "fewer" (Delta(U-LS, U) <= -2) at some k: the paper must say that a least-squares write lowers the dense store's
  requirement there (by the factor) and must restate "two to four times fewer than the dense store" against U-LS
  wherever U-LS is the dense store's best; tie at every k: the paper may say that the decorrelating write that helped at
  equal bits does not lower the device store's requirement (within one grid step). If U-LS ties or beats H-SBC at some
  k, H-SBC is no longer "the best vector store we built" there.
- (B) free: if store-best-J needs fewer cells than the static tables at some k, the paper may say so only together with
  "given the list of stored keys at no cost, which the tables are not given (with it free a table needs no directory)";
  charged: if store-best-J (charged) ties or beats the static / insertable tables at some k, the comparison sentences
  must name joint decoding and the k; if charged store-best-J does not need fewer cells than H-SBC / U, the paper may
  say that full-store joint decoding does not lower the requirement once the key list it needs is charged.
- (C) Delta(O2-best, H-SBC) in [-1, +1] at every run k: "matched a table accepting new keys" stands as "within one grid
  step", now against a near-optimal insertable table (name it); Delta <= -2 at some k: the sentence must change at those
  k ("an insertable table with short fingerprints and a conflict stash needed N fewer cells"); Delta >= +2: the
  sparse-block store needs fewer cells than this table there.
- Static tables: reported for every comparison; "error-corrected tables with a static directory needed fewer cells
  than any store" stands only if the static tables need fewer cells than U-LS, and than store-best-J (charged), at every
  run k.

## Predictions (from the closed-form reasoning and the synthetic self-test; stated so that they can be wrong)

- PA1. U-LS needs fewer cells than U by 1-3 steps at k1, k2, k4 and k8 (U-LS 54-64 / 54-64 / 54-64 / 91-108), by 0-1
  at k16 and k32; label against U at k1-k8: at least two "fewer".
- PA2. U-LS needs more cells than H-SBC at k1, k2, k4 (Delta >= +2) and is within +-1 step of H-SBC at k8 and k32;
  the static tables need fewer cells than U-LS at every k (Delta <= -2).
- PB1 (free). U-J needs fewer cells than U by >= 3 steps at k1 and >= 1 at k8; H-SBC-J fewer than H-SBC by >= 2 steps at
  k1 (H-SBC-J free at k1 <= 23).
- PB2 (charged). Charged, neither U-J nor H-SBC-J needs fewer cells than H-SBC (free, B61) by 2 or more steps at any of
  k1, k8, k16; the static tables need fewer cells than store-best-J charged at every k (Delta <= -2).
- PC1. O2-best needs 1 or 2 steps fewer than the cuckoo table at every k (k1: 23 or 27 against 32).
- PC2. Delta(O2-best, H-SBC) is -1 or -2 at k1, -1 or 0 at k8 and k16; label TIE (|Delta| <= 1 at every k) or
  TABLE-side "fewer" at k1 only.
- PC3. O2-best still needs more cells than the static tables (online penalty +1 to +3 steps at every k).
- PC4. No lost insertion or stash overflow on a store of >= 1,000 pairs.
- PD. On the fresh stores the (C) and (A) Deltas against the same comparators are within one step of the RULER ones.

## Checks (b75_check.py; written and committed before the test output is read)

- **Plain-loop recount**: every pooled share and N* recomputed with plain loops from the raw test rows (independent of
  b69_vocab / b75_strong code), each row's exact count from its failed-key list; 0 differences required.
- **Cells audit**: every feasible test row's cells (charged: store cells + key-list cells) <= floor(B n), B on the grid;
  O2: logical bits = m x bucket width + stash entries x entry width; 0 over budget.
- **Determinism re-run**: a seeded sample of >= 5% of the test jobs (>= 20, every memory) re-run from scratch; failed-key
  lists identical.
- **No memory reads more than it should**: (i) U-LS with (inf, 0) and U-J / H-SBC-J with T = 0 reproduce U's / H-SBC's
  registered failed-key lists on the same budgets (they use the same cells and draw); (ii) the joint decoders are re-run
  on a sample with the stored values scrambled after the write (the decode functions' inputs: read cells, slot list
  (keys, positions, LAST flags), probed keys; no stored value): identical decoded lists required; (iii) a source audit
  that the decode functions (uj_decode, sbcj_decode, _all_slots, CuckooStash.read) do not reference stored values (pv,
  want) - the write (build) does, by definition.
- **Frozen-design audit**: every counted test row used the dev design at its (memory, accounting, condition, budget).
- **Sample reading by the AI-assisted process**: for each new memory one step below its k8 N* (k1 for (B)), which probed
  keys fail and why (stash / fingerprint / undecoded codeword for O2; wrong value / LAST for the stores).

## Compute, scope and time cut (registered now)

CPU only. Dev and test on the A800 box's CPUs (~/b75_run, code and stores MD5-identical to the local commit, nice 10,
one BLAS / numba thread per worker, at most 48 workers, never the GPUs; every ssh command in the background; no retry
loops around state-changing commands; processes stopped only by exact PID; no other user's process touched); checks
and the determinism re-run locally (at most 6 workers) or on the box (box-vs-local differences reported). Time box: 5
hours from 22:26 machine clock (until 03:26), including the block wait. Cut rule, applied by a dated compute note
committed before any test output is read: if the work cannot finish in the box, cut in this order: (D) fresh stores
((B) first, then (A), then (C)); (B) H-SBC-J at k16, then U-J at k16, then (B) entirely; k32 for (A) then (C); k2 and k4
for (A) then (C). (C) and (A) at k1, k8, k16 on the RULER stores are not cut.

Not run: multi-level cells, damage conditions, e05, U-LS's multi-level formats and B72's full-candidate objective,
resonator decoding (one unknown factor, B61), a key list protected by a real code (the capacity bound is used), other
insertable designs (quotient filters: B63's note; theoretical dynamic retrieval structures).

## Files

b75_strong.py, b75_box.sh, b75_verdict.py, b75_check.py; results/b75_dev*.json.gz, results/b75_choice.json,
results/b75_plan*.json, results/b75_test*.json.gz, results/b75_verdict.{txt,json}, results/b75_check.txt,
results/b75_sample_reading.txt; prereg/B75_compute_note*.md, prereg/B75_note_after_test.md.

## Self-test record (synthetic stores only: b73_cells.make_store with seed tag 'b75-selftest', which no test or dev store uses; b75_strong.py selftest)

`python3 b75_strong.py selftest` on this machine (one thread), 22:41-22:48 machine clock, **PASS** (417 s); two earlier
`--quick` runs (code1k5 store only) during coding, same outputs. Synthetic stores: code1k5 (1,491 pairs, 150 probed),
mv (23 / 18), tiny (9 / 9), wp6k (5,993 / 150). Nothing was chosen from these runs except the code fixes listed last.

- (C) O2, every (g, ss) on every store: every placed pair in one of its two buckets under its own fingerprint, no
  (bucket pair, fingerprint) conflict, placed + stashed + lost = n, half build + online continuation = full build
  (24 / 24 builds); 0 lost; stash use <= capacity (largest 358 / 420 at g = 6 on 5,993 pairs; 75 / 119 at g = 8;
  29 / 38 at g = 10). Logical bits per pair (g 6-10, ss 1 / 0) against the cuckoo table (f = 12) and the static MPHF
  table: wp6k 22.74-25.52 vs 27.38 / 17.03; code1k5 19.61-22.46 vs 24.25 / 14.02; mv 19.39-21.48 vs 21.91 / 7.96;
  tiny 20.89-23.11 vs 21.33 / 5.56. Noiseless: every probed key exact in every build. Noisy (code1k5): O2-BCH g8 ss1
  B 45 k4 150 / 150; O2-LDPC g8 ss1 R 1/2 B 76 k8 150 / 150; O2-LDPC g6 ss0 R 1/4 B 256 k16 150 / 150; O2-BCH g8 ss0
  B 27 k1 150 / 150; every evaluation within its budget.
- (A) U-LS (code1k5, mv): (inf, 0) reproduces U's failed-key list exactly (6 / 6); code1k5 B 38 k1: U 138 / 150, the
  seven least-squares designs 140-144; B 54 k1: U 145, LS 148-150; B 91 k8: U 147, LS 148-150; mv 18 / 18 for all.
- (B) U-J: T = 0 reproduces U's failed list (3 / 3 free cases). code1k5: B 38 k1 free 138 -> 150 (T 1-3), charged
  (17,737 key-list bits = 11.9 per key, counts 0 bits) 103 -> 132 / 149 / 149 (T 1 / 2 / 3); B 128 k8 charged (38,300
  cells at the capacity bound) 148 -> 149. wp6k (71,273 key-list bits): B 38 k1 free 125 -> 150 (T 1), charged 72 ->
  127 (T 3); B 54 k1 free 144 -> 150, charged 136 -> 150; B 128 k8 free 150, charged 148 -> 150. H-SBC-J (T = 0
  reproduces B61's H-SBC failed list in every feasible free case): code1k5 lam 0.5 B 16 k1 44 -> 120 / 148, B 23 k1 142
  -> 150; wp6k lam 0.5 B 16 k1 0 -> 9 / 10, B 23 k1 110 -> 150, B 64 k8 139 -> 150; lam 1.0 wp6k B 23 k1 104 -> 149 /
  150, B 64 k8 115 -> 148 / 150; charged at B 23 k1 0-9 / 150 (the list takes 11.9 of 23 cells per pair), B 64 k8
  charged 85 -> 129 (lam 0.5), 57 -> 84 (lam 1.0).
- Leakage: U-J's decode with the stored values scrambled after the write is identical (code1k5, B 45 k1, T 1, 2).
- Timing (this machine, one thread): U-J about 26 s per evaluation (all T, one accounting) on 5,993 pairs; H-SBC-J 6-59
  s; U-LS 0.3-1.2 s and O2 < 1 s per evaluation on 1,491 pairs.
- Code fixed during the self-test, before this commit: the key-list charge first used B69's BCH vocabulary rule (82 and
  325 cells per key at k8 and k16 on code1k5, because it demands the whole list error-free with 99%); it was replaced
  by the capacity bound (26 and 79 cells per key), the charge most favourable to the store; the per-shard sum of the
  decoded codes computed by reduceat; a dead helper removed.
