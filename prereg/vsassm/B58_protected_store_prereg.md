# B58 pre-registration: the VSA store protected by the tables' channel codes; value-vocabulary accounting; the LDPC decoder discrepancy

Date: 2026-11-20 (US Eastern; machine clock Asia/Shanghai, 20:40). Written and committed, then OpenTimestamps-stamped,
before any B58 run. Nothing has been run for B58 except a timing of a numba port of B38's second LDPC decoder on
synthetic all-zero codewords (speed only; its outputs were not compared with anything); that port (b58_bp.py) is
committed with this document. The rest of the B58 code is written after this commit and before any development run;
it implements this document. Local CPU only (16 cores,
<= 12 workers, free memory watched); no GPU, no paid API, no remote machine.

## Why (referee)

B38 (equal cells), B52 (c) (one-day rates) and B54 (multi-level cells): at equal reliability (>= 99.5% of 4,776 RULER
facts exact) the best error-corrected table needed 2-4 times fewer cells per stored pair than the VSA store. The tables
were searched over code family, length, rate and interleaving; the store only over its read rule and load. Objections:
(a) the store's quantized bundle is itself a bit string and could carry the same channel codes; is the table's
advantage cross-talk, or merely the store lacking channel coding? (b) the store's clean-up codebook is "regenerated
from the value strings and not counted"; is the tables' index-to-value dictionary treated identically? (c) B38's two
LDPC decoders agreed on only 79% of codewords, the second decoding many more under stuck cells without an interleaver;
which is correct, and does Fig. 5b (failed lines, stuck clusters; with and without the interleaver) still hold?

## Fixed (B38's protocol unless stated)

Stores, probed keys and metric: B38's 50 RULER test stores (b34_out/test_stores.json.gz), 4,776 probed keys; a key is
exact when all its values come back in order with nothing extra. Dev: B34's 12 synthetic dev stores
(b34_out/dev_stores.json.gz, 1,254 probed keys). Cells: every stored bit is one cell (B38's differential PCM pair read
as a sign). Budget B = cells per stored (key, j) pair, at most floor(B n) cells on a store of n pairs; every bit kept
between write and read is on them. Grid: B = 2^(i/4), 8 ... 4,096. N*(variant, condition) = the smallest grid budget
with >= 99.5% of probed keys exact there and at every larger evaluated budget. Conditions: B38's i.i.d. cell errors at
the device model's 30-day bit-error rates (k1 0, k2 1.3e-4, k4 0.0208, e05 0.05, k8 0.1226, k16 0.2760, k32 0.3744)
and B38's localized damage (each damaged cell reads a fixed random sign; f = 2 eps of the matched i.i.d. condition;
failed lines or 16 x 16 stuck clusters; layout il = seeded random interleaver over the array, nat = natural order with
codewords and copies contiguous). Free for every arm: hashing, seeded code generation from strings, binding, majority,
decoders, similarity search. B38's table results (T-rep, E-BCH, E-LDPC, E-CAM, T-vhdd) and its store results (V-last
etc.) are used as registered, except E-LDPC where part (c) re-scores it.

## Part (a), PRIMARY: the protected store

**Store.** B38's V-last store (each key's last value bound with a LAST code; clean-up over value and value x LAST; stop
when the LAST variant wins; no threshold; B38's best store read at every k but k8, where it tied V-Snt) built at an
**information budget b** (info bits per stored pair): exactly b38_eq.vsa_build(store, b, "last"), i.e. S = ceil(b n /
8,192) shards of d = floor(b n / S) <= 8,192 sign bits; codebook regenerated from the value strings (0 cells, as in
B38). Its S d sign bits, shard-major, are the logical bit string that the channel code protects. b in INFO = {76.1,
90.5, 107.6, 128.0} (grid steps 13-16): 76 is the store's noise-free requirement on dev and test (B38: dev share at 64
is 99.36%, below the bar even with perfect bits), so smaller b cannot reach the bar; larger b buys tolerance of residual
errors after decoding.

**Codes (the tables' families, lengths, rates, repetition and layouts).**
- **P-LDPC:** the store's bits in B38's systematic IRA LDPC codewords (b38_eq.ldpc_layout / ldpc_encode_stream: <= 1,024
  information bits per codeword, rate R in {1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 9/10}, the registered column weights),
  each coded bit repeated r or r + 1 times to fill floor(B n) cells; copies soft-combined into channel log-likelihood
  ratios at the condition's (matched) rate, as E-LDPC; sum-product belief propagation, flooding, 60 iterations, with
  the decoder that part (c) establishes as correct (below). Design (b, R): 4 x 7 = 28 per budget, chosen on dev.
- **P-BCH:** the store's bits in B38's shortened narrow-sense binary BCH codewords (n = 63 ... 1,023, every t with k >=
  2: the same 223 designs), copies r or r + 1 filling the budget, majority per coded bit with ties as erasures, B38's
  errors-and-erasures bounded-distance decoder (b38_eq.bch_decode_stream, galois-matched codec). (n, t) by **the
  tables' closed-form rule** (b38_eq.bch_choose): minimize the expected number of failed stored keys at the
  condition's (matched) rate, a key failing when any codeword holding a bit it needs fails; ties at the next harder
  rate, then the shorter code. A store key needs every bit of its shard (computed by shard with key counts as
  weights; identical to bch_choose with per-key needs, checked in the self-test). b chosen on dev.
- Decoded information bits -> +-1 -> V-last read, hard decisions (as the tables read hard decoded bits).
- Layout: cells in natural order (codewords contiguous, copies adjacent; B38's table layout) or B38's interleaver
  (il), exactly as the tables.
- **P-best** = per condition the smaller N* of P-LDPC and P-BCH (the protected store). **P-any** = the smaller of
  P-best and B38's uncoded V-last.

**Development (registered selection rule = the tables').** i.i.d. conditions only (damage uses the matched i.i.d.
design, B38's rule). Pass 1: every design at the octave budgets 128, 256, 512, 1,024, 2,048, 4,096 (and 90.5, the
smallest budget where a code fits at b = 76.1); pass 2: every design at the quarter steps within 3 grid steps of the
pass-1 N* of each (variant, condition) (the top 4 budgets if not reached). Per (variant, condition, budget) the design
with the highest dev share; ties: the higher share at the next harder condition at the same budget; then design-list
order (b ascending, then R in the order above). Infeasible designs (fewer than one copy) are skipped. **Censoring (a
compute rule, exact):** an evaluation on one dev store stops once that store has 7 failed probed keys (7 > 0.5% of
1,254, so the design cannot reach the bar at that budget); censored designs rank below every completed design; if every
design at a budget is censored, that budget is "not reached" and its design is the one chosen at the nearest evaluated
budget with a completed choice (first feasible design in list order if none). Shard by shard decoding and reading is
used for this stop; reads of shards whose decoded bits equal the written bits are the store's noise-free reads (exact:
every similarity is an integer count over +-1 components, so the result does not depend on summation order).

**Test (run once).** Conditions: the 7 i.i.d. conditions (primary) and 16 damage conditions (lines and clusters x il
and nat x k4, e05, k8, k16; secondary). Per (variant, condition): dev N* of the matched i.i.d. condition +- 4 grid steps
(B38's window), each budget with its dev design (nearest dev-evaluated budget's design where dev did not evaluate it);
the top 5 grid budgets if dev never reached 99.5% by 4,096; then B38's extension rule (4 more steps in the needed
direction, repeated until the test N* is bracketed or the grid ends; gap fill as B38). A test evaluation on one store
stops once the store has 40 failed probed keys (> 0.5% of any store-cluster bootstrap total, so exact for N* and for
the bootstrap). Each test cell runs once.

**Primary bar H1-P (i.i.d. k1 ... k32).** Delta(k) = grid steps of the best table's N* minus grid steps of P-best's
N* (best table = the smallest N* among T-rep, E-BCH, E-LDPC, E-CAM; E-LDPC from part (c)'s correct decoder; positive:
the protected store needs fewer cells). Per k "store fewer cells" if Delta >= +2, "table fewer cells" if <= -2, "tie"
if |Delta| <= 1; VSA WIN / TABLE WIN / MIXED / TIE as B38's H1. Store-cluster bootstrap (2,000 resamples, B38's
method) of Delta and of P-best's N*: 95% percentile ranges. e05 reported, not in the label.

**Secondary.** H1-Pany (P-any in place of P-best). G (does coding help the store?): steps(V-last N*) -
steps(P-best N*) per i.i.d. k: "coding helps" >= +2, "coding hurts" <= -2, else tie. H3-P (interleaved damage) and
H4-P (natural layout): P-best vs the best table (re-scored E-LDPC) with H1's rule over the 8 conditions per layout;
position independence of P-best (|N*(damage) - N*(matched i.i.d.)| <= 1 step). Attribution: the store's cross-talk
requirement is its N* with no cell errors (k1; B38: 76 against the best table's 19).

## Part (b): value-vocabulary accounting

Audit (code reading, stated with file and line): whether every B38 arm receives the value vocabulary (index -> value
string) identically. The registered reading of B38's code: the store regenerates its codebook from the vocabulary's
strings by seeded hashing (b25_pcm.codes; free stateless logic), every table stores vocabulary indices and maps them
to strings through the same vocabulary (b34_inmem.Store.vi / cd["vocab"]), T-vhdd regenerates its value codes from the
same strings; nothing else is free. If the audit confirms this, B38's V-last-vs-tables accounting is "both free"
(symmetric) and V-cb is the only asymmetric arm (it pays its codebook, which no table has; the tables' dictionary stays
free). **Symmetric "both charged" variant (computed, not simulated):** the vocabulary's strings (UTF-8 bytes x 8 bits)
placed on the same cells for every memory, at the condition's binary-symmetric-channel capacity, 1 / (1 - H2(eps))
cells per bit (a bound neutral to both; at k1 one cell per bit); the store regenerates its codebook from the charged
strings (no further cost). Per memory N*' = N* + v x c(eps), with v = vocabulary bits per pair: pooled over the 50
stores v = the maximum over stores (each store's budget must hold its own vocabulary), and per store family. Delta'
in grid steps = 4 log2(N*'(table) / N*'(store)), labels as H1 (rounded steps). Reported for V-last, P-any and V-cb.

## Part (c): the LDPC decoder discrepancy

Decoders: **PKG** = the ldpc package's BpDecoder (product_sum, flooding, 60 iterations, received-vector input with
per-bit channel probabilities; B38's registered decoder; locally ldpc 2.4.1, B38 ran 2.3.9 on the box and 2.4.1
locally with identical self-test counts); **OWN** = b38_check._sp_decode (B38's second decoder, numpy) and its numba
port (b58_bp.py, used for speed; must agree with _sp_decode, check below). Same channel inputs as B38: copy votes x
log((1 - eps) / eps), probabilities clipped to [1e-15, 0.5].

- **(c1) Cause.** On B38's 10 decoder-check cells (b38_check.decoders' picks and seeds; 1,007 codewords re-simulated
  from B38's test stores, which B38 already read), per codeword: convergence, info-bit agreement, correctness of each
  decoder, PKG's posterior log-probability ratios (finite or not), the number of input bits at the LLR clip and of
  confidently wrong inputs (wrong sign, |LLR| >= 20). Mechanism test: both decoders again with the channel
  probabilities floored at 1e-6 (|LLR| <= 13.8). Candidate causes tested: (i) non-finite messages in PKG's
  product-sum check update when tanh saturates (confident, conflicting inputs); (ii) tie handling of hard decisions;
  (iii) the convergence / early-stop test; (iv) an input convention mismatch (sign, probabilities vs LLRs). The cause
  is the candidate whose removal (floor, or a matched convention) restores agreement on >= 99% of the codewords.
- **(c2) Reference (third angle): exhaustive maximum-likelihood decoding.** Short IRA codes from B38's construction with
  K = 16 information bits (b38_eq.ira_code(16, R), R in {1/4, 1/3, 1/2, 2/3}); every one of the 65,536 codewords
  scored against each frame's LLRs (block ML). Channels, 400 frames per setting, random messages: (A) i.i.d. cell
  flips at eps with rho copies soft-combined (rho in {4, 16, 30}, eps in {0.0208, 0.1226, 0.276}); (B) stuck copies, as
  the natural layout: a share f = 2 eps of the coded bits has every copy stuck at a random sign, the rest read
  correctly, LLR at eps (eps in {0.0208, 0.05, 0.1226}; rho in {4, 16, 30}). Per decoder: block success (decoded word
  = sent), agreement with ML's word, frames where ML is right and the decoder wrong. **Correct** = a decoder whose
  block success is within 3 points of ML's in every setting of both channels; if both qualify, the disagreement is not a
  decoding error and both are reported; if only one qualifies, it is used for P-LDPC and for the re-scoring; if neither,
  ML-closer per setting is reported and OWN is used (stated as such).
- **(c3) Re-score.** E-LDPC with the correct decoder (if PKG is correct, B38's numbers stand and nothing is re-run) on
  the 50 test stores at B38's designs (the dev E-LDPC rate for the matched i.i.d. condition and budget): the 16 Fig. 5b
  conditions (lines, clusters x il, nat x k4, e05, k8, k16) and the 7 i.i.d. conditions (the primary best table), at
  every budget B38's test evaluated for E-LDPC in that condition, then B38's extension rule until bracketed. Report N*,
  position independence, B38's H3 / H4 labels and Delta (best store read vs best non-HD table) per condition with the
  re-scored E-LDPC.
- **Fig. 5b sentence** ("superposition helped only where a table's codewords lie in contiguous cells, until an
  interleaver gave tables the same tolerance") **holds** if, with the re-scored E-LDPC: (1) in nat lines and clusters
  the best store read needs >= 2 grid steps fewer cells than the best non-HD table in >= 6 of 8 conditions; (2) with the
  interleaver every non-HD table's N* is within 1 step of its matched i.i.d. N* in >= 7 of 8 conditions for E-LDPC and
  the interleaved best table needs fewer cells than the store (H3 TABLE WIN). Otherwise it is restated with the numbers.

## Predictions (Q1..Q11; checked against the verdict, reporting only)

- **Q1 (H1-P)** TABLE WIN; Delta <= -4 at every i.i.d. k (k1 ... k32).
- **Q2 (G)** coding never helps the store by >= 2 steps at any i.i.d. k; coding hurts (P-best >= V-last + 2 steps) at
  k8, k16 and k32 (the store's 76 information bits per pair cost at least 76 / capacity cells once protected: 164 at
  k8, 506 at k16, 1,650 at k32, against the uncoded store's 128 / 431 / 1,722).
- **Q3** P-best N* = 90.5 at k1 and k2 (one step above V-last's 76: parity alone), within 1 step of 90.5 at k4.
- **Q4** P-LDPC <= P-BCH at k8, k16, k32; P-BCH <= P-LDPC at k1, k2, k4.
- **Q5 (H3-P)** interleaved damage: TABLE WIN; P-best within 1 step of its matched i.i.d. N* in >= 7 of 8 conditions.
- **Q6 (H4-P)** natural layout: P-best needs >= 2 steps more than its matched i.i.d. N* (or never reaches the bar) in
  >= 6 of 8 conditions: protected, the store loses the uncoded store's position independence.
- **Q7 (b)** the audit finds the accounting symmetric ("both free"); both charged, pooled (v = 288 bits per pair, the
  36-character UUID values of niah_multikey_3), every k is a TIE (|Delta'| <= 1); for the niah_multikey_2 family (v =
  56) the table still needs >= 2 steps fewer cells at k1, k2, k4 and k8.
- **Q8 (c1)** PKG's posterior LLRs are non-finite in >= 90% of the codewords where PKG fails and OWN succeeds; with
  probabilities floored at 1e-6 the two decoders agree on >= 99% of the 1,007 codewords (cause (i)).
- **Q9 (c2)** channel A: both decoders within 3 points of ML in every setting; channel B: OWN within 3 points of ML in
  every setting, PKG >= 10 points below ML in at least one setting with rho >= 16.
- **Q10 (c3)** with OWN, E-LDPC lines-nat N* <= 152 at k4 (B38 PKG: > 4,096); the best store read's lead in the natural
  layout stays >= +2 steps in >= 6 of 8 conditions and the Fig. 5b sentence holds by the rule above; i.i.d. E-LDPC N*
  unchanged (within 1 step) at every k.
- **Q11 (checks)** second scorer 0 differences; 0 evaluations over budget; numba port = _sp_decode on >= 99.9% of
  codewords; P-BCH bounded-distance shortcut correct on every word it covers (galois).

## Checks (fixed now)

- **Second scorer by plain loops** (b58_check.py scorer): for a sample of test cells (every variant; i.i.d. and damage;
  stores of every family), the run is repeated without the shard cache, the read recomputed by explicit loops over
  probed keys, positions and candidate values (one dot product per pair), the exact-read test re-implemented from the
  store's own pairs (not Store38.want); failed-key lists must be identical.
- **Bit and cell accounting** (b58_check.py bits): for every test evaluation, information bits S d <= floor(b n),
  coded bits x copies = cells <= floor(B n); counts over budget (must be 0) and the share of the budget used.
- **Decoder agreement**: numba port vs _sp_decode on (c1)'s 1,007 codewords and on 2,000 synthetic words; PKG vs OWN on
  a sample of P-LDPC test words at i.i.d. conditions; P-BCH words against galois's BCH decoder (B38's method).
- **N* recomputed** from the per-store failed-key counts by a separately written rule (0 mismatches).
- **Sample reading by the AI-assisted process** (results/b58_sample_reading.txt): failed keys of P-LDPC and P-BCH one
  grid step below N* at k8 and k16 and in a nat condition, with the cause (residual errors in the key's shard after
  decoding; similarity of the true value vs the winner).

## How the paper should read each outcome (fixed now)

- **H1-P TABLE WIN and G "coding hurts or tie" at every k:** "Protected by the same channel codes as the tables, the
  store needs no fewer cells than unprotected, and the tables still need 2-4 times fewer: the gap is the store's
  cross-talk (76 information bits per pair with perfect cells, against about 14 for the table), not a missing channel
  code." The store's 76-cell noise-free requirement is then the number to cite.
- **H1-P TIE / MIXED / VSA WIN at some k:** name the k where the protected store reaches the table, with the code; then
  the B38 gap at that k was due to the store lacking coding, and the paper must say so.
- **G "coding helps" at some k:** the paper reports the protected store as the store's best form at that k.
- **(b):** if the audit confirms symmetry, say "the value vocabulary is shared and uncounted for every memory"; the
  both-charged numbers are reported with the warning that a common cost dilutes the ratio but leaves the difference in
  cells per pair unchanged. If the audit finds an asymmetry, the symmetric numbers replace B38's.
- **(c):** name the cause; if PKG is defective, every PKG-decoded E-LDPC number in damage conditions is replaced by the
  re-scored one (and i.i.d. ones if they change); the Fig. 5b sentence is kept only if it holds by the rule above.

## Scope and cuts (registered now, before any run)

Not run: multi-level cells (B54's conditions; time), contiguous loss and known-location damage for the protected
store, other store reads (V-pub, V-Snt, V-am), soft (posterior) decoded inputs to the read, b outside INFO. If the
test has not started by 23:00 machine clock, the 16 damage conditions of part (a) are cut and recorded in a compute
note before any test output is read; the i.i.d. part (a) and part (c) are not cut.

## Files

Code: b58_bp.py (numba sum-product), b58_store.py (protected store, dev/test runner, plan, extension), b58_dec.py
(part c), b58_check.py, b58_verdict.py. Results: results/b58_dec_*.{json,txt}, results/b58_dev*.json.gz,
results/b58_choice.json, results/b58_plan*.json, results/b58_test*.json.gz, results/b58_ldpc_rescore*.json.gz,
results/b58_check.txt, results/b58_sample_reading.txt, results/b58_verdict.{txt,json}; notes:
prereg/B58_compute_note*.md, prereg/B58_note_after_test.md.
