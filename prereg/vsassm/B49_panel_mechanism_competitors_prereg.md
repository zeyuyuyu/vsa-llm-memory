# B49 pre-registration: a pre-specified encoder panel, a test of the mechanism by changing the codes' geometry, and two more equal-bit competitors

Date: 2026-11-19 (US Eastern; box clock Asia/Shanghai). Written and committed before any memory (B32's VSA store and
index, B36's key-free families, or the new competitors below) has answered a dev or test query on the codes of
Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, bge-small-en-v1.5 or all-MiniLM-L6-v2, before the new competitors have
answered any query on any codes, and before any transformed codes of the mechanism test exist.

Done before this file (no memory answered any query):
- Reading B32 / B36 / B46 (prereg/B32_*.md, B36_*.md, B46_*.md; results/b36_verdict.txt, b46_verdict.txt, b46_summary.txt).
- Copying the Llama and Mistral codes that B46 encoded on the A800 box (every registered variant was encoded and scored
  by B32's criterion in B46, before any memory comparison on them) as float16: emb_llama31i_mean_L24_f16.npy (md5
  a55892a0ae069b160cb2dfaedce914a4), emb_mistral7i_eol_L24_f16.npy (md5 52207e3edaddb67145252805c30954be). The
  sentence-encoder codes are B32's (b32_data/emb_{bge_small,minilm}.npy, encoded on the CPU in B32 with
  b32_embed.st, the models' own sentence-transformers pooling, before B32's comparisons).
- Geometry of the codes (b49_geom.py; training split for spectra, dev split for alias-label cosines; no memory):
  b49_out/geom_existing.json, b49_out/geom_candidates.json. Table below.
- The learned map of the learned-hashing competitor, chosen per encoder by B32's dev criterion (b49_lhchoose.py; no
  memory): b49_out/lh_choice.json (running when this file was written; its choice is mechanical and is recorded in the
  dev record before any memory uses it).
- Code: b49_common.py (LAM, learned map, transforms), b49_run.py / b49_dev.sh / b49_test.sh (B32 + B36 unchanged on
  the new encoders, as b46_run.py). The runner of the new competitors and of the mechanism runs is written after this
  file and must implement the specification below; any deviation is recorded as an amendment before its dev results
  are read.
- A literature check by a subagent (web only): results/b49_literature.txt (written with this file).
- Disclosure: B32's pilot 1 used bge-small codes (centred, one dev store, M = 1,024, an earlier code; prereg/B32 'Pilot
  record'): the VSA was level with the index at V = 2 and 8 bits per fact and ahead at V = 16. Every other number about
  the four new encoders' memories is unseen. For Qwen3-8B, Granite and e5 the B36 / B46 test results of B32's and
  B36's memories are public (results/b36_verdict.txt, b46_verdict.txt) and are reused unchanged, not re-run.

## Why

The paper's equal-bits claim: in R1 (V in {2, 16}, 4-16 bits per fact) the superposed VSA store beats the best
dev-chosen key-free or per-fact classical memory on frozen LLM hidden-state codes (Qwen3-8B +6.77 [+5.82, +7.74];
Granite-3.3-8B +6.06 [+5.04, +7.10]) but ties on e5-base-v2 codes (+0.60 [-0.27, +1.43]). Referee (7th read), M2a:
(1) a pre-specified panel of at least four LLM families and at least two sentence encoders, all reported; (2) a
mechanistic account of why LLM codes favour superposition and e5 does not (anisotropy, effective dimensionality),
tested by isotropising LLM codes and anisotropising e5 codes and re-running the comparison; (3) more equal-bit
competitors: an optimal least-squares / pseudo-inverse linear associative memory and learned hashing trained on the
training split; (4) the bits per fact each memory needs to reach a useful accuracy (e.g. 90% of the exact index's), or
an explicit statement that the store has no such regime; (5) per-cell reporting wherever any competitor wins (a count
sketch reached B36's lift target with fewer bits at M = 256, V = 2). B49 does all five. It goes into the paper whatever
it shows.

## 1. Panel (fixed now; every member reported)

LLM families (frozen instruction-tuned decoders; per family the variant with the highest B32 dev criterion among the
eight registered variants, two poolings x four depths; all scored in B32 / B46 before any memory comparison on them):

| tag | model | variant | criterion | source |
| --- | --- | --- | --- | --- |
| qwen8 | Qwen3-8B | mean, layer 9 of 36 | 0.7035 | B32 (B36 test reused) |
| b46 | Granite-3.3-8B-Instruct | mean, layer 30 of 40 | 0.6538 | B46 (test reused) |
| llama | Llama-3.1-8B-Instruct | mean, layer 24 of 32 | 0.6478 | B46 criterion; new here |
| mistral | Mistral-7B-Instruct-v0.3 | PromptEOL, layer 24 of 32 | 0.6427 | B46 criterion; new here |

Sentence encoders (one per family among B32's five sentence-encoder candidates, the highest B32 criterion within the
family; the models' own pooling, as B32 encoded them):

| tag | model | dims | criterion | source |
| --- | --- | --- | --- | --- |
| e5 | intfloat/e5-base-v2 ("query: ") | 768 | 0.6980 | B32 (B36 test reused) |
| bge | BAAI/bge-small-en-v1.5 | 384 | 0.6547 | B32 codes; new here |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.6296 | B32 codes; new here (all-mpnet-base-v2 0.6264 and paraphrase-multilingual-mpnet-base-v2 0.6227 are the same family, lower) |

The tag seeds every fixed random component of an encoder's memories (as 'qwen8', 'e5', 'b46' did). Codes are float16
for the LLMs (as B32 / B46) and float32 for the sentence encoders (as B32).

## 2. Protocol on the new encoders (B32 + B36, unchanged)

Exactly B46's protocol (b49_run.py = b46_run.py with the encoder as an argument; working directory
b49_out/enc_{tag}/): B32's data, splits, stores, queries; B32's VSA (72 configurations) and index (48) at 4-64 bits per
fact (b32_run dev) and 1-2 (b36_run dev32); B36's SF, CM, BF, FBF, SDM at 1-64 (b36_run dev, amendment 1); B32's and
B36's choice rules; test once; merge (b36_out/test/all_{tag}_M{M}_V{V}.jsonl.gz). B36's verdict code (b36_verdict.main)
gives each encoder's B36-protocol cells and regions (BEST36 = B36's six competitor families).

## 3. New competitors (all seven encoders)

Budgets, stores, queries, preprocessing ('center', 'whiten' = centre + ZCA, eps 0.1), accounting (ideal primary, real
secondary) and the choice rule (per (M, V, bpf, acct) and family the configuration with the highest mean dev accuracy
over the 36 dev stores; within 0.1 points tied, first in grid order) as in B36. bpf in {1, 2, 4, 8, 16, 32, 64}.

**LAM, least-squares linear associative memory** (Kohonen 1972 'optimal linear associative memory'; Kohonen &
Ruohonen 1973 pseudo-inverse; Personnaz et al. 1986). Features f(x) of the preprocessed code: 'rp' (a fixed Gaussian
projection, blocks of 8,192 columns drawn from seed ('LAMG', tag, prep, block); a p-feature map is the prefix; p <=
65,536) or 'pca' (the first p principal directions of the preprocessed training codes; p <= n). Stored: W (p x c) =
F^T (F F^T + lam I)^-1 T, F the M x p features of the stored labels, lam = alpha x mean diagonal of F F^T. Targets T:
'onehot' = centred one-hot rows (e_v - 1/V); the V solution columns sum to zero, so only values 1..V-1 are stored
(c = V - 1) and value 0's score is minus their sum; for V >= 128 also 'code32' / 'code128' (random +-1 value codes, c =
32 / 128, read by correlation with each value's code). W is quantized per column and 8,192-row block (q = 1: sign
with the block's mean |w| as scale; q = 2, 4: Max's uniform quantizer, as the VSA), one 32-bit scale per (column,
block). Read: argmax over values of f(query)^T W_hat t(v). Bits: ideal c p H_q + 32 c ceil(p / 8192) (H_q as the VSA);
real q instead of H_q. p = the largest that fits (capped as above; budget left unused when capped). Grid: prep (2) x
feat {rp, pca} x alpha {1e-4, 1e-2, 1, 100} x q {1, 2, 4} x target (onehot; + code32, code128 for V >= 128) = 48 / 144.
(alpha = 1e-4 is the pseudo-inverse in practice; alpha = 100 approaches the Hebbian outer-product rule, i.e. the
superposed store's own write rule in the LAM's feature space.)

**LH, learned hashing.** A learned linear map from the training split's alias-label pairs (2,803 pairs of 1,075 training
entities): shrinkage LDA (LDAHash, Strecha et al. 2012): codes row-normalised and centred on the training mean, projected
on the first r generalized eigenvectors of (S_t, S_w + s mean eig(S_w) I), S_t the training codes' covariance, S_w the
covariance of alias - label differences; rows unit-norm. (s, r) in {0.01, 0.1, 1, 10} x {32, 64, 128, 256} chosen per
encoder by B32's dev entity-retrieval criterion on the learned codes (highest; ties within 0.001 to the smaller r,
then the larger s; b49_lhchoose.py). The learned codes are then hashed by the existing machinery, each a family of its
own (prep = the learned map only):
- **LHI**: B32's per-fact index on the learned codes (families LSH, ITQ, PCA-sign, PQ4, PQ8 x stored share {1, 1/2,
  1/4, 1/8} = 20 configurations; ITQ on the learned projection = supervised ITQ as in CCA-ITQ, Gong et al. 2013).
- **LHC**: B36's LSH bucket sketch (CM grid, 66 / 22 configurations for one prep) with SimHash / PQ signatures of the
  learned codes (projection pool and PQ codebooks of the learned space, fitted on the training split).
- **LHS**: B36's static function (SF grid, 88 configurations for one prep) on the same signatures.
- Secondary, not a competitor: **VSA-L**, B32's VSA grid (fmap x q x share = 36) on the learned codes, to tell a better
  metric from a better memory.

**Unbudgeted references** (like EXACT; not competitors): **VSA-inf**, the store's read at infinite width, score(v) =
sum over stored facts with value v of cos(query, key) on the preprocessed codes (prep chosen on dev); **LAM-inf**, the
ridge read on the exact preprocessed codes (prep and alpha chosen on dev).

**BEST** (primary in B49) = per cell, the dev-chosen competitor family with the highest dev accuracy among IDX, SF, CM,
BF, FBF, SDM, LAM, LHI, LHC, LHS (ties: that order). **BEST36** = B36's (IDX ... SDM), for continuity with B36 / B46. **BEST-noLAM** = the same rule
without LAM (classical memories and learned hashing only).

## 4. Metric, statistics, regions (B36's, unchanged)

Accuracy on alias queries; cell difference VSA - X paired on the same queries; 95% cluster bootstrap (entity within a
store; 10,000 resamples for BEST and BEST36, 2,000 for single families); Bonferroni over the 56 cells of an encoder and
accounting; exact sign test. Regions R1 (V 2/16, 4-16 bpf; 12 cells), R0 (V 2/16, 1-2), RH (V 2/16, 32-64), R2
(V 1,024); ORACLE (per cell the competitor with the highest TEST accuracy). The verdict (b49_verdict.py) reuses
b36_verdict's functions (per_query, boot_region, sign_test, rate) on merged files that add the new families.

## 5. Bars

- **H1-B49 (primary, per encoder, ideal accounting):** over R1, VSA - BEST >= +2.0 points and the 95% interval above 0.
  Loss: <= -2.0 with the interval below 0. Otherwise tie.
- **Panel summary:** the number of encoders passing H1-B49: of 7, of the 4 LLM families, of the 3 sentence encoders.
- Secondary per encoder: the same bar against BEST36, under real accounting, against ORACLE; R0, RH, R2; VSA - LAM,
  VSA - LHI / LHC / LHS over R1; VSA-L - VSA over R1.
- **Cell map (item 5):** for every encoder, every cell where BEST, or any single competitor family's own dev-chosen
  configuration, leads the VSA by >= 2 points with its 56-cell Bonferroni interval below 0 (per family), listed by
  family; and every (M, V) where a family reaches the lift target (B36's rate statistic, lift 0.25 and 0.5) with fewer
  bits per fact than the VSA.
- **Useful accuracy (item 4):** U = 0.9 x EXACT's test accuracy in the setting (encoder, M, V); for every memory, the
  bits per fact at which the running maximum of its test accuracy curve first reaches U (b36_verdict.rate,
  log-linear interpolation; '> 64' if never within the grid). The store has a useful regime in a setting only if it
  reaches U within 64 bits per fact; if VSA-inf < U, no budget at all brings the store to U there (the infinite-width
  read is the limit of the finite store as its width grows).
- Reporting rule (B36): no advantage of either side in a cell or region whose interval includes 0.

## 6. Mechanism

### Geometry (measured before this file; b49_geom.py)

Training-split spectra of the preprocessed codes (participation ratio PR = (sum l)^2 / sum l^2, and PR / n), dev-split
alias-label cosine (mu+), spread of an alias's cosine with unrelated labels (sd-), d' = mu+ / sd-.

| tag | n | raw mean cos | centred PR (PR/n) | whitened PR (PR/n) | d' centred | d' whitened |
| --- | --- | --- | --- | --- | --- | --- |
| qwen8 | 4096 | 0.39 | 170 (0.042) | 3168 (0.773) | 7.0 | 18.5 |
| b46 (Granite) | 4096 | 0.76 | 192 (0.047) | 3058 (0.747) | 7.9 | 19.1 |
| llama | 4096 | 0.54 | 199 (0.048) | 3008 (0.734) | 8.3 | 19.4 |
| mistral | 4096 | 0.53 | 67 (0.016) | 2757 (0.673) | 4.7 | 17.4 |
| e5 | 768 | 0.73 | 135 (0.176) | 570 (0.743) | 6.8 | 10.3 |
| bge | 384 | 0.47 | 134 (0.349) | 367 (0.955) | 6.6 | 8.8 |
| minilm | 384 | 0.17 | 109 (0.284) | 332 (0.864) | 5.8 | 8.1 |

Observation: the raw common direction (raw mean cosine) does not separate the two kinds (0.39-0.76 vs 0.17-0.73), and
the centred spectra have similar absolute concentration (PR 67-199 vs 109-135); what separates them is (a) anisotropy
relative to the ambient dimension (centred PR / n 0.016-0.048 vs 0.18-0.35) and (b) the number of usable directions after
whitening (PR 2,757-3,168 vs 332-570), with which the alias-label signal stands out about twice as far from unrelated
similarities (whitened d' 17-19 vs 8-10). The two accounts below predict the same panel; the interventions separate them.

### Two accounts (both stated so that they can fail)

- **HA, anisotropy (the referee's candidate):** LLM codes favour superposition because they are anisotropic (low PR / n
  after centring); isotropic codes do not.
- **HD, usable dimensionality:** the store's read sums graded similarities over every stored key; at infinite width its
  cross-talk falls as sd- (about 1 / sqrt(whitened PR)), so many usable directions after whitening help it, while a
  per-key or per-bucket competitor at 4-16 bits per fact cannot use directions beyond its few bits. Anisotropy as such
  is irrelevant once the memories may whiten.

### Interventions (codes transformed before the unchanged pipeline; every fit on the training split)

- **T1, isotropise LLM codes:** the codes are replaced by their ZCA-whitened version (B32's 'whiten' preprocessing, eps
  0.1, rows unit-norm), so every memory sees near-isotropic codes under both preprocessings (centred PR / n about 0.75
  instead of 0.04). Primary on qwen8; replication on b46.
- **T2, anisotropise e5 codes:** e5's centred spectrum is raised to a power gamma > 1 (each principal direction scaled
  by l^((gamma - 1) / 2); b49_common.spectrum_power), gamma chosen (bisection on the training codes) so that the
  centred PR / n equals qwen8's (0.042). Memories may still whiten (ZCA with eps 0.1 of the mean eigenvalue, which
  cannot fully undo a steep spectrum). Primary on e5; replication on bge.
- **T3, cap the LLM codes' dimension:** the raw codes are multiplied by a fixed 4096 x 768 Gaussian matrix (seed
  ('B49JL', 768)), rows unit-norm: e5's ambient dimension, the spectrum's head kept. Primary on qwen8; replication on b46.

Each intervention run is R1 only (V in {2, 16}, 4-16 bits per fact, both M; 18 dev and 50 test stores): every memory
and competitor family of sections 2-3 (VSA, IDX, SF, CM, BF, FBF, SDM, LAM, LHI, LHC, LHS; their grids and choice rules;
learned map re-chosen on the transformed codes), dev choice, test once. Fixed random components are seeded by the base
encoder's tag. Statistic: Delta = (VSA - BEST over R1 on transformed codes) - (VSA - BEST over R1 on the original codes,
from the panel run; same queries), paired cluster bootstrap. "Changed" if |Delta| >= 2 points with the 95% interval
excluding 0. Primary runs first (qwen8 T1, e5 T2, qwen8 T3); replications after them, reported as not run if the
deadline (2026-11-20 06:00 box clock) comes first.

Readings (fixed now):
- **HD supported:** T1 Delta > -2 (no drop by the bar) and T2 Delta < +2 (no rise by the bar) and T3 Delta <= -2 with
  its interval below 0.
- **HA supported:** T1 Delta <= -2 (interval below 0) and T2 Delta >= +2 (interval above 0).
- Otherwise: neither account explains the interventions; the paper reports the geometry and the interventions without
  a mechanism claim. T3 alone cannot separate HD from relative anisotropy (it also raises PR / n); it tests whether
  capping the dimension is sufficient to remove the advantage.
- Panel-level association (descriptive, 7 encoders): Spearman correlation of the R1 margin (VSA - BEST) with whitened
  PR, whitened d', and centred PR / n.

## 7. How each outcome will be read (fixed now)

- H1-B49 passes on all four LLM families and on no sentence encoder: "At equal bits, in the few-values, low-budget
  region, the superposed store beats the best dev-chosen competitor (key-free sketches, static functions, Bloom
  filters, SDM, the per-fact index, a least-squares associative memory, learned hashing) on hidden-state codes of four
  LLM families (each layer and pooling chosen by an alias-retrieval criterion) and on none of three sentence encoders."
  With the margins, accuracies (and distance from EXACT), and where the store loses.
- Passes on k < 4 LLM families: the paper names the families with and without a win; no statement about "LLM codes".
- Any sentence encoder passes: named; the LLM / sentence-encoder dichotomy is not claimed.
- If BEST in R1 is mostly LAM and LAM ties or beats the store: the paper says a least-squares superposition memory
  matches or beats the MAP store; the advantage over classical (key-storing or bucket) memories stands only if
  VSA - BEST-noLAM (BEST over the nine families other than LAM, reported for every encoder) still passes, and the
  VSA-specific claim does not.
- If the learned-hashing families make BEST and remove the win: the paper says supervised hashing trained on aliases
  closes the gap, and reports VSA-L (whether the store gains as much from the learned metric).
- Mechanism: the reading of section 6. Under HD supported: "the store's advantage tracks the number of usable
  dimensions of the codes after whitening, not their anisotropy"; under HA supported the converse; otherwise no
  mechanism claim.
- Useful accuracy: wherever the store reaches U within 64 bits per fact, the paper gives that budget beside the best
  competitor's; wherever it does not and VSA-inf < U, the paper must say that in that setting no budget makes the
  store reach 90% of the exact index's accuracy, and that the R1 wins are at accuracies far below EXACT.

## 8. Predictions (stated so that they can be wrong)

P1. Against BEST36 (B36's competitors), R1, ideal: llama and mistral pass H1 (>= +2, CI > 0); bge and minilm do not.
P2. Against BEST (all ten families): between 2 and 4 encoders pass, all of them LLM families; no sentence encoder passes.
P3. |VSA - LAM| <= 3 points over R1 on at least 5 of the 7 encoders; LAM is the dev BEST in at most 4 of the 12 R1 cells
    of any encoder.
P4. The learned map raises B32's dev criterion over the centred codes on every encoder; LHI beats IDX over R1 by >= +2
    points on at least 4 of the 7 encoders.
P5. Over R1, VSA - BEST is at least 1 point lower than VSA - BEST36 on average over the 7 encoders.
P6. VSA-L - VSA over R1 lies between -3 and +3 on every LLM encoder.
P7. Useful accuracy: the store reaches U within 64 bits per fact in none of the 28 settings (7 encoders x M x V in
    {2, 16}); VSA-inf < U in all 16 LLM settings; BEST reaches U within 64 bits per fact in at least 3 of the 12
    sentence-encoder settings and in at most 4 of the 16 LLM settings.
P8. Mechanism (HD): T1 Delta > -2 on qwen8; T2 Delta < +2 on e5; T3 Delta <= -2 with the interval below 0 on qwen8, and
    the T3 margin is at most +3.7 (half-way to e5's +0.6).
P9. Spearman(R1 margin vs BEST, whitened PR) >= 0.6 over the 7 encoders.
P10. Checks: independent scorer |difference| < 1e-9; 0 configurations over budget; re-derivation agreement >= 99% for
    B36's families and the VSA lin / LSH index on the new encoders (as B36 / B46), >= 98% for LAM (an independent primal
    implementation; near-ties of a ridge solve may flip) and for LHI LSH.

## 9. Checks (second angle)

- b36_check.py score / bits / rederive on each new encoder (as b46_check.py, encoder substituted).
- b49_check.py: an independent scorer of every cell of the merged files (VSA - BEST, VSA - each family; |difference|
  from the verdict < 1e-9); bits audit of LAM (p from the written rule, bits <= budget) and of the LH families (logged
  bits <= budget); re-derivation of LAM on 10% of test stores by an independent implementation (primal ridge,
  numpy.linalg.lstsq on augmented rows; own quantizer), and of LHI LSH predictions with the learned map recomputed by
  scipy.linalg.eigh's generalized solver.
- Hand reading: per new encoder, 10 queries only the VSA answers right and 10 only BEST answers right, in the R1 cell
  with the largest difference and in the R1 cell where BEST leads most (b36_handread.py pattern); notes by me.
- Literature: results/b49_literature.txt, updated after the result (support / conflict / gap).

## 10. Compute

Encoding: none (all codes exist). Memory runs: local CPU (16 cores, shared). A800 box: file copies only; GPU cap 10
GPU-hours, none planned. No paid API. Ledger: b49_out/compute_ledger.txt.

## Files

Code: b49_geom.py, b49_lhchoose.py, b49_common.py, b49_run.py, b49_dev.sh, b49_test.sh, b49_fam.py (new families),
b49_mech.py (interventions), b49_verdict.py, b49_check.py, b49_handread.py. Working directory b49_out/ (codes and dumps
not committed). Results: results/b49_*.
