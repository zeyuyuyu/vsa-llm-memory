# B67 pre-registration: theory predictions against the measurements (referee M2)

Date: 2026-11-23 (machine clock Asia/Shanghai, about 16:35). Written and committed, then OpenTimestamps-stamped, before
any B67 theory number is computed. The paper is not edited by B67. CPU only, this machine. Time box 2.5 hours from
about 15:20.

Referee M2: "Give quantitative theory predictions and compare them with the measurements: (a) the predicted
Hebbian-versus-least-squares gap from the measured key-covariance spectra, and (b) the predicted cells per pair from the
capacity expression in the Methods for the dense and sparse-block stores. Show where measurement departs from theory."

## What has been seen before this file (disclosed)

- (a) Every measured number is known: B60's STORE - LAMc and B66's STORE - LSFR region and cell accuracies
  (results/b60_verdict.txt, results/b66_B_verdict.{txt,json}); the store's, LSFR's and LAMc's dev choices
  (results/b53_choice_*, b66_lsf_choice_*, b60_choice_*). Tally of the store's 84 ideal R1 choices, read while designing:
  whiten-lin-q1-f1 in 58, center-lin-q1 in 7, center-sgn3-q1 in 6, other whiten lin / sgn in 13. **No theory number of
  B67 has been computed**; no key spectrum has been computed in this session.
- (b) Every measured N* is known (B38 / B59 / B61 / B63) and so is B61's scratch closed-form model, registered in
  prereg/B61: dense 76 / 76 / 91 / 108 / 152 / 431 / 1,218, sparse block 32 / 32 / 45 / 54 / 91 / 304 / 1,024 at k1 /
  k2 / k4 / e05 / k8 / k16 / k32 (measured: dense 76 / 76 / 76 / 91 / 128 / 431 / 1,722; sparse block 32 / 32 / 45 / 54
  / 91 / 304 / 1,448; best table 19 / 19 / 27 / 38 / 54 / 215 / 724). (b) is therefore NOT blind; its contribution is
  the Methods expression itself, the published Bloom-filter bound, the tables' Shannon bounds, a re-derivation of the
  finite-size model in new code with the stores' own structure, and the ratios.

## (a) Hebbian matched filter against least squares at equal bits: Gaussian cross-talk theory

Cells: R1 = V in {2, 16} x 4, 8, 16 bits per fact x M in {256, 1,024}, 7 encoders = 84 cells, ideal accounting.
Arms: STORE = B53's final design (its dev-chosen prep, fmap, q, stored share f; d = b32_common.vsa_dim(bpf M, q,
'ideal')); LSFR = B66's least squares on the store's own features (its dev-chosen scale, target, alpha, q; p by
b60_run.p_of with pmax 65,536); LAMc = B60's least squares (its dev-chosen scale, target, prep, feat, alpha, q).
Measured: the test accuracies in results/b66_B_verdict.json 'cells' (ideal). **Primary pair: STORE - LSFR** (same
features, only the write differs: the comparison the theory speaks to). Secondary: STORE - LAMc.

**Inputs (no test data).** Per encoder and prep: the preprocessed codes of the DEV split (labels and aliases; prep fitted
on the training split as in B32), because the training split is in-sample for the whitening and would understate the
cross-talk. C2 = second-moment matrix of the preprocessed dev labels and aliases (n x n); the alias-label pairs (q, k) of
the dev entities (2,152 pairs); the store's own projection blocks (b32_run.Feat seeds) and LAMc's maps (b60_run.FeatC).
The code refuses to build any test store or to read any test code row (runtime guard; check C2 below).

**Theory H (Hebbian, matched filter).** Normalised score of value v for query q: z_v = sum_i K(q, k_i) [v_i = v] + N_v,
where K is the feature map's kernel in the infinite-dimension limit (lin: q.k; sgn: (2/pi) arcsin(q.k); sgn3: its cube;
codes unit-norm after prep) and N_v the finite-dimension and quantization noise. Standard results: Gaussian cross-talk
of a superposition (Plate 2003; Frady, Kleyko & Sommer 2018: p_corr = integral phi(x) Phi(x + s)^(V-1) dx); Max
quantizer gain / distortion (sign: the factor pi / 2). Per dev pair: signal mu = K(q, k); intrinsic cross-talk from the
covariance spectrum, m2(q) = E_k'[K(q, k')^2] (lin: q^T C2 q; sgn / sgn3: Gaussian expectation of the kernel with
variance q^T C2 q, by Gauss-Hermite quadrature); kappa_q = E[e^2] / (a^2 sigma^2) of b32_common's quantizer on a unit
Gaussian (computed numerically; q = 1 gives pi / 2 - 1); finite-dimension noise sigma_N^2 = (1 + kappa_q) Ms / d (Ms =
floor(f M) stored facts). V = 2 (antipodal codes): P = Phi(mu / sqrt((Ms - 1) m2 + sigma_N^2)). V = 16 (orthogonal
codes; cross-value terms are noise of order 1 / d): race with V - 1 competitors, per-score variance
(Ms - 1) m2 / V + sigma_N^2, P = E_x[Phi((mu + sigma x) / sigma)^(V-1)] (Gauss-Hermite, 80 nodes). Cell prediction =
f x mean over dev pairs + (1 - f) / V.

**Theory L (least squares = the matched filter after decorrelation).** The ridge read is score_v = sum_i K_LS(q, k_i)
T_iv with K_LS(x, y) = f(x)^T (F^T F + lam I)^-1 f(y): the Hebbian read in the whitened feature space (Kohonen 1972;
Personnaz, Guyon & Dreyfus 1986). Its statistics from the feature covariance Sigma (p x p; lin / rp / pca: the exact
push-through of C2 through the map; sgn: Gaussian-equivalent features, (2/pi) on the normalised linear part plus white
noise 1 - 2/pi (Mei & Montanari 2022); sgn3: white, Sigma = I, its linear part vanishing) by the ridge deterministic
equivalent (Dobriban & Wager 2018; Hastie et al. 2022): lam_e = alpha tr(Sigma) / M; kappa solves kappa - lam_e =
(kappa / M) tr(Sigma (Sigma + kappa)^-1); kappa' = 1 / (1 - tr(Sigma^2 (Sigma + kappa)^-2) / M). Per dev pair (f_t =
f(k), delta = f(q) - f(k)): a = (kappa / (lam_e M)) f_t^T (Sigma + kappa)^-1 f_t, self-recall h = a / (1 + a); signal
mu = h + (kappa / (lam_e M)) delta^T (Sigma + kappa)^-1 f_t / (1 + a); residual cross-talk among stored keys
r = (1 - h)^2 (kappa' - 1); alias noise g = (kappa' / M) delta^T Sigma (Sigma + kappa)^-2 delta; quantization of W:
per-entry variance v_W = tau^2 (kappa' / M) tr(Sigma (Sigma + kappa)^-2) / p (tau^2 = (1 / V)(1 - 1 / V)), score noise
kappa_q v_W ||f(q)||^2 per stored column read. Same race as H with per-score variance (r + g) / V + quantization noise
(target 'V-1' at V = 2: one column read twice, P = Phi(mu / sqrt(r + g + 4 kappa_q v_W ||f(q)||^2)); 'V' and 'V-1' at
V = 16: per-score kappa_q v_W ||f(q)||^2, value 0 of 'V-1' (V - 1) times it). All M facts are stored (f = 1).

**Secondary (compute permitting; cut first): theory H-G**, the same as H with m2 replaced by the exact cross-talk
sum over each DEV store's other keys (B32's dev stores), to tell the homogeneous-spectrum approximation from the
Gaussian-race approximation.

**Agreement metrics (fixed now).** Over the 84 cells, for STORE - LSFR (primary) and STORE - LAMc:
- M1 (primary): Spearman rank correlation of predicted and measured gap; 95% interval by a bootstrap over cells
  (10,000 draws).
- M2: sign agreement (share of cells whose predicted and measured gaps have the same sign).
- M3: band: share of cells with |measured gap - predicted gap| <= 3 points (also <= 5).
- M4: per arm, measured - predicted accuracy: mean (bias) and mean absolute error, overall and by V; Spearman of
  accuracy over the 84 cells and within each V.
- M5: per encoder, region R1 mean of the predicted and the measured gap and their ratio.
Departure map: the cells with the largest |measured - predicted| per arm, by V, M, budget and encoder group.

**Predictions (stated so that they can fail).**
- PA1: theory predicts least squares ahead of the store at the same features (predicted STORE - LSFR < 0) in at least
  60 of 84 cells and at the R1 level on all 7 encoders; STORE - LAMc predicted < 0 in at least 70 of 84.
- PA2: M1 >= 0.4 for STORE - LSFR and for STORE - LAMc.
- PA3: accuracy Spearman >= 0.9 for each arm over the 84 cells; store MAE <= 5 points, least-squares MAE <= 7.
- PA4: M3 (3-point band) holds in at least 60% of cells for STORE - LSFR.
- PA5: the departures are larger at V = 16 than at V = 2 for the store (MAE), and the theory over-predicts the store
  there (mean bias > 0): the Gaussian race misses the heavy-tailed cross-talk of near-neighbour keys.
- PA6: predicted R1 STORE - LSFR within +-2 points of the measured on at least 4 of 7 encoders.

## (b) Cells per stored pair at 99.5% exact recall

Conditions: B38's i.i.d. cell-error rates k1 0, k2 1.3175e-4, k4 0.0208065, e05 0.05, k8 0.12263475, k16 0.27600645,
k32 0.374429. Store structure from B34's 50 RULER test stores (counts only: pairs n, value vocabulary N_v, values per
probed key, shard loads by B38's hash; no cell is read, no outcome used); a key is read exactly when all its slots
(one per stored value) are right; target: mean over stores of the share of failed probed keys <= 0.5%. Grid B =
2^(i/4), 8 ... 4,096; a prediction is reported both as a real number and rounded up to the grid.

- **D-a (Methods 4.1, as written, leading order).** k* = d / (2 ln(N / eps)) with N = 2 N_v - 1 candidates (value,
  value x LAST), eps = per-slot error, sign storage (power SNR x 2 / pi) and cell errors (x (1 - 2 e)^2):
  B = pi ln(N / eps) / (1 - 2 e)^2; eps from the key target with the mean slots per probed key.
- **D-b (finite-size Gaussian race).** Both the true and the N wrong candidates' overlaps noisy:
  P_slot = 1 - integral phi(x) Phi(x + s)^N dx, s^2 = (2 / pi) (d / k_shard) (1 - 2 e)^2 with each shard's own load
  k_shard (B38's hash), per store and key, averaged over the 50 stores.
- **S-a (Bloom bound; Bloom 1970; Clarkson, Ubaru & Yang 2023).** OR superposition, no cell errors, optimal load
  ln 2: B = log2(N / eps) / ln 2.
- **S-b (binomial overlap, B61's model re-derived).** True overlap ~ Binomial(nb, 1 - e), each wrong ~ Binomial(nb,
  rho'), rho' = rho (1 - e) + (1 - rho) e, rho = 1 - exp(-lam), nb = lam B (per shard load), ties half; best lam in
  {0.35, 0.5, 0.7, 1.0}.
- **T-a / T-b (tables).** T-a: B38's directory-free Shannon bound (ceil(log2 N_v) + last flag) / (1 - H2(e)) (max over
  stores). T-b: the same with the directory: the stores' logical bits per pair (MPHF displacements + rows, b38_eq's
  mphf_content) / (1 - H2(e)). Code rate: the measured best table's cells / T-b's bits = the effective rate.
Report: per condition and memory, measured N*, each prediction, measured / predicted, grid steps apart.

**Predictions.**
- PB1: D-a under-predicts the dense store at every condition (measured / D-a >= 1.2 at all 7).
- PB2: D-b within 2 grid steps of the measured dense N* at k1 to k16; measured above D-b by at least 1 step at k32.
- PB3: S-a within 1 grid step of the measured sparse-block N* at k1; S-b within 2 steps at k1 to k16; measured above
  S-b by >= 2 steps at k32.
- PB4: best table / T-b between 1.0 and 2.5 at every condition; best table / T-b <= 1.6 at k1.

## Checks (b67_check.py, a second short script written separately)

- C1 independent re-computation: (a) for every one of the 84 cells, H and L re-derived by Monte Carlo of the Gaussian
  model (draw the cross-talk and noise, 20,000 draws per cell on a 200-pair subsample) and the deterministic equivalent
  solved by fixed-point iteration (main: bisection); agreement with the main numbers within 1.5 points in every cell
  (Monte Carlo error included); (b) every prediction recomputed with scipy.stats (binom / norm) in plain loops; the
  grid-rounded N* identical.
- C2 inputs: the split of every row that enters a theory number is 'dev' (assert on the entity split field; counts
  reported); a guard makes b32_common.make_store raise on split 'test'; (b) reads only the stores' structure.
- C3 selftest of the formulas on synthetic isotropic keys (exact and alias-perturbed queries): H and L against a direct
  simulation of the store and of ridge least squares at small sizes, agreement within 2 points.

## Disclosure and cuts

Any cut is written into a dated compute note before the comparison is read. Order if short: (a) on the 4 language
models (48 cells), then (b), then (a) on the 3 sentence encoders, then H-G.

## Files

b67_theory.py (theory (a) and (b), comparison, verdict), b67_check.py (C1-C3); results/b67_*.
