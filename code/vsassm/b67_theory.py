"""B67 (prereg/B67_theory_vs_measurement_prereg.md): theory predictions against the measurements (referee M2).

(a) Gaussian cross-talk theory of the Hebbian matched filter (the store) and of ridge least squares read as the matched
    filter after decorrelation (deterministic equivalent), from the DEV split's keys only, at each R1 cell's chosen
    configuration; compared with the measured test accuracies of B66 / B60.
(b) cells per stored pair at 99.5% exact recall: Methods 4.1's leading-order expression, a finite-size Gaussian race,
    the Bloom-filter bound, a binomial-overlap model and the tables' Shannon bounds, from the RULER test stores'
    structure only (counts; no cell read), against the measured N*.

python3 b67_theory.py selftest
python3 b67_theory.py a ENC            # -> results/b67_a_ENC.json
python3 b67_theory.py b                # -> results/b67_b.json
python3 b67_theory.py verdict          # -> results/b67_verdict.{json,txt}
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402

RES = os.path.join(ROOT, 'results')
DATA = os.path.join(ROOT, 'b49_out')
ENC_TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
           'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
CELLS = [(M, V, b) for M in (256, 1024) for V in (2, 16) for b in (4, 8, 16)]
CH = 8192
HX, HW = np.polynomial.hermite_e.hermegauss(80)  # probabilists' Gauss-Hermite: E[g(x)], x ~ N(0, 1)
HW = HW / HW.sum()


# ----------------------------------------------------------------------------------------------- guard (check C2)

_orig_make_store = C.make_store


def _guard_make_store(pool, split, *a, **k):
    if split == 'test':
        raise RuntimeError('B67 guard: no test store may be built')
    return _orig_make_store(pool, split, *a, **k)


C.make_store = _guard_make_store


# ----------------------------------------------------------------------------------------------- quantizer constants

def kappa_q(q):
    """E[e^2] / (a^2 sigma^2) of b32_common's quantizer on a unit Gaussian (Q(x) = a x + e, e uncorrelated)."""
    if q >= 32:
        return 0.0
    x = np.linspace(-12, 12, 400001)
    w = stats.norm.pdf(x)
    w = w / w.sum()
    if q == 1:
        Qx = np.sign(x)
    else:
        step = C.MAX_STEP[q]
        L = 2 ** q
        Qx = (np.clip(np.floor(x / step), -L // 2, L // 2 - 1) + 0.5) * step
    a = float((w * Qx * x).sum())
    eq2 = float((w * Qx * Qx).sum())
    return (eq2 - a * a) / (a * a)


KQ = {q: kappa_q(q) for q in (1, 2, 4)}


# ----------------------------------------------------------------------------------------------- race

def race(mu, s_true, s_comp):
    """P(true score > every competitor): true ~ N(mu, s_true^2), competitors independent N(0, s_w^2).
    mu, s_true: arrays (pairs); s_comp: list of (array of std, multiplicity). Gauss-Hermite over the true score."""
    mu = np.asarray(mu, float)[:, None]
    st = np.asarray(s_true, float)[:, None]
    t = mu + st * HX[None, :]
    lp = np.zeros_like(t)
    for sw, m in s_comp:
        lp += m * stats.norm.logcdf(t / np.asarray(sw, float)[:, None])
    return (np.exp(lp) * HW[None, :]).sum(1)


# ----------------------------------------------------------------------------------------------- kernels

def kern(u, fmap):
    u = np.clip(u, -1.0, 1.0)
    if fmap == 'lin':
        return u
    k = (2 / np.pi) * np.arcsin(u)
    return k if fmap == 'sgn' else k ** 3


def kern_m2(s2, fmap):
    """E[K(u)^2], u ~ N(0, s2) (the cross-talk second moment from the spectrum: s2 = q^T C2 q)."""
    if fmap == 'lin':
        return s2
    u = np.sqrt(s2)[:, None] * HX[None, :]
    return (kern(u, fmap) ** 2 * HW[None, :]).sum(1)


# ----------------------------------------------------------------------------------------------- dev data

class DevData:
    """Preprocessed codes of the dev split only: labels, aliases, (alias, label) pairs, C2 (second moment of
    preprocessed dev labels and aliases)."""

    def __init__(self, enc):
        cwd = os.getcwd()
        os.chdir(DATA)
        self.pool = C.Pool(ENC_TAG[enc])
        os.chdir(cwd)
        pool = self.pool
        Tr = pool.train_matrix()
        self.Tr = Tr
        self.preps = {p: C.Prep(p, Tr) for p in ('center', 'whiten')}
        dv = pool.by_split['dev']
        assert all(e['split'] == 'dev' for e in dv)
        self.lab_rows = [pool.idx[e['name']] for e in dv]
        self.pair_lab, self.pair_ali = [], []
        for j, e in enumerate(dv):
            for a in e['aliases']:
                self.pair_lab.append(j)
                self.pair_ali.append(pool.idx[a])
        self.pair_lab = np.array(self.pair_lab)
        self.n = pool.X.shape[1]
        self.split_rows = set(self.lab_rows) | set(self.pair_ali)
        self.cache = {}

    def prepped(self, prep):
        if prep not in self.cache:
            P = self.preps[prep]
            L = P(self.pool.X[self.lab_rows]).astype(np.float64)
            A = P(self.pool.X[self.pair_ali]).astype(np.float64)
            Y = np.vstack([L, A])
            C2 = Y.T @ Y / len(Y)
            self.cache[prep] = (L, A, C2)
        return self.cache[prep]


# ----------------------------------------------------------------------------------------------- theory H

def theory_H(dd, prep, fmap, q, f, M, V, d):
    L, A, C2 = dd.prepped(prep)
    Lp = L[dd.pair_lab]
    u = (A * Lp).sum(1)
    mu = kern(u, fmap)
    s2 = np.einsum('ij,jk,ik->i', A, C2, A)
    m2 = kern_m2(s2, fmap)
    Ms = max(1, int(math.floor(f * M)))
    if V == 2:
        sN2 = (1 + KQ[q]) * Ms / d
        P = stats.norm.cdf(mu / np.sqrt((Ms - 1) * m2 + sN2))
    else:
        # mean kernel between distinct keys (enters the stored coordinates' variance through same-value pairs)
        if fmap == 'lin':
            sL = L.sum(0)
            mbar = float((sL @ sL - (L * L).sum()) / (len(L) * (len(L) - 1)))
        else:
            h = len(L) // 2
            mbar = float(np.mean(kern((L[:h] @ L[h:2 * h].T).ravel(), fmap)))
        s2S = Ms + Ms * (Ms - 1) * mbar / V
        sN2 = (1 + KQ[q]) * max(s2S, 1.0) / d
        sig = np.sqrt((Ms - 1) * m2 / V + sN2)
        P = race(mu, sig, [(sig, V - 1)])
    acc = f * float(P.mean()) + (1 - f) / V
    return acc, {'mu': float(mu.mean()), 'm2': float(m2.mean()), 'sN2': float(sN2), 'Ms': Ms, 'd': d,
                 'snr_mean': float(np.mean(mu / np.sqrt((Ms - 1) * m2 + sN2)))}


# ----------------------------------------------------------------------------------------------- features for theory L

def store_G(enc, prep, fmap, n, p):
    """The store's own projection (b32_run.Feat seeds), first factor, first p columns (n x p)."""
    K = 1 if fmap in ('lin', 'sgn') else 3
    blocks = []
    for j0 in range(0, p, CH):
        w = min(CH, p - j0)
        blocks.append(np.random.default_rng(C.seed_of('G', enc, prep, fmap, 0, j0, CH)).standard_normal(
            (n, CH), dtype=np.float32)[:, :w])
    return np.hstack(blocks).astype(np.float64), K


def lam_G(enc, prep, n, p):
    """LAMc's rp map (b49_common.LamFeat blocks, seed ('LAMG', enc, prep, j0), / sqrt(n))."""
    blocks = []
    for j0 in range(0, p, CH):
        w = min(CH, p - j0)
        blocks.append((np.random.default_rng(C.seed_of('LAMG', enc, prep, j0)).standard_normal(
            (n, CH), dtype=np.float32) / np.float32(math.sqrt(n)))[:, :w])
    return np.hstack(blocks).astype(np.float64)


def pca_U0(dd):
    if 'U0' not in dd.cache:
        Tr = np.asarray(dd.Tr, np.float64)
        w, U = np.linalg.eigh(np.cov((Tr - Tr.mean(0)).T))
        dd.cache['U0'] = U[:, np.argsort(-w)]
    return dd.cache['U0']


class Spec:
    """Sigma = Q diag(lam) Q^T + nu (I - Q Q^T) on R^p, with per-row coordinates u = Q^T f and rest = ||f||^2 - ||u||^2.
    Built from the feature map applied to the dev rows."""

    def __init__(self, lam, nu, p, uK, rK, uD, rD, cKD, fq2):
        self.lam, self.nu, self.p = lam, nu, p
        self.uK, self.rK, self.uD, self.rD, self.cKD, self.fq2 = uK, rK, uD, rD, cKD, fq2


def spec_linear(L, A, C2, G, pair_lab):
    """Linear features f = G^T y. p <= n: eigen of G^T C2 G (p x p); p > n: Cholesky of P = G G^T and eigen of
    L^T C2 L (n x n), same nonzero spectrum and quadratic forms (push-through)."""
    n, p = G.shape
    Lp = L[pair_lab]
    D = A - Lp
    if p <= n:
        S = G.T @ C2 @ G
        lam, W = np.linalg.eigh((S + S.T) / 2)
        T = G @ W
    else:
        P = G @ G.T
        Lc = np.linalg.cholesky(P + 1e-9 * np.trace(P) / n * np.eye(n))
        B = Lc.T @ C2 @ Lc
        lam, W = np.linalg.eigh((B + B.T) / 2)
        T = Lc @ W
    lam = np.maximum(lam, 0)
    uK, uD = Lp @ T, D @ T
    fq2 = np.einsum('ij,ij->i', A @ G, A @ G) if p <= 4 * n else np.einsum('ij,jk,ik->i', A, G @ G.T, A)
    z = np.zeros(len(Lp))
    return Spec(lam, 0.0, p, uK, z, uD, z, z, fq2)


def spec_sign(L, A, C2, G, pair_lab, K):
    """Sign features: Gaussian-equivalent covariance. K = 1: (2/pi) Gh^T C2 Gh + (1 - 2/pi) I with Gh = G / sd(g.y);
    K = 3: I. Per-row coordinates from the actual sign features of the first factor (K = 1) or the product of three
    (K = 3, factors 1..3 from the store's seeds)."""
    n, p = G.shape
    Lp = L[pair_lab]
    if K == 1:
        fK = np.where(Lp @ G >= 0, 1.0, -1.0)
        fA = np.where(A @ G >= 0, 1.0, -1.0)
        sd = np.sqrt(np.maximum(np.einsum('ji,jk,ki->i', G, C2, G), 1e-30))
        Gh = G / sd[None, :]
        nu = 1 - 2 / np.pi
        if p <= n:
            S = (2 / np.pi) * (Gh.T @ C2 @ Gh)
            lam, W = np.linalg.eigh((S + S.T) / 2)
            Q = W
        else:
            P = Gh @ Gh.T
            Lc = np.linalg.cholesky(P + 1e-9 * np.trace(P) / n * np.eye(n))
            B = (2 / np.pi) * (Lc.T @ C2 @ Lc)
            lam, W = np.linalg.eigh((B + B.T) / 2)
            Q = Gh.T @ np.linalg.solve(Lc.T, W)  # p x n orthonormal: Gh^T L^-T W
        lam = np.maximum(lam, 0) + nu
        uK, uA = fK @ Q, fA @ Q
        uD = uA - uK
        D = fA - fK
        rK = p - (uK ** 2).sum(1)
        rD = (D ** 2).sum(1) - (uD ** 2).sum(1)
        cKD = (fK * D).sum(1) - (uK * uD).sum(1)
        return Spec(lam, nu, p, uK, rK, uD, rD, cKD, np.full(len(Lp), float(p)))
    raise ValueError(K)


def spec_white(fK, fA):
    p = fK.shape[1]
    D = fA - fK
    e = np.zeros((len(fK), 0))
    return Spec(np.zeros(0), 1.0, p, e, (fK ** 2).sum(1), e, (D ** 2).sum(1), (fK * D).sum(1),
                (fA ** 2).sum(1))


def de_solve(sp, lam_e, M, method='bisect'):
    """kappa - lam_e = (kappa / M) df1(kappa); df1 = tr Sigma (Sigma + kappa)^-1 (rest of the spectrum at nu)."""
    lam, nu, p = sp.lam, sp.nu, sp.p
    nrest = p - len(lam)

    def df1(k):
        return float((lam / (lam + k)).sum() + (nrest * nu / (nu + k) if nu > 0 else 0.0))

    def F(k):
        return k * (1 - df1(k) / M) - lam_e
    if method == 'bisect':
        lo, hi = 1e-300, max(1.0, float(lam.sum() + nrest * nu)) * 10 + lam_e * 10
        while F(hi) < 0:
            hi *= 10
        for _ in range(400):
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 2
            if F(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-12:
                break
        k = math.sqrt(lo * hi)
    else:  # fixed point kappa = lam_e / (1 - df1(kappa) / M), damped (used by the check)
        k = lam_e + 1e-12
        for _ in range(20000):
            k_new = lam_e / max(1e-300, 1 - df1(k) / M) if df1(k) < M else k * 2
            k = math.sqrt(k * k_new)
    df2 = float((lam ** 2 / (lam + k) ** 2).sum() + (nrest * nu ** 2 / (nu + k) ** 2 if nu > 0 else 0.0))
    kp = 1.0 / max(1e-12, 1 - df2 / M)
    trS2 = float((lam / (lam + k) ** 2).sum() + (nrest * nu / (nu + k) ** 2 if nu > 0 else 0.0))
    return k, kp, trS2


def theory_L(sp, M, V, alpha, q, target, corrected=False):
    trS = float(sp.lam.sum() + (sp.p - len(sp.lam)) * sp.nu)
    lam_e = alpha * trS / M
    k, kp, trS2 = de_solve(sp, lam_e, M)
    g1 = 1.0 / (sp.lam + k)
    g2 = sp.lam / (sp.lam + k) ** 2
    nu = sp.nu
    A1KK = (sp.uK ** 2 * g1).sum(1) + (sp.rK / (nu + k) if nu > 0 else 0.0)
    A1DK = (sp.uD * sp.uK * g1).sum(1) + (sp.cKD / (nu + k) if nu > 0 else 0.0)
    A2DD = (sp.uD ** 2 * g2).sum(1) + (sp.rD * nu / (nu + k) ** 2 if nu > 0 else 0.0)
    c = lam_e * M / k  # (kappa / (lam_e M))^-1
    h = A1KK / (c + A1KK)
    shift = A1DK / (c + A1KK)
    mu = h + shift
    r = (1 - h) ** 2 * (kp - 1)
    g = (kp / M) * A2DD
    if corrected:
        # EXPLORATORY (compute note 2, post hoc): the cross-talk of the query's own residual v = f(q) - mu f(k) through
        # the leave-one-out inverse, (kappa' / M) v^T Sigma (Sigma + kappa)^-2 v, keeping the cross term that the
        # registered r + g drops
        A2KK = (sp.uK ** 2 * g2).sum(1) + (sp.rK * nu / (nu + k) ** 2 if nu > 0 else 0.0)
        A2KD = (sp.uK * sp.uD * g2).sum(1) + (sp.cKD * nu / (nu + k) ** 2 if nu > 0 else 0.0)
        tot = (kp / M) * ((1 - mu) ** 2 * A2KK + 2 * (1 - mu) * A2KD + A2DD)
        r, g = np.maximum(tot, 1e-30), np.zeros_like(tot)
    tau2 = (1 / V) * (1 - 1 / V)
    vW = tau2 * (kp / M) * trS2 / sp.p
    b = KQ[q] * vW * sp.fq2
    if V == 2:
        if target == 'V-1':
            P = stats.norm.cdf(mu / np.sqrt(r + g + 4 * b))
        else:
            P = stats.norm.cdf(mu / np.sqrt(r + g + 2 * b))
    else:
        cx = (r + g) / V
        if target == 'V':
            s = np.sqrt(cx + b)
            P = race(mu, s, [(s, V - 1)])
        else:
            s, s0 = np.sqrt(cx + b), np.sqrt(cx + (V - 1) * b)
            P = (V - 1) / V * race(mu, s, [(s, V - 2), (s0, 1)]) + (1 / V) * race(mu, s0, [(s, V - 1)])
    return float(P.mean()), {'kappa': k, 'kprime': kp, 'lam_e': lam_e, 'h': float(h.mean()), 'shift':
                             float(shift.mean()), 'r': float(r.mean()), 'g': float(g.mean()), 'b': float(b.mean())}


# ----------------------------------------------------------------------------------------------- (a) per encoder

def ls_p(Bud, V, target, q, scale, pmax):
    import b60_run as L60
    return L60.p_of(Bud, L60.ncols(V, target), q, 'ideal', scale, pmax)


def run_a(enc):
    t0 = time.time()
    dd = DevData(enc)
    n = dd.n
    chS = json.load(open(f'{RES}/b53_choice_{enc}.json'))['choice']
    chR = json.load(open(f'{RES}/b66_lsf_choice_{enc}.json'))['choice']
    chL = json.load(open(f'{RES}/b60_choice_{enc}.json'))['choice']
    out = {'enc': enc, 'n': n, 'dev_pairs': int(len(dd.pair_ali)), 'dev_labels': len(dd.lab_rows),
           'splits_used': sorted({e['split'] for e in dd.pool.by_split['dev']}), 'cells': {}}
    specs = {}
    for (M, V, b) in CELLS:
        key = f'{M}|{V}|{b}|ideal'
        prep, fmap, q, f = chS[key + '|VSA']['cfg']
        d = C.vsa_dim(b * M, q, 'ideal')
        aH, iH = theory_H(dd, prep, fmap, q, f, M, V, d)
        rec = {'STORE': aH, 'STORE_info': iH, 'STORE_cfg': [prep, fmap, q, f]}
        # LSFR: least squares on the store's own features
        cr = chR[key + '|LSFR']
        scale, target, alpha, qL = cr['cfg']
        rprep, rfmap = cr['feat']
        p = ls_p(b * M, V, target, qL, scale, 65536)
        L_, A_, C2 = dd.prepped(rprep)
        sk = ('R', rprep, rfmap, p)
        if sk not in specs:
            G, K = store_G(enc, rprep, rfmap, n, p)
            if rfmap == 'lin':
                specs[sk] = spec_linear(L_, A_, C2, G, dd.pair_lab)
            elif rfmap == 'sgn':
                specs[sk] = spec_sign(L_, A_, C2, G, dd.pair_lab, 1)
            else:
                fK = np.ones((len(dd.pair_lab), p))
                fA = np.ones((len(dd.pair_lab), p))
                Lp = L_[dd.pair_lab]
                for m in range(3):
                    Gm = np.hstack([np.random.default_rng(C.seed_of('G', enc, rprep, rfmap, m, j0, CH)).standard_normal(
                        (n, CH), dtype=np.float32)[:, :min(CH, p - j0)] for j0 in range(0, p, CH)]).astype(np.float64)
                    fK *= np.where(Lp @ Gm >= 0, 1.0, -1.0)
                    fA *= np.where(A_ @ Gm >= 0, 1.0, -1.0)
                specs[sk] = spec_white(fK, fA)
        aR, iR = theory_L(specs[sk], M, V, alpha, qL, target)
        aR2, _ = theory_L(specs[sk], M, V, alpha, qL, target, corrected=True)
        rec.update({'LSFR': aR, 'LSFR_info': iR, 'LSFR_cfg': [scale, target, alpha, qL, rprep, rfmap, p], 'LSFR2': aR2})
        # LAMc
        cl = chL[key + '|LAMC']['cfg']
        scale, target, lprep, feat, alpha, qL = cl
        n_max = 65536 if feat == 'rp' else n
        p = ls_p(b * M, V, target, qL, scale, n_max)
        L_, A_, C2 = dd.prepped(lprep)
        sk = ('L', lprep, feat, p)
        if sk not in specs:
            G = lam_G(enc, lprep, n, p) if feat == 'rp' else pca_U0(dd)[:, :p]
            specs[sk] = spec_linear(L_, A_, C2, G, dd.pair_lab)
        aL, iL = theory_L(specs[sk], M, V, alpha, qL, target)
        aL2, _ = theory_L(specs[sk], M, V, alpha, qL, target, corrected=True)
        rec.update({'LAMC': aL, 'LAMC_info': iL, 'LAMC_cfg': [scale, target, lprep, feat, alpha, qL, p], 'LAMC2': aL2})
        out['cells'][f'{M}|{V}|{b}'] = rec
        print(f'{enc} {key} STORE {aH:.4f} LSFR {aR:.4f} LAMC {aL:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if len(specs) > 6:
            specs.pop(next(iter(specs)))
    json.dump(out, open(f'{RES}/b67_a_{enc}.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------------- (b)

COND = {'k1': 0.0, 'k2': 0.00013175, 'k4': 0.0208065, 'e05': 0.05, 'k8': 0.12263475, 'k16': 0.27600645,
        'k32': 0.374429}
KS = list(COND)
GRIDB = [2 ** (i / 4) for i in range(12, 49)]  # 8 ... 4096
MEAS = {'dense': [76, 76, 76, 91, 128, 431, 1722], 'sbc': [32, 32, 45, 54, 91, 304, 1448],
        'best_table': [19, 19, 27, 38, 54, 215, 724], 'E-BCH': [19, 19, 27, 38, 108, 431, 1722],
        'E-LDPC': [23, 23, 27, 38, 54, 215, 724], 'T-rep': [19, 23, 91, 128, 256, 1024, 3444]}
B61_MODEL = {'dense': [76, 76, 91, 108, 152, 431, 1218], 'sbc': [32, 32, 45, 54, 91, 304, 1024]}
LAMS = [0.35, 0.5, 0.7, 1.0]
TARGET = 0.005


def store_structure():
    """Per RULER test store: n pairs, N_v, slots (values) per probed key, shard of every key (B38's hash) - counts
    only, no cell is read."""
    import gzip
    import b34_inmem as I34
    import ruler_vsa as RV
    stores = json.load(gzip.open(os.path.join(ROOT, 'b34_out/test_stores.json.gz'), 'rt'))
    out = []
    for sid, pairs in stores:
        s = I34.Store(pairs, 'b26b:%s' % sid.replace('/', ':'))
        allk = list(s.cd['allk'])
        slots = [s.count[k] for k in s.keys]
        mp = s.mphf(5)
        disp_bits = int(np.asarray(I34.bits_pm(mp['sigma'], mp['db'])).size)
        row_bits = int(mp['R']) * int(s.payload_bits().shape[1])
        out.append({'sid': sid, 'n': s.n, 'Nv': s.Nv, 'slots': slots, 'keys_all': allk,
                    'pair_keys': list(s.rk), 'logical_bits': disp_bits + row_bits, 'b': s.b, 'lastbit': s.lastbit,
                    'shard_seed': [RV._seed('shard:' + k) for k in allk]})
    return out


def gridup(B):
    for g in GRIDB:
        if g >= B - 1e-9:
            return g
    return None


def steps(a, b):
    """grid steps of a above b (both on the grid or real numbers)."""
    return 4 * math.log2(a / b)


def dense_slot_fail(snr, N):
    """Gaussian race: the true overlap ~ N(snr, 1), N wrong ~ N(0, 1)."""
    t = snr + HX
    lp = N * stats.norm.logcdf(t)
    return 1 - float((np.exp(lp) * HW).sum())


def sbc_slot_fail(nb, e, lam, N):
    """Binomial overlap: true ~ Bin(nb, 1 - e), wrong ~ Bin(nb, rho'), ties half (as a probability split)."""
    rho = 1 - math.exp(-lam)
    rp = rho * (1 - e) + (1 - rho) * e
    ks = np.arange(nb + 1)
    pt = stats.binom.pmf(ks, nb, 1 - e)
    cw = stats.binom.cdf(ks - 1, nb, rp)  # P(wrong < k)
    pe = stats.binom.pmf(ks, nb, rp)
    # P(true strictly beats N wrong) + tie share (first-order tie split)
    win = np.exp(N * np.log(np.maximum(cw, 1e-300)))
    tie = np.exp(N * np.log(np.maximum(cw + pe, 1e-300))) - win
    return 1 - float((pt * (win + 0.5 * tie)).sum())


def key_fail(stores, slot_fail_fn):
    """mean over stores of the share of probed keys with any failed slot."""
    fr = []
    for st in stores:
        pf = slot_fail_fn(st)
        fr.append(float(np.mean([1 - (1 - pf) ** m for m in st['slots']])))
    return float(np.mean(fr))


def run_b():
    t0 = time.time()
    S = store_structure()
    print(f'structure {time.time() - t0:.0f}s', flush=True)
    Nv = [st['Nv'] for st in S]
    slots_mean = float(np.mean([np.mean(st['slots']) for st in S]))
    eps_slot = 1 - (1 - TARGET) ** (1 / slots_mean)
    Nc = 2 * np.array(Nv) - 1
    res = {'stores': len(S), 'Nv_min': int(min(Nv)), 'Nv_max': int(max(Nv)), 'Nv_median': float(np.median(Nv)),
           'slots_mean': slots_mean, 'eps_slot': eps_slot, 'pred': {}}
    Nmed = float(np.median(Nc))
    for kname, e in COND.items():
        r = {}
        # D-a: Methods 4.1 leading order with sign storage and cell errors
        r['D-a'] = math.pi * math.log(Nmed / eps_slot) / (1 - 2 * e) ** 2
        # D-b: finite-size Gaussian race with each store's shard loads (bisection on the grid)
        def dense_fail(B):
            def f(st):
                n = st['n']
                total = int(B * n)
                Ssh = max(1, math.ceil(total / 8192))
                d = min(8192, total // Ssh)
                sh = np.array([int(x) % Ssh for x in st['shard_seed']], dtype=np.int64)
                kidx = {k: i for i, k in enumerate(st['keys_all'])}
                load = np.bincount(sh[[kidx[k] for k in st['pair_keys']]], minlength=Ssh)
                # a probed key's slots sit in its shard: use the load-weighted mean slot failure
                N = 2 * st['Nv'] - 1
                pf = [dense_slot_fail(math.sqrt((2 / math.pi) * d / max(1, L) * (1 - 2 * e) ** 2), N) for L in load]
                w = load / load.sum()
                return float(np.dot(w, pf))
            return key_fail(S, f)
        r['D-b'] = next((g for g in GRIDB if dense_fail(g) <= TARGET), None)
        # S-a: Bloom bound
        r['S-a'] = math.log2(Nmed / eps_slot) / math.log(2)
        # S-b: binomial overlap, best lam, per-store mean load (pairs per shard / Lb = lam)
        def sbc_fail(B, lam):
            def f(st):
                return sbc_slot_fail(max(1, int(round(lam * B))), e, lam, 2 * st['Nv'] - 1)
            return key_fail(S, f)
        best = None
        for lam in LAMS:
            g = next((g for g in GRIDB if sbc_fail(g, lam) <= TARGET), None)
            if g is not None and (best is None or g < best[0]):
                best = (g, lam)
        r['S-b'] = best[0] if best else None
        r['S-b_lam'] = best[1] if best else None
        # tables
        H2 = (-e * math.log2(e) - (1 - e) * math.log2(1 - e)) if e > 0 else 0.0
        r['T-a'] = max((st['b'] + st['lastbit']) for st in S) / (1 - H2)
        r['T-b'] = max(st['logical_bits'] / st['n'] for st in S) / (1 - H2)
        r['T-b_mean'] = float(np.mean([st['logical_bits'] / st['n'] for st in S])) / (1 - H2)
        res['pred'][kname] = r
        print(kname, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()},
              f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(f'{RES}/b67_b.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------------- selftest (C3)

def selftest():
    """Synthetic isotropic keys only: theory H and L against direct simulation at small sizes."""
    rng = np.random.default_rng(0)
    n, Npool = 64, 4000
    X = rng.standard_normal((Npool, n))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Aq = X + 0.6 * rng.standard_normal((Npool, n)) / math.sqrt(n)
    Aq /= np.linalg.norm(Aq, axis=1, keepdims=True)
    C2 = np.vstack([X, Aq]).T @ np.vstack([X, Aq]) / (2 * Npool)

    class DD:
        pass
    dd = DD()
    dd.pair_lab = np.arange(Npool)
    dd.prepped = lambda prep: (X, Aq, C2)
    for (M, V, d, q) in [(64, 2, 256, 1), (128, 2, 512, 2), (64, 16, 1024, 1), (128, 16, 2048, 4)]:
        th, _ = theory_H(dd, 'x', 'lin', q, 1.0, M, V, d)
        accs = []
        for rep in range(30):
            idx = rng.choice(Npool, M, replace=False)
            vals = rng.integers(0, V, M)
            G = rng.standard_normal((n, d))
            r = np.where(rng.random(d) < 0.5, -1.0, 1.0)
            if V == 2:
                Cv = np.stack([r, -r])
            else:
                H = np.array([[1.0]])
                while H.shape[0] < 16:
                    H = np.block([[H, H], [H, -H]])
                Cv = np.tile(H, (1, d // 16)) * r[None, :]
            S = ((X[idx] @ G) * Cv[vals]).sum(0)
            Sq = C.quantize(S, q, rng)
            sc = ((Aq[idx] @ G) * Sq) @ Cv.T
            accs.append(float((sc.argmax(1) == vals).mean()))
        print(f'H M={M} V={V} d={d} q={q}: theory {th:.3f} sim {np.mean(accs):.3f}')
    for (M, V, p, alpha) in [(256, 2, 32, 1e-2), (128, 2, 64, 1.0), (256, 16, 48, 0.1), (64, 2, 64, 1e-4)]:
        G = rng.standard_normal((n, p))
        sp = spec_linear(X, Aq, C2, G, dd.pair_lab)
        th, info = theory_L(sp, M, V, alpha, 4, 'V')
        th2, _ = theory_L(sp, M, V, alpha, 4, 'V', corrected=True)
        accs = []
        T0 = np.eye(V) - 1.0 / V
        for rep in range(30):
            idx = rng.choice(Npool, M, replace=False)
            vals = rng.integers(0, V, M)
            F = X[idx] @ G
            K = F @ F.T
            lam = alpha * np.trace(K) / M
            W = F.T @ np.linalg.solve(K + lam * np.eye(M), T0[vals])
            Wq = np.column_stack([C.quantize(W[:, j].astype(np.float32), 4, None) for j in range(V)])
            sc = (Aq[idx] @ G) @ Wq
            accs.append(float((sc.argmax(1) == vals).mean()))
        print(f'L M={M} V={V} p={p} alpha={alpha}: theory {th:.3f} corrected {th2:.3f} sim {np.mean(accs):.3f}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selftest':
        selftest()
    elif cmd == 'a':
        run_a(sys.argv[2])
    elif cmd == 'b':
        run_b()
    else:
        raise SystemExit(__doc__)
