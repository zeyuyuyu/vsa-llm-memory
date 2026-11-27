"""B49 (prereg/B49_panel_mechanism_competitors_prereg.md): the two new equal-bit competitors and the code transforms of
the mechanism test.

LAM  least-squares (pseudo-inverse / ridge) linear associative memory (Kohonen 1972, 'optimal linear associative
     memory'; Personnaz et al. 1986 projection rule). Keys are mapped to p features f(x) (a fixed Gaussian projection
     'rp' of the preprocessed code, or its first p principal components 'pca', fitted on the training split; both free).
     The store keeps one matrix W (p x c): the ridge solution of F W = T over the store's facts (F: M x p features of the
     labels; T: targets), W = F^T (F F^T + lam I)^-1 T, lam = alpha * mean diag(F F^T) (alpha -> 0: the pseudo-inverse).
     Targets: 'onehot' (centred one-hot rows e_v - 1/V; the V solution columns sum to zero, so only the columns of
     values 1..V-1 are stored, c = V - 1, and value 0's score is minus the sum of the others) or, for V >= 128,
     'codeR' (R random +-1 value codes, c = R; read by correlation with every value's code). W is quantized per column and per block of 8,192 rows with Max's uniform
     quantizer (q = 1: sign), each (column, block) with a 32-bit scale. Read: score(v) = f(q)^T W_hat t(v), argmax.
     Bits: ideal c * p * H_q + 32 c ceil(p / 8192) (entropy of the quantizer output, as the VSA); real: q instead of H_q.
     p = the largest number of features that fits (rp: at most 65,536; pca: at most n).
LH   learned hashing: a linear map learned on the training split's alias-label pairs (shrinkage LDA: the generalized
     eigenvectors of the training codes' covariance against the covariance of alias - label differences + s * mean
     eigenvalue * I; the first r directions; Strecha et al. 2012 'LDAHash'), applied after centring, rows unit-norm.
     (s, r) chosen by B32's dev entity-retrieval criterion (no memory). The learned codes are then hashed by the
     existing machinery: the per-fact index's binary / PQ codes (LSH, ITQ, PCA-sign, PQ4, PQ8: 'LHI'), the LSH bucket
     sketch ('LHC') and the static function ('LHS') with SimHash / PQ signatures of the learned codes; ITQ on the learned
     projection is supervised binary hashing in the sense of CCA-ITQ (Gong et al. 2013).
"""
import math

import numpy as np

import b32_common as C

CH = 8192
P_MAX = 65536


# ----------------------------------------------------------------------------------------------- LAM

def lam_cols(V, target):
    return V - 1 if target == 'onehot' else int(target[4:])


def lam_bits(p, c, q, acct):
    per = C.H_Q[q] if acct == 'ideal' else q
    return c * p * per + 32 * c * math.ceil(p / CH) if p > 0 else 0


def lam_p(B, c, q, acct, pmax):
    per = C.H_Q[q] if acct == 'ideal' else q
    p = int(max(0, (B - 32 * c) // (c * per)))
    while p > 0 and lam_bits(p, c, q, acct) > B:
        p -= max(1, int((lam_bits(p, c, q, acct) - B) // (c * per)))
    return max(0, min(p, pmax))


def value_targets(V, target, seed):
    """(T, D): T = V x c stored-column targets (centred one-hot, columns of values 1..V-1; or random +-1 codes, seeded
    per (V, R) only); D = c x V decoding matrix from the c stored scores to the V value scores."""
    if target == 'onehot':
        T = np.eye(V)[:, 1:] - 1.0 / V
        D = np.hstack([-np.ones((V - 1, 1)), np.eye(V - 1)])
        return T, D
    R = int(target[4:])
    r = np.random.default_rng(C.seed_of('LAMcode', V, R))
    T = np.where(r.random((V, R)) < 0.5, -1.0, 1.0)
    return T, T.T


class LamFeat:
    """Fixed feature maps of one (encoder, prep): 'rp' Gaussian projection blocks drawn from seeds
    ('LAMG', enc, prep, j0) (n x 8192 each; a p-feature map is the prefix of the blocks), 'pca' the principal directions
    of the preprocessed training codes (descending variance)."""

    def __init__(self, enc, prep, train_y):
        self.enc, self.prep = enc, prep
        self.n = train_y.shape[1]
        mu = train_y.mean(0)
        Cv = np.cov((train_y - mu).T)
        w, U = np.linalg.eigh(Cv)
        o = np.argsort(-w)
        self.mu, self.U = mu.astype(np.float32), U[:, o].astype(np.float32)
        self.cache = {}

    def block(self, j0):
        if j0 not in self.cache:
            self.cache[j0] = np.random.default_rng(C.seed_of('LAMG', self.enc, self.prep, j0)).standard_normal(
                (self.n, CH), dtype=np.float32) / np.float32(math.sqrt(self.n))
        return self.cache[j0]

    def pmax(self, feat):
        return P_MAX if feat == 'rp' else self.n

    def feats(self, Y, feat, j0, w):
        if feat == 'rp':
            return Y @ self.block(j0)[:, :w]
        return (Y - self.mu) @ self.U[:, j0:j0 + w]


def quant_cols(Wb, q, rng):
    """Max quantizer per column of one block (q = 1: sign with the column's mean |w| as scale)."""
    out = np.empty_like(Wb)
    for c in range(Wb.shape[1]):
        x = Wb[:, c]
        if q == 1:
            s = np.sign(x)
            z = s == 0
            if z.any():
                s[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
            out[:, c] = s * (np.abs(x).mean() or 1.0)
        else:
            out[:, c] = C.quantize(x.astype(np.float32), q, None)
    return out


def run_lam(st, Yk, Yq, lf, cfgs, bpfs, accts, seed):
    """cfgs: list of (feat, alpha, q, target) for one prep; Yk, Yq: preprocessed keys / queries. Returns
    {(cfg, bpf, acct): (pred, bits)}. Every configuration sharing (feat, p, target, alpha) shares one ridge solve."""
    M, V = st['M'], st['V']
    vals = st['vals']
    out = {}
    need = {}  # (feat, target) -> set of p
    plan = []
    for (feat, alpha, q, target) in cfgs:
        c = lam_cols(V, target)
        for bpf in bpfs:
            for acct in accts:
                p = lam_p(bpf * M, c, q, acct, lf.pmax(feat))
                plan.append(((feat, alpha, q, target), bpf, acct, p, c))
                need.setdefault((feat, target), set()).add(p)
    default = int(np.bincount(vals, minlength=V).argmax())
    for (feat, target), ps in need.items():
        ps = sorted(x for x in ps if x > 0)
        T, D = value_targets(V, target, seed)
        Y = T[vals]  # M x c
        if not ps:
            continue
        pm = ps[-1]
        # pass 1: Gram matrices of the feature prefixes
        K = np.zeros((M, M), np.float64)
        Ks = {}
        blocks = list(range(0, pm, CH))
        for j0 in blocks:
            w = min(CH, pm - j0)
            Fk = lf.feats(Yk, feat, j0, w).astype(np.float64)
            a0 = 0
            for p in ps:  # accumulate column segments between consecutive breakpoints
                if j0 < p <= j0 + w:
                    K += Fk[:, a0:p - j0] @ Fk[:, a0:p - j0].T
                    a0 = p - j0
                    Ks[p] = K.copy()
            if a0 < w:
                K += Fk[:, a0:w] @ Fk[:, a0:w].T
        # ridge solves: A = (K_p + lam I)^-1 Y
        sols = {}
        for (cfg, bpf, acct, p, c) in plan:
            if cfg[0] != feat or cfg[3] != target or p <= 0:
                continue
            alpha = cfg[1]
            if (p, alpha) not in sols:
                Kp = Ks[p]
                lam = alpha * float(np.trace(Kp)) / M
                sols[p, alpha] = np.linalg.solve(Kp + lam * np.eye(M), Y)
        # pass 2: W blocks, quantized, and scores
        scores = {}
        for j0 in blocks:
            w = min(CH, pm - j0)
            Fk = lf.feats(Yk, feat, j0, w).astype(np.float64)
            Fq = lf.feats(Yq, feat, j0, w).astype(np.float64)
            for (cfg, bpf, acct, p, c) in plan:
                if cfg[0] != feat or cfg[3] != target or p <= j0:
                    continue
                ww = min(w, p - j0)
                key = (cfg, bpf, acct)
                Wb = Fk[:, :ww].T @ sols[p, cfg[1]]
                Wq = quant_cols(Wb, cfg[2], np.random.default_rng(C.seed_of('LAMQ', seed, cfg, bpf, acct, j0)))
                s = Fq[:, :ww] @ Wq  # Q x c
                scores[key] = scores.get(key, 0.0) + s
        for (cfg, bpf, acct, p, c) in plan:
            if cfg[0] != feat or cfg[3] != target:
                continue
            key = (cfg, bpf, acct)
            if p <= 0 or key not in scores:
                out[key] = (np.full(Yq.shape[0], default), 0.0)
                continue
            S = scores[key] @ D  # Q x V
            out[key] = (S.argmax(1), float(lam_bits(p, c, cfg[2], acct)))
    for (cfg, bpf, acct, p, c) in plan:
        if (cfg, bpf, acct) not in out:
            out[cfg, bpf, acct] = (np.full(Yq.shape[0], default), 0.0)
    return out


# ----------------------------------------------------------------------------------------------- LH (learned map)

class LearnedPrep:
    """Shrinkage-LDA map fitted on the training split (labels and their aliases): codes are centred on the training
    mean (after the raw codes' row normalisation), projected on the first r generalized eigenvectors of
    (S_t, S_w + s * mean eig(S_w) I), rows unit-norm. S_t: covariance of all training codes; S_w: covariance of
    (alias - label) differences of training entities (one per alias)."""

    def __init__(self, X, pool, s, r):
        tr = pool.by_split['train']
        rows = [pool.idx[e['name']] for e in tr] + [pool.idx[a] for e in tr for a in e['aliases']]
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        T = Xn[rows].astype(np.float64)
        self.mu = T.mean(0)
        St = np.cov((T - self.mu).T)
        D = np.array([Xn[pool.idx[a]] - Xn[pool.idx[e['name']]] for e in tr for a in e['aliases']], np.float64)
        Sw = D.T @ D / len(D)
        ev = np.linalg.eigvalsh(Sw)
        Sw = Sw + s * float(ev.mean()) * np.eye(len(Sw))
        # generalized symmetric eigenproblem St v = l Sw v via Cholesky of Sw
        Lc = np.linalg.cholesky(Sw)
        Li = np.linalg.inv(Lc)
        Mx = Li @ St @ Li.T
        w, U = np.linalg.eigh((Mx + Mx.T) / 2)
        o = np.argsort(-w)[:r]
        self.A = (Li.T @ U[:, o]).astype(np.float32)  # n x r, columns = generalized eigenvectors
        self.eig = w[o]
        self.mu = self.mu.astype(np.float32)
        self.kind = f'lda_s{s}_r{r}'

    def __call__(self, X):
        Xn = X.astype(np.float32)
        Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
        Y = (Xn - self.mu) @ self.A
        return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


LH_GRID = {'s': [0.01, 0.1, 1.0, 10.0], 'r': [32, 64, 128, 256]}


def criterion(Y, pool):
    """B32's dev criterion on already-mapped codes: every dev label stored, every dev alias a query, exact cosine
    (the learned map is already centred)."""
    dv = pool.by_split['dev']
    K = Y[[pool.idx[e['name']] for e in dv]]
    qs = [(pool.idx[a], j) for j, e in enumerate(dv) for a in e['aliases']]
    Q = Y[[q for q, _ in qs]]
    g = np.array([j for _, j in qs])
    return float(((Q @ K.T).argmax(1) == g).mean())


# ----------------------------------------------------------------------------------------------- mechanism transforms

def spectrum_power(X, pool, gamma, eps=1e-6):
    """Centre (training mean) and rescale every principal direction of the training covariance by l^((gamma-1)/2):
    the centred codes' spectrum becomes l^gamma (gamma < 1 flattens, gamma > 1 steepens). Rows unit-norm."""
    Tr = pool.train_matrix().astype(np.float64)
    Tr /= np.linalg.norm(Tr, axis=1, keepdims=True)
    mu = Tr.mean(0)
    w, U = np.linalg.eigh(np.cov((Tr - mu).T))
    w = np.maximum(w, eps * w.max())
    S = (U * w ** ((gamma - 1) / 2)) @ U.T
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = (Xn - mu) @ S
    return (Y / np.linalg.norm(Y, axis=1, keepdims=True)).astype(np.float32)


def pr_of(Y, pool):
    rows = [pool.idx[e['name']] for e in pool.by_split['train']] + \
           [pool.idx[a] for e in pool.by_split['train'] for a in e['aliases']]
    T = Y[rows].astype(np.float64)
    T = T - T.mean(0)
    w = np.linalg.eigvalsh(np.cov(T.T))
    w = np.maximum(w, 0)
    return float(w.sum() ** 2 / (w ** 2).sum())
