"""B32 (prereg/B32_*.md): the vector form against an equally engineered vector index, at a fixed number of bits.

Shared by b32_run.py (dev grid, test run) and b32_check.py (independent re-scoring). Everything here is
deterministic given its seeds.

A store holds M facts 'entity -> value'. The entities are PopQA subjects (b32_data/entities.json); each store draws
M of them from one split and gives each a value drawn uniformly at random from V symbols, so no value can be guessed
from a name. Every entity is written under its label; each alias of a stored entity is one query, whose gold answer
is that entity's value. Keys are continuous codes from a frozen encoder (b32_embed.py), the same for every memory.

Memory budget: B = bpf * M bits per store (bpf = bits per fact). What a memory may keep for free is the same for all:
the encoder, the value vocabulary (V symbols), and any fixed transform fitted on the training split (preprocessing,
PCA, PQ codebooks, ITQ rotation) or drawn from a seed (projections, value codes). Everything that depends on the
store's facts is counted.

Memories:
  VSA  one vector of d numbers at q bits each (d = floor((B - 32) / q)); the 32 bits hold the vector's scale.
       Writing a fact adds phi(key) * c(value) (elementwise; c(value) a random +-1 code: MAP binding); phi is a fixed
       feature map of the preprocessed key code into d dimensions. Reading: score(v) = sum_j Q(S)_j phi_j(query) c_j(v),
       answer = argmax over the V symbols.
  IDX  a per-fact vector index: each fact keeps a b-bit code of its key; the value is either kept per fact
       (ceil(log2 V) bits) or implied by grouping the facts by value (V * ceil(log2(M + 1)) bits for the group sizes),
       whichever is cheaper; b = floor((B - value bits) / M), rounded down to what the code family can use. Reading:
       the nearest stored code to the query (asymmetric: the query stays exact), answer = that fact's value.
  EXACT the same index with exact float32 codes (unconstrained bits): the upper bound of similarity look-up.
"""
import hashlib
import json
import math

import numpy as np

ENT = 'b32_data/entities.json'
STR = 'b32_data/emb_strings.json'


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ----------------------------------------------------------------------------------------------- data and stores

class Pool:
    def __init__(self, enc_tag):
        d = json.load(open(ENT))
        self.ents = d['entities']
        S = json.load(open(STR))
        self.idx = {s: i for i, s in enumerate(S)}
        self.strings = S
        self.X = np.load(f'b32_data/emb_{enc_tag}.npy').astype(np.float32)
        self.by_split = {s: [e for e in self.ents if e['split'] == s] for s in ('train', 'dev', 'test')}

    def train_matrix(self):
        rows = []
        for e in self.by_split['train']:
            rows.append(self.idx[e['name']])
            rows += [self.idx[a] for a in e['aliases']]
        return self.X[rows]


def make_store(pool, split, M, V, rep):
    """Draw M entities of the split (all of them if M exceeds the split) and a random value for each.
    Returns dict: keys (M x n raw codes), vals (M,), queries (Q x n), qgold (Q,), qent (Q,), qalias (Q list), names."""
    ents = pool.by_split[split]
    rng = np.random.default_rng(seed_of('store', split, M, V, rep))
    pick = rng.permutation(len(ents))[:min(M, len(ents))]
    pick = np.sort(pick)
    E = [ents[i] for i in pick]
    vals = rng.integers(0, V, size=len(E))
    keys = pool.X[[pool.idx[e['name']] for e in E]]
    qi, qg, qe, qa = [], [], [], []
    for j, e in enumerate(E):
        for a in e['aliases']:
            qi.append(pool.idx[a])
            qg.append(vals[j])
            qe.append(j)
            qa.append(a)
    return {'split': split, 'M': len(E), 'V': V, 'rep': rep, 'keys': keys, 'vals': vals,
            'queries': pool.X[qi], 'qgold': np.array(qg), 'qent': np.array(qe), 'qalias': qa,
            'names': [e['name'] for e in E], 'uris': [e['uri'] for e in E]}


# ----------------------------------------------------------------------------------------------- preprocessing

class Prep:
    """'none': L2-normalize. 'center': subtract the training mean, then normalize. 'whiten': subtract the mean,
    apply the symmetric (ZCA) whitening U diag(1 / sqrt(eigenvalue + eps * mean eigenvalue)) U^T of the training
    covariance, normalize. (ZCA rather than PCA whitening: the same inner products, but coordinates that do not depend
    on the eigensolver's sign or basis choices, so that two machines produce the same codes.)"""

    def __init__(self, kind, train, eps=0.1):
        self.kind = kind
        self.mu = train.mean(0) if kind != 'none' else None
        if kind == 'whiten':
            C = np.cov((train - self.mu).T)
            w, U = np.linalg.eigh(C)
            w = np.maximum(w, 0)
            self.W = ((U / np.sqrt(w + eps * w.mean())) @ U.T).astype(np.float32)

    def __call__(self, X):
        Y = X.astype(np.float32)
        if self.kind != 'none':
            Y = Y - self.mu
        if self.kind == 'whiten':
            Y = Y @ self.W
        return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------------------------- VSA store

# Max (1960) optimal uniform quantizer step for a unit Gaussian, by bits
MAX_STEP = {2: 0.9957, 3: 0.5860, 4: 0.3352, 8: 0.0237}


def quantize(S, q, rng):
    """q = 1: sign (zeros to +-1 at random); q >= 2: uniform mid-rise quantizer with Max's step on S / std(S)."""
    if q == 32:
        return S.astype(np.float32)
    if q == 1:
        out = np.sign(S).astype(np.float32)
        z = out == 0
        if z.any():
            out[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
        return out
    sd = float(S.std()) or 1.0
    step = MAX_STEP[q] * sd
    L = 2 ** q
    k = np.clip(np.floor(S / step), -L // 2, L // 2 - 1)
    return ((k + 0.5) * step).astype(np.float32)


def features(Y, fmap, seed, j0, dc):
    """phi for dimensions [j0, j0 + dc): 'lin' Gaussian projection; 'sgn' its sign; 'sgnK' the product of K
    independent signs (kernel = angular kernel ** K). Projections are drawn from seeds per (chunk, factor)."""
    n = Y.shape[1]
    K = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
    out = None
    for m in range(K):
        G = np.random.default_rng(seed_of('G', seed, m, j0, dc)).standard_normal((n, dc), dtype=np.float32)
        P = Y @ G
        f = P if fmap == 'lin' else np.where(P >= 0, 1.0, -1.0).astype(np.float32)
        out = f if out is None else out * f
    return out


def value_codes(V, seed, j0, dc):
    r = np.random.default_rng(seed_of('C', seed, j0, dc))
    return np.where(r.random((V, dc), dtype=np.float32) < 0.5, -1.0, 1.0).astype(np.float32)


# Entropy (bits) of the quantizer's output for a Gaussian input: what ideal entropy coding of the stored vector costs
# per number (q = 1: the sign, 1 bit). Computed from the Gaussian CDF at Max's steps (see MAX_STEP).
H_Q = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


CH = 8192  # the stored vector is processed (projected, quantized) in blocks of CH numbers


def vsa_dim(B, q, acct='raw'):
    """Dimensions of the VSA vector at B bits. Each block of CH numbers carries a 32-bit scale (charged for every q,
    although a sign needs none). acct 'raw': q bits per number; 'ideal': the entropy of the quantized numbers (ideal
    entropy coding of the stored vector)."""
    per = q if acct == 'raw' else H_Q[q]
    d = max(0, int((B - 32) // per))
    while d > 0 and d * per + 32 * math.ceil(d / CH) > B:
        d -= max(1, int((d * per + 32 * math.ceil(d / CH) - B) // per))
    return max(0, d)


def vsa_bits(d, q, acct='raw'):
    per = q if acct == 'raw' else H_Q[q]
    return d * per + 32 * math.ceil(d / CH) if d > 0 else 0


def vsa_answer(K, vals, Qy, V, d, q, fmap, seed, chunk=8192):
    """K: preprocessed keys (M x n), vals: their values, Qy: preprocessed queries. Returns (pred, scores)."""
    scores = np.zeros((Qy.shape[0], V), np.float64)
    if d <= 0:
        return np.zeros(Qy.shape[0], int), scores
    rngq = np.random.default_rng(seed_of('Q', seed))
    for j0 in range(0, d, chunk):
        dc = min(chunk, d - j0)
        C = value_codes(V, seed, j0, dc)
        S = (features(K, fmap, seed, j0, dc) * C[vals]).sum(0)
        Sq = quantize(S, q, rngq)
        scores += ((features(Qy, fmap, seed, j0, dc) * Sq) @ C.T).astype(np.float64)
    return scores.argmax(1), scores


# ----------------------------------------------------------------------------------------------- vector index

def value_bits(M, V):
    return min(V * math.ceil(math.log2(M + 1)), M * math.ceil(math.log2(max(V, 2))))


def key_bits(B, M, V, acct='raw'):
    """Bits per stored key code. 'raw': fixed-width codes, values kept per fact or implied by grouping (value_bits).
    'ideal': the facts stored as an unordered multiset of (code, value) pairs with ideal enumerative coding, which
    costs log2 C(2^b V + M - 1, M) bits; the largest b that fits in B (order carries no information, so a set of codes
    costs about M (b + log2 V - log2 M + log2 e) bits rather than M (b + log2 V))."""
    if acct == 'raw':
        return max(0, (B - value_bits(M, V)) // M)
    i = np.arange(M, dtype=np.float64)
    lf = math.lgamma(M + 1) / math.log(2)

    def cost(b):  # log2 C(N + M - 1, M), N = 2^b V, as sum_i log2(N + i) - log2 M!, stable for large N
        lN = b + math.log2(V)
        return M * lN + float(np.log1p(i / 2.0 ** lN).sum()) / math.log(2) - lf
    b = 0
    while b < 4096 and cost(b + 1) <= B:
        b += 1
    return max(b, key_bits(B, M, V, 'raw'))


DIVS = {}
OPQ_NITER = 4  # prereg/B32_compute_note1.md


def divisors(n):
    if n not in DIVS:
        DIVS[n] = [m for m in range(1, n + 1) if n % m == 0]
    return DIVS[n]


class IndexFamily:
    """Fitted once per (encoder, preprocessing) on the training split; codes keys at a given number of bits."""

    def __init__(self, family, train_y, seed=0, cache_tag=None):
        self.family = family
        self.cache_tag = cache_tag  # with B32_FIT_CACHE set, fitted PQ / OPQ indexes are kept on disk under this tag
        self.train = train_y
        self.n = train_y.shape[1]
        self.seed = seed
        self._cache = {}
        if family in ('itq', 'sq1', 'sq2', 'sq4'):
            mu = train_y.mean(0)
            C = np.cov((train_y - mu).T)
            w, U = np.linalg.eigh(C)
            o = np.argsort(-w)
            self.pca_mu, self.pca_U, self.pca_w = mu, U[:, o].astype(np.float32), w[o]

    def config(self, b):
        """The largest code this family can make with at most b bits; None if none fits. Returns (bits, params)."""
        n = self.n
        if self.family == 'lsh':
            return (b, b) if b >= 1 else None
        if self.family == 'itq':
            c = min(b, n)
            return (c, c) if c >= 1 else None
        if self.family.startswith('sq'):
            qb = int(self.family[2:])
            c = min(b // qb, n)
            return (c * qb, c) if c >= 1 else None
        if self.family in ('pq8', 'opq8', 'pq4'):
            nb = 4 if self.family == 'pq4' else 8
            ms = [m for m in divisors(n) if m * nb <= b]
            return (max(ms) * nb, max(ms)) if ms else None
        raise ValueError(self.family)

    def _disk(self, p):
        import os
        d = os.environ.get('B32_FIT_CACHE')
        if not d or not self.cache_tag or self.family not in ('pq4', 'pq8', 'opq8'):
            return None
        os.makedirs(d, exist_ok=True)
        return f'{d}/{self.cache_tag}_{self.family}_{p}.faiss'

    def _fit(self, p):
        """Fitted pieces depend only on the training split, the family and p (deterministic given the seeds). PQ / OPQ
        indexes may be kept on disk (compute note 2): a fit is then computed once and read back by later processes."""
        key = p
        if key in self._cache:
            return self._cache[key]
        import faiss
        import os
        faiss.omp_set_num_threads(int(os.environ.get('B32_FAISS_THREADS', '8')))
        fn = self._disk(p)
        if fn and os.path.exists(fn):
            self._cache[key] = faiss.read_index(fn)
            return self._cache[key]
        f = self.family
        if f == 'lsh':
            G = np.random.default_rng(seed_of('LSH', self.seed, p)).standard_normal((self.n, p), dtype=np.float32)
            obj = G
        elif f == 'itq':
            U = self.pca_U[:, :p]
            Z = (self.train - self.pca_mu) @ U
            R = np.linalg.qr(np.random.default_rng(seed_of('ITQ', self.seed, p)).standard_normal((p, p)))[0]
            for _ in range(50):
                Bc = np.sign(Z @ R)
                Uu, _, Vt = np.linalg.svd(Z.T @ Bc)
                R = Uu @ Vt
            obj = (U @ R).astype(np.float32)
        elif f.startswith('sq'):
            qb = int(f[2:])
            U = self.pca_U[:, :p]
            Z = (self.train - self.pca_mu) @ U
            sd = Z.std(0) + 1e-8
            obj = (U, sd, qb)
        else:
            nb = 4 if f == 'pq4' else 8
            if f == 'opq8':
                opq = faiss.OPQMatrix(self.n, p)
                opq.niter = OPQ_NITER  # compute note 1: FAISS's default of 50 iterations does not fit the CPU budget
                idx = faiss.IndexPreTransform(opq, faiss.IndexPQ(self.n, p, nb))
            else:
                idx = faiss.IndexPQ(self.n, p, nb)
            idx.train(np.ascontiguousarray(self.train))
            obj = idx
        if fn:
            faiss.write_index(obj, fn + '.tmp')
            os.replace(fn + '.tmp', fn)
        self._cache[key] = obj
        return obj

    def nearest(self, Ky, Qy, p):
        """Index of the nearest stored code for each query (asymmetric scoring)."""
        f = self.family
        obj = self._fit(p)
        if f == 'lsh':
            codes = np.where(Ky @ obj >= 0, 1.0, -1.0).astype(np.float32)
            return (Qy @ obj @ codes.T).argmax(1)
        if f == 'itq':
            codes = np.where((Ky - self.pca_mu) @ obj >= 0, 1.0, -1.0).astype(np.float32)
            return (((Qy - self.pca_mu) @ obj) @ codes.T).argmax(1)
        if f.startswith('sq'):
            U, sd, qb = obj
            Z = ((Ky - self.pca_mu) @ U) / sd
            if qb == 1:
                Zh = np.where(Z >= 0, 1.0, -1.0) * np.sqrt(2 / np.pi)
            else:
                step = MAX_STEP[qb]
                L = 2 ** qb
                Zh = (np.clip(np.floor(Z / step), -L // 2, L // 2 - 1) + 0.5) * step
            Kh = (Zh * sd) @ U.T + self.pca_mu
            return (Qy @ Kh.T).argmax(1)
        # faiss PQ / OPQ (ADC, L2 on unit vectors)
        import faiss
        obj.reset()
        obj.add(np.ascontiguousarray(Ky))
        _, I = obj.search(np.ascontiguousarray(Qy), 1)
        obj.reset()
        return I[:, 0]


def exact_nearest(Ky, Qy):
    return (Qy @ Ky.T).argmax(1)


# ----------------------------------------------------------------------------------------------- one memory, one store

def subset_mask(st, f):
    """The facts a memory keeps when it stores only a share f of them (the same facts for every memory; the order is
    a seeded permutation of the store, independent of which entities are queried)."""
    M = st['M']
    order = np.random.default_rng(seed_of('subset', st['split'], M, st['V'], st['rep'])).permutation(M)
    keep = np.zeros(M, bool)
    keep[order[:max(1, int(math.floor(f * M)))]] = True
    return keep


def answer(st, arm, cfg, bpf, preps, fams, acct='raw'):
    """Predictions of one memory configuration on one store at bpf bits per fact.
    arm 'VSA': cfg = (prep, fmap, q, f); arm 'IDX': cfg = (prep, family, f); arm 'EXACT': cfg = (prep,).
    preps: {name: Prep}; fams: {(prep, family): IndexFamily}. Returns (pred values, info dict)."""
    M, V = st['M'], st['V']
    B = bpf * M
    P = preps[cfg[0]]
    Q = P(st['queries'])
    if arm == 'EXACT':
        K = P(st['keys'])
        return st['vals'][exact_nearest(K, Q)], {'bits': None}
    keep = subset_mask(st, cfg[-1])
    K, vals = P(st['keys'][keep]), st['vals'][keep]
    Ms = int(keep.sum())
    if arm == 'VSA':
        _, fmap, q, _ = cfg
        d = vsa_dim(B, q, acct)
        pred, _ = vsa_answer(K, vals, Q, V, d, q, fmap, seed_of('vsa', st['split'], M, V, st['rep'], cfg, bpf))
        return pred, {'bits': d * q + 32, 'd': d, 'stored': Ms}
    if arm == 'IDX':
        _, family, _ = cfg
        kb = key_bits(B, Ms, V, acct)
        c = fams[cfg[0], family].config(kb)
        if c is None:  # no key bits: answer the most frequent stored value
            mode = int(np.bincount(vals, minlength=V).argmax())
            return np.full(len(Q), mode), {'keybits': 0, 'stored': Ms}
        nn = fams[cfg[0], family].nearest(K, Q, c[1])
        return vals[nn], {'keybits': c[0], 'stored': Ms}
    raise ValueError(arm)


# ----------------------------------------------------------------------------------------------- string look-ups

def norm(s):
    import re
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def string_answers(st):
    """Unconstrained-memory string look-ups (context for B26): DICT, the exact table on normalized labels;
    TRI, the nearest label by the cosine of character-trigram counts (B26's N with its chosen n-gram set {3}, over
    the normalized string padded with one space at each end); WR, rapidfuzz WRatio with its default processor."""
    from collections import Counter
    from rapidfuzz import fuzz, process, utils
    names = st['names']
    nn = [norm(x) for x in names]
    d = {}
    for j, x in enumerate(nn):
        d.setdefault(x, j)

    def tri(s):
        s = ' ' + norm(s) + ' '
        return Counter(s[i:i + 3] for i in range(max(1, len(s) - 2)))

    vocab = {}
    rows, cols, data = [], [], []
    for j, x in enumerate(names):
        for g, c in tri(x).items():
            rows.append(j)
            cols.append(vocab.setdefault(g, len(vocab)))
            data.append(c)
    from scipy import sparse
    Kt = sparse.csr_matrix((data, (rows, cols)), shape=(len(names), len(vocab) + 1), dtype=np.float64)
    Kn = np.sqrt(np.asarray(Kt.multiply(Kt).sum(1)).ravel()) + 1e-12
    out = {'DICT': [], 'TRI': [], 'WR': []}
    for a in st['qalias']:
        j = d.get(norm(a))
        out['DICT'].append(-1 if j is None else int(st['vals'][j]))
        t = tri(a)
        v = np.zeros(len(vocab) + 1)
        for g, c in t.items():
            v[vocab.get(g, len(vocab))] += c
        sims = (Kt @ v) / (Kn * (np.linalg.norm(v) + 1e-12))
        out['TRI'].append(int(st['vals'][int(np.argmax(sims))]))
        best = process.extractOne(a, names, scorer=fuzz.WRatio, processor=utils.default_process)
        out['WR'].append(int(st['vals'][best[2]]))
    return {k: np.array(v) for k, v in out.items()}
