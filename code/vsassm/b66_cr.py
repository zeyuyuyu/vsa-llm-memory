# b66_cr.py -- B66 part (A) (prereg/B66_reconciliation_prereg.md): the clean-room re-implementation cr.py, copied
# verbatim from the scratchpad (cleanroom4_keyfree/cr.py, md5 1c1b9c1ea634881c962be95361705465; written by a separate
# agent from the manuscript / supplementary text only, without reading pipeline code). Changes, all marked "B66":
#   (1) the grid: a second SDM-with-training-locations family 'sdmt' on B64's exact configuration space (prep x location
#       set LA / L / A x a in 1, 2, 4, 8, 16, 32, 64 x counters onehot / bits x counter type cnt / clip3 / tern x share
#       1, 1/2 = 504), evaluated by cr.py's own SDM code (sdm_eval; the location-set columns and the counter transform
#       are the only additions); the original grid (family_grid) is unchanged and comes first, so every original
#       configuration sees the same random stream;
#   (2) harness: ROOT / output directory from the environment; the run is split into a dev phase (all dev accuracies and
#       the dev choices written to disk) and a test phase (reads the committed choices), with the same seeds, so the
#       predictions equal a one-pass run; extra arms (SDM with Gaussian addresses only, SDMt on the clean room's own
#       grid, SDMt on B64's grid, best of seven per grid) are evaluated after the original six with their own rng, and
#       per-query correctness is saved with the test-store index of every query.
# Clean-room re-implementation of the equal-bit store and six unsupervised key-free memories.
# Written only from the manuscript/supplementary text; no pipeline code was read.
import os, sys, json, math, time
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from math import lgamma, log
from scipy import sparse
np.seterr(over='ignore')
import warnings; warnings.filterwarnings('ignore')

ROOT = os.environ.get('B66_CR_ROOT', '/path/to/vsassm')  # B66: from the environment
WD = os.environ.get('B66_CR_OUT', os.path.join(ROOT, 'b66_out', 'cr'))  # B66: output directory
HQ = {1: 1.0, 2: 1.904, 4: 3.602}
MAXSTEP = {4: 0.99569, 16: 0.33520}
SHARES = [1.0, 0.5, 0.25, 0.125]
PREPS = ['centre', 'whiten']
NHYP = 512           # hyperplanes for signatures: table t uses columns 16t..16t+k-1 (<=32 tables, k<=16)
NLSH = 256
NPCA = 256
G_D = 16320          # largest store dimension in the region (M=1024, b=16, 1 bit)
SDM_NMAX = 16384
FLY_MMAX = 32768
FLY_FANIN = 16

G = {}  # shared globals filled in parent before fork

def h64(x):
    x = np.asarray(x, dtype=np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

def hmix(*parts):
    acc = np.uint64(0x9E3779B97F4A7C15)
    out = None
    for p in parts:
        p = np.asarray(p).astype(np.uint64)
        out = h64((acc if out is None else out) ^ (p + np.uint64(0x632BE59BD9B4E019)))
        acc = out
    return out

def entropy_hist(counts):
    c = np.asarray(counts, dtype=np.float64); c = c[c > 0]
    p = c / c.sum(); return float(-(p * np.log2(p)).sum())

def H2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def log2_binom(U, n):
    if n == 0: return 0.0
    if U < n: return float('inf')
    return (lgamma(U + 1) - lgamma(n + 1) - lgamma(U - n + 1)) / log(2)

def argmax_rt(S, rng):
    # argmax with random tie-breaking (B36 amendment: ties broken at random)
    S = S + rng.random(S.shape) * 1e-9 * (np.abs(S).max() + 1e-12) if S.dtype.kind == 'f' else S + rng.random(S.shape) * 1e-3
    return S.argmax(1)

# ------------------------------------------------------------------ data
def load_data(enc):
    ents = json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']
    strings = json.load(open(f'{ROOT}/b32_data/emb_strings.json'))
    idx = {s: i for i, s in enumerate(strings)}
    path = {'qwen8': f'{ROOT}/b32_data/emb_qwen8_mean_L9_f16.npy',
            'llama': f'{ROOT}/b49_out/codes/emb_llama31i_mean_L24_f16.npy'}[enc]
    X = np.load(path).astype(np.float32)
    split = {'train': [], 'dev': [], 'test': []}
    for e in ents:
        split[e['split']].append((idx[e['name']], [idx[a] for a in e['aliases']]))
    return X, split

def setup(enc, seed=12345):
    t0 = time.time()
    X, split = load_data(enc)
    tr = [l for l, _ in split['train']] + [a for _, al in split['train'] for a in al]
    T = X[tr].astype(np.float64)
    mu = T.mean(0)
    C = np.cov((T - mu).T)
    w, U = np.linalg.eigh(C); w = np.maximum(w, 0)
    Wz = ((U * (1.0 / np.sqrt(w + 0.1 * w.mean()))) @ U.T).astype(np.float32)
    Xc = X - mu.astype(np.float32)
    del X
    Xw = Xc @ Wz
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    Xw /= np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12
    G['split'] = split; G['train_rows'] = np.array(tr)
    G['n_lab'] = len(split['train'])  # B66: train_rows = the labels (entity order), then the aliases
    rng = np.random.default_rng(seed)
    n = Xc.shape[1]; G['n'] = n
    G['Xp'] = {'centre': Xc, 'whiten': Xw}
    hyp = rng.standard_normal((n, NHYP)).astype(np.float32)
    lsh = rng.standard_normal((n, NLSH)).astype(np.float32)
    G['Zhyp'] = {}; G['Zlsh'] = {}; G['Zpca'] = {}; G['pca_mu'] = {}
    for p, Xp in G['Xp'].items():
        G['Zhyp'][p] = Xp @ hyp
        G['Zlsh'][p] = Xp @ lsh
        Tp = Xp[tr].astype(np.float64); m = Tp.mean(0)
        wp, Up = np.linalg.eigh(np.cov((Tp - m).T))
        Up = Up[:, ::-1][:, :NPCA].astype(np.float32)
        G['Zpca'][p] = (Xp - m.astype(np.float32)) @ Up
    # VSA store projections (three independent Gaussian maps)
    G['Gm'] = [rng.standard_normal((n, G_D)).astype(np.float32) for _ in range(3)]
    # SDM random addresses (unit Gaussian) in code space
    A = rng.standard_normal((n, SDM_NMAX)).astype(np.float32); A /= np.linalg.norm(A, axis=0, keepdims=True)
    G['sdm_gauss'] = A
    # fly projection: sparse binary, each of FLY_MMAX units sums FLY_FANIN random input coordinates
    rows = np.concatenate([rng.choice(n, FLY_FANIN, replace=False) for _ in range(FLY_MMAX)])
    cols = np.repeat(np.arange(FLY_MMAX), FLY_FANIN)
    G['fly'] = sparse.csc_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, FLY_MMAX))
    print(f'setup {enc} done {time.time()-t0:.0f}s', flush=True)

# ------------------------------------------------------------------ stores
def make_stores(which, M, V, nst, seed):
    rng = np.random.default_rng(seed)
    L = G['split'][which]
    out = []
    for s in range(nst):
        ids = rng.choice(len(L), M, replace=False)
        vals = rng.integers(0, V, M)
        keys = np.array([L[i][0] for i in ids])
        qs, qv, qe, qp = [], [], [], []
        for j, i in enumerate(ids):
            for a in L[i][1]:
                qs.append(a); qv.append(vals[j]); qe.append(i); qp.append(j)
        out.append(dict(keys=keys, vals=vals, ents=ids, q=np.array(qs), qv=np.array(qv), qe=np.array(qe), qpos=np.array(qp)))
    return out

# ------------------------------------------------------------------ VSA store
def dfor(B, q):
    d = int(B / HQ[q])
    while d * HQ[q] + 32 * math.ceil(d / 8192) > B: d -= 1
    return d

def quant(S, q, rng):
    out = np.empty_like(S)
    for b0 in range(0, len(S), 8192):
        blk = S[b0:b0 + 8192]
        if q == 1:
            sg = np.sign(blk); z = sg == 0
            if z.any(): sg[z] = rng.choice([-1.0, 1.0], z.sum())
            out[b0:b0 + 8192] = sg * (np.abs(blk).mean() + 1e-12)
        else:
            N = 2 ** q; D = MAXSTEP[N] * (blk.std() + 1e-12)
            i = np.clip(np.floor(blk / D), -N // 2, N // 2 - 1)
            out[b0:b0 + 8192] = (i + 0.5) * D
    return out

def value_codes(V, d, design, rng):
    r = rng.choice([-1.0, 1.0], d).astype(np.float32)
    if design == 'first':
        return rng.choice([-1.0, 1.0], (V, d)).astype(np.float32)
    if V == 2:
        return np.stack([r, -r])
    if V == 16:
        H = np.array([[1.0]])
        while H.shape[0] < 16: H = np.block([[H, H], [H, -H]])
        return (r[None, :] * H[:, np.arange(d) % 16]).astype(np.float32)
    return rng.choice([-1.0, 1.0], (V, d)).astype(np.float32)

def store_eval(st, prep, B, V, codes, seed):
    """Returns dict cfg -> predictions over st's queries for both designs."""
    dmax = dfor(B, 1)
    Xp = G['Xp'][prep]
    rows = np.concatenate([st['keys'], st['q']])
    P = [Xp[rows] @ G['Gm'][i][:, :dmax] for i in range(3)]
    nk = len(st['keys'])
    feats = {'lin': P[0], 'sign': np.sign(P[0]), 'prod': np.sign(P[0]) * np.sign(P[1]) * np.sign(P[2])}
    del P
    res = {}
    rng = np.random.default_rng(seed)
    for mp, F in feats.items():
        Fk, Fq = F[:nk], F[nk:]
        for design in ('first', 'final'):
            Cv = codes[design]
            for sh in SHARES:
                ns = max(1, int(round(sh * nk)))
                S = (Cv[st['vals'][:ns]] * Fk[:ns]).sum(0)
                for q in (1, 2, 4):
                    d = dfor(B, q)
                    Sq = quant(S[:d].copy(), q, rng)
                    sc = (Fq[:, :d] * Sq[None, :]) @ Cv[:, :d].T
                    res[(design, prep, mp, q, sh)] = sc.argmax(1)   # first maximum
    return res

# ------------------------------------------------------------------ signatures
def hyp_sigs(prep, rows, T, k, P):
    """Per table: base signature int (len rows) and probe signatures (rows, P+1)."""
    Z = G['Zhyp'][prep][rows]
    out = []
    for t in range(T):
        z = Z[:, 16 * t:16 * t + k]
        bits = (z > 0).astype(np.int64)
        base = (bits << np.arange(k)).sum(1)
        pr = [base]
        if P > 0:
            order = np.argsort(np.abs(z), axis=1)[:, :P]
            for j in range(min(P, k)):
                pr.append(base ^ (np.int64(1) << order[:, j]))
        out.append(np.stack(pr, 1))
    return out  # list of (rows, P+1)

# ------------------------------------------------------------------ family 1: per-fact index
_itq_cache = {}; _pq_cache = {}
def itq_rot(prep, c):
    key = (prep, c)
    if key not in _itq_cache:
        Vt = G['Zpca'][prep][G['train_rows'], :c].astype(np.float64)
        rng = np.random.default_rng(c)
        R = np.linalg.qr(rng.standard_normal((c, c)))[0]
        for _ in range(50):
            Bm = np.sign(Vt @ R); Bm[Bm == 0] = 1
            U, s, Wt = np.linalg.svd(Vt.T @ Bm)
            R = U @ Wt
        _itq_cache[key] = R.astype(np.float32)
    return _itq_cache[key]

def kmeans(X, K, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    Cn = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ Cn.T + (Cn ** 2).sum(1)[None]
        a = d.argmin(1)
        cnt = np.bincount(a, minlength=K)
        sm = np.zeros_like(Cn); np.add.at(sm, a, X)
        nz = cnt > 0
        Cn[nz] = sm[nz] / cnt[nz, None]
    return Cn

PQ_DIM = 128
def pq_books(prep, m, K):
    key = (prep, m, K)
    if key not in _pq_cache:
        Xt = G['Zpca'][prep][G['train_rows'], :PQ_DIM]
        subs = np.array_split(np.arange(PQ_DIM), m)
        _pq_cache[key] = (subs, [kmeans(Xt[:, s], K, seed=j) for j, s in enumerate(subs)])
    return _pq_cache[key]

def index_bits(U_log2, V, n):
    return log2_binom_pow(U_log2 + math.log2(V), n)

def log2_binom_pow(lu, n):
    # information-theoretic minimum for an unordered multiset of n entries from a universe of 2^lu:
    # log2 C(U + n - 1, n)
    # = sum_{i<n} log2(U + i) - log2(n!), computed stably as n*lu + sum log2(1 + i/U) - log2(n!)
    i = np.arange(n, dtype=np.float64)
    corr = np.log1p(i * 2.0 ** (-lu)).sum() / log(2) if lu < 1000 else 0.0
    return float(n * lu + corr - lgamma(n + 1) / log(2))

def index_eval(st, prep, B, V, fam, sh, rng):
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    keys = st['keys'][:ns]; vals = st['vals'][:ns]; q = st['q']
    if fam.startswith('pq'):
        K = int(fam[2:].split('_')[0])
        m = 0
        for mm in range(1, PQ_DIM + 1):
            if log2_binom_pow(mm * math.log2(K) + math.log2(V), ns) <= B: m = mm
        if m == 0: return None
        subs, books = pq_books(prep, m, K)
        Zk = G['Zpca'][prep][keys, :PQ_DIM]; Zq = G['Zpca'][prep][q, :PQ_DIM]
        dist = np.zeros((len(q), ns), np.float32)
        for s, Cb in zip(subs, books):
            dk = ((Zk[:, s][:, None, :] - Cb[None]) ** 2).sum(2).argmin(1)
            dq = ((Zq[:, s][:, None, :] - Cb[None]) ** 2).sum(2)  # ADC table (Q, K)
            dist += dq[:, dk]
        pos = argmax_rt(-dist, rng)
        return vals[pos]
    kind, rd = fam.split('_')
    c = 0
    for cc in range(1, (NLSH if kind == 'lsh' else NPCA) + 1):
        if log2_binom_pow(cc + math.log2(V), ns) <= B: c = cc
    if c == 0: return None
    if kind == 'lsh':
        Zk = G['Zlsh'][prep][keys, :c]; Zq = G['Zlsh'][prep][q, :c]
    elif kind == 'pca':
        Zk = G['Zpca'][prep][keys, :c]; Zq = G['Zpca'][prep][q, :c]
    else:
        R = itq_rot(prep, c)
        Zk = G['Zpca'][prep][keys, :c] @ R; Zq = G['Zpca'][prep][q, :c] @ R
    Bk = np.where(Zk > 0, 1.0, -1.0).astype(np.float32)
    if rd == 'sym':
        Bq = np.where(Zq > 0, 1.0, -1.0).astype(np.float32)
        sc = Bq @ Bk.T            # = c - 2*Hamming
    else:
        sc = Zq @ Bk.T            # asymmetric distance
    pos = argmax_rt(sc.astype(np.float64), rng)
    return vals[pos]

# ------------------------------------------------------------------ family 2: static function (ribbon retrieval)
def static_eval(st, prep, B, V, k, f, P, sh, rng):
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    lv = math.log2(V)
    T = int(B // (ns * (lv + f)))
    T = min(T, 32)
    if T < 1: return None
    ks = hyp_sigs(prep, st['keys'][:ns], T, k, 0)
    qs = hyp_sigs(prep, st['q'], T, k, P)
    vals = st['vals'][:ns]
    Q = len(st['q'])
    votes = np.zeros((Q, V))
    fmask = np.uint64((1 << f) - 1) if f > 0 else np.uint64(0)
    for t in range(T):
        s = ks[t][:, 0]
        us, first = np.unique(s, return_index=True)   # first stored key wins a collision
        uv = vals[first]
        pr = qs[t]
        pos = np.searchsorted(us, pr); pos = np.clip(pos, 0, len(us) - 1)
        member = us[pos] == pr
        fp_true = hmix(t, pr, 7) & fmask
        garb = hmix(t, pr, 11)
        out_v = np.where(member, uv[pos], (garb % np.uint64(V)).astype(np.int64))
        out_fp = np.where(member, fp_true, (garb >> np.uint64(20)) & fmask)
        acc = out_fp == fp_true
        for j in range(pr.shape[1]):
            np.add.at(votes, (np.nonzero(acc[:, j])[0], out_v[acc[:, j], j]), 1.0)
    return argmax_rt(votes, rng)

# ------------------------------------------------------------------ family 3: count-min sketch
def cms_eval(st, prep, B, V, k, hold, rd, P, sh, rng):
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    w = 1 << k
    ks = hyp_sigs(prep, st['keys'][:ns], 32, k, 0)
    vals = st['vals'][:ns]
    tabs = []; T = 0
    hist = np.zeros(ns + 2) if hold == 'counts' else np.zeros(V + 1)
    for t in range(32):
        cnt = np.bincount(ks[t][:, 0] * V + vals, minlength=w * V).reshape(w, V)
        if hold == 'counts':
            nh = hist + np.bincount(cnt.ravel(), minlength=ns + 2)[:ns + 2]
            cost = (t + 1) * w * V * entropy_hist(nh)
        else:
            tot = cnt.sum(1); mv = np.where(tot > 0, cnt.argmax(1), V)  # V = empty symbol
            nh = hist + np.bincount(mv, minlength=V + 1)
            cost = (t + 1) * w * entropy_hist(nh)
        if cost > B: break
        hist = nh; T = t + 1
        tabs.append(cnt if hold == 'counts' else mv)
    if T == 0: return None
    qs = hyp_sigs(prep, st['q'], T, k, P)
    Q = len(st['q'])
    if hold == 'counts':
        per = np.stack([tabs[t][qs[t]].sum(1) for t in range(T)], 0)  # (T, Q, V)
        sc = per.min(0) if rd == 'min' else per.sum(0)
    else:
        sc = np.zeros((Q, V + 1))
        for t in range(T):
            v = tabs[t][qs[t]]  # (Q, P+1)
            for j in range(v.shape[1]): np.add.at(sc, (np.arange(Q), v[:, j]), 1.0)
        sc = sc[:, :V]
    return argmax_rt(sc.astype(np.float64), rng)

# ------------------------------------------------------------------ family 4: Bloom filter over (table, signature, value)
def bloom_eval(st, prep, B, V, k, T, h, P, sh, rng):
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    ks = hyp_sigs(prep, st['keys'][:ns], T, k, 0)
    vals = st['vals'][:ns]
    ins = np.concatenate([np.stack([hmix(t, ks[t][:, 0], vals, j) for j in range(h)], 1) for t in range(T)], 0)  # (ns*T, h)
    nins = ns * T * h
    # largest m whose filter fits m*H(fill) <= B; start from the expected-fill solution then verify
    def cost(m):
        bits = np.zeros(m, bool); bits[(ins % np.uint64(m)).astype(np.int64).ravel()] = True
        return m * float(H2(bits.mean())), bits
    lo, hi = 8, max(16, 64 * B)
    while hi - lo > max(1, lo // 200):
        mid = (lo + hi) // 2
        exp_p = 1 - math.exp(-nins / mid)
        if mid * H2(exp_p) <= B: lo = mid
        else: hi = mid
    m = lo
    c, bits = cost(m)
    while c > B and m > 8:
        m = int(m * 0.98); c, bits = cost(m)
    qs = hyp_sigs(prep, st['q'], T, k, P)
    Q = len(st['q'])
    sc = np.zeros((Q, V))
    mm = np.uint64(m)
    for t in range(T):
        pr = qs[t]
        for v in range(V):
            ok = np.ones(pr.shape, bool)
            for j in range(h):
                ok &= bits[(hmix(t, pr, np.full(pr.shape, v), j) % mm).astype(np.int64)]
            sc[:, v] += ok.sum(1)
    return argmax_rt(sc, rng)

# ------------------------------------------------------------------ family 5: fly Bloom filter per value
def fly_feats(st, prep):
    rows = np.concatenate([st['keys'], st['q']])
    return np.asarray((G['fly'].T @ G['Xp'][prep][rows].T).T, dtype=np.float32)

def fly_tags(F, m, a):
    idx = np.argpartition(-F[:, :m], a - 1, axis=1)[:, :a]
    return idx

def fly_eval(st, Fall, B, V, a, norm, sh, rng):
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    vals = st['vals'][:ns]
    nv = np.bincount(vals, minlength=V)
    # choose filter length m (per value) from expected fill, then verify actual entropy
    def exp_cost(m):
        p = 1 - (1 - a / m) ** nv
        return float((m * H2(p)).sum())
    lo, hi = a + 1, FLY_MMAX
    if exp_cost(lo) > B: return None
    while hi - lo > max(1, lo // 200):
        mid = (lo + hi) // 2
        if exp_cost(mid) <= B: lo = mid
        else: hi = mid
    m = lo
    while True:
        tk = fly_tags(Fall[:ns], m, a)
        filt = np.zeros((V, m), bool)
        for v in range(V):
            filt[v, tk[vals == v].ravel()] = True
        c = float((m * H2(filt.mean(1))).sum())
        if c <= B or m <= a + 1: break
        m = int(m * 0.97)
    tq = fly_tags(Fall[nk:], m, a)
    ov = filt[:, tq].sum(2).T.astype(np.float64) / a  # (Q, V)
    if norm == 'corr': ov = ov - filt.mean(1)[None, :]
    return argmax_rt(ov, rng)

# ------------------------------------------------------------------ family 6: sparse distributed memory
def sdm_sims(st, prep, addr):
    rows = np.concatenate([st['keys'], st['q']])
    Xr = G['Xp'][prep][rows]
    if addr == 'gauss':
        return Xr @ G['sdm_gauss']
    A = G['Xp'][prep][G['train_rows']]
    return Xr @ A.T

def sdm_eval(st, Sims, B, V, a, ctr, sh, rng):
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    vals = st['vals'][:ns]
    Nmax = Sims.shape[1]
    if ctr == 'bits':
        nb = max(1, int(math.ceil(math.log2(V))))
        vb = ((np.arange(V)[:, None] >> np.arange(nb)) & 1) * 2 - 1  # (V, nb) bipolar value bits
    Ns = sorted(set([int(x) for x in np.unique(np.round(np.geomspace(max(a, 8), Nmax, 64)))]))
    def write(N):
        act = np.argpartition(-Sims[:ns, :N], a - 1, axis=1)[:, :a] if a < N else np.tile(np.arange(N), (ns, 1))
        if ctr == 'onehot':
            C = np.zeros((N, V), np.int64)
            np.add.at(C, (act.ravel(), np.repeat(vals, act.shape[1])), 1)
        else:
            C = np.zeros((N, nb), np.int64)
            np.add.at(C, act.ravel(), np.repeat(vb[vals], act.shape[1], axis=0))
        u, cn = np.unique(C, return_counts=True)
        return C.size * entropy_hist(cn), C
    # bisection over the ascending size grid (cost grows with N), then step down until it fits
    lo, hi = 0, len(Ns) - 1
    c_lo, C_lo = write(Ns[0])
    if c_lo > B: return None
    c_hi, C_hi = write(Ns[hi])
    if c_hi <= B: lo = hi; C_lo = C_hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        c, Cm = write(Ns[mid])
        if c <= B: lo, C_lo = mid, Cm
        else: hi = mid
    N, C = Ns[lo], C_lo
    Sq = Sims[nk:, :N]
    actq = np.argpartition(-Sq, a - 1, axis=1)[:, :a] if a < N else np.tile(np.arange(N), (len(Sq), 1))
    R = C[actq].sum(1).astype(np.float64)  # (Q, V) or (Q, nb)
    if ctr == 'onehot':
        return argmax_rt(R, rng)
    return argmax_rt(R @ vb.T.astype(np.float64), rng)


# ------------------------------------------------------------------ B66: SDM with training locations on B64's grid
SDMT_LOCS = ['LA', 'L', 'A']
SDMT_AS = [1, 2, 4, 8, 16, 32, 64]
SDMT_QS = ['cnt', 'clip3', 'tern']

def sdmt_grid():
    """B64's configuration space, B64's tie order (prep > loc > a > ctr > q > share)."""
    return [('sdmt', p, loc, a, ctr, q, sh) for p in PREPS for loc in SDMT_LOCS for a in SDMT_AS
            for ctr in ('onehot', 'bits') for q in SDMT_QS for sh in SHARES[:2]]

def qtrans(C, ctr, q):
    # B66: B64's counter types; 'cnt' exact, 'clip3' saturating, 'tern' presence / sign
    if q == 'cnt': return C
    if q == 'clip3': return np.minimum(C, 3) if ctr == 'onehot' else np.clip(C, -3, 3)
    return np.minimum(C, 1) if ctr == 'onehot' else np.sign(C)

def sdmt_eval(st, Sims, B, V, loc, a, ctr, q, sh, rng):
    """cr.py's sdm_eval (copied line for line) on the location-set columns of the data-address similarities, with the
    counters passed through B64's counter type before the cost and the read (the two B66 additions)."""
    nl = G['n_lab']
    Sims = Sims if loc == 'LA' else (Sims[:, :nl] if loc == 'L' else Sims[:, nl:])  # B66: location set
    nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
    vals = st['vals'][:ns]
    Nmax = Sims.shape[1]
    if ctr == 'bits':
        nb = max(1, int(math.ceil(math.log2(V))))
        vb = ((np.arange(V)[:, None] >> np.arange(nb)) & 1) * 2 - 1  # (V, nb) bipolar value bits
    Ns = sorted(set([int(x) for x in np.unique(np.round(np.geomspace(max(a, 8), Nmax, 64)))]))
    def write(N):
        act = np.argpartition(-Sims[:ns, :N], a - 1, axis=1)[:, :a] if a < N else np.tile(np.arange(N), (ns, 1))
        if ctr == 'onehot':
            C = np.zeros((N, V), np.int64)
            np.add.at(C, (act.ravel(), np.repeat(vals, act.shape[1])), 1)
        else:
            C = np.zeros((N, nb), np.int64)
            np.add.at(C, act.ravel(), np.repeat(vb[vals], act.shape[1], axis=0))
        C = qtrans(C, ctr, q)  # B66: counter type
        u, cn = np.unique(C, return_counts=True)
        return C.size * entropy_hist(cn), C
    # bisection over the ascending size grid (cost grows with N), then step down until it fits
    lo, hi = 0, len(Ns) - 1
    c_lo, C_lo = write(Ns[0])
    if c_lo > B: return None
    c_hi, C_hi = write(Ns[hi])
    if c_hi <= B: lo = hi; C_lo = C_hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        c, Cm = write(Ns[mid])
        if c <= B: lo, C_lo = mid, Cm
        else: hi = mid
    N, C = Ns[lo], C_lo
    Sq = Sims[nk:, :N]
    actq = np.argpartition(-Sq, a - 1, axis=1)[:, :a] if a < N else np.tile(np.arange(N), (len(Sq), 1))
    R = C[actq].sum(1).astype(np.float64)  # (Q, V) or (Q, nb)
    if ctr == 'onehot':
        return argmax_rt(R, rng)
    return argmax_rt(R @ vb.T.astype(np.float64), rng)

# ------------------------------------------------------------------ grids
def family_grid():
    g = []
    for p in PREPS:
        for fam in ['lsh_sym', 'lsh_asym', 'pca_sym', 'pca_asym', 'itq_sym', 'itq_asym', 'pq16', 'pq256']:
            for sh in SHARES: g.append(('index', p, fam, sh))
    for p in PREPS:
        for k in (4, 8, 12, 16):
            for f in (0, 2, 4):
                for P in (0, 4):
                    for sh in SHARES[:3]: g.append(('static', p, k, f, P, sh))
    for p in PREPS:
        for k in (4, 6, 8, 10, 12):
            for hold, rd in (('counts', 'min'), ('counts', 'sum'), ('value', 'vote')):
                for P in (0, 4):
                    for sh in SHARES[:2]: g.append(('cms', p, k, hold, rd, P, sh))
    for p in PREPS:
        for k in (6, 8, 12, 16):
            for T in (1, 4, 16):
                for h in (1, 3):
                    for P in (0, 4):
                        for sh in SHARES[:2]: g.append(('bloom', p, k, T, h, P, sh))
    for p in PREPS:
        for a in (4, 8, 16, 32):
            for norm in ('raw', 'corr'):
                for sh in SHARES[:2]: g.append(('fly', p, a, norm, sh))
    for p in PREPS:
        for addr in ('gauss', 'data'):
            for a in (1, 4, 16, 64):
                for ctr in ('onehot', 'bits'):
                    for sh in SHARES[:2]: g.append(('sdm', p, addr, a, ctr, sh))
    return g

def run_family_cfgs(st, B, V, cfgs, seed):
    """Evaluate a list of family configs on one store; returns {cfg: preds or None}."""
    rng = np.random.default_rng(seed)
    out = {}
    cache = {}
    for cfg in cfgs:
      fam = cfg[0]
      try:
        if fam == 'index':
            pr = index_eval(st, cfg[1], B, V, cfg[2], cfg[3], rng)
        elif fam == 'static':
            pr = static_eval(st, cfg[1], B, V, cfg[2], cfg[3], cfg[4], cfg[5], rng)
        elif fam == 'cms':
            pr = cms_eval(st, cfg[1], B, V, cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], rng)
        elif fam == 'bloom':
            pr = bloom_eval(st, cfg[1], B, V, cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], rng)
        elif fam == 'fly':
            key = ('fly', cfg[1])
            if key not in cache: cache = {k: v for k, v in cache.items() if k[0] != 'fly'}; cache[key] = fly_feats(st, cfg[1])
            pr = fly_eval(st, cache[key], B, V, cfg[2], cfg[3], cfg[4], rng)
        elif fam == 'sdm':
            key = ('sdm', cfg[1], cfg[2])
            if key not in cache: cache = {k: v for k, v in cache.items() if k[0] != 'sdm'}; cache[key] = sdm_sims(st, cfg[1], cfg[2])
            pr = sdm_eval(st, cache[key], B, V, cfg[3], cfg[4], cfg[5], rng)
        elif fam == 'sdmt':  # B66
            key = ('sdm', cfg[1], 'data')
            if key not in cache: cache = {k: v for k, v in cache.items() if k[0] != 'sdm'}; cache[key] = sdm_sims(st, cfg[1], 'data')
            pr = sdmt_eval(st, cache[key], B, V, cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], rng)
      except Exception as ex:
        print(f'CFG-ERROR {cfg}: {type(ex).__name__} {ex}', flush=True)
        pr = None
      out[cfg] = pr
    return out

def chance_preds(st, V, rng):
    return rng.integers(0, V, len(st['q']))

def acc_of(stores, predlist):
    c = np.concatenate([(p == st['qv']) for st, p in zip(stores, predlist)])
    return 100.0 * c.mean(), c

def choose(grid, devacc):
    best = max(devacc[g] for g in grid)
    for g in grid:
        if devacc[g] >= best - 0.1: return g

# ------------------------------------------------------------------ one condition
# B66: split into a dev phase and a test phase (same seeds and order of computation as cr.py's one-pass run_condition)
FAMS = ['index', 'static', 'cms', 'bloom', 'fly', 'sdm']
EXTRA = ['sdmG', 'sdmTcr', 'sdmT64']        # B66 arms: SDM Gaussian addresses only; SDMt on cr's grid; SDMt on B64's grid

def best_of(fams, ch, devacc):
    best_f = None
    for f in fams:  # cr.py's rule: best family by dev accuracy, ties within 0.1 to first in this order
        if best_f is None or devacc[ch[f]] > devacc[ch[best_f]] + 0.1: best_f = f
    return best_f

def cond_setup(M, V, b):
    B = b * M
    cs = M * 1000 + V * 10 + b
    ndev, ntest = (6, 20) if M == 256 else (3, 5)
    dev = make_stores('dev', M, V, ndev, 1000 + cs)
    test = make_stores('test', M, V, ntest, 2000 + cs)
    crng = np.random.default_rng(3000 + cs)
    codes = {'first': value_codes(V, dfor(B, 1), 'first', crng), 'final': value_codes(V, dfor(B, 1), 'final', crng)}
    return B, dev, test, codes

def store_preds(stores, tag, B, V, codes):
    allp = [dict() for _ in stores]
    for si, st in enumerate(stores):
        for p in PREPS:
            allp[si].update(store_eval(st, p, B, V, codes, seed=(0 if tag == 'dev' else 50000) + 100 * si + PREPS.index(p)))
    return allp

SGRID = {d: [(d, p, mp, q, sh) for p in PREPS for mp in ('lin', 'sign', 'prod') for q in (1, 2, 4) for sh in SHARES] for d in ('first', 'final')}

def dev_file(enc, M, V, b):
    return f'{WD}/out_{enc}/dev_M{M}_V{V}_b{b}.json'

def run_dev(args):
    enc, M, V, b = args
    t0 = time.time()
    B, dev, test, codes = cond_setup(M, V, b)
    devp = store_preds(dev, 'dev', B, V, codes)
    sdev = {c: acc_of(dev, [dp[c] for dp in devp])[0] for c in SGRID['first'] + SGRID['final']}
    schosen = {d: choose(SGRID[d], sdev) for d in SGRID}
    del devp
    grid = family_grid() + sdmt_grid()
    devpreds = [run_family_cfgs(st, B, V, grid, seed=5000 + si) for si, st in enumerate(dev)]
    devacc = {}
    for g in grid:
        pl = [dp[g] if dp[g] is not None else None for dp in devpreds]
        if any(p is None for p in pl): devacc[g] = -1.0
        else: devacc[g] = acc_of(dev, pl)[0]
    fg = family_grid()
    ch = {f: choose([g for g in fg if g[0] == f], devacc) for f in FAMS}
    ch['sdmG'] = choose([g for g in fg if g[0] == 'sdm' and g[2] == 'gauss'], devacc)
    ch['sdmTcr'] = choose([g for g in fg if g[0] == 'sdm' and g[2] == 'data'], devacc)
    ch['sdmT64'] = choose(sdmt_grid(), devacc)
    best = {'best6': best_of(FAMS, ch, devacc),
            'best7cr': best_of(FAMS[:5] + ['sdmG', 'sdmTcr'], ch, devacc),
            'best7_64': best_of(FAMS[:5] + ['sdmG', 'sdmT64'], ch, devacc)}
    out = {'enc': enc, 'M': M, 'V': V, 'b': b, 'B': B,
           'store_choice': {d: list(schosen[d]) for d in schosen}, 'store_dev': {d: sdev[schosen[d]] for d in schosen},
           'choice': {f: list(ch[f]) for f in ch}, 'choice_dev': {f: devacc[ch[f]] for f in ch}, 'best': best,
           'devacc_all': {'|'.join(map(str, g)): round(devacc[g], 4) for g in grid}, 'n_grid': len(grid),
           'secs': time.time() - t0}
    os.makedirs(f'{WD}/out_{enc}', exist_ok=True)
    json.dump(out, open(dev_file(enc, M, V, b) + '.tmp', 'w'), indent=1)
    os.replace(dev_file(enc, M, V, b) + '.tmp', dev_file(enc, M, V, b))
    print(f'[{enc} M{M} V{V} b{b}] DEV done {time.time()-t0:.0f}s ' + ' '.join(f'{f}={devacc[ch[f]]:.1f}' for f in ch), flush=True)
    return out

def tup(c):
    return tuple(int(x) if isinstance(x, str) and x.lstrip('-').isdigit() else x for x in c)

def run_test(args):
    enc, M, V, b = args
    t0 = time.time()
    D = json.load(open(dev_file(enc, M, V, b)))
    B, dev, test, codes = cond_setup(M, V, b)
    schosen = {d: tuple(D['store_choice'][d]) for d in SGRID}
    testp = store_preds(test, 'test', B, V, codes)
    res = {'enc': enc, 'M': M, 'V': V, 'b': b, 'B': B, 'n_test_queries': int(sum(len(s['q']) for s in test))}
    corr = {}
    for d in SGRID:
        a, c = acc_of(test, [tp[schosen[d]] for tp in testp])
        res[f'store_{d}'] = {'cfg': list(map(str, schosen[d])), 'dev': D['store_dev'][d], 'test': a}
        corr[f'store_{d}'] = c
    del testp
    ch = {f: tuple(D['choice'][f]) for f in D['choice']}
    first = [ch[f] for f in FAMS]
    extra = []
    for f in EXTRA:
        if ch[f] not in first and ch[f] not in extra: extra.append(ch[f])
    tpreds = [run_family_cfgs(st, B, V, first + extra, seed=6000 + si) for si, st in enumerate(test)]
    rng = np.random.default_rng(7)
    for f in FAMS + EXTRA:
        if f == 'sdmG': rng = np.random.default_rng(8)  # B66: the extra arms' chance fallback has its own stream
        pl = [tp[ch[f]] for tp in tpreds]
        pl = [p if p is not None else chance_preds(st, V, rng) for p, st in zip(pl, test)]
        a, c = acc_of(test, pl)
        res[f] = {'cfg': list(map(str, ch[f])), 'dev': D['choice_dev'][f], 'test': a}
        corr[f] = c
    for k, f in D['best'].items():
        res[k] = {'family': f, 'dev': D['choice_dev'][f], 'test': res[f]['test']}
        corr[k] = corr[f]
    corr['qe'] = np.concatenate([s['qe'] for s in test])
    corr['qstore'] = np.concatenate([np.full(len(s['q']), i) for i, s in enumerate(test)])
    res['secs'] = time.time() - t0
    np.savez(f'{WD}/out_{enc}/corr_M{M}_V{V}_b{b}.npz', **corr)
    json.dump(res, open(f'{WD}/out_{enc}/cell_M{M}_V{V}_b{b}.json', 'w'), indent=1)
    print(f'[{enc} M{M} V{V} b{b}] TEST done {time.time()-t0:.0f}s ' + ' '.join(f'{f}={res[f]["test"]:.1f}' for f in FAMS + EXTRA), flush=True)
    return res

def selftest(enc):
    """B66 selftest on TRAINING-split stores only: (1) sdmt_eval on ('LA', a, ctr, 'cnt', share) equals cr.py's sdm_eval
    with data addresses (same rng state) on two training stores; (2) every sdmt configuration returns predictions of the
    right shape and the chosen N fits the budget (recomputed); (3) the L / A column split equals the label / alias rows."""
    setup(enc)
    n_ok = n = 0
    for M, V, b in ((256, 2, 4), (256, 16, 8), (1024, 16, 16)):
        B = b * M
        st = make_stores('train', M, V, 1, 99)[0]
        S = sdm_sims(st, 'whiten', 'data')
        for a in (1, 4, 16, 64):
            for ctr in ('onehot', 'bits'):
                for sh in (1.0, 0.5):
                    p1 = sdm_eval(st, S, B, V, a, ctr, sh, np.random.default_rng(5))
                    p2 = sdmt_eval(st, S, B, V, 'LA', a, ctr, 'cnt', sh, np.random.default_rng(5))
                    n += 1
                    n_ok += (p1 is None and p2 is None) or (p1 is not None and p2 is not None and (p1 == p2).all())
        for cfg in sdmt_grid()[::7]:
            pr = sdmt_eval(st, S, B, V, *cfg[2:], np.random.default_rng(1))
            assert pr is None or pr.shape == (len(st['q']),), cfg
    tr = G['train_rows']; nl = G['n_lab']
    labs = np.array([l for l, _ in G['split']['train']])
    assert (tr[:nl] == labs).all() and len(tr) - nl == sum(len(al) for _, al in G['split']['train'])
    print(f'selftest {enc}: sdmt_eval(LA, cnt) == sdm_eval(data) on {n_ok}/{n}; grid {len(sdmt_grid())}; split ok', flush=True)
    assert n_ok == n

if __name__ == '__main__':
    # B66: python3 b66_cr.py ENC dev|test NPROC [M,V,b]   |   python3 b66_cr.py ENC selftest
    enc, phase = sys.argv[1], sys.argv[2]
    if phase == 'selftest':
        selftest(enc); sys.exit(0)
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    only = sys.argv[4] if len(sys.argv) > 4 else None
    setup(enc)
    conds = [(enc, M, V, b) for M in (1024, 256) for V in (2, 16) for b in (4, 8, 16)]
    if only:
        M, V, b = map(int, only.split(','))
        conds = [(enc, M, V, b)]
    fn = run_dev if phase == 'dev' else run_test
    if phase == 'dev':
        conds = [c for c in conds if not os.path.exists(dev_file(*c))]
    if nproc == 1:
        for c in conds: fn(c)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(nproc) as pool:
            for r in pool.imap_unordered(fn, conds): pass
