"""Clean-room equal-bit comparison: MAP store (final design) vs corrected least-squares memory.
Written only from the paper's text (Methods 4.5, SI Notes 5-10, SI Table 28). Ideal accounting.
Usage: python cr_equalbit.py <enc> <outjson>
"""
import sys, json, time, math
import numpy as np

T0 = time.time()
ENC = sys.argv[1]
OUT = sys.argv[2]
DATA = '/path/to/vsassm/b32_data'
FILES = {'qwen8': 'emb_qwen8_mean_L9_f16.npy', 'e5': 'emb_e5_base.npy'}
H_BITS = {1: 1.0, 2: 1.904, 4: 3.602}          # ideal accounting: output entropy for Gaussian input
MAX_STEP = {2: 0.9957, 4: 0.3352}              # Max's Gaussian-optimal uniform step (4 and 16 levels)
BLOCK = 8192
SCALE_BITS = 32
DMAX = 16320
MS = [256, 1024]
VS = [2, 16]
BS = [4, 8, 16]
KS = [1, 2, 4]
SHARES = [1.0, 0.5, 0.25, 0.125]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # ridge strength relative to mean diag of F F^T (ASSUMPTION)
EPS_WHITE_REL = 0.01                           # whitening floor relative to mean eigenvalue (ASSUMPTION)


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


# ---------------- data ----------------
ents = json.load(open(f'{DATA}/entities.json'))['entities']
strings = json.load(open(f'{DATA}/emb_strings.json'))
sidx = {s: i for i, s in enumerate(strings)}
E = np.load(f'{DATA}/{FILES[ENC]}').astype(np.float32)
n = E.shape[1]
log('codes', E.shape)


def split_block(sp):
    es = [x for x in ents if x['split'] == sp]
    lab_rows = [sidx[x['name']] for x in es]
    al_rows, al_owner = [], []
    for j, x in enumerate(es):
        for a in x['aliases']:
            al_rows.append(sidx[a]); al_owner.append(j)
    rows = np.array(lab_rows + al_rows)
    ne = len(es)
    lab_local = np.arange(ne)
    al_local = {}
    for t, j in enumerate(al_owner):
        al_local.setdefault(j, []).append(ne + t)
    return dict(rows=rows, ne=ne, lab=lab_local, al=al_local)


SPL = {sp: split_block(sp) for sp in ['train', 'dev', 'test']}
# training statistics: labels + aliases of training entities
Xtr = E[SPL['train']['rows']].astype(np.float64)
mu = Xtr.mean(0)
Cov = (Xtr - mu).T @ (Xtr - mu) / len(Xtr)
lam, U = np.linalg.eigh(Cov)
o = np.argsort(lam)[::-1]
lam = np.clip(lam[o], 0, None); U = U[:, o]
eps = EPS_WHITE_REL * lam.mean()
log('train codes', len(Xtr), 'eps', eps)


def prep(rows, white):
    Y = (E[rows].astype(np.float64) - mu) @ U       # rotate into principal basis (raw-code covariance)
    if white:
        Y = Y / np.sqrt(lam + eps)                  # PCA/ZCA whitening (rotation-equivalent)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y.astype(np.float32)


# ---------------- stores (shared by both memories) ----------------
NST = {('dev', 256): 6, ('dev', 1024): 3, ('test', 256): 20, ('test', 1024): 5}


def make_stores(sp, M, V):
    blk = SPL[sp]
    rng = np.random.default_rng([7, {'dev': 1, 'test': 2}[sp], M, V])
    out = []
    for _ in range(NST[(sp, M)]):
        e = rng.choice(blk['ne'], M, replace=False)
        v = rng.integers(0, V, M)
        qr, qv = [], []
        for j, ej in enumerate(e):
            for r in blk['al'].get(int(ej), []):
                qr.append(r); qv.append(v[j])
        out.append(dict(lab=blk['lab'][e], vals=v, qrows=np.array(qr, dtype=np.int64), qvals=np.array(qv)))
    return out


STORES = {(sp, M, V): make_stores(sp, M, V) for sp in ['dev', 'test'] for M in MS for V in VS}
for k_, s_ in STORES.items():
    log('stores', k_, len(s_), 'queries', sum(len(x['qvals']) for x in s_))


# ---------------- quantizer ----------------
QRNG = np.random.default_rng(12345)


def quant_vec(x, k):
    """Quantize a 1-D array in blocks of 8192 numbers, each with its own (32-bit) scale."""
    y = np.empty_like(x, dtype=np.float32)
    for s in range(0, len(x), BLOCK):
        b = x[s:s + BLOCK].astype(np.float64)
        sd = b.std()
        if sd == 0:
            sd = 1.0
        if k == 1:
            sg = np.sign(b)
            z = sg == 0
            if z.any():
                sg[z] = QRNG.choice([-1.0, 1.0], int(z.sum()))
            y[s:s + BLOCK] = sg * sd * math.sqrt(2 / math.pi)
        else:
            N = 2 ** k
            D = MAX_STEP[k] * sd
            y[s:s + BLOCK] = D * (np.clip(np.floor(b / D), -N // 2, N // 2 - 1) + 0.5)
    return y


def nblocks(cnt):
    return -(-cnt // BLOCK)


def store_dim(Bud, k):
    d = int((Bud - SCALE_BITS) / H_BITS[k]) + 2
    while d * H_BITS[k] + SCALE_BITS * nblocks(d) > Bud:
        d -= 1
    return d


def ls_dim(Bud, k, C, scale, cap):
    p = int(Bud / (C * H_BITS[k])) + 2
    def bits(p):
        nb = nblocks(p * C) if scale == 'shared' else C * nblocks(p)
        return p * C * H_BITS[k] + SCALE_BITS * nb
    while p > 0 and bits(p) > Bud:
        p -= 1
    return min(p, cap) if cap else p


H16 = np.array([[1]], dtype=np.float32)
while H16.shape[0] < 16:
    H16 = np.block([[H16, H16], [H16, -H16]])
RV = np.random.default_rng([ENC == 'qwen8', 99]).choice([-1.0, 1.0], DMAX).astype(np.float32)  # value-code base vector r


def value_codes(V, d):
    if V == 2:
        return np.stack([RV[:d], -RV[:d]])              # antipodal
    return RV[:d][None, :] * H16[:, np.arange(d) % 16]  # Sylvester-Hadamard rows: c_j(v)=r_j H16[v, j mod 16]


# results[cond_key][cfg_key] = [dev_correct, dev_n, test_correct, test_n]
RES_STORE = {}
RES_LS = {}


def add(res, cond, cfg, sp, corr, tot):
    r = res.setdefault(cond, {}).setdefault(cfg, [0, 0, 0, 0])
    o_ = 0 if sp == 'dev' else 2
    r[o_] += int(corr); r[o_ + 1] += int(tot)


def gaussian(seed, rows, cols):
    g = np.random.default_rng(seed)
    out = np.empty((rows, cols), dtype=np.float32)
    for s in range(0, rows, 2048):
        out[s:s + 2048] = g.standard_normal((min(2048, rows - s), cols), dtype=np.float32)
    return out


for white in [False, True]:
    pname = 'white' if white else 'centred'
    P = {sp: prep(SPL[sp]['rows'], white) for sp in ['dev', 'test']}
    log('prep', pname)
    # ---------- VSA store: three maps ----------
    G1 = gaussian([ENC == 'qwen8', 1], DMAX, n)
    MAPS = {}
    for sp in ['dev', 'test']:
        Z = P[sp] @ G1.T
        MAPS[(sp, 'lin')] = Z
        MAPS[(sp, 'sign')] = np.where(Z >= 0, 1, -1).astype(np.int8)
    del G1
    for t in [2, 3]:
        G = gaussian([ENC == 'qwen8', t], DMAX, n)
        for sp in ['dev', 'test']:
            S = np.where(P[sp] @ G.T >= 0, 1, -1).astype(np.int8)
            if (sp, 'prod3') not in MAPS:
                MAPS[(sp, 'prod3')] = MAPS[(sp, 'sign')] * S
            else:
                MAPS[(sp, 'prod3')] *= S
        del G
    log('store maps done', pname)
    for M in MS:
        for V in VS:
            for b in BS:
                Bud = b * M
                cond = f'{M}|{V}|{b}'
                for k in KS:
                    d = store_dim(Bud, k)
                    CV = value_codes(V, d)                       # V x d
                    for mp in ['lin', 'sign', 'prod3']:
                        for sp in ['dev', 'test']:
                            F = MAPS[(sp, mp)]
                            for st in STORES[(sp, M, V)]:
                                Fl = F[st['lab'], :d].astype(np.float32)
                                Fq = F[st['qrows'], :d].astype(np.float32)
                                for sh in SHARES:
                                    m = int(round(sh * M))
                                    bundle = (Fl[:m] * CV[st['vals'][:m]]).sum(0)
                                    bq = quant_vec(bundle, k)
                                    sc = (Fq * bq[None, :]) @ CV.T   # matched filter
                                    pred = sc.argmax(1)
                                    add(RES_STORE, cond, f'{pname}|{mp}|{k}|{sh}', sp,
                                        (pred == st['qvals']).sum(), len(pred))
                log('store', pname, M, V, 'done')
    del MAPS
    # ---------- corrected least squares ----------
    Gl = gaussian([ENC == 'qwen8', 50], DMAX, n)
    RF = {sp: P[sp] @ Gl.T for sp in ['dev', 'test']}
    del Gl
    log('ls random features done', pname)
    for M in MS:
        for V in VS:
            for sp in ['dev', 'test']:
                for st in STORES[(sp, M, V)]:
                    T = (np.eye(V)[st['vals']] - 1.0 / V)
                    groups = {}
                    for b in BS:
                        Bud = b * M
                        for feat in ['random', 'principal']:
                            for k in KS:
                                for C in [V - 1, V]:
                                    for scale in ['percol', 'shared']:
                                        p = ls_dim(Bud, k, C, scale, n if feat == 'principal' else None)
                                        if p >= 1:
                                            groups.setdefault((feat, p), []).append((b, k, C, scale))
                    for (feat, p), combos in groups.items():
                        src = RF[sp] if feat == 'random' else P[sp]
                        Fl = src[st['lab'], :p].astype(np.float64)
                        Fq = src[st['qrows'], :p].astype(np.float64)
                        K = Fl @ Fl.T
                        w, Uk = np.linalg.eigh(K)
                        w = np.clip(w, 0, None)
                        md = np.trace(K) / M
                        for (b, k, C, scale) in combos:
                            cond = f'{M}|{V}|{b}'
                            UT = Uk.T @ T[:, :C]
                            for a in ALPHAS:
                                A = Uk @ (UT / (w + a * md)[:, None])
                                W = Fl.T @ A                       # p x C ridge solution
                                if scale == 'shared':
                                    Wq = quant_vec(W.ravel().astype(np.float32), k).reshape(W.shape)
                                else:
                                    Wq = np.stack([quant_vec(W[:, c].astype(np.float32), k)
                                                   for c in range(C)], 1)
                                S = Fq @ Wq.astype(np.float64)
                                if C == V - 1:
                                    S = np.hstack([S, -S.sum(1, keepdims=True)])
                                pred = S.argmax(1)
                                add(RES_LS, cond, f'{pname}|{feat}|{k}|{C}|{scale}|{a}', sp,
                                    (pred == st['qvals']).sum(), len(pred))
                        del Fl, Fq, K, Uk
            log('ls', pname, M, V, 'done')
    del RF, P

json.dump(dict(store=RES_STORE, ls=RES_LS), open(OUT, 'w'))
log('wrote', OUT)
