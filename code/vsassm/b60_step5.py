"""The diagnosis's step5 implementation (EXPLORATORY code of the LAM diagnosis, not B60's), reconstructed for B60's
registered agreement check (prereg/B60_corrected_lam_prereg.md, 'Checks'; compute note 1): the original files
(scratchpad lsdiag/step5_main.py, flex.py, pl.py and the clean-room common.py's quantize / train_stats) were erased by a
reboot; their text is copied here verbatim from the session transcript, merged into one file. Nothing of B60's code is
imported. The clean-room statistics file stats.npz (mu, lam, U) is recomputed by the clean-room train_stats.

python3 b60_step5.py M V       # -> b60_out/step5/step5_M{M}_V{V}.pkl (dev and test predictions of every configuration)
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np

ROOT = '/path/to/vsassm'
sys.path.insert(0, ROOT)
os.chdir(f'{ROOT}/b49_out')
import b32_common as C  # noqa: E402
import b49_common as B  # noqa: E402

OUTD = f'{ROOT}/b60_out/step5'

# ---------------------------------------------------------------- clean-room common.py (verbatim parts)
DATA = '/path/to/vsassm/b32_data'
BLOCK = 8192
SCALE_BITS = 32
GAUSS_STEP = {2: 0.9957, 4: 0.3352}


def load_data():
    ents = json.load(open(f'{DATA}/entities.json'))['entities']
    strings = json.load(open(f'{DATA}/emb_strings.json'))
    E = np.load(f'{DATA}/emb_qwen8_mean_L9_f16.npy', mmap_mode='r')
    name_row, alias_rows, split = [], [], []
    r = 0
    for e in ents:
        assert strings[r] == e['name']
        name_row.append(r); r += 1
        al = list(range(r, r + len(e['aliases']))); r += len(e['aliases'])
        alias_rows.append(al); split.append(e['split'])
    assert r == len(strings) == E.shape[0]
    return ents, E, np.array(name_row), alias_rows, np.array(split)


def train_stats(E, name_row, alias_rows, split):
    rows = []
    for i in np.where(split == 'train')[0]:
        rows.append(name_row[i]); rows.extend(alias_rows[i])
    X = np.asarray(E[np.array(rows)], dtype=np.float64)
    mu = X.mean(0)
    Cv = np.cov(X - mu, rowvar=False)
    lam, U = np.linalg.eigh(Cv)
    order = np.argsort(lam)[::-1]
    return mu, lam[order], U[:, order], len(rows)


def cc_quantize(x, q, rng):
    """x: 1-D float array. Blocks of 8192 numbers with one 32-bit scale each."""
    out = np.empty_like(x, dtype=np.float64)
    for b0 in range(0, len(x), BLOCK):
        blk = np.asarray(x[b0:b0 + BLOCK], dtype=np.float64)
        if q == 1:
            s = np.sign(blk)
            z = s == 0
            if z.any():
                s[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
            out[b0:b0 + BLOCK] = s * np.mean(np.abs(blk))
        else:
            sig = np.sqrt(np.mean(blk ** 2)) + 1e-30
            step = GAUSS_STEP[q] * sig
            L = 2 ** (q - 1)
            k = np.floor(blk / step)
            k = np.clip(k, -L, L - 1)
            out[b0:b0 + BLOCK] = (k + 0.5) * step
    return out


# ---------------------------------------------------------------- pl.py (verbatim)
TAG = 'qwen8_mean_L9_f16'
PREPS = ['center', 'whiten']
ALPHAS = [1e-4, 1e-2, 1.0, 100.0]
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}


class S:
    pass


def setup(eps=0.1):
    s = S()
    s.pool = C.Pool(TAG)
    Tr = s.pool.train_matrix()
    s.Tr = Tr
    s.preps = {p: C.Prep(p, Tr, eps=eps) for p in PREPS}
    s.lamf = {p: B.LamFeat('qwen8', p, s.preps[p](Tr)) for p in PREPS}
    return s


def lam_grid(V):
    return [(p, ft, a, q, 'onehot') for p in PREPS for ft in ('rp', 'pca') for a in ALPHAS for q in (1, 2, 4)]


def run_pipeline_store(s, st, cfgs, bpfs, accts=('ideal',)):
    M, V = st['M'], st['V']
    seed = C.seed_of('b49', 'qwen8', st['split'], M, V, st['rep'])
    res = {}
    for prep in PREPS:
        cs = [c for c in cfgs if c[0] == prep]
        if not cs: continue
        P = s.preps[prep]
        o = B.run_lam(st, P(st['keys']), P(st['queries']), s.lamf[prep], [c[1:] for c in cs], list(bpfs), list(accts), seed)
        for (c1, b, a), v in o.items():
            res[(prep,) + tuple(c1), b, a] = v
    return res


# ---------------------------------------------------------------- step5_main.py (verbatim apart from paths)
def main(M, V):
    BPF = [4, 8, 16]; QS = [1, 2, 4]; AL = ALPHAS
    s = setup()
    ents, E, nr, ar, sp = load_data()
    MU, LAM, U, _ = train_stats(E, nr, ar, sp)
    FEATS = {
        ('center', 'rp'): lambda Xr, w: np.hstack([s.lamf['center'].feats(s.preps['center'](Xr), 'rp', j0, min(8192, w - j0)) for j0 in range(0, w, 8192)]),
        ('center', 'pca'): lambda Xr, w: s.lamf['center'].feats(s.preps['center'](Xr), 'pca', 0, min(w, 4096)),
        ('whiten', 'rp'): lambda Xr, w: np.hstack([s.lamf['whiten'].feats(s.preps['whiten'](Xr), 'rp', j0, min(8192, w - j0)) for j0 in range(0, w, 8192)]),
        ('whiten', 'pcafix'): lambda Xr, w: pcafix(Xr)[:, :min(w, 4096)],
    }
    _U32 = U.astype(np.float32); _MU32 = MU.astype(np.float32)
    _SC = (1.0 / np.sqrt(np.maximum(LAM, 0) + 0.1 * LAM.mean())).astype(np.float32)

    def pcafix(Xr):
        Z = ((Xr.astype(np.float32) - _MU32) @ _U32) * _SC
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    PMAX = {('center', 'rp'): 65536, ('whiten', 'rp'): 65536, ('center', 'pca'): 4096, ('whiten', 'pcafix'): 4096}

    def p_of(rule, Bb, nc, q, pmax):
        if rule == 'col':
            return B.lam_p(Bb, nc, q, 'ideal', pmax)
        per = C.H_Q[q]
        p = int(Bb // (per * nc))
        while p > 0 and p * nc * per + 32 * math.ceil(p * nc / 8192) > Bb:
            p -= 1
        return min(p, pmax)

    def quant(W, q, rule, rng):
        if rule == 'col':
            out = np.empty_like(W)
            for j0 in range(0, W.shape[0], 8192):
                out[j0:j0 + 8192] = B.quant_cols(W[j0:j0 + 8192], q, rng)
            return out
        flat = W.T.ravel()
        return cc_quantize(flat, q, rng).reshape(W.shape[1], W.shape[0]).T

    FIXGRID = []
    for rule in ('col', 'flat'):
        for cols in ('V-1', 'V'):
            for (pr, ft) in [('center', 'rp'), ('center', 'pca'), ('whiten', 'rp'), ('whiten', 'pcafix')]:
                for a in AL:
                    for q in QS:
                        FIXGRID.append((pr, ft, a, q, cols, rule))

    def run_fixed(st, rng):
        g = st['qgold']; vals = st['vals']; Mm = st['M']
        T = np.eye(V)[vals] - 1.0 / V
        out = {}
        need = {}
        for cfg in FIXGRID:
            pr, ft, a, q, cols, rule = cfg
            nc = V - 1 if cols == 'V-1' else V
            for b in BPF:
                need.setdefault((pr, ft), []).append((cfg, b, p_of(rule, b * Mm, nc, q, PMAX[pr, ft]), nc))
        for (pr, ft), lst in need.items():
            pm = max(p for _, _, p, _ in lst)
            Fk = FEATS[pr, ft](st['keys'], pm).astype(np.float64)
            Fq = FEATS[pr, ft](st['queries'], pm).astype(np.float32)
            sols = {}
            for cfg, b, p, nc in lst:
                a, q, rule = cfg[2], cfg[3], cfg[5]
                if p <= 0:
                    out[cfg, b] = np.full(len(g), np.bincount(vals, minlength=V).argmax()); continue
                if (p, a) not in sols:
                    F = Fk[:, :p]; K = F @ F.T
                    A = np.linalg.solve(K + a * np.trace(K) / Mm * np.eye(Mm), T)
                    sols[p, a] = F.T @ A
                W = sols[p, a]
                if nc == V - 1:
                    Wq = quant(W[:, 1:], q, rule, rng)
                    Sx = Fq[:, :p].astype(np.float64) @ Wq
                    Sx = np.hstack([-Sx.sum(1, keepdims=True), Sx])
                else:
                    Sx = Fq[:, :p].astype(np.float64) @ quant(W, q, rule, rng)
                out[cfg, b] = Sx.argmax(1)
        return out

    rng = np.random.default_rng(M * 31 + V)
    R = {'dev': {}, 'test': {}}
    t0 = time.time()
    for split, reps in (('dev', DEV_REPS[M]), ('test', TEST_REPS[M])):
        for rep in range(reps):
            st = C.make_store(s.pool, split, M, V, rep); g = st['qgold']
            r = run_pipeline_store(s, st, lam_grid(V), BPF)
            rec = {('PL',) + tuple(c): {b: r[c, b, 'ideal'][0] for b in BPF} for c in lam_grid(V)}
            f = run_fixed(st, rng)
            for (cfg, b), p in f.items():
                rec.setdefault(('FX',) + cfg, {})[b] = p
            R[split][rep] = {'qgold': g, 'qent': st['qent'], 'pred': rec}
            print(split, rep, '%.0fs' % (time.time() - t0), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    pickle.dump(R, open(f'{OUTD}/step5_M{M}_V{V}.pkl', 'wb'))
    print('done')


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
