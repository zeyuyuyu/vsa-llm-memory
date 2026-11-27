"""B32 second angle (prereg/B32_vector_form_equal_bits_prereg.md): an independent scorer and an independent
re-derivation of a sample of the test predictions. Shares no code with b32_common / b32_run / b32_verdict: the seeds,
projections, codes and read-outs are re-implemented from the written specification (prereg, 'Memories').

python3 b32_check.py score  OUTDIR                    # recompute every cell and region from the dumps; compare with
                                                       # results/b32_verdict.json; store-level bootstrap as a robustness CI
python3 b32_check.py rederive OUTDIR CHOICE_qwen8.json CHOICE_e5.json [--frac 0.1]
python3 b32_check.py bits CHOICE_qwen8.json CHOICE_e5.json   # every chosen configuration fits its budget
"""
import gzip
import hashlib
import json
import math
import random
import re
import sys

import numpy as np

MS, VS, BPFS = (256, 1024), (2, 16, 128, 1024), (4, 8, 16, 32, 64)
BLOCK = 8192
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}
STEP = {2: 0.9957, 3: 0.5860, 4: 0.3352}


def sd_(*parts):  # the registered seed: first 15 hex digits of sha256 over '|'-joined parts
    return int(hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def recs(outdir, enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{outdir}/preds_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


# ------------------------------------------------------------------------------------------------ score

def score(outdir):
    J = json.load(open('results/b32_verdict.json'))
    worst = 0.0
    n = 0
    region = {}
    for enc in ('qwen8', 'e5'):
        for M in MS:
            for V in VS:
                R = recs(outdir, enc, M, V)
                for acct in ('ideal', 'raw'):
                    for b in BPFS:
                        tot = cv = cx = 0
                        per_store = []
                        for r in R:
                            g = r['qgold']
                            pv, px = r['pred'][f'VSA|{b}|{acct}'], r['pred'][f'IDX|{b}|{acct}']
                            a = sum(1 for i in range(len(g)) if pv[i] == g[i])
                            c = sum(1 for i in range(len(g)) if px[i] == g[i])
                            cv += a
                            cx += c
                            tot += len(g)
                            per_store.append((a - c, len(g)))
                        mine = (cv - cx) / tot
                        theirs = J['cells'][f'{enc}|{acct}|{M}|{V}|{b}']['diff']
                        worst = max(worst, abs(mine - theirs))
                        n += 1
                        region.setdefault((enc, acct), {})[M, V, b] = (mine, per_store)
    print(f'cells re-scored: {n}; largest |difference| from the verdict: {worst:.2e}')
    # regions with a store-level bootstrap (stores, not entities, resampled: a coarser, more conservative CI)
    R1 = [(M, V, b) for M in MS for V in (2, 16) for b in (4, 8, 16)]
    R2 = [(M, 1024, b) for M in MS for b in BPFS]
    rng = np.random.default_rng(12345)
    for (enc, acct), cells in region.items():
        for name, R in (('R1', R1), ('R2', R2)):
            stat = float(np.mean([cells[c][0] for c in R]))
            theirs = J['regions'][f'{enc}|{acct}|{name}']['diff']
            draws = np.zeros(4000)
            for c in R:
                ps = cells[c][1]
                dif = np.array([x for x, _ in ps], float)
                cnt = np.array([y for _, y in ps], float)
                idx = rng.integers(0, len(ps), size=(4000, len(ps)))
                draws += dif[idx].sum(1) / cnt[idx].sum(1)
            draws /= len(R)
            print(f'{enc:5s} {acct:5s} {name}: {100 * stat:+.2f} (verdict {100 * theirs:+.2f}); store-level 95% CI '
                  f'[{100 * np.quantile(draws, 0.025):+.2f}, {100 * np.quantile(draws, 0.975):+.2f}]')


# ------------------------------------------------------------------------------------------------ re-derivation

class Codes:
    def __init__(self, enc_file):
        S = json.load(open('b32_data/emb_strings.json'))
        self.ix = {s: i for i, s in enumerate(S)}
        self.X = np.load(enc_file).astype(np.float64)
        E = json.load(open('b32_data/entities.json'))['entities']
        self.E = {e['uri']: e for e in E}
        tr = [e for e in E if e['split'] == 'train']
        self.T = self.X[[self.ix[e['name']] for e in tr] + [self.ix[a] for e in tr for a in e['aliases']]]

    def prep(self, kind):
        mu = self.T.mean(0)
        if kind == 'center':
            def f(Z):
                Y = Z - mu
                return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            return f
        Cv = np.cov((self.T - mu).T)
        w, U = np.linalg.eigh(Cv)
        w = np.maximum(w, 0)
        W = (U / np.sqrt(w + 0.1 * w.mean())) @ U.T  # ZCA

        def g(Z):
            Y = (Z - mu) @ W
            return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return g


def vsa_dim(B, q, acct):
    per = q if acct == 'raw' else HQ[q]
    best = 0
    lo, hi = 0, int(B / per) + 1
    while lo <= hi:  # largest d with d*per + 32*ceil(d/BLOCK) <= B
        mid = (lo + hi) // 2
        if mid * per + 32 * math.ceil(mid / BLOCK) <= B:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def vsa_predict(K, vals, Q, V, d, q, fixed, sseed, stored):
    """Materialise the d-dimensional codes block by block: projection blocks (n x BLOCK Gaussian) from
    sd_('G', enc, prep, fmap, factor, start, BLOCK); value codes = the first V rows of a 1024 x BLOCK random sign block
    from sd_('C', enc, prep, fmap, start, BLOCK); bind, bundle the stored facts, quantize each block with its own scale
    (sign ties broken by sd_('Q', store seed, d, q, n_stored, start)), unbind, clean up against all V codes."""
    enc, prep, fmap = fixed
    import torch
    n = K.shape[1]
    nf = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
    score = torch.zeros((Q.shape[0], V), dtype=torch.float64)
    Kt, Qt = torch.from_numpy(K.astype(np.float32)), torch.from_numpy(Q.astype(np.float32))
    for s0 in range(0, d, BLOCK):
        w = min(BLOCK, d - s0)
        fk = torch.ones((K.shape[0], w))
        fq = torch.ones((Q.shape[0], w))
        for m in range(nf):
            G = torch.from_numpy(np.random.default_rng(sd_('G', enc, prep, fmap, m, s0, BLOCK)).standard_normal(
                (n, BLOCK), dtype=np.float32)[:, :w])
            a, b = Kt @ G, Qt @ G
            if fmap != 'lin':
                a, b = torch.where(a >= 0, 1.0, -1.0), torch.where(b >= 0, 1.0, -1.0)
            fk, fq = fk * a, fq * b
        r = np.random.default_rng(sd_('C', enc, prep, fmap, s0, BLOCK))
        cv = torch.from_numpy(np.where(r.random((1024, BLOCK), dtype=np.float32) < 0.5, -1.0, 1.0).astype(
            np.float32)[:V, :w])
        S = (fk[stored] * cv[torch.from_numpy(vals[stored])]).sum(0).double()
        rq = np.random.default_rng(sd_('Q', sseed, d, q, len(stored), s0))
        if q == 1:
            Sq = torch.sign(S)
            z = (Sq == 0).numpy()
            if z.any():
                Sq[torch.from_numpy(z)] = torch.from_numpy(rq.choice([-1.0, 1.0], size=int(z.sum())))
        else:
            sdv = float(S.float().std(unbiased=False)) or 1.0
            st = STEP[q] * sdv
            L = 2 ** q
            Sq = (torch.clamp(torch.floor(S / st), -L // 2, L // 2 - 1) + 0.5) * st
        score += ((fq.double() * Sq) @ cv.double().T)
    return score.argmax(1).numpy()


def multiset_bits(b, V, M):
    lN = b + math.log2(V)
    return M * lN + sum(math.log1p(i / 2.0 ** lN) for i in range(M)) / math.log(2) - math.lgamma(M + 1) / math.log(2)


def key_bits(B, M, V, acct):
    vb = min(V * math.ceil(math.log2(M + 1)), M * math.ceil(math.log2(max(V, 2))))
    raw = max(0, (B - vb) // M)
    if acct == 'raw':
        return raw
    b = 0
    while b < 4096 and multiset_bits(b + 1, V, M) <= B:
        b += 1
    return max(b, raw)


_MEMO = {}


def _memo(key, fn):
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


def idx_predict(codes, prepf, family, K, vals, Q, kb, seed=0):
    """lsh / itq / sq1 re-implemented; pq / opq via faiss's codes and centroids with our own ADC search.
    Fitted pieces (training codes, PCA, ITQ rotations, PQ / OPQ) are memoised per (preprocessing, family, size)."""
    n = K.shape[1]
    T = _memo((id(prepf), 'T'), lambda: prepf(codes.T))
    if family == 'lsh':
        p = kb
        if p < 1:
            return None
        G = np.random.default_rng(sd_('LSH', seed, p)).standard_normal((n, p), dtype=np.float32).astype(np.float64)
        return vals[((Q @ G) @ np.where(K @ G >= 0, 1.0, -1.0).T).argmax(1)]
    def pca():
        mu_ = T.mean(0)
        w_, U_ = np.linalg.eigh(np.cov((T - mu_).T))
        return mu_, U_[:, np.argsort(-w_)]
    mu, U = _memo((id(prepf), 'pca'), pca)
    if family == 'itq':
        p = min(kb, n)
        if p < 1:
            return None
        Up = U[:, :p]

        def itq():
            Z = (T - mu) @ Up
            R = np.linalg.qr(np.random.default_rng(sd_('ITQ', seed, p)).standard_normal((p, p)))[0]
            for _ in range(50):
                u, _, vt = np.linalg.svd(Z.T @ np.sign(Z @ R))
                R = u @ vt
            return Up @ R
        A = _memo((id(prepf), 'itq', p), itq)
        return vals[(((Q - mu) @ A) @ np.where((K - mu) @ A >= 0, 1.0, -1.0).T).argmax(1)]
    if family == 'sq1':
        p = min(kb, n)
        if p < 1:
            return None
        Up = U[:, :p]
        s = ((T - mu) @ Up).std(0) + 1e-8
        Zh = np.where(((K - mu) @ Up) / s >= 0, 1.0, -1.0) * math.sqrt(2 / math.pi)
        return vals[(Q @ ((Zh * s) @ Up.T + mu).T).argmax(1)]
    import faiss
    nb = 4 if family == 'pq4' else 8
    ms = [m for m in range(1, n + 1) if n % m == 0 and m * nb <= kb]
    if not ms:
        return None
    m = max(ms)

    def fit():
        Tf = np.ascontiguousarray(T.astype(np.float32))
        if family == 'opq8':
            opq = faiss.OPQMatrix(n, m)
            opq.niter = 4  # prereg/B32_compute_note1.md
            opq.train(Tf)
            rot_ = lambda Z: opq.apply_py(np.ascontiguousarray(Z.astype(np.float32)))
        else:
            rot_ = lambda Z: np.ascontiguousarray(Z.astype(np.float32))
        pq_ = faiss.ProductQuantizer(n, m, nb)
        pq_.train(rot_(Tf))
        return rot_, pq_, faiss.vector_to_array(pq_.centroids).reshape(m, 2 ** nb, n // m)
    rot, pq, cent = _memo((id(prepf), family, m), fit)
    codes_k = pq.compute_codes(rot(K))
    Qr = rot(Q)
    if nb == 4:  # codes packed two per byte, low nibble first
        cb = np.zeros((codes_k.shape[0], m), np.int64)
        for j in range(m):
            byte = codes_k[:, j // 2].astype(np.int64)
            cb[:, j] = (byte >> (4 * (j % 2))) & 15
    else:
        cb = codes_k.astype(np.int64)
    dist = np.zeros((Q.shape[0], K.shape[0]))
    for j in range(m):
        sub = Qr[:, j * (n // m):(j + 1) * (n // m)].astype(np.float64)
        tab = ((sub[:, None, :] - cent[j][None]) ** 2).sum(-1)  # Q x 2^nb
        dist += tab[:, cb[:, j]]
    return vals[dist.argmin(1)]


def rederive(outdir, chq, che, frac, encs=('qwen8', 'e5')):
    files = {'qwen8': 'b32_data/emb_qwen8_mean_L9_f16.npy', 'e5': 'b32_data/emb_e5_base.npy'}
    chs = {e: json.load(open(p))['choice'] for e, p in (('qwen8', chq), ('e5', che)) if e in encs}
    rnd = random.Random(2026)
    tot = {'VSA': [0, 0], 'IDX': [0, 0]}
    for enc in encs:
        tot = {'VSA': [0, 0], 'IDX': [0, 0]}
        fam_tot = {}  # per family: agreeing queries, queries, logged right, re-derived right
        codes = Codes(files[enc])
        _MEMO.clear()  # memo keys use id() of this encoder's preprocessing functions
        preps = {'center': codes.prep('center'), 'whiten': codes.prep('whiten')}
        for M in MS:
            for V in VS:
                for r in recs(outdir, enc, M, V):
                    if rnd.random() > frac:
                        continue
                    ents = [codes.E[u] for u in r['uris']]
                    Kraw = codes.X[[codes.ix[e['name']] for e in ents]]
                    Qraw = codes.X[[codes.ix[a] for a in r['qalias']]]
                    vals = np.array(r['vals'])
                    order = np.random.default_rng(sd_('subset', 'test', M, V, r['rep'])).permutation(M)
                    for b in BPFS:
                        for acct in ('ideal', 'raw'):
                            prep, fmap, q, f = chs[enc][f'{M}|{V}|{b}|{acct}|VSA']['cfg']
                            pf = preps[prep]
                            ns = max(1, int(math.floor(f * M)))
                            d = vsa_dim(b * M, q, acct)
                            sseed = sd_('vsa', enc, 'test', M, V, r['rep'])
                            p = vsa_predict(pf(Kraw), vals, pf(Qraw), V, d, q, (enc, prep, fmap), sseed, order[:ns])
                            lp = np.array(r['pred'][f'VSA|{b}|{acct}'])
                            same = int((p == lp).sum())
                            tot['VSA'][0] += same
                            tot['VSA'][1] += len(p)
                            g = np.array(r['qgold'])
                            fam_tot.setdefault(f'VSA:{fmap}', [0, 0, 0, 0])
                            for i, x in enumerate((same, len(p), int((lp == g).sum()), int((p == g).sum()))):
                                fam_tot[f'VSA:{fmap}'][i] += x
                            prep, family, f = chs[enc][f'{M}|{V}|{b}|{acct}|IDX']['cfg']
                            pf = preps[prep]
                            ns = max(1, int(math.floor(f * M)))
                            kb = key_bits(b * M, ns, V, acct)
                            keep = order[:ns]
                            p = idx_predict(codes, pf, family, pf(Kraw[keep]), vals[keep], pf(Qraw), kb)
                            if p is None:
                                p = np.full(len(Qraw), int(np.bincount(vals[keep], minlength=V).argmax()))
                            lp = np.array(r['pred'][f'IDX|{b}|{acct}'])
                            same = int((p == lp).sum())
                            tot['IDX'][0] += same
                            tot['IDX'][1] += len(p)
                            fam_tot.setdefault(f'IDX:{family}', [0, 0, 0, 0])
                            for i, x in enumerate((same, len(p), int((lp == g).sum()), int((p == g).sum()))):
                                fam_tot[f'IDX:{family}'][i] += x
                    print(f'{enc} M={M} V={V} rep={r["rep"]}: VSA agree {tot["VSA"][0]}/{tot["VSA"][1]}, '
                          f'IDX agree {tot["IDX"][0]}/{tot["IDX"][1]}', flush=True)
        for k, (a, n, l, m) in sorted(fam_tot.items()):
            print(f'{enc} {k:10s} agree {a}/{n} = {a / n:.4f}; accuracy logged {l / n:.4f} re-derived {m / n:.4f} '
                  f'(difference {100 * (m - l) / n:+.2f} points)', flush=True)
        json.dump({'total': tot, 'by_family': fam_tot}, open(f'results/b32_rederive_{enc}.json', 'w'))


def bits(chq, che):
    bad = 0
    for path in (chq, che):
        ch = json.load(open(path))['choice']
        for k, v in ch.items():
            M, V, b, acct, arm = k.split('|')
            if arm == 'EXACT':
                continue
            M, V, b = int(M), int(V), int(b)
            B = b * M
            if arm == 'VSA':
                prep, fmap, q, f = v['cfg']
                d = vsa_dim(B, q, acct)
                used = d * (q if acct == 'raw' else HQ[q]) + 32 * math.ceil(d / BLOCK)
            else:
                prep, family, f = v['cfg']
                ns = max(1, int(math.floor(f * M)))
                kb = key_bits(B, ns, V, acct)
                vb = min(V * math.ceil(math.log2(ns + 1)), ns * math.ceil(math.log2(max(V, 2))))
                used = (ns * kb + vb) if acct == 'raw' else multiset_bits(max(kb, 0), V, ns)
            if used > B + 1e-6:
                bad += 1
                print('OVER BUDGET', path, k, used, B)
    print(f'bits audit: {bad} configurations over budget')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'score':
        score(a[1])
    elif a[0] == 'rederive':
        rederive(a[1], a[2], a[3], float(a[a.index('--frac') + 1]) if '--frac' in a else 0.1,
                 tuple(a[a.index('--encs') + 1].split(',')) if '--encs' in a else ('qwen8', 'e5'))
    elif a[0] == 'bits':
        bits(a[1], a[2])
