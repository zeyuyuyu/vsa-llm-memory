"""B49 checks (prereg/B49_panel_mechanism_competitors_prereg.md, section 9), a second angle on the verdict.

python3 b49_check.py score [--encs ...]       # independent scorer of every cell (plain loops) vs results/b49_verdict.json
python3 b49_check.py bits [--encs ...]        # bits audit of LAM (own re-implementation of the size rule), LH*, VSAL
python3 b49_check.py lam ENC [--frac 0.1]     # LAM re-derived on a share of test stores by an independent implementation
python3 b49_check.py lh ENC                   # learned map re-derived with scipy's generalized eigensolver; LHI LSH
                                              # predictions re-derived on 10% of test stores with the re-derived map
Writes results/b49_check.txt (appends) and results/b49_check_{mode}_{enc}.json.
"""
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
M_GRID, V_GRID, BPF = [256, 1024], [2, 16, 128, 1024], [1, 2, 4, 8, 16, 32, 64]
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
OUT = f'{ROOT}/results/b49_check.txt'
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}
MAXSTEP = {2: 0.9957, 4: 0.3352}


def seed_of(*parts):  # own copy of the seed rule (sha256 of the joined parts, first 15 hex digits)
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def recs(enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{ROOT}/b49_out/fam/{enc}/test/all49_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


def log(s):
    print(s, flush=True)
    open(OUT, 'a').write(s + '\n')


def score(encs):
    J = json.load(open(f'{ROOT}/results/b49_verdict.json'))
    worst, n = 0.0, 0
    for enc in encs:
        for M in M_GRID:
            for V in V_GRID:
                R = recs(enc, M, V)
                for b in BPF:
                    for a in ('ideal', 'real'):
                        tot, cnt = 0.0, 0
                        for r in R:
                            for g, x, y in zip(r['qgold'], r['pred'][f'VSA|{b}|{a}'], r['pred'][f'BEST|{b}|{a}']):
                                tot += (x == g) - (y == g)
                                cnt += 1
                        d = tot / cnt
                        worst = max(worst, abs(d - J['cells'][f'{enc}|{a}|{M}|{V}|{b}']['diff']))
                        n += 1
        # store-level bootstrap of R1 (stores resampled within each cell), a second interval
        rng = np.random.default_rng(4900)
        cells = []
        for M in M_GRID:
            for V in (2, 16):
                R = recs(enc, M, V)
                for b in (4, 8, 16):
                    per = [(sum((x == g) - (y == g) for g, x, y in zip(r['qgold'], r['pred'][f'VSA|{b}|ideal'],
                                                                         r['pred'][f'BEST|{b}|ideal'])), len(r['qgold']))
                           for r in R]
                    cells.append(np.array(per, float))
        draws = []
        for _ in range(2000):
            vals = []
            for c in cells:
                s = c[rng.integers(0, len(c), len(c))]
                vals.append(s[:, 0].sum() / s[:, 1].sum())
            draws.append(np.mean(vals))
        stat = np.mean([c[:, 0].sum() / c[:, 1].sum() for c in cells])
        log(f'score {enc}: R1 VSA - BEST {100 * stat:+.2f}, store-level bootstrap [{100 * np.quantile(draws, 0.025):+.2f}, '
            f'{100 * np.quantile(draws, 0.975):+.2f}]')
    log(f'score: {n} cells re-scored by plain loops; max |difference| from the verdict {worst:.2e}')


def lam_p(B, c, q, acct, pmax):  # own implementation of the registered size rule
    per = HQ[q] if acct == 'ideal' else q
    best = 0
    lo, hi = 0, pmax
    while lo <= hi:
        mid = (lo + hi) // 2
        bits = c * mid * per + 32 * c * math.ceil(mid / 8192) if mid > 0 else 0
        if bits <= B:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def bits(encs):
    bad, n = [], 0
    for enc in encs:
        ch = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
        n_ref = json.load(open(f'{ROOT}/b49_out/b32_data/../fam/{enc}/lh_choice.json'))
        for M in M_GRID:
            for V in V_GRID:
                for r in recs(enc, M, V):
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            B = b * r['M']
                            for f in ('LAM', 'LHC', 'LHS', 'VSAL'):
                                x = r['bits'][f'{f}|{b}|{a}']
                                n += 1
                                if x > B + 1e-6:
                                    bad.append((enc, M, V, r['rep'], f, b, a, x, B))
                            cfg = r['cfg'][f'LAM|{b}|{a}']
                            c = V - 1 if cfg[4] == 'onehot' else int(cfg[4][4:])
                            per = HQ[cfg[3]] if a == 'ideal' else cfg[3]
                            n_feat = 65536 if cfg[1] == 'rp' else None
                            if n_feat:
                                p = lam_p(B, c, cfg[3], a, n_feat)
                                want = c * p * per + 32 * c * math.ceil(p / 8192) if p > 0 else 0
                                if abs(want - r['bits'][f'LAM|{b}|{a}']) > 1e-6 and r['bits'][f'LAM|{b}|{a}'] > 0:
                                    bad.append((enc, M, V, r['rep'], 'LAM-rule', b, a, want, r['bits'][f'LAM|{b}|{a}']))
    log(f'bits: {n} logged new-family bit counts checked; over budget or off the written rule: {len(bad)}'
        + (f' e.g. {bad[:3]}' if bad else ''))


# ----------------------------------------------------------------------------------------------- LAM, independent

def lam_independent(Yk, Yq, vals, V, enc, prep, feat, alpha, q, target, B, acct, Ucols, seed):
    M = len(vals)
    c = V - 1 if target == 'onehot' else int(target[4:])
    p = lam_p(B, c, q, acct, 65536 if feat == 'rp' else Yk.shape[1])
    if p <= 0:
        return None
    if target == 'onehot':
        T = np.eye(V)[:, 1:] - 1.0 / V
        D = np.vstack([-np.ones((1, V - 1)), np.eye(V - 1)]).T  # c x V
    else:
        r = np.random.default_rng(seed_of('LAMcode', V, c))
        T = np.where(r.random((V, c)) < 0.5, -1.0, 1.0)
        D = T.T
    Y = T[vals]
    n = Yk.shape[1]

    def F(X, j0, w):
        if feat == 'rp':
            G = np.random.default_rng(seed_of('LAMG', enc, prep, j0)).standard_normal((n, 8192), dtype=np.float32)
            return (X.astype(np.float32) @ (G[:, :w] / np.float32(math.sqrt(n)))).astype(np.float64)
        return (X - Ucols[0]) @ Ucols[1][:, j0:j0 + w].astype(np.float64)
    Kmat = np.zeros((M, M))
    for j0 in range(0, p, 8192):
        w = min(8192, p - j0)
        f = F(Yk, j0, w)
        Kmat += f @ f.T
    lam = alpha * np.trace(Kmat) / M
    from scipy.linalg import solve
    A = solve(Kmat + lam * np.eye(M), Y, assume_a='pos')
    S = np.zeros((Yq.shape[0], c))
    for j0 in range(0, p, 8192):
        w = min(8192, p - j0)
        Wb = F(Yk, j0, w).T @ A
        Wq = np.empty_like(Wb)
        rng = np.random.default_rng(seed_of('LAMQ', seed, (feat, alpha, q, target), B // M, acct, j0))
        for k in range(c):
            x = Wb[:, k]
            if q == 1:
                s = np.sign(x)
                z = s == 0
                if z.any():
                    s[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
                Wq[:, k] = s * (np.abs(x).mean() or 1.0)
            else:
                x32 = x.astype(np.float32)
                sd = float(x32.std()) or 1.0
                step = MAXSTEP[q] * sd
                L = 2 ** q
                Wq[:, k] = (np.clip(np.floor(x32 / step), -L // 2, L // 2 - 1) + 0.5) * step
        S += F(Yq, j0, w) @ Wq
    return (S @ D).argmax(1)


def lam(enc, frac):
    sys.path.insert(0, ROOT)
    os.chdir(f'{ROOT}/b49_out')
    import b32_common as C
    import b49_fam as Fm
    pool = C.Pool(Fm.ENC_TAG[enc])
    Tr = pool.train_matrix()
    preps = {p: C.Prep(p, Tr) for p in ('center', 'whiten')}
    pcs = {}
    for p in preps:
        T = preps[p](Tr)
        mu = T.mean(0)
        w, U = np.linalg.eigh(np.cov((T - mu).T))
        pcs[p] = (mu.astype(np.float64), U[:, np.argsort(-w)])
    rng = np.random.default_rng(49)
    agree, tot, per = 0, 0, {}
    for M in M_GRID:
        for V in V_GRID:
            R = recs(enc, M, V)
            pick = sorted(set([int(rng.integers(0, len(R)))] + [i for i in range(len(R)) if rng.random() < frac]))
            for i in pick:
                r = R[i]
                st = C.make_store(pool, 'test', M, V, r['rep'])
                assert st['qgold'].tolist() == r['qgold']
                seed = C.seed_of('b49', enc, 'test', M, V, r['rep'])
                for b in BPF:
                    for a in ('ideal', 'real'):
                        cfg = r['cfg'][f'LAM|{b}|{a}']
                        prep = cfg[0]
                        P = preps[prep]
                        pr = lam_independent(P(st['keys']).astype(np.float64), P(st['queries']).astype(np.float64),
                                             st['vals'], V, enc, prep, cfg[1], cfg[2], cfg[3], cfg[4], b * M, a,
                                             pcs[prep], seed)
                        if pr is None:
                            continue
                        x = np.array(r['pred'][f'LAM|{b}|{a}'])
                        k = int((pr == x).sum())
                        agree += k
                        tot += len(x)
                        key = f'{M}|{V}|{b}|{a}'
                        per[key] = per.get(key, 0) + k / len(x) / len(pick)
    out = {'agreement': agree / max(tot, 1), 'n_queries': tot, 'per_cell': per}
    json.dump(out, open(f'{ROOT}/results/b49_check_lam_{enc}.json', 'w'), indent=1)
    worst = min(per.items(), key=lambda kv: kv[1]) if per else None
    log(f'lam {enc}: independent LAM re-derivation agreement {agree / max(tot, 1):.4f} on {tot} query predictions '
        f'(lowest cell {worst})')


def lh(enc):
    sys.path.insert(0, ROOT)
    os.chdir(f'{ROOT}/b49_out')
    import b32_common as C
    import b49_common as B
    import b49_fam as Fm
    from scipy.linalg import eigh
    pool = C.Pool(Fm.ENC_TAG[enc])
    ch = json.load(open(f'fam/{enc}/lh_choice.json'))
    lp = B.LearnedPrep(pool.X, pool, ch['s'], ch['r'])
    # independent: scipy generalized eigensolver on the same definition
    X = pool.X.astype(np.float64)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    tr = pool.by_split['train']
    rows = [pool.idx[e['name']] for e in tr] + [pool.idx[a] for e in tr for a in e['aliases']]
    T = Xn[rows]
    mu = T.mean(0)
    St = np.cov((T - mu).T)
    D = np.array([Xn[pool.idx[a]] - Xn[pool.idx[e['name']]] for e in tr for a in e['aliases']])
    Sw = D.T @ D / len(D)
    Sw += ch['s'] * float(np.trace(Sw) / len(Sw)) * np.eye(len(Sw))
    n = len(Sw)
    w, U = eigh(St, Sw, subset_by_index=[n - ch['r'], n - 1])
    U = U[:, ::-1]
    A0 = lp.A.astype(np.float64)
    cos = np.abs((A0 * U).sum(0)) / (np.linalg.norm(A0, axis=0) * np.linalg.norm(U, axis=0))
    sgn = np.sign((A0 * U).sum(0))
    U = U * sgn
    log(f'lh {enc}: learned map re-derived (scipy.linalg.eigh generalized): |cos| with the run\'s directions median '
        f'{np.median(cos):.6f}, min {cos.min():.6f} (r = {ch["r"]}, s = {ch["s"]})')

    def mapped(Z):
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        Y = (Zn - mu) @ U
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)
    rng = np.random.default_rng(491)
    agree, tot = 0, 0
    for M in M_GRID:
        for V in V_GRID:
            R = recs(enc, M, V)
            pick = sorted(set([int(rng.integers(0, len(R)))] + [i for i in range(len(R)) if rng.random() < 0.1]))
            for i in pick:
                r = R[i]
                st = C.make_store(pool, 'test', M, V, r['rep'])
                for b in BPF:
                    for a in ('ideal', 'real'):
                        cfg = r['cfg'][f'LHI|{b}|{a}']
                        if cfg[1] != 'lsh':
                            continue
                        order = np.random.default_rng(seed_of('subset', 'test', M, V, r['rep'])).permutation(M)
                        ns = max(1, int(math.floor(cfg[2] * M)))
                        keep = order[:ns]
                        kb = C.key_bits(b * M, ns, V, 'ideal' if a == 'ideal' else 'raw')
                        if kb < 1:
                            continue
                        G = np.random.default_rng(seed_of('LSH', 0, kb)).standard_normal((ch['r'], kb), dtype=np.float32)
                        Kc = np.where(mapped(st['keys'][keep].astype(np.float64)) @ G >= 0, 1.0, -1.0)
                        Qp = mapped(st['queries'].astype(np.float64)) @ G
                        pr = st['vals'][keep][(Qp @ Kc.T).argmax(1)]
                        x = np.array(r['pred'][f'LHI|{b}|{a}'])
                        agree += int((pr == x).sum())
                        tot += len(x)
    log(f'lh {enc}: LHI LSH predictions re-derived with the re-derived map: agreement {agree / max(tot, 1):.4f} on {tot}')
    json.dump({'map_cos_median': float(np.median(cos)), 'map_cos_min': float(cos.min()),
               'lhi_lsh_agreement': agree / max(tot, 1), 'n': tot}, open(f'{ROOT}/results/b49_check_lh_{enc}.json', 'w'))


if __name__ == '__main__':
    a = sys.argv[1:]
    encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
    frac = float(a[a.index('--frac') + 1]) if '--frac' in a else 0.1
    if a[0] == 'score':
        score(encs)
    elif a[0] == 'bits':
        bits(encs)
    elif a[0] == 'lam':
        lam(a[1], frac)
    elif a[0] == 'lh':
        lh(a[1])
