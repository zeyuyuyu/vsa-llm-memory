"""B57 checks (prereg/B57_pca_store_maps_online_prereg.md, 'Checks').

python3 b57_check.py score | bits | rls | handread
python3 b57_check.py rederive ENC
Appends to results/b57_check.txt (handread: results/b57_handread.txt).
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b56_check as K56  # noqa: E402  (own seed rule, codes, quantizer, size rules: independent of the run code)

PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
MAP_ENCS = ['qwen8', 'e5', 'minilm']
NDIM = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
OUT = f'{ROOT}/results/b57_check.txt'
seed_of = K56.seed_of


def log(s):
    print(s, flush=True)
    open(OUT, 'a').write(s + '\n')


def recs(enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{ROOT}/b57_out/fam/{enc}/all57_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


def score():
    J = json.load(open(f'{ROOT}/results/b57_verdict.json'))
    worst = wreg = wmap = 0.0
    n = 0
    for enc in PANEL:
        reg = {}
        for M in M_GRID:
            for V in VN:
                R = recs(enc, M, V)
                for b in BPF:
                    for a in ('ideal', 'real'):
                        arms = ['VSAP', 'NLMS', 'RLS', 'LAM', 'BEST', 'VSAS', 'BEST36']
                        if enc in MAP_ENCS:
                            arms += [f'{x}m{m}' for x in ('VSAS', 'LAM', 'BEST') for m in (1, 2, 3)]
                        for x in arms:
                            v = K56.acc_loop(R, f'{x}|{b}|{a}')
                            if x in J['cells'][f'{enc}|{a}|{M}|{V}|{b}']:
                                worst = max(worst, abs(v - J['cells'][f'{enc}|{a}|{M}|{V}|{b}'][x]))
                                n += 1
                            if a == 'ideal' and b in (4, 8, 16):
                                reg.setdefault(x, []).append(v)
        for x, y in (('VSAP', 'LAM'), ('VSAP', 'BEST'), ('NLMS', 'LAM'), ('RLS', 'LAM')):
            d = float(np.mean(reg[x])) - float(np.mean(reg[y]))
            wreg = max(wreg, abs(d - J['regions'][f'{enc}|ideal|R1|{x}-{y}']['diff']))
        if enc in MAP_ENCS:
            for y in ('LAM', 'BEST'):
                for m in (1, 2, 3):
                    d = float(np.mean(reg[f'VSASm{m}'])) - float(np.mean(reg[f'{y}m{m}']))
                    wmap = max(wmap, abs(d - J['maps'][f'{enc}|ideal|R1|VSAS-{y}']['per_map'][m]))
    log(f'- score: {n} cell accuracies re-scored by plain loops; max |difference| {worst:.2e}; R1 regions {wreg:.2e}; '
        f'per-map R1 points {wmap:.2e}')


def bits():
    over = notmax = n = 0
    for enc in PANEL:
        nd = NDIM[enc]
        for M in M_GRID:
            for V in VN:
                for r in recs(enc, M, V):
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            B_ = b * M
                            for x in ('VSAPL', 'VSAPT'):
                                d, q = r['b57'][f'{x}|{b}|{a}'][:2]
                                n += 1
                                over += K56.vbits(d, q, a) > B_ + 1e-9 or d > nd
                                notmax += d < nd and K56.vbits(d + 1, q, a) <= B_
                            prep, ft, mu, q, bt = r['b57'][f'NLMS|{b}|{a}']
                            n += 1
                            over += bt > B_ + 1e-9
                            rc = r['b57'][f'RLS|{b}|{a}']
                            n += 1
                            over += rc[-1] > B_ + 1e-9
                            if enc in MAP_ENCS:
                                for m in (1, 2, 3):
                                    for arm in ('LM', 'RT'):
                                        d, q = r['b57'][f'{arm}m{m}|{b}|{a}'][:2]
                                        n += 1
                                        over += K56.vbits(d, q, a) > B_ + 1e-9
                                        notmax += K56.vbits(d + 1, q, a) <= B_
                                    p, q, bt = r['b57'][f'LRm{m}|{b}|{a}'][:3]
                                    n += 1
                                    over += K56.lbits(p, V - 1, q, a) > B_ + 1e-9
    log(f'- bits: {n} (store, cell, arm) configurations; over budget {over}; not maximal under the cap {notmax}')


def rls():
    import b32_common as C
    import b49_common as B
    import b53_run as R53
    import b57_run as R57
    worst = 0.0
    agree = tot = 0
    for enc in ('e5', 'minilm'):
        ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice']
        pool, preps = R53.setup(enc)
        Tr = pool.train_matrix()
        for V in VN:
            st = C.make_store(pool, 'test', 256, V, 0)
            for b in (4, 16):
                prep, ft, a, q, t = ch[f'256|{V}|{b}|ideal|RLS']['cfg']
                lf = B.LamFeat(enc, prep, preps[prep](Tr))
                p = B.lam_p(b * 256, V - 1, q, 'ideal', lf.pmax(ft))
                P = preps[prep]
                Fk = lf.feats(P(st['keys']), 'pca', 0, p).astype(np.float64)
                Tt, D = B.value_targets(V, 'onehot', 0)
                Y = Tt[st['vals']]
                K = Fk @ Fk.T
                lam = a * float(np.trace(K)) / 256
                Wb = Fk.T @ np.linalg.solve(K + lam * np.eye(256), Y)
                Wr = R57.rls_recursive(Fk, Y, lam)
                worst = max(worst, float(np.abs(Wr - Wb).max() / np.abs(Wb).max()))
                Fq = lf.feats(P(st['queries']), 'pca', 0, p).astype(np.float64)
                agree += int(((Fq @ Wr @ D).argmax(1) == (Fq @ Wb @ D).argmax(1)).sum())
                tot += Fq.shape[0]
    log(f'- rls: genuine recursion vs batch ridge (e5, MiniLM; M = 256; V 2 / 16; 4 / 16 bpf; chosen pca configurations): '
        f'max relative |W difference| {worst:.2e}; unquantized read agreement {agree / tot:.5f} over {tot} queries')


def rederive(enc):
    import b32_common as C
    import b53_run as R53
    ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice']
    pool, preps = R53.setup(enc)
    Tr = pool.train_matrix()
    basis = {}
    ag = {'VSAPL': [0, 0], 'VSAPT': [0, 0]}
    worst = {'VSAPL': 1.0, 'VSAPT': 1.0}
    for M in M_GRID:
        for V in VN:
            R = recs(enc, M, V)
            rep = 1 if M == 256 else 0
            st = C.make_store(pool, 'test', M, V, rep)
            rec = [r for r in R if r['rep'] == rep][0]
            sseed = seed_of('vsa', enc, 'test', M, V, rep)
            order = np.random.default_rng(seed_of('subset', 'test', M, V, rep)).permutation(M)
            for b in (4, 16, 64):
                for arm, nm in (('PLM', 'VSAPL'), ('PRT', 'VSAPT')):
                    q, x = ch[f'{M}|{V}|{b}|ideal|{arm}']['cfg']
                    d, q_, ns, prep, fmap = rec['b57'][f'{nm}|{b}|ideal'][:5]
                    P = preps[prep]
                    if prep not in basis and os.environ.get('B57_LAMFEAT_BASIS'):  # diagnostic: the run's own basis
                        import b49_common as B
                        lf = B.LamFeat(enc, prep, P(Tr))
                        basis[prep] = (lf.mu.astype(np.float64), lf.U.astype(np.float64))
                    if prep not in basis:  # own principal basis (float64)
                        Y = P(Tr).astype(np.float64)
                        mu = Y.mean(0)
                        w, U = np.linalg.eigh(np.cov((Y - mu).T))
                        basis[prep] = (mu, U[:, np.argsort(-w)])
                    mu, U = basis[prep]

                    def feats(Y):
                        Z = (P(Y).astype(np.float64) - mu) @ U[:, :d]
                        return Z if fmap == 'plin' else np.where(Z >= 0, 1.0, -1.0)
                    Fk, Fq = feats(st['keys']), feats(st['queries'])
                    Cv = K56.own_codes(enc, prep, 'lin' if fmap == 'plin' else 'sgn', V, d)
                    idx = order[:ns]
                    vals = st['vals']
                    if arm == 'PLM':
                        S = (Fk[idx] * Cv[vals[idx]]).sum(0)
                        z = K56.own_block_quant(S, q, ('Q', sseed, d, q, ns)) * Cv[0]
                        sl = [slice(c, d, 16) for c in range(16)] if V == 16 else [slice(0, d)]
                        if math.isinf(x):
                            m = [Fq[:, s] @ z[s] for s in sl]
                        else:
                            T = 1024 if V == 2 else 512
                            rows = np.sort(np.random.default_rng(seed_of('b56T', enc, prep, fmap, V)).choice(len(Tr), T,
                                                                                                         replace=False))
                            F = feats(Tr[rows])
                            m = []
                            for s in sl:
                                Fc = F[:, s]
                                lam = x * float((Fc ** 2).sum()) / (T * Fc.shape[1])
                                Sig = Fc.T @ Fc / T
                                m.append(Fq[:, s] @ np.linalg.solve(Sig + lam * np.eye(Fc.shape[1]), z[s]))
                        m = np.stack(m, 1)
                        if V == 16:
                            from scipy.linalg import hadamard
                            pred = (m @ hadamard(16).T).argmax(1)
                        else:
                            pred = np.where(m[:, 0] >= 0, 0, 1)
                    else:
                        X, y = Fk[idx], vals[idx]
                        E, eta = int(x.split(':')[0]), float(x.split(':')[1])
                        S = (X * Cv[y]).sum(0)
                        for ep in range(E):
                            for b0 in range(0, ns, 32):
                                S0 = S.copy()
                                for i in range(b0, min(ns, b0 + 32)):
                                    vh = int(np.argmax([(X[i] * S0 * Cv[v]).sum() for v in range(V)]))
                                    if vh != y[i]:
                                        S += eta * X[i] * (Cv[y[i]] - Cv[vh])
                        Sq = K56.own_block_quant(S, q, ('Q56', sseed, d, q, ns, E, eta))
                        pred = (Fq @ (Cv * Sq).T).argmax(1)
                    got = np.array(rec['pred'][f'{nm}|{b}|ideal'])
                    a_ = float((pred == got).mean())
                    ag[nm][0] += int((pred == got).sum())
                    ag[nm][1] += len(got)
                    worst[nm] = min(worst[nm], a_)
                    if a_ < 0.99:
                        log(f'  cell below 0.99: {enc} {nm} M={M} V={V} bpf={b} cfg={rec["b57"][f"{nm}|{b}|ideal"]} '
                            f'read={q, x} agreement {a_:.4f}')
    for nm in ag:
        log(f'- rederive {enc} {nm}{" (diagnostic: LamFeat basis)" if os.environ.get("B57_LAMFEAT_BASIS") else ""}: float64 own basis / codes / quantizer / read, 1 test store per (M, V), bpf 4 / 16 / 64: '
            f'agreement {ag[nm][0] / ag[nm][1]:.4f} over {ag[nm][1]} queries; worst cell {worst[nm]:.4f}')


def handread():
    ents = {e['uri']: e for e in json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']}
    J = json.load(open(f'{ROOT}/results/b57_verdict.json'))
    L = ['# B57 hand reading: per encoder the R1 cell (ideal) where |VSAP - VSAS| is largest; 10 queries only VSAP answers',
         '# right and 10 only VSAS answers right (alias -> stored label, gold value). Notes at the end.', '']
    for enc in ('qwen8', 'e5'):
        cells = [(abs(J['cells'][f'{enc}|ideal|{M}|{V}|{b}']['VSAP'] - J['cells'][f'{enc}|ideal|{M}|{V}|{b}']['VSAS']),
                  J['cells'][f'{enc}|ideal|{M}|{V}|{b}']['VSAP'] - J['cells'][f'{enc}|ideal|{M}|{V}|{b}']['VSAS'], M, V, b)
                 for M in M_GRID for V in VN for b in (4, 8, 16)]
        _, dd, M, V, b = max(cells)
        R = recs(enc, M, V)
        rng = np.random.default_rng(57)
        L.append(f'## {enc}: M = {M}, V = {V}, {b} bits per fact; VSAP - VSAS = {100 * dd:+.2f} points; VSAP '
                 f'{R[0]["b57"][f"VSAP|{b}|ideal"]}')
        for want, nm in (((1, 0), 'only VSAP right'), ((0, 1), 'only VSAS right')):
            pool = []
            for r in R:
                for i, (g, p, y) in enumerate(zip(r['qgold'], r['pred'][f'VSAP|{b}|ideal'], r['pred'][f'VSAS|{b}|ideal'])):
                    if (int(p == g), int(y == g)) == want:
                        pool.append((r['qalias'][i], ents[r['uris'][r['qent'][i]]]['name'], g, p, y))
            L.append(f'### {nm}: {len(pool)} queries; 10 drawn')
            for k in rng.choice(len(pool), min(10, len(pool)), replace=False):
                a, lab, g, p, y = pool[k]
                L.append(f'  {a!r} -> {lab!r} gold {g} VSAP {p} VSAS {y}')
        L.append('')
    open(f'{ROOT}/results/b57_handread.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    m = sys.argv[1]
    {'score': score, 'bits': bits, 'rls': rls, 'handread': handread}.get(m, lambda: rederive(sys.argv[2]))()
