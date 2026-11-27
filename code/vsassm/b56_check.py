"""B56 checks (prereg/B56_decoders_retraining_prereg.md, 'Checks'), a second angle on the verdict.

python3 b56_check.py score            # plain-loop scorer vs results/b56_verdict.json
python3 b56_check.py bits             # own size rules for every logged d / p
python3 b56_check.py repro            # VSAL at alpha = inf (and B53's q) vs B53's VSA test predictions
python3 b56_check.py rederive ENC     # float64, own features / codes / quantizer / decoder / retraining, 1 store per (M, V)
python3 b56_check.py handread         # Qwen3-8B, e5: the R1 cell where VSAS - LAM is lowest, 10 + 10 queries
Appends to results/b56_check.txt (handread: results/b56_handread.txt).
"""
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}
MAXSTEP = {2: 0.9957, 4: 0.3352}
CHK = 8192
OUT = f'{ROOT}/results/b56_check.txt'


def log(s):
    print(s, flush=True)
    open(OUT, 'a').write(s + '\n')


def seed_of(*parts):  # own copy of the registered seed rule
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def recs(enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{ROOT}/b56_out/fam/{enc}/all56_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


def acc_loop(R, arm):
    ok = n = 0
    for r in R:
        for g, p in zip(r['qgold'], r['pred'][arm]):
            ok += int(p == g)
            n += 1
    return ok / n


def score():
    J = json.load(open(f'{ROOT}/results/b56_verdict.json'))
    worst = wreg = 0.0
    n = 0
    for enc in PANEL:
        reg = {}
        for M in M_GRID:
            for V in VN:
                R = recs(enc, M, V)
                for b in BPF:
                    for a in ('ideal', 'real'):
                        for x in ('VSAS', 'LAM', 'BEST', 'VSA', 'VSAL', 'VSAT', 'VSAW', 'LAMRF'):
                            v = acc_loop(R, f'{x}|{b}|{a}')
                            worst = max(worst, abs(v - J['cells'][f'{enc}|{a}|{M}|{V}|{b}'][x]))
                            n += 1
                            if a == 'ideal' and b in (4, 8, 16):
                                reg.setdefault(x, []).append(v)
        for y in ('LAM', 'BEST'):
            d = float(np.mean(reg['VSAS'])) - float(np.mean(reg[y]))
            wreg = max(wreg, abs(d - J['regions'][f'{enc}|ideal|R1|VSAS-{y}']['diff']))
    log(f'- score: {n} cell accuracies re-scored by plain loops; max |difference| from the verdict {worst:.2e}; '
        f'R1 VSAS - LAM / BEST max |difference| {wreg:.2e}')


def vbits(d, q, acct):
    per = HQ[q] if acct == 'ideal' else q
    return d * per + 32 * math.ceil(d / CHK) if d > 0 else 0


def lbits(p, c, q, acct):
    per = HQ[q] if acct == 'ideal' else q
    return c * p * per + 32 * c * math.ceil(p / CHK) if p > 0 else 0


def bits():
    over = notmax = n = 0
    for enc in PANEL:
        for M in M_GRID:
            for V in VN:
                for r in recs(enc, M, V):
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            B = b * M
                            for x in ('VSAL', 'VSAT', 'VSAW'):
                                d, q, ns = r['b56'][f'{x}|{b}|{a}']
                                n += 1
                                over += vbits(d, q, a) > B + 1e-9
                                notmax += vbits(d + 1, q, a) <= B
                            p, q, bt = r['b56'][f'LAMRF|{b}|{a}']
                            n += 1
                            over += lbits(p, V - 1, q, a) > B + 1e-9 or abs(lbits(p, V - 1, q, a) - bt) > 1e-6
                            notmax += p < 65536 and lbits(p + 1, V - 1, q, a) <= B
    log(f'- bits: {n} (store, cell, arm) configurations; over budget or logged bits inconsistent {over}; not maximal '
        f'{notmax}')


def repro():
    for enc in PANEL:
        agree = tot = cells = 0
        for M in M_GRID:
            for V in VN:
                R = recs(enc, M, V)
                for b in BPF:
                    for a in ('ideal', 'real'):
                        c = R[0]['cfg'][f'VSAL|{b}|{a}']
                        if not (c[1] == float('inf') and c[0] == R[0]['cfg'][f'VSA|{b}|{a}'][2]):
                            continue
                        cells += 1
                        for r in R:
                            x, y = np.array(r['pred'][f'VSAL|{b}|{a}']), np.array(r['pred'][f'VSA|{b}|{a}'])
                            agree += int((x == y).sum())
                            tot += len(x)
        log(f'- repro {enc}: VSAL with alpha = inf at B53\'s q in {cells} cell-accountings: agreement with B53\'s VSA '
            f'{agree / max(tot, 1):.5f} ({tot - agree} of {tot} queries differ)')


# ----------------------------------------------------------------------------------------------- re-derivation

def own_codes(enc, prep, fmap, V, d):
    from scipy.linalg import hadamard
    key = (enc, prep, 'sgn' if fmap.startswith('tern') else fmap)
    rows = []
    for j0 in range(0, d, CHK):
        rr = np.random.default_rng(seed_of('C', *key, j0, CHK))
        rows.append(np.where(rr.random((1024, CHK), dtype=np.float32) < 0.5, -1.0, 1.0)[0])
    r = np.concatenate(rows)[:d].astype(np.float64)
    if V == 2:
        return np.stack([r, -r])
    H = hadamard(16).astype(np.float64)
    return np.stack([r * H[v, np.arange(d) % 16] for v in range(16)])


def own_feats(enc, prep, fmap, Y, d):
    key = (enc, prep, 'sgn' if fmap.startswith('tern') else fmap)
    K = 1 if fmap in ('lin', 'sgn') or fmap.startswith('tern') else int(fmap[3:])
    out = []
    for j0 in range(0, d, CHK):
        f = None
        for m in range(K):
            G = np.random.default_rng(seed_of('G', *key, m, j0, CHK)).standard_normal((Y.shape[1], CHK),
                                                                                     dtype=np.float32)
            P = Y.astype(np.float64) @ G.astype(np.float64)
            if fmap == 'lin':
                g = P
            elif fmap.startswith('tern'):
                from scipy.stats import norm
                th = norm.ppf(1 - {'tern4': 0.25, 'tern16': 1 / 16, 'tern64': 1 / 64}[fmap] / 2)
                g = np.where(np.abs(P) >= th, np.sign(P), 0.0)
            else:
                g = np.where(P >= 0, 1.0, -1.0)
            f = g if f is None else f * g
        out.append(f)
    return np.concatenate(out, 1)[:, :d]


def own_quant(S, q, rng):
    if q == 1:
        o = np.sign(S)
        z = o == 0
        if z.any():
            o[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
        return o
    sd = S.std() or 1.0
    st = MAXSTEP[q] * sd
    L = 2 ** q
    return (np.clip(np.floor(S / st), -L // 2, L // 2 - 1) + 0.5) * st


def own_block_quant(S, q, seedparts):
    return np.concatenate([own_quant(S[j0:j0 + CHK].astype(np.float32).astype(np.float64), q,
                                     np.random.default_rng(seed_of(*seedparts, j0))) for j0 in range(0, len(S), CHK)])


def rederive(enc):
    sys.path.insert(0, ROOT)
    import b32_common as C
    import b53_run as R53
    ch = json.load(open(f'{ROOT}/results/b56_choice_{enc}.json'))['choice']
    ch53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice']
    pool, preps = R53.setup(enc)
    Tr = pool.train_matrix()
    agree = {'VSAL': [0, 0], 'VSAT': [0, 0]}
    worst = {'VSAL': 1.0, 'VSAT': 1.0}
    for M in M_GRID:
        for V in VN:
            R = recs(enc, M, V)
            rep = 1 if M == 256 else 0
            st = C.make_store(pool, 'test', M, V, rep)
            rec = [r for r in R if r['rep'] == rep][0]
            assert rec['qgold'] == st['qgold'].tolist()
            sseed = seed_of('vsa', enc, 'test', M, V, rep)
            order = np.random.default_rng(seed_of('subset', 'test', M, V, rep)).permutation(M)
            for b in (4, 16, 64):
                a3 = 'ideal'
                prep, fmap, q53, f = ch53[f'{M}|{V}|{b}|{a3}|VSA']['cfg']
                ns = max(1, int(math.floor(f * M)))
                P = preps[prep]
                for arm, nm in (('LM', 'VSAL'), ('RT', 'VSAT')):
                    q, x = ch[f'{M}|{V}|{b}|{a3}|{arm}']['cfg']
                    d = rec['b56'][f'{nm}|{b}|ideal'][0]
                    Fk = own_feats(enc, prep, fmap, P(st['keys']), d)
                    Fq = own_feats(enc, prep, fmap, P(st['queries']), d)
                    Cv = own_codes(enc, prep, fmap, V, d)
                    idx = order[:ns]
                    vals = st['vals']
                    if arm == 'LM':
                        S = (Fk[idx] * Cv[vals[idx]]).sum(0)
                        Sq = own_block_quant(S, q, ('Q', sseed, d, q, ns))
                        z = Sq * Cv[0]
                        if math.isinf(x):
                            m = [Fq[:, c::16] @ z[c::16] for c in range(16)] if V == 16 else [Fq @ z]
                        else:
                            T = 1024 if V == 2 else 512
                            rows = np.sort(np.random.default_rng(seed_of('b56T', enc, prep, fmap, V)).choice(
                                len(Tr), T, replace=False))
                            F = own_feats(enc, prep, fmap, P(Tr[rows]), d)
                            m = []
                            for c in (range(16) if V == 16 else [None]):
                                sl = slice(c, d, 16) if c is not None else slice(0, d)
                                Fc = F[:, sl]
                                lam = x * float((Fc ** 2).sum()) / (T * Fc.shape[1])
                                # dual-free check: solve (Sigma + lam I) y = z in the primal when small, else via
                                # the push-through identity with an explicit solve of the T x T system
                                if Fc.shape[1] <= 2048:
                                    Sig = Fc.T @ Fc / T
                                    m.append(Fq[:, sl] @ np.linalg.solve(Sig + lam * np.eye(Fc.shape[1]), z[sl]))
                                else:
                                    y = (z[sl] - Fc.T @ np.linalg.solve(T * lam * np.eye(T) + Fc @ Fc.T, Fc @ z[sl])) / lam
                                    m.append(Fq[:, sl] @ y)
                        m = np.stack(m, 1)
                        if V == 16:
                            from scipy.linalg import hadamard
                            pred = (m @ hadamard(16).T).argmax(1)
                        else:
                            pred = np.where(m[:, 0] >= 0, 0, 1)
                    else:
                        X = Fk[idx]
                        y = vals[idx]
                        E, eta = int(x.split(':')[0]), float(x.split(':')[1])
                        S = (X * Cv[y]).sum(0)
                        for ep in range(E):
                            for b0 in range(0, ns, 32):
                                S0 = S.copy()  # every fact of a mini-batch is read with the bundle at the batch start
                                for i in range(b0, min(ns, b0 + 32)):
                                    sc = [(X[i] * S0 * Cv[v]).sum() for v in range(V)]
                                    vh = int(np.argmax(sc))
                                    if vh != y[i]:
                                        S += eta * X[i] * (Cv[y[i]] - Cv[vh])
                        Sq = own_block_quant(S, q, ('Q56', sseed, d, q, ns, E, eta))
                        pred = (Fq @ (Cv * Sq).T).argmax(1)
                    got = np.array(rec['pred'][f'{nm}|{b}|ideal'])
                    ag = float((pred == got).mean())
                    agree[nm][0] += int((pred == got).sum())
                    agree[nm][1] += len(got)
                    worst[nm] = min(worst[nm], ag)
    for nm in agree:
        log(f'- rederive {enc} {nm}: float64 own implementation, 1 test store per (M, V), bpf 4 / 16 / 64 (ideal): '
            f'agreement {agree[nm][0] / agree[nm][1]:.4f} over {agree[nm][1]} queries; worst cell {worst[nm]:.4f}')


def handread():
    ents = {e['uri']: e for e in json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']}
    J = json.load(open(f'{ROOT}/results/b56_verdict.json'))
    L = ['# B56 hand reading: per encoder the R1 cell (ideal) where VSAS - LAM is lowest; 10 queries only VSAS answers',
         '# right and 10 only LAM answers right (alias -> stored label, gold value). Notes below each block.', '']
    for enc in ('qwen8', 'e5'):
        cells = [(J['cells'][f'{enc}|ideal|{M}|{V}|{b}']['VSAS'] - J['cells'][f'{enc}|ideal|{M}|{V}|{b}']['LAM'], M, V, b)
                 for M in M_GRID for V in VN for b in (4, 8, 16)]
        dd, M, V, b = min(cells)
        R = recs(enc, M, V)
        rng = np.random.default_rng(56)
        L.append(f'## {enc}: M = {M}, V = {V}, {b} bits per fact; VSAS - LAM = {100 * dd:+.2f} points; '
                 f'VSAS = {R[0]["cfg"][f"VSAS|{b}|ideal"]}')
        for want, nm in (((1, 0), 'only VSAS right'), ((0, 1), 'only LAM right')):
            pool = []
            for r in R:
                for i, (g, p, y) in enumerate(zip(r['qgold'], r['pred'][f'VSAS|{b}|ideal'], r['pred'][f'LAM|{b}|ideal'])):
                    if (int(p == g), int(y == g)) == want:
                        pool.append((r['qalias'][i], ents[r['uris'][r['qent'][i]]]['name'], g, p, y))
            L.append(f'### {nm}: {len(pool)} queries in the cell; 10 drawn')
            for k in rng.choice(len(pool), min(10, len(pool)), replace=False):
                a, lab, g, p, y = pool[k]
                L.append(f'  {a!r} -> {lab!r} gold {g} VSAS {p} LAM {y}')
        L.append('')
    open(f'{ROOT}/results/b56_handread.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def ties():
    """Where VSAL at alpha = inf differs from B53's VSA (sgn features, q = 2), recompute the scores on the quantizer's
    lattice (S_hat = (k + 1/2) step: score / step * 2 is an integer) and test that both predictions tie exactly."""
    sys.path.insert(0, ROOT)
    import b32_common as C
    import b53_run as R53
    cases = [('e5', 256, 16, 1, 'real'), ('mistral', 256, 2, 1, 'real'), ('mistral', 256, 2, 2, 'ideal'),
             ('mistral', 256, 2, 2, 'real'), ('mistral', 256, 2, 64, 'real')]
    for enc, M, V, b, a in cases:
        pool, preps = R53.setup(enc)
        R = recs(enc, M, V)
        nd = nt = 0
        for rec in R:
            x, y = np.array(rec['pred'][f'VSAL|{b}|{a}']), np.array(rec['pred'][f'VSA|{b}|{a}'])
            ii = np.nonzero(x != y)[0]
            if not len(ii):
                continue
            prep, fmap, q, f = rec['cfg'][f'VSA|{b}|{a}']
            d, q_, ns = rec['vsa_d'][f'VSA|{b}|{a}']
            st = C.make_store(pool, 'test', M, V, rec['rep'])
            P = preps[prep]
            Fk = own_feats(enc, prep, fmap, P(st['keys']), d)
            Fq = own_feats(enc, prep, fmap, P(st['queries'][ii]), d)
            Cv = own_codes(enc, prep, fmap, V, d)
            order = np.random.default_rng(seed_of('subset', 'test', M, V, rec['rep'])).permutation(M)[:ns]
            S = (Fk[order] * Cv[st['vals'][order]]).sum(0)
            sseed = seed_of('vsa', enc, 'test', M, V, rec['rep'])
            lat = []
            for j0 in range(0, d, CHK):
                blk = S[j0:j0 + CHK].astype(np.float32).astype(np.float64)
                st_ = MAXSTEP[q] * (blk.std() or 1.0)
                lat.append(2 * np.clip(np.floor(blk / st_), -2, 1) + 1)  # odd integers: 2 (k + 1/2)
            assert len(lat) == 1, 'one block expected'
            sc = Fq @ (Cv * lat[0]).T  # integer lattice scores
            nd += len(ii)
            nt += int(np.sum(sc[np.arange(len(ii)), x[ii]] == sc[np.arange(len(ii)), y[ii]]))
        log(f'- ties {enc} M={M} V={V} {b} bpf {a}: {nd} differing queries, {nt} of them exact ties on the quantizer lattice')


if __name__ == '__main__':
    m = sys.argv[1]
    {'score': score, 'bits': bits, 'repro': repro, 'handread': handread, 'ties': ties}.get(m, lambda: rederive(sys.argv[2]))()
