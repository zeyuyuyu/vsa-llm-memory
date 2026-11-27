"""B53 checks (prereg/B53_value_codes_prereg.md, 'Checks'), a second angle on the verdict.

python3 b53_check.py score              # independent scorer (plain loops) of every re-run cell vs results/b53_verdict.json
python3 b53_check.py bits               # own size rule: bits(d) <= budget < bits(d + 1) for every logged configuration
python3 b53_check.py rederive ENC       # numpy float64 re-implementation of the B53 store, one test store per cell
python3 b53_check.py handread           # Qwen3-8B / e5: the R1-V2 cell where LAM leads most, 10 + 10 queries
Appends to results/b53_check.txt; rederive writes results/b53_rederive_{enc}.json; handread results/b53_handread.txt.
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
OUT = f'{ROOT}/results/b53_check.txt'


def log(s):
    print(s, flush=True)
    open(OUT, 'a').write(s + '\n')


def seed_of(*parts):  # own copy of the registered seed rule
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def recs(enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{ROOT}/b53_out/fam/{enc}/all53_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


def score():
    J = json.load(open(f'{ROOT}/results/b53_verdict.json'))
    worst, n = 0.0, 0
    reg = {}
    for enc in PANEL:
        for M in M_GRID:
            for V in VN:
                R = recs(enc, M, V)
                for b in BPF:
                    for a in ('ideal', 'real'):
                        for x in ('LAM', 'BEST', 'VSAR'):
                            tot = cnt = 0
                            for r in R:
                                for g, p, y in zip(r['qgold'], r['pred'][f'VSA|{b}|{a}'], r['pred'][f'{x}|{b}|{a}']):
                                    tot += int(p == g) - int(y == g)
                                    cnt += 1
                            d = tot / cnt
                            worst = max(worst, abs(d - J['cells'][f'{enc}|{a}|{M}|{V}|{b}']['cmp'][x]['diff']))
                            n += 1
                            if a == 'ideal' and b in (4, 8, 16):
                                reg.setdefault((enc, x), []).append(d)
    wr = max(abs(sum(v) / len(v) - J['regions'][f'{e}|ideal|R1|{x}']['diff']) for (e, x), v in reg.items())
    log(f'score: {n} cell differences re-scored by plain loops; max |difference| from the verdict {worst:.2e}; '
        f'R1 regions max |difference| {wr:.2e}')


def bits_of(d, q, acct):
    per = q if acct == 'raw' else HQ[q]
    return d * per + 32 * math.ceil(d / CHK) if d > 0 else 0


def bits():
    n = bad = notmax = badcfg = 0
    for enc in PANEL:
        for M in M_GRID:
            for V in VN:
                for r in recs(enc, M, V):
                    for b in BPF:
                        for a, acct in (('ideal', 'ideal'), ('real', 'raw')):
                            d, q, ns = r['vsa_d'][f'VSA|{b}|{a}']
                            cfg = r['cfg'][f'VSA|{b}|{a}']
                            B = b * r['M']
                            n += 1
                            bad += bits_of(d, q, acct) > B
                            notmax += bits_of(d + 1, q, acct) <= B
                            badcfg += (q != cfg[2]) or (ns != max(1, int(math.floor(cfg[3] * r['M']))))
    log(f'bits: {n} (store, cell) configurations; over budget {bad}; not maximal (d + 1 fits) {notmax}; '
        f'logged q / stored share inconsistent with the chosen configuration {badcfg}')


def sylvester_codes(V, r):
    from scipy.linalg import hadamard
    if V == 2:
        return np.stack([r, -r])
    H = hadamard(16).astype(np.float64)
    return H[:, np.arange(len(r)) % 16] * r[None, :]


def rederive(enc):
    sys.path.insert(0, ROOT)
    import b32_common as C  # data only (codes, stores, preprocessing)
    tags = {'qwen8': ('', 'qwen8_mean_L9_f16'), 'e5': ('', 'e5_base'), 'b46': ('b46_out', 'b46_granite33i_mean_L30_f16')}
    sub, tag = tags.get(enc, (f'b49_out/enc_{enc}', None))
    if tag is None:
        tag = json.load(open(f'{ROOT}/{sub}/b49_encoder.json'))['file_tag']
    os.chdir(os.path.join(ROOT, sub))
    pool = C.Pool(tag)
    Tr = pool.train_matrix()
    preps = {}
    agree = tot = 0
    rows = []
    for M in M_GRID:
        for V in VN:
            R = recs(enc, M, V)
            for i, b in enumerate(BPF):
                rep = i % len(R)
                r = R[rep]
                st = C.make_store(pool, 'test', M, V, r['rep'])
                assert st['qgold'].tolist() == r['qgold']
                prep, fmap, q, f = r['cfg'][f'VSA|{b}|ideal']
                d, q2, ns = r['vsa_d'][f'VSA|{b}|ideal']
                if prep not in preps:
                    preps[prep] = C.Prep(prep, Tr)
                K = preps[prep](st['keys']).astype(np.float64)
                Q = preps[prep](st['queries']).astype(np.float64)
                keep = np.random.default_rng(seed_of('subset', 'test', M, V, r['rep'])).permutation(M)[:ns]
                sseed = seed_of('vsa', enc, 'test', M, V, r['rep'])
                nf = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
                sc = np.zeros((Q.shape[0], V))
                for j0 in range(0, d, CHK):
                    w = min(CHK, d - j0)
                    Gs = [np.random.default_rng(seed_of('G', enc, prep, fmap, m, j0, CHK)).standard_normal(
                        (K.shape[1], CHK), dtype=np.float32)[:, :w].astype(np.float64) for m in range(nf)]
                    r0 = np.where(np.random.default_rng(seed_of('C', enc, prep, fmap, j0, CHK)).random(
                        (1, CHK), dtype=np.float32)[0] < 0.5, -1.0, 1.0)
                    Cv = sylvester_codes(V, r0)[:, :w]

                    def phi(Y):
                        o = np.ones((Y.shape[0], w))
                        for G in Gs:
                            Z = Y @ G
                            o = o * (Z if fmap == 'lin' else np.where(Z >= 0, 1.0, -1.0))
                        return o
                    S = (phi(K[keep]) * Cv[st['vals'][keep]]).sum(0)
                    if q == 1:
                        Sq = np.sign(S)
                        z = Sq == 0
                        if z.any():
                            Sq[z] = np.random.default_rng(seed_of('Q', sseed, d, q, ns, j0)).choice([-1.0, 1.0],
                                                                                                    size=int(z.sum()))
                    else:
                        step = MAXSTEP[q] * (S.std() or 1.0)
                        Lq = 2 ** q
                        Sq = (np.clip(np.floor(S / step), -Lq // 2, Lq // 2 - 1) + 0.5) * step
                    sc += (phi(Q) * Sq) @ Cv.T
                p2 = sc.argmax(1)
                a = int((p2 == np.array(r['pred'][f'VSA|{b}|ideal'])).sum())
                agree += a
                tot += len(p2)
                rows.append({'M': M, 'V': V, 'b': b, 'rep': r['rep'], 'cfg': [prep, fmap, q, f], 'd': d,
                             'agree': a / len(p2), 'acc_rederived': float((p2 == st['qgold']).mean())})
                print(rows[-1], flush=True)
    json.dump({'enc': enc, 'agree': agree / tot, 'n': tot, 'rows': rows},
              open(f'{ROOT}/results/b53_rederive_{enc}.json', 'w'), indent=1)
    log(f'rederive {enc}: float64 numpy re-implementation (own Hadamard codes, own quantizer) on {len(rows)} test '
        f'stores x cells, {tot} queries: agreement {agree / tot:.4f}; worst cell {min(x["agree"] for x in rows):.4f}')


def handread():
    J = json.load(open(f'{ROOT}/results/b53_verdict.json'))
    ents = {e['uri']: e for e in json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']}
    L = ['# B53 hand reading (prereg "Checks"): the R1-V2 cell where LAM leads the B53 store most; 10 queries only the',
         '# VSA answers right, 10 only LAM; alias -> stored label (gold value); EXACT = the exact nearest stored label',
         '# is the queried entity', '']
    for enc in ('qwen8', 'e5'):
        cells = [(J['cells'][f'{enc}|ideal|{M}|2|{b}']['cmp']['LAM']['diff'], M, b) for M in M_GRID for b in (4, 8, 16)]
        d, M, b = min(cells)
        R = recs(enc, M, 2)
        A, B = [], []
        for r in R:
            for i, (g, x, y, ee) in enumerate(zip(r['qgold'], r['pred'][f'VSA|{b}|ideal'], r['pred'][f'LAM|{b}|ideal'],
                                                   r['pred']['EXACT_ENT'])):
                row = (r['rep'], r['qalias'][i], ents[r['uris'][r['qent'][i]]]['name'], g, ee == r['qent'][i])
                if x == g and y != g:
                    A.append(row)
                elif y == g and x != g:
                    B.append(row)
        rng = np.random.default_rng(53)
        L.append(f'## {enc}: M {M}, V 2, {b} bits per fact (VSA - LAM {100 * d:+.2f}); only-VSA {len(A)}, only-LAM '
                 f'{len(B)}')
        for nm, X in (('only VSA right', A), ('only LAM right', B)):
            L.append(f'  {nm} (EXACT right in {sum(x[4] for x in X)} of {len(X)}):')
            for k in rng.choice(len(X), size=min(10, len(X)), replace=False):
                rep, al, name, g, ex = X[k]
                L.append(f'    store {rep}: "{al}" -> "{name}" (value {g}; EXACT {"right" if ex else "wrong"})')
        L.append('')
    open(f'{ROOT}/results/b53_handread.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    a = sys.argv[1:]
    {'score': score, 'bits': bits, 'handread': handread}.get(a[0], lambda: rederive(a[1]))()
