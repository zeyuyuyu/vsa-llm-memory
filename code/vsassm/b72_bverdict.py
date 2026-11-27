"""B72 part (B) verdict, run where the (B) test jobs are (the A800 box; compute note 2: slow link): symmetric
side-information contrasts per K from the (B) test jobs at the committed choices, the records rebuilt from B32's
make_store (qgold, qent per test store), b66_storeboot.analyse (one-level = b36_verdict.boot_region, two-level = B66's
store-level scheme; 2,000 draws), plain-loop recomputation of every point statistic, and the bits audit of every (B)
dev and test row (b72_check.bits' rules).

python3 b72_bverdict.py ENC [ENC ...]     # -> results/b72_bpart_{enc}.json
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b66_storeboot as SB  # noqa: E402
import b72_run as E  # noqa: E402

A = 'ideal'
KT = {1e3: 'k3', 1e4: 'k4', 1e5: 'k5'}
MEMS = ['STORE', 'VSAR', 'LAMC', 'NLMSC', 'SDMT']
PAIRS = [('STORE', 'LAMC'), ('VSAR', 'SDMT'), ('STORE', 'NLMSC')]


def test_index(enc):
    out = {}
    jd = os.path.join(E.OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_') or '_test_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['enc'] != enc or s['arm'] not in ('BV', 'BL', 'BS'):
            continue
        gk = (s['mem'], s['K'], s['M'], s['V'], s['rep'])
        out.setdefault(('_q',) + gk, r['qgold'])
        assert out[('_q',) + gk] == r['qgold']
        for x in r['rows']:
            out[gk + (x['bpf'], tuple(x['cfg']))] = x['pred']
    return out


def load_b(enc):
    import b32_common as C
    import b66_lsfc as LS
    ch = E.choice(enc)
    pool = LS.setup(enc)['pool']
    ti = test_index(enc)
    data = {}
    for M in E.M_GRID:
        for V in E.VN:
            recs = []
            for rep in range(E.TEST_REPS[M]):
                st = C.make_store(pool, 'test', M, V, rep)
                r = {'rep': rep, 'qgold': [int(x) for x in st['qgold']], 'qent': [int(x) for x in st['qent']],
                     'pred': {}}
                dflt = int(np.bincount(st['vals'], minlength=V).argmax())
                for b in E.BPF:
                    for K in E.KS:
                        for mem in MEMS:
                            c = ch[f'{M}|{V}|{b}|{mem}|K{E.kstr(K)}']
                            if c['cfg'] is None:
                                r['pred'][f'{mem}{KT[K]}|{b}|{A}'] = [dflt] * len(r['qgold'])
                                continue
                            gk = (mem, K, M, V, rep)
                            assert ti[('_q',) + gk] == r['qgold'], ('qgold', enc, gk)
                            r['pred'][f'{mem}{KT[K]}|{b}|{A}'] = ti[gk + (b, tuple(c['cfg']))]
                recs.append(r)
            data[M, V] = recs
    return data


def plain_region(data, x, y, R):
    tot = 0.0
    for (M, V, b) in R:
        s, n = 0.0, 0
        for r in data[M, V]:
            px, py, g = r['pred'][f'{x}|{b}|{A}'], r['pred'][f'{y}|{b}|{A}'], r['qgold']
            for i in range(len(g)):
                s += (1.0 if int(px[i]) == int(g[i]) else 0.0) - (1.0 if int(py[i]) == int(g[i]) else 0.0)
                n += 1
        tot += s / n
    return tot / len(R)


def run(enc):
    data = load_b(enc)
    ei = E.PANEL.index(enc)
    cons = []
    for K in E.KS:
        for (x, y) in PAIRS:
            cons.append((f'{x}{KT[K]}', f'{y}{KT[K]}', f'B{E.kstr(K)}'))
    J = SB.analyse(data, ei + 7400, cons, accts=[A], nb={A: 2000}, regions=SB.REG)
    mx = 0.0
    for key, e in J['regions'].items():
        _, rn, nm = key.split('|')
        x, y = nm.split('-')
        mx = max(mx, abs(plain_region(data, x, y, SB.REG[rn]) - e['diff']))
    acc = {}
    for (M, V), recs in data.items():
        for b in E.BPF:
            for K in E.KS:
                for mem in MEMS:
                    c = SB.correct(recs, f'{mem}{KT[K]}|{b}|{A}')
                    acc[f'{M}|{V}|{b}|{mem}{KT[K]}'] = float(c.mean())
    out = {'enc': enc, 'B': J['regions'], 'missing': J['missing'], 'cell_acc': acc, 'plain_max_abs_diff': mx}
    json.dump(out, open(f'{E.RES}/b72_bpart_{enc}.json', 'w'))
    print('bpart', enc, 'missing', len(J['missing']), 'plain max diff', mx, flush=True)


def bits_b(encs):
    """b72_check.bits' rules on the (B) rows (dev and test) where they are."""
    import b72_check as K72
    K72.OUT = E.OUT
    res = K72.bits(encs)
    os.replace(f'{K72.RES}/b72_check_bits.json', f'{E.RES}/b72_check_bits_box.json')
    return res


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'bits':
        bits_b(a[1:])
    else:
        for enc in a:
            run(enc)
