"""B59 checks (prereg/B59_maps_all_encoders_protected_full_prereg.md, 'Checks'), part 1.

python3 b59_check.py score | bits | boot | xmachine | reading
Appends to results/b59_check.txt (reading: results/b59_sample_reading.txt).
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b56_check as K56  # noqa: E402  (own size rules and plain-loop scorer, independent of the run code)

PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
NEW = ['b46', 'llama', 'mistral', 'bge']
NDIM = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
OUT = f'{ROOT}/results/b59_check.txt'


def log(s):
    print(s, flush=True)
    open(OUT, 'a').write(s + '\n')


def recs(enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


def skey(s, m):
    return ('VSAS' if m == 0 else f'VSASm{m}') if s == 'VSAS' else f'VSAm{m}'


def ckey(y, m):
    return y if (m == 0 or y == 'BEST36') else f'{y}m{m}'


def score():
    worst = wpt = 0.0
    n = 0
    for enc in PANEL:
        P = json.load(open(f'{ROOT}/results/b59_part_{enc}.json'))
        acc = {}
        for M in M_GRID:
            for V in VN:
                R = recs(enc, M, V)
                for b in (4, 8, 16):
                    for a in ('ideal', 'real'):
                        cell = P['cells'][f'{enc}|{a}|{M}|{V}|{b}']
                        for m in range(4):
                            for k, arm in ([(f'{s}m{m}', skey(s, m)) for s in ('VSAS', 'VSA')] +
                                           [(f'{y}m{m}', ckey(y, m)) for y in ('LAM', 'BEST', 'LAMRF', 'BEST36')]):
                                v = K56.acc_loop(R, f'{arm}|{b}|{a}')
                                acc[(M, V, b, a, k)] = v
                                worst = max(worst, abs(v - cell[k]))
                                n += 1
        for a in ('ideal', 'real'):
            for s in ('VSAS', 'VSA'):
                for y in ('LAM', 'BEST', 'LAMRF', 'BEST36'):
                    x = P['maps'][f'{enc}|{a}|R1|{s}-{y}']
                    for m in range(4):
                        d = np.mean([acc[(M, V, b, a, f'{s}m{m}')] - acc[(M, V, b, a, f'{y}m{m}')]
                                     for M in M_GRID for V in VN for b in (4, 8, 16)])
                        wpt = max(wpt, abs(d - x['per_map'][m]))
    log(f'- score: {n} cell accuracies re-scored by plain loops; max |difference| {worst:.2e}; per-map R1 points '
        f'(both accountings, VSAS / VSA x 4 comparisons) max |difference| {wpt:.2e}')


def bits():
    over = notmax = n = 0
    for enc in PANEL:
        for M in M_GRID:
            for V in VN:
                for r in recs(enc, M, V):
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            B_ = b * M
                            for m in (1, 2, 3):
                                prep, fmap, q53, f, d, q, ns = r['b59'][f'VSAm{m}|{b}|{a}']
                                n += 1
                                over += K56.vbits(d, q, a) > B_ + 1e-9
                                notmax += K56.vbits(d + 1, q, a) <= B_
                                if enc in NEW:
                                    for arm in ('LM', 'RT'):
                                        d, q = r['b59'][f'{arm}m{m}|{b}|{a}'][:2]
                                        n += 1
                                        over += K56.vbits(d, q, a) > B_ + 1e-9
                                        notmax += K56.vbits(d + 1, q, a) <= B_
                                    p, q, bt = r['b59'][f'LRm{m}|{b}|{a}'][:3]
                                    n += 1
                                    over += K56.lbits(p, V - 1, q, a) > B_ + 1e-9
                                    bt = r['b59'][f'LAMm{m}|{b}|{a}']
                                    n += 1
                                    over += float(bt) > B_ + 1e-9
    log(f'- bits: {n} (store, cell, arm, map) configurations of the B59 map arms; over budget {over}; store not maximal '
        f'under the cap {notmax}')


def boot():
    """Independent two-level bootstrap by explicit loops (different seed): per draw, 4 maps drawn with replacement and
    entity clusters drawn once and shared; R1 VSAS - LAM and VSAS - BEST, ideal, on b46 and bge."""
    import b36_verdict as V36
    for enc in ('b46', 'bge'):
        P = json.load(open(f'{ROOT}/results/b59_part_{enc}.json'))
        data = {(M, V): recs(enc, M, V) for M in M_GRID for V in VN}
        for y in ('LAM', 'BEST'):
            cells = []
            for M in M_GRID:
                for V in VN:
                    for b in (4, 8, 16):
                        _, cl = V36.per_query(data[M, V], f'VSAS|{b}|ideal')
                        ds = [V36.per_query(data[M, V], f'{skey("VSAS", m)}|{b}|ideal')[0] -
                              V36.per_query(data[M, V], f'{ckey(y, m)}|{b}|ideal')[0] for m in range(4)]
                        u, inv = np.unique(np.asarray(cl), return_inverse=True)
                        cells.append((ds, inv, len(u)))
            rng = np.random.default_rng(5959)
            out = []
            for it in range(3000):
                maps = rng.integers(0, 4, 4)
                tot = 0.0
                for ds, inv, k in cells:
                    pick = rng.integers(0, k, k)
                    cnt = np.bincount(pick, minlength=k)
                    w = cnt[inv]
                    tot += np.mean([(ds[m] * w).sum() / w.sum() for m in maps])
                out.append(tot / len(cells))
            lo, hi = np.quantile(out, [0.025, 0.975])
            ref = P['maps'][f'{enc}|ideal|R1|VSAS-{y}']['two95']
            log(f'- boot {enc} VSAS-{y}: loop two-level 95% [{100 * lo:+.2f}, {100 * hi:+.2f}] vs run [{100 * ref[0]:+.2f}, '
                f'{100 * ref[1]:+.2f}]; max |end difference| {100 * max(abs(lo - ref[0]), abs(hi - ref[1])):.2f} points')


def xmachine():
    a = [json.loads(l) for l in open(f'{ROOT}/b59_out/bge/devm_bge_m1_center_sgn.jsonl')]
    b = [json.loads(l) for l in open(f'{ROOT}/b59_out_local/bge/devm_bge_m1_center_sgn.jsonl')]
    ka = {(r['M'], r['V'], r['rep']): {(x['bpf'], x['acct'], tuple(x['cfg'])): x['acc'] for x in r['rows']} for r in a}
    kb = {(r['M'], r['V'], r['rep']): {(x['bpf'], x['acct'], tuple(x['cfg'])): x['acc'] for x in r['rows']} for r in b}
    nd = sum(1 for k in ka for c in ka[k] if abs(ka[k][c] - kb.get(k, {}).get(c, -1)) > 1e-12)
    nt = sum(len(v) for v in ka.values())
    worst = max(abs(ka[k][c] - kb[k][c]) for k in ka for c in ka[k])
    log(f'- xmachine bge devm m1 center:sgn (box vs local): {len(ka)} / {len(kb)} stores, {nt} configuration '
        f'accuracies, {nd} differ, max |difference| {worst:.2e}')
    fa = f'{ROOT}/b59_out/b46/decm_dev_b46_m1_M256_V2.jsonl.gz'
    fb = f'{ROOT}/b59_out_local/b46/decm_dev_b46_m1_M256_V2.jsonl.gz'
    if os.path.exists(fb):
        A = [json.loads(l) for l in gzip.open(fa, 'rt')]
        Bb = [json.loads(l) for l in gzip.open(fb, 'rt')]
        npred = ndiff = 0
        cells = set()
        for x, y in zip(A, Bb):
            for k, p in x['pred'].items():
                q = y['pred'][k]
                npred += len(p)
                dd = sum(int(u != v) for u, v in zip(p, q))
                ndiff += dd
                if dd:
                    cells.add(k)
        log(f'- xmachine b46 decm m1 dev M256 V2 (box vs local): {npred} predictions, {ndiff} differ, in {len(cells)} '
            f'of {len(A[0]["pred"])} configurations')
    else:
        log('- xmachine b46 engine job: not run locally (see compute note)')


def reading():
    ents = {e['uri']: e for e in json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']}
    L = ['# B59 sample reading by the AI-assisted process (not independent): per encoder the R1 cell (ideal) with the',
         '# largest between-map spread of VSAS - LAM; 10 queries where the four maps\' store answers disagree (alias ->',
         '# stored label, gold value, store answer on maps 0-3, LAM answer on maps 0-3). Notes at the end.', '']
    for enc in ('b46', 'bge'):
        P = json.load(open(f'{ROOT}/results/b59_part_{enc}.json'))
        best = None
        for M in M_GRID:
            for V in VN:
                for b in (4, 8, 16):
                    c = P['cells'][f'{enc}|ideal|{M}|{V}|{b}']
                    ds = [c[f'VSASm{m}'] - c[f'LAMm{m}'] for m in range(4)]
                    sp = max(ds) - min(ds)
                    if best is None or sp > best[0]:
                        best = (sp, M, V, b, ds)
        sp, M, V, b, ds = best
        R = recs(enc, M, V)
        rng = np.random.default_rng(59)
        pool = []
        for r in R:
            st = [r['pred'][f'{skey("VSAS", m)}|{b}|ideal'] for m in range(4)]
            lm = [r['pred'][f'{ckey("LAM", m)}|{b}|ideal'] for m in range(4)]
            for i, g in enumerate(r['qgold']):
                a = [s[i] for s in st]
                if len(set(int(x == g) for x in a)) > 1:
                    pool.append((r['qalias'][i], ents[r['uris'][r['qent'][i]]]['name'], g, a, [x[i] for x in lm]))
        L.append(f'## {enc}: M = {M}, V = {V}, {b} bits per fact; VSAS - LAM per map ' +
                 ' '.join(f'{100 * d:+.2f}' for d in ds) + f' (spread {100 * sp:.2f}); {len(pool)} queries where the '
                 f'store\'s maps disagree on correctness')
        for k in rng.choice(len(pool), min(10, len(pool)), replace=False):
            a, lab, g, st, lm = pool[k]
            L.append(f'  {a!r} -> {lab!r} gold {g} store maps0-3 {st} LAM maps0-3 {lm}')
        L.append('')
    open(f'{ROOT}/results/b59_sample_reading.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'score': score, 'bits': bits, 'boot': boot, 'xmachine': xmachine, 'reading': reading}[sys.argv[1]]()
