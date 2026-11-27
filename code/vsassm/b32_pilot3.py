"""B32 pilot 3 (dev split only; before the pre-registration): the Qwen3-8B codes, one dev store per (M, V),
ideal accounting, a reduced grid (no sgn3, q in {1, 2}, no pq8/opq8), with b32_run's shared pass.

python3 b32_pilot3.py OUT.json
"""
import itertools
import json
import sys
import time

import numpy as np

import b32_common as C
import b32_run as R


def main():
    out = sys.argv[1]
    pool = C.Pool(R.ENCS['qwen8'])
    Tr = pool.train_matrix()
    preps = {p: C.Prep(p, Tr) for p in R.GRID['prep']}
    fl = ['lsh', 'itq', 'sq1', 'pq4']
    fams = {(p, f): C.IndexFamily(f, preps[p](Tr)) for p in R.GRID['prep'] for f in fl}
    res = []
    for M in (1024, 256):
        for V in R.GRID['V']:
            st = C.make_store(pool, 'dev', M, V, 0)
            g = st['qgold']
            t0 = time.time()
            best = {}
            for prep in R.GRID['prep']:
                for fmap in ('lin', 'sgn'):
                    r = R.run_vsa(st, preps, prep, fmap, R.GRID['bpf'], ['ideal'], [1, 2], R.GRID['f'], 'qwen8')
                    for (bpf, acct, q, f), p in r.items():
                        a = float((p == g).mean())
                        res.append({'M': M, 'V': V, 'bpf': bpf, 'arm': 'VSA', 'cfg': [prep, fmap, q, f], 'acc': a})
                        if a > best.get(('VSA', bpf), (-1,))[0]:
                            best['VSA', bpf] = (a, (prep, fmap, q, f))
                for family, bpf, f in itertools.product(fl, R.GRID['bpf'], R.GRID['f']):
                    p, kb = R.run_idx(st, preps, fams, prep, family, bpf, 'ideal', f)
                    a = float((p == g).mean())
                    res.append({'M': M, 'V': V, 'bpf': bpf, 'arm': 'IDX', 'cfg': [prep, family, f], 'acc': a,
                                'keybits': kb})
                    if a > best.get(('IDX', bpf), (-1,))[0]:
                        best['IDX', bpf] = (a, (prep, family, f))
            ex = {p: float((st['vals'][C.exact_nearest(preps[p](st['keys']), preps[p](st['queries']))] == g).mean())
                  for p in R.GRID['prep']}
            res.append({'M': M, 'V': V, 'arm': 'EXACT', 'acc': ex})
            print(f'M={M} V={V} Q={len(g)} EXACT {ex} ({time.time() - t0:.0f}s)', flush=True)
            for bpf in R.GRID['bpf']:
                print(f'   bpf={bpf:3d} IDX {best["IDX", bpf][0]:.3f} {best["IDX", bpf][1]}   VSA {best["VSA", bpf][0]:.3f} '
                      f'{best["VSA", bpf][1]}   diff {best["VSA", bpf][0] - best["IDX", bpf][0]:+.3f}', flush=True)
            json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    main()
