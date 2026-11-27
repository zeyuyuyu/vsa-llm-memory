"""B32 pilot 2 (dev split only; before the pre-registration): stored share f, preprocessing and q, per arm.

python3 b32_pilot2.py ENC OUT.json --M 1024 --V 16 --bpf 4,8,16 [--reps 1] [--fams lsh,itq,sq1] [--fmaps lin,sgn]
"""
import json
import sys
import time

import numpy as np

import b32_common as C


def arg(a, k, d, typ=int):
    return [typ(x) for x in a[a.index(k) + 1].split(',')] if k in a else d


def main():
    a = sys.argv[1:]
    enc, out = a[0], a[1]
    Ms, Vs, bpfs = arg(a, '--M', [1024]), arg(a, '--V', [16]), arg(a, '--bpf', [4, 8, 16])
    reps = arg(a, '--reps', [1])[0]
    fams_l = arg(a, '--fams', ['lsh', 'itq', 'sq1', 'pq4'], str)
    fmaps = arg(a, '--fmaps', ['lin', 'sgn', 'sgn3'], str)
    qs = arg(a, '--q', [1, 2, 4])
    fs = arg(a, '--f', [1.0, 0.5, 0.25, 0.125], float)
    prep_l = arg(a, '--prep', ['center', 'whiten'], str)
    accts = arg(a, '--acct', ['raw', 'ideal'], str)
    pool = C.Pool(enc)
    Tr = pool.train_matrix()
    preps = {p: C.Prep(p, Tr) for p in prep_l}
    fams = {(p, f): C.IndexFamily(f, preps[p](Tr)) for p in prep_l for f in fams_l}
    res = []
    for M in Ms:
        for V in Vs:
            for rep in range(reps):
                st = C.make_store(pool, 'dev', M, V, rep)
                g = st['qgold']
                for p in prep_l:
                    pr, _ = C.answer(st, 'EXACT', (p,), 0, preps, fams)
                    res.append({'M': st['M'], 'V': V, 'rep': rep, 'arm': 'EXACT', 'cfg': [p], 'acc': float((pr == g).mean())})
                print(f'M={st["M"]} V={V} rep={rep} Q={len(g)} EXACT ' +
                      ' '.join(f'{r["cfg"][0]}:{r["acc"]:.3f}' for r in res[-len(prep_l):]), flush=True)
                for bpf in bpfs:
                  for acct in accts:
                    best = {}
                    for p in prep_l:
                        for f in fs:
                            for fam in fams_l:
                                cfg = (p, fam, f)
                                pr, info = C.answer(st, 'IDX', cfg, bpf, preps, fams, acct)
                                acc = float((pr == g).mean())
                                res.append({'M': st['M'], 'V': V, 'rep': rep, 'bpf': bpf, 'acct': acct, 'arm': 'IDX', 'cfg': list(cfg),
                                            'acc': acc, **info})
                                if acc > best.get('IDX', (-1,))[0]:
                                    best['IDX'] = (acc, cfg)
                            for fm in fmaps:
                                for q in qs:
                                    cfg = (p, fm, q, f)
                                    t0 = time.time()
                                    pr, info = C.answer(st, 'VSA', cfg, bpf, preps, fams, acct)
                                    acc = float((pr == g).mean())
                                    res.append({'M': st['M'], 'V': V, 'rep': rep, 'bpf': bpf, 'acct': acct, 'arm': 'VSA',
                                                'cfg': list(cfg), 'acc': acc, 'sec': time.time() - t0, **info})
                                    if acc > best.get('VSA', (-1,))[0]:
                                        best['VSA'] = (acc, cfg)
                    print(f'  bpf={bpf:4d} {acct:5s} best IDX {best["IDX"][0]:.3f} {best["IDX"][1]}  best VSA {best["VSA"][0]:.3f} '
                          f'{best["VSA"][1]}', flush=True)
                    json.dump(res, open(out, 'w'), indent=0)


if __name__ == '__main__':
    main()
