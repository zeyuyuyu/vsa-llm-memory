"""B32 pilot (dev split only; written before the pre-registration, to see the operating range).

python3 b32_pilot.py ENC PREP OUT.json [--M 256,1024] [--V 2,16,256,1024] [--bpf 8,32,128,512] [--reps 1]
"""
import json
import sys
import time

import numpy as np

import b32_common as C

FAMS = ['lsh', 'itq', 'sq1', 'sq4', 'pq4', 'pq8', 'opq8']
VSAS = [(f, q) for f in ('lin', 'sgn', 'sgn3') for q in (1, 4)]


def arg(a, k, d):
    return [int(x) for x in a[a.index(k) + 1].split(',')] if k in a else d


def main():
    a = sys.argv[1:]
    enc, prep, out = a[0], a[1], a[2]
    Ms, Vs, bpfs = arg(a, '--M', [256, 1024]), arg(a, '--V', [2, 16, 256, 1024]), arg(a, '--bpf', [8, 32, 128, 512])
    reps = arg(a, '--reps', [1])[0]
    pool = C.Pool(enc)
    P = C.Prep(prep, pool.train_matrix())
    T = P(pool.train_matrix())
    fams = {f: C.IndexFamily(f, T) for f in FAMS}
    res = []
    for M in Ms:
        for V in Vs:
            for rep in range(reps):
                st = C.make_store(pool, 'dev', M, V, rep)
                K, Q, g = P(st['keys']), P(st['queries']), st['qgold']
                ex = (st['vals'][C.exact_nearest(K, Q)] == g).mean()
                ent_ok = (C.exact_nearest(K, Q) == st['qent']).mean()
                res.append({'M': st['M'], 'V': V, 'rep': rep, 'arm': 'EXACT', 'acc': float(ex), 'ent': float(ent_ok),
                            'nq': len(g)})
                print(f'M={st["M"]} V={V} rep={rep} Q={len(g)} EXACT {ex:.3f} (entity {ent_ok:.3f})', flush=True)
                for bpf in bpfs:
                    B = bpf * st['M']
                    line = []
                    kb = C.key_bits(B, st['M'], V)
                    for f, fam in fams.items():
                        cfg = fam.config(kb)
                        if cfg is None:
                            acc = float((np.bincount(st['vals'], minlength=V).argmax() == g).mean())
                            res.append({'M': st['M'], 'V': V, 'rep': rep, 'bpf': bpf, 'arm': 'IDX-' + f, 'acc': acc,
                                        'bits': 0})
                            continue
                        nn = fam.nearest(K, Q, cfg[1])
                        acc = float((st['vals'][nn] == g).mean())
                        res.append({'M': st['M'], 'V': V, 'rep': rep, 'bpf': bpf, 'arm': 'IDX-' + f, 'acc': acc,
                                    'bits': cfg[0]})
                        line.append(f'{f}:{acc:.3f}')
                    for fm, q in VSAS:
                        d = C.vsa_dim(B, q)
                        t0 = time.time()
                        pred, _ = C.vsa_answer(K, st['vals'], Q, V, d, q, fm, C.seed_of('vsa', M, V, rep))
                        acc = float((pred == g).mean())
                        res.append({'M': st['M'], 'V': V, 'rep': rep, 'bpf': bpf, 'arm': f'VSA-{fm}-q{q}', 'acc': acc,
                                    'd': d, 'sec': time.time() - t0})
                        line.append(f'{fm}q{q}:{acc:.3f}')
                    print(f'  bpf={bpf:4d} keybits={kb:4d} ' + ' '.join(line), flush=True)
                json.dump(res, open(out, 'w'), indent=0)


if __name__ == '__main__':
    main()
