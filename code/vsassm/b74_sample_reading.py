"""B74 sample-reading material (prereg "Sample reading by the AI-assisted process"): Qwen3-8B and e5-base-v2,
(i) the R1 cell with the largest |I_T|: 10 queries with a non-zero interaction term (LT - HT - LR + HR), alias, label,
gold and the four answers; (ii) 10 label queries the store misses and LAMc answers (storage errors), same cell.

python3 b74_sample_reading.py      # -> results/b74_sample_material.txt
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b74_verdict as VV  # noqa: E402

RES = VV.RES


def main():
    import b66_lsfc as LS
    import b32_common as C
    L = ['# B74 sample-reading material (prereg: Qwen3-8B and e5-base-v2; R1 cell with the largest |I_T|)']
    for enc in ('qwen8', 'e5'):
        J = json.load(open(f'{RES}/b74_part_{enc}.json'))
        cells = [(abs(J['A_cells'][f'{M}|{V}|{b}|I_T']), M, V, b) for M in VN_M() for V in (2, 16) for b in (4, 8, 16)]
        _, M, V, b = max(cells)
        data = VV.load_A(enc)
        ldata, _ = VV.lab_preds(enc, data)
        pool = LS.setup(enc)['pool']
        recs = data[M, V]
        inter, store_miss = [], []
        for r, lr in zip(recs, ldata[M, V]):
            st = C.make_store(pool, 'test', M, V, r['rep'])
            g = r['qgold']
            for i in range(len(g)):
                ok = {a: int(r['pred'][f'{a}|{b}|ideal'][i] == g[i]) for a in ('HR', 'LR', 'HT', 'LT')}
                v = ok['LT'] - ok['HT'] - ok['LR'] + ok['HR']
                if v != 0:
                    e = st['qent'][i]
                    inter.append((v, r['rep'], st['qalias'][i], st['names'][e], int(g[i]),
                                  {a: r['pred'][f'{a}|{b}|ideal'][i] for a in ('HR', 'LR', 'HT', 'LT')}))
            for e in range(len(lr['qgold'])):
                s_ok = lr['pred'][f'STORE|{b}|ideal'][e] == lr['qgold'][e]
                l_ok = lr['pred'][f'LAMC|{b}|ideal'][e] == lr['qgold'][e]
                if l_ok and not s_ok:
                    store_miss.append((r['rep'], st['names'][e], int(lr['qgold'][e]),
                                       lr['pred'][f'STORE|{b}|ideal'][e], lr['pred'][f'LAMC|{b}|ideal'][e]))
        rng = np.random.default_rng(74)
        L.append(f'## {VV.NAMES[enc]} M={M} V={V} bpf={b}: I_T = {100 * J["A_cells"][f"{M}|{V}|{b}|I_T"]:+.2f}; '
                 f'queries with a non-zero interaction term {len(inter)} (+ {sum(x[0] > 0 for x in inter)}, '
                 f'- {sum(x[0] < 0 for x in inter)}); label queries store wrong / LAMc right {len(store_miss)}')
        L.append('  interaction queries: (term, rep, alias, stored label, gold, answers HR / LR / HT / LT)')
        for j in rng.choice(len(inter), size=min(10, len(inter)), replace=False):
            L.append(f'    {inter[j]}')
        L.append('  storage errors (label queries): (rep, label, gold, store answer, LAMc answer)')
        for j in rng.choice(len(store_miss), size=min(10, len(store_miss)), replace=False):
            L.append(f'    {store_miss[j]}')
    open(f'{RES}/b74_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def VN_M():
    return (256, 1024)


if __name__ == '__main__':
    main()
