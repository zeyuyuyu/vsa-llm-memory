"""B49 hand reading (prereg/B49 section 9; b36_handread.py pattern): per encoder, in the R1 cell with the largest
VSA - BEST and in the R1 cell where BEST leads most (ideal accounting), 10 queries only the VSA answers right and 10 only
BEST answers right, with the alias, its entity's label, the label EXACT retrieves, and the centred alias-label cosine.

python3 b49_handread.py ENC [ENC ...]        # -> results/b49_handread_{enc}.txt
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    os.chdir(f'{ROOT}/b49_out')
    import b32_common as C
    import b49_fam as F
    J = json.load(open(f'{ROOT}/results/b49_verdict.json'))
    for enc in sys.argv[1:]:
        pool = C.Pool(F.ENC_TAG[enc])
        P = C.Prep('center', pool.train_matrix())
        Y = P(pool.X)
        cells = {k: c for k, c in J['cells'].items() if k.startswith(f'{enc}|ideal|') and
                 int(k.split('|')[3]) in (2, 16) and int(k.split('|')[4]) in (4, 8, 16)}
        hi = max(cells, key=lambda k: cells[k]['diff'])
        lo = min(cells, key=lambda k: cells[k]['diff'])
        L = [f'# B49 hand reading, {enc}: R1 cells with the largest and the smallest VSA - BEST (ideal)', '']
        rng = np.random.default_rng(49)
        for k in (hi, lo):
            _, _, M, V, b = k.split('|')
            M, V, b = int(M), int(V), int(b)
            c = cells[k]
            L.append(f'## cell M={M} V={V} bpf={b}: VSA {100 * c["VSA"]:.1f}, BEST {c["BEST_fam"]} {100 * c["BEST"]:.1f}, '
                     f'diff {100 * c["diff"]:+.2f}, EXACT {100 * c["EXACT"]:.1f}')
            recs = [json.loads(l) for l in __import__('gzip').open(
                f'{ROOT}/b49_out/fam/{enc}/test/all49_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            only = {'VSA': [], 'BEST': []}
            for r in recs:
                names = [None] * len(r['uris'])
                st = C.make_store(pool, 'test', M, V, r['rep'])
                names = st['names']
                for i, (g, a, e) in enumerate(zip(r['qgold'], r['qalias'], r['qent'])):
                    v = r['pred'][f'VSA|{b}|ideal'][i] == g
                    x = r['pred'][f'BEST|{b}|ideal'][i] == g
                    if v != x:
                        cos = float(Y[pool.idx[a]] @ Y[pool.idx[names[e]]])
                        ex = names[r['pred']['EXACT_ENT'][i]]
                        only['VSA' if v else 'BEST'].append((a, names[e], ex, cos, r['pred']['EXACT'][i] == g))
            for side in ('VSA', 'BEST'):
                lst = only[side]
                L.append(f'  only {side} right: {len(lst)} queries; 10 drawn at random '
                         f'(alias | label | EXACT\'s label | centred cosine alias-label | EXACT right)')
                for j in rng.permutation(len(lst))[:10]:
                    a, lab, ex, cos, er = lst[j]
                    L.append(f'    {a!r:40s} | {lab!r:40s} | {ex!r:40s} | {cos:+.3f} | {er}')
                if lst:
                    L.append(f'    mean centred cosine over all {len(lst)}: {np.mean([x[3] for x in lst]):+.3f}; '
                             f'EXACT right in {np.mean([x[4] for x in lst]):.2f}')
            L.append('')
        open(f'{ROOT}/results/b49_handread_{enc}.txt', 'w').write('\n'.join(L) + '\n')
        print('\n'.join(L))


if __name__ == '__main__':
    main()
