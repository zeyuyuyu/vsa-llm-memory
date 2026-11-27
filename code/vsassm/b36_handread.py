"""B36 hand reading (prereg/B36_keyfree_equal_bits_prereg.md, Checks): queries that only the VSA answers right and
queries that only BEST answers right, in the R1 cell with the largest VSA - BEST difference and in one cell where BEST
leads most, with the alias, the stored label, the exact index's nearest label and the alias-label cosine.

python3 b36_handread.py OUTDIR [ENC]   -> results/b36_handread_{enc}.txt
"""
import gzip
import json
import sys

import numpy as np

import b32_common as C

outdir = sys.argv[1]
enc = sys.argv[2] if len(sys.argv) > 2 else 'qwen8'
J = json.load(open('results/b36_verdict.json'))
cells = {k: v for k, v in J['cells'].items() if k.startswith(f'{enc}|ideal|')}
r1 = [k for k in cells if int(k.split('|')[3]) in (2, 16) and int(k.split('|')[4]) in (4, 8, 16)]
top = max(r1, key=lambda k: cells[k]['diff'])
low = min(cells, key=lambda k: cells[k]['diff'])
pool = C.Pool({'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}[enc])
Pw = C.Prep('whiten', pool.train_matrix())
ent = {e['uri']: e for e in pool.ents}
rng = np.random.default_rng(7)
out = []
for key in (top, low):
    _, acct, M, V, b = key.split('|')
    M, V, b = int(M), int(V), int(b)
    c = cells[key]
    out.append(f'## {key}: VSA {100 * c["VSA"]:.2f}, BEST ({c["BEST_fam"]}) {100 * c["BEST"]:.2f}, diff '
               f'{100 * c["diff"]:+.2f}, EXACT {100 * c["EXACT"]:.2f}')
    R = [json.loads(l) for l in gzip.open(f'{outdir}/all_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
    only = {'VSA': [], 'BEST': []}
    for r in R:
        pv, pb, g = r['pred'][f'VSA|{b}|{acct}'], r['pred'][f'BEST|{b}|{acct}'], r['qgold']
        for i in range(len(g)):
            if (pv[i] == g[i]) != (pb[i] == g[i]):
                only['VSA' if pv[i] == g[i] else 'BEST'].append((r, i))
    out.append(f'   queries only the VSA gets right: {len(only["VSA"])}; only BEST: {len(only["BEST"])}; '
               f'BEST configuration (store 0): {R[0]["cfg"][f"BEST|{b}|{acct}"]}')
    for who in ('VSA', 'BEST'):
        lst = only[who]
        pick = rng.choice(len(lst), size=min(10, len(lst)), replace=False) if lst else []
        out.append(f'   -- only {who} right (10 at random) --')
        cs_all = []
        for r, i in lst:
            e = ent[r['uris'][r['qent'][i]]]
            k = Pw(pool.X[[pool.idx[e['name']]]])[0]
            q = Pw(pool.X[[pool.idx[r['qalias'][i]]]])[0]
            cs_all.append(float(k @ q))
        for j in pick:
            r, i = lst[j]
            e = ent[r['uris'][r['qent'][i]]]
            nn = ent[r['uris'][r['pred']['EXACT_ENT'][i]]]['name']
            out.append(f'   alias {r["qalias"][i]!r:40s} label {e["name"]!r:35s} exact-NN {nn!r:30s} '
                       f'cos(whitened) {cs_all[j]:+.2f}')
        if cs_all:
            out.append(f'   median alias-label cosine (whitened) over all {len(cs_all)}: {np.median(cs_all):+.3f}')
    out.append('')
txt = '\n'.join(out)
open(f'results/b36_handread_{enc}.txt', 'w').write(txt + '\n')
print(txt)
