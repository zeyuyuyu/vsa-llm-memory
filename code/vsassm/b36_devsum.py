"""B36 dev summary: per (M, V, bpf, acct), each family's best mean dev accuracy (and its configuration), with B32's
VSA / IDX dev bests for reference (B32 dev files for bpf >= 4, B36 dev32 files for 1-2).

python3 b36_devsum.py ENC DEVFILE [OUT.txt]
"""
import json
import sys
from collections import defaultdict

import numpy as np

enc, fn = sys.argv[1], sys.argv[2]
acc = defaultdict(dict)
for l in open(fn):
    r = json.loads(l)
    for x in r['rows']:
        acc[r['M'], r['V'], x['bpf'], x['acct'], x['fam'], tuple(x['cfg'])][r['rep']] = x['acc']
amap = {'ideal': 'ideal', 'real': 'raw'}
ref = defaultdict(dict)
for f32 in (f'b32_out/dev_{enc}_vsa.jsonl', f'b32_out/dev_{enc}_idx.jsonl', f'b36_out/dev32_{enc}_vsa.jsonl',
            f'b36_out/dev32_{enc}_idx.jsonl'):
    try:
        for l in open(f32):
            r = json.loads(l)
            for x in r['rows']:
                if x['arm'] == 'EXACT':
                    ref[r['M'], r['V'], None, None, 'EXACT', tuple(x['cfg'])][r['rep']] = x['acc']
                else:
                    a = 'ideal' if x['acct'] == 'ideal' else 'real'
                    ref[r['M'], r['V'], x['bpf'], a, x['arm'], tuple(x['cfg'])][r['rep']] = x['acc']
    except FileNotFoundError:
        pass
best = {}
for d in (acc, ref):
    for k, v in d.items():
        M, V, b, a, f, c = k
        m = float(np.mean(list(v.values())))
        key = (M, V, b, a, f)
        if key not in best or m > best[key][0]:
            best[key] = (m, c, len(v))
out = []
fams = ['VSA', 'IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
cells = sorted({k[:4] for k in best if k[4] in fams and k[2] is not None})
for acct in ('ideal', 'real'):
    out.append(f'## {enc} {acct}: best mean dev accuracy per family (points); n = dev stores')
    out.append(f'{"M":>5} {"V":>5} {"bpf":>4} ' + ' '.join(f'{f:>6}' for f in fams) + '  VSA-maxcomp  bestcomp')
    for (M, V, b, a) in cells:
        if a != acct:
            continue
        row = {f: best.get((M, V, b, a, f)) for f in fams}
        comp = {f: row[f][0] for f in fams[1:] if row[f]}
        bc = max(comp, key=comp.get) if comp else None
        vs = row['VSA'][0] if row['VSA'] else float('nan')
        out.append(f'{M:5d} {V:5d} {b:4d} ' + ' '.join(f'{100 * row[f][0]:6.2f}' if row[f] else '     -' for f in fams)
                   + (f'  {100 * (vs - comp[bc]):+7.2f}  {bc} {row[bc][1]}' if bc else ''))
    out.append('')
txt = '\n'.join(out)
print(txt)
if len(sys.argv) > 3:
    open(sys.argv[3], 'w').write(txt + '\n')
