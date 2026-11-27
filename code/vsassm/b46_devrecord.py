"""B46 dev summary from the committed choice file (dev split only): dev VSA - BEST per region, BEST family counts, and
what the grids chose; written before any test run.

python3 b46_devrecord.py      # reads b46_out/results/b36_choice_b46.json (or results/b46_choice_b36.json)
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
fn = os.path.join(ROOT, 'b46_out/results/b36_choice_b46.json')
ch = json.load(open(fn))['choice']
Ms, Vs, B = (256, 1024), (2, 16, 128, 1024), (1, 2, 4, 8, 16, 32, 64)
REG = {'R1': [(M, V, b) for M in Ms for V in (2, 16) for b in (4, 8, 16)],
       'R0': [(M, V, b) for M in Ms for V in (2, 16) for b in (1, 2)],
       'RH': [(M, V, b) for M in Ms for V in (2, 16) for b in (32, 64)],
       'R2': [(M, 1024, b) for M in Ms for b in B]}
out = []
for acct in ('ideal', 'real'):
    for name, cells in REG.items():
        d = [ch[f'{M}|{V}|{b}|{acct}|VSA']['dev_acc'] - ch[f'{M}|{V}|{b}|{acct}|BEST']['dev_acc'] for M, V, b in cells]
        fams = collections.Counter(ch[f'{M}|{V}|{b}|{acct}|BEST']['fam'] for M, V, b in cells)
        out.append(f'{acct:5s} {name}: dev VSA - BEST mean {100 * sum(d) / len(d):+.2f} (cells {min(d) * 100:+.1f} to '
                   f'{max(d) * 100:+.1f}); BEST families {dict(fams)}')
    allf = collections.Counter(ch[f'{M}|{V}|{b}|{acct}|BEST']['fam'] for M in Ms for V in Vs for b in B)
    out.append(f'{acct:5s} all 56 cells: BEST families {dict(allf)}')
out.append('')
out.append('ideal, R1 cells: dev accuracy VSA / BEST (family) / each family; VSA and IDX configurations')
for M, V, b in REG['R1']:
    k = f'{M}|{V}|{b}|ideal'
    fam = {f: round(100 * ch[f'{k}|{f}']['dev_acc'], 2) for f in ('IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM')}
    out.append(f'  M {M:4d} V {V:2d} bpf {b:2d}: VSA {100 * ch[k + "|VSA"]["dev_acc"]:.2f} {ch[k + "|VSA"]["cfg"]}; '
               f'BEST {ch[k + "|BEST"]["fam"]} {100 * ch[k + "|BEST"]["dev_acc"]:.2f}; {fam}; IDX {ch[k + "|IDX"]["cfg"]}')
vs = collections.Counter(tuple(ch[f'{M}|{V}|{b}|ideal|VSA']['cfg'][1:3]) for M in Ms for V in Vs for b in B)
ix = collections.Counter(ch[f'{M}|{V}|{b}|ideal|IDX']['cfg'][1] for M in Ms for V in Vs for b in B)
out.append(f'VSA (fmap, q) over 56 ideal cells: {dict(vs)}; IDX families: {dict(ix)}')
txt = '\n'.join(out)
print(txt)
open(os.path.join(ROOT, 'results/b46_devrecord.txt'), 'w').write(txt + '\n')
