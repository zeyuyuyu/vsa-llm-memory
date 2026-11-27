import json, sys, os
import numpy as np
WD = os.path.dirname(os.path.abspath(__file__))
R = '/path/to/vsassm/results'
enc = sys.argv[1]
mine = json.load(open(f'{WD}/my_cells_{enc}.json'))
b53 = json.load(open(f'{R}/b53_part_{enc}.json'))
b60 = json.load(open(f'{R}/b60_part_{enc}.json')) if os.path.exists(f'{R}/b60_part_{enc}.json') else None
b49 = json.load(open(f'{R}/b49_verdict_part_{enc}.json'))
chf = f'{R}/b36_choice_{enc}.json' if os.path.exists(f'{R}/b36_choice_{enc}.json') else f'{R}/b49_choice36_{enc}.json'
ch = json.load(open(chf))['choice']
fmap = {'index': 'IDX', 'static': 'SF', 'cms': 'CM', 'bloom': 'BF', 'fly': 'FBF', 'sdm': 'SDM'}
rows = []
print(f'== {enc}: per condition, mine / pipeline (test accuracy %)')
hdr = ['M', 'V', 'b', 'first', 'final'] + list(fmap) + ['best6', 'fam', 'first-best6', 'final-best6']
print(' | '.join(hdr))
D = {'first_mine': [], 'first_pipe': [], 'final_mine': [], 'final_pipe': [], 'best_mine': [], 'best_pipe': []}
fam_mine = {f: [] for f in fmap}; fam_pipe = {f: [] for f in fmap}
for r in mine['conditions']:
    M, V, b = r['M'], r['V'], r['b']
    c = b53['cells'][f'{enc}|ideal|{M}|{V}|{b}']
    a = c['acc']
    pf = 100 * a['VSAR']; pfin = 100 * c['VSA']
    if b60 is not None:
        s60 = b60['cells'][f'{enc}|ideal|{M}|{V}|{b}']
        assert abs(s60['STORE'] - c['VSA']) < 1e-9 and abs(s60['VSAR'] - a['VSAR']) < 1e-9
    pb = 100 * a['BEST36']
    bestfam = ch[f'{M}|{V}|{b}|ideal|BEST']['fam']
    D['first_mine'].append(r['store_first']); D['first_pipe'].append(pf)
    D['final_mine'].append(r['store_final']); D['final_pipe'].append(pfin)
    D['best_mine'].append(r['best_keyfree']); D['best_pipe'].append(pb)
    cells = [str(M), str(V), str(b), f"{r['store_first']:.1f}/{pf:.1f}", f"{r['store_final']:.1f}/{pfin:.1f}"]
    for f, F in fmap.items():
        fam_mine[f].append(r[f]); fam_pipe[f].append(100 * a[F])
        cells.append(f"{r[f]:.1f}/{100*a[F]:.1f}")
    cells += [f"{r['best_keyfree']:.1f}/{pb:.1f}", f"{r['best_keyfree_family']}/{bestfam}",
              f"{r['store_first']-r['best_keyfree']:+.1f}/{pf-pb:+.1f}", f"{r['store_final']-r['best_keyfree']:+.1f}/{pfin-pb:+.1f}"]
    print(' | '.join(cells))
print('\n== region means (12 conditions), mine / pipeline')
for k in ('first', 'final', 'best'):
    print(k, f"{np.mean(D[k+'_mine']):.2f} / {np.mean(D[k+'_pipe']):.2f}")
for f in fmap: print(f, f"{np.mean(fam_mine[f]):.2f} / {np.mean(fam_pipe[f]):.2f}",
                     f"store_first - {f}: {np.mean(D['first_mine'])-np.mean(fam_mine[f]):+.2f} / {np.mean(D['first_pipe'])-np.mean(fam_pipe[f]):+.2f}")
print('first - best6 (point):', f"{np.mean(D['first_mine'])-np.mean(D['best_mine']):+.2f} / {np.mean(D['first_pipe'])-np.mean(D['best_pipe']):+.2f}")
print('final - best6 (point):', f"{np.mean(D['final_mine'])-np.mean(D['best_mine']):+.2f} / {np.mean(D['final_pipe'])-np.mean(D['best_pipe']):+.2f}")
print('my bootstrap:', mine['region']['store_first_minus_best_keyfree'], mine['region']['store_final_minus_best_keyfree'])
print('pipeline B49 R1 BEST36 (first design):', b49['regions'].get(f'{enc}|ideal|R1|BEST36'))
print('pipeline B53 R1 BEST36 (final design):', b53['regions'].get(f'{enc}|ideal|R1|BEST36'))
print('\n== pipeline dev-chosen key-free configs (from', os.path.basename(chf), ')')
for r in mine['conditions']:
    M, V, b = r['M'], r['V'], r['b']
    print(M, V, b, {F: ch[f'{M}|{V}|{b}|ideal|{F}']['cfg'] for F in fmap.values()})
