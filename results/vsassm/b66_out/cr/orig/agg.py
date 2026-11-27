import json, sys, glob, os
import numpy as np
WD = os.path.dirname(os.path.abspath(__file__))
enc = sys.argv[1]
FAMS = ['index', 'static', 'cms', 'bloom', 'fly', 'sdm']
conds = [(M, V, b) for M in (256, 1024) for V in (2, 16) for b in (4, 8, 16)]
cells, corr = {}, {}
for M, V, b in conds:
    f = f'{WD}/out_{enc}/cell_M{M}_V{V}_b{b}.json'
    if not os.path.exists(f): continue
    cells[(M, V, b)] = json.load(open(f))
    corr[(M, V, b)] = dict(np.load(f'{WD}/out_{enc}/corr_M{M}_V{V}_b{b}.npz'))
done = [c for c in conds if c in cells]
print(f'{len(done)} of 12 conditions')
NE = 4622; R = 2000
rng = np.random.default_rng(20261123)
W = rng.multinomial(NE, np.full(NE, 1.0 / NE), size=R).astype(np.float64)  # (R, NE) entity weights

def boot_acc(c, qe):
    w = W[:, qe]
    return 100.0 * (w * c[None]).sum(1) / w.sum(1)

arms = ['store_first', 'store_final'] + FAMS + ['best_keyfree']
out = {'encoder': enc, 'conditions': [], 'region': {}}
bt = {a: [] for a in arms}
for k in done:
    M, V, b = k
    row = {'M': M, 'V': V, 'b': b, 'n_test_queries': cells[k]['n_test_queries'],
           'best_keyfree_family': cells[k]['best_keyfree']['family']}
    for a in arms:
        row[a] = round(cells[k][a]['test'], 3)
        if a != 'best_keyfree' and 'cfg' in cells[k][a]: row[a + '_cfg'] = cells[k][a]['cfg']
        bt[a].append(boot_acc(corr[k][a].astype(np.float64), corr[k]['qe']))
    for d in ('first', 'final'):
        row[f'store_{d}_minus_best_keyfree'] = round(row[f'store_{d}'] - row['best_keyfree'], 3)
    out['conditions'].append(row)
for a in arms:
    out['region'][a] = round(float(np.mean([r[a] for r in out['conditions']])), 3)
for d in ('first', 'final'):
    pt = float(np.mean([r[f'store_{d}'] - r['best_keyfree'] for r in out['conditions']]))
    bs = np.mean(np.stack(bt[f'store_{d}']) - np.stack(bt['best_keyfree']), 0)
    out['region'][f'store_{d}_minus_best_keyfree'] = {'mean': round(pt, 3), 'ci95': [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    for f in FAMS:
        pt = float(np.mean([r[f'store_{d}'] - r[f] for r in out['conditions']]))
        bs = np.mean(np.stack(bt[f'store_{d}']) - np.stack(bt[f]), 0)
        out['region'][f'store_{d}_minus_{f}'] = {'mean': round(pt, 3), 'ci95': [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    # per-condition intervals
    for i, r in enumerate(out['conditions']):
        bs = bt[f'store_{d}'][i] - bt['best_keyfree'][i]
        r[f'store_{d}_minus_best_keyfree_ci95'] = [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
out['notes'] = 'cluster bootstrap over test entities (2000 replicates, shared across conditions); ideal accounting; region R1 = V in {2,16}, b in {4,8,16}, M in {256,1024}'
json.dump(out, open(f'{WD}/my_cells_{enc}.json', 'w'), indent=1)
print(json.dumps(out['region'], indent=1))
for r in out['conditions']:
    print(r['M'], r['V'], r['b'], 'first %.1f final %.1f' % (r['store_first'], r['store_final']),
          ' '.join(f'{f}={r[f]:.1f}' for f in FAMS), 'best=%s %.1f' % (r['best_keyfree_family'], r['best_keyfree']))
