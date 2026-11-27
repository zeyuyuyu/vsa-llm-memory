# Best of six with the SDM family restricted to random (Gaussian) hard-location addresses.
import json, sys, os
import numpy as np
WD = os.path.dirname(os.path.abspath(__file__))
enc = sys.argv[1]
FAMS = ['index', 'static', 'cms', 'bloom', 'fly']
NE = 4622; R = 2000
rng = np.random.default_rng(20261123)
W = rng.multinomial(NE, np.full(NE, 1.0 / NE), size=R).astype(np.float64)
def boot(c, qe):
    w = W[:, qe]; return 100.0 * (w * c[None]).sum(1) / w.sum(1)
rows = []; bs = {'first': [], 'final': [], 'sdmg': []}; pts = {'first': [], 'final': [], 'sdmg': [], 'best': []}
for M in (256, 1024):
    for V in (2, 16):
        for b in (4, 8, 16):
            cell = json.load(open(f'{WD}/out_{enc}/cell_M{M}_V{V}_b{b}.json'))
            corr = dict(np.load(f'{WD}/out_{enc}/corr_M{M}_V{V}_b{b}.npz'))
            sg = json.load(open(f'{WD}/out_{enc}/sens_sdm_gauss_M{M}_V{V}_b{b}.json'))
            sgc = np.load(f'{WD}/out_{enc}/senscorr_sdm_gauss_M{M}_V{V}_b{b}.npy')
            cand = [(cell[f]['dev'], i, f, cell[f]['test'], corr[f]) for i, f in enumerate(FAMS)] + [(sg['dev'], 5, 'sdm_gauss', sg['test'], sgc)]
            best = None
            for c in cand:
                if best is None or c[0] > best[0] + 0.1: best = c
            qe = corr['qe']
            for d in ('first', 'final'):
                bs[d].append(boot(corr[f'store_{d}'].astype(float), qe) - boot(best[4].astype(float), qe))
                pts[d].append(cell[f'store_{d}']['test'] - best[3])
            pts['sdmg'].append(sg['test']); pts['best'].append(best[3])
            rows.append((M, V, b, sg['test'], best[2], best[3]))
            print(M, V, b, f"sdm_gauss={sg['test']:.1f} cfg={sg['cfg'][1:]}", f"best6'={best[2]} {best[3]:.1f}")
out = {}
for d in ('first', 'final'):
    x = np.mean(np.stack(bs[d]), 0)
    out[f'store_{d}_minus_best6_sdmgauss'] = {'mean': round(float(np.mean(pts[d])), 3), 'ci95': [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]}
out['sdm_gauss_region_mean'] = round(float(np.mean(pts['sdmg'])), 3)
out['best6_sdmgauss_region_mean'] = round(float(np.mean(pts['best'])), 3)
out['per_condition'] = [dict(M=r[0], V=r[1], b=r[2], sdm_gauss=round(r[3], 3), best_family=r[4], best=round(r[5], 3)) for r in rows]
json.dump(out, open(f'{WD}/my_cells_{enc}_sdmgauss.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_condition'}, indent=1))
