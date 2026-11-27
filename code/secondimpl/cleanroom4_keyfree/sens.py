# Sensitivity arms: re-select a restricted sub-grid on the same dev stores and score once on the same test stores.
import sys, os, json, time
import numpy as np
import cr

FILTERS = {
    # Kanerva-style SDM only: random (Gaussian) hard-location addresses, no data-derived addresses
    'sdm_gauss': lambda g: g[0] == 'sdm' and g[2] == 'gauss',
    # SDM with data addresses only
    'sdm_data': lambda g: g[0] == 'sdm' and g[2] == 'data',
}

def run(args):
    enc, name, M, V, b = args
    t0 = time.time()
    B = b * M; cs = M * 1000 + V * 10 + b
    ndev, ntest = (6, 20) if M == 256 else (3, 5)
    dev = cr.make_stores('dev', M, V, ndev, 1000 + cs)
    test = cr.make_stores('test', M, V, ntest, 2000 + cs)
    grid = [g for g in cr.family_grid() if FILTERS[name](g)]
    devpreds = [cr.run_family_cfgs(st, B, V, grid, seed=5000 + si) for si, st in enumerate(dev)]
    devacc = {}
    for g in grid:
        pl = [dp[g] for dp in devpreds]
        devacc[g] = -1.0 if any(p is None for p in pl) else cr.acc_of(dev, pl)[0]
    ch = cr.choose(grid, devacc)
    pl = [cr.run_family_cfgs(st, B, V, [ch], seed=6000 + si)[ch] for si, st in enumerate(test)]
    rng = np.random.default_rng(7)
    pl = [p if p is not None else cr.chance_preds(st, V, rng) for p, st in zip(pl, test)]
    a, c = cr.acc_of(test, pl)
    out = {'enc': enc, 'arm': name, 'M': M, 'V': V, 'b': b, 'cfg': list(map(str, ch)), 'dev': devacc[ch], 'test': a}
    json.dump(out, open(f'{cr.WD}/out_{enc}/sens_{name}_M{M}_V{V}_b{b}.json', 'w'), indent=1)
    np.save(f'{cr.WD}/out_{enc}/senscorr_{name}_M{M}_V{V}_b{b}.npy', c)
    print(f'[{enc} {name} M{M} V{V} b{b}] test={a:.2f} cfg={ch} {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    enc, name, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cr.setup(enc)
    conds = [(enc, name, M, V, b) for M in (1024, 256) for V in (2, 16) for b in (4, 8, 16)]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(nproc) as pool:
        for _ in pool.imap_unordered(run, conds): pass
