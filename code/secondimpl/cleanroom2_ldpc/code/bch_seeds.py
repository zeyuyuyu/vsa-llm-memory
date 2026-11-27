import sys, json, pickle, numpy as np
sys.path.insert(0, '.')
from sim_bch import *
Ts, Ds = pickle.load(open('../out/tables.pkl', 'rb'))
p = 0.123; out = {}
for B in [GRID[15], GRID[14]]:
    sh = []
    for seed in range(100, 112):
        rng = np.random.default_rng([seed, int(B * 1000), int(p * 1e6)])
        good = tot = 0
        for T in Ts:
            ok, d = simulate(T, p, B, rng); good += int(ok.sum()); tot += len(ok)
        sh.append(good / tot); print(round(B, 1), seed, good / tot, flush=True)
    out[str(round(B, 1))] = sh
json.dump(out, open('../out/bch_k8_extra_seeds.json', 'w'), indent=1)
