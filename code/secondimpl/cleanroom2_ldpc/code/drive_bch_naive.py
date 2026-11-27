import sys, time, json, pickle, numpy as np
sys.path.insert(0, '.')
import sim_bch as sb
from sim_bch import *
# variant: BCH dimension taken as k = n - m*t (the usual upper bound on parity), instead of the exact
# cyclotomic-coset dimension
for m in sb.BCH:
    n, d = sb.BCH[m]
    sb.BCH[m] = (n, {t: n - m * t for t in d if n - m * t >= 1})
Ts, Ds = pickle.load(open('../out/tables.pkl','rb'))
p = 0.123
out = []
for B in GRID[15:18]:
    for seed in range(3):
        rng = np.random.default_rng([seed, int(B*1000), int(p*1e6)])
        good = tot = 0; ds = []
        for T in Ts:
            ok, d = simulate(T, p, B, rng); good += int(ok.sum()); tot += len(ok); ds.append(d)
        out.append(dict(B=B, seed=seed, share=good/tot, big=dict(m=ds[0]['m'], t=ds[0]['t'], rho=ds[0]['rho'], pf=ds[0]['pf'])))
        print('naive_dim', round(B,1), seed, good/tot, out[-1]['big'], flush=True)
json.dump(out, open('../out/bch_k8_naive_dim.json','w'), indent=1)
