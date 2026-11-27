# repeat channel realizations for fixed designs near the threshold
import sys, json, pickle, numpy as np
sys.path.insert(0, '.')
from sim_ldpc import *
Ts, Ds = pickle.load(open('../out/tables.pkl','rb'))
cond = sys.argv[1]; p = float(sys.argv[2]); B = float(sys.argv[3]); rate = (int(sys.argv[4]), int(sys.argv[5])); nseed = int(sys.argv[6])
res = []
for s in range(100, 100 + nseed):
    rng = np.random.default_rng([s, int(B*1000), rate[0], rate[1], int(p*1e6)])
    good = tot = 0
    for T in Ts:
        ok, d = simulate_store(T, p, B, rate, rng); good += int(ok.sum()); tot += len(ok)
    res.append(good / tot); print(cond, round(B,1), rate, s, good/tot, flush=True)
fn = f'../out/seeds_{cond}_{round(B,1)}_{rate[0]}-{rate[1]}.json'
json.dump(dict(cond=cond, p=p, B=B, rate=rate, shares=res), open(fn, 'w'))
