import sys, json, pickle, copy, numpy as np
sys.path.insert(0, '.')
from sim_bch import *
Ts, Ds = pickle.load(open('../out/tables.pkl','rb'))
p = 0.123; out = []
for pad in [0.4, 0.8, 1.5]:
    Tp = []
    for T in Ts:
        T2 = copy.copy(T); extra = int(round(pad * T.n))
        T2.bits = np.concatenate([T.bits, np.zeros(extra, dtype=np.uint8)]); T2.K = len(T2.bits)
        Tp.append(T2)
    for B in GRID[15:17]:
        for seed in range(3):
            rng = np.random.default_rng([seed, int(B*1000), int(p*1e6)])
            good = tot = 0
            for T in Tp:
                ok, d = simulate(T, p, B, rng); good += int(ok[:len(T.probe)].sum()); tot += len(ok)
            out.append(dict(pad_bits_per_pair=pad, B=B, seed=seed, share=good/tot))
            print('pad', pad, round(B,1), seed, good/tot, flush=True)
json.dump(out, open('../out/bch_k8_directory_pad.json','w'), indent=1)
