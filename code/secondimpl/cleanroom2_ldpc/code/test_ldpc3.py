import sys, time, numpy as np
sys.path.insert(0, '.')
from ldpc_own import *
rng = np.random.default_rng(3)
for seed in [0,1]:
    code = build_ira(1024, (1,3), seed=seed)
    for p in [0.11, 0.123]:
        B=1000
        err = rng.random((B, code['n'])) < p
        Lc=np.log((1-p)/p); L=np.where(err, -Lc, Lc)
        out, conv = decode(code, L, 60)
        ie = out[:, :1024].sum(1)
        fe = ie>0
        print('seed',seed,'p',p,'FER',fe.mean(),'info-bit errors in failures', sorted(ie[fe].tolist())[:40], 'nonconv', (~conv).sum())
