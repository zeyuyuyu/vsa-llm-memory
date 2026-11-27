import sys, time, numpy as np
sys.path.insert(0, '.')
from ldpc_own import *
rng = np.random.default_rng(2)
code = build_ira(1024, (1,4))
p=0.16; B=400
err = rng.random((B, code['n'])) < p
Lc=np.log((1-p)/p); L=np.where(err, -Lc, Lc)
out, conv = decode(code, L, 60)
fe = out[:, :1024].any(1)
w = out.sum(1)
print('FER', fe.mean(), 'undetected', (conv&fe).sum(), 'weights of undetected', sorted(w[conv&fe])[:30])
idx = np.nonzero(conv&fe)[0]
for i in idx[:5]:
    nz = np.nonzero(out[i])[0]; print('info ones', nz[nz<1024], 'parity ones count', (nz>=1024).sum(), 'parity range', nz[nz>=1024][:5]-1024, nz[nz>=1024][-5:]-1024)
