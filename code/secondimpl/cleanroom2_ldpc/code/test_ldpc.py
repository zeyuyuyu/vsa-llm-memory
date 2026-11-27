import sys, time, numpy as np
sys.path.insert(0, '.')
from ldpc_own import *
rng = np.random.default_rng(1)
for rate in [(1,4),(1,3),(1,2),(9,10)]:
    t0=time.time(); code = build_ira(1024, rate); tb=time.time()-t0
    # syndrome check for random codewords
    u = rng.integers(0,2,1024).astype(np.uint8); c = encode(code, u)
    cp, cv = code['chk_ptr'], code['chk_var']
    syn = [c[cv[cp[i]:cp[i+1]]].sum()&1 for i in range(code['m'])]
    print(rate, 'n',code['n'],'E',code['E'],'build %.1fs'%tb,'syndrome ok', not any(syn))
# FER vs p, nonzero codewords (test symmetry) and zero
for rate, ps in [((1,2),[0.06,0.07,0.08]),((1,3),[0.12,0.13,0.14]),((1,4),[0.16,0.18,0.19])]:
    code = build_ira(1024, rate)
    for p in ps:
        B=200
        U = rng.integers(0,2,(B,1024)).astype(np.uint8)
        C = np.array([encode(code,u) for u in U[:50]])
        C = np.concatenate([C, np.zeros((B-50, code['n']),dtype=np.uint8)])
        U2 = np.concatenate([U[:50], np.zeros((B-50,1024),dtype=np.uint8)])
        err = rng.random(C.shape) < p
        y = C ^ err
        Lc = np.log((1-p)/p)
        L = np.where(y==0, Lc, -Lc)
        t0=time.time(); out, conv = decode(code, L, 60); td=time.time()-t0
        fe = (out[:, :1024] != U2).any(1)
        print(rate, p, 'FER %.3f (nonzero %.3f, zero %.3f)'%(fe.mean(), fe[:50].mean(), fe[50:].mean()), 'conv', conv.mean(), 'undetected', (conv & fe).sum(), '%.2fs'%td)
