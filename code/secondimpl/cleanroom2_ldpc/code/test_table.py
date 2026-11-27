import sys, time, numpy as np
sys.path.insert(0, '.')
from table import *
S = load_stores('/path/to/vsassm/b34_out/test_stores.json.gz')
for lam in [4.0, 5.0, 6.0]:
    t0=time.time(); tot=0; rows=[]
    for name, tr in S[:1]+S[10:11]+S[20:21]+S[30:31]+S[40:41]:
        T = Table(name, tr, lam=lam)
        ok = T.read_exact(T.bits)
        rows.append((name, T.n, T.N, T.b, T.r, T.w, round(T.dir_bits/T.n,2), round(T.K/T.n,2), ok.all(), len(T.probe)))
    print(lam, time.time()-t0)
    for r in rows: print(r)
