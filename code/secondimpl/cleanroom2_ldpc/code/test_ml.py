import sys, numpy as np, itertools
sys.path.insert(0,'.')
from ldpc_own import *
rng=np.random.default_rng(5)
for k, rate, p in [(12,(1,2),0.05),(12,(1,2),0.1),(10,(1,3),0.12),(14,(1,2),0.08)]:
    code=build_ira(k,rate,seed=1)
    U=np.array(list(itertools.product([0,1],repeat=k)),dtype=np.uint8)
    CW=np.array([encode(code,u) for u in U])
    B=3000
    err=(rng.random((B,code['n']))<p).astype(np.uint8)
    Lc=np.log((1-p)/p); L=np.where(err==1,-Lc,Lc)
    out,conv=decode(code,L,60)
    bp_fail=out[:,:k].any(1)
    # ML on BSC: min Hamming distance from received (= err, since all-zero sent)
    d=(err[:,None,:]^CW[None,:,:]).sum(2)
    dmin=d.min(1); ml_fail = d[:,0] > dmin  # strictly closer codeword exists
    ml_tie = (d[:,0]==dmin) & ((d==dmin[:,None]).sum(1)>1)
    print(k,rate,p,'n',code['n'],'BP FER %.4f'%bp_fail.mean(),'ML FER %.4f (+ties %.4f)'%(ml_fail.mean(), ml_tie.mean()/2), 'mind', CW[1:].sum(1).min())
