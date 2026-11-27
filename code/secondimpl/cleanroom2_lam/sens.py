"""POST-HOC sensitivity (not used for closure): whitening floor eps for Qwen store and LS on fixed configs."""
import sys, json
import numpy as np
sys.argv = ['x', 'qwen8', '/dev/null', 'centred']
src = open('cr_equalbit_fast2.py').read()
exec(src[:src.index('for white in')])
G1 = gaussian([True, 1], DMAX, n)
for rel in [1e-4, 1e-3, 1e-2, 1e-1]:
    eps = rel * lam.mean()
    Pt = prep(SPL['test']['rows'], True)
    Z = Pt @ G1.T
    res = []
    for (M, V, b) in [(1024, 2, 16), (256, 2, 16), (256, 16, 16)]:
        d = store_dim(b * M, 1); CV = value_codes(V, d); c = t = 0
        for st in STORES[('test', M, V)]:
            Fl = Z[st['lab'], :d]; Fq = Z[st['qrows'], :d]
            bq = quant_vec((Fl * CV[st['vals']]).sum(0), 1)
            pr = ((Fq * bq[None]) @ CV.T).argmax(1); c += (pr == st['qvals']).sum(); t += len(pr)
        res.append(f'store {M}|{V}|{b} {100*c/t:.2f}')
    print(f'eps_rel={rel:g}', ' ; '.join(res), flush=True)
