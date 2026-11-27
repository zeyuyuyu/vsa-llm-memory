import sys, time, json, math, pickle, numpy as np
sys.path.insert(0, '.')
from sim_ldpc import *
from math import comb

def cap_rep(p, rho):
    # capacity of rho soft-combined copies of BSC(p), uniform input, per coded bit
    if p == 0: return 1.0
    I = 0.0
    for j in range(rho + 1):  # j = number of copies reading opposite to x
        a = comb(rho, j) * p**j * (1-p)**(rho-j)      # P(y=j | x=0)
        b = comb(rho, j) * p**(rho-j) * (1-p)**j      # P(y=j | x=1)
        py = 0.5*(a+b)
        for q in (a, b):
            if q > 0: I += 0.5 * q * math.log2(q/py)
    return I

cond = sys.argv[1]; p = float(sys.argv[2]); j0, j1 = int(sys.argv[3]), int(sys.argv[4])
Ts, Ds = pickle.load(open('../out/tables.pkl','rb'))
budgets = GRID[j0:j1+1]
out = {'cond': cond, 'p': p, 'results': []}
t0 = time.time()
for B in budgets:
    for rate in RATES:
        # feasibility + Shannon pruning on the largest test store
        T = max(Ts, key=lambda T: T.K / T.n)
        mcw, kmax, sizes, mp, coded, rho = design(T.K, int(math.floor(B*T.n)), rate)
        if rho < 1:
            continue
        R = rate[0]/rate[1]
        if R > cap_rep(p, rho):
            out['results'].append(dict(B=B, rate=rate, pruned=True, rho=rho))
            continue
        rec = dict(B=B, rate=rate, rho=rho, pruned=False)
        for lab, S in (('dev', Ds), ('test', Ts)):
            r = run(S, p, [B], [rate], 7, lab)
            if (B, rate) in r:
                share, good, tot, cwf = r[(B, rate)]
                rec[lab] = dict(share=share, good=good, tot=tot, cw_fail=cwf)
            else:
                rec[lab] = None
        out['results'].append(rec)
        print(cond, round(B,1), rate, 'rho', rho, 'dev', rec['dev'] and round(rec['dev']['share'],4), 'test', rec['test'] and round(rec['test']['share'],4), '%.0fs' % (time.time()-t0), flush=True)
    json.dump(out, open('../out/ldpc_%s.json' % cond, 'w'), indent=1)
print('done', time.time()-t0)
