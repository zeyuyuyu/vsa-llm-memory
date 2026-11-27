import sys, time, json, pickle, numpy as np
sys.path.insert(0, '.')
from sim_bch import *
Ts, Ds = pickle.load(open('../out/tables.pkl','rb'))
p = float(sys.argv[1]); cond = sys.argv[2]; j0, j1 = int(sys.argv[3]), int(sys.argv[4])
variants = {'main': dict(shorten=True, allow_even=True),
            'no_shorten': dict(shorten=False, allow_even=True),
            'odd_rho_only': dict(shorten=True, allow_even=False)}
out = {'p': p, 'cond': cond, 'results': []}
t0 = time.time()
for vn, kw in variants.items():
    for B in GRID[j0:j1+1]:
        for seed in range(3):
            rng = np.random.default_rng([seed, int(B*1000), int(p*1e6)])
            good = tot = 0; designs = []
            feas = True
            for T in Ts:
                r = simulate(T, p, B, rng, **kw)
                if r is None: feas = False; break
                ok, d = r; good += int(ok.sum()); tot += len(ok); designs.append(d)
            rec = dict(variant=vn, B=B, seed=seed, feasible=feas)
            if feas:
                big = designs[0]
                rec.update(share=good/tot, good=good, tot=tot, big_design=dict(m=big['m'], t=big['t'], rho=big['rho'], pf=big['pf'], ncw=big['ncw']))
            out['results'].append(rec)
            print(vn, round(B,1), seed, rec.get('share'), rec.get('big_design'), '%.0fs'%(time.time()-t0), flush=True)
            if seed == 0 and feas and rec['share'] < 0.9: break
    json.dump(out, open('../out/bch_%s.json' % cond, 'w'), indent=1)
