import sys, time, numpy as np, pickle
sys.path.insert(0, '.')
from sim_ldpc import *
t0=time.time()
Ts = build_tables('/path/to/vsassm/b34_out/test_stores.json.gz')
Ds = build_tables('/path/to/vsassm/b34_out/dev_stores.json.gz')
pickle.dump((Ts, Ds), open('../out/tables.pkl','wb'))
print('tables', time.time()-t0, 'probed', sum(len(T.probe) for T in Ts), sum(len(T.probe) for T in Ds))
print('max K/n', max(T.K/T.n for T in Ts), 'dir bits/pair big', [round(T.dir_bits/T.n,2) for T in Ts[:10]])
for B in [GRID[11], GRID[12]]:
    for rate in [(1,3),(2,5)]:
        t0=time.time()
        r = run(Ts, 0.123, [B], [rate], 0, 't')
        print(round(B,1), rate, r, '%.1fs'%(time.time()-t0))
