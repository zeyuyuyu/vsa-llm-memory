# Analysis only (disclosed after the test): accuracy by the fact's position in the write order (terciles), pooled over
# the R1 cells of each V, for STORE, LAMc, NLMSc. Write order = b32_run.store_order (seed ('subset', split, M, V, rep)).
import sys, numpy as np
sys.path.insert(0, '/path/to/vsassm')
import b62_verdict as VV, b62_run as R62
for enc in sys.argv[1:]:
    data, ch = VV.load(enc)
    for V in (2, 16):
        acc = {a: np.zeros((3, 2)) for a in ('STORE', 'LAMC', 'NLMSC', 'RLSC')}
        for M in (256, 1024):
            for r in data[M, V]:
                o = R62.store_order({'split': 'test', 'M': M, 'V': V, 'rep': r['rep']})
                pos = np.empty(M, int); pos[o] = np.arange(M)
                ter = np.minimum(3 * pos[np.array(r['qent'])] // M, 2)
                g = np.array(r['qgold'])
                for b in (4, 8, 16):
                    for a in acc:
                        ok = np.array(r['pred'][f'{a}|{b}|ideal']) == g
                        for t in range(3):
                            acc[a][t, 0] += ok[ter == t].sum(); acc[a][t, 1] += (ter == t).sum()
        print(enc, f'V={V}', '  '.join(f'{a}: ' + ' / '.join(f'{100 * x[t, 0] / x[t, 1]:.1f}' for t in range(3))
                                     for a, x in acc.items()), '(first / middle / last third written)', flush=True)
