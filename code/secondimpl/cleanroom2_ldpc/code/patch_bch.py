src = open('sim_bch.py').read()
start = src.index('def choose(')
end = src.index('def simulate(')
new = '''from functools import lru_cache


@lru_cache(maxsize=None)
def pfail_c(nn, t, a, b):
    return pfail(nn, t, a, b)


def choose(T, p, cells, shorten=True, allow_even=True, rho_max=64):
    best = None
    K = T.K
    for m, (n, dims) in BCH.items():
        tt = np.array(sorted(dims)); kk = np.array([dims[t] for t in tt])
        c = -(-K // kk)
        tot = (K + c * (n - kk)) if shorten else c * n
        for rho in range(1, rho_max + 1):
            if not allow_even and rho % 2 == 0:
                continue
            if rho * K > cells:
                break
            fit = np.nonzero(rho * tot <= cells)[0]
            if len(fit) == 0:
                continue
            a, b = maj_probs(p, rho)
            for i in fit[-6:]:
                t, k = int(tt[i]), int(kk[i])
                sizes, lens = layout(K, n, k, shorten)
                pf = 0.0
                for L in set(lens):
                    w = sum(s for s, LL in zip(sizes, lens) if LL == L)
                    pf += w * pfail_c(int(L), t, round(a, 12), round(b, 12))
                pf /= K
                if best is None or pf < best[0]:
                    best = (pf, m, t, rho, sizes, lens)
    return best


'''
open('sim_bch.py','w').write(src[:start] + new + src[end:])
