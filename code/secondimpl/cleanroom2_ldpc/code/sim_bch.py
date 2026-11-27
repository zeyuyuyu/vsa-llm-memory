"""Clean-room shortened-BCH-protected table on i.i.d. cell errors.
Narrow-sense binary BCH, n = 2^m - 1 (m = 6..10), designed t; dimension from cyclotomic cosets.
Each coded bit repeated rho times; majority per bit, ties = erasures; errors-and-erasures
bounded-distance decoding succeeds iff 2e + f <= 2t. (n, t, rho) chosen per store by closed form.
"""
import sys, os, math, pickle, json, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table import Table

GRID = [8 * 2 ** (j / 4) for j in range(37)]


def bch_dims(m):
    n = 2 ** m - 1
    covered = set()
    dims = {}
    for t in range(1, n // 2 + 1):
        i = 2 * t - 1
        if i not in covered:
            x = i
            while True:
                covered.add(x); x = (2 * x) % n
                if x == i:
                    break
        # designed distance 2t+1 needs roots alpha^1..alpha^2t; even powers are in cosets of odd ones
        k = n - len(covered)
        if k < 1:
            break
        dims[t] = k
    return n, dims


BCH = {m: bch_dims(m) for m in range(6, 11)}


def maj_probs(p, rho):
    """per-bit after majority over rho copies: (P(error), P(erasure))"""
    j = np.arange(rho + 1)
    pm = binom.pmf(j, rho, p)  # j = number of wrong copies
    err = pm[j * 2 > rho].sum()
    era = pm[j * 2 == rho].sum()
    return float(err), float(era)


def pfail(nn, t, a, b):
    """P(2E + F > 2t) for nn bits with per-bit error a, erasure b."""
    if b == 0:
        return float(binom.sf(t, nn, a))
    f = np.arange(0, min(nn, 2 * t) + 1)
    pf = binom.pmf(f, nn, b)
    q = a / (1 - b) if b < 1 else 0.0
    emax = (2 * t - f) // 2
    pe = binom.cdf(emax, nn - f, q)
    return float(1 - (pf * pe).sum())


def layout(K, n, k, shorten=True):
    c = math.ceil(K / k)
    if shorten:
        sizes = [K // c + (1 if i < K % c else 0) for i in range(c)]
        lens = [s + (n - k) for s in sizes]
    else:
        sizes = [min(k, K - i * k) for i in range(c)]
        lens = [n] * c
    return sizes, lens


from functools import lru_cache


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


def simulate(T, p, B, rng, shorten=True, allow_even=True):
    cells = int(math.floor(B * T.n))
    ch = choose(T, p, cells, shorten, allow_even)
    if ch is None:
        return None
    pf, m, t, rho, sizes, lens = ch
    est = []
    for s, L in zip(sizes, lens):
        wrong = rng.binomial(rho, p, size=L)
        err = wrong * 2 > rho
        era = wrong * 2 == rho
        ok = 2 * err.sum() + era.sum() <= 2 * t
        if ok:
            est.append(np.zeros(s, dtype=np.uint8))
        else:
            # failure: hand back the majority bits (ties broken at random) for the info part
            e = err[:s].copy()
            e |= era[:s] & (rng.random(s) < 0.5)
            est.append(e.astype(np.uint8))
    e = np.concatenate(est)
    bits = T.bits ^ e
    return T.read_exact(bits), dict(m=m, t=t, rho=rho, pf=pf, ncw=len(sizes))
