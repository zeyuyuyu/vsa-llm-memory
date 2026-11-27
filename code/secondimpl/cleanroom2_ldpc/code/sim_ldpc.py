"""Clean-room LDPC-protected table on i.i.d. cell errors (BSC with crossover p).
Every logical bit (MPH directory + rows) is LDPC-coded; codewords of <= 1024 info bits;
rho copies of every coded bit, soft-combined; sum-product decoding, 60 iterations.
All-zero-codeword simulation is exact for BP on a BSC (linear code, symmetric channel and decoder):
the decoded error pattern is XOR-ed onto the true logical bits.
"""
import sys, os, json, math, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table import Table, load_stores
from ldpc_own import build_ira, decode, RATES, parity_len

GRID = [8 * 2 ** (j / 4) for j in range(37)]
CODES = {}
BIG = 40.0


def get_code(k, rate):
    key = (k, rate)
    if key not in CODES:
        CODES[key] = build_ira(k, rate, seed=0)
    return CODES[key]


def design(K, cells, rate):
    mcw = math.ceil(K / 1024)
    kmax = math.ceil(K / mcw)
    sizes = [K // mcw + (1 if i < K % mcw else 0) for i in range(mcw)]
    mp = parity_len(kmax, rate)
    coded = sum(s + mp for s in sizes)
    rho = cells // coded
    return mcw, kmax, sizes, mp, coded, rho


def simulate_store(T, p, B, rate, rng, max_iter=60):
    cells = int(math.floor(B * T.n))
    mcw, kmax, sizes, mp, coded, rho = design(T.K, cells, rate)
    if rho < 1:
        return None
    code = get_code(kmax, rate)
    n = code['n']
    Lc = math.log((1 - p) / p) if p > 0 else BIG
    L = np.full((mcw, n), BIG)
    errs = rng.binomial(rho, p, size=(mcw, n)) if p > 0 else np.zeros((mcw, n), dtype=np.int64)
    L = Lc * (rho - 2 * errs).astype(np.float64)
    # shortened (known-zero) info positions: the tail of each smaller codeword's info part
    for i, s in enumerate(sizes):
        if s < kmax:
            L[i, s:kmax] = BIG
    out, conv = decode(code, L, max_iter)
    e = np.concatenate([out[i, :s] for i, s in enumerate(sizes)])
    bits = T.bits ^ e
    ok = T.read_exact(bits)
    return ok, dict(rho=rho, mcw=mcw, kmax=kmax, cells_used=rho * coded, cells=cells,
                    cw_fail=int((out[:, :kmax].any(1)).sum()))


def build_tables(path, lam_choices=(3.0, 4.0, 5.0)):
    S = load_stores(path)
    Ts = []
    for name, tr in S:
        best = None
        for lam in lam_choices:
            T = Table(name, tr, lam=lam)
            if best is None or T.K < best.K:
                best = T
        Ts.append(best)
    return Ts


def run(Ts, p, budgets, rates, seed, label):
    res = {}
    for B in budgets:
        for rate in rates:
            rng = np.random.default_rng([seed, int(B * 1000), rate[0], rate[1], int(p * 1e6)])
            tot = 0; good = 0; feas = True; info = []
            for T in Ts:
                r = simulate_store(T, p, B, rate, rng)
                if r is None:
                    feas = False; break
                ok, d = r
                tot += len(ok); good += int(ok.sum()); info.append(d['cw_fail'])
            if feas:
                res[(B, rate)] = (good / tot, good, tot, sum(info))
    return res


if __name__ == '__main__':
    pass
