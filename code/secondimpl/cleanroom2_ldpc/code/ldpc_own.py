"""Clean-room IRA LDPC codes and a sum-product (belief-propagation) decoder.
No external LDPC package is used.

IRA parity-check matrix H = [H1 | H2], H2 the m x m dual-diagonal accumulator.
Info-bit degree profiles follow the DVB-S2 IRA profiles for each rate (fraction of
high-degree info bits, rest degree 3); edges placed at random with balanced check
degrees, no repeated edges, and a greedy pass that removes 4-cycles where it can.
"""
import numpy as np
from numba import njit, prange

RATES = [(1, 4), (1, 3), (2, 5), (1, 2), (3, 5), (2, 3), (3, 4), (4, 5), (5, 6), (8, 9), (9, 10)]
# (fraction of info bits with high degree, high degree) -- DVB-S2 normal-frame profiles
PROFILE = {(1, 4): (1/3, 12), (1, 3): (1/3, 12), (2, 5): (1/3, 12), (1, 2): (0.4, 8),
           (3, 5): (1/3, 12), (2, 3): (0.1, 13), (3, 4): (1/9, 12), (4, 5): (1/8, 11),
           (5, 6): (0.1, 13), (8, 9): (1/8, 4), (9, 10): (1/9, 4)}


def parity_len(k, rate):
    a, b = rate
    return int(round(k * (b - a) / a))


def build_ira(k, rate, seed=0):
    """Returns dict with n, m, check->var CSR (chk_ptr, chk_var) and var->edge CSR."""
    m = parity_len(k, rate)
    n = k + m
    rng = np.random.default_rng(seed * 1000003 + k * 31 + rate[0] * 7 + rate[1])
    frac, dh = PROFILE[rate]
    dh = min(dh, m)
    nh = int(round(frac * k))
    deg = np.full(k, min(3, m), dtype=np.int64)
    deg[:nh] = dh
    rng.shuffle(deg)
    E1 = int(deg.sum())
    # balanced check targets
    base = E1 // m; extra = E1 - base * m
    cap = np.full(m, base, dtype=np.int64)
    cap[rng.choice(m, extra, replace=False)] += 1
    # socket lists
    var_sockets = np.repeat(np.arange(k), deg)
    chk_sockets = np.repeat(np.arange(m), cap)
    rng.shuffle(chk_sockets)
    # assign; fix duplicates by swapping
    rows = [set() for _ in range(m)]
    cols = [[] for _ in range(k)]
    pairs = list(zip(var_sockets.tolist(), chk_sockets.tolist()))
    # greedy placement with duplicate repair
    chk_of = np.array(chk_sockets)
    for it in range(20):
        seen = set(); bad = []
        for i, (v, c) in enumerate(zip(var_sockets.tolist(), chk_of.tolist())):
            if (v, c) in seen:
                bad.append(i)
            else:
                seen.add((v, c))
        if not bad:
            break
        for i in bad:
            j = rng.integers(len(chk_of))
            chk_of[i], chk_of[j] = chk_of[j], chk_of[i]
    edges = sorted(set(zip(var_sockets.tolist(), chk_of.tolist())))
    H1cols = [[] for _ in range(k)]
    for v, c in edges:
        H1cols[v].append(c)
    # 4-cycle removal (greedy, limited): for each pair of info cols sharing 2 checks, move one edge
    chk_rows = [set() for _ in range(m)]
    for v in range(k):
        for c in H1cols[v]:
            chk_rows[c].add(v)

    def cycles_of(v):
        cnt = {}
        for c in H1cols[v]:
            for u in chk_rows[c]:
                if u != v:
                    cnt[u] = cnt.get(u, 0) + 1
        return [u for u, x in cnt.items() if x >= 2]

    for sweep in range(2):
        moved = 0
        for v in range(k):
            if not cycles_of(v):
                continue
            for ci, c in enumerate(list(H1cols[v])):
                # try moving edge (v,c) to another check that gives no 4-cycle
                for _ in range(30):
                    c2 = int(rng.integers(m))
                    if c2 in H1cols[v]:
                        continue
                    # neighbours of v through other checks
                    nb = set()
                    for cc in H1cols[v]:
                        if cc != c:
                            nb |= chk_rows[cc]
                    nb.discard(v)
                    if nb & chk_rows[c2]:
                        continue
                    chk_rows[c].discard(v); chk_rows[c2].add(v)
                    H1cols[v][ci] = c2; moved += 1
                    break
                if not cycles_of(v):
                    break
        if moved == 0:
            break
    # low-weight-codeword repair for the accumulator: a set S of info bits gives a codeword of
    # weight |S| + sum of interval lengths between the sorted (XOR-ed) check positions.
    # Re-draw degree-<=3 info nodes whose single or pairwise codeword weight is below wmin.
    small = np.array([v for v in range(k) if len(H1cols[v]) <= 3], dtype=np.int64)
    wmin = max(8, min(40, m // 40))
    for rep in range(6):
        C = np.full((len(small), 3), -1, dtype=np.int64)
        for i, v in enumerate(small):
            cs = sorted(H1cols[v]); C[i, :len(cs)] = cs
        badw = _lowweight(C, m, wmin)
        bad = small[badw]
        if len(bad) == 0:
            break
        for v in bad:
            d = len(H1cols[v])
            for c in H1cols[v]:
                chk_rows[c].discard(v)
            for _ in range(200):
                cs = sorted(rng.choice(m, d, replace=False).tolist())
                nb = set()
                for cc in cs:
                    nb |= chk_rows[cc]
                # avoid 4-cycles: no other var shares two of these checks
                cnt = {}
                okc = True
                for cc in cs:
                    for u in chk_rows[cc]:
                        cnt[u] = cnt.get(u, 0) + 1
                        if cnt[u] >= 2:
                            okc = False
                if okc:
                    break
            H1cols[v] = cs
            for c in cs:
                chk_rows[c].add(v)
    self_lw = _lowweight
    # assemble edge list (check, var)
    ec = []; ev = []
    for v in range(k):
        for c in H1cols[v]:
            ec.append(c); ev.append(v)
    for j in range(m):  # parity column j: rows j and j+1
        ec.append(j); ev.append(k + j)
        if j + 1 < m:
            ec.append(j + 1); ev.append(k + j)
    ec = np.array(ec, dtype=np.int64); ev = np.array(ev, dtype=np.int64)
    o = np.lexsort((ev, ec))
    ec = ec[o]; ev = ev[o]
    E = len(ec)
    chk_ptr = np.zeros(m + 1, dtype=np.int64)
    np.add.at(chk_ptr, ec + 1, 1); chk_ptr = np.cumsum(chk_ptr)
    ov = np.argsort(ev, kind='stable')
    var_ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(var_ptr, ev + 1, 1); var_ptr = np.cumsum(var_ptr)
    return dict(n=n, m=m, k=k, E=E, chk_ptr=chk_ptr, chk_var=ev, var_ptr=var_ptr, var_edge=ov.astype(np.int64))


def encode(code, u):
    """Systematic IRA encoding (used only for self-tests). u: (k,) uint8 -> codeword (n,)"""
    k, m = code['k'], code['m']
    s = np.zeros(m, dtype=np.uint8)
    cp, cv = code['chk_ptr'], code['chk_var']
    for c in range(m):
        vs = cv[cp[c]:cp[c + 1]]
        vs = vs[vs < k]
        s[c] = u[vs].sum() & 1
    p = np.cumsum(s) & 1
    return np.concatenate([u, p.astype(np.uint8)])


@njit(cache=True)
def _decode_one(L, chk_ptr, chk_var, var_ptr, var_edge, max_iter, out, tmp_fw):
    n = L.shape[0]
    m = chk_ptr.shape[0] - 1
    E = chk_var.shape[0]
    c2v = np.zeros(E)
    v2c = np.empty(E)
    hard = np.empty(n, dtype=np.uint8)
    post = np.empty(n)
    # init
    for e in range(E):
        v2c[e] = L[chk_var[e]]
    iters_used = max_iter
    converged = False
    for it in range(max_iter):
        # check update (sum-product, leave-one-out by prefix/suffix products)
        for c in range(m):
            a = chk_ptr[c]; b = chk_ptr[c + 1]
            d = b - a
            # forward products
            p = 1.0
            for i in range(d):
                x = v2c[a + i]
                if x > 40.0:
                    x = 40.0
                elif x < -40.0:
                    x = -40.0
                t = np.tanh(0.5 * x)
                tmp_fw[i] = p
                p *= t
            q = 1.0
            for i in range(d - 1, -1, -1):
                x = v2c[a + i]
                if x > 40.0:
                    x = 40.0
                elif x < -40.0:
                    x = -40.0
                t = np.tanh(0.5 * x)
                pr = tmp_fw[i] * q
                if pr > 0.999999999999:
                    pr = 0.999999999999
                elif pr < -0.999999999999:
                    pr = -0.999999999999
                c2v[a + i] = 2.0 * np.arctanh(pr)
                q *= t
        # variable update
        for v in range(n):
            a = var_ptr[v]; b = var_ptr[v + 1]
            s = L[v]
            for i in range(a, b):
                s += c2v[var_edge[i]]
            post[v] = s
            hard[v] = 1 if s < 0 else 0
            for i in range(a, b):
                e = var_edge[i]
                v2c[e] = s - c2v[e]
        # syndrome
        ok = True
        for c in range(m):
            par = 0
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                par ^= hard[chk_var[e]]
            if par != 0:
                ok = False
                break
        if ok:
            converged = True
            iters_used = it + 1
            break
    for v in range(n):
        out[v] = hard[v]
    return converged


@njit(parallel=True, cache=True)
def decode_batch(Ls, chk_ptr, chk_var, var_ptr, var_edge, max_iter):
    B, n = Ls.shape
    out = np.empty((B, n), dtype=np.uint8)
    conv = np.empty(B, dtype=np.bool_)
    maxd = 0
    for c in range(chk_ptr.shape[0] - 1):
        d = chk_ptr[c + 1] - chk_ptr[c]
        if d > maxd:
            maxd = d
    for bi in prange(B):
        tmp = np.empty(maxd)
        o = np.empty(n, dtype=np.uint8)
        conv[bi] = _decode_one(Ls[bi], chk_ptr, chk_var, var_ptr, var_edge, max_iter, o, tmp)
        out[bi] = o
    return out, conv


def decode(code, Ls, max_iter=60):
    return decode_batch(np.ascontiguousarray(Ls, dtype=np.float64), code['chk_ptr'], code['chk_var'],
                        code['var_ptr'], code['var_edge'], max_iter)


@njit(cache=True)
def _lowweight(C, m, wmin):
    """C: (s,3) sorted check positions of low-degree info nodes (-1 padded).
    Returns bool mask of nodes involved in a single/pair codeword of weight < wmin."""
    s = C.shape[0]
    bad = np.zeros(s, dtype=np.bool_)
    buf = np.empty(6, dtype=np.int64)
    for i in range(s):
        # single
        d = 0
        for a in range(3):
            if C[i, a] >= 0:
                buf[d] = C[i, a]; d += 1
        w = 1
        j = 0
        while j < d:
            if j + 1 < d:
                w += buf[j + 1] - buf[j]
            else:
                w += m - buf[j]
            j += 2
        if w < wmin:
            bad[i] = True
    for i in range(s):
        for u in range(i + 1, s):
            # quick reject: need every position of i near some position of u
            d = 0
            for a in range(3):
                if C[i, a] >= 0:
                    buf[d] = C[i, a]; d += 1
            for a in range(3):
                if C[u, a] >= 0:
                    buf[d] = C[u, a]; d += 1
            # sort
            for x in range(1, d):
                y = buf[x]; z = x - 1
                while z >= 0 and buf[z] > y:
                    buf[z + 1] = buf[z]; z -= 1
                buf[z + 1] = y
            # cancel equal pairs
            tmp = np.empty(6, dtype=np.int64)
            t = 0
            x = 0
            while x < d:
                if x + 1 < d and buf[x] == buf[x + 1]:
                    x += 2
                else:
                    tmp[t] = buf[x]; t += 1; x += 1
            w = 2
            j = 0
            while j < t:
                if j + 1 < t:
                    w += tmp[j + 1] - tmp[j]
                else:
                    w += m - tmp[j]
                j += 2
                if w >= wmin:
                    break
            if w < wmin:
                bad[u] = True
    return bad
