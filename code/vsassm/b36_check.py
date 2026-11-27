"""B36 second angle (prereg/B36_keyfree_equal_bits_prereg.md). Shares no code with b36_common / b36_run / b36_verdict /
b32_common: seeds, projections, signatures, ribbon, sketches, Bloom filters and SDM are re-implemented from the written
specification (prereg, 'Memories'); only the fitted PQ codebooks are read from b36_out/pqcache (as B32's checker read
FAISS's fitted codebooks).

python3 b36_check.py score OUTDIR                 # recompute every cell / region from the dumps; store-level bootstrap
python3 b36_check.py bits  OUTDIR                 # every logged configuration within its budget (all stores, all cells)
python3 b36_check.py rederive OUTDIR ENC [--frac 0.1]   # re-derive the new families' test predictions on a sample
"""
import gzip
import hashlib
import json
import math
import sys

import numpy as np

MS, VS, BPFS = (256, 1024), (2, 16, 128, 1024), (1, 2, 4, 8, 16, 32, 64)
NEWF = ('SF', 'CM', 'BF', 'FBF', 'SDM')
M64 = (1 << 64) - 1


def recs(outdir, enc, M, V):
    return [json.loads(l) for l in gzip.open(f'{outdir}/all_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]


# ------------------------------------------------------------------------------------------------ score

def score(outdir):
    J = json.load(open('results/b36_verdict.json'))
    worst, n = 0.0, 0
    cells = {}
    for enc in ('qwen8', 'e5'):
        for M in MS:
            for V in VS:
                R = recs(outdir, enc, M, V)
                for acct in ('ideal', 'real'):
                    for b in BPFS:
                        cv = cb = tot = 0
                        per_store = []
                        for r in R:
                            g = r['qgold']
                            pv, pb = r['pred'][f'VSA|{b}|{acct}'], r['pred'][f'BEST|{b}|{acct}']
                            a = sum(1 for i in range(len(g)) if pv[i] == g[i])
                            c = sum(1 for i in range(len(g)) if pb[i] == g[i])
                            cv, cb, tot = cv + a, cb + c, tot + len(g)
                            per_store.append((a - c, len(g)))
                        mine = (cv - cb) / tot
                        theirs = J['cells'][f'{enc}|{acct}|{M}|{V}|{b}']['diff']
                        worst = max(worst, abs(mine - theirs))
                        n += 1
                        cells[enc, acct, M, V, b] = (mine, per_store)
    print(f'cells re-scored: {n}; largest |difference| from the verdict: {worst:.2e}')
    R1 = [(M, V, b) for M in MS for V in (2, 16) for b in (4, 8, 16)]
    rng = np.random.default_rng(4242)
    for enc in ('qwen8', 'e5'):
        for acct in ('ideal', 'real'):
            stat = float(np.mean([cells[enc, acct, M, V, b][0] for (M, V, b) in R1]))
            theirs = J['regions'][f'{enc}|{acct}|R1|BEST']['diff']
            draws = np.zeros(4000)
            for (M, V, b) in R1:
                ps = cells[enc, acct, M, V, b][1]
                d = np.array([x for x, _ in ps], float)
                c = np.array([y for _, y in ps], float)
                idx = rng.integers(0, len(ps), size=(4000, len(ps)))
                draws += d[idx].sum(1) / c[idx].sum(1)
            draws /= len(R1)
            print(f'{enc:5s} {acct:5s} R1 VSA - BEST: {100 * stat:+.2f} (verdict {100 * theirs:+.2f}); store-level '
                  f'95% CI [{100 * np.quantile(draws, 0.025):+.2f}, {100 * np.quantile(draws, 0.975):+.2f}]')


# ------------------------------------------------------------------------------------------------ bits

HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


def vsa_bits_ok(B, q, acct):
    per = q if acct == 'real' else HQ[q]
    d = int((B - 32) // per)
    while d > 0 and d * per + 32 * math.ceil(d / 8192) > B:
        d -= 1
    return d * per + 32 * math.ceil(d / 8192) <= B, d


def multiset_bits(b, V, M):
    lN = b + math.log2(V)
    return M * lN + sum(math.log1p(i / 2.0 ** lN) for i in range(M)) / math.log(2) - math.lgamma(M + 1) / math.log(2)


def bits(outdir):
    over, n, worst = [], 0, 0.0
    for enc in ('qwen8', 'e5'):
        for M in MS:
            for V in VS:
                for r in recs(outdir, enc, M, V):
                    Ms = r['M']
                    for key, bused in r['bits'].items():
                        f, b, a = key.split('|')
                        B = int(b) * Ms
                        n += 1
                        worst = max(worst, bused / B)
                        if bused > B + 1e-6:
                            over.append((enc, M, V, r['rep'], key, bused, B))
                    # VSA: dimension from the written rule fits; index: logged key bits fit as a multiset / fixed width
                    for b in BPFS:
                        for a in ('ideal', 'real'):
                            q = r['cfg'][f'VSA|{b}|{a}'][2]
                            ok, _ = vsa_bits_ok(b * Ms, q, a)
                            n += 1
                            if not ok:
                                over.append((enc, M, V, r['rep'], f'VSA|{b}|{a}'))
                            kb = r.get('keybits', {}).get(f'IDX|{b}|{"ideal" if a == "ideal" else "raw"}')
                            if kb is not None:
                                f_ = r['cfg'][f'IDX|{b}|{a}'][2]
                                ms = max(1, int(math.floor(f_ * Ms)))
                                vb = min(V * math.ceil(math.log2(ms + 1)), ms * math.ceil(math.log2(max(V, 2))))
                                need = multiset_bits(kb, V, ms) if a == 'ideal' else kb * ms + vb
                                n += 1
                                if kb > 0 and need > b * Ms + 1e-6:
                                    over.append((enc, M, V, r['rep'], f'IDX|{b}|{a}', need, b * Ms))
    print(f'configurations audited: {n}; over budget: {len(over)}; largest used/budget among new families: {worst:.4f}')
    for o in over[:20]:
        print('  OVER', o)


# ------------------------------------------------------------------------------------------------ re-derivation

def sd15(*parts):
    return int(hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def sd16(*parts):
    return int(hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def sm64(x):
    """splitmix64 finaliser on Python ints (the numpy version in b36_common is the implementation under test)."""
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


class World:
    """Codes, preprocessing (float64 re-implementation) and the projection pool for one encoder."""

    def __init__(self, enc):
        self.enc = enc
        tag = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}[enc]
        S = json.load(open('b32_data/emb_strings.json'))
        self.ix = {s: i for i, s in enumerate(S)}
        self.X = np.load(f'b32_data/emb_{tag}.npy').astype(np.float32)
        E = json.load(open('b32_data/entities.json'))['entities']
        self.E = {e['uri']: e for e in E}
        tr = [e for e in E if e['split'] == 'train']
        T = self.X[[self.ix[e['name']] for e in tr] + [self.ix[a] for e in tr for a in e['aliases']]]
        self.mu = T.mean(0)
        Cv = np.cov((T - self.mu).T)
        w, U = np.linalg.eigh(Cv)
        w = np.maximum(w, 0)
        self.Wz = ((U / np.sqrt(w + 0.1 * w.mean())) @ U.T).astype(np.float32)
        self.n = self.X.shape[1]
        self.G = {}

    def prep(self, kind, Z):
        Y = Z.astype(np.float32) - self.mu
        if kind == 'whiten':
            Y = Y @ self.Wz
        return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)

    def cols(self, Y, j0, j1):
        out = []
        for b0 in range(j0 - j0 % 8192, j1, 8192):
            if b0 not in self.G:
                self.G[b0] = np.random.default_rng(sd15('B36G', self.enc, b0)).standard_normal(
                    (self.n, 8192), dtype=np.float32)
            a, z = max(j0, b0), min(j1, b0 + 8192)
            out.append(Y @ self.G[b0][:, a - b0:z - b0])
        return np.concatenate(out, 1)


def store_from_dump(Wd, r):
    keys = Wd.X[[Wd.ix[Wd.E[u]['name']] for u in r['uris']]]
    qs = Wd.X[[Wd.ix[a] for a in r['qalias']]]
    return keys, np.array(r['vals']), qs


def sm64np(x):
    """splitmix64 finaliser, numpy uint64 (second implementation, used for vectorised reads; sm64 above is the scalar
    one)."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over='ignore'):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class Sig:
    """Signatures of the stored keys ('K') and queries ('Q') re-derived from the projections (cached per table)."""

    _n = 0

    def __init__(self, Wd, prep, PK, PQ):
        self.Wd, self.prep, self.PK, self.PQ = Wd, prep, PK, PQ
        z = np.load(f'b36_out/pqcache/pq_{Wd.enc}_{prep}.npz')
        self.cent = z['c']
        self.cache = {}
        Sig._n += 1
        self.uid = Sig._n
        _SF.clear()
        _CM.clear()

    def get(self, hashf, t, k, T, which):
        key = (hashf, t, k, T, which)
        if key in self.cache:
            return self.cache[key]
        P = self.PK if which == 'K' else self.PQ
        n = P.shape[0]
        out = np.zeros((n, T), np.int64)
        if hashf == 'sim':
            X = P[:, t * k:(t + 1) * k]
            pw = np.array([1 << j for j in range(k)], dtype=np.int64)
            sig = ((X >= 0).astype(np.int64) * pw).sum(1)
            out[:, 0] = sig
            if T > 1:
                o = np.argsort(np.abs(X), axis=1, kind='stable')
                for p in range(min(T - 1, k)):
                    out[:, 1 + p] = sig ^ pw[o[:, p]]
                if T - 1 > k:
                    out[:, 1 + k] = sig ^ pw[o[:, 0]] ^ pw[o[:, 1]]
        else:
            ns = k // 4
            codes, alts, gaps = [], [], []
            for u in range(ns):
                sidx = t * ns + u
                Xs = P[:, 16 * sidx:16 * sidx + 16].astype(np.float64)
                c = self.cent[sidx].astype(np.float64)
                d = ((Xs[:, None, :] - c[None]) ** 2).sum(-1)
                o = np.argsort(d, axis=1, kind='stable')[:, :4]
                codes.append(o[:, 0])
                for j in (1, 2, 3):
                    alts.append((u, o[:, j]))
                    gaps.append(d[np.arange(n), o[:, j]] - d[np.arange(n), o[:, 0]])
            codes = np.stack(codes, 1).astype(np.int64)
            sig = (codes << (4 * np.arange(ns, dtype=np.int64))).sum(1)
            out[:, 0] = sig
            if T > 1:
                G = np.stack(gaps, 1)
                o = np.argsort(G, axis=1, kind='stable')[:, :T - 1]
                for p in range(T - 1):
                    for i in range(n):
                        u, alt = alts[o[i, p]]
                        out[i, 1 + p] = sig[i] ^ ((int(codes[i, u]) ^ int(alt[i])) << (4 * u))
        self.cache[key] = out
        return out


def ribbon_solve(keys, words, seed):
    """Standard ribbon, re-implemented from the specification (scalar Python): rows kept in a dict keyed by pivot."""
    n = len(keys)
    m = max(math.ceil(n * 1.03) + 4, 65)
    att = 0
    while True:
        sd = sd16('ribbon', seed, att)
        rows = {}
        ok = True
        for x, v in zip(keys, words):
            h1 = sm64((int(x) & M64) ^ sd)
            c = sm64(h1 ^ 0x5851F42D4C957F2D) | 1
            s_ = h1 % (m - 63)
            while True:
                if s_ not in rows:
                    rows[s_] = (c, v)
                    break
                c2, v2 = rows[s_]
                c ^= c2
                v ^= v2
                if c == 0:
                    ok = v == 0
                    break
                while c & 1 == 0:
                    c >>= 1
                    s_ += 1
            if not ok:
                break
        if ok:
            break
        att += 1
        if att % 3 == 0:
            m = math.ceil(m * 1.04) + 1
    free = np.random.default_rng(sd).integers(0, 1 << 16, size=m).tolist()
    Z = [0] * (m + 64)
    for i in range(m - 1, -1, -1):
        if i not in rows:
            Z[i] = free[i]
            continue
        c, v = rows[i]
        j = 1
        c >>= 1
        while c:
            if c & 1:
                v ^= Z[i + j]
            c >>= 1
            j += 1
        Z[i] = v
    return m, sd, np.array(Z, dtype=np.int64)


def ribbon_read(m, sd, Z, xs):
    xs = np.asarray(xs, dtype=np.int64).astype(np.uint64)
    h1 = sm64np(xs ^ np.uint64(sd))
    h2 = sm64np(h1 ^ np.uint64(0x5851F42D4C957F2D)) | np.uint64(1)
    s_ = (h1 % np.uint64(m - 63)).astype(np.int64)
    o = np.zeros(len(xs), np.int64)
    for j in range(64):
        b = ((h2 >> np.uint64(j)) & np.uint64(1)).astype(bool)
        o[b] ^= Z[s_[b] + j]
    return o


def fp4(sig, t, tag):
    return (sm64np(np.asarray(sig, dtype=np.int64).astype(np.uint64) ^ np.uint64(sd16('fp', tag, t)))
            >> np.uint64(60)).astype(np.int64)


def plural(sigs, vals, V, seed):
    """Plurality value per distinct signature; ties broken by the registered jitter: U(0, 0.5) per (signature, value)
    from sd15('tie', *seed), rows in increasing signature order."""
    d = {}
    for s_, v in zip(sigs.tolist(), vals):
        d.setdefault(s_, [0] * V)[v] += 1
    keys = sorted(d)
    jit = np.random.default_rng(sd15('tie', *seed)).random((len(keys), V)) * 0.5
    return {s_: max(range(V), key=lambda v: d[s_][v] + jit[i, v]) for i, s_ in enumerate(keys)}


def answer_vec(score, default):
    sc = score.copy()
    sc[:, default] += 1e-6
    return sc.argmax(1).tolist()


def ent(counts):
    c = np.array([x for x in counts if x > 0], float)
    return float(-(c / c.sum() * np.log2(c / c.sum())).sum()) if len(c) else 0.0


def dflt(vals, V):
    return max(range(V), key=lambda v: (int((vals == v).sum()), -v))


_SF = {}


def sf_table(sg, hashf, k, t, M, V, vals, tag):
    key = (sg.uid, hashf, k, t)
    if key not in _SF:
        sk = sg.get(hashf, t, k, 1, 'K')[:, 0]
        pv = plural(sk, vals.tolist(), V, (tag, 'sf', hashf, k, t))
        u = sorted(pv)
        words = [(pv[x] << 4) | int(fp4([x], t, tag)[0]) for x in u]
        m, sd, Z = ribbon_solve(u, words, (tag, 'sf', hashf, k, t))
        _SF[key] = (len(u), m, sd, Z)
    return _SF[key]


def red_sf(sg, cfg, bpf, acct, M, V, vals, order, tag):
    hashf, k, fp, T = cfg
    r = int(round(math.log2(V)))
    default = dflt(vals, V)
    Bav = bpf * M - math.ceil(math.log2(V))
    used, tabs, t = 0.0, [], 0
    while (16 * (t + 1) * (k // 4) <= 65536) if hashf == 'pq' else ((t + 1) * k <= 65536):
        n_u, m, sd, Z = sf_table(sg, hashf, k, t, M, V, vals, tag)
        cost = n_u * (r + fp) if acct == 'ideal' else m * (r + fp) + 32
        if used + cost > Bav:
            break
        used += cost
        tabs.append(t)
        t += 1
    if not tabs:
        return None  # partial table: not re-derived (counted)
    nq = sg.PQ.shape[0]
    score = np.zeros((nq, V))
    for t in tabs:
        n_u, m, sd, Z = sf_table(sg, hashf, k, t, M, V, vals, tag)
        sq = sg.get(hashf, t, k, T, 'Q')
        for p in range(T):
            w = ribbon_read(m, sd, Z, sq[:, p])
            ok = np.ones(nq, bool) if fp == 0 else (((w & 15) >> (4 - fp)) == (fp4(sq[:, p], t, tag) >> (4 - fp)))
            v = (w >> 4) & (V - 1)
            np.add.at(score, (np.nonzero(ok)[0], v[ok]), 1.0 if p == 0 else 0.5)
    return answer_vec(score, default)


_CM = {}


def cm_table(sg, hashf, k, t, ct, V, vals, tag):
    key = (sg.uid, hashf, k, t, ct)
    if key not in _CM:
        nb = 2 ** k
        sk = sg.get(hashf, t, k, 1, 'K')[:, 0]
        cnt = {}
        for x, v in zip(sk.tolist(), vals.tolist()):
            cnt.setdefault(x, [0] * V)[v] += 1
        from collections import Counter
        if ct == 'plural':
            keys_ = sorted(cnt)
            jit = np.random.default_rng(sd15('tie', tag, 'cm', hashf, k, t)).random((len(keys_), V)) * 0.5
            cont = {x: max(range(V), key=lambda v: cnt[x][v] + jit[i, v]) for i, x in enumerate(keys_)}
            syms = [nb - len(cnt)] + [sum(1 for y in cont.values() if y == v) for v in range(V)]
            real = nb * math.ceil(math.log2(V + 1))
        elif ct == 'pres':
            cont = {x: tuple(int(y > 0) for y in c) for x, c in cnt.items()}
            syms = [nb - len(cnt)] + list(Counter(cont.values()).values())
            real = nb * V
        else:
            cont = {x: tuple(c) for x, c in cnt.items()}
            syms = [nb - len(cnt)] + list(Counter(cont.values()).values())
            real = nb * V * max(1, math.ceil(math.log2(max(max(c) for c in cnt.values()) + 1)))
        _CM[key] = (cont, nb * ent(syms), real)
    return _CM[key]


def red_cm(sg, cfg, bpf, acct, M, V, vals, tag):
    hashf, k, ct, T = cfg
    default = dflt(vals, V)
    Bav = bpf * M - math.ceil(math.log2(V))
    used, tabs, t = 0.0, [], 0
    while t < 16384 and ((16 * (t + 1) * (k // 4) <= 65536) if hashf == 'pq' else ((t + 1) * k <= 65536)):
        cont, ideal, real = cm_table(sg, hashf, k, t, ct, V, vals, tag)
        cost = ideal if acct == 'ideal' else real
        if used + cost > Bav:
            break
        used += cost
        tabs.append(t)
        t += 1
    nq = sg.PQ.shape[0]
    score = np.zeros((nq, V))
    for t in tabs:
        cont = cm_table(sg, hashf, k, t, ct, V, vals, tag)[0]
        sq = sg.get(hashf, t, k, T, 'Q')
        for p in range(T):
            w = 1.0 if p == 0 else 0.5
            for i, x in enumerate(sq[:, p].tolist()):
                c = cont.get(x)
                if c is None:
                    continue
                if ct == 'plural':
                    score[i, c] += w
                else:
                    score[i] += w * np.array(c)
    return answer_vec(score, default)


def red_bf(sg, cfg, bpf, M, V, vals, tag):
    hashf, k, L = cfg
    default = dflt(vals, V)
    m = bpf * M - math.ceil(math.log2(V))
    vk = np.array([sm64(((v * 0x9E3779B97F4A7C15) & M64) ^ sd16('bfv', tag)) for v in range(V)], dtype=np.uint64)
    items = set()
    for t in range(L):
        ts = sd16('bft', tag, hashf, k, t)
        for x, v in zip(sg.get(hashf, t, k, 1, 'K')[:, 0].tolist(), vals.tolist()):
            items.add(sm64(sm64(x ^ ts) ^ int(vk[v])))
    h = int(min(16, max(1, round(math.log(2) * m / len(items)))))
    bitset = np.zeros(m, bool)
    for x in items:
        h2 = sm64(x ^ 0xD6E8FEB86659FD93) | 1
        for j in range(h):
            bitset[((x + j * h2) & M64) % m] = True
    nq = sg.PQ.shape[0]
    score = np.zeros((nq, V))
    for t in range(L):
        ts = np.uint64(sd16('bft', tag, hashf, k, t))
        base = sm64np(sg.get(hashf, t, k, 1, 'Q')[:, 0].astype(np.uint64) ^ ts)
        x = sm64np(base[:, None] ^ vk[None, :])
        h2 = sm64np(x ^ np.uint64(0xD6E8FEB86659FD93)) | np.uint64(1)
        allset = np.ones(x.shape, bool)
        with np.errstate(over='ignore'):
            for j in range(h):
                allset &= bitset[((x + np.uint64(j) * h2) % np.uint64(m)).astype(np.int64)]
        score += allset
    return answer_vec(score, default)


def answer(score, V, default):
    best, arg = None, None
    for v in range(V):
        sc = score.get(v, 0.0) + (1e-6 if v == default else 0.0)
        if best is None or sc > best:
            best, arg = sc, v
    return arg


def h2b(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def red_fbf(PK, PQ, cfg, bpf, acct, M, V, vals):
    (rho,) = cfg
    default = max(range(V), key=lambda v: (int((vals == v).sum()), -v))
    Bav = bpf * M - math.ceil(math.log2(V))
    Mv = [int((vals == v).sum()) for v in range(V)]

    def build(m):
        F = np.zeros((V, m), bool)
        for i in range(M):
            top = np.argsort(-PK[i, :m], kind='stable')[:rho]
            F[vals[i], top] = True
        return F
    if acct == 'real':
        m = min(65536, Bav // V)
        F = build(m) if m > rho else None
    else:
        def est(m):
            return sum(m * h2b(1 - (1 - rho / m) ** c) for c in Mv)
        lo, hi = rho + 1, 65536
        if est(lo) > Bav:
            m = 0
        else:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if est(mid) <= Bav else (lo, mid - 1)
            m = lo
        F = None
        while m > rho:
            F = build(m)
            if sum(m * h2b(F[v].mean()) for v in range(V)) <= Bav:
                break
            m = int(m * 0.98)
        if m <= rho:
            F = None
    if F is None:
        return [default] * PQ.shape[0]
    m = F.shape[1]
    out = []
    for i in range(PQ.shape[0]):
        top = np.argsort(-PQ[i, :m], kind='stable')[:rho]
        sc = {v: float(F[v, top].sum()) for v in range(V)}
        out.append(answer(sc, V, default))
    return out


_SDM = {}


def red_sdm(Wd, Pfull, cfg, bpf, acct, M, V, vals, ckey):
    """SDM re-derived: counters u < Dcap = min(65,536, 128 M); location of counter u = u // nd = pool column; active if
    projection >= Phi^-1(1 - p); codes = first V rows of the 1024 x 8192 sign block per 8,192 counters."""
    from scipy.stats import norm
    p, nd, q = cfg
    default = dflt(vals, V)
    Bav = bpf * M - math.ceil(math.log2(V))
    Dcap = min(65536, 2 * 64 * M)
    key = (ckey, p, nd)
    if key not in _SDM:
        tau = float(norm.ppf(1 - p))
        nloc = (Dcap + nd - 1) // nd
        A = Pfull[:, :nloc] >= tau
        codes = np.concatenate([np.where(np.random.default_rng(sd15('B36SDMC', Wd.enc, u0)).random(
            (1024, 8192), dtype=np.float32) < 0.5, -1, 1).astype(np.int8)[:V] for u0 in range(0, Dcap, 8192)],
            1)[:, :Dcap]
        loc = np.arange(Dcap) // nd
        S = np.zeros(Dcap, np.int64)
        for i in range(M):  # stored facts one by one
            S += A[i, loc] * codes[vals[i]].astype(np.int64)
        _SDM.clear()
        _SDM[key] = (A, codes, loc, S)
    A, codes, loc, S = _SDM[key]
    sym = np.sign(S) if q == 'tern' else S

    def nbits(D):
        if D <= 0:
            return 0.0
        s_ = sym[:D]
        if acct == 'ideal':
            return D * ent(np.unique(s_, return_counts=True)[1])
        if q == 'tern':
            return 2.0 * D
        mx = int(np.abs(s_).max())
        return D * (math.ceil(math.log2(2 * mx + 1 + 1e-9)) if mx > 0 else 1) + 8
    lo, hi = 0, min(Dcap, 2 * bpf * M)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if nbits(mid) <= Bav else (lo, mid - 1)
    D = lo
    while D > 0 and nbits(D) > Bav:
        D = int(D * 0.99)
    nq = A.shape[0] - M
    if D == 0:
        return [default] * nq
    Aq = A[M:][:, loc[:D]].astype(np.float64)                    # (Q, D) active counters
    sc = (Aq * sym[:D].astype(np.float64)) @ codes[:, :D].T.astype(np.float64)
    return answer_vec(sc, default)


def rederive(outdir, enc, frac):
    Wd = World(enc)
    rng = np.random.default_rng(36)
    agree = {}
    skipped = []
    for M in MS:
        for V in VS:
            R = recs(outdir, enc, M, V)
            k = max(1, int(round(frac * len(R))))
            pick = sorted(rng.choice(len(R), size=k, replace=False).tolist())
            for ri in pick:
                r = R[ri]
                keys, vals, qs = store_from_dump(Wd, r)
                order = np.random.default_rng(sd15('subset', 'test', M, V, r['rep'])).permutation(M)
                gold = r['qgold']
                for prep in ('center', 'whiten'):
                    Y = Wd.prep(prep, np.vstack([keys, qs]))
                    P = None
                    ckey = (M, V, r['rep'], prep)
                    for key in r['pred']:
                        fam, b, a = (key.split('|') + ['', ''])[:3]
                        if fam not in NEWF or r['cfg'][key][0] != prep:
                            continue
                        b = int(b)
                        if a == 'real' and b not in (4, 16):
                            continue
                        cfg = r['cfg'][key][1:]
                        tag = f'{enc}|test|{M}|{V}|{r["rep"]}|{prep}'
                        if P is None:
                            P = Wd.cols(Y, 0, 65536)
                            sg = Sig(Wd, prep, P[:M], P[M:])
                        if fam == 'SF':
                            red = red_sf(sg, tuple(cfg), b, a, M, V, vals, order, tag)
                        elif fam == 'CM':
                            red = red_cm(sg, tuple(cfg), b, a, M, V, vals, tag)
                        elif fam == 'BF':
                            red = red_bf(sg, tuple(cfg), b, M, V, vals, tag)
                        elif fam == 'FBF':
                            red = red_fbf(P[:M], P[M:], tuple(cfg), b, a, M, V, vals)
                        else:
                            red = red_sdm(Wd, P, tuple(cfg), b, a, M, V, vals, ckey)
                        if red is None:
                            skipped.append((M, V, r['rep'], key))
                            continue
                        logged = r['pred'][key]
                        same = sum(1 for x, y in zip(red, logged) if x == y)
                        acc_red = sum(1 for x, g in zip(red, gold) if x == g)
                        acc_log = sum(1 for x, g in zip(logged, gold) if x == g)
                        e = agree.setdefault(fam, [0, 0, 0, 0])
                        e[0] += same
                        e[1] += len(logged)
                        e[2] += acc_red
                        e[3] += acc_log
                print(f'rederive {enc} M={M} V={V} rep={r["rep"]} done', flush=True)
    out = {f: {'agreement': e[0] / e[1], 'queries': e[1], 'acc_rederived': e[2] / e[1], 'acc_logged': e[3] / e[1]}
           for f, e in agree.items()}
    out['skipped_partial_tables'] = len(skipped)
    json.dump(out, open(f'results/b36_rederive_{enc}.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'score':
        score(a[1])
    elif a[0] == 'bits':
        bits(a[1])
    elif a[0] == 'rederive':
        rederive(a[1], a[2], float(a[a.index('--frac') + 1]) if '--frac' in a else 0.1)
