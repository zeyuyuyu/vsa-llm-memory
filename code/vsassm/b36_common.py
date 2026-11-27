"""B36 (prereg/B36_keyfree_equal_bits_prereg.md): the VSA store against classical memories that keep a key -> value map
WITHOUT storing keys, at equal bits, on B32's stores (PopQA entities, random values, unseen Wikidata aliases as
queries, keys = frozen-encoder codes of the label).

Families implemented here (B32's VSA and per-fact index are reused from b32_common / b32_run):
  SF   static-function retrieval (standard ribbon, Dillinger & Walzer 2021) keyed by an LSH (SimHash) or PQ signature
       of the key code, in L independent tables, each storing (value, f-bit fingerprint of the signature); a query
       probes its own signature (and T-1 multi-probe neighbours) in every table and the accepted outputs vote.
  CM   count-min-style LSH bucket sketch: per table 2^k buckets, each holding its stored facts' plurality value (or
       EMPTY), or, for V <= 16, the set of values present (one Bloom bit per value); entropy-coded; probes vote.
  BF   one Bloom filter over the items (table, signature, value); a query tests every value in every table.
  FBF  fly Bloom filter per value (Dasgupta et al. 2018; FlyNN, Sinha & Ram 2021): winner-take-all fly hash of a random
       expansion, one bit array per value; read = how many of the query's winners are set in each value's array.
  SDM  sparse distributed memory (Kanerva 1988) with continuous addresses: random Gaussian address vectors, location
       active if the projection exceeds the (1-p) Gaussian quantile; counters per location bundle random +-1 value codes;
       ternary or exact (entropy-coded) counters; read = activated counters correlated with each value's code.

Budget per store: B = bpf * M bits. Free for every memory (as in B32): encoder, value vocabulary, preprocessing fitted on
the training split, every fixed random component drawn from a seed (projections, hash seeds, value codes, PQ codebooks
fitted on training codes). Counted: everything that depends on the store's facts, plus ceil(log2 V) bits for a default
value (the store's plurality value: answered when no table votes).

Accounting ('ideal', primary; 'real', secondary):
  SF   ideal: n_t (r + f) bits per table (n_t distinct stored signatures, r = log2 V): the retrieval lower bound
       (BuRR reaches it within 1%). real: the ribbon actually built, m_t (r + f) + 32 bits per table.
  CM   ideal: 2^k * H0 per table (zeroth-order empirical entropy of the bucket symbols). real: 2^k * ceil(log2(V+1))
       (plurality) or 2^k * V (value sets) per table.
  BF   m bits (a Bloom filter at its optimal number of hash functions is incompressible) in both.
  FBF  ideal: V * m * H2(fill); real: V * m.
  SDM  ideal: D * H0 (empirical entropy of the stored counter symbols); real: fixed width (2 bits per ternary counter;
       ceil(log2(2 max|S| + 1)) per exact counter) + 8.
"""
import hashlib
import math

import numpy as np

import b32_common as C

seed_of = C.seed_of
CH = 8192          # projection pool block width
POOL = 65536       # projection pool for signatures / fly cells (columns)
SDM_MAX = 65536    # SDM counters cap (and <= 2 B)
W = 64             # ribbon width
FPMAX = 4          # fingerprint bits stored (lower f used by a configuration: the top f of these 4)
FLIP_W = 0.5       # vote weight of a multi-probe neighbour (exact signature: 1)
U64 = np.uint64
MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------------------------------- hashing

def mix64(x):
    """splitmix64 finaliser, vectorised on uint64 arrays."""
    with np.errstate(over='ignore'):
        z = (np.asarray(x, dtype=U64) + U64(0x9E3779B97F4A7C15))
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def s64(*parts):
    return U64(int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16))


# ----------------------------------------------------------------------------------------------- projection pool

class Pool:
    """Fixed Gaussian projection pool per encoder: column block j0 = N(0,1) n x CH from seed ('B36G', enc, j0).
    Every family reads its columns from this pool (SimHash hyperplanes, PQ sub-spaces, fly cells, SDM addresses).
    Codes are unit-norm after preprocessing, so a projection is N(0, 1) for a random direction."""

    def __init__(self, enc, n):
        self.enc, self.n = enc, n
        self.cache = {}

    def block(self, j0):
        if j0 not in self.cache:
            self.cache[j0] = np.random.default_rng(seed_of('B36G', self.enc, j0)).standard_normal(
                (self.n, CH), dtype=np.float32)
        return self.cache[j0]

    def project(self, Y, D):
        out = np.empty((Y.shape[0], D), np.float32)
        for j0 in range(0, D, CH):
            w = min(CH, D - j0)
            out[:, j0:j0 + w] = Y @ self.block(j0)[:, :w]
        return out


class PQBank:
    """PQ sub-quantizers: sub-quantizer s = 16 centroids in the 16-dimensional sub-space of pool columns
    [16 s, 16 s + 16), fitted on the training codes by k-means (init: 16 distinct training points drawn from seed
    ('B36PQ', tag, s); 20 Lloyd iterations; an empty cluster keeps its centroid). A k-bit PQ signature is k/4
    consecutive sub-quantizers' codes. All 4,096 sub-quantizers of the pool are fitted once (b36_run.py pqfit) and read
    from the cache; workers never write it."""

    NSUB = POOL // 16

    def __init__(self, pool, train_y, tag, cache_dir=None):
        self.pool, self.tag, self.cache_dir = pool, tag, cache_dir
        self.train = train_y
        self.cent = None
        fn = self._fn()
        if fn:
            import os
            if os.path.exists(fn):
                self.cent = np.load(fn)['c']

    def _fn(self):
        return f'{self.cache_dir}/pq_{self.tag}.npz' if self.cache_dir else None

    def fit(self, chunk=256):
        TP = self.pool.project(self.train, POOL).astype(np.float32)
        N = TP.shape[0]
        out = np.zeros((self.NSUB, 16, 16), np.float32)
        for s0 in range(0, self.NSUB, chunk):
            S = min(chunk, self.NSUB - s0)
            X = TP[:, 16 * s0:16 * (s0 + S)].reshape(N, S, 16).transpose(1, 0, 2).copy()  # (S, N, 16)
            Cc = np.stack([X[j, np.random.default_rng(seed_of('B36PQ', self.tag, s0 + j)).choice(N, 16, replace=False)]
                           for j in range(S)])
            x2 = (X * X).sum(2)
            for _ in range(20):
                d = x2[:, :, None] - 2 * np.matmul(X, Cc.transpose(0, 2, 1)) + (Cc * Cc).sum(2)[:, None, :]
                a = d.argmin(2)
                oh = np.zeros((S, N, 16), np.float32)
                np.put_along_axis(oh, a[:, :, None], 1.0, axis=2)
                cnt = oh.sum(1)
                sums = np.matmul(oh.transpose(0, 2, 1), X)
                nz = cnt > 0
                Cc[nz] = sums[nz] / cnt[nz][:, None]
            out[s0:s0 + S] = Cc
        self.cent = out
        fn = self._fn()
        if fn:
            import os
            np.savez(fn + '.tmp.npz', c=out)
            os.replace(fn + '.tmp.npz', fn)

    def ensure(self, S):
        if self.cent is None:
            self.fit()


# ----------------------------------------------------------------------------------------------- signatures

def sim_sig(P, t, k, T):
    """SimHash table t, k bits: pool columns [t k, t k + k). Returns (sig (Q, T) int64, probe weights (T,)).
    Probe 0 = the exact signature; probe i >= 1 flips the i-th least confident bit (smallest |projection|); with
    k < T - 1 the last probe flips the two least confident bits."""
    X = P[:, t * k:(t + 1) * k]
    bits = (X >= 0).astype(np.int64)
    sig = (bits << np.arange(k, dtype=np.int64)).sum(1)
    out = np.empty((P.shape[0], T), np.int64)
    out[:, 0] = sig
    if T > 1:
        order = np.argsort(np.abs(X), axis=1, kind='stable').astype(np.int64)
        ns = min(T - 1, k)
        out[:, 1:1 + ns] = sig[:, None] ^ (np.int64(1) << order[:, :ns])
        if T - 1 > k:  # k < T - 1 (k = 2): the double flip of the two least confident bits
            out[:, 1 + k] = sig ^ (np.int64(1) << order[:, 0]) ^ (np.int64(1) << order[:, 1])
    return out


def pq_sig(P, bank, t, k, T):
    """PQ table t, k bits: sub-quantizers [t k/4, (t+1) k/4). Probe i >= 1 is the i-th cheapest single replacement
    of one sub-quantizer's code by its second-, third- or fourth-nearest centroid (cost = squared-distance gap)."""
    ns = k // 4
    subs = range(t * ns, (t + 1) * ns)
    bank.ensure((t + 1) * ns)
    codes, alt, gap = [], [], []
    for s in subs:
        X = P[:, 16 * s:16 * s + 16]
        c = bank.cent[s]
        d = (X * X).sum(1)[:, None] - 2 * X @ c.T + (c * c).sum(1)[None]
        o = np.argsort(d, axis=1)[:, :4]
        codes.append(o[:, 0])
        alt += [o[:, 1], o[:, 2], o[:, 3]]
        d0 = np.take_along_axis(d, o[:, :1], 1)[:, 0]
        gap += [np.take_along_axis(d, o[:, j:j + 1], 1)[:, 0] - d0 for j in (1, 2, 3)]
    codes = np.stack(codes, 1).astype(np.int64)
    alt, gap = np.stack(alt, 1).astype(np.int64), np.stack(gap, 1)  # candidate 3u+j: sub u -> its (j+2)-th centroid
    sh = 4 * np.arange(ns, dtype=np.int64)
    sig = (codes << sh).sum(1)
    out = np.empty((P.shape[0], T), np.int64)
    out[:, 0] = sig
    if T > 1:
        order = np.argsort(gap, axis=1, kind='stable')[:, :T - 1]
        r = np.arange(P.shape[0])
        for i in range(T - 1):
            cand = order[:, i]
            u = cand // 3
            out[:, i + 1] = sig ^ ((codes[r, u] ^ alt[r, cand]) << (4 * u))
    return out


def signatures(P, bank, hashf, t, k, T):
    return sim_sig(P, t, k, T) if hashf == 'sim' else pq_sig(P, bank, t, k, T)


def probe_w(T):
    return np.array([1.0] + [FLIP_W] * (T - 1))


def cols_needed(hashf, L, k):
    return L * k if hashf == 'sim' else 16 * L * (k // 4)


# ----------------------------------------------------------------------------------------------- votes

def tie_argmax(cnt, seed):
    """Row-wise argmax of integer counts with ties broken uniformly at random: jitter U(0, 0.5) per entry from
    seed_of('tie', *seed), rows in the order given (sorted distinct signatures)."""
    j = np.random.default_rng(seed_of('tie', *seed)).random(cnt.shape) * 0.5
    return (cnt + j).argmax(1)


def plurality(sig, vals, V, seed):
    """Distinct signatures (sorted) and the plurality value of the facts sharing each (ties: random, seeded)."""
    u, inv = np.unique(sig, return_inverse=True)
    cnt = np.zeros((len(u), V), np.int32)
    np.add.at(cnt, (inv, vals), 1)
    return u, tie_argmax(cnt, seed)


def decide(score, default):
    """argmax; ties go to the default value (the store's plurality value), then to the smallest value."""
    s = score.copy()
    s[:, default] += 1e-6
    return s.argmax(1)


def default_value(vals, V):
    return int(np.bincount(vals, minlength=V).argmax())


def def_bits(V):
    return math.ceil(math.log2(V))


def H0(counts):
    c = np.asarray(counts, np.float64)
    c = c[c > 0]
    n = c.sum()
    return float(-(c / n * np.log2(c / n)).sum()) if n > 0 else 0.0


def H2(p):
    return 0.0 if p <= 0 or p >= 1 else float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


# ----------------------------------------------------------------------------------------------- ribbon retrieval

class Ribbon:
    """Standard ribbon retrieval (Dillinger & Walzer 2021), width 64, bit-sliced words of up to 16 bits.
    Key -> (start s in [0, m - 64], coefficient c: 64 bits with bit 0 set) from two splitmix64 hashes of the key and
    the table seed. Build: on-the-fly banded Gaussian elimination over GF(2); a dependent row with a different value
    is a failure (retry with the next seed, then 4% more slots). Back substitution; slots without a pivot get
    seeded random words (so a key outside the set reads an arbitrary word). Query: XOR of the solution words selected
    by c from s."""

    def __init__(self, keys, words, seed, eps=0.03, nb=16):
        n = len(keys)
        self.n = n
        m = max(int(math.ceil(n * (1 + eps))) + 4, W + 1) if n > 0 else 0
        attempt = 0
        while True:
            self.seed = s64('ribbon', seed, attempt)
            ok = self._build(keys, words, m, nb) if n > 0 else self._empty()
            if ok:
                break
            attempt += 1
            if attempt % 3 == 0:
                m = int(math.ceil(m * 1.04)) + 1
        self.attempts = attempt + 1

    def _empty(self):
        self.m = 0
        self.Z = np.zeros(W + 1, np.uint16)
        return True

    def _hash(self, keys):
        h1 = mix64(np.asarray(keys, dtype=np.int64).astype(U64) ^ self.seed)
        h2 = mix64(h1 ^ U64(0x5851F42D4C957F2D)) | U64(1)
        s = (h1 % U64(self.m - W + 1)).astype(np.int64)
        return s, h2

    def _build(self, keys, words, m, nb):
        self.m = m
        s_arr, c_arr = self._hash(keys)
        coeff = [0] * m
        res = [0] * m
        for s, c, v in zip(s_arr.tolist(), c_arr.tolist(), list(map(int, words))):
            while True:
                cc = coeff[s]
                if cc == 0:
                    coeff[s] = c
                    res[s] = v
                    break
                c ^= cc
                v ^= res[s]
                if c == 0:
                    if v != 0:
                        return False
                    break
                tz = (c & -c).bit_length() - 1
                s += tz
                c >>= tz
        rng = np.random.default_rng(int(self.seed) & MASK64)
        free = rng.integers(0, 1 << nb, size=m).tolist()
        Z = [0] * (m + W)
        for i in range(m - 1, -1, -1):
            c = coeff[i]
            if c == 0:
                Z[i] = free[i]
                continue
            v = res[i]
            c >>= 1
            j = 1
            while c:
                if c & 1:
                    v ^= Z[i + j]
                c >>= 1
                j += 1
            Z[i] = v
        self.Z = np.array(Z, dtype=np.uint16)
        return True

    def query(self, keys):
        if self.m == 0:
            return np.zeros(len(keys), np.uint16)
        s, c = self._hash(keys)
        out = np.zeros(len(s), np.uint16)
        for j in range(W):
            sel = ((c >> U64(j)) & U64(1)).astype(bool)
            out[sel] ^= self.Z[s[sel] + j]
        return out


def fph(sig, t, tag):
    """FPMAX-bit fingerprint of a signature in table t (top bits of a splitmix64 hash)."""
    return (mix64(np.asarray(sig, dtype=np.int64).astype(U64) ^ s64('fp', tag, t)) >> U64(64 - FPMAX)).astype(np.int64)


# ----------------------------------------------------------------------------------------------- family runners
# Each runner gets the store, the projections P (rows: the M stored keys, then the queries) of the preprocessed codes,
# the configurations to run, and the budgets; it returns {(cfg, bpf, acct): (pred, bits_used)}.

def budgets(M, bpfs):
    return {b: b * M for b in bpfs}


def _snap_points(costs, Bav):
    """Number of whole tables whose cumulative cost fits Bav."""
    cum = np.cumsum(costs) if len(costs) else np.zeros(0)
    return int(np.searchsorted(cum, Bav + 1e-9, side='right'))


def run_sf(st, P, bank, cfgs, bpfs, accts, tag, order):
    """cfgs: list of (hashf, k, fp, T). Tables are built per (hashf, k) and shared by every fp, T, budget."""
    M, V = st['M'], st['V']
    r = int(round(math.log2(V)))
    vals = st['vals']
    default = default_value(vals, V)
    db = def_bits(V)
    PK, PQ = P[:M], P[M:]
    Qn = PQ.shape[0]
    out = {}
    Bmax = max(bpfs) * M
    by_hk = {}
    for c in cfgs:
        by_hk.setdefault(c[:2], []).append(c)
    for (hashf, k), lst in by_hk.items():
        Tmax = max(c[3] for c in lst)
        fmin = min(c[2] for c in lst)
        tables = []   # per table: n_t, m_t, out words (Qn, Tmax), query fingerprints (Qn, Tmax)
        cum_min = 0
        t = 0
        while True:
            if cols_needed(hashf, t + 1, k) > P.shape[1]:
                break
            sk = signatures(PK, bank, hashf, t, k, 1)[:, 0]
            u, pv = plurality(sk, vals, V, (tag, 'sf', hashf, k, t))
            cost_min = len(u) * (r + fmin)
            if cum_min + cost_min > Bmax - db:
                break
            cum_min += cost_min
            words = (pv.astype(np.int64) << FPMAX) | fph(u, t, tag)
            rb = Ribbon(u, words, (tag, 'sf', hashf, k, t))
            sq = signatures(PQ, bank, hashf, t, k, Tmax)
            tables.append({'n': len(u), 'm': rb.m, 'w': rb.query(sq.ravel()).reshape(Qn, Tmax).astype(np.int64),
                           'fq': fph(sq.ravel(), t, tag).reshape(Qn, Tmax)})
            t += 1
        for (_, _, fp, T) in lst:
            wts = probe_w(T)
            Ls = {}
            for bpf in bpfs:
                for acct in accts:
                    if acct == 'ideal':
                        costs = [tb['n'] * (r + fp) for tb in tables]
                    else:
                        costs = [tb['m'] * (r + fp) + 32 for tb in tables]
                    Ls[bpf, acct] = (_snap_points(costs, bpf * M - db), float(np.sum(costs[:_snap_points(costs, bpf * M - db)])))
            need = sorted({L for L, _ in Ls.values()})
            score = np.zeros((Qn, V), np.float64)
            snaps = {0: None}
            for ti in range(max(need) if need else 0):
                tb = tables[ti]
                w = tb['w'][:, :T]
                val = (w >> FPMAX) & (V - 1)
                if fp > 0:
                    acc = ((w & (2 ** FPMAX - 1)) >> (FPMAX - fp)) == (tb['fq'][:, :T] >> (FPMAX - fp))
                else:
                    acc = np.ones_like(val, bool)
                for p in range(T):
                    np.add.at(score, (np.arange(Qn)[acc[:, p]], val[acc[:, p], p]), wts[p])
                if ti + 1 in need:
                    snaps[ti + 1] = decide(score, default)
            for (bpf, acct), (L, used) in Ls.items():
                if L > 0:
                    out[(hashf, k, fp, T), bpf, acct] = (snaps[L], used + db)
                else:  # not even one whole table fits: one partial table over a prefix of the store's order
                    pred, used = _sf_partial(st, PK, PQ, bank, hashf, k, fp, T, bpf * M - db, acct, tag, order, r,
                                             default)
                    out[(hashf, k, fp, T), bpf, acct] = (pred, used + db)
    return out


def _sf_partial(st, PK, PQ, bank, hashf, k, fp, T, Bav, acct, tag, order, r, default):
    M, V = st['M'], st['V']
    Qn = PQ.shape[0]
    if Bav < (r + fp) + (32 if acct == 'real' else 0):
        return np.full(Qn, default), 0
    sk_all = signatures(PK, bank, hashf, 0, k, 1)[:, 0]
    sk_o = sk_all[order]
    # distinct signatures among the first j facts in the store's order
    first = np.zeros(len(sk_o), bool)
    _, idx = np.unique(sk_o, return_index=True)
    first[idx] = True
    ndist = np.cumsum(first)
    if acct == 'ideal':
        j = int(np.searchsorted(ndist * (r + fp), Bav + 1e-9, side='right'))
    else:
        j = int(np.searchsorted((np.maximum(np.ceil(ndist * 1.03) + 4, W + 1)) * (r + fp) + 32, Bav + 1e-9,
                                side='right'))
    while j > 0:
        keep = order[:j]
        u, pv = plurality(sk_all[keep], st['vals'][keep], V, (tag, 'sfp', hashf, k, j))
        words = (pv.astype(np.int64) << FPMAX) | fph(u, 0, tag)
        rb = Ribbon(u, words, (tag, 'sfp', hashf, k, j))
        used = len(u) * (r + fp) if acct == 'ideal' else rb.m * (r + fp) + 32
        if used <= Bav:
            break
        j = int(j * 0.97)
    if j == 0:
        return np.full(Qn, default), 0
    sq = signatures(PQ, bank, hashf, 0, k, T)
    w = rb.query(sq.ravel()).reshape(Qn, T).astype(np.int64)
    fq = fph(sq.ravel(), 0, tag).reshape(Qn, T)
    val = (w >> FPMAX) & (V - 1)
    acc = (((w & (2 ** FPMAX - 1)) >> (FPMAX - fp)) == (fq >> (FPMAX - fp))) if fp > 0 else np.ones_like(val, bool)
    score = np.zeros((Qn, V))
    wts = probe_w(T)
    for p in range(T):
        np.add.at(score, (np.arange(Qn)[acc[:, p]], val[acc[:, p], p]), wts[p])
    return decide(score, default), used


_ROWMUL = mix64(np.arange(1, 1025, dtype=np.uint64)) | U64(1)


def run_cm(st, P, bank, cfgs, bpfs, accts, tag):
    """cfgs: list of (hashf, k, content, T); content 'plural' (plurality value or EMPTY) or 'pres' (set of values
    present; V <= 16)."""
    M, V = st['M'], st['V']
    vals = st['vals']
    default = default_value(vals, V)
    db = def_bits(V)
    PK, PQ = P[:M], P[M:]
    Qn = PQ.shape[0]
    out = {}
    Bmax = max(bpfs) * M
    by_hk = {}
    for c in cfgs:
        by_hk.setdefault(c[:2], []).append(c)
    rqi = np.arange(Qn)
    for (hashf, k), lst in by_hk.items():
        nb = 2 ** k
        contents = sorted({c[2] for c in lst})
        Tmax = max(c[3] for c in lst)
        tabs = []
        cum = {ct: 0.0 for ct in contents}
        t = 0
        while t < 16384 and cols_needed(hashf, t + 1, k) <= P.shape[1]:
            sk = signatures(PK, bank, hashf, t, k, 1)[:, 0]
            u, inv = np.unique(sk, return_inverse=True)
            cnt = np.zeros((len(u), V), np.int32)
            np.add.at(cnt, (inv, vals), 1)
            tb = {'u': u}
            ok = False
            for ct in contents:
                if ct == 'plural':
                    pv = tie_argmax(cnt, (tag, 'cm', hashf, k, t))
                    tb['plural'] = pv
                    ideal = nb * H0(np.concatenate([[nb - len(u)], np.bincount(pv, minlength=V)]))
                    real = nb * math.ceil(math.log2(V + 1))
                elif ct == 'pres':
                    masks = ((cnt > 0).astype(np.int64) << np.arange(V, dtype=np.int64)).sum(1)
                    tb['pres'] = masks
                    _, mc = np.unique(masks, return_counts=True)
                    ideal = nb * H0(np.concatenate([[nb - len(u)], mc]))
                    real = nb * V
                else:  # 'count': the vector of per-value counts (count sketch / RACE-style counters)
                    tb['count'] = cnt
                    with np.errstate(over='ignore'):  # distinct count vectors via a 64-bit row hash
                        rh = (cnt.astype(np.uint64) * _ROWMUL[:V]).sum(1)
                    _, mc = np.unique(rh, return_counts=True)
                    ideal = nb * H0(np.concatenate([[nb - len(u)], mc]))
                    real = nb * V * max(1, math.ceil(math.log2(int(cnt.max()) + 1)))
                tb['cost', ct, 'ideal'] = ideal
                tb['cost', ct, 'real'] = real
                if cum[ct] + ideal <= Bmax - db:
                    ok = True
                cum[ct] += ideal
            if not ok:
                break
            tabs.append(tb)
            t += 1
        # one pass over the tables updates every (content, T) configuration of this (hashf, k)
        Ls, need, score, snaps = {}, {}, {}, {}
        for c in lst:
            ct = c[2]
            for bpf in bpfs:
                for acct in accts:
                    costs = [tb['cost', ct, acct] for tb in tabs]
                    L = _snap_points(costs, bpf * M - db)
                    Ls[c, bpf, acct] = (L, float(np.sum(costs[:L])))
            need[c] = {L for (cc, _, _), (L, _) in Ls.items() if cc == c}
            score[c] = np.zeros((Qn, V), np.float64)
            snaps[c] = {0: np.full(Qn, default)}
        Lmax = max(max(v) for v in need.values()) if need else 0
        for ti in range(Lmax):
            tb = tabs[ti]
            sq = signatures(PQ, bank, hashf, ti, k, Tmax)
            u = tb['u']
            pos = np.minimum(np.searchsorted(u, sq), len(u) - 1)
            hit = u[pos] == sq
            for c in lst:
                if ti >= max(need[c]):
                    continue
                ct, T = c[2], c[3]
                wts = probe_w(T)
                sc = score[c]
                for p in range(T):
                    h = hit[:, p]
                    if ct == 'plural':
                        np.add.at(sc, (rqi[h], tb['plural'][pos[h, p]]), wts[p])
                    elif ct == 'count':
                        sc[h] += wts[p] * tb['count'][pos[h, p]]
                    else:
                        mk = tb['pres'][pos[h, p]]
                        sc[h] += wts[p] * ((mk[:, None] >> np.arange(V)) & 1)
                if ti + 1 in need[c]:
                    snaps[c][ti + 1] = decide(sc, default)
        for (c, bpf, acct), (L, used) in Ls.items():
            out[c, bpf, acct] = (snaps[c][L], used + db if L else 0)
    return out


def run_bf(st, P, bank, cfgs, bpfs, accts, tag):
    """cfgs: list of (hashf, k, L). One Bloom filter of m = B - default bits over the distinct items
    (table, signature, value) of the stored facts; h = round(ln 2 * m / items) hash functions (1..16) by double
    hashing; a query scores each value by the number of tables in which (table, its signature, value) tests present."""
    M, V = st['M'], st['V']
    vals = st['vals']
    default = default_value(vals, V)
    db = def_bits(V)
    PK, PQ = P[:M], P[M:]
    Qn = PQ.shape[0]
    out = {}
    vkey = mix64(np.arange(V, dtype=np.int64).astype(U64) * U64(0x9E3779B97F4A7C15) ^ s64('bfv', tag))
    by_hk = {}
    for c in cfgs:
        by_hk.setdefault(c[:2], []).append(c)
    for (hashf, k), lst in by_hk.items():
        Lmax = max(c[2] for c in lst)
        if cols_needed(hashf, Lmax, k) > P.shape[1]:
            Lmax = max(L for L in [c[2] for c in lst] if cols_needed(hashf, L, k) <= P.shape[1])
        tsk, tbase = [], []
        for t in range(Lmax):
            sk = signatures(PK, bank, hashf, t, k, 1)[:, 0]
            sq = signatures(PQ, bank, hashf, t, k, 1)[:, 0]
            st_seed = s64('bft', tag, hashf, k, t)
            tsk.append(mix64(sk.astype(U64) ^ st_seed))
            tbase.append(mix64(sq.astype(U64) ^ st_seed))
        for (_, _, L) in lst:
            if L > Lmax:
                continue
            ikeys = np.unique(np.concatenate([mix64(tsk[t] ^ vkey[vals]) for t in range(L)]))
            n_items = len(ikeys)
            for bpf in bpfs:
                m = bpf * M - db
                h = int(min(16, max(1, round(math.log(2) * m / n_items))))
                h1 = ikeys
                with np.errstate(over='ignore'):
                    h2 = mix64(h1 ^ U64(0xD6E8FEB86659FD93)) | U64(1)
                    bits = np.zeros(m, bool)
                    for j in range(h):
                        bits[((h1 + U64(j) * h2) % U64(m)).astype(np.int64)] = True
                    score = np.zeros((Qn, V), np.float64)
                    for t in range(L):
                        q1 = mix64(tbase[t][:, None] ^ vkey[None, :])
                        q2 = mix64(q1 ^ U64(0xD6E8FEB86659FD93)) | U64(1)
                        allset = np.ones((Qn, V), bool)
                        for j in range(h):
                            allset &= bits[((q1 + U64(j) * q2) % U64(m)).astype(np.int64)]
                        score += allset
                pred = decide(score, default)
                for acct in accts:
                    out[(hashf, k, L), bpf, acct] = (pred, m + db)
    return out


def run_fbf(st, P, cfgs, bpfs, accts, tag):
    """cfgs: list of (rho,). Fly hash: the rho largest of the first m pool projections (winner-take-all); one bit array
    of m bits per value: a key sets its winners in its value's array. m: the largest that fits the budget (ideal:
    V m H2(fill); real: V m)."""
    M, V = st['M'], st['V']
    vals = st['vals']
    default = default_value(vals, V)
    db = def_bits(V)
    PK, PQ = P[:M], P[M:]
    Qn = PQ.shape[0]
    out = {}
    Mv = np.bincount(vals, minlength=V)

    def build(m, rho):
        topk = np.argpartition(-PK[:, :m], rho - 1, axis=1)[:, :rho]
        F = np.zeros((V, m), bool)
        F[np.repeat(vals, rho), topk.ravel()] = True
        return F

    for (rho,) in cfgs:
        for bpf in bpfs:
            Bav = bpf * M - db
            for acct in accts:
                if acct == 'real':
                    m = min(P.shape[1], Bav // V)
                    F = build(m, rho) if m > rho else None
                    used = V * m
                else:
                    # analytic start: fill of value v's array ~ 1 - (1 - rho/m)^{M_v}; then verify exactly
                    def est(m):
                        f = 1 - (1 - rho / m) ** Mv
                        return sum(m * H2(x) for x in f)
                    lo, hi = rho + 1, P.shape[1]
                    if est(lo) > Bav:
                        m = 0
                    else:
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            if est(mid) <= Bav:
                                lo = mid
                            else:
                                hi = mid - 1
                        m = lo
                    F, used = None, 0
                    while m > rho:
                        F = build(m, rho)
                        used = sum(m * H2(x) for x in F.mean(1))
                        if used <= Bav:
                            break
                        m = int(m * 0.98)
                    if m <= rho:
                        F = None
                if F is None:
                    out[(rho,), bpf, acct] = (np.full(Qn, default), 0)
                    continue
                m = F.shape[1]
                tq = np.argpartition(-PQ[:, :m], rho - 1, axis=1)[:, :rho]
                score = F[:, tq].sum(2).T.astype(np.float64)  # (Qn, V)
                out[(rho,), bpf, acct] = (decide(score, default), used + db)
    return out


_SDMC = {}


def sdm_codes(enc, u0, w, V):
    """Value codes of counters [u0, u0 + w): the first V rows of a 1024 x CH random sign block from seed
    ('B36SDMC', enc, u0) (cached per process)."""
    key = (enc, u0, V)
    if key not in _SDMC:
        if len(_SDMC) > 64:
            _SDMC.clear()
        r = np.random.default_rng(seed_of('B36SDMC', enc, u0))
        _SDMC[key] = np.where(r.random((1024, CH), dtype=np.float32) < 0.5, -1.0, 1.0).astype(np.float32)[:V]
    return _SDMC[key][:, :w]


def run_sdm(st, Y, P, pool, cfgs, bpfs, accts, enc):
    """cfgs: list of (p, nd, q). Counter u belongs to location u // nd, whose address is pool column u // nd; a code
    activates a location if its projection >= the (1 - p) quantile of N(0, 1). Counter u stores
    S_u = sum over stored facts activating its location of c_u(value) (random +-1 codes, seeded per block);
    q 'tern' keeps sign(S) in {-1, 0, 1}, 'cnt' keeps S. Read: score(v) = sum over the query's active counters of
    Q(S_u) c_u(v). D (counters) = the largest prefix that fits the budget (<= min(2 B, SDM_MAX))."""
    from scipy.stats import norm
    M, V = st['M'], st['V']
    vals = st['vals']
    default = default_value(vals, V)
    db = def_bits(V)
    Qn = Y.shape[0] - M
    out = {}
    Bmax = max(bpfs) * M
    Dcap = min(SDM_MAX, 2 * Bmax)

    def proj_cols(j0, j1):  # pool projections for columns [j0, j1)
        if j1 <= P.shape[1]:
            return P[:, j0:j1]
        parts = []
        for b0 in range(j0 - j0 % CH, j1, CH):
            a, z = max(j0, b0), min(j1, b0 + CH)
            if z <= P.shape[1]:
                parts.append(P[:, a:z])
            else:
                parts.append(Y @ pool.block(b0)[:, a - b0:z - b0])
        return np.concatenate(parts, 1)

    by_pn = {}
    for c in cfgs:
        by_pn.setdefault(c[:2], []).append(c)
    for (p, nd), lst in by_pn.items():
        tau = float(norm.ppf(1 - p))
        qs = sorted({c[2] for c in lst})
        # phase 1: stored counters (all blocks up to Dcap)
        S = np.zeros(Dcap, np.float32)
        for u0 in range(0, Dcap, CH):
            w = min(CH, Dcap - u0)
            loc = np.arange(u0, u0 + w) // nd
            Ak = (proj_cols(int(loc[0]), int(loc[-1]) + 1)[:M] >= tau).astype(np.float32)  # (M, nloc)
            Ak = Ak[:, loc - loc[0]]
            S[u0:u0 + w] = (Ak * sdm_codes(enc, u0, w, V)[vals]).sum(0)
        S = np.rint(S).astype(np.int64)
        for q in qs:
            sym = np.sign(S) if q == 'tern' else S
            Ds = {}
            for bpf in bpfs:
                Bav = bpf * M - db
                for acct in accts:
                    D = _sdm_fit(sym, q, Bav, acct, min(Dcap, 2 * bpf * M))
                    Ds[bpf, acct] = D
            need = sorted({d for d, _ in Ds.values() if d > 0})
            score = np.zeros((Qn, V), np.float64)
            snaps = {}
            done = 0
            for D in need:
                for u0 in range(done - done % CH, D, CH):
                    a, z = max(done, u0), min(D, u0 + CH)
                    if z <= a:
                        continue
                    loc = np.arange(a, z) // nd
                    Aq = (proj_cols(int(loc[0]), int(loc[-1]) + 1)[M:] >= tau).astype(np.float32)[:, loc - loc[0]]
                    Cb = sdm_codes(enc, u0, CH, V)[:, a - u0:z - u0]
                    score += ((Aq * sym[a:z].astype(np.float32)) @ Cb.T).astype(np.float64)
                done = D
                snaps[D] = decide(score, default)
            for (bpf, acct), (D, used) in Ds.items():
                out[(p, nd, q), bpf, acct] = ((snaps[D], used + db) if D > 0 else (np.full(Qn, default), 0))
    return out


def _sdm_bits(sym, D, q, acct):
    if D <= 0:
        return 0.0
    s = sym[:D]
    if acct == 'ideal':
        return D * H0(np.bincount(s - s.min()))
    if q == 'tern':
        return 2.0 * D
    return D * math.ceil(math.log2(2 * int(np.abs(s).max()) + 1 + 1e-9) if np.abs(s).max() > 0 else 1) + 8


def _sdm_fit(sym, q, Bav, acct, Dmax):
    lo, hi = 0, Dmax
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _sdm_bits(sym, mid, q, acct) <= Bav:
            lo = mid
        else:
            hi = mid - 1
    D = lo
    while D > 0 and _sdm_bits(sym, D, q, acct) > Bav:
        D = int(D * 0.99)
    return D, _sdm_bits(sym, D, q, acct)
