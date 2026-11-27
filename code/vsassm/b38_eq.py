"""B38 (prereg/B38_ecc_equal_cells_prereg.md): cells per stored fact that the VSA store and error-corrected tables need
for >= 99.5% exact reads, on simulated phase-change cells, under i.i.d. cell errors and under localized damage.

python3 b38_eq.py selftest                                   # synthetic pairs only
python3 b38_eq.py dev  STORES.json.gz OUT.json [--workers N]  # synthetic dev stores: every design x budget x i.i.d. condition
python3 b38_eq.py test STORES.json.gz PLAN.json OUT.json [--workers N]   # the registered run (plan from b38_design.py)

Every stored bit of every arm is one cell (a differential pair of PCM devices read as a sign; B25/B34's cell model).
B34 established that the device model's sign errors are i.i.d. per cell and symmetric, so the i.i.d. conditions flip
each cell independently at the device model's measured 30-day bit-error rate p(k) (results/b34_ber.json); a check
(b38_check.py device) re-reads a subset with the device model itself.

Budget B = cells per stored (key, j) pair; an arm on a store with n pairs may use at most floor(B * n) cells, and
everything it keeps between write and read is on those cells (directory, parity, codebook where stored). Stateless
logic is free in every arm: hashing the query key, generating seeded codes from strings, binding, majority, decoders.
The value vocabulary (index -> value string) is the shared output dictionary of every arm (B34's convention).

Arms
  V-plain  the B22/B25 bipolar VSA store (sign of the bundle), load set by the budget (S shards of d <= 8192 cells,
           d = floor(B n / S), S = ceil(B n / 8192)); codebook regenerated from the value strings (0 cells). Read rules
           from one set of clean-up similarities:
             V-pub   B25's rule, threshold 0.5 on the rms-scaled similarity at every j (the published store).
             V-Snt   B34's: first value by argmax, threshold 0.5 at j >= 1.
             V-am:c  first value by argmax; a later value is accepted if its best similarity >= c x the first value's.
  V-last   the store with the key's last value bound with a LAST code; clean-up over 2 x N_values candidates (value,
           value x LAST); stop when the LAST variant wins. No threshold (the analogue of the tables' last flag).
  V-cb     V-last with the value codebook (N_values x d) stored on the same cells and read as signs (item memory
           counted); load m (pairs per shard) chosen on dev.
  T-rep    B34's T-v: minimal perfect hash (displacements) + rows (last flag, value index), every bit repeated r times.
  T-vhdd   B34's T-vhdd: displacements and values as random high-dimensional codes decoded by similarity.
  E-BCH    T-rep's logical bits (displacements + rows) in shortened binary BCH codewords (n = 63 ... 1023, any t),
           each coded bit repeated rho >= 1 times (fractional: r or r + 1), majority per coded bit (ties = erasures),
           errors-and-erasures bounded-distance decoding; (n, t) chosen per store by the closed form at the condition's
           cell-error rate.
  E-LDPC   the same logical bits in IRA LDPC codewords (<= 1024 information bits, rate R in RATES), each coded bit
           repeated rho >= 1 times, soft combining of the copies into channel log-likelihood ratios, sum-product belief
           propagation (ldpc package, 60 iterations); R chosen on dev.
  E-CAM    B34's T-iix (hashed-key CAM: h-bit hash of (key, j), last flag, value index per row) in BCH codewords as
           E-BCH; every row decoded, exact match of the decoded hash, nearest Hamming as fallback.

Physical array: cells in rows of 256, tiles of 256 x 256 cells; an arm's cells in its natural order (vectors / codewords
/ copies contiguous) or permuted by a seeded random interleaver over the whole array.
"""
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

import b25_pcm as P
import b34_inmem as I34
import ruler_vsa as V

D = 8192
GRID = [float(2 ** (i / 4)) for i in range(12, 49)]          # 8 ... 4096 cells per stored pair, quarter octaves
TILE_ROWS, ROW = 256, 256
TARGET = 0.995
P30 = {1: 0.0, 2: 0.00013175, 4: 0.0208065, 8: 0.12263475, 16: 0.27600645, 32: 0.374429}   # b34_ber.json, 30 d
IID = collections.OrderedDict([("k1", P30[1]), ("k2", P30[2]), ("k4", P30[4]), ("e05", 0.05), ("k8", P30[8]),
                               ("k16", P30[16]), ("k32", P30[32])])
M3_EPS = collections.OrderedDict([("k8", P30[8]), ("e16", 0.16), ("e20", 0.20), ("e24", 0.24), ("k16", P30[16]),
                                  ("e32", 0.32), ("k32", P30[32])])
DAMAGE_EPS = ("k4", "e05", "k8", "k16")                       # matched i.i.d. rates; damaged fraction f = 2 eps
GEOMS = ("lines", "clust", "contig")
LAYOUTS = ("il", "nat")
NEXT_HARDER = {"k1": "k2", "k2": "k4", "k4": "e05", "e05": "k8", "k8": "k16", "k16": "k32", "k32": None}
AM_C = (0.5, 0.6, 0.7, 0.8, 0.9)
CB_LOADS = (1, 2, 4, 8, 16, 32, 64)
VHDD_SHARES = (0.1, 0.2, 0.3, 0.4, 0.5)
RATES = (0.25, 1 / 3, 0.5, 2 / 3, 0.75, 5 / 6, 0.9)
IRA_DV = {0.25: 6, 1 / 3: 6, 0.5: 5, 2 / 3: 6, 0.75: 5, 5 / 6: 4, 0.9: 5}   # synthetic BSC design check (record)
LDPC_K = 1024
LDPC_ITERS = 60
BCH_M = (6, 7, 8, 9, 10)
LAM = 5
ARMS = ("V-plain", "V-last", "V-cb", "T-rep", "T-vhdd", "E-BCH", "E-LDPC", "E-CAM")


def seed_of(*parts):
    return V._seed("b38:" + ":".join(str(p) for p in parts)) % (2 ** 62)


# =============================================================================================== conditions
def parse_cond(name):
    """'k8' -> iid; 'lines-il-k8' / 'contig-nat-e05' / 'lines-il-known-k4' -> damage at f = 2 eps(matched)."""
    parts = name.split("-")
    if len(parts) == 1:
        eps = IID.get(name, M3_EPS.get(name))
        return dict(kind="iid", eps=eps, match=name)
    geom, lay = parts[0], parts[1]
    known = parts[2] == "known"
    m = parts[-1]
    eps = IID.get(m, M3_EPS.get(m))
    return dict(kind="dmg", geom=geom, layout=lay, known=known, f=2 * eps, eps=eps, match=m)


def all_conditions():
    c = list(IID)
    for g in GEOMS:
        for lay in LAYOUTS:
            for m in DAMAGE_EPS:
                c.append("%s-%s-%s" % (g, lay, m))
    for m in DAMAGE_EPS:
        c.append("lines-il-known-%s" % m)
    return c


class Channel:
    """Applies a condition to an arm's cells (natural order). Common random numbers: one uniform per cell for the i.i.d.
    flips (per store, arm, budget); the damage map is a property of the physical array (per store, budget, geometry),
    shared by every arm; the interleaver is per arm."""

    def __init__(self, sid, arm_tag, B, ncells, budget_cells):
        self.sid, self.tag, self.B, self.N = sid, arm_tag, B, ncells
        self.total = max(budget_cells, ncells)
        self._u = None
        self._perm = None

    def u(self):
        if self._u is None:
            self._u = np.random.default_rng(seed_of(self.sid, self.tag, "%.4f" % self.B, "u")).random(self.N,
                                                                                                       dtype=np.float32)
        return self._u

    def positions(self, layout):
        if layout == "nat":
            return np.arange(self.N, dtype=np.int64)
        if self._perm is None:
            self._perm = np.random.default_rng(seed_of(self.sid, self.tag, "%.4f" % self.B, "perm")).permutation(
                self.total)[:self.N].astype(np.int64)
        return self._perm

    def dead_mask(self, geom, f, layout):
        pos = self.positions(layout)
        nrows = -(-self.total // ROW)
        ntiles = -(-nrows // TILE_ROWS)
        sb = (self.sid, "%.4f" % self.B, geom)
        if geom == "lines":
            q = 1 - math.sqrt(max(0.0, 1 - f))
            ur = np.random.default_rng(seed_of(*sb, "rows")).random(nrows)
            uc = np.random.default_rng(seed_of(*sb, "cols")).random((ntiles, ROW))
            row = pos // ROW
            return (ur[row] < q) | (uc[row // TILE_ROWS, pos % ROW] < q)
        if geom == "clust":
            nbr = TILE_ROWS // 16
            ub = np.random.default_rng(seed_of(*sb, "blocks")).random((ntiles, nbr, ROW // 16))
            row = pos // ROW
            return ub[row // TILE_ROWS, (row % TILE_ROWS) // 16, (pos % ROW) // 16] < f
        if geom == "contig":
            u0 = np.random.default_rng(seed_of(*sb, "start")).random()
            # the arm's own array: a band of round(f N) cells starting at a uniform cell, wrapping
            start = int(u0 * self.N)
            band = int(round(f * self.N))
            rel = (np.arange(self.N, dtype=np.int64) if layout == "nat" else np.argsort(np.argsort(pos))) - start
            return (rel % self.N) < band
        raise ValueError(geom)

    def stuck(self, layout):
        pos = self.positions(layout)
        sg = np.random.default_rng(seed_of(self.sid, "%.4f" % self.B, "stuck")).integers(0, 2, self.total,
                                                                                        dtype=np.int8) * 2 - 1
        return sg[pos]

    def apply(self, cells, cond):
        """cells: int8 +-1 (natural order). Returns int8 signs as read, 0 at known-erased cells."""
        c = parse_cond(cond) if isinstance(cond, str) else cond
        if c["kind"] == "iid":
            if c["eps"] <= 0:
                return cells.copy()
            return np.where(self.u() < c["eps"], -cells, cells).astype(np.int8)
        dead = self.dead_mask(c["geom"], c["f"], c["layout"])
        if c["known"]:
            return np.where(dead, 0, cells).astype(np.int8)
        return np.where(dead, self.stuck(c["layout"]), cells).astype(np.int8)


# =============================================================================================== store
class Store38:
    def __init__(self, sid, pairs):
        self.sid = sid
        self.s = I34.Store(pairs, "b26b:%s" % sid.replace("/", ":"))
        s = self.s
        self.n = s.n
        self.Nv = s.Nv
        cd = s.cd
        self.Kc = cd["Kc"].astype(np.int8)                      # all keys x 8192
        self.Vc = cd["Vc"].astype(np.int8)
        self.Pc = cd["Pc"].astype(np.int8)
        self.J = cd["maxj"]                                    # B25's loop bound: j = 0 .. max j + 1
        self.L = P.bip(["last"], "lastflag", D)[0].astype(np.int8)
        ki = {k: i for i, k in enumerate(cd["allk"])}
        self.pk = np.array([ki[k] for k in s.rk])               # key index per pair
        self.pv = s.rv
        self.pj = s.rj
        self.plast = s.rlast
        self.qk = np.array([ki[k] for k in s.keys])             # probed keys
        self.want = s.want
        self.Vf = None

    def vf(self, d):
        if self.Vf is None or self.Vf.shape[1] != d:
            self.Vf = self.Vc[:, :d].astype(np.float32)
        return self.Vf


def exact(got, want):
    return [i for i, (g, w) in enumerate(zip(got, want)) if g != w]


# =============================================================================================== VSA arms
def vsa_build(S38, B, kind, load=None):
    """kind: plain / last / cb. Returns dict(cells int8 natural order, S, d, shard, mem rows, rms0) or None."""
    n, Nv = S38.n, S38.Nv
    total = int(B * n)
    if kind == "cb":
        S = max(1, math.ceil(n / load))
        d = min(D, total // (S + Nv))
    else:
        S = max(1, math.ceil(total / D))
        d = min(D, total // S)
    if d < 16:
        return None
    shard_k = np.array([V._seed("shard:" + k) % S for k in S38.s.cd["allk"]])
    ps = shard_k[S38.pk]
    prod = (S38.Kc[S38.pk, :d] * S38.Pc[S38.pj, :d] * S38.Vc[S38.pv, :d]).astype(np.int32)
    if kind in ("last", "cb"):
        prod = np.where(S38.plast[:, None] > 0, prod * S38.L[None, :d], prod)
    mem = np.zeros((S, d), np.int32)
    order = np.argsort(ps, kind="stable")
    sp_ = ps[order]
    starts = np.flatnonzero(np.r_[True, sp_[1:] != sp_[:-1]])
    mem[sp_[starts]] = np.add.reduceat(prod[order], starts, axis=0)
    b1 = np.sign(mem).astype(np.int8)
    ties = b1 == 0
    b1[ties] = np.random.default_rng(0).choice(np.array([-1, 1], np.int8), size=int(ties.sum()))
    rms0 = np.sqrt(np.mean(mem.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    cells = b1.ravel()
    if kind == "cb":
        cells = np.concatenate([cells, S38.Vc[:, :d].ravel()])
    return dict(cells=cells, S=S, d=d, shard_k=shard_k, rms0=rms0, kind=kind, load=load)


def vsa_read(S38, bd, sig):
    """Clean-up similarities of every probed key at j = 0 .. J-1. Returns per-key arrays (best value, best sim scaled
    as B25, raw best sim, LAST-tag of the winner for last/cb stores)."""
    S, d = bd["S"], bd["d"]
    rows = sig[:S * d].reshape(S, d).astype(np.float32)
    if bd["kind"] == "cb":
        Vn = sig[S * d:S * d + S38.Nv * d].reshape(S38.Nv, d).astype(np.float32)
    else:
        Vn = S38.vf(d)
    sh = bd["shard_k"][S38.qk]
    X = rows[sh] * S38.Kc[S38.qk, :d]
    cnt = np.maximum((X != 0).sum(1), 1).astype(np.float32)          # components not erased
    J = S38.J
    nq = len(sh)
    best = np.zeros((nq, J), np.int64)
    sim = np.zeros((nq, J), np.float32)
    tag = np.zeros((nq, J), bool)
    Lf = S38.L[:d].astype(np.float32)
    for j in range(J):
        Y = X * S38.Pc[j, :d]
        s0 = (Y @ Vn.T) / cnt[:, None]
        if bd["kind"] == "plain":
            b = s0.argmax(1)
            best[:, j] = b
            sim[:, j] = s0[np.arange(nq), b]
        else:
            s1 = ((Y * Lf) @ Vn.T) / cnt[:, None]
            m0, m1 = s0.max(1), s1.max(1)
            useL = m1 > m0
            best[:, j] = np.where(useL, s1.argmax(1), s0.argmax(1))
            sim[:, j] = np.maximum(m0, m1)
            tag[:, j] = useL
    scale = bd["rms0"][sh]
    return dict(best=best, sim=sim, simsc=sim * scale[:, None], tag=tag)


def vsa_rule(rd, rule):
    best, sim, simsc, tag = rd["best"], rd["sim"], rd["simsc"], rd["tag"]
    nq, J = best.shape
    got = []
    for a in range(nq):
        g = []
        for j in range(J):
            if rule == "last":
                g.append(int(best[a, j]))
                if tag[a, j]:
                    break
                continue
            if rule == "pub":
                ok = simsc[a, j] >= P.THR
            elif rule == "snt":
                ok = j == 0 or simsc[a, j] >= P.THR
            else:                                              # am:c
                c = float(rule.split(":")[1])
                ok = j == 0 or sim[a, j] >= c * sim[a, 0]
            if not ok:
                break
            g.append(int(best[a, j]))
        got.append(g)
    return got


VPLAIN_RULES = ["pub", "snt"] + ["am:%.1f" % c for c in AM_C]


# =============================================================================================== tables: logical content
def mphf_content(S38):
    s = S38.s
    mp = s.mphf(LAM)
    disp = (I34.bits_pm(mp["sigma"], mp["db"]) > 0).astype(np.uint8)
    pay = s.payload_bits()
    rowpay = np.full((mp["R"], pay.shape[1]), -1, np.int8)
    rowpay[mp["slot"]] = pay
    rows01 = (rowpay > 0).astype(np.uint8)
    bits = np.concatenate([disp.ravel(), rows01.ravel()])
    return dict(mp=mp, bits=bits, nd=disp.size, db=mp["db"], pb=pay.shape[1], R=mp["R"])


def mphf_read(S38, mc, bits, rng=None):
    """Reads every probed key from the (decoded) logical bits of the MPHF table (B34 T-v's read loop)."""
    s = S38.s
    mp = mc["mp"]
    disp = I34.pm_to_int((bits[:mc["nd"]].reshape(mp["B"], mc["db"]).astype(np.int8) * 2 - 1))
    rows = bits[mc["nd"]:].reshape(mc["R"], mc["pb"]).astype(np.int8) * 2 - 1
    last, val = s.decode_payload(rows)
    got = []
    for a, k in enumerate(s.keys):
        g = []
        for j in range(s.maxj + 1):
            i = s.row_of.get((k, j))
            if i is not None:
                gg, h1, h2 = mp["h0"][i], mp["h1"][i], mp["h2"][i]
            else:
                gg, h1, h2 = I34.mphf_hash(k, j, mp["R"], mp["B"])
            slot = I34.mphf_slot(h1, h2, int(disp[gg]), mp["R"])
            g.append(int(val[slot]))
            if last[slot]:
                break
        got.append(g)
    return got


def mphf_needs(S38, mc, keys=None):
    """Logical bit positions each key needs (its buckets' displacements and its rows), for the closed form; keys
    defaults to the probed keys."""
    s = S38.s
    mp = mc["mp"]
    out = []
    for a, k in enumerate(s.keys if keys is None else keys):
        pos = []
        for j in range(s.count[k]):
            i = s.row_of[(k, j)]
            g = int(mp["h0"][i])
            pos += list(range(g * mc["db"], (g + 1) * mc["db"]))
            sl = int(mp["slot"][i])
            pos += list(range(mc["nd"] + sl * mc["pb"], mc["nd"] + (sl + 1) * mc["pb"]))
        out.append(np.unique(np.array(pos, np.int64)))
    return out


def cam_content(S38):
    s = S38.s
    kc, h = I34.hash_code(s.rk, s.rj)
    pay = s.payload_bits()
    rows = np.concatenate([kc, pay], 1)
    return dict(h=h, w=rows.shape[1], bits=(rows > 0).astype(np.uint8).ravel(), n=s.n)


def cam_read(S38, cc, bits, rng):
    s = S38.s
    rows = bits.reshape(cc["n"], cc["w"]).astype(np.int8) * 2 - 1
    fp = rows[:, :cc["h"]]
    last, val = s.decode_payload(rows[:, cc["h"]:])
    fpk = {}
    for i in range(cc["n"]):
        fpk.setdefault(fp[i].tobytes(), []).append(i)
    got = []
    for a, k in enumerate(s.keys):
        g = []
        for j in range(s.maxj + 1):
            dig = np.frombuffer(hashlib.sha256(("b34h:%s\x00%d" % (k, j)).encode("utf-8")).digest(), np.uint8)
            q = (2 * np.unpackbits(dig)[:cc["h"]].astype(np.int8) - 1)
            hit = fpk.get(q.tobytes())
            if hit:
                row = hit[0]
            else:
                dots = fp.astype(np.int32) @ q.astype(np.int32) + rng.random(cc["n"])
                row = int(np.argmax(dots))
            g.append(int(val[row]))
            if last[row]:
                break
        got.append(g)
    return got


def cam_needs(S38, cc, keys=None):
    s = S38.s
    out = []
    for a, k in enumerate(s.keys if keys is None else keys):
        pos = []
        for j in range(s.count[k]):
            i = s.row_of[(k, j)]
            pos += list(range(i * cc["w"], (i + 1) * cc["w"]))
        out.append(np.unique(np.array(pos, np.int64)))
    return out


# =============================================================================================== repetition helpers
def rep_counts(Nc, cells):
    """Copies per coded bit: r or r + 1, spread evenly, summing to `cells` (None if r < 1)."""
    r = cells // Nc
    if r < 1:
        return None
    extra = cells - r * Nc
    i = np.arange(Nc, dtype=np.int64)
    plus = (((i + 1) * extra) // Nc - (i * extra) // Nc).astype(np.int64)
    return r + plus


def rep_sum(sig, counts):
    """sig: int8 cells (copies adjacent, natural order); returns (#(+1) - #(-1)) per coded bit."""
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return np.add.reduceat(sig.astype(np.int32), starts)


# =============================================================================================== BCH
_BCH = {}


def cyclotomic_k(m, t):
    n = 2 ** m - 1
    seen = set()
    for i in range(1, 2 * t + 1):
        x = i % n
        while x not in seen:
            seen.add(x)
            x = (2 * x) % n
    return n - len(seen)


def bch_designs():
    out = []
    for m in BCH_M:
        n = 2 ** m - 1
        t = 1
        while True:
            k = cyclotomic_k(m, t)
            if k < 2:
                break
            if not out or (out[-1][0], out[-1][1]) != (n, k):
                out.append((n, k, t))
            t += 1
    # keep only the largest t for each (n, k)
    best = {}
    for n, k, t in out:
        best[(n, k)] = max(t, best.get((n, k), 0))
    return sorted((n, k, t) for (n, k), t in best.items())


BCH_DESIGNS = None


# Own binary BCH codec (narrow sense, roots alpha^1 .. alpha^2t, alpha = x in GF(2^m) over the same primitive field
# polynomials as the galois library's BCH codes, so b38_check.py can compare generator polynomials, encodings and
# decoders word by word).
FIELD_POLY = {6: 67, 7: 137, 8: 285, 9: 529, 10: 1033}   # galois.BCH extension fields (primitive)


class GF2m:
    def __init__(self, m):
        self.m, self.q = m, 2 ** m - 1
        poly = FIELD_POLY[m]
        self.exp = np.zeros(2 * (self.q + 1), np.int64)
        self.log = np.zeros(self.q + 1, np.int64)
        x = 1
        for i in range(self.q):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << m):
                x ^= poly
        for i in range(self.q, 2 * (self.q + 1)):
            self.exp[i] = self.exp[i - self.q]

    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else int(self.exp[(self.log[a] + self.log[b]) % self.q])

    def inv(self, a):
        return int(self.exp[(self.q - self.log[a]) % self.q])


_GF = {}


def gf(m):
    if m not in _GF:
        _GF[m] = GF2m(m)
    return _GF[m]


class BCH:
    """(n, k, t) narrow-sense binary BCH code; systematic encoding [message | parity], polynomial coefficients highest
    degree first; shortened codes drop leading message positions. decode() is errors-only bounded-distance decoding
    (syndromes, Berlekamp-Massey, Chien search)."""

    def __init__(self, n, k, t):
        self.n, self.k, self.t = n, k, t
        self.m = int(round(math.log2(n + 1)))
        F = gf(self.m)
        self.F = F
        seen, g = set(), [1]                                     # g(x) over GF(2^m), lowest degree first
        for i in range(1, 2 * t + 1):
            if i % n in seen:
                continue
            coset, x = [], i % n
            while x not in seen:
                seen.add(x)
                coset.append(x)
                x = (2 * x) % n
            for r in coset:                                      # g(x) *= (x - alpha^r)
                ar = int(F.exp[r])
                ng = [0] * (len(g) + 1)
                for d, cf in enumerate(g):
                    ng[d + 1] ^= cf
                    ng[d] ^= F.mul(cf, ar)
                g = ng
        assert all(c in (0, 1) for c in g), "generator not binary"
        assert len(g) - 1 == n - k, (n, k, t, len(g) - 1)
        self.g_low = np.array(g, np.uint8)                       # lowest degree first
        # parity of message bit i (coefficient of x^(n-1-i)): x^(n-1-i) mod g(x), as n-k bits highest degree first
        nk = n - k
        P = np.zeros((k, nk), np.uint8)
        reg = np.zeros(nk, np.uint8)                             # remainder of x^nk mod g, then multiply by x
        gh = self.g_low[:-1][::-1]                               # g without its leading term, highest degree first
        # remainder of x^(nk + j) for j = 0 .. k-1 via an LFSR: rem(x^(d+1)) = x * rem(x^d) mod g
        rem = gh.copy()                                          # x^nk mod g = g - x^nk
        rows = []
        for j in range(k):
            rows.append(rem.copy())
            top = rem[0]
            rem = np.r_[rem[1:], 0].astype(np.uint8)
            if top:
                rem ^= gh
        # message bit i multiplies x^(n-1-i) = x^(nk + (k-1-i))
        for i in range(k):
            P[i] = rows[k - 1 - i]
        self.P = P

    def encode(self, M):
        """M: (words, ks) 0/1 messages (ks <= k, shortened). Returns (words, ks + n - k) codewords."""
        ks = M.shape[1]
        par = (M.astype(np.int64) @ self.P[self.k - ks:].astype(np.int64)) % 2
        return np.concatenate([M.astype(np.uint8), par.astype(np.uint8)], 1)

    def decode(self, w):
        """w: 0/1 received word of length ks + n - k. Returns (ok, message bits, corrections)."""
        F, t, q = self.F, self.t, self.F.q
        L = len(w)
        ks = L - (self.n - self.k)
        ones = np.flatnonzero(w)
        pw = (L - 1 - ones).astype(np.int64)
        js = np.arange(1, 2 * t + 1, dtype=np.int64)
        if len(pw):
            S = np.bitwise_xor.reduce(F.exp[np.outer(js, pw) % q], axis=1)
        else:
            S = np.zeros(2 * t, np.int64)
        if not S.any():
            return True, w[:ks].copy(), 0
        # Berlekamp-Massey in the log domain (numpy), C(x) = connection polynomial, lowest degree first
        lg, ex = F.log, F.exp
        T2 = 2 * t
        C = np.zeros(T2 + 2, np.int64)
        Bp = np.zeros(T2 + 2, np.int64)
        C[0] = Bp[0] = 1
        Lc, mm, bb = 0, 1, 1
        for r_ in range(T2):
            idx = np.arange(0, Lc + 1)
            a_, b_ = C[idx], S[r_ - idx]
            nzm = (a_ != 0) & (b_ != 0)
            d = int(np.bitwise_xor.reduce(ex[(lg[a_[nzm]] + lg[b_[nzm]]) % q])) if nzm.any() else 0
            if d == 0:
                mm += 1
                continue
            lcoef = (int(lg[d]) - int(lg[bb])) % q
            T_ = C.copy()
            nzb = np.flatnonzero(Bp[:T2 + 2 - mm])
            C[nzb + mm] ^= ex[(lg[Bp[nzb]] + lcoef) % q]
            if 2 * Lc <= r_:
                Lc, Bp, bb, mm = r_ + 1 - Lc, T_, d, 1
            else:
                mm += 1
        C = [int(x) for x in C[:Lc + 1]]
        nz = [(j, int(F.log[c])) for j, c in enumerate(C) if c]
        ii = np.arange(L, dtype=np.int64)
        val = np.zeros(L, np.int64)
        for j, lc in nz:
            val ^= F.exp[(lc - (ii * j) % q) % q]
        roots = np.flatnonzero(val == 0)
        if len(roots) != Lc:
            return False, w[:ks].copy(), -1
        out = w.copy()
        out[L - 1 - roots] ^= 1
        return True, out[:ks].copy(), int(Lc)


_BCHC = {}


def bch_code(n, k, t):
    if (n, k) not in _BCHC:
        _BCHC[(n, k)] = BCH(n, k, t)
    return _BCHC[(n, k)]


def blocks(L, kmax):
    ncw = max(1, math.ceil(L / kmax))
    kk = math.ceil(L / ncw)
    sizes = [kk] * (L // kk) + ([L % kk] if L % kk else [])
    while len(sizes) < ncw:
        sizes.append(0)
    return [s for s in sizes if s > 0]


_QR = {}


def q_rep(r, p):
    """(P(wrong), P(tie)) of the majority of r i.i.d. copies at flip rate p."""
    key = (r, p)
    if key not in _QR:
        if p <= 0:
            _QR[key] = (0.0, 0.0)
        elif r % 2:
            _QR[key] = (float(stats.binom.sf(r // 2, r, p)), 0.0)
        else:
            _QR[key] = (float(stats.binom.sf(r // 2, r, p)), float(stats.binom.pmf(r // 2, r, p)))
    return _QR[key]


_LF = {}


def bch_cw_logfail(clen, t, r, n1, p):
    """log P(codeword not decoded) for bounded-distance decoding of a codeword of clen coded bits, n1 of them with r + 1
    copies and the rest with r; a tie counts as half an error (the design rule's conservative approximation; the decoder
    itself uses ties as erasures)."""
    key = (clen, t, r, n1, p)
    if key in _LF:
        return _LF[key]
    if p <= 0:
        _LF[key] = -np.inf
        return -np.inf
    w1, t1 = q_rep(r, p)
    w2, t2 = q_rep(r + 1, p)
    pa, pb = w1 + t1 / 2, w2 + t2 / 2
    m0, m1 = clen - n1, n1
    if m1 == 0 or m0 == 0:
        m, pe = (m0, pa) if m1 == 0 else (m1, pb)
        v = float(stats.binom.logsf(t, m, pe)) if pe > 0 else -np.inf
    else:
        e = np.arange(0, t + 1)
        lp = stats.binom.logpmf(e, m0, pa) + stats.binom.logcdf(t - e, m1, pb)
        okl = np.logaddexp.reduce(lp)
        if okl < -1e-12:
            v = float(np.log(-np.expm1(okl)))
        else:   # failure below double precision of 1 - ok: union of the two halves' tails
            v = float(np.logaddexp(stats.binom.logsf(t // 2, m0, pa), stats.binom.logsf(t - t // 2 - 1, m1, pb)))
    _LF[key] = v
    return v


def ecc_layout(L, n, k, t, cells):
    """Blocks of a bitstream of L bits in shortened (n, k) codewords; copies per coded bit. None if infeasible."""
    sizes = blocks(L, k)
    clen = [s + (n - k) for s in sizes]
    Nc = sum(clen)
    counts = rep_counts(Nc, cells)
    if counts is None:
        return None
    return dict(n=n, k=k, t=t, sizes=sizes, clen=clen, Nc=Nc, counts=counts, cells=int(counts.sum()))


def needs_cat(needs):
    pos = np.concatenate(needs) if needs else np.zeros(0, np.int64)
    kid = np.concatenate([np.full(len(x), i, np.int64) for i, x in enumerate(needs)]) if needs else pos
    return pos, kid, len(needs)


def ecc_key_objective(lay, nc, eps):
    """(expected number of failed keys, log of the summed failure probabilities of the needed codewords). The first is
    the design objective; the second separates designs whose expected failures are below 1e-6. Positions are sorted
    within each key (needs_cat), so a key's codewords are found by adjacent de-duplication."""
    if eps <= 0:
        return 0.0, -np.inf
    pos, kid, nk = nc
    ncw = len(lay["sizes"])
    kk = lay["sizes"][0]                                         # blocks are consecutive, all of size kk but the last
    cst = np.r_[0, np.cumsum(lay["clen"])]
    r = int(lay["counts"].min())
    plus = np.r_[0, np.cumsum(lay["counts"] > r)]
    lf = np.array([bch_cw_logfail(lay["clen"][i], lay["t"], r, int(plus[cst[i + 1]] - plus[cst[i]]), eps)
                   for i in range(ncw)])
    cw = pos // kk
    keep = np.r_[True, (kid[1:] != kid[:-1]) | (cw[1:] != cw[:-1])]
    lfu = lf[cw[keep]]
    pf = np.exp(lfu)
    ok = np.bincount(kid[keep], weights=np.log1p(-np.minimum(pf, 1 - 1e-16)), minlength=nk)
    E = float(-np.expm1(ok).sum())
    return E, (float(np.logaddexp.reduce(lfu)) if len(lfu) else -np.inf)


def bch_choose(L, needs, cells, eps, eps_next):
    """The (n, k, t) minimizing the closed-form expected number of failed keys (every stored key) at eps (below 1e-6
    compared by the log of the summed codeword failure probabilities); ties: the same at eps_next, then the shortest
    code. Returns (n, k, t) or None."""
    global BCH_DESIGNS
    if BCH_DESIGNS is None:
        BCH_DESIGNS = bch_designs()
    nc = needs_cat(needs)
    best = None

    def kk(e, lg):
        return (round(e, 9), round(lg, 6)) if e < 1e-6 else (round(e, 9), 0.0)
    for n, k, t in BCH_DESIGNS:
        lay = ecc_layout(L, n, k, t, cells)
        if lay is None:
            continue
        e0, l0 = ecc_key_objective(lay, nc, eps)
        e1, l1 = ecc_key_objective(lay, nc, eps_next) if eps_next else (0.0, -np.inf)
        key = (kk(e0, l0), kk(e1, l1), n)
        if best is None or key < best[0]:
            best = (key, (n, k, t))
    return None if best is None else best[1]


def bch_encode_stream(bits, lay):
    code = bch_code(lay["n"], lay["k"], lay["t"])
    out, o = [], 0
    for s_ in lay["sizes"]:
        out.append(code.encode(bits[o:o + s_].astype(np.uint8)[None, :])[0])
        o += s_
    return np.concatenate(out)


def bch_decode_stream(coded_true, sums, lay, rng):
    """sums: copy sums per coded bit (+ = bit 1). Ties are erasures. Errors-and-erasures bounded-distance decoding:
    a codeword with 2 e + s <= 2t (e errors, s erasures) is corrected (guaranteed for a bounded-distance decoder;
    b38_check.py verifies it against the galois library); otherwise the decoder runs on both erasure fillings and the
    decodable result with fewer corrections is kept; if neither decodes, the received information bits (erasures at
    random) are returned."""
    code = bch_code(lay["n"], lay["k"], lay["t"])
    out = []
    o = 0
    st = [0, 0, 0]                                                # corrected-by-bound, decoded, failed
    for s_, cl in zip(lay["sizes"], lay["clen"]):
        sm = sums[o:o + cl]
        tr = coded_true[o:o + cl]
        hard = (sm > 0).astype(np.uint8)
        era = sm == 0
        e = int(((hard != tr) & ~era).sum())
        se = int(era.sum())
        if 2 * e + se <= 2 * lay["t"]:
            out.append(tr[:s_])
            st[0] += 1
        else:
            cands = []
            for fill in ((0, 1) if se else (0,)):
                w = hard.copy()
                w[era] = fill
                ok, msg, ne = code.decode(w)
                if ok:
                    cands.append((ne, fill, msg))
            if cands:
                cands.sort(key=lambda x: (x[0], x[1]))
                out.append(cands[0][2])
                st[1] += 1
            else:
                w = hard.copy()
                w[era] = rng.integers(0, 2, se)
                out.append(w[:s_])
                st[2] += 1
        o += cl
    return np.concatenate(out), st


# =============================================================================================== LDPC (IRA)
_IRA = {}


def ira_code(K, R):
    """Systematic IRA LDPC code: H = [H1 | H2], H2 dual-diagonal (accumulator); info columns of weight dv placed on
    the currently lightest rows (random among ties); fixed seed per (K, R)."""
    import scipy.sparse as sp
    key = (K, round(R, 4))
    if key in _IRA:
        return _IRA[key]
    M = max(2, int(round(K * (1 - R) / R)))
    dv = min(IRA_DV[min(IRA_DV, key=lambda r: abs(r - R))], M)
    rng = np.random.default_rng(seed_of("ira", K, "%.4f" % R))
    deg = np.zeros(M, np.int64)
    rows, cols = [], []
    for c in rng.permutation(K):
        r = np.argpartition(deg + rng.random(M), dv - 1)[:dv] if dv < M else np.arange(M)
        deg[r] += 1
        rows += list(r)
        cols += [c] * len(r)
    i = np.arange(M)
    H1 = sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(M, K))
    H = sp.csr_matrix((np.ones(len(rows) + 2 * M - 1, np.uint8),
                       (np.r_[rows, i, i[1:]], np.r_[cols, K + i, K + i[:-1]])), shape=(M, K + M))
    _IRA[key] = dict(H=H, H1=H1, K=K, M=M, N=K + M, dv=dv, dec={})
    return _IRA[key]


def ira_encode(code, U):
    s = (code["H1"] @ U.T.astype(np.int64) % 2).T.astype(np.uint8)
    p = np.bitwise_xor.accumulate(s, axis=1)
    return np.concatenate([U.astype(np.uint8), p], 1)


def ldpc_layout(L, R, cells):
    ncw = max(1, math.ceil(L / LDPC_K))
    K = math.ceil(L / ncw)
    code = ira_code(K, R)
    pad = ncw * K - L                                             # shortened (known-zero) bits, not stored
    Nstored = ncw * code["N"] - pad
    counts = rep_counts(Nstored, cells)
    if counts is None:
        return None
    return dict(R=R, K=K, ncw=ncw, pad=pad, N=code["N"], Nc=Nstored, counts=counts, cells=int(counts.sum()))


def ldpc_encode_stream(bits, lay):
    code = ira_code(lay["K"], lay["R"])
    U = np.zeros(lay["ncw"] * lay["K"], np.uint8)
    U[:len(bits)] = bits
    C = ira_encode(code, U.reshape(lay["ncw"], lay["K"]))
    # stored = all coded bits except the padded info bits (the last pad info positions of the last codeword)
    keep = np.ones(C.size, bool)
    if lay["pad"]:
        last = (lay["ncw"] - 1) * lay["N"]
        keep[last + lay["K"] - lay["pad"]:last + lay["K"]] = False
    return C.ravel()[keep], keep


def ldpc_decode_stream(sums, lay, keep, eps, rng):
    from ldpc import BpDecoder
    code = ira_code(lay["K"], lay["R"])
    e = min(max(eps, 1e-6), 0.4999)
    Le = math.log((1 - e) / e)
    full = np.zeros(lay["ncw"] * lay["N"], np.float64)            # LLR for bit = 1 minus bit = 0 (votes x Le)
    full[keep] = sums * Le
    full[~keep] = -60.0                                          # shortened bits: known 0
    hard = (full > 0).astype(np.uint8)
    ties = full == 0
    hard[ties] = rng.integers(0, 2, int(ties.sum()))
    pr = 1.0 / (1.0 + np.exp(np.minimum(np.abs(full), 60.0)))
    pr = np.clip(pr, 1e-15, 0.5)
    if "bp" not in code["dec"]:
        code["dec"]["bp"] = BpDecoder(code["H"], error_rate=0.1, max_iter=LDPC_ITERS, bp_method="product_sum",
                                      input_vector_type="received_vector")
    bpd = code["dec"]["bp"]
    out = []
    nfail = 0
    own = os.environ.get("B38_LDPC_OWN") == "1"                 # exploratory: b38_check's own sum-product decoder
    if own:
        import b38_check as K
    for w in range(lay["ncw"]):
        sl = slice(w * lay["N"], (w + 1) * lay["N"])
        if own:
            llr = np.where(hard[sl] == 0, 1, -1) * np.log((1 - pr[sl]) / pr[sl])
            d, conv = K._sp_decode(code["H"], llr, LDPC_ITERS)
            nfail += int(not conv)
        else:
            bpd.update_channel_probs(pr[sl])
            d = bpd.decode(hard[sl])
            nfail += int(not bpd.converge)
        out.append(np.asarray(d[:lay["K"]], np.uint8))
    return np.concatenate(out), nfail


# =============================================================================================== one arm, one budget
def build_arm(S38, arm, B, design):
    """Returns an object with .cells (int8 natural order) and .read(sig, cond, rng) -> list of probed keys' values."""
    total = int(B * S38.n)
    if arm in ("V-plain", "V-last", "V-cb"):
        kind = {"V-plain": "plain", "V-last": "last", "V-cb": "cb"}[arm]
        bd = vsa_build(S38, B, kind, load=design if arm == "V-cb" else None)
        if bd is None:
            return None
        return dict(cells=bd["cells"], bd=bd, params=dict(S=bd["S"], d=bd["d"], load=design))
    s = S38.s
    if arm in ("T-rep", "T-vhdd"):
        info = I34.layout(s, "T-v" if arm == "T-rep" else "T-vhdd", total,
                          share=design if arm == "T-vhdd" else None, lam=LAM)
        if info is None:
            return None
        names = list(info["arrays"])
        shapes = [info["arrays"][a][0].shape for a in names]
        cells = np.concatenate([info["arrays"][a][0].ravel() for a in names]).astype(np.int8)
        p = {kk: v for kk, v in info.items() if kk not in ("arrays", "fields_logical", "mphf")}
        return dict(cells=cells, info=info, names=names, shapes=shapes, params=p)
    if arm in ("E-BCH", "E-CAM"):
        cont = mphf_content(S38) if arm == "E-BCH" else cam_content(S38)
        n, k, t = design
        lay = ecc_layout(len(cont["bits"]), n, k, t, total)
        if lay is None:
            return None
        coded = bch_encode_stream(cont["bits"], lay)
        cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
        return dict(cells=cells, cont=cont, lay=lay, coded=coded,
                    params=dict(n=n, k=k, t=t, ncw=len(lay["sizes"]), rho=lay["cells"] / lay["Nc"], L=len(cont["bits"])))
    if arm == "E-LDPC":
        cont = mphf_content(S38)
        lay = ldpc_layout(len(cont["bits"]), design, total)
        if lay is None:
            return None
        coded, keep = ldpc_encode_stream(cont["bits"], lay)
        cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
        return dict(cells=cells, cont=cont, lay=lay, coded=coded, keep=keep,
                    params=dict(R=design, K=lay["K"], ncw=lay["ncw"], rho=lay["cells"] / lay["Nc"],
                                L=len(cont["bits"])))
    raise ValueError(arm)


def read_arm(S38, arm, A, sig, cond, rng):
    """Returns {rule: list of values per probed key} (one rule for every arm except V-plain) and diagnostics."""
    if arm in ("V-plain", "V-last", "V-cb"):
        rd = vsa_read(S38, A["bd"], sig)
        if arm == "V-plain":
            return {r: vsa_rule(rd, r) for r in VPLAIN_RULES}, {}
        return {"last": vsa_rule(rd, "last")}, {}
    s = S38.s
    if arm in ("T-rep", "T-vhdd"):
        sigd, o = {}, 0
        for a, shp in zip(A["names"], A["shapes"]):
            sz = int(np.prod(shp))
            sigd[a] = sig[o:o + sz].reshape(shp)
            o += sz
        return {"": I34.read_table(s, A["info"], sigd, rng)}, {}
    lay = A["lay"]
    sums = rep_sum(sig, lay["counts"])
    if arm in ("E-BCH", "E-CAM"):
        bits, st = bch_decode_stream(A["coded"], sums, lay, rng)
        rd = mphf_read(S38, A["cont"], bits) if arm == "E-BCH" else cam_read(S38, A["cont"], bits, rng)
        return {"": rd}, {"cw_bound": st[0], "cw_lib": st[1], "cw_fail": st[2]}
    c = parse_cond(cond)
    bits, nf = ldpc_decode_stream(sums, lay, A["keep"], c["eps"], rng)
    return {"": mphf_read(S38, A["cont"], bits[:len(A["cont"]["bits"])])}, {"cw_nonconv": nf}


# =============================================================================================== jobs
def run_job(job):
    """job: dict(sid, pairs, arm, evals=[(B, design, [conds])]). Returns per (B, design, cond, rule): [n_exact, n,
    failed probed-key indices]."""
    t0 = time.time()
    S38 = Store38(job["sid"], job["pairs"])
    nq = len(S38.want)
    out = []
    for ev in job["evals"]:
        B, design, conds = ev[0], ev[1], ev[2]
        dkey = ev[3] if len(ev) > 3 else None
        A = build_arm(S38, job["arm"], B, design)
        if A is None:
            for c in conds:
                out.append(dict(B=B, design=design, cond=c, rule=None, exact=0, n=nq, failed=None, infeasible=True,
                                dkey=dkey))
            continue
        tag = "%s|%s" % (job["arm"], json.dumps(design))
        ch = Channel(job["sid"], tag, B, len(A["cells"]), int(B * S38.n))
        for c in conds:
            sig = ch.apply(A["cells"], c)
            rng = np.random.default_rng(seed_of(job["sid"], tag, "%.4f" % B, c, "dec"))
            res, diag = read_arm(S38, job["arm"], A, sig, c, rng)
            for rule, got in res.items():
                f = exact(got, S38.want)
                out.append(dict(B=B, design=design, cond=c, rule=rule, exact=nq - len(f), n=nq, failed=f,
                                cells=int(len(A["cells"])), diag=diag, dkey=dkey))
        params = A["params"]
        out.append(dict(B=B, design=design, params=params, cells=int(len(A["cells"])), budget=int(B * S38.n)))
    return dict(sid=job["sid"], arm=job["arm"], n_pairs=S38.n, n_values=S38.Nv, nq=nq, res=out, sec=time.time() - t0)


def load_stores(path):
    return [(sid, [tuple(p) for p in pairs]) for sid, pairs in json.load(gzip.open(path, "rt"))]


def pool_run(jobs, workers, out_path=None):
    from multiprocessing import Pool
    done = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            done.append(r)
            print("done %-28s %-7s %5.0fs  %d/%d" % (r["sid"], r["arm"], r["sec"], len(done), len(jobs)), flush=True)
            if out_path and len(done) % 25 == 0:
                json.dump(done, open(out_path + ".partial", "w"))
    return done


# =============================================================================================== dev: designs x budgets x i.i.d.
def dev_jobs(stores, arms, budgets=None, chunk=6, only_designs=None):
    """Every design of every arm at every grid budget and every i.i.d. condition (E-BCH / E-CAM: the closed-form choice
    per store, computed inside the job)."""
    budgets = budgets or GRID
    conds = list(IID)
    jobs = []
    for sid, pairs in stores:
        for arm in arms:
            if arm == "V-plain" or arm == "V-last" or arm == "T-rep":
                designs = [None]
            elif arm == "V-cb":
                designs = list(CB_LOADS)
            elif arm == "T-vhdd":
                designs = list(VHDD_SHARES)
            elif arm == "E-LDPC":
                designs = list(RATES)
            else:
                designs = ["auto"]
            if only_designs is not None:
                designs = [d for d in designs if d in only_designs]
            ev = [(B, d, conds) for B in budgets for d in designs]
            for i in range(0, len(ev), chunk):
                jobs.append(dict(sid=sid, pairs=pairs, arm=arm, evals=ev[i:i + chunk]))
    jobs.sort(key=lambda j: -len(j["pairs"]))
    return jobs


def run_job_auto(job):
    """As run_job, but designs 'auto' (E-BCH / E-CAM) are resolved per (budget, condition) by the closed form."""
    if not any(isinstance(d, str) and d.startswith("auto") for _, d, _ in job["evals"]):
        return run_job(job)
    S38 = Store38(job["sid"], job["pairs"])
    cont = mphf_content(S38) if job["arm"] == "E-BCH" else cam_content(S38)
    allk = list(S38.s.count)                                     # the design rule sees every stored key, not the probe
    needs = mphf_needs(S38, cont, allk) if job["arm"] == "E-BCH" else cam_needs(S38, cont, allk)
    ev = []
    for B, d, conds in job["evals"]:
        for c in conds:
            # "auto": designed for the evaluated condition's (matched) cell-error rate; "auto@k8": designed for k8
            m = d.split("@")[1] if isinstance(d, str) and "@" in d else parse_cond(c)["match"]
            nh = NEXT_HARDER.get(m)
            des = bch_choose(len(cont["bits"]), needs, int(B * S38.n), parse_cond(m)["eps"], IID[nh] if nh else None)
            ev.append((B, None if des is None else list(des), [c], d))
    # group conditions sharing a design (and the same design rule, so "auto" and "auto@k8" rows stay apart)
    grp = collections.OrderedDict()
    for B, d, cs, dk in ev:
        grp.setdefault((B, json.dumps(d), dk), []).extend(cs)
    j2 = dict(job)
    j2["evals"] = [(B, json.loads(d), sorted(set(cs)), dk) for (B, d, dk), cs in grp.items()]
    r = run_job_feasible(j2)
    return r


def run_job_feasible(job):
    ev = []
    infeasible = []
    for e in job["evals"]:
        if e[1] is None:
            infeasible += [(e[0], c, e[3] if len(e) > 3 else None) for c in e[2]]
        else:
            ev.append(e)
    j2 = dict(job)
    j2["evals"] = ev
    r = run_job(j2)
    for B, c, dk in infeasible:
        r["res"].append(dict(B=B, design=None, cond=c, rule=None, exact=0, n=r["nq"], failed=None, infeasible=True,
                             dkey=dk))
    return r


def pool_run_auto(jobs, workers, out_path=None):
    from multiprocessing import Pool
    done = []
    last = time.time()
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_job_auto, jobs):
            done.append(r)
            print("done %-28s %-7s %5.0fs  %d/%d" % (r["sid"], r["arm"], r["sec"], len(done), len(jobs)), flush=True)
            if out_path and time.time() - last > 900:           # checkpoint every 15 minutes (I/O only)
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    return done


# =============================================================================================== self-test
def selftest():
    rng = np.random.default_rng(1)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(600):
        k = "key%d" % rng.integers(0, 500)
        pairs.append((k, "val%d" % (i % 400), seen[k]))
        seen[k] += 1
    S38 = Store38("selftest/x/0", pairs)
    print("n", S38.n, "Nv", S38.Nv, "probed", len(S38.want), "J", S38.J)
    # 1. noiseless exactness of every arm at a generous budget
    for arm, design in [("V-plain", None), ("V-last", None), ("V-cb", 4), ("T-rep", None), ("T-vhdd", 0.3),
                        ("E-BCH", [255, 131, 18]), ("E-CAM", [255, 131, 18]), ("E-LDPC", 0.5)]:
        Bst = 4096.0 if arm == "V-cb" else 512.0                 # V-cb: its codebook costs N_values x d cells
        A = build_arm(S38, arm, Bst, design)
        res, diag = read_arm(S38, arm, A, A["cells"].copy(), "k1", np.random.default_rng(0))
        for rule, got in res.items():
            f = exact(got, S38.want)
            print("%-7s %-7s cells/budget %.3f noiseless failed %d %s" % (arm, rule, len(A["cells"]) / (Bst * S38.n),
                                                                          len(f), A["params"]))
            assert not f or rule in ("am:0.8", "am:0.9"), (arm, rule)   # a high c can reject a true later value
    # 2. channel: i.i.d. flip rate and damage fractions
    ch = Channel("selftest/x/0", "t", 64.0, 200000, 200000)
    cells = np.where(rng.random(200000) < 0.5, 1, -1).astype(np.int8)
    for c in ("k8", "lines-il-k8", "lines-nat-k8", "clust-nat-k8", "contig-nat-k8", "lines-il-known-k8"):
        sig = ch.apply(cells, c)
        pc = parse_cond(c)
        print("%-18s error rate %.4f (target eps %.4f), erased %.4f" % (c, float(((sig != cells) & (sig != 0)).mean()),
                                                                         pc["eps"], float((sig == 0).mean())))
    # 3. BCH closed-form design and a noisy run of every arm
    cont = mphf_content(S38)
    needs = mphf_needs(S38, cont)
    for eps in (0.02, 0.12, 0.27):
        for B in (64.0, 256.0):
            print("bch_choose eps %.2f B %g ->" % (eps, B), bch_choose(len(cont["bits"]), needs, int(B * S38.n), eps,
                                                                      None))
    job = dict(sid="selftest/x/0", pairs=pairs, arm="E-BCH", evals=[(128.0, "auto", ["k4", "k8", "k16"])])
    r = run_job_auto(job)
    for x in r["res"]:
        if "cond" in x:
            print("E-BCH", x["B"], x["cond"], x["design"], x["exact"], "/", x["n"], x.get("diag"))
    for arm, d in [("V-plain", None), ("V-last", None), ("E-LDPC", 1 / 3), ("T-rep", None), ("T-vhdd", 0.3)]:
        r = run_job(dict(sid="selftest/x/0", pairs=pairs, arm=arm, evals=[(128.0, d, ["k4", "k8", "k16", "lines-nat-k8"])]))
        for x in r["res"]:
            if "cond" in x:
                print(arm, x["B"], x["cond"], x["rule"], x["exact"], "/", x["n"], x.get("diag"))
    print("selftest ok")


if __name__ == "__main__":
    mode = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    if mode == "selftest":
        selftest()
    elif mode == "dev":                                         # dev STORES OUT [ARMS] --workers N
        stores = load_stores(sys.argv[2])
        arms = sys.argv[4].split(",") if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else ARMS
        only = [float(x) if "." in x else int(x) for x in sys.argv[sys.argv.index("--designs") + 1].split(",")] \
            if "--designs" in sys.argv else None
        jobs = dev_jobs(stores, arms, only_designs=only)
        print(len(jobs), "jobs", flush=True)
        res = pool_run_auto(jobs, workers, sys.argv[3])
        json.dump(res, open(sys.argv[3], "w"))
    elif mode == "test":                                        # test STORES PLAN OUT --workers N
        stores = dict(load_stores(sys.argv[2]))
        plan = json.load(open(sys.argv[3]))
        jobs = [dict(sid=j["sid"], pairs=stores[j["sid"]], arm=j["arm"],
                     evals=[(e[0], e[1], e[2]) for e in j["evals"]]) for j in plan["jobs"]]
        jobs.sort(key=lambda j: -len(j["pairs"]))
        print(len(jobs), "jobs", flush=True)
        res = pool_run_auto(jobs, workers, sys.argv[4])
        json.dump(res, open(sys.argv[4], "w"))
