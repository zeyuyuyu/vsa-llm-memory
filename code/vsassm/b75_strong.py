"""B75 (prereg/B75_strongest_options_prereg.md): the strongest options on both sides of the equal-cell device
comparison, on the paper's simulated phase-change cells (B38's binary cells, 30-day i.i.d. cell errors x 1 ... 32).

Memories run here (every bit on cells; decoders, directory logic and clean-up search not counted, as in the paper):
  U-LS     (A) the dense store U (B38's V-last: same shards, dimension, codes, cells and matched-filter read) with its
           stored vector written by a ridge least-squares solve of the stored pairs' matched-filter scores instead of
           the Hebbian sum: per shard w = Z^T (Z Z^T + alpha d I)^-1 y, y = 1 (Z = the shard's bound pair codes), then
           sign(w) on the cells. alpha = None is the Hebbian limit (alpha -> inf). R refit rounds re-solve with targets
           y <- y + (mean(s) - s) mean(y) / mean(s), s = Z sign(w) (equalises the stored pairs' sign-domain scores).
           (alpha, R) = (None, 0) is U itself (built by b38_eq.vsa_build, so its rows equal U's).
  U-J      (B) U's cells read by a joint decoder over EVERY stored pair: pass 1 decodes every stored (key, j) slot by
           U's matched filter (the slot list gives each slot's LAST flag); each round t then forms W^ = the shard's sum of
           the decoded pair codes, calibrates a per-shard error scale sigma from the disagreement of sign(W^) with the
           read cells, and scores candidates by the log-likelihood of the read cells given the other slots' estimates
           (b_t = sign(R_t + x_t + N(0, sigma^2 + 1/4)), flipped with the condition's cell-error rate): a weighted
           matched filter with weights g_t = (log P(b_t | R_t + 1) - log P(b_t | R_t - 1)) / 2, R = W^ - own estimate.
           The probed keys are read after round T with U's V-last rule on these scores. T = 0 is U itself.
  H-SBC-J  (B) B61's sparse block code store read with explaining away over EVERY stored slot (B61's H-SIC weights and
           kernel; B61 explained away only the probe batch). T = 0 is H-SBC itself; T = 1 = H-SIC over all stored slots;
           T = 2 re-decodes every slot once with explaining away before the probe read.
  O2-BCH / O2-LDPC  (C) an insertable table with fewer bits per pair than B63's cuckoo table: the same bucketized cuckoo
           table (4 slots, two buckets, partial-key cuckoo hashing, load <= 0.95, 500 kicks, B63's hashes) with SHORT
           g-bit fingerprints made unambiguous by a conflict stash: a new key whose fingerprint already occurs in one of
           its two buckets (the only way two entries can share (bucket pair, fingerprint); moves never create one) goes
           to a stash of (ceil(log2 n) + 12)-bit key hashes with its row, provisioned for mu + 3 sqrt(mu) + 1 entries
           (mu = 4 x 0.95 x n / (2^g - 1), a design constant from n, as m). ss = 1: semi-sorted buckets (Fan et al. 2014:
           the four fingerprints' top 4 bits stored as one 12-bit index of the sorted 4-tuple; 1 bit per slot saved).
           Every slot, every stash entry and every index bit on the cells; protected by B38 / B63's BCH and LDPC
           machinery unchanged. Insertion is simulated logically, one pair at a time in store order (as B63).

Accounting for the joint decoders (B): 'free' = the decoder is given the stored slot list (keys and counts) at no cost;
'chg' = the list is charged on the same cells at its information-theoretic minimum for h-bit key fingerprints
(h = ceil(log2 n_keys) + 10): ceil(log2 C(2^h, n_keys)) + ceil(n_keys H(counts)) bits, on cells at the binary
symmetric channel's capacity bound ceil(L / (1 - h2(eps))) (B69's 'cap'; the list then read without error: a lower
bound on any code's cells, the charge most favourable to the store), deducted from the store's budget; a probed key
whose fingerprint equals another stored key's counts as failed (its codes would be shared).

python3 b75_strong.py selftest
python3 b75_strong.py devplan OUT.json
python3 b75_strong.py run STORES.json.gz PLAN.json OUT.json [--workers N] [--only MEM,..] [--conds k1,..]
python3 b75_strong.py choose DEV.json[,..] OUT.json
python3 b75_strong.py testplan CHOICE.json OUT.json
python3 b75_strong.py extend CHOICE.json PLAN.json[,..] TEST.json[,..] OUT.json STORES.json.gz
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
from scipy.special import log_ndtr  # noqa: E402

import b38_eq as E  # noqa: E402
import b61_hd as H  # noqa: E402
import b63_online as O  # noqa: E402
import b69_vocab as B69  # noqa: E402
import ruler_vsa as V  # noqa: E402

GRID = E.GRID
NG = len(GRID)
TARGET = E.TARGET
D = E.D
CONDS = ("k1", "k2", "k4", "k8", "k16", "k32")           # the six labelled conditions (e05 not run)
PRIMARY = ("k1", "k8", "k16")
NEXT = {"k1": "k2", "k2": "k4", "k4": "k8", "k8": "k16", "k16": "k32", "k32": None}
MEMS = ("U-LS", "U-J", "H-SBC-J", "O2-BCH", "O2-LDPC")
COND_OF = {"U-LS": CONDS, "U-J": PRIMARY, "H-SBC-J": PRIMARY, "O2-BCH": CONDS, "O2-LDPC": CONDS}
ACCTS = {"U-LS": ("free",), "U-J": ("free", "chg"), "H-SBC-J": ("free", "chg"), "O2-BCH": ("free",),
         "O2-LDPC": ("free",)}
# design lists in tie-break order (the incumbent first, so a new design must beat it strictly)
ULS_DESIGNS = [[None, 0], [None, 3], [1.0, 0], [1.0, 3], [0.1, 0], [0.1, 3], [0.0, 0], [0.0, 3]]
UJ_T = (0, 1, 2, 3)
SBCJ_T = (0, 1, 2)
SBCJ_LAMS = (0.5, 0.7, 1.0, 1.4)
O2_GS = (6, 8, 10)
O2_SS = (0, 1)
O2_RATES = {"k1": (0.75, 5 / 6, 0.9), "k2": (0.75, 5 / 6, 0.9), "k4": (0.5, 2 / 3, 0.75, 0.9),
            "k8": (0.25, 1 / 3, 0.5), "k16": (0.25, 1 / 3, 0.5), "k32": (0.25, 1 / 3)}
KEY_H_EXTRA = 10
ALPHA = O.ALPHA
BUCKET = O.BUCKET
# the paper's development N* (window centres): B38 dev (U), B61 dev (H-SBC), B63 dev (O-BCH, O-LDPC)
DEV_CENTRE = {"U": {"k1": 76, "k2": 76, "k4": 76, "k8": 128, "k16": 431, "k32": 1722},
              "H-SBC": {"k1": 32, "k2": 32, "k4": 45, "k8": 91, "k16": 362, "k32": 1448},
              "O-BCH": {"k1": 32, "k2": 32, "k4": 45, "k8": 181, "k16": 724, "k32": 2435},
              "O-LDPC": {"k1": 32, "k2": 32, "k4": 45, "k8": 91, "k16": 304, "k32": 1218}}
DEV_WINDOW = {"U-LS": ("U", -6, 3), "U-J": ("U", -10, 1), "H-SBC-J": ("H-SBC", -8, 1), "O2-BCH": ("O-BCH", -5, 3),
              "O2-LDPC": ("O-LDPC", -5, 3)}


def gidx(B):
    return min(range(NG), key=lambda i: abs(GRID[i] - B))


def dj(x):
    return json.dumps(x)


# =============================================================================================== key list charge (B)
def keylist(S38):
    """The stored slot list at its information-theoretic minimum (h-bit key fingerprints + counts) and the probed keys
    whose fingerprint collides with another stored key's."""
    s = S38.s
    keys = list(s.count)
    nk = len(keys)
    h = (math.ceil(math.log2(nk)) if nk > 1 else 0) + KEY_H_EXTRA
    N = 2 ** h
    lset = math.ceil((math.lgamma(N + 1) - math.lgamma(nk + 1) - math.lgamma(N - nk + 1)) / math.log(2))
    cc = collections.Counter(s.count[k] for k in keys)
    Hc = -sum(c / nk * math.log2(c / nk) for c in cc.values()) if len(cc) > 1 else 0.0
    lcnt = math.ceil(nk * Hc - 1e-9)
    fp = {k: int.from_bytes(hashlib.sha256(("b75key:" + k).encode("utf-8")).digest()[:8], "little") % N
          for k in keys}
    mult = collections.Counter(fp.values())
    coll = [a for a, k in enumerate(s.keys) if mult[fp[k]] > 1]
    return dict(h=h, nk=nk, L=lset + lcnt, lset=lset, lcnt=lcnt, coll=coll)


def key_cells(L, eps):
    """Cells for the key list: the binary symmetric channel's capacity bound L / (1 - h2(eps)) (B69's 'cap'), with the
    list then read without error -- a lower bound on any code's cells, i.e. the charge most favourable to the store."""
    return B69.vocab_cost(L, eps, "cap")


def key_cells_bch(L, eps):
    """For reporting only: B69's repetition / BCH rule (P(any key-list bit wrong) <= 1%)."""
    return B69.vocab_cost(L, eps, "bch")


def charged_total(S38, B, cond, acct):
    """(cells left for the store, key-list cells, collided probed keys)."""
    budget = int(B * S38.n)
    if acct == "free":
        return budget, 0, [], None
    kl = keylist(S38)
    kc = key_cells(kl["L"], E.parse_cond(cond)["eps"])
    return budget - kc, kc, kl["coll"], kl


# =============================================================================================== (A) U-LS
def uls_build(S38, B, alpha, R):
    """U's geometry and codes with the least-squares write. Returns a vsa_build-like dict or None."""
    if alpha is None and R == 0:
        return E.vsa_build(S38, B, "last")
    n = S38.n
    total = int(B * n)
    S = max(1, math.ceil(total / D))
    d = min(D, total // S)
    if d < 16:
        return None
    shard_k = np.array([V._seed("shard:" + k) % S for k in S38.s.cd["allk"]])
    ps = shard_k[S38.pk]
    prod = (S38.Kc[S38.pk, :d] * S38.Pc[S38.pj, :d] * S38.Vc[S38.pv, :d]).astype(np.int8)
    prod = np.where(S38.plast[:, None] > 0, prod * S38.L[None, :d], prod).astype(np.int8)
    cells = np.ones((S, d), np.int8)
    order = np.argsort(ps, kind="stable")
    sp_ = ps[order]
    starts = np.flatnonzero(np.r_[True, sp_[1:] != sp_[:-1]])
    ends = np.r_[starts[1:], len(sp_)]
    for a, b in zip(starts, ends):
        idx = order[a:b]
        Z = prod[idx].astype(np.float64)
        m = len(idx)
        y = np.ones(m)
        if alpha is not None:
            G = Z @ Z.T
            G[np.diag_indices(m)] += alpha * d
            try:
                Lc = np.linalg.cholesky(G)
                solve = lambda t, Lc=Lc: np.linalg.solve(Lc.T, np.linalg.solve(Lc, t))  # noqa: E731
            except np.linalg.LinAlgError:
                Gp = np.linalg.pinv(G)
                solve = lambda t, Gp=Gp: Gp @ t  # noqa: E731
            w = Z.T @ solve(y)
        else:
            w = Z.T @ y
        for _ in range(R):
            bsg = np.where(w >= 0, 1.0, -1.0)
            s_ = Z @ bsg
            sb = s_.mean()
            if sb <= 0:
                break
            y = y + (sb - s_) * (y.mean() / sb)
            w = Z.T @ (solve(y) if alpha is not None else y)
        cells[sp_[a]] = np.where(w >= 0, 1, -1).astype(np.int8)
    return dict(cells=cells.ravel(), S=S, d=d, shard_k=shard_k, rms0=np.ones(S, np.float32), kind="last", load=None)


def uls_eval(S38, alpha, R, B, conds):
    bd = uls_build(S38, B, alpha, R)
    des = [alpha, R]
    if bd is None:
        return [dict(mem="U-LS", acct="free", design=des, B=B, cond=c, infeasible=True) for c in conds]
    tag = "V-last|null"                                     # U's channel draw (common random numbers with U's rows)
    ch = E.Channel(S38.sid, tag, B, len(bd["cells"]), int(B * S38.n))
    out = []
    for c in conds:
        sig = ch.apply(bd["cells"], c)
        rd = E.vsa_read(S38, bd, sig)
        got = E.vsa_rule(rd, "last")
        fl = E.exact(got, S38.want)
        out.append(dict(mem="U-LS", acct="free", design=des, B=B, cond=c, infeasible=False, exact=len(S38.want) - len(fl),
                        n=len(S38.want), failed=fl, cells=int(len(bd["cells"])), budget=int(B * S38.n), S=bd["S"],
                        d=bd["d"]))
    return out


# =============================================================================================== (B) U-J
def _sigma_shard(b, What, eps):
    """Error scale sigma of W = W^ + N(0, sigma^2): matches the observed disagreement of sign(W^) with the read cells."""
    aw = np.abs(What).astype(np.int64)
    bad = (np.sign(What) != b) & (What != 0)
    dobs = (bad.sum() + 0.5 * (What == 0).sum()) / len(b)
    vals, cnt = np.unique(aw, return_counts=True)
    nz = vals > 0
    v, cn = vals[nz].astype(np.float64), cnt[nz].astype(np.float64)
    z0 = cnt[~nz].sum() if (~nz).any() else 0

    def model(sig):
        s = math.sqrt(sig * sig + 0.25)
        p = np.exp(log_ndtr(-v / s))
        return ((cn * ((1 - eps) * p + eps * (1 - p))).sum() + 0.5 * z0) / len(b)
    lo, hi = 0.0, 3.0 * math.sqrt(max(1.0, float(aw.max()))) + 3.0
    if dobs <= model(lo):
        return lo
    if dobs >= model(hi):
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if model(mid) < dobs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gtable(sig, eps, Rmax):
    """g[b_idx, R + Rmax] = (log P(b | R + 1) - log P(b | R - 1)) / 2, b_idx 0 -> b = -1, 1 -> b = +1."""
    s = math.sqrt(sig * sig + 0.25)
    e = min(max(eps, 1e-9), 0.5 - 1e-9)
    r = np.arange(-Rmax, Rmax + 1, dtype=np.float64)
    out = np.empty((2, len(r)), np.float32)
    for bi, bv in ((0, -1.0), (1, 1.0)):
        def lp(w):
            return np.logaddexp(math.log(1 - e) + log_ndtr(bv * w / s), math.log(e) + log_ndtr(-bv * w / s))
        out[bi] = 0.5 * (lp(r + 1) - lp(r - 1))
    return out


def _decode_slots(Wfn, Us, lastm, Lf, Vn, chunk=384):
    """argmax over values of (weights * unbinding (* LAST)) @ Vn^T per stored slot, LAST flag known per slot."""
    n = Us.shape[0]
    out = np.empty(n, np.int64)
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        Y = Wfn(sl) * Us[sl].astype(np.float32)
        lm = lastm[sl]
        if lm.any():
            Y[lm] *= Lf[None, :]
        out[sl] = (Y @ Vn.T).argmax(1)
    return out


def uj_decode(bmat, shard_k, pk, pj, plast, Kc, Pc, Vc, L, qk, J, eps, Ts, Vn):
    """The joint decoder. Inputs are the read cells and the slot list (keys, positions, LAST flags) and the probed key
    indices; the stored values are NOT an input. Returns {T: list of decoded value lists per probed key}."""
    S, d = bmat.shape
    n = len(pk)
    ps = shard_k[pk]
    Us = (Kc[pk, :d] * Pc[pj, :d]).astype(np.int8)
    lastm = plast > 0
    Lf = L[:d].astype(np.float32)
    bf = bmat.astype(np.float32)
    bi_all = (bmat > 0).astype(np.int64)
    slot_of = {(int(k), int(j)): i for i, (k, j) in enumerate(zip(pk, pj))}
    qsh = shard_k[qk]
    order = np.argsort(ps, kind="stable")
    sp_ = ps[order]
    starts = np.flatnonzero(np.r_[True, sp_[1:] != sp_[:-1]])
    vhat = _decode_slots(lambda sl: bf[ps[sl]], Us, lastm, Lf, Vn)        # pass 1: U's matched filter per slot
    res = {}
    Tmax = max(Ts)
    for t in range(1, Tmax + 1):
        Zh = (Us * Vc[vhat, :d]).astype(np.int8)
        Zh[lastm] *= L[None, :d]
        W = np.zeros((S, d), np.int32)
        W[sp_[starts]] = np.add.reduceat(Zh[order].astype(np.int32), starts, axis=0)
        Rmax = int(np.abs(W).max()) + 2
        G = np.empty((S, 2, 2 * Rmax + 1), np.float32)
        for sh in range(S):
            G[sh] = _gtable(_sigma_shard(bmat[sh], W[sh], eps), eps, Rmax)
        if t in Ts:                                                        # probe read after round t (V-last rule)
            nq = len(qk)
            best = np.zeros((nq, J), np.int64)
            tag = np.zeros((nq, J), bool)
            for j in range(J):
                own = np.zeros((nq, d), np.int32)
                for a in range(nq):
                    i = slot_of.get((int(qk[a]), j))
                    if i is not None:
                        own[a] = Zh[i]
                Rh = W[qsh] - own + Rmax
                Wt = G[qsh[:, None], bi_all[qsh], Rh]
                Y = Wt * (Kc[qk, :d] * Pc[j, :d]).astype(np.float32)
                s0 = Y @ Vn.T
                s1 = (Y * Lf[None, :]) @ Vn.T
                m0, m1 = s0.max(1), s1.max(1)
                useL = m1 > m0
                best[:, j] = np.where(useL, s1.argmax(1), s0.argmax(1))
                tag[:, j] = useL
            got = []
            for a in range(nq):
                g = []
                for j in range(J):
                    g.append(int(best[a, j]))
                    if tag[a, j]:
                        break
                got.append(g)
            res[t] = got
        if t < Tmax:
            def wfn(sl, W=W, G=G, Rmax=Rmax, Zh=Zh):
                Rh = W[ps[sl]] - Zh[sl].astype(np.int32) + Rmax
                return G[ps[sl][:, None], bi_all[ps[sl]], Rh]
            vhat = _decode_slots(wfn, Us, lastm, Lf, Vn)
    return res


def uj_eval(S38, B, cond, acct, Ts=UJ_T):
    nq = len(S38.want)
    tot, kc, coll, kl = charged_total(S38, B, cond, acct)
    base = dict(mem="U-J", acct=acct, B=B, cond=cond)
    if tot <= 0:
        return [dict(base, design=[t], infeasible=True) for t in Ts]
    Bs = B if acct == "free" else (tot + 0.5) / S38.n
    bd = E.vsa_build(S38, Bs, "last")
    if bd is None:
        return [dict(base, design=[t], infeasible=True) for t in Ts]
    ch = E.Channel(S38.sid, "V-last|null", Bs, len(bd["cells"]), tot)
    sig = ch.apply(bd["cells"], cond)
    out = []
    got0 = E.vsa_rule(E.vsa_read(S38, bd, sig), "last")                  # T = 0: U's own read
    gots = {0: got0}
    Tj = [t for t in Ts if t > 0]
    if Tj:
        eps = E.parse_cond(cond)["eps"]
        S, d = bd["S"], bd["d"]
        gots.update(uj_decode(sig[:S * d].reshape(S, d), bd["shard_k"], S38.pk, S38.pj, S38.plast, S38.Kc, S38.Pc,
                              S38.Vc, S38.L, S38.qk, S38.J, eps, Tj, S38.vf(d)))
    for t in Ts:
        fl = sorted(set(E.exact(gots[t], S38.want)) | set(coll))
        out.append(dict(base, design=[t], infeasible=False, exact=nq - len(fl), n=nq, failed=fl,
                        cells=int(len(bd["cells"])) + kc, budget=int(B * S38.n), key_cells=kc, S=bd["S"], d=bd["d"],
                        key_bits=None if kl is None else kl["L"],
                        key_cells_bch=None if kl is None else key_cells_bch(kl["L"], E.parse_cond(cond)["eps"])))
    return out


# =============================================================================================== (B) H-SBC-J
@njit(cache=True)
def _all_slots(yb, bases, Om, tf, cov, own, Vb, Lc, Lb, Ash, Csh, shard, hascov):
    """Decision for every stored slot with its LAST variant known (tf): B61's slot kernel restricted to one variant,
    weights A (unexplained active) / -C (unexplained inactive) per shard, explained cells (cov minus the slot's own
    item) carry no evidence; exact pruning as B61's kernel; argmax = first index."""
    n, nb = Om.shape
    Nv = Vb.shape[0]
    out = np.empty(n, np.int64)
    for i in range(n):
        base = bases[i]
        A = Ash[shard[i]]
        Cc = Csh[shard[i]]
        Ap = A if A > 0 else 0.0
        best = -1e300
        bi = -1
        for v in range(Nv):
            na = 0
            ni = 0
            pruned = False
            for b in range(nb):
                p = Om[i, b] + Vb[v, b]
                if p >= Lb:
                    p -= Lb
                if tf[i]:
                    p += Lc[b]
                    if p >= Lb:
                        p -= Lb
                cell = base + b * Lb + p
                if hascov:
                    cv = cov[cell]
                    if own[i, b] == cell:
                        cv -= 1
                    if cv > 0:
                        continue
                if yb[cell]:
                    na += 1
                else:
                    ni += 1
                    if (na + nb - b - 1) * Ap - ni * Cc < best:
                        pruned = True
                        break
            if pruned:
                continue
            a = na * A - ni * Cc
            if a > best:
                best = a
                bi = v
        out[i] = bi
    return out


def _sbc_items(st, Om, vhat, tf, bases):
    p = (Om + st.Vb[vhat] + np.where(tf[:, None], st.Lc[None, :], 0)) % st.Lb
    return bases[:, None] + np.arange(st.nb, dtype=np.int64)[None, :] * st.Lb + p


def _sbc_ac(st, yb, cov, eps):
    e = max(eps, H.EPS_FLOOR)
    A = np.empty(st.S)
    C = np.empty(st.S)
    for sh in range(st.S):
        seg = slice(sh * st.d, sh * st.d + st.nb * st.Lb)
        un = cov[seg] == 0
        q = float(yb[seg][un].mean()) if un.any() else 0.5
        q = min(max(q, 1e-3), 0.999)
        A[sh], C[sh] = math.log((1 - e) / q), math.log((1 - q) / e)
    return A, C


def sbcj_decode(st, yb, pk, pj, plast, qk, J, eps, Ts):
    """Explaining away over every stored slot. Inputs: the read cells, the slot list and the probed keys (no stored
    value). Returns {T: decoded value lists per probed key} for T >= 1."""
    tf = (plast > 0)
    shard = st.shard_k[pk].astype(np.int64)
    bases = shard * st.d
    Om = ((st.Kb[pk] + st.Pb[pj]) % st.Lb).astype(np.int32)
    one, zero = np.ones(st.S), np.zeros(st.S)
    nocov = np.zeros(1, np.int32)
    noown = np.full((1, 1), -1, np.int64)
    vhat = _all_slots(yb, bases, Om, tf, nocov, noown, st.Vb, st.Lc, st.Lb, one, zero, shard, False)
    slot_of = {(int(k), int(j)): i for i, (k, j) in enumerate(zip(pk, pj))}
    res = {}
    for t in range(1, max(Ts) + 1):
        items = _sbc_items(st, Om, vhat, tf, bases)
        cov = np.bincount(items.ravel(), minlength=len(yb)).astype(np.int32)
        A, C = _sbc_ac(st, yb, cov, eps)
        if t in Ts:
            owns = {}
            for a in range(len(qk)):
                for j in range(J):
                    i = slot_of.get((int(qk[a]), j))
                    if i is not None:
                        owns[(a, j)] = items[i]
            AC = {sh: (A[sh], C[sh]) for sh in range(st.S)}
            res[t], _ = _sbc_probe(st, yb, qk, J, cov, owns, AC)
        if t < max(Ts):
            vhat = _all_slots(yb, bases, Om, tf, cov, items, st.Vb, st.Lc, st.Lb, A, C, shard, True)
    return res


def _sbc_probe(st, yb, qk, J, cov, owns, AC):
    """B61's sbc_read (V-last rule, explaining-away pass) with the probed keys given by key index."""
    got = []
    none = np.full(st.nb, -1, np.int64)
    for a, k in enumerate(qk):
        sh = int(st.shard_k[k])
        base = sh * st.d
        g = []
        for j in range(J):
            o = ((st.Kb[k] + st.Pb[j]) % st.Lb).astype(np.int32)
            A, Cc = AC[sh]
            v, last = H._slot(yb, base, cov, owns.get((a, j), none), o, st.Vb, st.Lc, st.Lb, A, Cc)
            g.append(int(v))
            if last:
                break
        got.append(g)
    return got, None


def sbcj_eval(S38, lam, B, cond, acct, Ts=SBCJ_T):
    nq = len(S38.want)
    tot, kc, coll, kl = charged_total(S38, B, cond, acct)
    base = dict(mem="H-SBC-J", acct=acct, B=B, cond=cond)
    if tot <= 0:
        return [dict(base, design=[lam, t], infeasible=True) for t in Ts]
    Bs = B if acct == "free" else (tot + 0.5) / S38.n
    st = H.SBCStore(S38, Bs, lam)
    if not st.ok:
        return [dict(base, design=[lam, t], infeasible=True) for t in Ts]
    ch = E.Channel(S38.sid, "H-SBC|%.4f" % lam, Bs, len(st.cells), st.total)
    sig = ch.apply(st.cells, cond)
    yb = (sig > 0).astype(np.uint8)
    keys = list(range(nq))
    g0, _ = H.sbc_read(S38, st, yb, keys)                                  # T = 0: H-SBC's own read
    gots = {0: [g0[a] for a in keys]}
    Tj = [t for t in Ts if t > 0]
    if Tj:
        gots.update(sbcj_decode(st, yb, S38.pk, S38.pj, S38.plast, S38.qk, S38.J, E.parse_cond(cond)["eps"], Tj))
    out = []
    for t in Ts:
        fl = sorted(set(E.exact(gots[t], S38.want)) | set(coll))
        out.append(dict(base, design=[lam, t], infeasible=False, exact=nq - len(fl), n=nq, failed=fl,
                        cells=int(len(st.cells)) + kc, budget=int(B * S38.n), key_cells=kc, S=st.S, d=st.d, Lb=st.Lb,
                        nb=st.nb, key_bits=None if kl is None else kl["L"]))
    return out


# =============================================================================================== (C) cuckoo + conflict stash
_NIB4 = [t for t in itertools.combinations_with_replacement(range(16), 4)]      # 3,876 sorted 4-tuples
_NIB4_IX = {t: i for i, t in enumerate(_NIB4)}


def stash_hash(k, j, hb):
    h = hashlib.sha256(("b75stash:%s\x00%d" % (k, j)).encode("utf-8")).digest()
    return 1 + int.from_bytes(h[:8], "little") % ((1 << hb) - 1)


class CuckooStash:
    """The insertable table of (C): m buckets x 4 slots x [g-bit fp | last | value] (semi-sorted if ss) + a stash."""

    def __init__(self, S38, g, ss, upto=None, state=None):
        s = S38.s
        self.g, self.ss, self.n = g, ss, s.n
        self.f = g
        self.m = m = max(1, math.ceil(s.n / (BUCKET * ALPHA)))
        self.P = s.lastbit + s.b
        mu = 4 * ALPHA * s.n / ((1 << g) - 1)
        self.cap = int(math.ceil(mu + 3 * math.sqrt(mu))) + 1
        self.hb = (math.ceil(math.log2(s.n)) if s.n > 1 else 0) + 12
        if state is None:
            self.fps = np.zeros((m, BUCKET), np.int64)
            self.rows = np.full((m, BUCKET), -1, np.int64)
            self.sh = np.zeros(self.cap, np.int64)
            self.srow = np.full(self.cap, -1, np.int64)
            self.nst = 0
            self.rng = np.random.default_rng(E.seed_of(S38.sid, "b75cs", g))
            self.lost = []
            start = 0
        else:
            self.fps, self.rows, self.sh, self.srow, self.nst, self.rng, self.lost, start = state
        stop = s.n if upto is None else upto
        for i in range(start, stop):
            k, j = s.rk[i], int(s.rj[i])
            i1, i2, fp = O.buckets_of(k, j, m, g)
            if (self.fps[i1] == fp).any() or (self.fps[i2] == fp).any():
                if self.nst < self.cap:
                    self.sh[self.nst], self.srow[self.nst] = stash_hash(k, j, self.hb), i
                    self.nst += 1
                else:
                    self.lost.append(i)
                continue
            r = O.cuckoo_insert(self.fps, self.rows, k, j, i, m, g, self.rng)
            if r is not None:
                self.lost.append(r)
        self.done = stop
        self.empty = int((self.fps == 0).sum())
        self._bits = None
        self._lay = None

    def layout(self):
        """Per bucket the slot order on the cells (semi-sorted: by fp, i.e. by the top nibble then the rest)."""
        if self._lay is None:
            if self.ss:
                self._lay = np.argsort(self.fps, axis=1, kind="stable")
            else:
                self._lay = np.tile(np.arange(BUCKET), (self.m, 1))
        return self._lay

    def bucket_width(self):
        return (12 + BUCKET * (self.g - 4) if self.ss else BUCKET * self.g) + BUCKET * self.P

    def bits(self, S38):
        if self._bits is None:
            s = S38.s
            g, P, m = self.g, self.P, self.m
            pay = (s.payload_bits() > 0).astype(np.uint8)
            lay = self.layout()
            fps = np.take_along_axis(self.fps, lay, 1)
            rows = np.take_along_axis(self.rows, lay, 1)
            pb = np.zeros((m, BUCKET, P), np.uint8)
            occ = rows >= 0
            pb[occ] = pay[rows[occ]]
            if self.ss:
                top = fps >> (g - 4)
                ix = np.array([_NIB4_IX[tuple(int(x) for x in r)] for r in top], np.int64)
                ib = ((ix[:, None] >> np.arange(11, -1, -1)[None, :]) & 1).astype(np.uint8)
                low = fps & ((1 << (g - 4)) - 1)
                lb = ((low[:, :, None] >> np.arange(g - 5, -1, -1)[None, None, :]) & 1).astype(np.uint8)
                tb = np.concatenate([ib, lb.reshape(m, -1), pb.reshape(m, -1)], 1)
            else:
                fb = ((fps[:, :, None] >> np.arange(g - 1, -1, -1)[None, None, :]) & 1).astype(np.uint8)
                tb = np.concatenate([fb, pb], 2).reshape(m, -1)
            spb = np.zeros((self.cap, P), np.uint8)
            so = self.srow >= 0
            spb[so] = pay[self.srow[so]]
            shb = ((self.sh[:, None] >> np.arange(self.hb - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
            self._bits = np.concatenate([tb.ravel(), np.concatenate([shb, spb], 1).ravel()])
        return self._bits

    def parse(self, S38, bits):
        """(fp per (bucket, cell slot), last, value per (bucket, cell slot), stash hashes, stash last, stash value)."""
        s = S38.s
        g, P, m, w = self.g, self.P, self.m, self.bucket_width()
        arr = bits[:m * w].reshape(m, w).astype(np.int64)
        if self.ss:
            ix = (arr[:, :12] * (1 << np.arange(11, -1, -1))[None, :]).sum(1) % len(_NIB4)
            top = np.array(_NIB4, np.int64)[ix]
            lw = arr[:, 12:12 + BUCKET * (g - 4)].reshape(m, BUCKET, g - 4)
            low = (lw * (1 << np.arange(g - 5, -1, -1))[None, None, :]).sum(2)
            fpr = (top << (g - 4)) | low
            pay = arr[:, 12 + BUCKET * (g - 4):].reshape(m * BUCKET, P)
        else:
            a3 = arr.reshape(m, BUCKET, g + P)
            fpr = (a3[:, :, :g] * (1 << np.arange(g - 1, -1, -1))[None, None, :]).sum(2)
            pay = a3[:, :, g:].reshape(m * BUCKET, P)
        last, val = s.decode_payload(pay.astype(np.int8) * 2 - 1)
        st = bits[m * w:m * w + self.cap * (self.hb + P)].reshape(self.cap, self.hb + P).astype(np.int64)
        shr = (st[:, :self.hb] * (1 << np.arange(self.hb - 1, -1, -1))[None, :]).sum(1)
        sl, sv = s.decode_payload(st[:, self.hb:].astype(np.int8) * 2 - 1)
        return fpr, last.reshape(m, BUCKET), val.reshape(m, BUCKET), shr, sl, sv

    def where(self):
        """row -> ('t', bucket, cell slot) or ('s', stash index)."""
        out = {}
        lay = self.layout()
        rows = np.take_along_axis(self.rows, lay, 1)
        for b, e in zip(*np.nonzero(rows >= 0)):
            out[int(rows[b, e])] = ("t", int(b), int(e))
        for i in range(self.nst):
            out[int(self.srow[i])] = ("s", i)
        return out

    def needs(self, S38):
        """Logical bit positions each stored key's read needs (every stored key; for the BCH closed form): in the table,
        the fingerprint region of both candidate buckets at each j and its own payload; in the stash, its own entry."""
        s = S38.s
        wh = self.where()
        g, P, m, w = self.g, self.P, self.m, self.bucket_width()
        fpw = (12 + BUCKET * (g - 4)) if self.ss else None
        sbase = m * w
        out = []
        for k in s.count:
            pos = []
            for j in range(s.count[k]):
                i = s.row_of[(k, j)]
                loc = wh.get(i)
                if loc is not None and loc[0] == "s":
                    e = loc[1]
                    pos.append(np.arange(sbase + e * (self.hb + P), sbase + (e + 1) * (self.hb + P)))
                    continue
                i1, i2, _ = O.buckets_of(k, j, m, g)
                for b in {i1, i2}:
                    if self.ss:
                        pos.append(np.arange(b * w, b * w + fpw))
                    else:
                        for e in range(BUCKET):
                            pos.append(np.arange(b * w + e * (g + P), b * w + e * (g + P) + g))
                if loc is not None:
                    _, b, e = loc
                    if self.ss:
                        o = b * w + fpw + e * P
                    else:
                        o = b * w + e * (g + P) + g
                    pos.append(np.arange(o, o + P))
            out.append(np.unique(np.concatenate(pos)) if pos else np.zeros(0, np.int64))
        return out

    def read(self, S38, bits, diag=None):
        """Stash first (exact hash match: one -> its row, two or more -> fail); else B63's table rule (one exact
        fingerprint match -> read; two or more -> fail; none -> the unique nearest slot in Hamming distance, a tie
        fails)."""
        s = S38.s
        m, g = self.m, self.g
        fpr, last, val, shr, sl, sv = self.parse(S38, bits)
        dg = diag if diag is not None else {}
        for kk in ("ambiguous", "nearest", "nearest_tie", "stash_hit", "stash_ambiguous"):
            dg.setdefault(kk, 0)
        sidx = {}
        for e, hv in enumerate(shr.tolist()):
            sidx.setdefault(hv, []).append(e)
        got = []
        for k in s.keys:
            gl = []
            for j in range(s.maxj + 1):
                hs = stash_hash(k, j, self.hb)
                se = sidx.get(hs, [])
                if len(se) > 1:
                    dg["stash_ambiguous"] += 1
                    gl.append(-1)
                    break
                if len(se) == 1:
                    dg["stash_hit"] += 1
                    gl.append(int(sv[se[0]]))
                    if sl[se[0]]:
                        break
                    continue
                i1, i2, fp = O.buckets_of(k, j, m, g)
                cand = [(i1, e) for e in range(BUCKET)] + ([(i2, e) for e in range(BUCKET)] if i2 != i1 else [])
                mt = [c for c in cand if fpr[c] == fp]
                if len(mt) > 1:
                    dg["ambiguous"] += 1
                    gl.append(-1)
                    break
                if len(mt) == 1:
                    c = mt[0]
                else:
                    dist = [bin(int(fpr[c]) ^ fp).count("1") for c in cand]
                    mn = min(dist)
                    wn = [c for c, x in zip(cand, dist) if x == mn]
                    if len(wn) > 1:
                        dg["nearest_tie"] += 1
                        gl.append(-1)
                        break
                    dg["nearest"] += 1
                    c = wn[0]
                gl.append(int(val[c]))
                if last[c]:
                    break
            got.append(gl)
        return got


def o2_evaluate(S38, ck, arm, R, B, conds, needs=None):
    """B63's evaluate (b63_online.evaluate) with this table and its own channel tags; code machinery unchanged."""
    total = int(B * S38.n)
    bits = ck.bits(S38)
    L = len(bits)
    des = [ck.g, ck.ss] + ([R] if arm == "O2-LDPC" else [])
    base = dict(mem=arm, acct="free", design=des, B=B, L=int(L), m=ck.m, lost=len(ck.lost), empty=ck.empty,
                nst=ck.nst, cap=ck.cap, budget=total)
    out = []
    if arm == "O2-BCH":
        if needs is None:
            needs = ck.needs(S38)
        groups = collections.OrderedDict()
        for c in conds:
            nh = NEXT.get(c)
            des_ = E.bch_choose(L, needs, total, E.IID[c], E.IID[nh] if nh else None)
            groups.setdefault(None if des_ is None else tuple(des_), []).append(c)
        for d_, cs in groups.items():
            if d_ is None:
                for c in cs:
                    out.append(dict(base, cond=c, infeasible=True))
                continue
            n_, k_, t_ = d_
            lay = E.ecc_layout(L, n_, k_, t_, total)
            coded = E.bch_encode_stream(bits, lay)
            cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
            tag = "O2-BCH|%s" % json.dumps([ck.g, ck.ss, n_, k_, t_])
            ch = E.Channel(S38.sid, tag, B, len(cells), total)
            for c in cs:
                sig = ch.apply(cells, c)
                rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, c, "dec"))
                dec, st = E.bch_decode_stream(coded, E.rep_sum(sig, lay["counts"]), lay, rng)
                dg = dict(cw_bound=st[0], cw_lib=st[1], cw_fail=st[2])
                got = ck.read(S38, dec, dg)
                fl = E.exact(got, S38.want)
                out.append(dict(base, cond=c, infeasible=False, bch=[n_, k_, t_], cells=int(len(cells)),
                                exact=len(S38.want) - len(fl), n=len(S38.want), failed=fl, diag=dg))
        return out
    lay = E.ldpc_layout(L, R, total)
    if lay is None:
        return [dict(base, cond=c, infeasible=True) for c in conds]
    coded, keep = E.ldpc_encode_stream(bits, lay)
    cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
    tag = "O2-LDPC|%s" % json.dumps([ck.g, ck.ss, R])
    ch = E.Channel(S38.sid, tag, B, len(cells), total)
    for c in conds:
        sig = ch.apply(cells, c)
        rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, c, "dec"))
        dec, nf = E.ldpc_decode_stream(E.rep_sum(sig, lay["counts"]), lay, keep, E.IID[c], rng)
        dg = dict(cw_nonconv=nf)
        got = ck.read(S38, dec[:L], dg)
        fl = E.exact(got, S38.want)
        out.append(dict(base, cond=c, infeasible=False, cells=int(len(cells)), exact=len(S38.want) - len(fl),
                        n=len(S38.want), failed=fl, diag=dg))
    return out


def mphf_bits_per_pair(S38):
    return len(E.mphf_content(S38)["bits"]) / S38.n


# =============================================================================================== jobs
def run_job(job):
    """job: dict(sid, pairs, mem, par, evals=[(B, R_or_None, [conds])], accts)."""
    t0 = time.time()
    mem, par = job["mem"], job["par"]
    S38 = E.Store38(job["sid"], job["pairs"])
    rows, meta = [], {}
    if mem == "U-LS":
        for B, _, cs in job["evals"]:
            rows += uls_eval(S38, par[0], par[1], B, cs)
    elif mem == "U-J":
        for acct in job["accts"]:
            for B, _, cs in job["evals"]:
                for c in cs:
                    rows += uj_eval(S38, B, c, acct)
    elif mem == "H-SBC-J":
        for acct in job["accts"]:
            for B, _, cs in job["evals"]:
                for c in cs:
                    rows += sbcj_eval(S38, par, B, c, acct)
    else:
        ck = CuckooStash(S38, par[0], par[1])
        meta = dict(m=ck.m, lost=len(ck.lost), nst=ck.nst, cap=ck.cap, L=int(len(ck.bits(S38))),
                    mphf_bits=len(E.mphf_content(S38)["bits"]))
        needs = ck.needs(S38) if mem == "O2-BCH" else None
        for B, R, cs in job["evals"]:
            rows += o2_evaluate(S38, ck, mem, R, B, cs, needs)
    return dict(sid=job["sid"], mem=mem, par=par, rows=rows, n_pairs=S38.n, n_values=S38.Nv, nq=len(S38.want),
                sec=time.time() - t0, **meta)


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done, last, t0 = [], time.time(), time.time()
    with Pool(workers, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            done.append(r)
            print("done %-28s %-8s %-14s %6.0fs %d/%d (%.0f s)" % (r["sid"], r["mem"], dj(r["par"]), r["sec"],
                                                                    len(done), len(jobs), time.time() - t0), flush=True)
            if time.time() - last > 120:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def par_of(mem, design):
    """The build parameter of a design (one job evaluates every nested design of its parameter)."""
    if mem == "U-LS":
        return list(design)
    if mem == "U-J":
        return None
    if mem == "H-SBC-J":
        return design[0]
    return [design[0], design[1]]


def rate_of(mem, design):
    return design[2] if mem == "O2-LDPC" else None


UNIT = {"U-LS": 4.0, "U-J": 60.0, "H-SBC-J": 20.0, "O2-BCH": 3.0, "O2-LDPC": 3.0}


def make_jobs(stores, need, only=None, conds=None, per_job=None):
    """need: {(mem, par_json, R_json, gi): set(conds)}."""
    grp = collections.defaultdict(list)
    for (mem, pj_, rj, gi), cs in need.items():
        if only and mem not in only:
            continue
        cs2 = sorted(c for c in cs if conds is None or c in conds)
        if cs2:
            grp[(mem, pj_)].append((GRID[gi], json.loads(rj), cs2))
    jobs = []
    for sid, pairs in stores:
        n = len(pairs)
        for (mem, pj_), ev in sorted(grp.items()):
            ev.sort(key=lambda e: (-1 if e[1] is None else e[1], e[0]))
            pj = per_job or (1 if mem in ("U-J", "H-SBC-J") and n > 3000 else (3 if mem in ("U-J", "H-SBC-J") else 6))
            for i in range(0, len(ev), pj):
                part = ev[i:i + pj]
                cost = UNIT[mem] * (n / 6000) ** (2 if mem == "U-J" else 1) * sum(len(e[2]) for e in part) + 0.01
                jobs.append(dict(sid=sid, pairs=pairs, mem=mem, par=json.loads(pj_), evals=part,
                                 accts=list(ACCTS[mem]), cost=cost))
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def designs(mem, cond=None):
    if mem == "U-LS":
        return [list(d) for d in ULS_DESIGNS]
    if mem == "U-J":
        return [[t] for t in UJ_T]
    if mem == "H-SBC-J":
        return [[lam, t] for t in SBCJ_T for lam in SBCJ_LAMS]
    if mem == "O2-BCH":
        return [[g, ss] for g in O2_GS for ss in O2_SS]
    rates = O2_RATES[cond] if cond else sorted({r for v in O2_RATES.values() for r in v})
    return [[g, ss, R] for g in O2_GS for ss in O2_SS for R in rates]


def window(mem, c):
    src, lo, hi = DEV_WINDOW[mem]
    g0 = gidx(DEV_CENTRE[src][c])
    return list(range(max(0, g0 + lo), min(NG, g0 + hi + 1)))


def devplan(out_path):
    need = collections.defaultdict(set)
    for mem in MEMS:
        for c in COND_OF[mem]:
            for gi in window(mem, c):
                for d in designs(mem, c):
                    need[(mem, dj(par_of(mem, d)), dj(rate_of(mem, d)), gi)].add(c)
    save_need(need, out_path)
    return need


def save_need(need, path):
    json.dump([[list(k), sorted(v)] for k, v in need.items()], open(path, "w"))
    print(len(need), "(memory, parameter, rate, budget) entries;", sum(len(v) for v in need.values()),
          "evaluations per store ->", path)


def load_need(path):
    return {tuple(k): set(v) for k, v in json.load(open(path))}


# =============================================================================================== choose / plans
def jl(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def aggregate(paths):
    """acc[(mem, acct)][design_json][cond][gi][sid] = (exact, n), feasible rows only."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict))))
    sids = set()
    for p in paths:
        for r in jl(p):
            sids.add(r["sid"])
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                acc[(x["mem"], x["acct"])][dj(x["design"])][x["cond"]][gidx(x["B"])][r["sid"]] = (x["exact"], x["n"])
    return acc, sids


def nstar_curve(curve):
    best = None
    for i in sorted(curve, reverse=True):
        if curve[i] is not None and curve[i] >= TARGET:
            best = i
        else:
            break
    return best


def choose(paths, out_path):
    """B38 / B58 / B63's rule per (memory, accounting, condition, budget): the design with the highest pooled dev
    share (counted at a budget only if feasible on every dev store); ties: the higher share at the next harder run
    condition at the same budget, then design-list order."""
    acc, sids = aggregate(paths)
    nst = len(sids)
    ch = dict(designs={}, dev_curve={}, dev_nstar={}, n_dev_stores=nst)

    def share(key, dk, c, gi):
        d = acc[key][dk][c].get(gi)
        if not d or len(d) < nst:
            return None
        return sum(v[0] for v in d.values()) / sum(v[1] for v in d.values())
    for mem in MEMS:
        for acct in ACCTS[mem]:
            key = (mem, acct)
            nm = "%s|%s" % key
            ch["designs"][nm], ch["dev_curve"][nm], ch["dev_nstar"][nm] = {}, {}, {}
            for c in COND_OF[mem]:
                nh = NEXT[c] if NEXT[c] in COND_OF[mem] else None
                dl = [dj(d) for d in designs(mem, c)]
                des, cur = {}, {}
                gis = sorted({gi for dk in dl for gi in acc[key][dk][c]})
                for gi in gis:
                    cands = []
                    for o, dk in enumerate(dl):
                        fr = share(key, dk, c, gi)
                        if fr is None:
                            continue
                        f2 = share(key, dk, nh, gi) if nh else None
                        cands.append((-fr, -(f2 or 0.0), o, dk, fr))
                    if not cands:
                        continue
                    cands.sort()
                    des[gi], cur[gi] = cands[0][3], cands[0][4]
                ch["designs"][nm][c] = {str(g): json.loads(v) for g, v in des.items()}
                ch["dev_curve"][nm][c] = {str(g): round(v, 6) for g, v in cur.items()}
                ns = nstar_curve(cur)
                ch["dev_nstar"][nm][c] = None if ns is None else GRID[ns]
    json.dump(ch, open(out_path, "w"), indent=1)
    print("dev N* (cells per stored pair), %d dev stores:" % nst)
    for nm in ch["dev_nstar"]:
        print("  %-16s" % nm, " ".join("%s %s" % (c, "%.0f" % v if v else "-") for c, v in ch["dev_nstar"][nm].items()))
        print("     design at dev N*:", {c: ch["designs"][nm][c].get(str(gidx(v))) if v else None
                                         for c, v in ch["dev_nstar"][nm].items()})
    return ch


def design_at(ch, nm, c, gi):
    des = ch["designs"][nm][c]
    if not des:
        return None
    if str(gi) in des:
        return des[str(gi)]
    near = min((int(g) for g in des), key=lambda g: (abs(g - gi), -g))
    return des[str(near)]


def testplan(choice_path, out_path, conds=None):
    """Per (memory, accounting, condition): every grid budget within 4 steps of the dev N* (the dev window's top 5 if
    dev never reached the bar), each with the dev design there (nearest dev budget's design; ties the larger)."""
    ch = json.load(open(choice_path))
    need = collections.defaultdict(set)
    for mem in MEMS:
        for acct in ACCTS[mem]:
            nm = "%s|%s" % (mem, acct)
            for c in COND_OF[mem]:
                if conds and c not in conds:
                    continue
                ns = ch["dev_nstar"][nm].get(c)
                if ns is None:
                    w = window(mem, c)
                    gis = w[-5:]
                else:
                    g0 = gidx(ns)
                    gis = range(max(0, g0 - 4), min(NG, g0 + 5))
                for gi in gis:
                    d = design_at(ch, nm, c, gi)
                    if d is None:
                        continue
                    need[(mem, dj(par_of(mem, d)), dj(rate_of(mem, d)), gi)].add(c)
    save_need(need, out_path)
    return need


def test_counts(ch, paths):
    """C[(mem, acct, cond)][gi][sid] = (failed, n) at the dev design of (mem, acct, cond, gi); infeasible skipped."""
    C = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in paths:
        for r in jl(p):
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                nm = "%s|%s" % (x["mem"], x["acct"])
                gi = gidx(x["B"])
                if x["design"] != design_at(ch, nm, x["cond"], gi):
                    continue
                C[(x["mem"], x["acct"], x["cond"])][gi][r["sid"]] = (len(x["failed"]), x["n"])
    return C


def extend(choice_path, plan_paths, test_paths, out_path, sids):
    """B38's extension rule, one round: top evaluated budget fails -> the next 4 up; N* = the lowest evaluated budget
    -> the 4 below; budgets already planned are not requested again."""
    ch = json.load(open(choice_path))
    planned = set()
    for p in plan_paths:
        for (mem, pj_, rj, gi), cs in load_need(p).items():
            for c in cs:
                planned.add((mem, c, gi))
    C = test_counts(ch, test_paths)
    need = collections.defaultdict(set)
    nadd = 0
    for (mem, acct, c), cur in C.items():
        ns, sh = B69.nstar_counts(cur, sids)
        if not sh:
            continue
        top, low = max(sh), min(sh)
        add = []
        if sh[top] < TARGET and top < NG - 1:
            add = list(range(top + 1, min(NG, top + 5)))
        elif ns == low and low > 0:
            add = list(range(max(0, low - 4), low))
        nm = "%s|%s" % (mem, acct)
        for gi in add:
            if (mem, c, gi) in planned:
                continue
            d = design_at(ch, nm, c, gi)
            need[(mem, dj(par_of(mem, d)), dj(rate_of(mem, d)), gi)].add(c)
            nadd += 1
    save_need(need, out_path)
    print(nadd, "(memory, condition, budget) additions")
    return nadd


# =============================================================================================== self-test (synthetic)
def selftest(quick=False):
    import b73_cells as F
    ok = True
    t00 = time.time()
    st = [F.make_store(f, "selftest", 0, "b75-selftest") for f in ("code1k5", "mv", "tiny")]
    big = F.make_store("wp6k", "selftest", 0, "b75-selftest")
    # ---- (C) the insertable table: insertion invariants, online property, bits per pair, noiseless reads
    for sid, pairs in st + [big]:
        S38 = E.Store38(sid, pairs)
        mph = mphf_bits_per_pair(S38)
        ckb = O.Cuckoo(S38, 12)
        cuckoo_bpp = len(ckb.bits(S38)) / S38.n
        for g in O2_GS:
            for ss in O2_SS:
                ck = CuckooStash(S38, g, ss)
                placed = int((ck.rows >= 0).sum())
                inv = placed + ck.nst + len(ck.lost) == S38.n
                # every placed pair in one of its buckets under its own fingerprint; no (pair, fp) conflict
                conf = 0
                for b in range(ck.m):
                    for e in range(BUCKET):
                        r = int(ck.rows[b, e])
                        if r < 0:
                            continue
                        i1, i2, fp = O.buckets_of(S38.s.rk[r], int(S38.s.rj[r]), ck.m, g)
                        inv &= (b in (i1, i2)) and int(ck.fps[b, e]) == fp
                        other = [x for x in ck.fps[i1].tolist() + (ck.fps[i2].tolist() if i2 != i1 else [])]
                        conf += other.count(fp) > 1
                # online: half build continued = full build
                h = CuckooStash(S38, g, ss, upto=S38.n // 2)
                h2 = CuckooStash(S38, g, ss, state=(h.fps, h.rows, h.sh, h.srow, h.nst, h.rng, h.lost, h.done))
                onl = (np.array_equal(h2.fps, ck.fps) and np.array_equal(h2.rows, ck.rows) and
                       np.array_equal(h2.sh, ck.sh) and h2.nst == ck.nst)
                bits = ck.bits(S38)
                got = ck.read(S38, bits)
                fl = E.exact(got, S38.want)
                w = ck.where()
                nd = ck.needs(S38)
                okc = inv and conf == 0 and onl
                ok &= okc
                print("  C %-24s g%-2d ss%d m %5d stash %4d/%4d lost %d bits/pair %.2f (cuckoo f12 %.2f, MPHF %.2f) "
                      "noiseless %d/%d exact, where %d, needs %d %s" % (
                          sid, g, ss, ck.m, ck.nst, ck.cap, len(ck.lost), len(bits) / S38.n, cuckoo_bpp, mph,
                          len(S38.want) - len(fl), len(S38.want), len(w), len(nd), "ok" if okc else "BAD"), flush=True)
        if quick:
            break
    # noisy O2 evaluations on the medium store
    sid, pairs = st[0]
    S38 = E.Store38(sid, pairs)
    for g, ss, arm, R, B, c in ((8, 1, "O2-BCH", None, 45, "k4"), (8, 1, "O2-LDPC", 0.5, 76, "k8"),
                                (6, 0, "O2-LDPC", 0.25, 256, "k16"), (8, 0, "O2-BCH", None, 27, "k1")):
        ck = CuckooStash(S38, g, ss)
        t0 = time.time()
        for x in o2_evaluate(S38, ck, arm, R, B, [c]):
            over = (not x["infeasible"]) and x["cells"] > x["budget"]
            ok &= not over
            print("  C %s g%d ss%d R %s B %d %s: %s/%s exact cells %s budget %s %s %.1fs" % (
                arm, g, ss, R, B, c, x.get("exact"), x.get("n"), x.get("cells"), x["budget"], x.get("diag"),
                time.time() - t0), flush=True)
    # ---- (A) U-LS: (None, 0) equals U; every design within budget; reads
    for sid, pairs in [st[0], st[1]]:
        S38 = E.Store38(sid, pairs)
        for B, c in ((38, "k1"), (54, "k1"), (91, "k8")):
            rowsU = None
            for des in ULS_DESIGNS:
                t0 = time.time()
                x = uls_eval(S38, des[0], des[1], B, [c])[0]
                if des == [None, 0]:
                    bd = E.vsa_build(S38, B, "last")
                    ch = E.Channel(S38.sid, "V-last|null", B, len(bd["cells"]), int(B * S38.n))
                    gu = E.vsa_rule(E.vsa_read(S38, bd, ch.apply(bd["cells"], c)), "last")
                    rowsU = E.exact(gu, S38.want)
                    ok &= rowsU == x["failed"]
                ok &= x["infeasible"] or x["cells"] <= x["budget"]
                print("  A %-24s B %3d %-3s design %-11s exact %4d/%4d cells %d/%d %.1fs" % (
                    sid, B, c, dj(des), x.get("exact", -1), x.get("n", -1), x.get("cells", -1), x.get("budget", -1),
                    time.time() - t0), flush=True)
    # ---- (B) joint decoders: T = 0 equals U / H-SBC; the decoder does not read stored values; budgets
    for sid, pairs in [st[0]] + ([] if quick else [big]):
        S38 = E.Store38(sid, pairs)
        for B, c in ((38, "k1"), (54, "k1"), (128, "k8")):
            for acct in ("free", "chg"):
                t0 = time.time()
                xs = uj_eval(S38, B, c, acct)
                bd = E.vsa_build(S38, B, "last")
                ch = E.Channel(S38.sid, "V-last|null", B, len(bd["cells"]), int(B * S38.n))
                gu = E.exact(E.vsa_rule(E.vsa_read(S38, bd, ch.apply(bd["cells"], c)), "last"), S38.want)
                if acct == "free":
                    ok &= xs[0]["failed"] == gu
                for x in xs:
                    ok &= x["infeasible"] or x["cells"] <= x["budget"]
                print("  B U-J %-22s B %3d %-3s %-4s " % (sid, B, c, acct) + " ".join(
                    "T%d %s" % (x["design"][0], x.get("exact")) for x in xs) + " /%s key cells %s (%.1fs; U %d)" % (
                    xs[0].get("n"), xs[0].get("key_cells"), time.time() - t0, len(S38.want) - len(gu)), flush=True)
        for lam in (0.5, 1.0):
            for B, c in ((16, "k1"), (23, "k1"), (64, "k8")):
                for acct in ("free", "chg"):
                    t0 = time.time()
                    xs = sbcj_eval(S38, lam, B, c, acct)
                    if acct == "free" and not xs[0].get("infeasible"):
                        r61 = H.sbc_evaluate(H.slim(E.Store38(sid, pairs)), lam, B, c, sic=False)
                        ok &= r61["rules"]["H-SBC"]["failed"] == xs[0]["failed"]
                    for x in xs:
                        ok &= x["infeasible"] or x["cells"] <= x["budget"]
                    print("  B H-SBC-J %-18s lam %.1f B %3d %-3s %-4s " % (sid, lam, B, c, acct) + " ".join(
                        "T%d %s" % (x["design"][1], x.get("exact")) for x in xs) + " /%s key cells %s (%.1fs)" % (
                        xs[0].get("n"), xs[0].get("key_cells"), time.time() - t0), flush=True)
    # leakage: decoders given scrambled stored values (after the write) decode identically
    sid, pairs = st[0]
    S38 = E.Store38(sid, pairs)
    B, c = 45, "k1"
    bd = E.vsa_build(S38, B, "last")
    ch = E.Channel(S38.sid, "V-last|null", B, len(bd["cells"]), int(B * S38.n))
    sig = ch.apply(bd["cells"], c)
    S, d = bd["S"], bd["d"]
    a1 = uj_decode(sig[:S * d].reshape(S, d), bd["shard_k"], S38.pk, S38.pj, S38.plast, S38.Kc, S38.Pc, S38.Vc, S38.L,
                   S38.qk, S38.J, 0.0, [1, 2], S38.vf(d))
    pv_save = S38.pv.copy()
    S38.pv = np.random.default_rng(5).permutation(S38.pv)
    a2 = uj_decode(sig[:S * d].reshape(S, d), bd["shard_k"], S38.pk, S38.pj, S38.plast, S38.Kc, S38.Pc, S38.Vc, S38.L,
                   S38.qk, S38.J, 0.0, [1, 2], S38.vf(d))
    S38.pv = pv_save
    ok &= a1 == a2
    print("  leakage U-J: decode with scrambled stored values identical:", a1 == a2)
    kl = keylist(S38)
    print("  key list:", {k: v for k, v in kl.items() if k != "coll"}, "collided probed", len(kl["coll"]),
          "cells k1/k8/k16:", [key_cells(kl["L"], E.IID[c]) for c in PRIMARY])
    print("SELFTEST", "PASS" if ok else "FAIL", "%.0f s" % (time.time() - t00))
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = lambda k, d: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)  # noqa: E731
    if cmd == "selftest":
        selftest("--quick" in sys.argv)
    elif cmd == "devplan":
        devplan(sys.argv[2])
    elif cmd == "run":
        stores = E.load_stores(sys.argv[2])
        only = arg("--only", None)
        conds = arg("--conds", None)
        jobs = make_jobs(stores, load_need(sys.argv[3]), only.split(",") if only else None,
                         conds.split(",") if conds else None)
        print("jobs", len(jobs), flush=True)
        pool_run(jobs, int(arg("--workers", 6)), sys.argv[4])
    elif cmd == "choose":
        choose(sys.argv[2].split(","), sys.argv[3])
    elif cmd == "testplan":
        cs = arg("--conds", None)
        testplan(sys.argv[2], sys.argv[3], cs.split(",") if cs else None)
    elif cmd == "extend":
        sids = [s for s, _ in E.load_stores(sys.argv[6])]
        extend(sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5], sids)
    else:
        raise SystemExit(cmd)
