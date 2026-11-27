"""B61 (prereg/B61_hd_protected_store_prereg.md): HD-native protected variants of the VSA store in B58 / B59's setting
(B38's 50 RULER test stores, binary PCM cells at the 30-day bit-error rates, exact-key metric, N* rule, budget grid).

Arms
  H-SBC  sparse block code store (Laiho et al. 2015; Frady et al. 2021 binding): every key, position, value and the
         LAST marker is a sparse block code of nb blocks of length Lb (one active position per block, positions from
         seeded hashes of the strings); binding = block-wise addition of the active indices mod Lb (block-wise circular
         convolution); a shard's superposition is the OR of its pairs' codes (one cell per position: active = +1,
         inactive = -1). Read: unbind block-wise with the query key and position, score each candidate (value, and
         value bound with LAST) by its overlap (number of blocks whose active position is set in the read), argmax,
         B38's V-last stop rule (stop when the LAST variant wins; ties: argmax first index, LAST only if strictly
         higher). Free parameter: the block load lam = (pairs per shard) / Lb, in LAMS.
  H-SIC  the same stored cells (same channel draw) read twice: pass 1 = H-SBC's read of every probed key; pass 2 =
         explaining away (successive interference cancellation, Kleyko et al. 2023): positions set by another decoded
         item of the probe batch are explained and carry no evidence; at unexplained positions a candidate scores
         log((1 - e) / q) if the cell reads active and log(e / (1 - q)) if inactive (e = max(cell-error rate, 1e-3),
         q = the shard's active fraction among unexplained cells), the same stop rule. Own lam choice.
  H-HAD  B38's V-last store (dense bipolar, sign of the bundle) with an error-correcting item codebook: the value codes
         are codewords of the first-order Reed-Muller code RM(1, m) (signed Walsh-Hadamard rows, minimum distance
         2^(m-1), the largest possible for 2^(m+1) words; 2^(m-1) < d <= 2^m), restricted to a seeded set of d of the
         2^m coordinates and scrambled by a seeded sign vector; LAST = the Hadamard row 2^(m-1), so value x LAST is again
         a codeword, distinct from every value's. The clean-up correlates with the vocabulary's codewords: the
         restriction of RM(1, m)'s maximum-likelihood decoder (fast Hadamard transform; checked in the self-test) to
         the vocabulary. No free parameter.

python3 b61_hd.py selftest
python3 b61_hd.py dev  OUT.json [--workers N]                  # dev pass 1 (octave budgets; H-HAD U's N* +- 4)
python3 b61_hd.py dev2 DEV1.json OUT.json [--workers N]        # dev pass 2 (quarter steps near pass-1 N*)
python3 b61_hd.py choose DEV.json[,DEV2.json] CHOICE.json
python3 b61_hd.py plan CHOICE.json PLAN.json
python3 b61_hd.py run STORES.json.gz PLAN.json OUT.json [--workers N] [--cap 40]
python3 b61_hd.py extend CHOICE.json TEST1.json[,...] PLAN_X.json
"""
import collections
import gzip
import json
import math
import sys
import time

import numpy as np
from numba import njit

import b38_eq as E
import ruler_vsa as V

GRID = E.GRID
NG = len(GRID)
D = E.D
IIDC = list(E.IID)
DMG4 = ["lines-nat-k4", "clust-nat-k4", "lines-nat-k8", "clust-nat-k8"]
LAMS = (0.35, 0.5, 0.7, 1.0)
RULES = ("H-SBC", "H-SIC", "H-HAD")
EPS_FLOOR = 1e-3
DEV_CAP, TEST_CAP = 7, 40
U_DEV_NSTAR = {"k1": 76, "k2": 76, "k4": 76, "e05": 91, "k8": 128, "k16": 431, "k32": 1722}   # B59 (B38's dev V-last)


def gidx(B):
    return min(range(NG), key=lambda i: abs(GRID[i] - B))


# =============================================================================================== hashed sparse codes
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def splitmix(x):
    with np.errstate(over="ignore"):
        z = x + _C1
        z = (z ^ (z >> np.uint64(30))) * _C2
        z = (z ^ (z >> np.uint64(27))) * _C3
        return z ^ (z >> np.uint64(31))


_SEEDS = {}


def seeds(strings, prefix):
    out = np.empty(len(strings), np.uint64)
    for i, s in enumerate(strings):
        k = prefix + s
        if k not in _SEEDS:
            if len(_SEEDS) > 200000:
                _SEEDS.clear()
            _SEEDS[k] = np.uint64(V._seed(k))
        out[i] = _SEEDS[k]
    return out


def block_codes(sd, nb, Lb):
    """Active index per block (nb blocks of length Lb) for each seed: (len(sd), nb) int32."""
    b = np.arange(nb, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = splitmix(sd[:, None] ^ splitmix(b[None, :] * _C2 + np.uint64(nb) * _C3 + np.uint64(Lb)))
    return (h % np.uint64(Lb)).astype(np.int32)


# =============================================================================================== H-SBC / H-SIC
def sbc_geometry(n, B, lam):
    total = int(B * n)
    S = max(1, math.ceil(total / D))
    d = min(D, total // S)
    Lb = max(2, int(round((n / S) / lam)))
    nb = d // Lb
    if nb < 1:
        return None
    return dict(total=total, S=S, d=d, Lb=Lb, nb=nb)


class SBCStore:
    def __init__(self, S38, B, lam):
        g = sbc_geometry(S38.n, B, lam)
        self.ok = g is not None
        if not self.ok:
            return
        self.__dict__.update(g)
        cd = S38.s.cd
        S, d, Lb, nb = self.S, self.d, self.Lb, self.nb
        self.shard_k = np.array([V._seed("shard:" + k) % S for k in cd["allk"]])       # B38's shard hash
        self.Kb = block_codes(seeds(cd["allk"], "b61k:"), nb, Lb)
        self.Pb = block_codes(seeds([str(j) for j in range(S38.J)], "b61p:"), nb, Lb)
        self.Vb = block_codes(seeds(cd["vocab"], "b61v:"), nb, Lb)
        self.Lc = block_codes(seeds(["last"], "b61l:"), nb, Lb)[0]
        ps = self.shard_k[S38.pk]
        pos = (self.Kb[S38.pk] + self.Pb[S38.pj] + self.Vb[S38.pv]
               + np.where(S38.plast[:, None] > 0, self.Lc[None, :], 0)) % Lb
        cell = ps[:, None].astype(np.int64) * d + np.arange(nb, dtype=np.int64)[None, :] * Lb + pos
        y = np.full(S * d, -1, np.int8)
        y[cell.ravel()] = 1
        self.cells = y
        self.qsh = self.shard_k[S38.qk]
        self.pair_cells = cell                                                          # (n, nb) for checks

    def offsets(self, S38, a, j):
        return ((self.Kb[S38.qk[a]] + self.Pb[j]) % self.Lb).astype(np.int32)


@njit(cache=True)
def _slot(yb, base, cov, own, o, Vb, Lc, Lb, A, Cc):
    """Decision for one slot: (value index, LAST won). yb: uint8 read (1 = active) of the whole array; base = shard
    offset; cov: int32 coverage by decoded items (length-1 zero array: none); own: the slot's own decoded item's cell
    per block (-1: none). Weight of a cell = 0 if covered by another decoded item, else A if active, -Cc if inactive;
    the score is computed from the integer counts as na A - ni Cc (one formula, so equal counts give equal scores).
    Candidates: value v (t = 0) and value v bound with LAST (t = 1). Decision = B38's V-last rule: LAST wins iff max s1 >
    max s0 (strict); argmax = first index. Exact pruning: a candidate whose bound (na + remaining) A - ni Cc (>= its final
    score, also in floating point: the same monotone formula) is below the best score seen so far (either type) cannot be
    the decision's winner nor change the comparison, so it stops (checked only after an inactive cell, where the bound
    drops)."""
    nb = o.shape[0]
    Nv = Vb.shape[0]
    hascov = cov.shape[0] > 1
    Ap = A if A > 0 else 0.0
    best0 = -1e300
    best1 = -1e300
    i0 = -1
    i1 = -1
    for v in range(Nv):
        for t in range(2):
            na = 0                                   # unexplained active cells
            ni = 0                                   # unexplained inactive cells
            pruned = False
            bestx = best0 if best0 > best1 else best1
            for b in range(nb):
                p = o[b] + Vb[v, b]
                if p >= Lb:
                    p -= Lb
                if t == 1:
                    p += Lc[b]
                    if p >= Lb:
                        p -= Lb
                cell = base + b * Lb + p
                if hascov:
                    cv = cov[cell]
                    if own[b] == cell:
                        cv -= 1
                    if cv > 0:
                        continue
                if yb[cell]:
                    na += 1
                else:
                    ni += 1
                    if (na + nb - b - 1) * Ap - ni * Cc < bestx:
                        pruned = True
                        break
            if pruned:
                continue
            a = na * A - ni * Cc                     # score from integer counts: exact, order-independent ties
            if t == 0:
                if a > best0:
                    best0 = a
                    i0 = v
            else:
                if a > best1:
                    best1 = a
                    i1 = v
    if best1 > best0:
        return i1, True
    return i0, False


def sbc_read(S38, st, yb, keys, cov=None, owns=None, AC=None):
    """V-last read of probed keys from the SBC cells. cov / owns / AC: explaining-away pass. Returns per key the
    decoded value list and per key the list of (j, value, last) items."""
    zc = np.zeros(1, np.int32)
    got, items = {}, {}
    none = np.full(st.nb, -1, np.int64)
    for a in keys:
        sh = int(st.qsh[a])
        base = sh * st.d
        g, it = [], []
        for j in range(S38.J):
            o = st.offsets(S38, a, j)
            if cov is None:
                v, last = _slot(yb, base, zc, none, o, st.Vb, st.Lc, st.Lb, 1.0, 0.0)
            else:
                A, Cc = AC[sh]
                v, last = _slot(yb, base, cov, owns.get((a, j), none), o, st.Vb, st.Lc, st.Lb, A, Cc)
            g.append(int(v))
            it.append((j, int(v), bool(last)))
            if last:
                break
        got[a] = g
        items[a] = it
    return got, items


def item_cells(S38, st, a, j, v, last):
    o = st.offsets(S38, a, j)
    p = (o + st.Vb[v] + (st.Lc if last else 0)) % st.Lb
    return int(st.qsh[a]) * st.d + np.arange(st.nb, dtype=np.int64) * st.Lb + p


def sbc_evaluate(S38, lam, B, cond, sic=True):
    """One (lam, budget, condition) on one store: H-SBC's and H-SIC's failed probed keys (same cells, same draw)."""
    st = SBCStore(S38, B, lam)
    nq = len(S38.want)
    if not st.ok:
        return dict(infeasible=True)
    tag = "H-SBC|%.4f" % lam
    ch = E.Channel(S38.sid, tag, B, len(st.cells), st.total)
    sig = ch.apply(st.cells, cond)
    yb = (sig > 0).astype(np.uint8)
    keys = list(range(nq))
    got1, items1 = sbc_read(S38, st, yb, keys)
    f1 = [a for a in keys if got1[a] != S38.want[a]]
    acc = dict(cells=int(len(st.cells)), budget=st.total, S=st.S, d=st.d, Lb=st.Lb, nb=st.nb,
               density=float((st.cells > 0).mean()))
    out = {"H-SBC": dict(exact=nq - len(f1), n=nq, failed=f1)}
    if sic:
        cov = np.zeros(len(yb), np.int32)
        owns = {}
        for a in keys:
            for (j, v, last) in items1[a]:
                c = item_cells(S38, st, a, j, v, last)
                np.add.at(cov, c, 1)
                owns[(a, j)] = c
        pc = E.parse_cond(cond)
        e = max(pc["eps"], EPS_FLOOR)
        AC = {}
        for sh in set(int(s) for s in st.qsh):
            seg = slice(sh * st.d, sh * st.d + st.nb * st.Lb)
            un = cov[seg] == 0
            q = float(yb[seg][un].mean()) if un.any() else 0.5
            q = min(max(q, 1e-3), 0.999)
            AC[sh] = (math.log((1 - e) / q), math.log((1 - q) / e))
        got2, _ = sbc_read(S38, st, yb, keys, cov=cov, owns=owns, AC=AC)
        f2 = [a for a in keys if got2[a] != S38.want[a]]
        out["H-SIC"] = dict(exact=nq - len(f2), n=nq, failed=f2)
    return dict(rules=out, **acc)


# =============================================================================================== H-HAD
def _parity(x):
    """Parity of the set bits of non-negative int64 values (< 2^32), by xor folding (numpy-version independent)."""
    x = x.astype(np.int64)
    for sh in (16, 8, 4, 2, 1):
        x = x ^ (x >> sh)
    return x & 1


def hadamard_rows(a, cols):
    """Rows a (int64 array) of the 2^m Sylvester-Hadamard matrix at columns cols: (len(a), len(cols)) int8 +-1."""
    out = np.empty((len(a), len(cols)), np.int8)
    for i in range(0, len(a), 512):
        x = _parity(a[i:i + 512, None] & cols[None, :])
        out[i:i + 512] = (1 - 2 * x).astype(np.int8)
    return out


class HADStore:
    def __init__(self, S38, B):
        n, Nv = S38.n, S38.Nv
        total = int(B * n)
        S = max(1, math.ceil(total / D))
        d = min(D, total // S)
        self.ok = d >= 16
        if not self.ok:
            return
        self.total, self.S, self.d = total, S, d
        m = max(1, math.ceil(math.log2(d)))
        N2 = 2 ** m
        self.m, self.N2 = m, N2
        rng = np.random.default_rng(E.seed_of(S38.sid, "b61had", d))
        cols = np.sort(rng.choice(N2, d, replace=False)).astype(np.int64)
        r = rng.choice(np.array([-1, 1], np.int8), d)
        slots = rng.permutation(N2)[:min(Nv, N2)].astype(np.int64)
        half = N2 // 2
        a = slots % half
        sg = np.where(slots >= half, -1, 1).astype(np.int8)
        VH = hadamard_rows(a, cols) * sg[:, None] * r[None, :]
        self.overflow = max(0, Nv - N2)
        if self.overflow:
            extra = np.random.default_rng(E.seed_of(S38.sid, "b61had-extra", d)).choice(
                np.array([-1, 1], np.int8), (self.overflow, d))
            VH = np.concatenate([VH, extra])
        self.VH = VH.astype(np.int8)
        self.LH = hadamard_rows(np.array([half], np.int64), cols)[0]
        self.cols, self.r, self.a, self.sg = cols, r, a, sg
        shard_k = np.array([V._seed("shard:" + k) % S for k in S38.s.cd["allk"]])       # B38's shard hash
        self.shard_k = shard_k
        ps = shard_k[S38.pk]
        mem = np.zeros((S, d), np.int32)
        order = np.argsort(ps, kind="stable")
        sp_ = ps[order]
        starts = np.flatnonzero(np.r_[True, sp_[1:] != sp_[:-1]])
        CH = 2048
        for i0 in range(0, len(order), CH):                     # chunked sum of the bound pairs per shard
            idx = order[i0:i0 + CH]
            prod = (S38.Kc[S38.pk[idx], :d] * S38.Pc[S38.pj[idx], :d] * self.VH[S38.pv[idx]]).astype(np.int32)
            prod = np.where(S38.plast[idx, None] > 0, prod * self.LH[None, :], prod)
            for s in np.unique(ps[idx]):
                mem[s] += prod[ps[idx] == s].sum(0)
        b1 = np.sign(mem).astype(np.int8)
        ties = b1 == 0
        b1[ties] = np.random.default_rng(0).choice(np.array([-1, 1], np.int8), size=int(ties.sum()))
        self.cells = b1.ravel()
        self.qsh = shard_k[S38.qk]


def had_read(S38, st, sig, keys):
    S, d = st.S, st.d
    rows = sig[:S * d].reshape(S, d).astype(np.float32)
    Vn = st.VH.astype(np.float32)
    Lf = st.LH.astype(np.float32)
    keys = np.asarray(keys)
    sh = st.qsh[keys]
    X = rows[sh] * S38.Kc[S38.qk[keys], :d]
    cnt = np.maximum((X != 0).sum(1), 1).astype(np.float32)
    nq = len(keys)
    best = np.zeros((nq, S38.J), np.int64)
    tag = np.zeros((nq, S38.J), bool)
    for j in range(S38.J):
        Y = X * S38.Pc[j, :d]
        s0 = (Y @ Vn.T) / cnt[:, None]
        s1 = ((Y * Lf) @ Vn.T) / cnt[:, None]
        m0, m1 = s0.max(1), s1.max(1)
        useL = m1 > m0
        best[:, j] = np.where(useL, s1.argmax(1), s0.argmax(1))
        tag[:, j] = useL
    out = []
    for i in range(nq):
        g = []
        for j in range(S38.J):
            g.append(int(best[i, j]))
            if tag[i, j]:
                break
        out.append(g)
    return out


def had_evaluate(S38, B, cond, cap):
    st = HADStore(S38, B)
    nq = len(S38.want)
    if not st.ok:
        return dict(infeasible=True)
    tag = "H-HAD|[]"
    ch = E.Channel(S38.sid, tag, B, len(st.cells), st.total)
    sig = ch.apply(st.cells, cond)
    failed, censored = [], False
    CHK = 50
    for i0 in range(0, nq, CHK):
        keys = list(range(i0, min(nq, i0 + CHK)))
        got = had_read(S38, st, sig, keys)
        failed += [a for a, g in zip(keys, got) if g != S38.want[a]]
        if len(failed) >= cap:
            censored = True
            break
    return dict(rules={"H-HAD": dict(exact=None if censored else nq - len(failed), n=nq, failed=failed,
                                     censored=censored)},
                cells=int(len(st.cells)), budget=st.total, S=st.S, d=st.d, m=st.m, overflow=st.overflow)


# =============================================================================================== jobs
def slim(S38):
    for dct in (S38.s.cd, S38.s.st):
        for k in ("Kc", "Vc", "Pc", "mem", "b1"):
            if k in dct:
                dct[k] = None
    return S38


def run_job(job):
    t0 = time.time()
    S38 = slim(E.Store38(job["sid"], job["pairs"]))
    out = []
    for arm, lam, B, conds in job["evals"]:
        for c in conds:
            if arm == "SBC":
                r = sbc_evaluate(S38, lam, B, c)
            else:
                r = had_evaluate(S38, B, c, job["cap"])
            r.update(arm=arm, lam=lam, B=B, cond=c)
            out.append(r)
    return dict(sid=job["sid"], n_pairs=S38.n, n_values=S38.Nv, nq=len(S38.want), res=out, sec=time.time() - t0)


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done, last = [], time.time()
    with Pool(workers, maxtasksperchild=20) as pool:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            done.append(r)
            print("done %-28s %5.0fs %d/%d" % (r["sid"], r["sec"], len(done), len(jobs)), flush=True)
            if time.time() - last > 120:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def make_jobs(stores, need, cap, per_job=None):
    """need: {(arm, lam, gi): set(conds)} -> one job per (store, arm, lam, budget); the conditions of an H-SBC budget
    with lam x B >= 256 (slow reads) are split one per job. Longest jobs first (estimated cost)."""
    jobs = []
    for sid, pairs in stores:
        for (arm, lam, gi), cs in sorted(need.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1],
                                                                        kv[0][2])):
            cs = sorted(cs)
            B = GRID[gi]
            split = arm == "SBC" and lam * B >= 256
            groups = [[c] for c in cs] if split else [cs]
            for g in groups:
                cost = len(pairs) * (lam * B if arm == "SBC" else 2000.0) * len(g)
                jobs.append(dict(sid=sid, pairs=pairs, evals=[(arm, lam, B, g)], cap=cap, cost=cost))
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def dev_need():
    """Pass 1: H-SBC / H-SIC every lam at the octave budgets 8, 16, ..., 4,096; H-HAD every budget within 4 grid steps
    of U's dev N*; the seven i.i.d. conditions."""
    need = collections.defaultdict(set)
    for lam in LAMS:
        for gi in range(0, NG, 4):
            for c in IIDC:
                need[("SBC", lam, gi)].add(c)
    for c in IIDC:
        u = gidx(U_DEV_NSTAR[c])
        for gi in range(max(0, u - 4), min(NG, u + 5)):
            need[("HAD", None, gi)].add(c)
    return need


def dev2_need(dev1_path):
    """Pass 2 (B58's rule): every lam at the quarter steps within 3 grid steps of each (rule, condition)'s pass-1 N*
    (H-SBC or H-SIC; the top 4 budgets if not reached), not already evaluated."""
    ch = choose([dev1_path], dev1_path + ".pass1_choice.json")
    need = collections.defaultdict(set)
    for rule in ("H-SBC", "H-SIC"):
        for c in IIDC:
            ns = ch["dev_nstar"][rule][c]
            gis = range(NG - 4, NG) if ns is None else range(max(0, gidx(ns) - 3), min(NG, gidx(ns) + 4))
            for gi in gis:
                if gi % 4 == 0:
                    continue
                for lam in LAMS:
                    need[("SBC", lam, gi)].add(c)
    return need


# =============================================================================================== dev / choose
def aggregate(paths):
    """acc[rule][design][cond][gi] = [exact, n, censored stores]."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0, 0]))))
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                for rule, y in x["rules"].items():
                    a = acc[rule][json.dumps(x["lam"])][x["cond"]][gidx(x["B"])]
                    a[1] += y["n"]
                    if y.get("censored"):
                        a[2] += 1
                    else:
                        a[0] += y["exact"]
    return acc


def nstar(curve):
    best = None
    for i in sorted(curve, reverse=True):
        if curve[i] is not None and curve[i] >= E.TARGET:
            best = i
        else:
            break
    return best


def choose(paths, out_path):
    """B38 / B58's rule per (rule, condition, budget): the design with the highest dev share; ties the higher share at
    the next harder condition at the same budget, then list order (lam ascending); censored below completed."""
    acc = aggregate(paths)
    ch = dict(designs={}, dev_curve={}, dev_nstar={})
    for rule in RULES:
        dl = [json.dumps(None)] if rule == "H-HAD" else [json.dumps(l) for l in LAMS]
        ch["designs"][rule], ch["dev_curve"][rule], ch["dev_nstar"][rule] = {}, {}, {}
        for c in IIDC:
            nh = E.NEXT_HARDER[c]
            des, cur, cens = {}, {}, []
            gis = sorted({gi for dk in dl for gi in acc[rule][dk][c]})
            for gi in gis:
                cands = []
                for o, dk in enumerate(dl):
                    a = acc[rule][dk][c].get(gi)
                    if a is None or a[1] == 0:
                        continue
                    if a[2]:
                        cands.append((1, 0.0, 0.0, o, dk, None))
                        continue
                    fr = a[0] / a[1]
                    a2 = acc[rule][dk][nh].get(gi) if nh else None
                    f2 = (a2[0] / a2[1]) if (a2 and a2[1] and not a2[2]) else 0.0
                    cands.append((0, -fr, -f2, o, dk, fr))
                if not cands:
                    continue
                cands.sort()
                if cands[0][0] == 0:
                    des[gi], cur[gi] = cands[0][4], cands[0][5]
                else:
                    cens.append(gi)
                    cur[gi] = None
            for gi in cens:
                if des:
                    near = min(des, key=lambda g: (abs(g - gi), -g))
                    des[gi] = des[near]
            ch["designs"][rule][c] = {str(g): json.loads(v) for g, v in des.items()}
            ch["dev_curve"][rule][c] = {str(g): (None if v is None else round(v, 6)) for g, v in cur.items()}
            ns = nstar(cur)
            ch["dev_nstar"][rule][c] = None if ns is None else GRID[ns]
    json.dump(ch, open(out_path, "w"), indent=1)
    print("dev N* (cells per stored pair):")
    print("%-7s" % "", " ".join("%8s" % c for c in IIDC))
    for rule in RULES:
        print("%-7s" % rule, " ".join("%8s" % ("%.0f" % x if x else "-") for x in
                                      (ch["dev_nstar"][rule].get(c) for c in IIDC)))
    for rule in ("H-SBC", "H-SIC"):
        print(" ", rule, "lam at dev N*:", {c: ch["designs"][rule][c].get(str(gidx(ch["dev_nstar"][rule][c])))
                                             if ch["dev_nstar"][rule][c] else None for c in IIDC})
    return ch


def design_at(ch, rule, cond, gi):
    m = E.parse_cond(cond)["match"]
    des = ch["designs"][rule][m]
    if str(gi) in des:
        return des[str(gi)]
    near = min((int(g) for g in des), key=lambda g: (abs(g - gi), -g))
    return des[str(near)]


def plan(choice_path, out_path):
    """Test: per (rule, condition) every grid budget from 8 to the matched i.i.d. dev N* + 4 grid steps (the whole grid
    if dev never reached the bar) with the rule's dev design there (H-SBC, H-SIC: i.i.d. and the 4 natural-layout
    damage conditions); H-HAD: dev N* - 4 ... + 4 (i.i.d. only)."""
    ch = json.load(open(choice_path))
    need = collections.defaultdict(set)
    for rule in RULES:
        for c in IIDC + (DMG4 if rule != "H-HAD" else []):
            m = E.parse_cond(c)["match"]
            ns = ch["dev_nstar"][rule][m]
            top = NG - 1 if ns is None else min(NG - 1, gidx(ns) + 4)
            lo = 0 if rule != "H-HAD" else (max(0, gidx(ns) - 4) if ns else NG - 5)
            for gi in range(lo, top + 1):
                lam = design_at(ch, rule, c, gi)
                need[("HAD" if rule == "H-HAD" else "SBC", lam, gi)].add(c)
    out = dict(need=[[list(k), sorted(v)] for k, v in need.items()])
    json.dump(out, open(out_path, "w"))
    print(len(need), "(arm, design, budget) entries;", sum(len(v) for v in need.values()), "evaluations per store")


def load_need(path):
    return {tuple(k): set(v) for k, v in json.load(open(path))["need"]}


def test_curves(paths):
    """cur[rule][cond][gi] = pooled share at the rule's evaluation with its design (None if a store censored)."""
    rows = collections.defaultdict(dict)
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                for rule, y in x["rules"].items():
                    rows[(rule, x["cond"], gidx(x["B"]), json.dumps(x["lam"]))][r["sid"]] = y
    return rows


def extend(choice_path, test_paths, out_path):
    """B38's extension rule: if the top evaluated budget fails, 4 more steps up; if N* is the lowest evaluated budget
    (H-HAD), 4 steps down."""
    ch = json.load(open(choice_path))
    rows = test_curves(test_paths)
    nst = 50
    need = collections.defaultdict(set)
    nadd = 0
    for rule in RULES:
        for c in IIDC + (DMG4 if rule != "H-HAD" else []):
            curve = {}
            for (ru, cc, gi, dk), d in rows.items():
                if ru != rule or cc != c or json.loads(dk) != design_at(ch, rule, c, gi):
                    continue
                if len(d) < nst:
                    continue
                n = sum(y["n"] for y in d.values())
                cens = any(y.get("censored") for y in d.values())
                curve[gi] = None if cens else sum(y["exact"] for y in d.values()) / n
            if not curve:
                continue
            ev = sorted(curve)
            ns = nstar(curve)
            add = []
            top_ok = curve[max(ev)] is not None and curve[max(ev)] >= E.TARGET
            if not top_ok and max(ev) < NG - 1:
                add = list(range(max(ev) + 1, min(NG, max(ev) + 5)))
            elif ns is not None and ns == min(ev) and ns > 0:
                add = list(range(max(0, ns - 4), ns))
            for gi in add:
                lam = design_at(ch, rule, c, gi)
                need[("HAD" if rule == "H-HAD" else "SBC", lam, gi)].add(c)
                nadd += 1
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(nadd, "(rule, condition, budget) additions")
    return nadd


# =============================================================================================== self-test (synthetic)
def synth(n_keys=600, n_pairs=700, nv=500, seed=1):
    rng = np.random.default_rng(seed)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(n_pairs):
        k = "key%d" % rng.integers(0, n_keys)
        pairs.append((k, "val%d" % (i % nv), seen[k]))
        seen[k] += 1
    return pairs


def plain_sbc_read(S38, st, yb, a):
    """Second implementation (explicit loops) of H-SBC's read of probed key a."""
    sh = int(st.qsh[a])
    k = S38.qk[a]
    g = []
    for j in range(S38.J):
        best0, b0, best1, b1 = -1, -1, -1, -1
        for v in range(S38.Nv):
            c0 = c1 = 0
            for b in range(st.nb):
                i0 = (int(st.Kb[k, b]) + int(st.Pb[j, b]) + int(st.Vb[v, b])) % st.Lb
                i1 = (i0 + int(st.Lc[b])) % st.Lb
                c0 += int(yb[sh * st.d + b * st.Lb + i0])
                c1 += int(yb[sh * st.d + b * st.Lb + i1])
            if c0 > best0:
                best0, b0 = c0, v
            if c1 > best1:
                best1, b1 = c1, v
        if best1 > best0:
            g.append(b1)
            break
        g.append(b0)
    return g


def selftest():
    t0 = time.time()
    pairs = synth()
    S38 = slim(E.Store38("selftest/x/0", pairs))
    nq = len(S38.want)
    print("synthetic store: n", S38.n, "Nv", S38.Nv, "probed", nq, "J", S38.J)
    # 1. geometry and accounting
    for lam in LAMS:
        for B in (32.0, 128.0, 512.0):
            st = SBCStore(S38, B, lam)
            assert len(st.cells) <= int(B * S38.n), "over budget"
            assert st.nb * st.Lb <= st.d
            dens = (st.cells > 0).mean()
            print("lam %.2f B %4.0f: S %d d %d Lb %d nb %d cells %d / %d density %.3f" % (
                lam, B, st.S, st.d, st.Lb, st.nb, len(st.cells), int(B * S38.n), dens))
    # 2. every stored pair's cells are active; the numba read = the explicit-loop read
    st = SBCStore(S38, 64.0, 0.7)
    assert (st.cells[st.pair_cells.ravel()] == 1).all()
    yb = (st.cells > 0).astype(np.uint8)
    got, _ = sbc_read(S38, st, yb, list(range(nq)))
    for a in range(0, nq, max(1, nq // 25)):
        assert plain_sbc_read(S38, st, yb, a) == got[a], a
    print("numba read == explicit loops on %d probed keys" % len(range(0, nq, max(1, nq // 25))))
    # 3. noise-free and noisy exactness, both rules
    for lam in LAMS:
        for B, c in ((32.0, "k1"), (64.0, "k1"), (128.0, "k4"), (256.0, "k8")):
            r = sbc_evaluate(S38, lam, B, c)
            print("lam %.2f B %4.0f %-3s exact SBC %d SIC %d of %d" % (
                lam, B, c, r["rules"]["H-SBC"]["exact"], r["rules"]["H-SIC"]["exact"], nq))
    r = sbc_evaluate(S38, 0.7, 512.0, "k1")
    assert r["rules"]["H-SBC"]["exact"] == nq and r["rules"]["H-SIC"]["exact"] == nq
    # 4. explaining away with no decoded items = the plain read (weights reduce to counts)
    st = SBCStore(S38, 48.0, 0.5)
    ch = E.Channel(S38.sid, "H-SBC|0.5000", 48.0, len(st.cells), st.total)
    yb = (ch.apply(st.cells, "k8") > 0).astype(np.uint8)
    g1, _ = sbc_read(S38, st, yb, list(range(nq)))
    cov = np.zeros(len(yb), np.int32)
    q = yb[:st.S * st.d].mean()
    AC = {s: (1.0, 0.0) for s in range(st.S)}
    g2, _ = sbc_read(S38, st, yb, list(range(nq)), cov=cov, owns={}, AC=AC)
    assert g1 == g2
    print("SIC with empty coverage and (A, C) = (1, 0) == plain read: OK")
    # 5. H-HAD: orthogonality at d = 2^m, FHT == matrix correlation, noise-free read, accounting
    pairs2 = synth(n_keys=300, n_pairs=256, nv=200, seed=3)
    S2 = slim(E.Store38("selftest/h/0", pairs2))
    st = HADStore(S2, 32.0)                                    # 8192 cells -> d = 8192 = 2^13
    print("H-HAD d %d m %d S %d overflow %d" % (st.d, st.m, st.S, st.overflow))
    G = st.VH.astype(np.int64) @ st.VH.T.astype(np.int64)
    off = G - np.diag(np.diag(G))
    print("d = 2^m: max |off-diagonal inner product| = %d (0 = orthogonal or antipodal)" % np.abs(off).max())
    assert np.abs(off).max() == 0 or set(np.unique(np.abs(off))) <= {0, st.d}
    GL = (st.VH.astype(np.int64) * st.LH) @ st.VH.T.astype(np.int64)
    assert np.abs(GL).max() == 0, "value x LAST must be orthogonal to every value"
    z = np.random.default_rng(5).standard_normal(st.d)
    zz = np.zeros(st.N2)
    zz[st.cols] = z * st.r
    h = zz.copy()
    w = 1
    while w < st.N2:                                           # fast Walsh-Hadamard transform
        h = h.reshape(-1, 2 * w)
        a_, b_ = h[:, :w].copy(), h[:, w:].copy()
        h[:, :w], h[:, w:] = a_ + b_, a_ - b_
        h = h.reshape(-1)
        w *= 2
    corr = st.VH.astype(np.float64) @ z
    fht = h[st.a] * st.sg
    print("FHT (RM(1,m) ML decoder) correlations == matrix correlations: max diff %.2e" % np.abs(corr - fht).max())
    assert np.abs(corr - fht).max() < 1e-8
    r = had_evaluate(S2, 128.0, "k1", 10 ** 9)
    print("H-HAD k1 B 128 exact %d of %d" % (r["rules"]["H-HAD"]["exact"], len(S2.want)))
    st3 = HADStore(S38, 100.0)
    G3 = st3.VH[:300].astype(np.int64) @ st3.VH[:300].T.astype(np.int64)
    o3 = np.abs(G3 - np.diag(np.diag(G3)))
    o3 = o3[o3 < st3.d].max()                                  # antipodal pairs (same row, both signs) excluded
    rnd = np.random.default_rng(7).choice([-1, 1], (300, st3.d))
    R3 = rnd @ rnd.T
    print("d = %d (2^m = %d): max |inner product| over 300 codes: RM(1,m) subset %d, random codes %d" % (
        st3.d, st3.N2, o3, np.abs(R3 - np.diag(np.diag(R3))).max()))
    for B, c in ((76.0, "k1"), (128.0, "k8")):
        r = had_evaluate(S38, B, c, 10 ** 9)
        assert r["cells"] <= int(B * S38.n)
        print("H-HAD B %.0f %s exact %d of %d" % (B, c, r["rules"]["H-HAD"]["exact"], nq))
    # 6. timing on a synthetic store of dev size (6,000 pairs, 6,000 values)
    pairs3 = synth(n_keys=6000, n_pairs=6000, nv=6000, seed=9)
    S3 = slim(E.Store38("selftest/t/0", pairs3))
    for B, c in ((64.0, "k1"), (512.0, "k16"), (2048.0, "k32")):
        t1 = time.time()
        sbc_evaluate(S3, 0.5, B, c)
        t2 = time.time()
        had_evaluate(S3, B, c, 10 ** 9) if B <= 512 else None
        t3 = time.time()
        print("timing n 6000 B %.0f: SBC+SIC %.1fs, HAD %.1fs" % (B, t2 - t1, t3 - t2))
    print("selftest PASS (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = lambda k, d: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)
    if cmd == "selftest":
        selftest()
    elif cmd == "dev":
        stores = E.load_stores("b34_out/dev_stores.json.gz")
        jobs = make_jobs(stores, dev_need(), DEV_CAP)
        print(len(jobs), "jobs")
        pool_run(jobs, int(arg("--workers", 8)), sys.argv[2])
    elif cmd == "dev2":
        stores = E.load_stores("b34_out/dev_stores.json.gz")
        jobs = make_jobs(stores, dev2_need(sys.argv[2]), DEV_CAP)
        print(len(jobs), "jobs")
        pool_run(jobs, int(arg("--workers", 8)), sys.argv[3])
    elif cmd == "choose":
        choose(sys.argv[2].split(","), sys.argv[3])
    elif cmd == "plan":
        plan(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        stores = E.load_stores(sys.argv[2])
        jobs = make_jobs(stores, load_need(sys.argv[3]), int(arg("--cap", TEST_CAP)))
        print(len(jobs), "jobs")
        pool_run(jobs, int(arg("--workers", 8)), sys.argv[4])
    elif cmd == "extend":
        extend(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
