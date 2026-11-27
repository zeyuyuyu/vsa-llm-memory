"""B63 (prereg/B63_online_directory_prereg.md): the best coded tables (E-BCH, E-LDPC) with their static minimal perfect
hash (MPHF) directory replaced by a directory that accepts online insertion, in B59 part 2 / B61's setting (B38's 50
RULER test stores, binary PCM cells at the 30-day bit-error rates, exact-key metric, N* rule, budget grid).

Online directory: a bucketized cuckoo hash table with partial-key cuckoo hashing (Fan, Andersen, Kaminsky and
Mitzenmacher 2014, "Cuckoo filter: practically better than Bloom"): m buckets of 4 slots, two candidate buckets per
item, i1 = h(x) mod m and i2 = alt(i1, fp) = (H(fp) - i1) mod m (an involution, so an entry can be moved knowing only
its fingerprint and its current bucket; the subtraction form works for any m, not only powers of two), fingerprint
fp in 1 .. 2^f - 1 (0 marks an empty slot), up to 500 kicks per insertion; m = ceil(n / (4 x 0.95)) (load <= 0.95).
Items are inserted one at a time in the store's pair order; an insertion sees only the table built so far and the new
pair (online). An item still homeless after 500 kicks is lost (the table is full) and its key then reads wrong.
Each slot holds [fp (f bits) | last flag | value index] - the MPHF table's row plus the fingerprint; every slot of
every bucket, empty or not, is on the cells. The table's bits (bucket-major) are the logical bitstream that B38's
E-BCH / E-LDPC machinery protects unchanged (shortened BCH with the closed-form (n, t) per store / IRA LDPC with rate
R, each coded bit repeated rho >= 1 times, majority / soft combining, bounded-distance / sum-product decoding).

Read of a probed key (B38's table loop, j = 0, 1, ...): compute i1, fp, i2 from (key, j) (stateless hashing, free);
the slots of the two buckets (8, or 4 if i1 = i2) are compared with fp on the decoded bits; exactly one exact match ->
that slot; two or more exact matches (a fingerprint false match) -> the read of that key fails (the reader cannot tell
which; no guess); no exact match -> the unique nearest slot in Hamming distance (a tie fails); return its value, stop
when its last flag is set.

Arms: O-BCH (design f; BCH (n, t) by B38's closed form per store, budget and condition, given f), O-LDPC (design
(f, R)). FS = fingerprint widths searched on dev.

python3 b63_online.py selftest
python3 b63_online.py dev  STORES.json.gz OUT.json [--workers N]            # dev pass 1 (octave budgets)
python3 b63_online.py dev2 STORES.json.gz DEV1.json OUT.json [--workers N]  # dev pass 2 (quarter steps near N*)
python3 b63_online.py choose DEV1.json[,DEV2.json] CHOICE.json
python3 b63_online.py plan CHOICE.json PLAN.json
python3 b63_online.py run STORES.json.gz PLAN.json OUT.json [--workers N]
python3 b63_online.py extend CHOICE.json PLAN.json[,PLAN_X1.json] TEST.json[,...] PLAN_X.json
"""
import collections
import gzip
import hashlib
import json
import math
import sys
import time

import numpy as np

import b38_eq as E

GRID = E.GRID
NG = len(GRID)
IIDC = list(E.IID)
BUCKET = 4
ALPHA = 0.95
MAX_KICKS = 500
FS = (12, 14, 16, 18)
RATES = E.RATES
ARMS = ("O-BCH", "O-LDPC")
M64 = (1 << 64) - 1


def gidx(B):
    return min(range(NG), key=lambda i: abs(GRID[i] - B))


# =============================================================================================== hashing
_QH = {}


def raw_hash(k, j):
    """(64-bit bucket hash, 64-bit fingerprint hash) of the identity (key, j): stateless, from SHA-256."""
    key = (k, j)
    v = _QH.get(key)
    if v is None:
        h = hashlib.sha256(("b63ck:%s\x00%d" % (k, j)).encode("utf-8")).digest()
        v = (int.from_bytes(h[0:8], "little"), int.from_bytes(h[8:16], "little"))
        if len(_QH) > 400000:
            _QH.clear()
        _QH[key] = v
    return v


def fp_of(hf, f):
    return 1 + hf % ((1 << f) - 1)


def mix64(x):
    """splitmix64 finalizer (the fingerprint's hash H(fp) for the alternate bucket)."""
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def alt(i, fp, m):
    return (mix64(fp) % m - i) % m


def buckets_of(k, j, m, f):
    hb, hf = raw_hash(k, j)
    fp = fp_of(hf, f)
    i1 = hb % m
    return i1, alt(i1, fp, m), fp


# =============================================================================================== the cuckoo table
def cuckoo_insert(fps, rows, k, j, row, m, f, rng):
    """Online insertion of pair `row` with identity (k, j) into the table (fps, rows), Fan et al.'s Algorithm 1 with
    partial-key cuckoo hashing. Sees only the current table and the new pair. Returns None, or the row index of the
    entry left homeless after MAX_KICKS (lost)."""
    i1, i2, fp = buckets_of(k, j, m, f)
    for b in (i1, i2):
        e = np.flatnonzero(fps[b] == 0)
        if len(e):
            fps[b, e[0]], rows[b, e[0]] = fp, row
            return None
    b = (i1, i2)[int(rng.integers(2))]
    for _ in range(MAX_KICKS):
        e = int(rng.integers(BUCKET))
        fp, fps[b, e] = int(fps[b, e]), fp
        row, rows[b, e] = int(rows[b, e]), row
        b = alt(b, fp, m)
        ee = np.flatnonzero(fps[b] == 0)
        if len(ee):
            fps[b, ee[0]], rows[b, ee[0]] = fp, row
            return None
    return row


class Cuckoo:
    """The online-insertion directory with the MPHF table's rows: m buckets x 4 slots x [fp | last | value]."""

    def __init__(self, S38, f, alpha=ALPHA, upto=None, state=None):
        s = S38.s
        self.f, self.n = f, s.n
        self.m = m = max(1, math.ceil(s.n / (BUCKET * alpha)))
        self.P = s.lastbit + s.b
        self.w = f + self.P
        if state is None:
            self.fps = np.zeros((m, BUCKET), np.int64)
            self.rows = np.full((m, BUCKET), -1, np.int64)
            self.rng = np.random.default_rng(E.seed_of(S38.sid, "b63cuckoo", f))
            self.lost = []
            start = 0
        else:                                                    # continue an earlier partial build (check)
            self.fps, self.rows, self.rng, self.lost, start = state
        stop = s.n if upto is None else upto
        for i in range(start, stop):
            r = cuckoo_insert(self.fps, self.rows, s.rk[i], int(s.rj[i]), i, m, f, self.rng)
            if r is not None:
                self.lost.append(r)
        self.done = stop
        self.empty = int((self.fps == 0).sum())
        self._bits = None

    def bits(self, S38):
        if self._bits is None:
            s = S38.s
            m, f = self.m, self.f
            fb = ((self.fps[:, :, None] >> np.arange(f - 1, -1, -1)[None, None, :]) & 1).astype(np.uint8)
            pay = (s.payload_bits() > 0).astype(np.uint8)            # (n, P)
            pb = np.zeros((m, BUCKET, self.P), np.uint8)
            occ = self.rows >= 0
            pb[occ] = pay[self.rows[occ]]
            self._bits = np.concatenate([fb, pb], 2).ravel()
        return self._bits

    def slot_of_rows(self):
        where = {}
        for b, e in zip(*np.nonzero(self.rows >= 0)):
            where[int(self.rows[b, e])] = (int(b), int(e))
        return where

    def needs(self, S38):
        """Logical bit positions each stored key's read needs (every stored key; for the BCH closed form): the
        fingerprints of every slot of both candidate buckets at each j, and the payload of its own slot."""
        s = S38.s
        where = self.slot_of_rows()
        f, w, m = self.f, self.w, self.m
        fpos = np.arange(f)
        ppos = np.arange(f, w)
        out = []
        for k in s.count:
            pos = []
            for j in range(s.count[k]):
                i1, i2, _ = buckets_of(k, j, m, f)
                for b in {i1, i2}:
                    for e in range(BUCKET):
                        pos.append((b * BUCKET + e) * w + fpos)
                i = s.row_of[(k, j)]
                if i in where:
                    b, e = where[i]
                    pos.append((b * BUCKET + e) * w + ppos)
            out.append(np.unique(np.concatenate(pos)))
        return out

    def read(self, S38, bits, diag=None):
        """Reads every probed key from the decoded logical bits (rule in the module docstring)."""
        s = S38.s
        m, f, w = self.m, self.f, self.w
        arr = bits[:m * BUCKET * w].reshape(m, BUCKET, w).astype(np.int64)
        fpr = (arr[:, :, :f] * (1 << np.arange(f - 1, -1, -1))[None, None, :]).sum(2)
        last, val = s.decode_payload(arr[:, :, f:].reshape(m * BUCKET, self.P).astype(np.int8) * 2 - 1)
        last, val = last.reshape(m, BUCKET), val.reshape(m, BUCKET)
        dg = diag if diag is not None else {}
        for kk in ("ambiguous", "nearest", "nearest_tie"):
            dg.setdefault(kk, 0)
        got = []
        for k in s.keys:
            g = []
            for j in range(s.maxj + 1):
                i1, i2, fp = buckets_of(k, j, m, f)
                cand = [(i1, e) for e in range(BUCKET)] + ([(i2, e) for e in range(BUCKET)] if i2 != i1 else [])
                mt = [c for c in cand if fpr[c] == fp]
                if len(mt) > 1:
                    dg["ambiguous"] += 1
                    g.append(-1)
                    break
                if len(mt) == 1:
                    c = mt[0]
                else:
                    dist = [bin(int(fpr[c]) ^ fp).count("1") for c in cand]
                    mn = min(dist)
                    wn = [c for c, x in zip(cand, dist) if x == mn]
                    if len(wn) > 1:
                        dg["nearest_tie"] += 1
                        g.append(-1)
                        break
                    dg["nearest"] += 1
                    c = wn[0]
                g.append(int(val[c]))
                if last[c]:
                    break
            got.append(g)
        return got


# =============================================================================================== one evaluation
def evaluate(S38, ck, arm, R, B, conds, needs=None):
    """One (arm, design, budget) on one store at the given conditions. Returns a list of result rows."""
    total = int(B * S38.n)
    bits = ck.bits(S38)
    L = len(bits)
    base = dict(arm=arm, f=ck.f, R=R, B=B, L=int(L), m=ck.m, lost=len(ck.lost), empty=ck.empty, budget=total)
    out = []
    if arm == "O-BCH":
        if needs is None:
            needs = ck.needs(S38)
        groups = collections.OrderedDict()
        for c in conds:
            mtc = E.parse_cond(c)["match"]
            nh = E.NEXT_HARDER.get(mtc)
            des = E.bch_choose(L, needs, total, E.parse_cond(mtc)["eps"], E.IID[nh] if nh else None)
            groups.setdefault(None if des is None else tuple(des), []).append(c)
        for des, cs in groups.items():
            if des is None:
                for c in cs:
                    out.append(dict(base, cond=c, infeasible=True))
                continue
            n_, k_, t_ = des
            lay = E.ecc_layout(L, n_, k_, t_, total)
            coded = E.bch_encode_stream(bits, lay)
            cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
            tag = "O-BCH|%s" % json.dumps([ck.f, n_, k_, t_])
            ch = E.Channel(S38.sid, tag, B, len(cells), total)
            for c in cs:
                sig = ch.apply(cells, c)
                rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, c, "dec"))
                dec, st = E.bch_decode_stream(coded, E.rep_sum(sig, lay["counts"]), lay, rng)
                dg = dict(cw_bound=st[0], cw_lib=st[1], cw_fail=st[2])
                got = ck.read(S38, dec, dg)
                fl = E.exact(got, S38.want)
                out.append(dict(base, cond=c, bch=[n_, k_, t_], cells=int(len(cells)), exact=len(S38.want) - len(fl),
                                n=len(S38.want), failed=fl, diag=dg))
        return out
    lay = E.ldpc_layout(L, R, total)
    if lay is None:
        return [dict(base, cond=c, infeasible=True) for c in conds]
    coded, keep = E.ldpc_encode_stream(bits, lay)
    cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
    tag = "O-LDPC|%s" % json.dumps([ck.f, R])
    ch = E.Channel(S38.sid, tag, B, len(cells), total)
    for c in conds:
        sig = ch.apply(cells, c)
        rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, c, "dec"))
        dec, nf = E.ldpc_decode_stream(E.rep_sum(sig, lay["counts"]), lay, keep, E.parse_cond(c)["eps"], rng)
        dg = dict(cw_nonconv=nf)
        got = ck.read(S38, dec[:L], dg)
        fl = E.exact(got, S38.want)
        out.append(dict(base, cond=c, cells=int(len(cells)), exact=len(S38.want) - len(fl), n=len(S38.want),
                        failed=fl, diag=dg))
    return out


def slim(S38):
    for dct in (S38.s.cd, S38.s.st):
        for k in ("Kc", "Vc", "Pc", "mem", "b1"):
            if k in dct:
                dct[k] = None
    S38.Kc = S38.Vc = S38.Pc = None
    return S38


def run_job(job):
    """job: dict(sid, pairs, arm, f, evals=[(R, B, conds)])."""
    t0 = time.time()
    S38 = slim(E.Store38(job["sid"], job["pairs"]))
    ck = Cuckoo(S38, job["f"])
    needs = ck.needs(S38) if job["arm"] == "O-BCH" else None
    out = []
    for R, B, conds in job["evals"]:
        out += evaluate(S38, ck, job["arm"], R, B, conds, needs)
    return dict(sid=job["sid"], arm=job["arm"], f=job["f"], n_pairs=S38.n, n_values=S38.Nv, nq=len(S38.want),
                m=ck.m, lost=len(ck.lost), empty=ck.empty, res=out, sec=time.time() - t0)


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done, last = [], time.time()
    with Pool(workers, maxtasksperchild=10) as pool:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            done.append(r)
            print("done %-28s %-6s f%-2d %5.0fs %d/%d" % (r["sid"], r["arm"], r["f"], r["sec"], len(done), len(jobs)),
                  flush=True)
            if time.time() - last > 120:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def make_jobs(stores, need, per_job=6):
    """need: {(arm, f, R, gi): set(conds)} -> jobs of at most per_job budgets per (store, arm, f) (largest first)."""
    jobs = []
    for sid, pairs in stores:
        byf = collections.defaultdict(list)
        for (arm, f, R, gi), cs in need.items():
            byf[(arm, f)].append((R, GRID[gi], sorted(cs)))
        for (arm, f), ev in sorted(byf.items()):
            ev.sort(key=lambda e: (-1 if e[0] is None else e[0], e[1]))
            for i in range(0, len(ev), per_job):
                part = ev[i:i + per_job]
                cost = len(pairs) * sum(len(e[2]) * (3.0 if arm == "O-BCH" else 1.0) for e in part)
                jobs.append(dict(sid=sid, pairs=pairs, arm=arm, f=f, evals=part, cost=cost))
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def designs(arm):
    """Design list in tie-break order: O-BCH f ascending; O-LDPC f ascending, then R in B38's RATES order."""
    if arm == "O-BCH":
        return [(f, None) for f in FS]
    return [(f, R) for f in FS for R in RATES]


def dev_need():
    """Pass 1: every design of both arms at the octave budgets 8, 16, ..., 4,096, the seven i.i.d. conditions."""
    need = collections.defaultdict(set)
    for arm in ARMS:
        for f, R in designs(arm):
            for gi in range(0, NG, 4):
                for c in IIDC:
                    need[(arm, f, R, gi)].add(c)
    return need


def dev2_need(dev1_path):
    """Pass 2 (B58 / B61's rule): every design at the quarter steps within 3 grid steps of each (arm, condition)'s
    pass-1 N* (the top 4 budgets if not reached), not already evaluated."""
    ch = choose([dev1_path], dev1_path + ".pass1_choice.json")
    need = collections.defaultdict(set)
    for arm in ARMS:
        for c in IIDC:
            ns = ch["dev_nstar"][arm][c]
            gis = range(NG - 4, NG) if ns is None else range(max(0, gidx(ns) - 3), min(NG, gidx(ns) + 4))
            for gi in gis:
                if gi % 4 == 0:
                    continue
                for f, R in designs(arm):
                    need[(arm, f, R, gi)].add(c)
    return need


# =============================================================================================== dev / choose / plan
def jload(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def aggregate(paths):
    """acc[arm][design key][cond][gi] = {sid: (exact, n)} (feasible evaluations only)."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict))))
    for p in paths:
        for r in jload(p):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                acc[x["arm"]][json.dumps([x["f"], x["R"]])][x["cond"]][gidx(x["B"])][r["sid"]] = (x["exact"], x["n"])
    return acc


def nstar(curve):
    best = None
    for i in sorted(curve, reverse=True):
        if curve[i] is not None and curve[i] >= E.TARGET:
            best = i
        else:
            break
    return best


def choose(paths, out_path, nstores=None):
    """B38 / B58's rule per (arm, condition, budget): the design with the highest pooled dev share (a design counts
    at a budget only if it was feasible on every dev store); ties: the higher share at the next harder condition at the
    same budget, then list order (designs())."""
    acc = aggregate(paths)
    sids = set()
    for arm in acc:
        for dk in acc[arm]:
            for c in acc[arm][dk]:
                for gi in acc[arm][dk][c]:
                    sids |= set(acc[arm][dk][c][gi])
    nst = nstores or len(sids)
    ch = dict(designs={}, dev_curve={}, dev_nstar={}, n_dev_stores=nst)

    def share(arm, dk, c, gi):
        d = acc[arm][dk][c].get(gi)
        if not d or len(d) < nst:
            return None
        return sum(v[0] for v in d.values()) / sum(v[1] for v in d.values())
    for arm in ARMS:
        dl = [json.dumps([f, R]) for f, R in designs(arm)]
        ch["designs"][arm], ch["dev_curve"][arm], ch["dev_nstar"][arm] = {}, {}, {}
        for c in IIDC:
            nh = E.NEXT_HARDER[c]
            des, cur = {}, {}
            gis = sorted({gi for dk in dl for gi in acc[arm][dk][c]})
            for gi in gis:
                cands = []
                for o, dk in enumerate(dl):
                    fr = share(arm, dk, c, gi)
                    if fr is None:
                        continue
                    f2 = share(arm, dk, nh, gi) if nh else None
                    cands.append((-fr, -(f2 or 0.0), o, dk, fr))
                if not cands:
                    continue
                cands.sort()
                des[gi], cur[gi] = cands[0][3], cands[0][4]
            ch["designs"][arm][c] = {str(g): json.loads(v) for g, v in des.items()}
            ch["dev_curve"][arm][c] = {str(g): round(v, 6) for g, v in cur.items()}
            ns = nstar(cur)
            ch["dev_nstar"][arm][c] = None if ns is None else GRID[ns]
    json.dump(ch, open(out_path, "w"), indent=1)
    print("dev N* (cells per stored pair), %d dev stores:" % nst)
    print("%-8s" % "", " ".join("%8s" % c for c in IIDC))
    for arm in ARMS:
        print("%-8s" % arm, " ".join("%8s" % ("%.0f" % x if x else "-") for x in
                                     (ch["dev_nstar"][arm].get(c) for c in IIDC)))
        print("  design at dev N*:", {c: ch["designs"][arm][c].get(str(gidx(ch["dev_nstar"][arm][c])))
                                      if ch["dev_nstar"][arm][c] else None for c in IIDC})
    return ch


def design_at(ch, arm, cond, gi):
    m = E.parse_cond(cond)["match"]
    des = ch["designs"][arm][m]
    if str(gi) in des:
        return des[str(gi)]
    near = min((int(g) for g in des), key=lambda g: (abs(g - gi), -g))
    return des[str(near)]


def plan(choice_path, out_path):
    """Test: per (arm, i.i.d. condition) every grid budget within 4 grid steps of the dev N* (B38's table rule; the top
    5 budgets if dev never reached the bar), each with the arm's dev design there."""
    ch = json.load(open(choice_path))
    need = collections.defaultdict(set)
    for arm in ARMS:
        for c in IIDC:
            ns = ch["dev_nstar"][arm][c]
            gis = range(NG - 5, NG) if ns is None else range(max(0, gidx(ns) - 4), min(NG, gidx(ns) + 5))
            for gi in gis:
                f, R = design_at(ch, arm, c, gi)
                need[(arm, f, R, gi)].add(c)
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(len(need), "(arm, design, budget) entries;", sum(len(v) for v in need.values()), "evaluations per store")


def load_need(path):
    return {tuple(k): set(v) for k, v in json.load(open(path))["need"]}


def test_counts(ch, paths):
    """C[(arm, cond)] = {gi: {sid: (failed, n)}} at the arm's dev design (infeasible rows skipped)."""
    C = collections.defaultdict(dict)
    for p in paths:
        for r in jload(p):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                gi = gidx(x["B"])
                if [x["f"], x["R"]] != design_at(ch, x["arm"], x["cond"], gi):
                    continue
                C[(x["arm"], x["cond"])].setdefault(gi, {})[r["sid"]] = (len(x["failed"]), x["n"])
    return C


def extend(choice_path, plan_paths, test_paths, out_path, nst=50):
    """B38's extension rule: if the top evaluated budget fails, 4 more steps up; if N* is the lowest evaluated budget, 4
    steps down; budgets already in a plan are not requested again (infeasible ones stay infeasible)."""
    ch = json.load(open(choice_path))
    planned = collections.defaultdict(set)
    for p in plan_paths:
        for (arm, f, R, gi), cs in load_need(p).items():
            for c in cs:
                planned[(arm, c)].add(gi)
    C = test_counts(ch, test_paths)
    need = collections.defaultdict(set)
    nadd = 0
    for arm in ARMS:
        for c in IIDC:
            curve = {}
            for gi, d in C.get((arm, c), {}).items():
                if len(d) < nst:
                    continue
                curve[gi] = 1 - sum(v[0] for v in d.values()) / sum(v[1] for v in d.values())
            if not curve:
                continue
            ev = sorted(curve)
            ns = nstar(curve)
            add = []
            if curve[max(ev)] < E.TARGET and max(ev) < NG - 1:
                add = list(range(max(ev) + 1, min(NG, max(ev) + 5)))
            elif ns is not None and ns == min(ev) and ns > 0:
                add = list(range(max(0, ns - 4), ns))
            for gi in add:
                if gi in planned[(arm, c)]:
                    continue
                f, R = design_at(ch, arm, c, gi)
                need[(arm, f, R, gi)].add(c)
                nadd += 1
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(nadd, "(arm, condition, budget) additions")
    return nadd


# =============================================================================================== self-test (synthetic)
def synth(n_keys=2000, n_pairs=3000, nv=3000, seed=1):
    rng = np.random.default_rng(seed)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(n_pairs):
        k = "key%d" % rng.integers(0, n_keys)
        pairs.append((k, "val%d" % rng.integers(0, nv), seen[k]))
        seen[k] += 1
    return pairs


def mphf_logical_per_pair(S38):
    mc = E.mphf_content(S38)
    return len(mc["bits"]) / S38.n


def selftest():
    t0 = time.time()
    # 1. insertion: load, lost items, online property (prefix build + continuation = full build)
    for n_pairs in (150, 700, 3000, 6000):
        pairs = synth(n_keys=max(50, n_pairs * 2 // 3), n_pairs=n_pairs, nv=min(n_pairs, 5000), seed=n_pairs)
        S38 = slim(E.Store38("selftest/x/%d" % n_pairs, pairs))
        for f in (12, 14, 16):
            ck = Cuckoo(S38, f)
            assert int((ck.rows >= 0).sum()) + len(ck.lost) == S38.n
            # every stored pair sits in one of its two buckets with its own fingerprint
            for (b, e) in zip(*np.nonzero(ck.rows >= 0)):
                i = int(ck.rows[b, e])
                i1, i2, fp = buckets_of(S38.s.rk[i], int(S38.s.rj[i]), ck.m, f)
                assert b in (i1, i2) and ck.fps[b, e] == fp
            half = Cuckoo(S38, f, upto=S38.n // 2)
            cont = Cuckoo(S38, f, state=(half.fps.copy(), half.rows.copy(), half.rng, list(half.lost), half.done))
            same = np.array_equal(cont.fps, ck.fps) and np.array_equal(cont.rows, ck.rows)
            bits = ck.bits(S38)
            assert len(bits) == ck.m * BUCKET * ck.w
            print("n %5d f %2d: m %4d load %.3f lost %d empty slots %d; online (half + continue = full) %s; "
                  "logical bits/pair %.2f (MPHF table %.2f)" % (
                      S38.n, f, ck.m, S38.n / (ck.m * BUCKET), len(ck.lost), ck.empty, same, len(bits) / S38.n,
                      mphf_logical_per_pair(S38)))
            assert same
            # 2. noiseless read: every probed key exact unless a fingerprint false match
            dg = {}
            got = ck.read(S38, bits, dg)
            fl = E.exact(got, S38.want)
            print("   noiseless read: %d / %d probed keys failed; diag %s" % (len(fl), len(S38.want), dg))
            assert len(fl) == dg["ambiguous"] + dg["nearest_tie"] + (0 if not ck.lost else len(fl))
    # 3. false-match rate on a large synthetic store vs the model 1 - (1 - 2^-f)^(7 x load)
    pairs = synth(n_keys=20000, n_pairs=20000, nv=5000, seed=7)
    S38 = slim(E.Store38("selftest/fm/0", pairs))
    for f in (8, 10, 12):
        ck = Cuckoo(S38, f)
        where = ck.slot_of_rows()
        amb = tot = 0
        for i in range(S38.n):
            i1, i2, fp = buckets_of(S38.s.rk[i], int(S38.s.rj[i]), ck.m, f)
            cand = {(i1, e) for e in range(BUCKET)} | {(i2, e) for e in range(BUCKET)}
            mt = sum(1 for c in cand if ck.fps[c] == fp)
            amb += mt > 1
            tot += 1
        load = S38.n / (ck.m * BUCKET)
        print("false matches f %2d: %.5f per pair (model %.5f)" % (f, amb / tot, 1 - (1 - 1 / ((1 << f) - 1)) ** (7 * load)))
    # 4. a noisy run of both arms on a synthetic store
    pairs = synth(n_keys=500, n_pairs=700, nv=500, seed=3)
    job = dict(sid="selftest/x/run", pairs=pairs, arm="O-BCH", f=14,
               evals=[(None, 38.05, ["k1", "k4"]), (None, 128.0, ["k8", "k16"])])
    r = run_job(job)
    for x in r["res"]:
        print("O-BCH f14", "%.0f" % x["B"], x["cond"], x.get("bch"), x.get("exact"), "/", x.get("n"), x.get("diag"),
              "cells %s <= budget %d" % (x.get("cells"), x["budget"]))
        assert x.get("infeasible") or x["cells"] <= x["budget"]
    job = dict(sid="selftest/x/run", pairs=pairs, arm="O-LDPC", f=14,
               evals=[(5 / 6, 38.05, ["k1", "k2"]), (1 / 3, 128.0, ["k8"]), (0.25, 512.0, ["k16"])])
    r = run_job(job)
    for x in r["res"]:
        print("O-LDPC f14", "%.0f" % x["B"], x["cond"], x["R"], x.get("exact"), "/", x.get("n"), x.get("diag"),
              "cells %s <= budget %d" % (x.get("cells"), x["budget"]))
        assert x.get("infeasible") or x["cells"] <= x["budget"]
    print("selftest ok (%.0f s)" % (time.time() - t0))


def timing():
    pairs = synth(n_keys=4000, n_pairs=6000, nv=5000, seed=11)
    for arm, R, B, c in (("O-BCH", None, 54.0, "k8"), ("O-LDPC", 1 / 3, 108.0, "k8"), ("O-LDPC", 0.25, 431.0, "k16"),
                         ("O-BCH", None, 32.0, "k1")):
        t = time.time()
        r = run_job(dict(sid="selftest/t/0", pairs=pairs, arm=arm, f=14, evals=[(R, B, [c])]))
        x = r["res"][0]
        print("timing %s R %s B %.0f %s: %.1f s, exact %s / %s" % (arm, R, B, c, time.time() - t, x.get("exact"),
                                                                   x.get("n")))


if __name__ == "__main__":
    mode = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    args = [a for i, a in enumerate(sys.argv[2:]) if not a.startswith("--") and sys.argv[1 + i] != "--workers"]
    if mode == "selftest":
        selftest()
    elif mode == "timing":
        timing()
    elif mode == "dev":
        stores = E.load_stores(args[0])
        jobs = make_jobs(stores, dev_need())
        print(len(jobs), "jobs", flush=True)
        pool_run(jobs, workers, args[1])
    elif mode == "dev2":
        stores = E.load_stores(args[0])
        jobs = make_jobs(stores, dev2_need(args[1]))
        print(len(jobs), "jobs", flush=True)
        pool_run(jobs, workers, args[2])
    elif mode == "choose":
        choose(args[0].split(","), args[1])
    elif mode == "plan":
        plan(args[0], args[1])
    elif mode == "run":
        stores = E.load_stores(args[0])
        jobs = make_jobs(stores, load_need(args[1]))
        print(len(jobs), "jobs", flush=True)
        pool_run(jobs, workers, args[2])
    elif mode == "extend":
        extend(args[0], args[1].split(","), args[2].split(","), args[3])
