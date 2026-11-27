"""B58 part (a) (prereg/B58_protected_store_prereg.md): the VSA store (B38's V-last) protected by the tables' channel
codes, and the E-LDPC re-score of part (c).

python3 b58_store.py selftest
python3 b58_store.py dev1   OUT.json [--workers N]              # pass 1 (octave budgets), dev stores, i.i.d.
python3 b58_store.py dev2   DEV1.json OUT.json [--workers N]    # pass 2 (quarter steps near pass-1 N*)
python3 b58_store.py choose DEV1.json,DEV2.json CHOICE.json
python3 b58_store.py plan   CHOICE.json PLAN.json [--iid-only]
python3 b58_store.py run    STORES.json.gz PLAN.json OUT.json [--workers N] [--cap 40]
python3 b58_store.py extend CHOICE.json TEST1.json[,TEST2...] PLAN.json [--iid-only]
python3 b58_store.py rescore_plan PLAN.json                     # E-LDPC (part c3): B38's evaluated budgets
python3 b58_store.py rescore STORES.json.gz PLAN.json OUT.json [--workers N]
python3 b58_store.py rescore_extend TEST1.json[,...] PLAN.json

Store = b38_eq.vsa_build(store, b, "last") at information budget b (info bits per stored pair); its S d sign bits
(shard-major) are the logical bit string. P-LDPC: b38_eq.ldpc_layout / ldpc_encode_stream (IRA, <= 1,024 info bits per
codeword, rate R), copies r / r + 1 filling floor(B n) cells, soft-combined LLR at the matched rate, B38's second
decoder OWN = b38_check._sp_decode run batched over codewords (b58_bp.sp_decode_batch, bit-identical; 60
iterations), the decoder part (c)'s registered rule selects. P-BCH: b38_eq.ecc_layout / bch_encode_stream with (n, k, t) by the
tables' closed-form rule (expected number of failed stored keys; a key needs every bit of its shard), B38's
errors-and-erasures decoder. Decoded bits -> +-1 -> the V-last read. Shards are decoded and read in order; a shard
whose decoded bits equal the written bits gives the store's noise-free read (cached; exact, integer similarities).
Censoring: an evaluation on a store stops once it has `cap` failed probed keys (dev 7, test 40).
"""
import collections
import gzip
import json
import math
import os
import sys
import time

import numpy as np

import b38_eq as E
import b58_bp as NB

GRID = E.GRID
INFO_GI = (13, 14, 15, 16)                                     # b = 76.1, 90.5, 107.6, 128.0 info bits per pair
INFO = [GRID[i] for i in INFO_GI]
VARS = ("P-LDPC", "P-BCH")
DEV1_GI = (14, 16, 20, 24, 28, 32, 36)                         # 90.5, 128, 256, 512, 1024, 2048, 4096
IIDC = list(E.IID)
DMG = ["%s-%s-%s" % (g, lay, m) for g in ("lines", "clust") for lay in ("il", "nat") for m in E.DAMAGE_EPS]
DEV_CAP, TEST_CAP = 7, 40


def gidx(B):
    return min(range(len(GRID)), key=lambda i: abs(GRID[i] - B))


def designs(var):
    """Design list in registered order: b ascending, then R in E.RATES order (P-LDPC); b ascending (P-BCH)."""
    if var == "P-LDPC":
        return [[b, R] for b in INFO for R in E.RATES]
    return [[b, None] for b in INFO]


# =============================================================================================== store cache
class PStore:
    """One store at one information budget b: the V-last bundle, its bits, the noise-free read, probed keys by shard,
    stored-key weights per shard."""

    def __init__(self, S38, b):
        self.S38, self.b = S38, b
        bd = E.vsa_build(S38, b, "last")
        self.bd = bd
        self.S, self.d = bd["S"], bd["d"]
        self.L = self.S * self.d
        self.bits = (bd["cells"] > 0).astype(np.uint8)          # logical bit string (shard-major)
        rd = E.vsa_read(S38, bd, bd["cells"])
        self.got0 = E.vsa_rule(rd, "last")
        self.qsh = bd["shard_k"][S38.qk]                         # shard of each probed key
        self.keys_of = collections.defaultdict(list)
        for a, s in enumerate(self.qsh):
            self.keys_of[int(s)].append(a)
        allk = S38.s.cd["allk"]
        stored = set(S38.s.count)
        w = np.zeros(self.S, np.int64)
        for i, k in enumerate(allk):
            if k in stored:
                w[bd["shard_k"][i]] += 1
        self.w = w                                               # stored keys per shard (closed-form weights)

    def read_keys(self, dec, shards, keys):
        """V-last read of probed keys `keys`, each from its shard's decoded bits (one batch; same arithmetic as
        read_shard / b38_eq.vsa_read: integer similarities)."""
        S38, d = self.S38, self.d
        sh = np.asarray(shards)
        rows = (dec.reshape(self.S, d)[sh].astype(np.int8) * 2 - 1).astype(np.float32)
        qk = S38.qk[keys]
        X = rows * S38.Kc[qk, :d]
        cnt = np.maximum((X != 0).sum(1), 1).astype(np.float32)
        Vn = S38.vf(d)
        Lf = S38.L[:d].astype(np.float32)
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
        for a in range(nq):
            g = []
            for j in range(S38.J):
                g.append(int(best[a, j]))
                if tag[a, j]:
                    break
            out.append(g)
        return out

    def read_shard(self, s, dec_bits_shard, keys):
        """V-last read (b38_eq.vsa_read / vsa_rule, kind 'last') of probed keys `keys` from shard s's decoded bits."""
        S38, d = self.S38, self.d
        row = (dec_bits_shard.astype(np.int8) * 2 - 1).astype(np.float32)
        qk = S38.qk[keys]
        X = row[None, :] * S38.Kc[qk, :d]
        cnt = np.maximum((X != 0).sum(1), 1).astype(np.float32)
        Vn = S38.vf(d)
        Lf = S38.L[:d].astype(np.float32)
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
        for a in range(nq):
            g = []
            for j in range(S38.J):
                g.append(int(best[a, j]))
                if tag[a, j]:
                    break
            out.append(g)
        return out


_PS = {}


def pstore(S38, b):
    key = (S38.sid, round(b, 4))
    if key not in _PS:
        _PS.clear() if len(_PS) > 4 else None
        _PS[key] = PStore(S38, b)
    return _PS[key]


# =============================================================================================== BCH closed form (store)
def store_bch_choose(L, d, w, cells, eps, eps_next):
    """b38_eq.bch_choose's rule with the store's needs: shard s (weight w[s] stored keys) needs bits [s d, (s+1) d).
    Objective = expected number of failed stored keys (a key fails if any codeword holding a bit of its shard fails);
    below 1e-6 compared by the log of the summed (key, codeword) failure probabilities; ties: the same at eps_next,
    then the shorter code."""
    if E.BCH_DESIGNS is None:
        E.BCH_DESIGNS = E.bch_designs()
    S = len(w)
    best = None

    def obj(lay, e):
        if e <= 0:
            return 0.0, -np.inf
        kk = lay["sizes"][0]
        ncw = len(lay["sizes"])
        cst = np.r_[0, np.cumsum(lay["clen"])]
        r = int(lay["counts"].min())
        plus = np.r_[0, np.cumsum(lay["counts"] > r)]
        lf = np.array([E.bch_cw_logfail(lay["clen"][i], lay["t"], r, int(plus[cst[i + 1]] - plus[cst[i]]), e)
                       for i in range(ncw)])
        lg1 = np.log1p(-np.minimum(np.exp(lf), 1 - 1e-16))
        c0 = (np.arange(S) * d) // kk
        c1 = ((np.arange(S) + 1) * d - 1) // kk
        cum = np.r_[0.0, np.cumsum(lg1)]
        ok = cum[c1 + 1] - cum[c0]
        Ev = float(-(w * np.expm1(ok)).sum())
        lse = []
        for s in range(S):
            if w[s]:
                lse.append(np.logaddexp.reduce(lf[c0[s]:c1[s] + 1]) + math.log(w[s]))
        return Ev, (float(np.logaddexp.reduce(lse)) if lse else -np.inf)

    def kk_(e, lg):
        return (round(e, 9), round(lg, 6)) if e < 1e-6 else (round(e, 9), 0.0)
    for n, k, t in E.BCH_DESIGNS:
        lay = E.ecc_layout(L, n, k, t, cells)
        if lay is None:
            continue
        e0, l0 = obj(lay, eps)
        e1, l1 = obj(lay, eps_next) if eps_next else (0.0, -np.inf)
        key = (kk_(e0, l0), kk_(e1, l1), n)
        if best is None or key < best[0]:
            best = (key, (n, k, t))
    return None if best is None else best[1]


# =============================================================================================== one evaluation
def llr0_from_sums(sums, keep, eps, ncw, N):
    """OWN's channel LLR of bit = 0 exactly as B38 (votes x Le, probabilities clipped to [1e-15, 0.5]; shortened bits
    known 0)."""
    e = min(max(eps, 1e-6), 0.4999)
    Le = math.log((1 - e) / e)
    full = np.zeros(ncw * N, np.float64)
    full[keep] = sums * Le
    full[~keep] = -60.0
    pr = np.clip(1.0 / (1.0 + np.exp(np.minimum(np.abs(full), 60.0))), 1e-15, 0.5)
    return np.where(full > 0, -1.0, 1.0) * np.log((1 - pr) / pr)


_NBD = {}


def nb_decoder(K, R):
    key = (K, round(R, 4))
    if key not in _NBD:
        _NBD[key] = NB.NbDecoder(E.ira_code(K, R)["H"], E.LDPC_ITERS)
    return _NBD[key]


def bch_decode_one(code, sm, tr, s_, t, rng):
    """b38_eq.bch_decode_stream's per-codeword logic (identical decisions and rng use)."""
    hard = (sm > 0).astype(np.uint8)
    era = sm == 0
    e = int(((hard != tr) & ~era).sum())
    se = int(era.sum())
    if 2 * e + se <= 2 * t:
        return tr[:s_], 0
    cands = []
    for fill in ((0, 1) if se else (0,)):
        w = hard.copy()
        w[era] = fill
        ok, msg, ne = code.decode(w)
        if ok:
            cands.append((ne, fill, msg))
    if cands:
        cands.sort(key=lambda x: (x[0], x[1]))
        return cands[0][2], 1
    w = hard.copy()
    w[era] = rng.integers(0, 2, se)
    return w[:s_], 2


def evaluate(S38, var, b, R, B, cond, cap, nbch_cache=None):
    """Returns dict with exact / failed / censored / accounting for one (variant, b, design, budget, condition)."""
    ps = pstore(S38, b)
    total = int(B * S38.n)
    pc = E.parse_cond(cond)
    m = pc["match"]
    nq = len(S38.want)
    acc = dict(L=ps.L, S=ps.S, d=ps.d, budget=total)
    if var == "P-LDPC":
        lay = E.ldpc_layout(ps.L, R, total)
        if lay is None:
            return dict(infeasible=True, **acc)
        coded, keep = E.ldpc_encode_stream(ps.bits, lay)
        counts = lay["counts"]
        design = [b, R]
        acc.update(K=lay["K"], ncw=lay["ncw"], Nc=lay["Nc"], cells=int(counts.sum()), rmin=int(counts.min()),
                   rmax=int(counts.max()))
    else:
        nh = E.NEXT_HARDER.get(m)
        des = store_bch_choose(ps.L, ps.d, ps.w, total, E.IID[m], E.IID[nh] if nh else None)
        if des is None:
            return dict(infeasible=True, **acc)
        n_, k_, t_ = des
        lay = E.ecc_layout(ps.L, n_, k_, t_, total)
        coded = E.bch_encode_stream(ps.bits, lay)
        counts = lay["counts"]
        design = [b, list(des)]
        acc.update(bch_n=n_, bch_k=k_, bch_t=t_, ncw=len(lay["sizes"]), Nc=lay["Nc"], cells=int(counts.sum()),
                   rmin=int(counts.min()), rmax=int(counts.max()))
    cells = np.repeat((coded.astype(np.int8) * 2 - 1), counts)
    tag = "%s|%s" % (var, json.dumps([round(b, 4), R]))
    ch = E.Channel(S38.sid, tag, B, len(cells), total)
    sig = ch.apply(cells, cond)
    sums = E.rep_sum(sig, counts)
    rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, cond, "dec"))
    dec = np.zeros(ps.L, np.uint8)
    done_upto = 0                                                 # info bits decoded so far (prefix)
    failed, ncw_fail, resid = [], 0, 0
    d = ps.d
    if var == "P-LDPC":
        Kc_, N_ = lay["K"], lay["N"]
        llr = llr0_from_sums(sums, keep, pc["eps"], lay["ncw"], N_).reshape(lay["ncw"], N_)
        Hm = E.ira_code(Kc_, R)["H"]                               # OWN (b38_check._sp_decode), batched
        next_cw = 0
    else:
        code = E.bch_code(lay["n"], lay["k"], lay["t"])
        cst = np.r_[0, np.cumsum(lay["clen"])]
        ist = np.r_[0, np.cumsum(lay["sizes"])]
        next_cw = 0
    censored = False
    CH = 8                                                        # shards per decode / read batch
    for s0 in range(0, ps.S, CH):
        s1 = min(ps.S, s0 + CH)
        need = s1 * d
        if var == "P-LDPC":
            last_cw = min(lay["ncw"] - 1, (need - 1) // Kc_)
            if last_cw >= next_cw:
                out, conv, _ = NB.sp_decode_batch(Hm, llr[next_cw:last_cw + 1], E.LDPC_ITERS)
                for i, w in enumerate(range(next_cw, last_cw + 1)):
                    lo = w * Kc_
                    hi = min(ps.L, lo + Kc_)
                    dec[lo:hi] = out[i, :hi - lo]
                    ncw_fail += int(not conv[i])
                next_cw = last_cw + 1
        else:
            while next_cw < len(lay["sizes"]) and ist[next_cw] < need:
                w = next_cw
                msg, st = bch_decode_one(code, sums[cst[w]:cst[w + 1]], coded[cst[w]:cst[w + 1]],
                                         lay["sizes"][w], lay["t"], rng)
                dec[ist[w]:ist[w + 1]] = msg
                ncw_fail += int(st == 2)
                next_cw += 1
        noisy_keys, noisy_sh = [], []
        for s in range(s0, s1):
            keys = ps.keys_of.get(s, [])
            db = dec[s * d:(s + 1) * d]
            nerr = int((db != ps.bits[s * d:(s + 1) * d]).sum())
            resid += nerr
            if not keys:
                continue
            if nerr == 0:
                for a in keys:
                    if ps.got0[a] != S38.want[a]:
                        failed.append(int(a))
            else:
                noisy_keys += keys
                noisy_sh += [s] * len(keys)
        if noisy_keys:
            got = ps.read_keys(dec, noisy_sh, noisy_keys)
            for a, g in zip(noisy_keys, got):
                if g != S38.want[a]:
                    failed.append(int(a))
        if len(failed) >= cap:
            censored = True
            break
    s = s1 - 1
    return dict(design=design, exact=None if censored else nq - len(failed), n=nq, failed=sorted(failed),
                censored=censored, ncw_fail=ncw_fail, resid=resid, shards_done=s + 1, **acc)


def slim(S38):
    """Drop the float32 code arrays that b34_inmem.Store keeps (Store38 holds int8 copies; nothing in B58 reads them)."""
    for dct in (S38.s.cd, S38.s.st):
        for k in ("Kc", "Vc", "Pc", "mem", "b1"):
            if k in dct and k != "Pc":
                dct[k] = None
    return S38


def run_job(job):
    t0 = time.time()
    S38 = slim(E.Store38(job["sid"], job["pairs"]))
    out = []
    for var, b, R, B, conds in job["evals"]:
        for c in conds:
            r = evaluate(S38, var, b, R, B, c, job["cap"])
            r.update(var=var, b=b, R=R, B=B, cond=c)
            out.append(r)
    return dict(sid=job["sid"], n_pairs=S38.n, nq=len(S38.want), res=out, sec=time.time() - t0)


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done, last = [], time.time()
    with Pool(workers, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            done.append(r)
            print("done %-28s %5.0fs %d/%d" % (r["sid"], r["sec"], len(done), len(jobs)), flush=True)
            if time.time() - last > 300:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def done_set(paths):
    out = set()
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                out.add((r["sid"], x["var"], round(x["b"], 4), x["R"], gidx(x["B"]), x["cond"]))
    return out


def make_jobs(stores, need, cap, per_job=6, done=None):
    """need: {(var, b, R, gi): set(conds)} -> jobs per store grouped by (var, b) so the store cache is reused; `done`
    (resume): evaluations already in a finished file are skipped."""
    jobs = []
    for sid, pairs in stores:
        grp = collections.defaultdict(list)
        for (var, b, R, gi), cs in sorted(need.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] is None
                                                                         else kv[0][2], kv[0][3])):
            if done is not None:
                cs = {c for c in cs if (sid, var, round(b, 4), R, gi, c) not in done}
                if not cs:
                    continue
            grp[(var, b)].append((var, b, R, GRID[gi], sorted(cs)))
        for (var, b), ev in grp.items():
            for i in range(0, len(ev), per_job):
                jobs.append(dict(sid=sid, pairs=pairs, evals=ev[i:i + per_job], cap=cap))
    jobs.sort(key=lambda j: -len(j["pairs"]))
    return jobs


# =============================================================================================== dev / choose
def dev_need(gis, conds=None, only=None):
    need = collections.defaultdict(set)
    for var in VARS:
        for b, R in designs(var):
            for gi in gis:
                if only is not None and (var, gi) not in only:
                    continue
                if GRID[gi] < b:                                  # no code fits below the information budget
                    continue
                for c in (conds or IIDC):
                    if only is not None and c not in only[(var, gi)]:
                        continue
                    need[(var, b, R, gi)].add(c)
    return need


def aggregate(paths):
    """acc[var][design json][cond][gi] = [exact, n, censored stores]."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0, 0]))))
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                dk = json.dumps([round(x["b"], 4), x["R"]])
                a = acc[x["var"]][dk][x["cond"]][gidx(x["B"])]
                a[1] += x["n"]
                if x["censored"]:
                    a[2] += 1
                else:
                    a[0] += x["exact"]
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
    acc = aggregate(paths)
    nst = 12                                                       # dev stores
    ch = dict(designs={}, dev_curve={}, dev_nstar={}, censored={})
    for var in VARS:
        dl = [json.dumps([round(b, 4), R]) for b, R in designs(var)]
        ch["designs"][var], ch["dev_curve"][var], ch["dev_nstar"][var], ch["censored"][var] = {}, {}, {}, {}
        for c in IIDC:
            nh = E.NEXT_HARDER[c]
            des, cur, cens = {}, {}, {}
            gis = sorted({gi for dk in dl for gi in acc[var][dk][c]})
            for gi in gis:
                cands = []
                for o, dk in enumerate(dl):
                    a = acc[var][dk][c].get(gi)
                    if a is None or a[1] == 0:
                        continue
                    if a[2]:
                        cands.append((1, 0.0, 0.0, o, dk, None))
                        continue
                    fr = a[0] / a[1]
                    a2 = acc[var][dk][nh].get(gi) if nh else None
                    f2 = (a2[0] / a2[1]) if (a2 and a2[1] and not a2[2]) else 0.0
                    cands.append((0, -fr, -f2, o, dk, fr))
                if not cands:
                    continue
                cands.sort()
                if cands[0][0] == 0:
                    des[gi], cur[gi] = cands[0][4], cands[0][5]
                else:
                    cens[gi] = True
                    cur[gi] = None
            # budgets where every design was censored: the design chosen at the nearest evaluated budget with a choice
            for gi in cens:
                if des:
                    near = min(des, key=lambda g: (abs(g - gi), -g))
                    des[gi] = des[near]
            ch["designs"][var][c] = {str(g): json.loads(v) for g, v in des.items()}
            ch["dev_curve"][var][c] = {str(g): (None if v is None else round(v, 6)) for g, v in cur.items()}
            ns = nstar(cur)
            ch["dev_nstar"][var][c] = None if ns is None else GRID[ns]
            ch["censored"][var][c] = sorted(cens)
    json.dump(ch, open(out_path, "w"), indent=1)
    print("dev N* (cells per stored pair):")
    print("%-7s" % "", " ".join("%8s" % c for c in IIDC))
    for var in VARS:
        print("%-7s" % var, " ".join("%8s" % ("%.0f" % x if x else "-") for x in
                                     (ch["dev_nstar"][var].get(c) for c in IIDC)))
    for var in VARS:
        for c in IIDC:
            ns = ch["dev_nstar"][var][c]
            if ns:
                print("  %s %s design at N*: %s" % (var, c, ch["designs"][var][c][str(gidx(ns))]))
    return ch


def dev2_need(dev1_path):
    """Pass 2: every design at the quarter steps within 3 grid steps of each (var, cond)'s pass-1 N* (top 4 budgets
    if not reached)."""
    paths = dev1_path.split(",")
    tmp = paths[-1] + ".choice_tmp.json"
    ch = choose(paths, tmp)
    only = collections.defaultdict(set)
    for var in VARS:
        for c in IIDC:
            ns = ch["dev_nstar"][var][c]
            if ns is None:
                gis = range(len(GRID) - 4, len(GRID))
            else:
                g = gidx(ns)
                # compute note 2: P-LDPC only the 3 quarter steps below its pass-1 N* (P-BCH the registered +-3)
                gis = range(max(0, g - 3), g) if var == "P-LDPC" else range(max(0, g - 3), min(len(GRID), g + 4))
            if var == "P-LDPC" and os.environ.get("B58_NO_LDPC_PASS2") == "1":    # compute note 3
                continue
            for gi in gis:
                if gi in DEV1_GI:
                    continue
                only[(var, gi)].add(c)
    gis = sorted({gi for (_, gi) in only})
    need = dev_need(gis, only=only)
    return need


# =============================================================================================== test plan / extension
def design_at(ch, var, m, gi):
    """The dev design at budget gi; where dev did not evaluate gi, the nearest dev-evaluated budget's design (ties:
    the larger budget) among those that fit at gi (b / R <= B; compute note 2), else the nearest one."""
    d = ch["designs"][var].get(m, {})
    if not d:
        return designs(var)[0]
    if str(gi) in d:
        return d[str(gi)]
    order = sorted((int(g) for g in d), key=lambda g: (abs(g - gi), -g))
    for g in order:
        b, R = d[str(g)]
        if GRID[gi] >= b / (R if R else 0.99):
            return d[str(g)]
    return d[str(order[0])]


def window(ch, var, m):
    ns = ch["dev_nstar"][var].get(m)
    if ns is None:
        return list(range(len(GRID) - 5, len(GRID)))
    g = gidx(ns)
    return list(range(max(0, g - 4), min(len(GRID), g + 5)))


DMG_ONLY = "--dmg-only" in sys.argv


def conds_all(iid_only):
    if DMG_ONLY:                                                 # B58_DMG_GEOM=lines|clust: run order only (note 2)
        g = os.environ.get("B58_DMG_GEOM")
        return [c for c in DMG if not g or c.startswith(g + "-")]
    return IIDC + ([] if iid_only else DMG)


def plan(choice_path, out_path, iid_only=False):
    ch = json.load(open(choice_path))
    need = collections.defaultdict(set)
    for c in conds_all(iid_only):
        m = E.parse_cond(c)["match"]
        for var in VARS:
            for gi in window(ch, var, m):
                b, R = design_at(ch, var, m, gi)
                need[(var, b, R, gi)].add(c)
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(len(need), "(variant, design, budget) entries;", sum(len(v) for v in need.values()), "evaluations per store")


def test_curves(paths):
    """curve[var][cond][gi] = share (None if any store censored) using every evaluated design (one per budget)."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0])))
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                a = acc[x["var"]][x["cond"]][gidx(x["B"])]
                a[1] += x["n"]
                a[2] += bool(x["censored"])
                a[0] += 0 if x["censored"] else x["exact"]
    cur = {v: {c: {gi: (None if a[2] else a[0] / a[1]) for gi, a in d.items()} for c, d in dd.items()}
           for v, dd in acc.items()}
    return cur


def extend(choice_path, test_paths, out_path, iid_only=False):
    ch = json.load(open(choice_path))
    cur = test_curves(test_paths)
    tried = set()
    for p in test_paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                tried.add((x["var"], x["cond"], gidx(x["B"])))
    need = collections.defaultdict(set)
    nadd = 0
    for c in conds_all(iid_only):
        m = E.parse_cond(c)["match"]
        for var in VARS:
            curve = cur.get(var, {}).get(c, {})
            if not curve:
                continue
            ev = sorted(curve)
            ns = nstar(curve)
            top_ok = curve[max(ev)] is not None and curve[max(ev)] >= E.TARGET
            add = []
            if not top_ok and max(ev) < len(GRID) - 1:
                add = list(range(max(ev) + 1, min(len(GRID), max(ev) + 5)))
            elif ns is not None and ns == min(ev) and ns > 0 and GRID[ns - 1] >= INFO[0]:
                add = list(range(max(0, ns - 4), ns))
            elif ns is not None and ns > 0 and (ns - 1) not in curve and GRID[ns - 1] >= INFO[0]:
                below = [i for i in ev if i < ns]
                add = list(range((max(below) + 1) if below else max(0, ns - 4), ns))
            for gi in add:
                b, R = design_at(ch, var, m, gi)
                if GRID[gi] < b / (R if R else 0.99) or (var, c, gi) in tried:
                    continue
                need[(var, b, R, gi)].add(c)
                nadd += 1
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(nadd, "(variant, condition, budget) additions")
    return nadd


def load_need(path):
    return {tuple(k): set(v) for k, v in json.load(open(path))["need"]}


# =============================================================================================== E-LDPC re-score (c3)
RESCORE_CONDS = IIDC + DMG


def rescore_plan(out_path):
    """Every (store, budget, design, condition) at which B38's test evaluated E-LDPC in the 23 conditions."""
    res = json.load(gzip.open("results/b38_test.json.gz", "rt"))
    need = collections.defaultdict(set)
    for r in res:
        if r["arm"] != "E-LDPC":
            continue
        for x in r["res"]:
            if "cond" in x and x["cond"] in RESCORE_CONDS and not x.get("infeasible"):
                need[(r["sid"], gidx(x["B"]), json.dumps(x["design"]))].add(x["cond"])
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(len(need), "entries;", sum(len(v) for v in need.values()), "evaluations")


def own_decode_stream(sums, lay, keep, eps, rng):
    """b38_eq.ldpc_decode_stream's B38_LDPC_OWN path (b38_check._sp_decode per codeword, same LLRs), with the
    codewords decoded together by b58_bp.sp_decode_batch (bit-identical per codeword)."""
    code = E.ira_code(lay["K"], lay["R"])
    e = min(max(eps, 1e-6), 0.4999)
    Le = math.log((1 - e) / e)
    full = np.zeros(lay["ncw"] * lay["N"], np.float64)
    full[keep] = sums * Le
    full[~keep] = -60.0
    hard = (full > 0).astype(np.uint8)
    ties = full == 0
    hard[ties] = rng.integers(0, 2, int(ties.sum()))
    pr = np.clip(1.0 / (1.0 + np.exp(np.minimum(np.abs(full), 60.0))), 1e-15, 0.5)
    llr = np.where(hard == 0, 1, -1) * np.log((1 - pr) / pr)
    out, conv, _ = NB.sp_decode_batch(code["H"], llr.reshape(lay["ncw"], lay["N"]), E.LDPC_ITERS)
    return np.ascontiguousarray(out[:, :lay["K"]]).ravel(), int((~conv).sum())


class _SlimStore38(E.Store38):
    def __init__(self, sid, pairs):
        super().__init__(sid, pairs)
        slim(self)


def rescore_job(job):
    E.ldpc_decode_stream = own_decode_stream                    # OWN (b38_check._sp_decode), batched
    E.Store38 = _SlimStore38                                     # memory only (E-LDPC reads no code array)
    return E.run_job(job)


def rescore(stores_path, plan_path, out_path, workers):
    stores = dict(E.load_stores(stores_path))
    need = json.load(open(plan_path))["need"]
    by = collections.defaultdict(list)
    for (sid, gi, dj), cs in need:
        by[sid].append((GRID[gi], json.loads(dj), cs))
    jobs = []
    for sid, ev in by.items():
        for i in range(0, len(ev), 3):
            jobs.append(dict(sid=sid, pairs=stores[sid], arm="E-LDPC", evals=ev[i:i + 3]))
    jobs.sort(key=lambda j: -len(j["pairs"]))
    print(len(jobs), "rescore jobs", flush=True)
    from multiprocessing import Pool
    done, last = [], time.time()
    with Pool(workers, maxtasksperchild=20) as pool:
        for r in pool.imap_unordered(rescore_job, jobs):
            done.append(r)
            print("done %-28s %5.0fs %d/%d" % (r["sid"], r["sec"], len(done), len(jobs)), flush=True)
            if time.time() - last > 300:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))


def rescore_curves(paths):
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if "cond" not in x or x.get("infeasible"):
                    continue
                a = acc[x["cond"]][gidx(x["B"])]
                a[0] += x["exact"]
                a[1] += x["n"]
    return {c: {gi: a[0] / a[1] for gi, a in d.items()} for c, d in acc.items()}


def rescore_extend(test_paths, out_path):
    """B38's extension rule for the re-scored E-LDPC curves; designs = B38's dev choice (results/b38_choice.json)."""
    ch = json.load(open("results/b38_choice.json"))
    cur = rescore_curves(test_paths)
    stores = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    need = collections.defaultdict(set)
    nadd = 0
    for c in RESCORE_CONDS:
        m = E.parse_cond(c)["match"]
        curve = cur.get(c, {})
        if not curve:
            continue
        ev = sorted(curve)
        ns = nstar(curve)
        add = []
        if curve[max(ev)] < E.TARGET and max(ev) < len(GRID) - 1:
            add = list(range(max(ev) + 1, min(len(GRID), max(ev) + 5)))
        elif ns is not None and ns == min(ev) and ns > 0:
            add = list(range(max(0, ns - 4), ns))
        elif ns is not None and ns > 0 and (ns - 1) not in curve:
            below = [i for i in ev if i < ns]
            add = list(range((max(below) + 1) if below else max(0, ns - 4), ns))
        for gi in add:
            d = ch["designs"]["E-LDPC"][m].get(str(gi))
            if d is None:
                near = min((int(g) for g in ch["designs"]["E-LDPC"][m]), key=lambda g: (abs(g - gi), -g))
                d = ch["designs"]["E-LDPC"][m][str(near)]
            for sid in stores:
                need[(sid, gi, json.dumps(json.loads(d)))].add(c)
            nadd += 1
    json.dump(dict(need=[[list(k), sorted(v)] for k, v in need.items()]), open(out_path, "w"))
    print(nadd, "(condition, budget) additions")
    return nadd


# =============================================================================================== self-test
def selftest():
    rng = np.random.default_rng(1)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(700):
        k = "key%d" % rng.integers(0, 600)
        pairs.append((k, "val%d" % (i % 500), seen[k]))
        seen[k] += 1
    S38 = E.Store38("selftest/x/0", pairs)
    print("n", S38.n, "Nv", S38.Nv, "probed", len(S38.want))
    # 1. store_bch_choose == b38_eq.bch_choose with per-key needs (every stored key needs its shard's bits)
    for b in (76.1, 128.0):
        ps = PStore(S38, b)
        allk = list(S38.s.count)
        ki = {k: i for i, k in enumerate(S38.s.cd["allk"])}
        needs = [np.arange(ps.bd["shard_k"][ki[k]] * ps.d, (ps.bd["shard_k"][ki[k]] + 1) * ps.d) for k in allk]
        for B in (128.0, 256.0, 512.0):
            for eps, en in ((0.0208065, 0.05), (0.12263475, 0.27600645), (0.27600645, 0.374429)):
                a = store_bch_choose(ps.L, ps.d, ps.w, int(B * S38.n), eps, en)
                c = E.bch_choose(ps.L, needs, int(B * S38.n), eps, en)
                print("bch choose b=%.0f B=%.0f eps=%.3f: store %s per-key %s %s" % (b, B, eps, a, c,
                                                                                    "OK" if a == c else "DIFF"))
                assert a == c
    # 2. noiseless: every protected design reads exactly what the noise-free store reads; the read_shard path equals
    #    b38_eq.vsa_read
    for var in VARS:
        for b, R in designs(var)[:3]:
            r = evaluate(S38, var, b, R, 512.0, "k1", cap=10 ** 9)
            ps = pstore(S38, b)
            nf = len(S38.want) - len(E.exact(ps.got0, S38.want))
            print(var, b, R, "k1 exact", r["exact"], "noise-free store", nf, "resid", r["resid"], "cells/budget",
                  r["cells"] / r["budget"])
            assert r["exact"] == nf and r["resid"] == 0 and r["cells"] <= r["budget"]
    ps = pstore(S38, 76.1)
    for s in range(ps.S):
        keys = ps.keys_of.get(s, [])
        if keys:
            flip = ps.bits[s * ps.d:(s + 1) * ps.d].copy()
            flip[rng.random(ps.d) < 0.1] ^= 1
            g1 = ps.read_shard(s, flip, keys)
            sig = ps.bd["cells"].copy()
            sig[s * ps.d:(s + 1) * ps.d] = flip.astype(np.int8) * 2 - 1
            g2 = E.vsa_rule(E.vsa_read(S38, ps.bd, sig), "last")
            assert g1 == [g2[a] for a in keys], s
    print("read_shard == vsa_read on noisy shards: OK")
    # 3. noisy runs
    for var, b, R in (("P-LDPC", 90.5, 0.5), ("P-LDPC", 76.1, 1 / 3), ("P-BCH", 90.5, None)):
        for B, c in ((256.0, "k8"), (512.0, "k16"), (256.0, "lines-nat-k8"), (256.0, "lines-il-k8")):
            t0 = time.time()
            r = evaluate(S38, var, b, R, B, c, cap=10 ** 9)
            print(var, b, R, B, c, {k: r.get(k) for k in ("exact", "n", "ncw_fail", "resid", "bch_t", "rmin", "rmax")},
                  "%.1fs" % (time.time() - t0))
    print("selftest ok")


if __name__ == "__main__":
    mode = sys.argv[1]
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    iid_only = "--iid-only" in sys.argv
    if mode == "selftest":
        selftest()
    elif mode == "dev1":
        stores = E.load_stores("b34_out/dev_stores.json.gz")
        done = done_set(sys.argv[sys.argv.index("--resume") + 1].split(",")) if "--resume" in sys.argv else None
        jobs = make_jobs(stores, dev_need(DEV1_GI), DEV_CAP, done=done)
        print(len(jobs), "jobs", flush=True)
        pool_run(jobs, W, sys.argv[2])
    elif mode == "dev2":
        stores = E.load_stores("b34_out/dev_stores.json.gz")
        jobs = make_jobs(stores, dev2_need(sys.argv[2]), DEV_CAP)
        print(len(jobs), "jobs", flush=True)
        pool_run(jobs, W, sys.argv[3])
    elif mode == "choose":
        choose(sys.argv[2].split(","), sys.argv[3])
    elif mode == "plan":
        plan(sys.argv[2], sys.argv[3], iid_only)
    elif mode == "run":
        stores = E.load_stores(sys.argv[2])
        cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else TEST_CAP
        jobs = make_jobs(stores, load_need(sys.argv[3]), cap)
        print(len(jobs), "jobs", flush=True)
        pool_run(jobs, W, sys.argv[4])
    elif mode == "extend":
        extend(sys.argv[2], sys.argv[3].split(","), sys.argv[4], iid_only)
    elif mode == "rescore_plan":
        rescore_plan(sys.argv[2])
    elif mode == "rescore":
        rescore(sys.argv[2], sys.argv[3], sys.argv[4], W)
    elif mode == "rescore_extend":
        rescore_extend(sys.argv[2].split(","), sys.argv[3])
