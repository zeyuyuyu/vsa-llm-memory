"""B63 checks (prereg/B63_online_directory_prereg.md, 'Checks'): second scorer by plain loops, cells audit, N* by a second
rule, insertion audit, sample reading by the AI-assisted process.

python3 b63_check.py scorer CHOICE.json TEST.json[,...] OUT.json       # local re-run of sampled cells, two readers
python3 b63_check.py bits TEST.json[,...] OUT.json                     # cells audit of every test evaluation
python3 b63_check.py nstar CHOICE.json TEST.json[,...] VERDICT.json    # N* by a second rule
python3 b63_check.py insert PLAN.json[,...] OUT.json                   # insertion audit on every test store
python3 b63_check.py handread CHOICE.json VERDICT.json OUT.txt         # sample reading (failed keys one step below N*)
"""
import collections
import gzip
import hashlib
import json
import math
import sys

import numpy as np

import b38_eq as E
import b63_online as O

MASK = 0xFFFFFFFFFFFFFFFF


def jload(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


# ------------------------------------------------------------------ independent hash and reader (no b63_online code)
def h2(k, j):
    d = hashlib.sha256(("b63ck:" + k + "\x00" + str(j)).encode("utf-8")).digest()
    a = 0
    for t in range(7, -1, -1):
        a = (a << 8) | d[t]
    b = 0
    for t in range(15, 7, -1):
        b = (b << 8) | d[t]
    return a, b


def sm(x):
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def cands(k, j, m, f):
    a, b = h2(k, j)
    fp = 1 + b % (2 ** f - 1)
    i1 = a % m
    i2 = (sm(fp) % m - i1) % m
    return i1, i2, fp


def plain_reader(pairs, m, f, P, lastbit, bits, keys, maxj):
    """Explicit loops over slots, bits and candidates; returns {key: list of value indices}."""
    w = f + P
    slots = {}
    for b in range(m):
        for e in range(4):
            o = (b * 4 + e) * w
            fp = 0
            for t in range(f):
                fp = fp * 2 + int(bits[o + t])
            last = int(bits[o + f]) if lastbit else 1
            v = 0
            for t in range(f + lastbit, w):
                v = v * 2 + int(bits[o + t])
            slots[(b, e)] = (fp, last, v)
    out = {}
    for k in keys:
        g = []
        for j in range(maxj + 1):
            i1, i2, fp = cands(k, j, m, f)
            cs = [(i1, e) for e in range(4)]
            if i2 != i1:
                cs += [(i2, e) for e in range(4)]
            hit = [c for c in cs if slots[c][0] == fp]
            if len(hit) >= 2:
                g.append(-1)
                break
            if not hit:
                ds = []
                for c in cs:
                    x, cnt = slots[c][0] ^ fp, 0
                    while x:
                        cnt += x & 1
                        x >>= 1
                    ds.append(cnt)
                best = min(ds)
                hit = [c for c, dd in zip(cs, ds) if dd == best]
                if len(hit) >= 2:
                    g.append(-1)
                    break
            c = hit[0]
            g.append(slots[c][2])
            if slots[c][1]:
                break
        out[k] = g
    return out


def own_want(pairs, keys, vocab_index):
    byk = collections.defaultdict(dict)
    for k, v, j in pairs:
        byk[k][j] = vocab_index[v]
    return {k: [byk[k][j] for j in sorted(byk[k])] for k in keys}


def decoded_bits(S38, ck, arm, R, B, cond, needs):
    """Re-runs one evaluation (b63_online.evaluate's sequence) and returns (decoded logical bits, failed list)."""
    total = int(B * S38.n)
    bits = ck.bits(S38)
    L = len(bits)
    if arm == "O-BCH":
        mtc = E.parse_cond(cond)["match"]
        nh = E.NEXT_HARDER.get(mtc)
        n_, k_, t_ = E.bch_choose(L, needs, total, E.parse_cond(mtc)["eps"], E.IID[nh] if nh else None)
        lay = E.ecc_layout(L, n_, k_, t_, total)
        coded = E.bch_encode_stream(bits, lay)
        cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
        tag = "O-BCH|%s" % json.dumps([ck.f, n_, k_, t_])
        ch = E.Channel(S38.sid, tag, B, len(cells), total)
        sig = ch.apply(cells, cond)
        rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, cond, "dec"))
        dec, _ = E.bch_decode_stream(coded, E.rep_sum(sig, lay["counts"]), lay, rng)
    else:
        lay = E.ldpc_layout(L, R, total)
        coded, keep = E.ldpc_encode_stream(bits, lay)
        cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
        tag = "O-LDPC|%s" % json.dumps([ck.f, R])
        ch = E.Channel(S38.sid, tag, B, len(cells), total)
        sig = ch.apply(cells, cond)
        rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, cond, "dec"))
        dec, _ = E.ldpc_decode_stream(E.rep_sum(sig, lay["counts"]), lay, keep, E.parse_cond(cond)["eps"], rng)
        dec = dec[:L]
    return dec, bits


def scorer(choice_path, test_paths, out_path):
    ch = json.load(open(choice_path))
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    rows = []
    for p in test_paths:
        for r in jload(p):
            for x in r["res"]:
                if not x.get("infeasible") and [x["f"], x["R"]] == O.design_at(ch, x["arm"], x["cond"], O.gidx(x["B"])):
                    rows.append((r["sid"], x))
    # sample: both arms x k1, k4, k8, k16; per (arm, cond) one large, one medium, one small store at a failing or
    # near-N* budget (the budget with the most failed keys among those with >= 99% exact, else the largest failing)
    by = collections.defaultdict(list)
    for sid, x in rows:
        n = len(stores[sid])
        fam = "large" if n > 5000 else ("medium" if n > 1000 else "small")
        by[(x["arm"], x["cond"], fam)].append((sid, x))
    pick = []
    rng = np.random.default_rng(631)
    for arm in O.ARMS:
        for c in ("k1", "k4", "k8", "k16"):
            for fam in ("large", "medium", "small"):
                cand = by.get((arm, c, fam), [])
                if not cand:
                    continue
                fails = [t for t in cand if t[1]["failed"]]
                pool = fails if fails else cand
                pick.append(pool[int(rng.integers(len(pool)))])
    res = []
    for sid, x in pick:
        pairs = stores[sid]
        S38 = O.slim(E.Store38(sid, pairs))
        ck = O.Cuckoo(S38, x["f"])
        needs = ck.needs(S38) if x["arm"] == "O-BCH" else None
        dec, _ = decoded_bits(S38, ck, x["arm"], x["R"], x["B"], x["cond"], needs)
        got1 = ck.read(S38, dec)
        f1 = E.exact(got1, S38.want)
        s = S38.s
        got2 = plain_reader(pairs, ck.m, ck.f, ck.P, s.lastbit, dec, s.keys, s.maxj)
        wv = own_want(pairs, s.keys, s.vi)
        f2 = [i for i, k in enumerate(s.keys) if got2[k] != wv[k]]
        res.append(dict(sid=sid, arm=x["arm"], f=x["f"], R=x["R"], B=x["B"], cond=x["cond"], box=x["failed"],
                        reader1=f1, reader2=f2, same_bits_identical=f1 == f2, box_vs_local=f1 == x["failed"]))
        print("%-26s %-6s B %6.0f %-4s box %3d local1 %3d local2 %3d  readers identical %s  box=local %s" % (
            sid, x["arm"], x["B"], x["cond"], len(x["failed"]), len(f1), len(f2), f1 == f2, f1 == x["failed"]),
            flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("second scorer: %d / %d cells identical on the same decoded bits; box vs local failed lists identical in %d / %d"
          % (sum(r["same_bits_identical"] for r in res), len(res), sum(r["box_vs_local"] for r in res), len(res)))


def bits_audit(test_paths, out_path):
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    meta = {}
    for sid, pairs in stores.items():
        vocab = sorted({v for _, v, _ in pairs})
        nv = len(vocab)
        b = max(1, math.ceil(math.log2(nv))) if nv > 1 else 1
        lastbit = 1 if max(j for _, _, j in pairs) > 0 else 0
        meta[sid] = (len(pairs), lastbit + b)
    n_ev = over = badL = 0
    share = []
    for p in test_paths:
        for r in jload(p):
            n, P = meta[r["sid"]]
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                n_ev += 1
                if x["cells"] > int(x["B"] * n):
                    over += 1
                if x["L"] != x["m"] * 4 * (x["f"] + P):
                    badL += 1
                share.append((x["m"] * 4 * x["f"] + x["empty"] * P) / x["L"])
    out = dict(evaluations=n_ev, over_budget=over, logical_bits_mismatch=badL,
               fp_and_empty_share=[float(np.min(share)), float(np.median(share)), float(np.max(share))])
    json.dump(out, open(out_path, "w"), indent=1)
    print(out)


def nstar2(choice_path, test_paths, verdict_path):
    """Second rule: pooled share per budget over the 50 stores; N* = the smallest evaluated budget g such that no
    evaluated budget >= g is below 99.5% (written without b58_verdict / b63_online.nstar)."""
    ch = json.load(open(choice_path))
    V = json.load(open(verdict_path))
    nst = len(E.load_stores("b34_out/test_stores.json.gz"))
    pool = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in test_paths:
        for r in jload(p):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                g = O.gidx(x["B"])
                if [x["f"], x["R"]] != O.design_at(ch, x["arm"], x["cond"], g):
                    continue
                pool[(x["arm"], x["cond"])][g][r["sid"]] = (len(x["failed"]), x["n"])
    mism = 0
    tot = 0
    for (arm, c), byg in pool.items():
        ev = {g: 1.0 - sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for g, d in byg.items()
              if len(d) == nst}
        gs = sorted(ev)
        ns = None
        for g in gs:
            if all(ev[h] >= 0.995 for h in gs if h >= g):
                ns = g
                break
        mine = None if ns is None else round(O.GRID[ns], 6)
        theirs = V["nstar"]["%s|%s" % (arm, c)]
        theirs = None if theirs is None else round(theirs, 6)
        tot += 1
        if mine != theirs:
            mism += 1
            print("MISMATCH", arm, c, mine, theirs)
    print("N* second rule: %d / %d mismatches" % (mism, tot))
    return mism, tot


def insert_audit(plan_paths, out_path):
    fs = set()
    for p in plan_paths:
        for (arm, f, R, gi), cs in O.load_need(p).items():
            fs.add(f)
    res = []
    for sid, pairs in E.load_stores("b34_out/test_stores.json.gz"):
        S38 = O.slim(E.Store38(sid, pairs))
        for f in sorted(fs):
            ck = O.Cuckoo(S38, f)
            placed = int((ck.rows >= 0).sum())
            ok_place = True
            for b, e in zip(*np.nonzero(ck.rows >= 0)):
                i = int(ck.rows[b, e])
                i1, i2, fp = cands(S38.s.rk[i], int(S38.s.rj[i]), ck.m, f)
                if b not in (i1, i2) or ck.fps[b, e] != fp:
                    ok_place = False
            half = O.Cuckoo(S38, f, upto=S38.n // 2)
            cont = O.Cuckoo(S38, f, state=(half.fps.copy(), half.rows.copy(), half.rng, list(half.lost), half.done))
            online = bool(np.array_equal(cont.fps, ck.fps) and np.array_equal(cont.rows, ck.rows))
            res.append(dict(sid=sid, f=f, n=S38.n, m=ck.m, load=S38.n / (4 * ck.m), placed=placed, lost=len(ck.lost),
                            placement_ok=ok_place, sum_ok=placed + len(ck.lost) == S38.n, online=online))
    json.dump(res, open(out_path, "w"), indent=1)
    bad = [r for r in res if not (r["placement_ok"] and r["sum_ok"] and r["online"])]
    print("insertion audit: %d (store, f) builds, %d bad, lost total %d, load %.3f-%.3f" % (
        len(res), len(bad), sum(r["lost"] for r in res), min(r["load"] for r in res), max(r["load"] for r in res)))


def handread(choice_path, verdict_path, test_paths, out_path, per_cell=6):
    """Failed keys one grid step below N* for OT-best at k1, k8, k16: cause of each."""
    ch = json.load(open(choice_path))
    V = json.load(open(verdict_path))
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    rows = collections.defaultdict(dict)
    for p in test_paths:
        for r in jload(p):
            for x in r["res"]:
                if not x.get("infeasible"):
                    rows[(x["arm"], x["cond"], O.gidx(x["B"]), json.dumps([x["f"], x["R"]]))][r["sid"]] = x
    out = ["# B63 sample reading by the AI-assisted process (not independent): failed keys one grid step below N* of",
           "# OT-best at k1, k8, k16, re-run locally on the same channel draw; cause per failed key.", ""]
    for c in ("k1", "k8", "k16"):
        ns = {a: V["nstar"]["%s|%s" % (a, c)] for a in O.ARMS}
        arm = min(O.ARMS, key=lambda a: (O.NG if ns[a] is None else O.gidx(ns[a])))
        gi = O.gidx(ns[arm]) - 1
        f, R = O.design_at(ch, arm, c, gi)
        d = rows.get((arm, c, gi, json.dumps([f, R])), {})
        out.append("== %s: OT-best = %s, N* %s; reading B = %.0f (one step below), design f %s R %s" % (
            c, arm, ns[arm], O.GRID[gi], f, R))
        fails = [(sid, x) for sid, x in d.items() if x["failed"]]
        fails.sort(key=lambda t: -len(t[1]["failed"]))
        out.append("   failed keys pooled: %d of %d; stores with failures: %d" % (
            sum(len(x["failed"]) for x in d.values()), sum(x["n"] for x in d.values()), len(fails)))
        cause_tot = collections.Counter()
        for sid, x in fails[:3]:
            pairs = stores[sid]
            S38 = O.slim(E.Store38(sid, pairs))
            ck = O.Cuckoo(S38, f)
            needs = ck.needs(S38) if arm == "O-BCH" else None
            dec, true_bits = decoded_bits(S38, ck, arm, R, O.GRID[gi], c, needs)
            s = S38.s
            m, w = ck.m, ck.w
            where = ck.slot_of_rows()
            arr = dec[:m * 4 * w].reshape(m, 4, w).astype(np.int64)
            tb = true_bits.reshape(m, 4, w).astype(np.int64)
            fpr = (arr[:, :, :f] * (1 << np.arange(f - 1, -1, -1))[None, None, :]).sum(2)
            out.append("   store %s (n %d, m %d, lost %d): %d failed; residual logical bit errors %.4f" % (
                sid, s.n, m, len(ck.lost), len(x["failed"]), float((dec[:len(true_bits)] != true_bits).mean())))
            for a in x["failed"][:per_cell]:
                k = s.keys[a]
                cause = None
                for j in range(s.count[k]):
                    i1, i2, fp = O.buckets_of(k, j, m, f)
                    cs = [(i1, e) for e in range(4)] + ([(i2, e) for e in range(4)] if i2 != i1 else [])
                    i = s.row_of[(k, j)]
                    own = where.get(i)
                    hits = [cc for cc in cs if fpr[cc] == fp]
                    if own is None:
                        cause = "lost insertion (j %d)" % j
                        break
                    own_ok = fpr[own] == fp
                    pay_err = int((arr[own][f:] != tb[own][f:]).sum())
                    if len(hits) >= 2:
                        cause = "fingerprint false match at j %d (%d exact matches, own slot %s)" % (
                            j, len(hits), "matches" if own_ok else "corrupted")
                        break
                    if not own_ok:
                        cause = "own fingerprint corrupted at j %d (%d bit errors), nearest-slot fallback %s" % (
                            j, int((arr[own][:f] != tb[own][:f]).sum()), "chose another slot or tied" if True else "")
                        break
                    if pay_err:
                        cause = "exact match, payload (last flag / value) has %d bit errors at j %d" % (pay_err, j)
                        break
                if cause is None:
                    cause = "read past the last value (last flag) or other"
                cause_tot[cause.split(" at j")[0].split(" (")[0]] += 1
                out.append("     key %-40s values %d: %s" % (k[:40], s.count[k], cause))
        out.append("   causes (sampled keys): %s" % dict(cause_tot))
        out.append("")
    open(out_path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "scorer":
        scorer(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
    elif mode == "bits":
        bits_audit(sys.argv[2].split(","), sys.argv[3])
    elif mode == "nstar":
        nstar2(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
    elif mode == "insert":
        insert_audit(sys.argv[2].split(","), sys.argv[3])
    elif mode == "handread":
        handread(sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5])
