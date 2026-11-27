"""B61 checks (prereg/B61_hd_protected_store_prereg.md, 'Checks'). Local machine.

python3 b61_check.py scorer CHOICE.json TEST1.json[,...] OUT.json       # second scorer (independent numpy / loops)
python3 b61_check.py bits TEST1.json[,...] OUT.json                     # cells <= floor(B n); share used; density
python3 b61_check.py nstar2 CHOICE.json VERDICT.json TEST1.json[,...]   # N* by a separately written rule
python3 b61_check.py handread CHOICE.json VERDICT.json TEST1.json[,...] OUT.txt   # sample reading (failed keys)
"""
import collections
import gzip
import json
import math
import sys

import numpy as np

import b38_eq as E
import b61_hd as H

GRID = E.GRID
NG = len(GRID)


def jl(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def want_from_pairs(S38, pairs):
    """Exact-read targets re-derived from the store's own pairs (not Store38.want)."""
    vi = {v: i for i, v in enumerate(S38.s.cd["vocab"])}
    by = collections.defaultdict(list)
    for k, v, j in pairs:
        by[k].append((j, vi[v]))
    return [[x for _, x in sorted(by[k])] for k in S38.s.keys]


# ------------------------------------------------------------------------------------------- independent SBC reads
def np_slot(st, yb, S38, a, j, cov=None, own=None, A=1.0, Cc=0.0):
    """Full candidate scores by numpy gathers (no pruning); returns (value, last, s0, s1)."""
    sh = int(st.qsh[a])
    base = sh * st.d
    o = (st.Kb[S38.qk[a]].astype(np.int64) + st.Pb[j]) % st.Lb
    blk = np.arange(st.nb, dtype=np.int64)[None, :] * st.Lb
    C0 = base + blk + (o[None, :] + st.Vb) % st.Lb
    C1 = base + blk + (o[None, :] + st.Vb + st.Lc[None, :]) % st.Lb
    out = []
    for Cm in (C0, C1):
        y = yb[Cm] > 0
        if cov is None:
            un = np.ones(y.shape, bool)
        else:
            co = cov[Cm] - (Cm == own[None, :]) if own is not None else cov[Cm]
            un = co <= 0
        na = (un & y).sum(1)
        ni = (un & ~y).sum(1)
        out.append(na * A - ni * Cc)
    s0, s1 = out
    if s1.max() > s0.max():
        return int(s1.argmax()), True, s0, s1
    return int(s0.argmax()), False, s0, s1


def np_sbc_eval(S38, pairs, lam, B, cond):
    st = H.SBCStore(S38, B, lam)
    ch = E.Channel(S38.sid, "H-SBC|%.4f" % lam, B, len(st.cells), st.total)
    yb = (ch.apply(st.cells, cond) > 0).astype(np.uint8)
    want = want_from_pairs(S38, pairs)
    nq = len(want)
    got1, items = [], []
    for a in range(nq):
        g, it = [], []
        for j in range(S38.J):
            v, last, _, _ = np_slot(st, yb, S38, a, j)
            g.append(v)
            it.append((j, v, last))
            if last:
                break
        got1.append(g)
        items.append(it)
    # explaining away, re-implemented
    blk = np.arange(st.nb, dtype=np.int64) * st.Lb
    cells_of = {}
    allc = []
    for a in range(nq):
        sh = int(st.qsh[a])
        for (j, v, last) in items[a]:
            o = (st.Kb[S38.qk[a]].astype(np.int64) + st.Pb[j]) % st.Lb
            c = sh * st.d + blk + (o + st.Vb[v] + (st.Lc if last else 0)) % st.Lb
            cells_of[(a, j)] = c
            allc.append(c)
    cov = np.bincount(np.concatenate(allc), minlength=len(yb)) if allc else np.zeros(len(yb), np.int64)
    e = max(E.parse_cond(cond)["eps"], H.EPS_FLOOR)
    AC = {}
    for sh in set(int(s) for s in st.qsh):
        seg = slice(sh * st.d, sh * st.d + st.nb * st.Lb)
        un = cov[seg] == 0
        q = float(yb[seg][un].mean()) if un.any() else 0.5
        q = min(max(q, 1e-3), 0.999)
        AC[sh] = (math.log((1 - e) / q), math.log((1 - q) / e))
    got2 = []
    for a in range(nq):
        A, Cc = AC[int(st.qsh[a])]
        g = []
        for j in range(S38.J):
            v, last, _, _ = np_slot(st, yb, S38, a, j, cov=cov, own=cells_of.get((a, j), np.full(st.nb, -1)),
                                    A=A, Cc=Cc)
            g.append(v)
            if last:
                break
        got2.append(g)
    return ([a for a in range(nq) if got1[a] != want[a]], [a for a in range(nq) if got2[a] != want[a]], got1)


def loop_had(S38, pairs, B, cond, keys):
    """H-HAD read by one float64 dot product per candidate, for probed keys `keys`."""
    st = H.HADStore(S38, B)
    ch = E.Channel(S38.sid, "H-HAD|[]", B, len(st.cells), st.total)
    sig = ch.apply(st.cells, cond)
    rows = sig[:st.S * st.d].reshape(st.S, st.d).astype(np.float64)
    VH = st.VH.astype(np.float64)
    LH = st.LH.astype(np.float64)
    want = want_from_pairs(S38, pairs)
    bad = []
    for a in keys:
        x = rows[st.qsh[a]] * S38.Kc[S38.qk[a], :st.d]
        g = []
        for j in range(S38.J):
            z = x * S38.Pc[j, :st.d]
            zl = z * LH
            s0 = np.array([np.dot(z, VH[v]) for v in range(len(VH))])
            s1 = np.array([np.dot(zl, VH[v]) for v in range(len(VH))])
            if s1.max() > s0.max():
                g.append(int(s1.argmax()))
                break
            g.append(int(s0.argmax()))
        if g != want[a]:
            bad.append(int(a))
    return bad


def scorer(choice_path, paths, out_path, per_rule=6):
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    rows = []
    for p in paths:
        for r in jl(p):
            for x in r["res"]:
                if not x.get("infeasible"):
                    rows.append((r["sid"], x))
    rng = np.random.default_rng(61)
    out = []
    sbc = [(s, x) for s, x in rows if x["arm"] == "SBC" and x["B"] <= 1448]
    had = [(s, x) for s, x in rows if x["arm"] == "HAD" and not x["rules"]["H-HAD"].get("censored")]
    fam = lambda s: s.split("/")[1] if "/" in s else s
    pick = []
    for pool, n in ((sbc, 2 * per_rule), (had, per_rule)):
        byf = collections.defaultdict(list)
        for s, x in pool:
            byf[(fam(s), x["cond"].count("-") > 0)].append((s, x))
        ks = sorted(byf)
        for i in range(n):
            if not ks:
                break
            k = ks[i % len(ks)]
            pick.append(byf[k][int(rng.integers(0, len(byf[k])))])
    for s, x in pick:
        pairs = stores[s]
        S38 = H.slim(E.Store38(s, pairs))
        rec = dict(sid=s, arm=x["arm"], lam=x["lam"], B=x["B"], cond=x["cond"])
        if x["arm"] == "SBC":
            f1, f2, _ = np_sbc_eval(S38, pairs, x["lam"], x["B"], x["cond"])
            rec.update(sbc_same=f1 == x["rules"]["H-SBC"]["failed"], sic_same=f2 == x["rules"]["H-SIC"]["failed"],
                       sbc_n=[len(f1), len(x["rules"]["H-SBC"]["failed"])],
                       sic_n=[len(f2), len(x["rules"]["H-SIC"]["failed"])])
        else:
            nq = len(S38.want)
            keys = list(range(nq)) if nq <= 40 else sorted(set(rng.choice(nq, 30, replace=False).tolist())
                                                          | set(x["rules"]["H-HAD"]["failed"][:10]))
            bad = loop_had(S38, pairs, x["B"], x["cond"], keys)
            ref = sorted(set(x["rules"]["H-HAD"]["failed"]) & set(keys))
            rec.update(keys=len(keys), had_same=sorted(bad) == ref, had_n=[len(bad), len(ref)])
        print(rec, flush=True)
        out.append(rec)
    json.dump(out, open(out_path, "w"), indent=1)
    ok = [all(v for k, v in r.items() if k.endswith("_same")) for r in out]
    print("second scorer: %d of %d sampled cells identical" % (sum(ok), len(ok)))


def bits(paths, out_path):
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    nn = {s: len(p) for s, p in stores.items()}
    over, tot, share, dens = 0, 0, [], collections.defaultdict(list)
    for p in paths:
        for r in jl(p):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                tot += 1
                cap = int(x["B"] * nn[r["sid"]])
                over += x["cells"] > cap
                share.append(x["cells"] / cap)
                if x["arm"] == "SBC":
                    dens[x["lam"]].append(x["density"])
    res = dict(evaluations=tot, over_budget=over, share_min=min(share), share_median=float(np.median(share)),
               density={str(k): [round(min(v), 3), round(float(np.median(v)), 3), round(max(v), 3)]
                        for k, v in sorted(dens.items())})
    json.dump(res, open(out_path, "w"), indent=1)
    print(res)


def nstar2(choice_path, verdict_path, paths):
    """Separately written: N* = smallest evaluated budget g such that every evaluated budget >= g has pooled share
    >= 0.995 (pooled over all 50 stores; a budget with a missing store is skipped)."""
    ch = json.load(open(choice_path))
    J = json.load(open(verdict_path))
    sids = set(s for s, _ in E.load_stores("b34_out/test_stores.json.gz"))
    agg = collections.defaultdict(dict)
    for p in paths:
        for r in jl(p):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                gi = H.gidx(x["B"])
                for rule, y in x["rules"].items():
                    if H.design_at(ch, rule, x["cond"], gi) != x["lam"]:
                        continue
                    agg[(rule, x["cond"], gi)][r["sid"]] = (10 ** 6 if y.get("censored") else len(y["failed"]), y["n"])
    mism, n = [], 0
    for rule in H.RULES:
        for c in list(E.IID) + H.DMG4:
            sh = {}
            for (ru, cc, gi), d in agg.items():
                if ru == rule and cc == c and set(d) == sids:
                    f = sum(v[0] for v in d.values())
                    m = sum(v[1] for v in d.values())
                    sh[gi] = 1 - min(f, m) / m
            if not sh:
                continue
            gs = sorted(sh)
            ns = None
            for g in gs:
                if all(sh[h] >= 0.995 for h in gs if h >= g):
                    ns = GRID[g]
                    break
            ref = J["nstar"].get("%s|%s" % (rule, c)) if c in E.IID else J["damage"].get(c, {}).get(rule)
            n += 1
            if (ns is None) != (ref is None) or (ns is not None and abs(ns - ref) > 1e-6):
                mism.append((rule, c, ns, ref))
    print("N* second rule: %d mismatches of %d (rule, condition)" % (len(mism), n), mism)


def handread(choice_path, verdict_path, paths, out_path):
    """Failed keys one grid step below N*: H-SBC at k1, k8, k16 (full scores) and H-HAD at k8 (scores)."""
    ch = json.load(open(choice_path))
    J = json.load(open(verdict_path))
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    rows = collections.defaultdict(dict)
    for p in paths:
        for r in jl(p):
            for x in r["res"]:
                if not x.get("infeasible"):
                    rows[(x["arm"], x["cond"], H.gidx(x["B"]), json.dumps(x["lam"]))][r["sid"]] = x
    L = ["# B61 sample reading by the AI-assisted process (not independent): failed keys one grid step below N*.", ""]
    for rule, arm, c in (("H-SBC", "SBC", "k1"), ("H-SBC", "SBC", "k8"), ("H-SBC", "SBC", "k16"),
                         ("H-HAD", "HAD", "k8")):
        ns = J["nstar"].get("%s|%s" % (rule, c))
        if not ns:
            continue
        gi = H.gidx(ns) - 1
        lam = H.design_at(ch, rule, c, gi)
        d = rows.get((arm, c, gi, json.dumps(lam)), {})
        fl = [(s, a) for s, x in sorted(d.items()) for a in x["rules"][rule]["failed"]]
        L.append("%s %s at B = %.0f (lam %s): %d failed keys in %d stores" % (
            rule, c, GRID[gi], lam, len(fl), len({s for s, _ in fl})))
        for s, a in fl[:6]:
            pairs = stores[s]
            S38 = H.slim(E.Store38(s, pairs))
            want = want_from_pairs(S38, pairs)[a]
            if arm == "SBC":
                st = H.SBCStore(S38, GRID[gi], lam)
                chn = E.Channel(s, "H-SBC|%.4f" % lam, GRID[gi], len(st.cells), st.total)
                yb = (chn.apply(st.cells, c) > 0).astype(np.uint8)
                for j in range(S38.J):
                    v, last, s0, s1 = np_slot(st, yb, S38, a, j)
                    tv = want[j] if j < len(want) else None
                    tl = j == len(want) - 1
                    tsc = None if tv is None else (s1[tv] if tl else s0[tv])
                    win = s1[v] if last else s0[v]
                    nwin = int(((s1 if last else s0) == win).sum())
                    L.append("  %s key %d j %d: want %s%s, got %s%s; true score %s, winner %s of nb %d (Lb %d); "
                             "candidates at the winning score %d; %s" % (
                                 s, a, j, tv, " LAST" if tl else "", v, " LAST" if last else "",
                                 None if tsc is None else int(tsc), int(win), st.nb, st.Lb, nwin,
                                 "ok" if (v == tv and last == tl) else (
                                     "wrong LAST decision" if v == tv else
                                     ("tie" if tsc is not None and tsc == win else "wrong value"))))
                    if last or v != tv:
                        break
            else:
                st = H.HADStore(S38, GRID[gi])
                chn = E.Channel(s, "H-HAD|[]", GRID[gi], len(st.cells), st.total)
                sig = chn.apply(st.cells, c)
                rws = sig[:st.S * st.d].reshape(st.S, st.d).astype(np.float64)
                VH = st.VH.astype(np.float64)
                x = rws[st.qsh[a]] * S38.Kc[S38.qk[a], :st.d]
                for j in range(S38.J):
                    z = x * S38.Pc[j, :st.d]
                    s0, s1 = VH @ z, VH @ (z * st.LH)
                    last = s1.max() > s0.max()
                    v = int(s1.argmax() if last else s0.argmax())
                    tv = want[j] if j < len(want) else None
                    tl = j == len(want) - 1
                    sc = s1 if tl else s0
                    sd = float(np.concatenate([s0, s1]).std())
                    tsc = None if tv is None else float(sc[tv])
                    win = float((s1 if last else s0)[v])
                    L.append("  %s key %d j %d: want %s%s, got %d%s; true correlation %s, winner %.0f (d %d; noise SD of "
                             "all candidates %.0f, true value %s SD above the candidates' mean); %s" % (
                                 s, a, j, tv, " LAST" if tl else "", v, " LAST" if last else "",
                                 None if tsc is None else "%.0f" % tsc, win, st.d, sd,
                                 None if tsc is None else "%.1f" % ((tsc - np.concatenate([s0, s1]).mean()) / sd),
                                 "ok" if (v == tv and last == tl) else (
                                     "wrong LAST decision" if v == tv else "wrong value")))
                    if last or v != tv:
                        break
        L.append("")
    open(out_path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scorer":
        scorer(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
    elif cmd == "bits":
        bits(sys.argv[2].split(","), sys.argv[3])
    elif cmd == "nstar2":
        nstar2(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
    elif cmd == "handread":
        handread(sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5])
