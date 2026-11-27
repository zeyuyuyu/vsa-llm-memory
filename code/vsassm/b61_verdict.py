"""B61 verdict (prereg/B61_hd_protected_store_prereg.md): HD-native stores H-SBC, H-SIC, H-HAD against the uncoded store U
(B38 V-last + B59 T1) and the best table (B38's T-rep / E-BCH / E-LDPC / E-CAM; damage: B58's re-scored E-LDPC*).

python3 b61_verdict.py CHOICE.json OUT_PREFIX TEST1.json[,TEST2.json...]
"""
import collections
import gzip
import json
import sys

import numpy as np

import b38_eq as E
import b58_verdict as V58
import b59_store_verdict as V59
import b61_hd as H

GRID = E.GRID
NG = len(GRID)
IIDC = list(E.IID)
LABK = [c for c in IIDC if c != "e05"]
TABLES = ("T-rep", "E-BCH", "E-LDPC", "E-CAM")
TABLES_DMG = ("T-rep", "E-BCH", "E-LDPC*", "E-CAM")
ARMS = H.RULES


def steps(gi):
    return NG if gi is None else gi


def fmt(gi):
    return ">4096" if gi is None or gi >= NG else "%.0f" % GRID[gi]


def b61_counts(ch, paths):
    """C[(rule, cond)] = {gi: {sid: (failed, n)}} at each rule's dev design (censored store -> 10^6 failed)."""
    C = collections.defaultdict(dict)
    acc = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in paths:
        for r in (json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                for rule, y in x["rules"].items():
                    acc[(rule, x["cond"])][(H.gidx(x["B"]), json.dumps(x["lam"]))][r["sid"]] = (
                        10 ** 6 if y.get("censored") else len(y["failed"]), y["n"])
    for (rule, c), rows in acc.items():
        for (gi, dk), d in rows.items():
            if json.loads(dk) == H.design_at(ch, rule, c, gi):
                C[(rule, c)][gi] = d
    return C


def main(choice_path, out_prefix, paths):
    ch = json.load(open(choice_path))
    sids = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    C0, _ = V59.counts(None, {"tiers": {}}, ["results/b59_test_T1_U.json.gz"])
    C = b61_counts(ch, paths)
    for k, v in C0.items():
        if k[0] in ("U",) + TABLES + ("E-LDPC*",):
            C[k] = v
    J = {"nstar": {}, "G": {}, "H1": {}, "boot": {}, "damage": {}, "coverage": {}}
    L = ["# B61 verdict (prereg/B61_hd_protected_store_prereg.md): HD-native protected stores vs the uncoded store U and",
         "# the best table. N* in cells per stored pair ('>4096' = not reached). Steps: one grid step = x 2^(1/4).",
         "# HD-best = per k the smallest of the three HD arms' own N* (test-selected; favours the store).", ""]

    def ns(name, c, w=None):
        if name == "HD-best":
            return min(steps(ns(a, c, w)) for a in ARMS if (a, c) in C) if any((a, c) in C for a in ARMS) else NG
        if name == "best table":
            return min(steps(V58.nstar_counts(C[(t, c)], sids, w)[0]) for t in TABLES if (t, c) in C)
        if name == "best non-HD table":
            return min(steps(V58.nstar_counts(C[(t, c)], sids, w)[0]) for t in TABLES_DMG if (t, c) in C)
        if (name, c) not in C:
            return None
        return steps(V58.nstar_counts(C[(name, c)], sids, w)[0])

    NS = {}
    rows = ARMS + ("HD-best", "U", "best table") + TABLES
    for f in rows:
        for c in IIDC:
            v = ns(f, c)
            NS[(f, c)] = v
            J["nstar"]["%s|%s" % (f, c)] = None if v is None or v >= NG else GRID[v]
    for a in ARMS:
        for c in IIDC:
            if (a, c) in C:
                ev = sorted(V58.nstar_counts(C[(a, c)], sids)[1])
                J["coverage"]["%s|%s" % (a, c)] = [GRID[g] for g in ev]
    L.append("Table 1 (i.i.d., test): N*")
    L.append("  %-11s" % "" + "".join("%9s" % c for c in IIDC))
    for f in rows:
        L.append("  %-11s" % f + "".join("%9s" % ("n/r" if NS[(f, c)] is None else fmt(NS[(f, c)])) for c in IIDC))
    L.append("")
    L.append("Evaluated budgets per arm (every store), i.i.d.:")
    for k, v in J["coverage"].items():
        L.append("  %-12s %s" % (k, ", ".join("%.0f" % x for x in v)))
    L.append("")
    for a in ARMS + ("HD-best",):
        L.append("%s: G = steps(U) - steps(arm) (+: fewer cells than U); Delta = steps(best table) - steps(arm) "
                 "(+: store fewer cells)" % a)
        gl, dl = [], []
        J["G"][a], J["H1"][a] = {}, {}
        for c in IIDC:
            if NS[(a, c)] is None:
                continue
            g = steps(NS[("U", c)]) - NS[(a, c)]
            d = NS[("best table", c)] - NS[(a, c)]
            glab = "fewer cells than U" if g >= 2 else ("more cells than U" if g <= -2 else "tie")
            dlab = V58.label(d)
            gap = "closes the gap" if d >= -1 else ("narrows the gap" if g >= 2 else "gap not narrowed")
            if c != "e05":
                gl.append(glab)
                dl.append(d)
            L.append("    %-4s arm %6s  U %6s  best table %6s | G %+d %s | Delta %+d %s | %s" % (
                c, fmt(NS[(a, c)]), fmt(NS[("U", c)]), fmt(NS[("best table", c)]), g, glab, d, dlab, gap))
            J["G"][a][c] = dict(G=g, label=glab)
            J["H1"][a][c] = dict(delta=d, label=dlab, gap=gap)
        J["H1"][a]["label"] = V58.h_label(dl)
        J["G"][a]["fewer_than_U"] = [sum(x == "fewer cells than U" for x in gl), len(gl)]
        L.append("  H1-HD %s: %s; fewer cells than U at %d of %d labelled k" % (
            a, J["H1"][a]["label"], J["G"][a]["fewer_than_U"][0], len(gl)))
        L.append("")
    # bootstrap (vectorized per-store counts; the point estimate is checked against b58_verdict.nstar_counts)
    sidx = {sd: i for i, sd in enumerate(sids)}
    MAT = {}

    def mat(key):
        if key not in MAT:
            F = np.zeros((NG, len(sids)))
            Nn = np.zeros((NG, len(sids)))
            M = np.zeros((NG, len(sids)), bool)
            for gi, d in C[key].items():
                for sd, (f, n) in d.items():
                    if sd in sidx:
                        F[gi, sidx[sd]], Nn[gi, sidx[sd]], M[gi, sidx[sd]] = f, n, True
            MAT[key] = (F, Nn, M)
        return MAT[key]

    def fast_ns(key, w):
        F, Nn, M = mat(key)
        pos = w > 0
        ok = M[:, pos].all(1) & (Nn @ w > 0)
        f, n = F @ w, Nn @ w
        best = NG
        for gi in range(NG - 1, -1, -1):
            if not ok[gi]:
                continue
            if 1 - min(f[gi], n[gi]) / n[gi] >= E.TARGET:
                best = gi
            else:
                break
        return best

    w1 = np.ones(len(sids))
    for f in ARMS + ("U",) + TABLES:
        for c in IIDC:
            if (f, c) in C:
                assert fast_ns((f, c), w1) == steps(V58.nstar_counts(C[(f, c)], sids)[0]), (f, c)
    rng = np.random.default_rng(61)
    L.append("Store-cluster bootstrap (2,000 resamples, seed 61): 95% ranges of G and Delta, and of the N*")
    B_ = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in range(2000):
        draw = rng.integers(0, len(sids), len(sids))
        w = np.bincount(draw, minlength=len(sids)).astype(float)
        for c in IIDC:
            u = fast_ns(("U", c), w)
            t = min(fast_ns((tb, c), w) for tb in TABLES if (tb, c) in C)
            arm_ns = {a: fast_ns((a, c), w) for a in ARMS if (a, c) in C}
            if arm_ns:
                arm_ns["HD-best"] = min(arm_ns.values())
            for a, sv in arm_ns.items():
                B_[(a, c)]["G"].append(u - sv)
                B_[(a, c)]["D"].append(t - sv)
                B_[(a, c)]["N"].append(sv)

    def rg(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return int(round(lo)), int(round(hi))
    for a in ARMS + ("HD-best",):
        J["boot"][a] = {}
        for c in IIDC:
            if (a, c) not in B_:
                continue
            g, d, n = rg(B_[(a, c)]["G"]), rg(B_[(a, c)]["D"]), rg(B_[(a, c)]["N"])
            J["boot"][a][c] = dict(G=g, delta=d, nstar=[fmt(n[0]), fmt(n[1])])
            L.append("    %-8s %-4s G [%+d, %+d]  Delta [%+d, %+d]  N* [%s, %s]" % (a, c, g[0], g[1], d[0], d[1],
                                                                                   fmt(n[0]), fmt(n[1])))
    # damage
    L.append("")
    L.append("Table 2 (damage, natural layout, f = 2 eps of the matched i.i.d. condition): N*; best non-HD table with "
             "B58's re-scored E-LDPC*")
    L.append("  %-14s" % "" + "".join("%9s" % v for v in ("H-SBC", "H-SIC", "U") + TABLES_DMG)
             + "  | per arm: Delta vs best non-HD table; G vs U; within 1 step of matched i.i.d.")
    dd = collections.defaultdict(list)
    for c in H.DMG4:
        row = {}
        for f in ("H-SBC", "H-SIC", "U") + TABLES_DMG:
            row[f] = ns(f, c) if (f, c) in C else None
        tb = ns("best non-HD table", c)
        m = E.parse_cond(c)["match"]
        txt = []
        J["damage"][c] = {f: (None if row[f] is None or row[f] >= NG else GRID[row[f]]) for f in row}
        for a in ("H-SBC", "H-SIC"):
            if row[a] is None:
                continue
            d = tb - row[a]
            g = steps(row["U"]) - row[a]
            pi = abs(row[a] - NS[(a, m)]) <= 1
            dd[a].append(d)
            txt.append("%s %+d %s, G %+d, pos.ind. %s" % (a, d, V58.label(d), g, pi))
            J["damage"][c][a + "_eval"] = dict(delta=d, label=V58.label(d), G=g, posind=pi)
        L.append("  %-14s" % c + "".join("%9s" % ("n/r" if row[f] is None else fmt(row[f])) for f in row)
                 + "  | " + "; ".join(txt))
    for a, ds in dd.items():
        J["damage"][a + "_label"] = V58.h_label(ds)
        L.append("  natural-layout damage, %s vs the best non-HD table: %s" % (a, V58.h_label(ds)))
    L.append("")
    L.append("Dev N* (12 dev stores): " + "; ".join(
        "%s %s" % (a, " ".join("%s=%s" % (c, ("%.0f" % x) if x else "-") for c, x in
                               ((c, ch["dev_nstar"][a][c]) for c in IIDC))) for a in ARMS))
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [p for p in sys.argv[3].split(",") if p])
