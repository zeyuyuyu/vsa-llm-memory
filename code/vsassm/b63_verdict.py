"""B63 verdict (prereg/B63_online_directory_prereg.md): the coded tables with an online-insertion directory (O-BCH, O-LDPC;
cuckoo table with fingerprints) against H-SBC (B61, primary), U (B38 V-last + B59 T1) and the MPHF tables (B38).

python3 b63_verdict.py CHOICE.json OUT_PREFIX TEST1.json[,TEST2.json...]
"""
import collections
import json
import sys

import numpy as np

import b38_eq as E
import b58_verdict as V58
import b59_store_verdict as V59
import b61_verdict as V61
import b63_online as O

GRID = E.GRID
NG = len(GRID)
IIDC = list(E.IID)
LABK = [c for c in IIDC if c != "e05"]
TABLES = ("T-rep", "E-BCH", "E-LDPC", "E-CAM")
ARMS = O.ARMS
B61_CHOICE = "results/b61_choice.json"
B61_TESTS = ["results/b61_test1.json.gz"]
MODEL = dict(k1=32, k2=32, k4=45, e05=64, k8=91, k16=362, k32=1218)      # prereg predictions (MPHF best + 3 steps)


def steps(gi):
    return NG if gi is None else gi


def fmt(gi):
    return ">4096" if gi is None or gi >= NG else "%.0f" % GRID[gi]


def all_counts(ch, paths):
    sids = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    C0, _ = V59.counts(None, {"tiers": {}}, ["results/b59_test_T1_U.json.gz"])
    C = collections.defaultdict(dict)
    for k, v in C0.items():
        if k[0] in ("U",) + TABLES and k[1] in IIDC:
            C[k] = v
    ch61 = json.load(open(B61_CHOICE))
    C61 = V61.b61_counts(ch61, B61_TESTS)
    for c in IIDC:
        C[("H-SBC", c)] = C61[("H-SBC", c)]
    for k, v in O.test_counts(ch, paths).items():
        C[k] = v
    return C, sids


def main(choice_path, out_prefix, paths):
    ch = json.load(open(choice_path))
    C, sids = all_counts(ch, paths)
    J = {"nstar": {}, "delta": {}, "label": {}, "pen": {}, "boot": {}, "coverage": {}, "diag": {}}
    L = ["# B63 verdict (prereg/B63_online_directory_prereg.md): coded tables with an online-insertion directory",
         "# (cuckoo table with f-bit fingerprints) vs the sparse block code store H-SBC (primary), the dense store U and",
         "# the MPHF tables. N* in cells per stored pair ('>4096' = not reached). Steps: one grid step = x 2^(1/4).",
         "# OT-best = per k the smaller of O-BCH's and O-LDPC's own N* (test-selected; favours the table).", ""]

    def ns(name, c):
        if name == "OT-best":
            return min(steps(ns(a, c)) for a in ARMS)
        if name == "MPHF best table":
            return min(steps(ns(t, c)) for t in TABLES)
        if (name, c) not in C:
            return NG
        return steps(V58.nstar_counts(C[(name, c)], sids)[0])

    rows = ARMS + ("OT-best", "H-SBC", "U", "MPHF best table") + TABLES
    NS = {(f, c): ns(f, c) for f in rows for c in IIDC}
    for (f, c), v in NS.items():
        J["nstar"]["%s|%s" % (f, c)] = None if v >= NG else GRID[v]
    for a in ARMS:
        for c in IIDC:
            if (a, c) in C:
                J["coverage"]["%s|%s" % (a, c)] = [GRID[g] for g in sorted(V58.nstar_counts(C[(a, c)], sids)[1])]
    L.append("Table 1 (i.i.d., test): N*")
    L.append("  %-16s" % "" + "".join("%9s" % c for c in IIDC))
    for f in rows:
        L.append("  %-16s" % f + "".join("%9s" % fmt(NS[(f, c)]) for c in IIDC))
    L.append("")
    L.append("Design at N* (dev choice): " + "; ".join(
        "%s %s" % (a, " ".join("%s=%s" % (c, O.design_at(ch, a, c, NS[(a, c)]) if NS[(a, c)] < NG else "-")
                               for c in IIDC)) for a in ARMS))
    L.append("Evaluated budgets per arm (every store):")
    for k, v in J["coverage"].items():
        L.append("  %-12s %s" % (k, ", ".join("%.0f" % x for x in v)))
    L.append("")
    for arm in ARMS + ("OT-best",):
        for S in ("H-SBC", "U"):
            key = "%s_vs_%s" % (arm, S)
            J["delta"][key] = {}
            L.append("%s vs %s: Delta = steps(%s) - steps(%s) (+: store fewer cells)" % (arm, S, arm, S))
            dl = []
            for c in IIDC:
                d = NS[(arm, c)] - NS[(S, c)]
                J["delta"][key][c] = dict(delta=d, label=V58.label(d))
                if c != "e05":
                    dl.append(d)
                L.append("    %-4s %-8s %6s  %-6s %6s | Delta %+d %s" % (c, arm, fmt(NS[(arm, c)]), S, fmt(NS[(S, c)]),
                                                                        d, V58.label(d)))
            J["label"][key] = V58.h_label(dl)
            L.append("  label (k1, k2, k4, k8, k16, k32): %s" % J["label"][key])
            L.append("")
    L.append("Online penalty Pen = steps(OT-best) - steps(MPHF best table):")
    for c in IIDC:
        p = NS[("OT-best", c)] - NS[("MPHF best table", c)]
        J["pen"][c] = p
        L.append("    %-4s OT-best %6s  MPHF best %6s  Pen %+d (x%.2f)" % (
            c, fmt(NS[("OT-best", c)]), fmt(NS[("MPHF best table", c)]), p, 2 ** (p / 4)))
    L.append("")
    # diagnostics at N* and one step below (pooled)
    for a in ARMS:
        for c in IIDC:
            gi = NS[(a, c)]
            for g in (gi - 1, gi):
                d = C.get((a, c), {}).get(g)
                if not d:
                    continue
                J["diag"]["%s|%s|%s" % (a, c, fmt(g))] = dict(failed=sum(v[0] for v in d.values()),
                                                              n=sum(v[1] for v in d.values()))
    # bootstrap
    sidx = {sd: i for i, sd in enumerate(sids)}
    MAT = {}

    def mat(key):
        if key not in MAT:
            F = np.zeros((NG, len(sids)))
            Nn = np.zeros((NG, len(sids)))
            M = np.zeros((NG, len(sids)), bool)
            for gi, d in C.get(key, {}).items():
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
    for f in ARMS + ("H-SBC", "U") + TABLES:
        for c in IIDC:
            if (f, c) in C:
                assert fast_ns((f, c), w1) == NS[(f, c)], (f, c)
    rng = np.random.default_rng(63)
    Bt = collections.defaultdict(list)
    for it in range(2000):
        draw = rng.integers(0, len(sids), len(sids))
        w = np.bincount(draw, minlength=len(sids)).astype(float)
        for c in IIDC:
            ot = min(fast_ns((a, c), w) for a in ARMS)
            mp = min(fast_ns((t, c), w) for t in TABLES)
            Bt[("dS", c)].append(ot - fast_ns(("H-SBC", c), w))
            Bt[("dU", c)].append(ot - fast_ns(("U", c), w))
            Bt[("pen", c)].append(ot - mp)
            Bt[("N", c)].append(ot)

    def rg(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return int(round(lo)), int(round(hi))
    L.append("Store-cluster bootstrap (2,000 resamples, seed 63): 95% ranges (OT-best)")
    for c in IIDC:
        a, b, p, n = rg(Bt[("dS", c)]), rg(Bt[("dU", c)]), rg(Bt[("pen", c)]), rg(Bt[("N", c)])
        J["boot"][c] = dict(delta_HSBC=a, delta_U=b, pen=p, nstar=[fmt(n[0]), fmt(n[1])])
        L.append("    %-4s Delta vs H-SBC [%+d, %+d]  Delta vs U [%+d, %+d]  Pen [%+d, %+d]  N* [%s, %s]" % (
            c, a[0], a[1], b[0], b[1], p[0], p[1], fmt(n[0]), fmt(n[1])))
    L.append("")
    L.append("Dev N* (12 dev stores): " + "; ".join(
        "%s %s" % (a, " ".join("%s=%s" % (c, ("%.0f" % x) if x else "-") for c, x in
                               ((c, ch["dev_nstar"][a][c]) for c in IIDC))) for a in ARMS))
    # predictions (reporting only)
    L.append("")
    L.append("Predictions (prereg S1-S6; reporting only):")
    s1 = {c: NS[("OT-best", c)] - O.gidx(MODEL[c]) for c in IIDC}
    pen_ok = all(J["pen"][c] in (2, 3, 4) for c in IIDC)
    L.append("  S1 %s: OT-best within 1 step of the model at every k %s; Pen in {+2,+3,+4} at every k %s" % (
        "HELD" if all(abs(v) <= 1 for v in s1.values()) and pen_ok else "FAILED", s1, J["pen"]))
    dS = {c: J["delta"]["OT-best_vs_H-SBC"][c]["delta"] for c in IIDC}
    L.append("  S2 %s: TIE vs H-SBC, |Delta| <= 1 at every labelled k: %s (label %s)" % (
        "HELD" if J["label"]["OT-best_vs_H-SBC"] == "TIE" else "FAILED", dS, J["label"]["OT-best_vs_H-SBC"]))
    dU = {c: J["delta"]["OT-best_vs_U"][c]["delta"] for c in LABK}
    s3 = J["label"]["OT-best_vs_U"] == "TABLE WIN" and sum(v <= -2 for v in dU.values()) >= 4
    L.append("  S3 %s: TABLE WIN vs U, Delta <= -2 at >= 4 of 6: %s" % ("HELD" if s3 else "FAILED", dU))
    s4 = all(NS[("O-BCH", c)] <= NS[("O-LDPC", c)] for c in ("k1", "k2", "k4")) and \
        all(NS[("O-LDPC", c)] <= NS[("O-BCH", c)] for c in ("k8", "k16", "k32"))
    L.append("  S4 %s: O-BCH <= O-LDPC at k1-k4 and O-LDPC <= O-BCH at k8-k32: %s" % (
        "HELD" if s4 else "FAILED", {c: (fmt(NS[("O-BCH", c)]), fmt(NS[("O-LDPC", c)])) for c in IIDC}))
    fdev = {}
    for a in ARMS:
        for c in IIDC:
            x = ch["dev_nstar"][a][c]
            fdev["%s|%s" % (a, c)] = ch["designs"][a][c].get(str(O.gidx(x)))[0] if x else None
    n1416 = sum(v in (14, 16) for v in fdev.values())
    L.append("  S5 %s: dev f at dev N* 14 or 16 in %d of 14 (>= 10): %s" % ("HELD" if n1416 >= 10 else "FAILED", n1416,
                                                                          fdev))
    lost = 0
    for p in paths:
        for r in O.jload(p):
            lost += r["lost"]
    L.append("  S6 %s: lost insertions summed over test jobs: %d" % ("HELD" if lost == 0 else "FAILED", lost))
    J["predictions"] = dict(S1=s1, S5=fdev, S6=lost)
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [p for p in sys.argv[3].split(",") if p])
