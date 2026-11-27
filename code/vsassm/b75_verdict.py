"""B75 verdict (prereg/B75_strongest_options_prereg.md): requirements (cells per stored pair for >= 99.5% exact) of the
new memories against the paper's registered comparator rows, Deltas in grid steps, labels, bootstrap, predictions.

python3 b75_verdict.py CHOICE.json OUT_PREFIX TEST.json[,..] [--fresh FRESH.json[,..]]
"""
import collections
import json
import sys

import numpy as np

import b38_eq as E
import b69_vocab as B69
import b75_strong as S

GRID = E.GRID
NG = len(GRID)
TARGET = E.TARGET
CONDS = S.CONDS
PRIMARY = S.PRIMARY
RULER_TABLES = ("T-rep", "E-BCH", "E-LDPC", "E-CAM")
FRESH_TABLES = ("T-rep", "E-BCH", "E-LDPC")


def fmt(gi, flag=""):
    return (">4096" if gi is None or gi >= NG else "%.0f" % GRID[gi]) + flag


def lab1(d, x="X", y="Y"):
    return ("%s fewer" % x) if d <= -2 else ("%s fewer" % y) if d >= 2 else "tie"


def hlab(ds, x, y):
    xf = any(d <= -2 for d in ds)
    yf = any(d >= 2 for d in ds)
    return "MIXED" if xf and yf else ("%s WIN" % x) if xf else ("%s WIN" % y) if yf else "TIE"


def ruler_comparators():
    import b63_verdict as V63
    ch63 = json.load(open("results/b63_choice.json"))
    C, sids = V63.all_counts(ch63, ["results/b63_test1.json.gz"])
    return {k: v for k, v in C.items()}, list(sids)


def fresh_comparators():
    import b73_cells as F
    fz = json.load(open("results/b73_cells_frozen.json"))
    C = F.counts(fz, ["b73_out/cells/test1.json"])
    sids = sorted({s for v in C.values() for d in v.values() for s in d})
    return {k: v for k, v in C.items()}, sids


def nstar_flag(curve, sids):
    """(N* grid index or NG, flag): flag '<=' if N* is the lowest evaluated budget above the grid bottom, '>' if the
    top evaluated budget fails (censored)."""
    gi, sh = B69.nstar_counts(curve, sids)
    if not sh:
        return NG, " (none)"
    if gi >= NG:
        return NG, ""
    if gi == min(sh) and gi > 0:
        return gi, "*"
    return gi, ""


class Panel:
    def __init__(self, ch, paths, which):
        self.which = which
        if which == "ruler":
            C0, self.sids = ruler_comparators()
            self.tables = RULER_TABLES
        else:
            C0, self.sids = fresh_comparators()
            self.tables = FRESH_TABLES
        self.C = dict(C0)
        Cn = S.test_counts(ch, paths)
        for (mem, acct, c), v in Cn.items():
            self.C[("%s|%s" % (mem, acct), c)] = v
        self.NS, self.FL = {}, {}

    def ns(self, name, c):
        if (name, c) in self.NS:
            return self.NS[(name, c)]
        if name == "static":
            v = min(self.ns(t, c) for t in self.tables)
        elif name == "cuckoo":
            v = min(self.ns(t, c) for t in ("O-BCH", "O-LDPC"))
        elif name == "O2-best":
            v = min(self.ns(t, c) for t in ("O2-BCH|free", "O2-LDPC|free"))
        elif name.startswith("J-best|"):
            a = name.split("|")[1]
            v = min(self.ns("U-J|" + a, c), self.ns("H-SBC-J|" + a, c))
        elif (name, c) in self.C:
            v, fl = nstar_flag(self.C[(name, c)], self.sids)
            self.FL[(name, c)] = fl
        else:
            v = NG
            self.FL[(name, c)] = " (not run)"
        self.NS[(name, c)] = v
        return v

    def ran(self, name, c):
        """False if the memory (or every component of a derived row) was not run in this panel."""
        if name == "static":
            return any(self.ran(t, c) for t in self.tables)
        if name == "cuckoo":
            return any(self.ran(t, c) for t in ("O-BCH", "O-LDPC"))
        if name == "O2-best":
            return any(self.ran(t, c) for t in ("O2-BCH|free", "O2-LDPC|free"))
        if name.startswith("J-best|"):
            a = name.split("|")[1]
            return self.ran("U-J|" + a, c) or self.ran("H-SBC-J|" + a, c)
        return (name, c) in self.C

    def f(self, name, c):
        if not self.ran(name, c):
            return "-"
        v = self.ns(name, c)
        fl = self.FL.get((name, c), "")
        if fl in (" (not run)", " (none)"):
            return "-"
        return fmt(v, fl)


COMPARE = [  # (part, X, Y, conds)
    ("A", "U-LS|free", "U", CONDS), ("A", "U-LS|free", "H-SBC", CONDS), ("A", "U-LS|free", "static", CONDS),
    ("A", "U-LS|free", "cuckoo", CONDS), ("A", "U-LS|free", "O2-best", CONDS),
    ("B", "U-J|free", "U", PRIMARY), ("B", "H-SBC-J|free", "H-SBC", PRIMARY), ("B", "J-best|free", "H-SBC", PRIMARY),
    ("B", "J-best|free", "static", PRIMARY), ("B", "J-best|free", "cuckoo", PRIMARY),
    ("B", "J-best|free", "O2-best", PRIMARY),
    ("B", "U-J|chg", "U", PRIMARY), ("B", "H-SBC-J|chg", "H-SBC", PRIMARY), ("B", "J-best|chg", "H-SBC", PRIMARY),
    ("B", "J-best|chg", "static", PRIMARY), ("B", "J-best|chg", "cuckoo", PRIMARY),
    ("B", "J-best|chg", "O2-best", PRIMARY),
    ("C", "O2-best", "H-SBC", CONDS), ("C", "O2-best", "cuckoo", CONDS), ("C", "O2-best", "static", CONDS),
    ("C", "O2-best", "U", CONDS), ("C", "O2-BCH|free", "H-SBC", CONDS), ("C", "O2-LDPC|free", "H-SBC", CONDS),
]
ROWS = ["U-LS|free", "U-J|free", "U-J|chg", "H-SBC-J|free", "H-SBC-J|chg", "J-best|free", "J-best|chg",
        "O2-BCH|free", "O2-LDPC|free", "O2-best", "U", "H-SBC", "static", "cuckoo"]


def short(nm):
    return nm.replace("|free", "").replace("|chg", " (charged)")


def boot(P, conds, B=2000, seed="b75-boot"):
    """Store-cluster bootstrap within families: share of resamples in which each primary label holds, 95% ranges."""
    rng = np.random.default_rng(abs(hash(seed)) % (2 ** 32) if False else 75)
    fam = collections.defaultdict(list)
    for i, s in enumerate(P.sids):
        fam[B69.family(s) if P.which == "ruler" else s.split("/")[1]].append(i)
    ns = len(P.sids)
    W = np.zeros((B, ns))
    for b in range(B):
        for f, idx in fam.items():
            pick = rng.choice(idx, size=len(idx), replace=True)
            np.add.at(W[b], pick, 1)
    base = [n for n in ROWS if not (n in ("static", "cuckoo", "O2-best") or n.startswith("J-best"))]
    base += list(P.tables) + ["O-BCH", "O-LDPC"]
    NSb = {}
    for name in base:
        for c in conds:
            cur = P.C.get((name, c))
            if not cur:
                NSb[(name, c)] = np.full(B, NG)
                continue
            gis = sorted(g for g, d in cur.items() if all(s in d for s in P.sids))
            if not gis:
                NSb[(name, c)] = np.full(B, NG)
                continue
            F = np.array([[cur[g][s][0] for s in P.sids] for g in gis], float)
            N = np.array([[cur[g][s][1] for s in P.sids] for g in gis], float)
            sh = 1 - np.minimum(W @ F.T, W @ N.T) / (W @ N.T)            # B x len(gis)
            ok = sh >= TARGET
            out = np.full(B, NG)
            for b in range(B):
                best = NG
                for k in range(len(gis) - 1, -1, -1):
                    if ok[b, k]:
                        best = gis[k]
                    else:
                        break
                out[b] = best
            NSb[(name, c)] = out
    for c in conds:
        NSb[("static", c)] = np.min([NSb[(t, c)] for t in P.tables], 0)
        NSb[("cuckoo", c)] = np.minimum(NSb[("O-BCH", c)], NSb[("O-LDPC", c)])
        NSb[("O2-best", c)] = np.minimum(NSb[("O2-BCH|free", c)], NSb[("O2-LDPC|free", c)])
        for a in ("free", "chg"):
            if ("U-J|" + a, c) in NSb:
                NSb[("J-best|" + a, c)] = np.minimum(NSb[("U-J|" + a, c)], NSb[("H-SBC-J|" + a, c)])
    return NSb


def main(choice_path, out_prefix, paths, fresh_paths=None):
    ch = json.load(open(choice_path))
    J = {"panels": {}}
    L = ["# B75 verdict (prereg/B75_strongest_options_prereg.md): cells per stored pair for >= 99.5% of probed keys exact,",
         "# pooled; 30-day i.i.d. cell errors x 1 ... 32; value vocabulary uncharged. Delta(X, Y) = steps(X) - steps(Y)",
         "# (one step = x 2^(1/4)); 'X fewer' if Delta <= -2, 'Y fewer' if >= +2, else tie. '*' = N* is the lowest",
         "# evaluated budget (bracket open below); '>4096' / not reached counts as 37 steps. Comparators = the paper's",
         "# registered rows (B38 / B59 / B61 / B63 on the RULER stores; B73 on the fresh stores).", ""]
    panels = [("ruler", paths, CONDS)] + ([("fresh", fresh_paths, PRIMARY)] if fresh_paths else [])
    for which, pths, conds in panels:
        P = Panel(ch, pths, which)
        JP = J["panels"][which] = {"nstar": {}, "delta": {}, "label": {}, "boot": {}}
        L.append("== %s stores (%d stores, %d probed keys)" % (which, len(P.sids), sum(
            next(iter(P.C[("U", "k1")].values()))[s][1] for s in P.sids) if ("U", "k1") in P.C else -1))
        L.append("  %-22s" % "memory" + "".join("%10s" % c for c in conds))
        for nm in ROWS:
            L.append("  %-22s" % short(nm) + "".join("%10s" % P.f(nm, c) for c in conds))
            for c in conds:
                v = P.ns(nm, c)
                JP["nstar"]["%s|%s" % (nm, c)] = None if v >= NG else GRID[v]
        L.append("")
        L.append("  Comparisons: Delta per condition (ratio Y / X) -> label")
        for part, X, Y, cs in COMPARE:
            cs = [c for c in cs if c in conds and P.ran(X, c) and P.ran(Y, c)]
            if not cs:
                continue
            ds = {c: P.ns(X, c) - P.ns(Y, c) for c in cs}
            lb = hlab([ds[c] for c in cs], short(X), short(Y))
            JP["delta"]["%s vs %s" % (X, Y)] = ds
            JP["label"]["%s vs %s" % (X, Y)] = lb
            L.append("  (%s) %-20s vs %-8s %s -> %s" % (part, short(X), short(Y), "  ".join(
                "%s %+d (%.2f)" % (c, ds[c], 2 ** (ds[c] / 4)) if max(P.ns(X, c), P.ns(Y, c)) < NG else
                "%s %+d (-)" % (c, ds[c]) for c in cs), lb))
        L.append("")
        # O2 logical bits per pair beyond the static table
        meta = []
        for p in pths:
            for r in S.jl(p):
                if r["mem"] in ("O2-BCH", "O2-LDPC") and "L" in r:
                    meta.append((r["sid"], r["mem"], json.dumps(r["par"]), r["L"] / r["n_pairs"],
                                 (r["L"] - r["mphf_bits"]) / r["n_pairs"], r["lost"], r["nst"], r["cap"], r["n_pairs"]))
        if meta:
            by = collections.defaultdict(list)
            for sid, mem, par, bpp, extra, lost, nst, cap, n in meta:
                by[par].append((bpp, extra, lost, nst, cap, n, sid))
            L.append("  O2 logical bits per pair (median [min, max] over stores) and beyond the static MPHF table's:")
            for par in sorted(by):
                v = sorted(set(by[par]), key=lambda t: t[-1])
                bpp = np.array([t[0] for t in v])
                ex = np.array([t[1] for t in v])
                big = [t for t in v if t[5] >= 1000]
                L.append("    (g, ss) %-8s bits/pair %.2f [%.2f, %.2f]; beyond static %.2f [%.2f, %.2f]; lost %d; "
                         "stash overflow stores %d; on >= 1,000-pair stores lost %d" % (
                             par, np.median(bpp), bpp.min(), bpp.max(), np.median(ex), ex.min(), ex.max(),
                             sum(t[2] for t in v), sum(t[3] >= t[4] for t in v), sum(t[2] for t in big)))
                JP.setdefault("o2_bits", {})[par] = dict(median=float(np.median(bpp)), extra_median=float(np.median(ex)),
                                                         lost=int(sum(t[2] for t in v)))
            L.append("")
        # bootstrap (secondary)
        NSb = boot(P, conds)
        L.append("  Store bootstrap (2,000 resamples within families): share in which the label holds; 95% Delta range")
        for part, X, Y, cs in COMPARE:
            cs = [c for c in cs if c in conds and P.ran(X, c) and P.ran(Y, c)]
            if not cs or (X, cs[0]) not in NSb or (Y, cs[0]) not in NSb:
                continue
            D = np.stack([NSb[(X, c)] - NSb[(Y, c)] for c in cs], 1)
            labs = [hlab(list(row), short(X), short(Y)) for row in D]
            lb0 = JP["label"]["%s vs %s" % (X, Y)]
            share = float(np.mean([lb == lb0 for lb in labs]))
            rng_ = {c: [int(np.percentile(D[:, i], 2.5)), int(np.percentile(D[:, i], 97.5))] for i, c in enumerate(cs)}
            JP["boot"]["%s vs %s" % (X, Y)] = dict(share=share, range=rng_)
            L.append("  (%s) %-20s vs %-8s label %-16s held in %.3f; %s" % (part, short(X), short(Y), lb0, share, " ".join(
                "%s [%+d, %+d]" % (c, *rng_[c]) for c in cs)))
        L.append("")
        J["panels"][which]["ns_raw"] = {"%s|%s" % k: int(v) for k, v in P.NS.items()}
    # predictions (RULER panel)
    R = J["panels"]["ruler"]
    d = R["delta"]
    pr = {}
    a_u = d["U-LS|free vs U"]
    pr["PA1"] = (all(-3 <= a_u[c] <= -1 for c in ("k1", "k2", "k4", "k8")) and all(-1 <= a_u[c] <= 0 for c in
                                                                                  ("k16", "k32")),
                 "Delta(U-LS, U) %s" % a_u)
    a_h = d["U-LS|free vs H-SBC"]
    a_s = d["U-LS|free vs static"]
    pr["PA2"] = (all(a_h[c] >= 2 for c in ("k1", "k2", "k4")) and all(abs(a_h[c]) <= 1 for c in ("k8", "k32")) and
                 all(a_s[c] >= 2 for c in CONDS), "Delta(U-LS, H-SBC) %s; Delta(U-LS, static) %s" % (a_h, a_s))
    uj, sj = d["U-J|free vs U"], d["H-SBC-J|free vs H-SBC"]
    pr["PB1"] = (uj["k1"] <= -3 and uj["k8"] <= -1 and sj["k1"] <= -2 and R["nstar"]["H-SBC-J|free|k1"] is not None and
                 R["nstar"]["H-SBC-J|free|k1"] <= 23.5, "Delta(U-J, U) %s; Delta(H-SBC-J, H-SBC) %s; H-SBC-J k1 %s" % (
                     uj, sj, R["nstar"]["H-SBC-J|free|k1"]))
    jc = d["J-best|chg vs H-SBC"]
    js = d["J-best|chg vs static"]
    pr["PB2"] = (all(jc[c] > -2 for c in jc) and all(js[c] >= 2 for c in js),
                 "Delta(J-best chg, H-SBC) %s; Delta(J-best chg, static) %s" % (jc, js))
    oc = d["O2-best vs cuckoo"]
    pr["PC1"] = (all(-2 <= oc[c] <= -1 for c in oc), "Delta(O2-best, cuckoo) %s" % oc)
    oh = d["O2-best vs H-SBC"]
    pr["PC2"] = (oh["k1"] in (-1, -2) and all(oh[c] in (-1, 0) for c in ("k8", "k16")),
                 "Delta(O2-best, H-SBC) %s, label %s" % (oh, R["label"]["O2-best vs H-SBC"]))
    os_ = d["O2-best vs static"]
    pr["PC3"] = (all(1 <= os_[c] <= 3 for c in os_), "Delta(O2-best, static) %s" % os_)
    lost_big = sum(v.get("lost", 0) for v in R.get("o2_bits", {}).values())
    pr["PC4"] = (None, "lost insertions over all O2 test builds (all sizes): %d (see the per-design lines)" % lost_big)
    if "fresh" in J["panels"]:
        F_ = J["panels"]["fresh"]["delta"]
        ok = True
        txt = []
        for k in ("U-LS|free vs U", "U-LS|free vs H-SBC", "O2-best vs H-SBC", "O2-best vs cuckoo", "O2-best vs static"):
            for c in PRIMARY:
                if c in F_.get(k, {}):
                    ok &= abs(F_[k][c] - d[k][c]) <= 1
                    txt.append("%s %s %+d/%+d" % (k, c, d[k][c], F_[k][c]))
        pr["PD"] = (ok, "; ".join(txt))
    else:
        pr["PD"] = (None, "fresh stores not run")
    J["predictions"] = {k: [v[0], v[1]] for k, v in pr.items()}
    L.append("Predictions (prereg):")
    for k, (v, t) in pr.items():
        L.append("  %s %s: %s" % (k, "held" if v else ("not scored" if v is None else "FAILED"), t))
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))
    return J


if __name__ == "__main__":
    a = sys.argv
    fr = a[a.index("--fresh") + 1].split(",") if "--fresh" in a else None
    main(a[1], a[2], a[3].split(","), fr)
