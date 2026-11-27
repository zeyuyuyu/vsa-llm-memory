"""B59 part 2 verdict (prereg/B59_maps_all_encoders_protected_full_prereg.md): per-family test curves (U = uncoded
store, P-BCH, P-LDPC), P-best (largest test share among the three at every budget), P-dev (the dev-selected union
design), N*, H1-P against B38's best table, G, store-cluster bootstrap.

python3 b59_store_verdict.py CHOICE.json PLAN.json OUT_PREFIX B59_TEST1.json[,...]
"""
import collections
import gzip
import json
import sys

import numpy as np

import b38_eq as E
import b58_store as S
import b58_verdict as V58

GRID = E.GRID
NG = len(GRID)
IIDC = list(E.IID)
TABLES = ("T-rep", "E-BCH", "E-LDPC", "E-CAM")
DMG4 = ["lines-nat-k4", "clust-nat-k4", "lines-nat-k8", "clust-nat-k8"]
TABLES_DMG = ("T-rep", "E-BCH", "E-LDPC*", "E-CAM")             # E-LDPC re-scored with OWN (B58 part c)
FAMS = ("U", "P-BCH", "P-LDPC")


def jload(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def steps(gi):
    return NG if gi is None else gi


def fmt(gi):
    return ">4096" if gi is None else "%.0f" % GRID[gi]


def merged_plan(paths):
    out = {"tiers": {}}
    for i, p in enumerate(paths):
        for t, need in json.load(open(p))["tiers"].items():
            out["tiers"]["%s_%d" % (t, i)] = need
    return out


def counts(choice, plan, test_paths):
    """C[(family, cond)] = {gi: {sid: (failed, n)}} for the plan's evaluations (B38's U rows, B58's reused rows that
    match the plan, B59's rows)."""
    want = set()
    for t, need in plan["tiers"].items():
        for k, cs in need:
            for c in cs:
                want.add((k[0], round(k[1], 4), k[2], k[3], c))
    C = collections.defaultdict(dict)
    # U from B38's test (rule 'last')
    for r in jload("results/b38_test.json.gz"):
        if r["arm"] != "V-last":
            continue
        for x in r["res"]:
            if "cond" in x and x["cond"] in IIDC + DMG4 and not x.get("infeasible") and x.get("rule") == "last":
                C[("U", x["cond"])].setdefault(S.gidx(x["B"]), {})[r["sid"]] = (len(x["failed"]), x["n"])
    src = collections.Counter()
    for p in list(test_paths) + ["results/b58_test_iid1p.json.gz", "results/b58_test_iid2.json.gz"]:
        b58 = "b58_test" in p
        for r in jload(p):
            if r.get("arm") == "V-last":                         # B59 T1 rows (b38_eq.run_job format)
                for x in r["res"]:
                    if "cond" in x and not x.get("infeasible") and x.get("rule") == "last":
                        C[("U", x["cond"])].setdefault(S.gidx(x["B"]), {})[r["sid"]] = (len(x["failed"]), x["n"])
                        src["U-b59"] += 1
                continue
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                gi = S.gidx(x["B"])
                k = (x["var"], round(x["b"], 4), x["R"], gi, x["cond"])
                if k not in want:
                    continue
                d = C[(x["var"], x["cond"])].setdefault(gi, {})
                if r["sid"] in d and b58:
                    continue                                     # a B59 row exists (identical by construction)
                f = 10 ** 6 if x["censored"] else len(x["failed"])
                d[r["sid"]] = (f, x["n"])
                src[("b58" if b58 else "b59") + "-" + x["var"]] += 1
    T = V58.b38_counts()
    for v in TABLES:
        for c in IIDC + DMG4:
            if (v, c) in T:
                C[(v, c)] = T[(v, c)]
    for p in ("results/b58_ldpc_rescore1p.json.gz", "results/b58_ldpc_rescore2.json.gz"):
        for r in jload(p):
            for x in r["res"]:
                if "cond" in x and x["cond"] in DMG4 and not x.get("infeasible"):
                    C[("E-LDPC*", x["cond"])].setdefault(S.gidx(x["B"]), {})[r["sid"]] = (len(x["failed"]), x["n"])
    return C, src


def shares(curve, sids, w=None):
    return V58.nstar_counts(curve, sids, w)


def nstar_from(sh):
    best = None
    for gi in sorted(sh, reverse=True):
        if sh[gi] >= E.TARGET:
            best = gi
        else:
            break
    return best


def pbest(C, c, sids, w=None):
    sh = {}
    for f in FAMS:
        if (f, c) in C:
            _, s = shares(C[(f, c)], sids, w)
            for gi, v in s.items():
                sh[gi] = max(sh.get(gi, -1.0), v)
    return nstar_from(sh), sh


def pdev(C, ch, c, sids, w=None):
    fs = {f: shares(C[(f, c)], sids, w)[1] for f in FAMS if (f, c) in C}
    sh = {}
    for g, (fam, des) in ch["union"][c].items():
        gi = int(g)
        if fam in fs and gi in fs[fam]:
            sh[gi] = fs[fam][gi]
    return nstar_from(sh), sh


def main(choice_path, plan_path, out_prefix, test_paths):
    ch = json.load(open(choice_path))
    plan = merged_plan(plan_path.split(","))
    sids = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    C, src = counts(ch, plan, test_paths)
    L = ["# B59 part 2 verdict (prereg/B59_maps_all_encoders_protected_full_prereg.md): the protected store on the full",
         "# grid with the uncoded store U inside the search. N* in cells per stored pair; '>4096' = not reached.",
         "# P-best = largest test share among U, P-BCH, P-LDPC at every budget; P-dev = dev-selected union design.",
         "# Evaluation sources: %s" % dict(src), ""]
    J = {"nstar": {}, "coverage": {}, "H1P": {}, "H1Pdev": {}, "G": {}, "boot": {}}
    NS = {}
    for f in FAMS + TABLES:
        for c in IIDC:
            if (f, c) not in C:
                continue
            ns, sh = shares(C[(f, c)], sids)
            NS[(f, c)] = ns
            J["nstar"]["%s|%s" % (f, c)] = None if ns is None else GRID[ns]
            if f in FAMS:
                ev = sorted(sh)
                full = [gi for gi in range(14 if f != "U" else 0, NG)]
                miss = [gi for gi in full if gi not in sh]
                J["coverage"]["%s|%s" % (f, c)] = dict(evaluated=len(ev), missing=[GRID[g] for g in miss])
    for c in IIDC:
        NS[("P-any", c)] = min((NS[(f, c)] if NS[(f, c)] is not None else NG for f in FAMS if (f, c) in NS), default=NG)
        NS[("P-any", c)] = None if NS[("P-any", c)] >= NG else NS[("P-any", c)]
        J["nstar"]["P-any|%s" % c] = None if NS[("P-any", c)] is None else GRID[NS[("P-any", c)]]
        NS[("P-best", c)], _ = pbest(C, c, sids)
        NS[("P-dev", c)], _ = pdev(C, ch, c, sids)
        J["nstar"]["P-best|%s" % c] = None if NS[("P-best", c)] is None else GRID[NS[("P-best", c)]]
        J["nstar"]["P-dev|%s" % c] = None if NS[("P-dev", c)] is None else GRID[NS[("P-dev", c)]]
    L.append("Table 1 (i.i.d.): test N*")
    L.append("  %-8s" % "" + "".join("%9s" % c for c in IIDC))
    for f in FAMS + ("P-best", "P-any", "P-dev") + TABLES:
        L.append("  %-8s" % f + "".join("%9s" % (fmt(NS.get((f, c))) if (f, c) in NS else "n/r") for c in IIDC))
    L.append("")
    L.append("Coverage of the full grid (budgets not evaluated on every store; codes 90.5-4096, U 8-4096):")
    for k, v in J["coverage"].items():
        if v["missing"]:
            L.append("  %-12s missing %d budgets: %s" % (k, len(v["missing"]), ", ".join("%.0f" % x for x in v["missing"])))
    for name, key in (("H1-P", "P-best"), ("H1-P (P-dev)", "P-dev")):
        L.append("")
        L.append("%s: Delta = steps(best table) - steps(%s); +: store fewer cells" % (name, key))
        ds = []
        jj = J["H1P"] if key == "P-best" else J["H1Pdev"]
        for c in IIDC:
            sb = steps(NS[(key, c)])
            tb, tv = min((steps(NS[(v, c)]), v) for v in TABLES if (v, c) in NS)
            dl = tb - sb
            if c != "e05":
                ds.append(dl)
            fam_at = None
            if key == "P-best" and NS[(key, c)] is not None:
                gi = NS[(key, c)]
                fam_at = [f for f in FAMS if (f, c) in C and shares(C[(f, c)], sids)[1].get(gi, -1) >= E.TARGET]
            L.append("    %-4s %-6s %6s %s| best table %-7s %6s | Delta %+d -> %s" % (
                c, key, fmt(NS[(key, c)]), ("(passing at N*: %s) " % ",".join(fam_at)) if fam_at else "", tv,
                fmt(None if tb >= NG else tb), dl, V58.label(dl)))
            jj[c] = dict(store_N=None if sb >= NG else GRID[sb], table=tv, table_N=None if tb >= NG else GRID[tb],
                         delta=dl, label=V58.label(dl), families_passing_at_N=fam_at)
        jj["label"] = V58.h_label(ds)
        L.append("  %s: %s" % (name, jj["label"]))
    L.append("")
    L.append("G (does coding help the store? steps(U) - steps(best code family); >= +2 helps, <= -2 hurts):")
    nh = 0
    for c in IIDC:
        cb = min(steps(NS.get((v, c))) for v in ("P-BCH", "P-LDPC") if (v, c) in NS)
        g = steps(NS.get(("U", c))) - cb
        lab = "coding helps" if g >= 2 else ("coding hurts" if g <= -2 else "tie")
        nh += lab == "coding helps"
        L.append("    %-4s U %6s  P-BCH %6s  P-LDPC %6s  G %+d -> %s" % (
            c, fmt(NS.get(("U", c))), fmt(NS.get(("P-BCH", c))), fmt(NS.get(("P-LDPC", c))), g, lab))
        J["G"][c] = dict(G=g, label=lab)
    J["G"]["share_helps"] = [nh, len(IIDC)]
    L.append("  coding helps at %d of %d conditions" % (nh, len(IIDC)))
    # bootstrap
    rng = np.random.default_rng(59)
    L.append("")
    L.append("Store-cluster bootstrap (2,000 resamples): 95% range of Delta (P-best), P-best N*, best table N*:")
    for c in IIDC:
        dl, pb, tb_ = [], [], []
        for it in range(2000):
            draw = rng.integers(0, len(sids), len(sids))
            w = collections.Counter(sids[i] for i in draw)
            s_ = steps(pbest(C, c, sids, w)[0])
            t_ = min(steps(V58.nstar_counts(C[(v, c)], sids, w)[0]) for v in TABLES if (v, c) in C)
            dl.append(t_ - s_)
            pb.append(s_)
            tb_.append(t_)

        def rg(a, f=lambda x: x):
            lo, hi = np.percentile(a, [2.5, 97.5])
            return f(int(round(lo))), f(int(round(hi)))
        g2 = lambda x: ">4096" if x >= NG else "%.0f" % GRID[x]
        L.append("    %-4s Delta [%+d, %+d]   P-best N* [%s, %s]   best table N* [%s, %s]" % (
            (c,) + rg(dl) + rg(pb, g2) + rg(tb_, g2)))
        J["boot"][c] = dict(delta=rg(dl), p_best=rg(pb, g2), table=rg(tb_, g2))
    # damage (T5)
    L.append("")
    L.append("Table 2 (damage, natural layout; f = 2 eps of the matched i.i.d. condition): test N*; best non-HD table "
             "with B58's re-scored E-LDPC*")
    L.append("  P-any = the smallest of the three families' own N* (each on its own evaluated budgets). P-best (largest")
    L.append("  share per budget) is shown but not used here: in damage U has only B38's budgets, so at budgets where only a")
    L.append("  failing code family was evaluated the per-budget maximum reads as a failure (an artefact; compute note 6).")
    L.append("  %-14s" % "" + "".join("%9s" % v for v in FAMS + ("P-any", "P-best*") + TABLES_DMG) + "  | Delta P-any vs best table | P-any within 1 step of P-any i.i.d.")
    J["damage"] = {}
    ds_d = []
    for c in DMG4:
        row = {}
        for f in FAMS + TABLES_DMG:
            if (f, c) in C:
                row[f] = shares(C[(f, c)], sids)[0]
        pa = min((steps(row[f]) for f in FAMS if f in row), default=NG)
        pb = pbest(C, c, sids)[0]
        tb, tv = min((steps(row[v]), v) for v in TABLES_DMG if v in row)
        dl = tb - pa
        ds_d.append(dl)
        m = E.parse_cond(c)["match"]
        pam = min(steps(NS[(f, m)]) for f in FAMS if (f, m) in NS)
        pi = abs(pa - pam) <= 1
        L.append("  %-14s" % c + "".join("%9s" % (fmt(row.get(f)) if f in row else "n/r") for f in FAMS)
                 + "%9s%9s" % (fmt(None if pa >= NG else pa), fmt(pb))
                 + "".join("%9s" % (fmt(row.get(f)) if f in row else "n/r") for f in TABLES_DMG)
                 + "  | %+d %s (vs %s) | %s" % (dl, V58.label(dl), tv, pi))
        J["damage"][c] = dict({f: (None if row.get(f) is None else GRID[row[f]]) for f in row},
                              P_any=None if pa >= NG else GRID[pa], P_best_maxshare=None if pb is None else GRID[pb],
                              delta=dl, label=V58.label(dl), table=tv, posind=pi)
    J["damage"]["label"] = V58.h_label(ds_d)
    L.append("  natural-layout damage, P-any vs the best non-HD table: %s" % J["damage"]["label"])
    # dev N* (reported)
    L.append("")
    L.append("Dev N* (12 dev stores): " + "; ".join(
        "%s %s" % (f, " ".join("%s=%s" % (c, ("%.0f" % x) if x else "-") for c, x in
                               ((c, (ch["U_dev_nstar"] if f == "U" else ch["union_dev_nstar"] if f == "P-dev" else
                                     ch["dev_nstar"][f])[c]) for c in IIDC))) for f in ("U", "P-BCH", "P-LDPC", "P-dev")))
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [p for p in sys.argv[4].split(",") if p] if len(sys.argv) > 4 else [])
