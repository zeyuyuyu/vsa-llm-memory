"""B73 equal-cell verdict (prereg/B73_cells_prereg.md): pooled N* per memory at k1 / k8 / k16 on the 50 fresh stores,
the registered criteria D1-D4 and O1, per-family N*, the store bootstrap, the predictions.

python3 b73_cells_verdict.py STORES.json.gz FROZEN.json TEST.json[,EXT.json] OUT_PREFIX
"""
import collections
import json
import random
import sys

import b38_eq as E
import b58_verdict as V58
import b73_cells as T

GRID, NG, CONDS = T.GRID, T.NG, T.CONDS
FAMS = T.FAMILIES
NBOOT = 2000
# criteria: (table, store, allowed Delta range (criterion), exact range, label required)
CRIT = collections.OrderedDict([
    ("D1", ("static", "U", (-9, -3), (-8, -4), "TABLE WIN")),
    ("D2", ("static", "H-SBC", (-5, -1), (-4, -2), "TABLE WIN")),
    ("D3", ("insert", "H-SBC", (-1, 1), (-1, 1), "TIE")),
    ("D4", ("insert", "U", (-6, -1), (-5, -2), "TABLE WIN")),
])
PAPER_GI = {m: {c: T.gidx(v) for c, v in d.items()} for m, d in T.PAPER.items()}


def nstar_all(C, sids, weights=None):
    """NS[(mem, c)] = (gi or NG, censored flag) for the seven memories and static / insert."""
    NS = {}
    for m in T.MEMS:
        for c in CONDS:
            cur = C.get((m, c), {})
            if not cur:
                NS[(m, c)] = (NG, "no rows")
                continue
            ns, sh = V58.nstar_counts(cur, sids, weights)
            if ns is None:
                ns = NG
            flag = ""
            if sh:
                if sh[max(sh)] < T.TARGET:
                    flag = "not reached (top %s)" % T.fmt(max(sh))
                elif ns == min(sh) and ns > 0:
                    flag = "lowest evaluated"
            NS[(m, c)] = (ns, flag)
    for agg, arms in (("static", T.STATIC), ("insert", T.INSERT)):
        for c in CONDS:
            best = min(arms, key=lambda a: (NS[(a, c)][0], arms.index(a)))
            NS[(agg, c)] = (NS[(best, c)][0], NS[(best, c)][1], best)
    return NS


def criteria(NS):
    out = collections.OrderedDict()
    for k, (tab, st, (lo, hi), (elo, ehi), lab) in CRIT.items():
        ds = {}
        decid = True
        for c in CONDS:
            a, b = NS[(tab, c)][0], NS[(st, c)][0]
            if a >= NG or b >= NG:
                decid = False
            ds[c] = a - b
        label = V58.h_label([ds[c] for c in CONDS])
        inr = all(lo <= ds[c] <= hi for c in CONDS)
        exact = all(elo <= ds[c] <= ehi for c in CONDS)
        conf = decid and inr and (label == lab)
        out[k] = dict(table=tab, store=st, delta=ds, label=label,
                      ratio={c: round(2 ** (-ds[c] / 4), 3) for c in CONDS}, in_criterion_range=inr,
                      exact_range=exact and decid and label == lab, decidable=decid, confirmed=conf)
    o1 = all(NS[("static", c)][0] < NS[("H-SBC", c)][0] < NS[("U", c)][0] and NS[("insert", c)][0] < NS[("U", c)][0]
             for c in CONDS) and all(NS[(m, c)][0] < NG for m in ("static", "H-SBC", "U", "insert") for c in CONDS)
    out["O1"] = dict(confirmed=o1, order={c: [T.fmt(NS[(m, c)][0]) for m in ("static", "insert", "H-SBC", "U")]
                                          for c in CONDS})
    return out


def main(stores_path, frozen_path, test_paths, out_prefix):
    fz = json.load(open(frozen_path))
    stores = E.load_stores(stores_path)
    sids = [s for s, _ in stores]
    fam_of = {s: T.family(s) for s in sids}
    C = T.counts(fz, test_paths)
    NS = nstar_all(C, sids)
    CR = criteria(NS)
    J = dict(nstar={}, flags={}, criteria=CR, family={}, boot={}, coverage={}, predictions={}, lost={}, nq={})
    for (m, c), v in NS.items():
        J["nstar"]["%s|%s" % (m, c)] = None if v[0] >= NG else GRID[v[0]]
        J["flags"]["%s|%s" % (m, c)] = v[1:]
    for m in T.MEMS:
        for c in CONDS:
            J["coverage"]["%s|%s" % (m, c)] = [round(GRID[g], 1) for g in sorted(C.get((m, c), {}))
                                                if all(s in C[(m, c)][g] for s in sids)]
    # probed keys and lost insertions
    lost = collections.defaultdict(int)
    nq = {}
    for p in test_paths:
        for r in T.jl(p):
            nq[r["sid"]] = r.get("nq")
            if r["mem"] in T.INSERT and "lost" in r:
                lost[(r["sid"], r["mem"], json.dumps(r["design"]))] = max(lost[(r["sid"], r["mem"],
                                                                                json.dumps(r["design"]))], r["lost"])
    J["nq"] = dict(total=sum(v for v in nq.values() if v), stores=len(nq))
    J["lost"] = {"|".join(k): v for k, v in lost.items() if v}
    L = ["# B73 equal-cell verdict (prereg/B73_cells_prereg.md): 50 fresh stores (b73cells generator), frozen designs",
         "# (B38 / B61 / B63 dev choices), independent cell errors at the 30-day rate x 1 / 8 / 16, vocabulary uncharged.",
         "# N* = cells per stored pair for >= 99.5%% of probed keys read exactly, pooled over the 50 stores (%d probed keys)."
         % J["nq"]["total"],
         "# Delta = steps(table) - steps(store), one step = x 2^(1/4); ratio = store's N* / table's N* = 2^(-Delta/4).", ""]
    L.append("Pooled N* (fresh) [paper]")
    L.append("  %-8s" % "" + "".join("%22s" % c for c in CONDS))
    for m in T.MEMS + ("static", "insert"):
        cells = []
        for c in CONDS:
            v = NS[(m, c)]
            s = "%s [%s]" % (T.fmt(v[0]), T.fmt(PAPER_GI[m][c]))
            if m in ("static", "insert"):
                s = "%s=%s" % (v[2], s)
            if v[1]:
                s += "*"
            cells.append(s)
        L.append("  %-8s" % m + "".join("%22s" % x for x in cells))
    fl = ["%s %s: %s" % (m, c, NS[(m, c)][1]) for m in T.MEMS for c in CONDS if NS[(m, c)][1]]
    L.append("  * censoring flags: " + ("; ".join(fl) if fl else "none"))
    L.append("")
    L.append("Criteria (confirmed iff at k1, k8 and k16: label as registered and Delta in the criterion range)")
    for k, v in CR.items():
        if k == "O1":
            L.append("  O1 ordering static < H-SBC < U and insert < U at every k: %s  (static / insert / H-SBC / U: %s)"
                     % ("CONFIRMED" if v["confirmed"] else "NOT CONFIRMED",
                        "; ".join("%s %s" % (c, "/".join(v["order"][c])) for c in CONDS)))
            continue
        tab, st, rng_, ex, lab = CRIT[k]
        L.append("  %s %s vs %s: Delta %s, ratio %s, label %s (required %s), criterion range %s -> %s; exact range %s: %s"
                 % (k, tab, st, " / ".join("%+d" % v["delta"][c] for c in CONDS),
                    " / ".join("%.2f" % v["ratio"][c] for c in CONDS), v["label"], lab, list(rng_),
                    "CONFIRMED" if v["confirmed"] else "NOT CONFIRMED", list(ex), "held" if v["exact_range"] else
                    "not held"))
    L.append("")
    # per family
    L.append("Per family (secondary): N* static / insert / H-SBC / U and Deltas D1 D2 D3 D4")
    for f in FAMS:
        ss = [s for s in sids if fam_of[s] == f]
        NSf = nstar_all(C, ss)
        CRf = criteria(NSf)
        J["family"][f] = dict(nstar={"%s|%s" % (m, c): (None if v[0] >= NG else GRID[v[0]]) for (m, c), v in NSf.items()},
                              criteria=CRf)
        L.append("  %-8s " % f + " | ".join(
            "%s %s/%s/%s/%s" % (c, T.fmt(NSf[("static", c)][0]), T.fmt(NSf[("insert", c)][0]),
                                T.fmt(NSf[("H-SBC", c)][0]), T.fmt(NSf[("U", c)][0])) for c in CONDS))
        L.append("  %-8s " % "" + "; ".join("%s %s %s" % (k, "/".join("%+d" % CRf[k]["delta"][c] for c in CONDS),
                                                          CRf[k]["label"]) for k in CRIT))
    L.append("")
    # bootstrap
    rng = random.Random("b73cells-boot")
    byf = {f: [s for s in sids if fam_of[s] == f] for f in FAMS}
    hold = collections.Counter()
    for b in range(NBOOT):
        w = collections.Counter()
        for f, ss in byf.items():
            for _ in ss:
                w[rng.choice(ss)] += 1
        NSb = nstar_all(C, sids, w)
        CRb = criteria(NSb)
        for k, v in CRb.items():
            hold[k] += v["confirmed"]
            if k != "O1":
                hold[k + "_exact"] += v["exact_range"]
    J["boot"] = {k: hold[k] / NBOOT for k in sorted(hold)}
    L.append("Store bootstrap (secondary; %d draws, stores resampled within family): share of draws in which each criterion "
             "holds: %s" % (NBOOT, ", ".join("%s %.3f" % (k, v) for k, v in J["boot"].items())))
    L.append("")
    # predictions
    within = [(m, c) for m in T.MEMS for c in CONDS if NS[(m, c)][0] < NG and abs(NS[(m, c)][0] - PAPER_GI[m][c]) <= 1]
    q1 = len(within) >= 19
    q2 = all(CR[k]["confirmed"] for k in ("D1", "D2", "D3", "D4", "O1")) and CR["D1"]["exact_range"] and \
        CR["D2"]["exact_range"]
    hs = byf["wp6k"]
    NSw = nstar_all(C, hs)
    q3a = NS[("static", "k1")][2] in ("T-rep", "E-BCH") and GRID[NS[("static", "k1")][0]] < 24
    fam_h = {f: nstar_all(C, byf[f])[("H-SBC", "k1")][0] for f in FAMS}
    q3b = fam_h["wp6k"] == max(fam_h.values())
    big = {s for s, p in stores if len(p) >= 1000}
    q4 = not any(k.split("|")[0] in big for k in J["lost"])
    CRw = criteria(NSw)
    q5 = all(CRw[k]["label"] == CR[k]["label"] for k in CRIT)
    J["predictions"] = dict(Q1=[q1, len(within), 21], Q2=q2, Q3=[q3a and q3b, q3a, q3b], Q4=q4, Q5=q5)
    L.append("Predictions: Q1 %s (%d of 21 within one step of the paper); Q2 %s; Q3 %s (static at k1 by T-rep/E-BCH < 24: %s;"
             " H-SBC's k1 N* set by wp6k: %s, per family %s); Q4 %s (lost insertions: %s); Q5 %s"
             % ("held" if q1 else "FAILED", len(within), "held" if q2 else "FAILED", "held" if q3a and q3b else "FAILED",
                q3a, q3b, {f: T.fmt(g) for f, g in fam_h.items()}, "held" if q4 else "FAILED",
                J["lost"] or "none", "held" if q5 else "FAILED"))
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4])
