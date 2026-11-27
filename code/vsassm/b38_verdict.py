"""B38 verdict (prereg/B38_ecc_equal_cells_prereg.md): tables, bars, predictions.

python3 b38_verdict.py CHOICE.json TEST.json[,TEST2.json,...] STORES.json.gz OUT.txt [CHECK files...]
"""
import collections
import json
import math
import sys

import numpy as np

import b38_design as Dz
import b38_eq as E

GRID = E.GRID
NG = len(GRID)
VSA_FREE = Dz.VSA_FREE
TABLES = Dz.TABLES
B512 = Dz.gidx(512.0)


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


MAXEV = {}


def fmtB(gi, top=None):
    """N* as a budget; a curve that never reaches 99.5% shows '>' and the largest budget evaluated."""
    if gi is None:
        return ">%.0f" % GRID[top] if top is not None else ">4096"
    return "%.0f" % GRID[gi]


def steps(gi):
    return NG if gi is None else gi                           # not reached by 4096 counts as one step beyond


def curves(acc, choice, conds, variants):
    out = {}
    for v in variants:
        for c in conds:
            cur, ns = Dz.test_nstar(acc, choice, v, c)
            out[(v, c)] = (cur, ns)
    return out


def best_of(cv, vs, c):
    ns = [(steps(cv[(v, c)][1]), v) for v in vs if (v, c) in cv and cv[(v, c)][0]]
    if not ns:
        return None, None
    s, v = min(ns)
    return (None if s >= NG else s), v


def label_delta(d):
    if d >= 2:
        return "VSA fewer cells"
    if d <= -2:
        return "table fewer cells"
    if abs(d) <= 1:
        return "tie"
    return "?"


def overall(labels):
    v = any(l == "VSA fewer cells" for l in labels)
    t = any(l == "table fewer cells" for l in labels)
    if v and t:
        return "MIXED"
    if v:
        return "VSA WIN"
    if t:
        return "TABLE WIN"
    return "TIE"


def bootstrap_delta(results, choice, cond, B=2000, seed=38):
    """Store-cluster bootstrap of steps(best table N*) - steps(best VSA N*) at one condition."""
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])))
    sids = sorted({r["sid"] for r in results})
    for r in results:
        for v, d, x in Dz.variant_rows(r):
            if x["cond"] != cond:
                continue
            per[r["sid"]][(v, d)][Dz.gidx(x["B"])][0] += x["exact"]
            per[r["sid"]][(v, d)][Dz.gidx(x["B"])][1] += x["n"]
    rng = np.random.default_rng(seed)
    m = E.parse_cond(cond)["match"]
    out = []
    for _ in range(B):
        samp = rng.integers(0, len(sids), len(sids))
        agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for i in samp:
            for key, gd in per[sids[i]].items():
                for gi, a in gd.items():
                    agg[key][gi][0] += a[0]
                    agg[key][gi][1] += a[1]

        def ns(v):
            cur = {}
            for gi in range(NG):
                if v == "V-am" or v in ("T-vhdd", "E-LDPC", "V-cb"):
                    d = Dz.design_for(choice, v, m, gi)
                elif v in ("E-BCH", "E-CAM"):
                    d = "auto"
                else:
                    d = None
                a = agg.get((v, d), {}).get(gi)
                if a and a[1]:
                    cur[gi] = a[0] / a[1]
            return steps(Dz.nstar(cur)) if cur else NG
        sv = min(ns(v) for v in VSA_FREE)
        st = min(ns(v) for v in TABLES)
        out.append(st - sv)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main(choice_f, test_fs, stores_f, out_f, checks):
    choice = json.load(open(choice_f))
    acc, results = Dz.test_curves(test_fs.split(","))
    stores = E.load_stores(stores_f)
    L = []
    w = L.append
    iid = list(E.IID)
    dmg = [c for c in E.all_conditions() if c not in E.IID]
    variants = Dz.VARIANTS
    cv = curves(acc, choice, iid + dmg, variants)
    nkeys = sum(x["n"] for r in results for x in r["res"] if "cond" in x and x["cond"] == "k1" and
                Dz.gidx(x["B"]) == B512 and r["arm"] == "T-rep")
    w("# B38 verdict (prereg/B38_ecc_equal_cells_prereg.md): cells per stored pair needed for >= 99.5% exact reads")
    w("# of the probed keys, VSA store vs error-corrected tables, every stored bit on simulated PCM cells")
    w("")
    w("Stores %d, probed keys %d (T-rep rows at B = 512, k1)." % (len(stores), nkeys))
    # capacity bound per condition (directory-free table at the BSC capacity): max over stores
    capb = {}
    for c in iid:
        eps = E.IID[c]
        C = 1 - h2(eps)
        need = []
        for sid, pairs in stores:
            nv = len({v for _, v, _ in pairs})
            multi = max(j for _, _, j in pairs) > 0
            need.append((math.log2(max(nv, 2)) + (1 if multi else 0)) / C)
        capb[c] = max(need)
    # ---------------- Table 1: N* under i.i.d. errors
    w("")
    w("Table 1: N* = the smallest budget (cells per stored pair, every stored bit counted) with >= 99.5% of probed")
    w("keys read exactly at that budget and every larger one evaluated; i.i.d. cell errors at the device model's")
    w("30-day bit-error rate x k (e05: 5% control). '>4096' = not reached on the grid.")
    w("  %-8s" % "" + "".join("%9s" % c for c in iid))
    w("  %-8s" % "eps" + "".join("%9.4f" % E.IID[c] for c in iid))
    for v in variants:
        w("  %-8s" % v + "".join("%9s" % fmtB(cv[(v, c)][1], max(cv[(v, c)][0]) if cv[(v, c)][0] else None)
                                 for c in iid))
    w("  %-8s" % "capacity" + "".join("%9.1f" % capb[c] for c in iid) + "   (directory-free table at BSC capacity, "
                                                                          "max over stores)")
    # ---------------- bars H1, H1cb, H2
    w("")
    w("Bars (Delta = grid steps of the best non-HD table's N* minus the best VSA read's N*; one step = 2^(1/4) = 19%;")
    w("'VSA fewer cells' if Delta >= +2, 'table fewer cells' if Delta <= -2, 'tie' if |Delta| <= 1)")
    lab1, rows1 = [], []
    for c in iid:
        sv, vv = best_of(cv, VSA_FREE, c)
        st, vt = best_of(cv, TABLES, c)
        d = steps(st) - steps(sv)
        lab = label_delta(d)
        if c != "e05":
            lab1.append(lab)
        rows1.append("    %-4s best VSA %-6s %6s | best table %-6s %6s | Delta %+d -> %s" % (
            c, vv, fmtB(sv), vt, fmtB(st), d, lab))
    w("  H1 (primary; k1..k32): " + overall(lab1))
    L.extend(rows1)
    lab, rows = [], []
    for c in iid:
        sv = cv[("V-cb", c)][1]
        st, vt = best_of(cv, TABLES, c)
        d = steps(st) - steps(sv)
        if c != "e05":
            lab.append(label_delta(d))
        rows.append("    %-4s V-cb %6s | best table %-6s %6s | Delta %+d -> %s" % (c, fmtB(sv), vt, fmtB(st), d,
                                                                                 label_delta(d)))
    w("  H1cb (VSA with its codebook on the cells, counted): " + overall(lab))
    L.extend(rows)
    lab, rows = [], []
    for c in iid:
        sv, vv = best_of(cv, VSA_FREE, c)
        sh = cv[("T-vhdd", c)][1]
        d = steps(sh) - steps(sv)
        if c != "e05":
            lab.append(label_delta(d))
        rows.append("    %-4s best VSA %-6s %6s | T-vhdd %6s | Delta %+d -> %s" % (c, vv, fmtB(sv), fmtB(sh), d,
                                                                                   label_delta(d)))
    w("  H2 (best VSA vs the HD table T-vhdd): " + overall(lab))
    L.extend(rows)
    # ---------------- Table 2: damage
    w("")
    w("Table 2: N* under localized damage (each damaged cell reads a fixed random sign; f = 2 eps of the matched")
    w("i.i.d. condition, so the overall cell-error rate matches). il = random interleaver over the array; nat = natural")
    w("layout (vectors / codewords / copies contiguous); known = locations given to every reader (erasures).")
    hdr = "  %-22s" % "condition" + "".join("%8s" % v for v in variants) + "  | Delta(VSA vs table)"
    w(hdr)
    lab3 = collections.defaultdict(list)
    for c in dmg:
        pc = E.parse_cond(c)
        sv, _ = best_of(cv, VSA_FREE, c)
        st, _ = best_of(cv, TABLES, c)
        d = steps(st) - steps(sv)
        grp = "known" if pc["known"] else pc["layout"]
        lab3[grp].append(label_delta(d))
        w("  %-22s" % c + "".join("%8s" % fmtB(cv[(v, c)][1], max(cv[(v, c)][0]) if cv[(v, c)][0] else None)
                                  for v in variants) + "  | %+d %s" % (d, label_delta(d)))
    w("  H3 (interleaved, unknown locations): " + overall(lab3["il"]))
    w("  H4 (natural layout, unknown locations): " + overall(lab3["nat"]))
    w("  H5 (known locations, lines, interleaved): " + overall(lab3["known"]))
    # position independence per variant: |steps(damage) - steps(matched iid)| <= 1
    w("")
    w("  Position independence (|N*(damage) - N*(matched i.i.d.)| <= 1 grid step), per variant: share of damage")
    w("  conditions that pass, interleaved / natural:")
    for v in variants:
        r = {"il": [0, 0], "nat": [0, 0]}
        for c in dmg:
            pc = E.parse_cond(c)
            if pc["known"]:
                continue
            a = steps(cv[(v, c)][1])
            b = steps(cv[(v, pc["match"])][1])
            r[pc["layout"]][0] += abs(a - b) <= 1
            r[pc["layout"]][1] += 1
        w("    %-8s il %d/%d   nat %d/%d" % (v, r["il"][0], r["il"][1], r["nat"][0], r["nat"][1]))
    # ---------------- Table 3: equal cells
    w("")
    w("Table 3: share of probed keys read exactly at equal cells, B = 512 cells per pair (B34's budget), designs")
    w("chosen for each condition")
    w("  %-22s" % "condition" + "".join("%8s" % v for v in variants))
    for c in iid + dmg:
        row = []
        for v in variants:
            cur = cv[(v, c)][0]
            row.append("%8.2f" % (100 * cur[B512]) if B512 in cur else "%8s" % "-")
        w("  %-22s" % c + "".join(row))
    w("")
    w("  Equal cells at the best VSA read's own N* (each arm's share at that budget):")
    for c in iid:
        sv, vv = best_of(cv, VSA_FREE, c)
        if sv is None:
            continue
        row = []
        for v in variants:
            cur = cv[(v, c)][0]
            row.append("%s %s" % (v, "%.2f" % (100 * cur[sv]) if sv in cur else "-"))
        w("    %-4s B = %s: %s" % (c, fmtB(sv), "; ".join(row)))
    # ---------------- Table 3b: M2b, equal cells below the tables' N* (under-provisioned memories)
    w("")
    w("Table 3b (M2b): equal cells where the memories are under-provisioned (i.i.d.): at each budget from 8 grid steps")
    w("below the best table's N* up to it, the best VSA read's share minus the best non-HD table's share (points);")
    w("'V' = VSA reads >= 5 points more, 'T' = table >= 5 more, '.' = within 5")
    lab7 = collections.Counter()
    for c in iid:
        st, _ = best_of(cv, TABLES, c)
        if st is None:
            continue
        cells = []
        for gi in range(max(0, st - 8), st + 1):
            bv = [cv[(v, c)][0].get(gi) for v in VSA_FREE]
            bt = [cv[(v, c)][0].get(gi) for v in TABLES]
            bv = [x for x in bv if x is not None]
            bt = [x for x in bt if x is not None]
            if not bv or not bt:
                continue
            dd = 100 * (max(bv) - max(bt))
            tag = "V" if dd >= 5 else "T" if dd <= -5 else "."
            if gi < st:
                lab7[tag] += 1
            cells.append("%s:%+.0f%s" % (fmtB(gi), dd, tag))
        w("    %-4s %s" % (c, " ".join(cells)))
    w("  H7 (under-provisioned, budgets below the best table's N*): VSA reads more at %d, table at %d, within 5 at %d" % (
        lab7["V"], lab7["T"], lab7["."]))
    # ---------------- Table 4: M3 graceful degradation
    if choice.get("B_g"):
        gi = Dz.gidx(choice["B_g"])
        w("")
        w("Table 4 (M3): graceful degradation at equal cells, B_g = %s cells per pair, every design fixed for k8" %
          fmtB(gi))
        m3 = list(E.M3_EPS) + ["%s-nat-%s" % (g, m) for g in ("lines", "contig") for m in E.M3_EPS]
        w("  %-18s" % "condition" + "".join("%8s" % v for v in variants) + "  | best VSA - best table")
        lab6 = []
        m3vals = {}
        for c in m3:
            row, vals = [], {}
            for v in variants:
                if v == "V-am":
                    d = Dz.design_for(choice, v, "k8", gi)
                    a = acc[v].get(d, {}).get(c, {}).get(gi)
                elif v in ("E-BCH", "E-CAM"):
                    a = acc[v].get("auto@k8", {}).get(c, {}).get(gi)
                elif v in ("V-pub", "V-Snt", "V-last", "T-rep"):
                    a = acc[v][None].get(c, {}).get(gi)
                else:
                    d = Dz.design_for(choice, v, "k8", gi)
                    a = acc[v].get(d, {}).get(c, {}).get(gi)
                if a and a[1]:
                    vals[v] = 100 * a[0] / a[1]
                row.append("%8.2f" % vals[v] if v in vals else "%8s" % "-")
            bv = max([vals[v] for v in VSA_FREE if v in vals] or [float("nan")])
            bt = max([vals[v] for v in TABLES if v in vals] or [float("nan")])
            m3vals[c] = dict(vals, bestV=bv, bestT=bt)
            dd = bv - bt
            lab = "VSA ahead" if dd >= 5 else "table ahead" if dd <= -5 else "tie" if abs(dd) <= 2 else "small"
            if c != "k8":
                lab6.append(lab)
            w("  %-18s" % c + "".join(row) + "  | %+.2f %s" % (dd, lab))
        w("  H6 (graceful degradation beyond the design point): VSA ahead at %d, table ahead at %d, tie at %d of %d" % (
            lab6.count("VSA ahead"), lab6.count("table ahead"), lab6.count("tie"), len(lab6)))
    # ---------------- predictions (registered; mechanical check, reporting only)
    w("")
    w("Predictions (prereg; mechanical check, reporting only)")
    dev = choice["dev_nstar"]
    ks = ["k1", "k2", "k4", "k8", "k16", "k32"]

    def ns_(v, c):
        return cv[(v, c)][1]

    def best_set(vs, c):
        st = min(steps(ns_(v, c)) for v in vs)
        return {v for v in vs if steps(ns_(v, c)) == st}, st

    res_p = []
    # P1
    deltas = {c: steps(best_of(cv, TABLES, c)[0]) - steps(best_of(cv, VSA_FREE, c)[0]) for c in ks}
    allowed = {"k1": {"E-BCH", "T-rep"}, "k2": {"E-BCH", "T-rep"}, "k4": {"E-BCH", "E-LDPC"}, "k8": {"E-LDPC"},
               "k16": {"E-LDPC"}, "k32": {"E-LDPC"}}
    idok = all(best_set(TABLES, c)[0] & allowed[c] for c in ks)
    vok = all(best_set(VSA_FREE, c)[0] & {"V-last", "V-Snt"} for c in ks)
    p1 = overall(lab1) == "TABLE WIN" and all(d <= -3 for d in deltas.values()) and idok and vok
    res_p.append(("P1 H1 TABLE WIN, Delta <= -3 at every k, best table / VSA identities",
                  "%s; Delta %s; best tables %s; best VSA %s" % (overall(lab1), deltas,
                  {c: sorted(best_set(TABLES, c)[0]) for c in ks}, {c: sorted(best_set(VSA_FREE, c)[0]) for c in ks}), p1))
    # P2
    misses = []
    for v in ("V-last", "E-LDPC", "E-BCH", "T-vhdd"):
        for c in ks:
            d0 = dev[v][c]
            t0 = ns_(v, c)
            if abs(steps(t0) - (Dz.gidx(d0) if d0 else NG)) > 2:
                misses.append("%s %s dev %s test %s" % (v, c, d0 and round(d0), fmtB(t0)))
    sv, _ = best_of(cv, VSA_FREE, "k1")
    st, _ = best_of(cv, TABLES, "k1")
    r3 = (GRID[sv] / GRID[st]) if (sv is not None and st is not None) else float("nan")
    res_p.append(("P2 test N* within 2 steps of dev (V-last, E-LDPC, E-BCH, T-vhdd); VSA >= 3x best table at k1",
                  "misses %s; k1 ratio %.2f" % (misses or "none", r3), not misses and r3 >= 3))
    # P3
    cbd = {c: steps(ns_("V-cb", c)) - steps(best_of(cv, TABLES, c)[0]) for c in ks}
    p3 = all(x >= 8 for x in cbd.values()) and ns_("V-cb", "k16") is None and ns_("V-cb", "k32") is None
    res_p.append(("P3 V-cb >= 8 steps above the best table at every k; not reached at k16, k32", str(cbd), p3))
    # P4
    h2d = {c: steps(ns_("T-vhdd", c)) - steps(best_of(cv, VSA_FREE, c)[0]) for c in ks}
    p4 = all(x <= -1 for x in h2d.values()) and sum(x <= -2 for x in h2d.values()) >= 4
    res_p.append(("P4 H2: T-vhdd fewer cells, Delta <= -1 everywhere, <= -2 at >= 4 of 6", str(h2d), p4))
    # P5
    pi = {}
    for v in ("V-last", "E-LDPC", "E-BCH", "T-rep", "T-vhdd"):
        n_ok = 0
        for c in dmg:
            pc = E.parse_cond(c)
            if pc["known"] or pc["layout"] != "il":
                continue
            n_ok += abs(steps(ns_(v, c)) - steps(ns_(v, pc["match"]))) <= 1
        pi[v] = n_ok
    p5 = overall(lab3["il"]) == "TABLE WIN" and all(x >= 10 for x in pi.values())
    res_p.append(("P5 H3 TABLE WIN; position-independent in >= 10 of 12 interleaved conditions",
                  "%s; %s" % (overall(lab3["il"]), pi), p5))
    # P6
    vs_ok, tb_bad = [], []
    for g in ("lines", "clust"):
        for m in ("k4", "e05", "k8"):
            c = "%s-nat-%s" % (g, m)
            vs_ok.append(abs(steps(ns_("V-last", c)) - steps(ns_("V-last", m))) <= 2)
        for m in ("e05", "k8", "k16"):
            c = "%s-nat-%s" % (g, m)
            for v in ("T-rep", "E-BCH", "E-LDPC", "E-CAM", "T-vhdd"):
                if steps(ns_(v, c)) - steps(ns_(v, m)) < 4:
                    tb_bad.append("%s %s" % (v, c))
    p6 = overall(lab3["nat"]) == "VSA WIN" and all(vs_ok) and not tb_bad
    res_p.append(("P6 H4 VSA WIN; V-last within 2 steps (lines/clust nat, k4/e05/k8); every table >= 4 steps worse",
                  "%s; V-last ok %d/%d; table exceptions %s" % (overall(lab3["nat"]), sum(vs_ok), len(vs_ok),
                                                                tb_bad or "none"), p6))
    # P7
    cn = [c for c in dmg if c.startswith("contig-nat")]
    none_all = all(ns_(v, c) is None for c in cn for v in variants)
    p7b, notes = True, []
    for c in cn:
        f = E.parse_cond(c)["f"]
        cur = cv[("V-last", c)][0]
        vl = cur.get(B512)
        others = [cv[(v, c)][0].get(B512) for v in ("T-rep", "E-BCH", "E-LDPC", "T-vhdd")]
        ok = vl is not None and abs(100 * vl - 100 * (1 - f)) <= 3 and all(o is None or vl >= o for o in others)
        p7b &= ok
        notes.append("%s V-last %.1f vs 1-f %.1f" % (c, 100 * vl if vl is not None else float("nan"), 100 * (1 - f)))
    res_p.append(("P7 contig-nat: no arm reaches 99.5%; V-last within 3 points of 1-f at 512 and >= coded tables",
                  "none reached %s; %s" % (none_all, "; ".join(notes)), none_all and p7b))
    # P8
    res_p.append(("P8 H5 TABLE WIN", overall(lab3["known"]), overall(lab3["known"]) == "TABLE WIN"))
    # P9
    if choice.get("B_g"):
        g = lambda c, k: m3vals.get(c, {}).get(k, float("nan"))
        a1 = g("e20", "E-LDPC") - g("e20", "bestV") >= 5 and g("e24", "E-LDPC") - g("e24", "bestV") >= 5
        a2 = g("k16", "bestV") - g("k16", "E-LDPC") >= 20 and g("e32", "bestV") - g("e32", "E-LDPC") >= 20
        a3 = all(m3vals["lines-nat-%s" % m]["bestV"] - m3vals["lines-nat-%s" % m]["bestT"] >= 5 for m in E.M3_EPS)
        a4 = lab6.count("VSA ahead") >= 8
        res_p.append(("P9 M3: E-LDPC ahead >= 5 at 0.20, 0.24; VSA ahead >= 20 at 0.276, 0.32; lines-nat VSA ahead at "
                      "every eps; H6 VSA ahead >= 8 of 20",
                      "%s %s %s %s (VSA ahead %d)" % (a1, a2, a3, a4, lab6.count("VSA ahead")), a1 and a2 and a3 and a4))
    # P10
    tot7 = sum(lab7.values())
    p10 = tot7 > 0 and lab7["V"] >= 2 * tot7 / 3 and lab7["T"] <= tot7 / 4
    res_p.append(("P10 H7: VSA reads more at >= 2/3 of under-provisioned budgets, table at <= 1/4",
                  "V %d, T %d, within %d of %d" % (lab7["V"], lab7["T"], lab7["."], tot7), p10))
    for name, val, ok in res_p:
        w("  %s: %s -> %s" % (name, val, "HELD" if ok else "FAILED"))
    w("  %d of %d held (P11, the checks, in the check sections below)" % (sum(r[2] for r in res_p), len(res_p)))
    # ---------------- bootstrap
    w("")
    w("Store-cluster bootstrap (2,000 resamples of the stores) of Delta (grid steps, best table - best VSA), 95% CI:")
    for c in ("k4", "k8", "k16", "k32"):
        lo, hi = bootstrap_delta(results, choice, c)
        w("    %-4s [%+.0f, %+.0f]" % (c, lo, hi))
    # ---------------- design parameters at N*
    w("")
    w("Design parameters at N* (the store with the most pairs):")
    big = max(results, key=lambda r: r["n_pairs"])["sid"]
    for c in ("k4", "k8", "k16", "k32"):
        for v in ("V-last", "E-BCH", "E-LDPC", "T-rep", "T-vhdd"):
            ns = cv[(v, c)][1]
            if ns is None:
                continue
            want = None if v in ("V-last", "T-rep") else ("auto" if v == "E-BCH" else
                                                          Dz.design_for(choice, v, c, ns))
            for r in results:
                if r["sid"] != big or Dz.FAMILY[v] != r["arm"]:
                    continue
                hit = [x for x in r["res"] if "params" in x and Dz.gidx(x["B"]) == ns and
                       (v == "E-BCH" or want is None or json.dumps(x["design"]) == want)]
                if hit:
                    w("    %-4s %-7s B=%s %s" % (c, v, fmtB(ns), json.dumps(hit[0]["params"])[:150]))
                    break
    for f in checks:
        w("")
        w("== check file %s" % f)
        try:
            ck = json.load(open(f))
            if isinstance(ck, dict) and "rows" in ck:
                w("  closed form: %d cells, %d flagged (|sim - cf| > max(1 point, 4 SE)); largest |diff| %.2f" % (
                    ck["n_cells"], ck["flagged"], max(abs(r["diff"]) for r in ck["rows"]) if ck["rows"] else 0))
                for r in ck["rows"]:
                    if r["flag"]:
                        w("    flagged %s" % json.dumps(r))
            elif isinstance(ck, list):
                for r in ck:
                    w("    %s" % json.dumps(r))
        except Exception:
            w(open(f).read())
    open(out_f, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
