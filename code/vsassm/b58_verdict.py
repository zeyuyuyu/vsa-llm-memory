"""B58 verdict (prereg/B58_protected_store_prereg.md).

python3 b58_verdict.py TEST.json[,...] RESCORE.json[,...] OUT_PREFIX
-> OUT_PREFIX.txt / OUT_PREFIX.json (results/b58_verdict.*)
"""
import collections
import gzip
import json
import math
import sys

import numpy as np

import b38_design as D
import b38_eq as E
import b58_store as P

GRID = E.GRID
IIDC = list(E.IID)
DMG = P.DMG
NG = len(GRID)


def steps(gi):
    return NG if gi is None else gi                               # 'not reached by 4,096' = one step beyond the grid


def fmt(gi):
    return ">4096" if gi is None else "%.0f" % GRID[gi]


def label(delta):
    return "store fewer cells" if delta >= 2 else ("table fewer cells" if delta <= -2 else "tie")


def h_label(ds):
    s = [label(d) for d in ds]
    v, t = "store fewer cells" in s, "table fewer cells" in s
    return "MIXED" if v and t else ("VSA WIN" if v else ("TABLE WIN" if t else "TIE"))


# ---------------------------------------------------------------------------------------------- per-store counts
def b38_counts():
    """B38's registered table / store curves as per-store (failed, n) at each (variant, cond, gi), with the registered
    designs (b38_design.test_nstar's selection)."""
    res = json.load(gzip.open("results/b38_test.json.gz", "rt"))
    ch = json.load(open("results/b38_choice.json"))
    out = collections.defaultdict(dict)                          # (variant, cond) -> {gi: {sid: (fail, n)}}
    for r in res:
        for v, d, x in D.variant_rows(r):
            if v not in ("V-last", "V-cb", "T-rep", "E-BCH", "E-LDPC", "E-CAM", "T-vhdd") or x.get("infeasible"):
                continue
            c = x["cond"]
            m = E.parse_cond(c)["match"]
            gi = D.gidx(x["B"])
            if v in ("E-BCH", "E-CAM"):
                if d != "auto":
                    continue
            elif v not in ("V-last", "T-rep"):
                if d != D.design_for(ch, v, m, gi):
                    continue
            out[(v, c)].setdefault(gi, {})[r["sid"]] = (len(x["failed"]), x["n"])
    return out


def p_counts(paths):
    out = collections.defaultdict(dict)
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if x.get("infeasible"):
                    continue
                f = 10 ** 6 if x["censored"] else len(x["failed"])
                out[(x["var"], x["cond"])].setdefault(P.gidx(x["B"]), {})[r["sid"]] = (f, x["n"])
    return out


def rescore_counts(paths):
    out = collections.defaultdict(dict)
    for p in paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if "cond" not in x or x.get("infeasible"):
                    continue
                out[("E-LDPC*", x["cond"])].setdefault(D.gidx(x["B"]), {})[r["sid"]] = (len(x["failed"]), x["n"])
    return out


def nstar_counts(curve, sids, weights=None):
    """N* from per-store counts; weights = multiplicity of each store (bootstrap). A budget counts only if every store
    with weight > 0 was evaluated there."""
    share = {}
    for gi, d in curve.items():
        f = n = 0
        ok = True
        for s in sids:
            w = 1 if weights is None else weights.get(s, 0)
            if not w:
                continue
            if s not in d:
                ok = False
                break
            f += w * d[s][0]
            n += w * d[s][1]
        if ok and n:
            share[gi] = 1 - min(f, n) / n
    best = None
    for gi in sorted(share, reverse=True):
        if share[gi] >= E.TARGET:
            best = gi
        else:
            break
    return best, share


def h2(p):
    return 0.0 if p <= 0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def part_b(C, sids, tables, L, J):
    """Both-charged vocabulary variant: the value strings (UTF-8 bytes x 8) on the cells of every memory at the BSC
    capacity of the condition; N*' = N* + v c(eps). Pooled: N* over the 50 stores, v = max over stores; per family:
    N* over the family's stores, v = the family's max."""
    st = dict(E.load_stores("b34_out/test_stores.json.gz"))
    vb = {}
    for sid, pairs in st.items():
        vals = list(dict.fromkeys(v for _, v, _ in pairs))
        vb[sid] = sum(8 * len(v.encode("utf-8")) for v in vals) / len(pairs)
    fams = collections.OrderedDict([("pooled (50 stores)", sids)])
    for sid in sids:
        fams.setdefault(sid.rsplit("/", 1)[0], []).append(sid)
    L.append("")
    L.append("Part (b): value-vocabulary accounting. Audit: b25_pcm.codes builds the store's value codes from the")
    L.append("vocabulary strings (bip(vocab, 'value', d)); every table stores vocabulary indices (b34_inmem.Store.vi,")
    L.append("payload_bits) and T-vhdd regenerates its value codes from the same strings (hd_code('value:' + vocab[v]));")
    L.append("every arm returns vocabulary indices scored against the same index lists. Both free (B38) = symmetric for")
    L.append("V-last vs the tables; V-cb alone pays an item memory (its codebook) that no table has.")
    L.append("Both charged: vocabulary strings on the cells of every memory at 1 / (1 - H2(eps)) cells per bit;")
    L.append("N*' = N* + v c; Delta' = 4 log2(N*'(best table) / N*'(store)) grid steps (+: store fewer cells).")
    J["part_b"] = {}
    for fam, ss in fams.items():
        v = max(vb[s] for s in ss)
        L.append("  %s: v = %.1f vocabulary bits per pair" % (fam, v))
        for c in IIDC:
            cc = 1.0 / (1.0 - h2(E.IID[c]))
            ns = {}
            for a in ("V-last", "P-LDPC", "P-BCH", "V-cb") + tables:
                if (a, c) in C:
                    g, _ = nstar_counts(C[(a, c)], ss)
                    ns[a] = None if g is None else GRID[g]
            pany = [ns[a] for a in ("V-last", "P-LDPC", "P-BCH") if ns.get(a)]
            ns["P-any"] = min(pany) if pany else None
            tb = [ns[a] for a in tables if ns.get(a)]
            t = min(tb) if tb else None
            row = []
            for a in ("V-last", "P-any", "V-cb"):
                if ns.get(a) is None or t is None:
                    row.append("%s n/a" % a)
                    continue
                dfree = 4 * math.log2(t / ns[a])
                dch = 4 * math.log2((t + v * cc) / (ns[a] + v * cc))
                row.append("%s %.0f vs %.0f: free %+.1f, charged %+.1f (%s; difference %.0f cells per pair)" % (
                    a, ns[a], t, dfree, dch, label(int(round(dch))), ns[a] - t))
                J["part_b"]["%s|%s|%s" % (fam, c, a)] = dict(store=ns[a], table=t, v=v, c=cc, delta_free=dfree,
                                                            delta_charged=dch, label_charged=label(int(round(dch))))
            L.append("    %-4s c %.2f | %s" % (c, cc, " | ".join(row)))


def part_c(L, J):
    c1 = json.load(open("results/b58_dec_c1.json"))["totals"]
    c2 = json.load(open("results/b58_dec_c2.json"))
    emu = json.load(open("results/b58_dec_emu.json"))
    emc = json.load(open("results/b58_dec_emuclip.json"))
    port = json.load(open("results/b58_dec_port.json"))["totals"]
    L.append("")
    L.append("Part (c): the LDPC decoder discrepancy (PKG = ldpc package BpDecoder, B38's registered decoder; OWN =")
    L.append("b38_check._sp_decode, B38's second decoder).")
    L.append("  c1, B38's 1,007 check codewords: agree %d; PKG decodes %d, OWN %d; PKG posterior non-finite in %d of the %d"
             % (c1["agree"], c1["pkg_ok"], c1["own_ok"], c1["own_ok_pkg_fail_nonfinite"], c1["own_ok_pkg_fail"]))
    L.append("    codewords OWN decodes and PKG does not (PKG better in %d); inputs floored at 1e-6: agree %d (registered"
             % (c1["pkg_ok_own_fail"], c1["fl6_agree"]))
    L.append("    mechanism test, needed >= 99%: not met).")
    L.append("  Emulation of PKG's arithmetic (disclosed addition): identical to PKG on %d / %d codewords and %d / %d"
             % (emu["c1_emu_eq_pkg"], emu["c1_words"], emu["c2B_emu_eq_pkg"], emu["c2B_frames"]))
    L.append("    stuck-copy frames; with OWN's two guards added it decodes %d codewords (OWN %d, PKG %d), output identical"
             % (emc["clip_ok"], emc["own_ok"], emc["pkg_ok"]))
    L.append("    to OWN on %d / %d, no non-finite value -> cause: PKG's unguarded product-sum update (a tanh product that"
             % (emc["clip_eq_own"], emc["words"]))
    L.append("    rounds to +-1 gives an infinite message; inf - inf and inf x 0 give NaN; 'posterior <= 0' reads NaN as 'no error').")
    L.append("  c2, exhaustive ML on K = 16 IRA codes (400 frames per setting): block successes ML / PKG / OWN")
    ok = {"PKG": True, "OWN": True}
    closer = collections.Counter()
    for r in c2:
        for d, k in (("PKG", "pkg_ok"), ("OWN", "own_ok")):
            if r["ml_ok"] - r[k] > 0.03 * r["frames"]:
                ok[d] = False
        if r["pkg_ok"] != r["own_ok"]:
            closer["PKG" if r["pkg_ok"] > r["own_ok"] else "OWN"] += 1
            closer["%s_%s" % (r["chan"], "PKG" if r["pkg_ok"] > r["own_ok"] else "OWN")] += 1
    for chan in ("A", "B"):
        rows = [r for r in c2 if r["chan"] == chan]
        L.append("    channel %s (%s):" % (chan, "i.i.d. flips, rho copies" if chan == "A" else
                                           "share 2 eps of coded bits with every copy stuck, the rest clean"))
        for r in rows:
            if chan == "A" and r["ml_ok"] == r["pkg_ok"] == r["own_ok"] == r["frames"]:
                continue
            L.append("      R %.3f eps %.4f rho %2d: %3d / %3d / %3d" % (r["R"], r["eps"], r["rho"], r["ml_ok"],
                                                                     r["pkg_ok"], r["own_ok"]))
    L.append("    (channel A settings where all three decode every frame are omitted)")
    L.append("    within 3 points of ML in every setting: PKG %s, OWN %s -> registered rule: %s" % (
        ok["PKG"], ok["OWN"], "OWN used (neither qualifies)" if not (ok["PKG"] or ok["OWN"]) else
        ("both qualify" if ok["PKG"] and ok["OWN"] else ("PKG" if ok["PKG"] else "OWN"))))
    L.append("    settings where the two differ: OWN ahead %d (channel B %d), PKG ahead %d (channel B %d)" % (
        closer["OWN"], closer["B_OWN"], closer["PKG"], closer["B_PKG"]))
    L.append("  Implementation: batched OWN bit-identical to _sp_decode on %d / %d synthetic words (numba port: %d / %d, "
             "not used)." % (port["batch_equal"], port["words"], port["equal"], port["words"]))
    J["part_c"] = dict(c1=c1, emu=emu, emuclip=emc, port=port, within3=ok, closer=dict(closer))


def predictions(J, NS, L):
    """Q1-Q10 checked mechanically where the verdict holds the numbers (Q11 in results/b58_check.txt)."""
    out = {}
    h = J["H1-P"]["rows"]
    k6 = ("k1", "k2", "k4", "k8", "k16", "k32")
    out["Q1"] = J["H1-P"]["label"] == "TABLE WIN" and all(h[c]["delta"] <= -4 for c in k6)
    g = J["G"]
    out["Q2"] = all(g[c]["G"] < 2 for c in IIDC) and all(g[c]["G"] <= -2 for c in ("k8", "k16", "k32"))
    pb = {c: h[c]["store_N"] for c in IIDC}
    gi = lambda x: None if x is None else min(range(NG), key=lambda i: abs(GRID[i] - x))
    out["Q3"] = (pb["k1"] is not None and gi(pb["k1"]) == 14 and pb["k2"] is not None and gi(pb["k2"]) == 14 and
                 pb["k4"] is not None and abs(gi(pb["k4"]) - 14) <= 1)
    st = lambda v, c: steps(NS.get((v, c)))
    out["Q4"] = all(st("P-LDPC", c) <= st("P-BCH", c) for c in ("k8", "k16", "k32")) and \
        all(st("P-BCH", c) <= st("P-LDPC", c) for c in ("k1", "k2", "k4"))
    d = J.get("damage", {})
    pi = J.get("posind", {})
    if "label_il_P" in d:
        out["Q5"] = d["label_il_P"] == "TABLE WIN" and pi.get("P-best|il", [0, 1])[0] >= 7
        nat_moved = 0
        for gg in ("lines", "clust"):
            for m in E.DAMAGE_EPS:
                c = "%s-nat-%s" % (gg, m)
                a = min(steps(NS.get(("P-LDPC", c))), steps(NS.get(("P-BCH", c))))
                b = min(steps(NS.get(("P-LDPC", m))), steps(NS.get(("P-BCH", m))))
                nat_moved += a - b >= 2
        out["Q6"] = nat_moved >= 6
        out["Q6_count"] = nat_moved
    pbj = J.get("part_b", {})
    if pbj:
        pooled = [v for k, v in pbj.items() if k.startswith("pooled") and k.endswith("|V-last")]
        mk2 = [v for k, v in pbj.items() if k.startswith("131072/niah_multikey_2|") and k.endswith("|V-last") and
               k.split("|")[1] in ("k1", "k2", "k4", "k8")]
        out["Q7"] = all(abs(int(round(v["delta_charged"]))) <= 1 for v in pooled) and \
            all(int(round(v["delta_charged"])) <= -2 for v in mk2)
    c1 = J["part_c"]["c1"]
    out["Q8"] = (c1["own_ok_pkg_fail_nonfinite"] >= 0.9 * c1["own_ok_pkg_fail"]) and (c1["fl6_agree"] >= 0.99 * c1["words"])
    c2 = json.load(open("results/b58_dec_c2.json"))
    A_ok = all(r["ml_ok"] - r[k] <= 0.03 * r["frames"] for r in c2 if r["chan"] == "A" for k in ("pkg_ok", "own_ok"))
    B_own = all(r["ml_ok"] - r["own_ok"] <= 0.03 * r["frames"] for r in c2 if r["chan"] == "B")
    B_pkg = any(r["ml_ok"] - r["pkg_ok"] >= 0.10 * r["frames"] for r in c2 if r["chan"] == "B" and r["rho"] >= 16)
    out["Q9"] = A_ok and B_own and B_pkg
    out["Q9_parts"] = dict(A_both_within3=A_ok, B_own_within3=B_own, B_pkg_10_below=B_pkg)
    if ("E-LDPC*", "lines-nat-k4") in NS:
        e = NS[("E-LDPC*", "lines-nat-k4")]
        iid_same = "not re-scored (compute note 5)"
        out["Q10"] = (e is not None and GRID[e] <= 153) and J["fig5b"]["holds"]
        out["Q10_parts"] = dict(lines_nat_k4=None if e is None else GRID[e], fig5b=J["fig5b"]["holds"],
                                iid_within1=iid_same)
    L.append("")
    L.append("Predictions (mechanical; Q11 = checks, see results/b58_check.txt):")
    for k in sorted(out, key=lambda z: (len(z.split("_")[0]), z)):
        L.append("  %s: %s" % (k, out[k]))
    J["predictions"] = out


def main(test_paths, rescore_paths, out_prefix):
    sids = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    C = b38_counts()
    C.update(p_counts(test_paths))
    if rescore_paths:
        C.update(rescore_counts(rescore_paths))
    NS = {}
    SH = {}
    BOUND = {}
    for key, curve in C.items():
        NS[key], SH[key] = nstar_counts(curve, sids)
        comp = sorted(SH[key])
        if not comp:
            BOUND[key] = "not run"
        elif SH[key][comp[-1]] < E.TARGET:
            BOUND[key] = "> %.0f" % GRID[comp[-1]]
        elif key[0] not in ("V-last", "V-cb", "T-rep", "E-BCH", "E-LDPC", "E-CAM", "T-vhdd") and \
                NS[key] == comp[0]:
            BOUND[key] = "<= %.0f" % GRID[comp[0]]
        else:
            BOUND[key] = ""
    for key in list(NS):
        if BOUND.get(key) == "not run":
            del NS[key]
    ldpc = "E-LDPC*" if rescore_paths else "E-LDPC"
    tables_dmg = ("T-rep", "E-BCH", ldpc, "E-CAM")
    tables = ("T-rep", "E-BCH", "E-LDPC", "E-CAM")               # i.i.d.: B38's E-LDPC (compute note 5)
    L = []
    J = dict(nstar={"%s|%s" % k: (None if v is None else GRID[v]) for k, v in NS.items()})

    def cellf(v, c):
        b = BOUND.get((v, c))
        if b is None:
            return ""
        if b == "not run":
            return "n/r"
        if b.startswith(">"):
            return b.replace(" ", "")
        return ("<=" if b else "") + fmt(NS.get((v, c)))

    J["bounds"] = {"%s|%s" % k: v for k, v in BOUND.items() if v}

    def best(names, c):
        cand = [(steps(NS.get((v, c))), v) for v in names if (v, c) in NS]
        cand.sort()
        return cand[0] if cand else (NG, None)

    L.append("# B58 verdict (prereg/B58_protected_store_prereg.md): the VSA store protected by the tables' channel codes")
    L.append("Stores 50, probed keys 4776. N* in cells per stored pair (>4096 = not reached on the grid).")
    L.append("Best table = smallest N* of T-rep, E-BCH, E-LDPC, E-CAM; i.i.d.: B38's E-LDPC (OWN = PKG on i.i.d. inputs,")
    L.append("compute note 5); damage: %s. P-best = smaller of P-LDPC, P-BCH (n/r = not run, compute note 5);" % (
        "E-LDPC* = E-LDPC re-scored with OWN" if rescore_paths else "B38's E-LDPC"))
    L.append("'<=' = N* at the bottom of the cut window (at k1-k4 no code fits below 91 cells, so 91 is exact).")
    L.append("P-any = smaller of P-best and B38's uncoded V-last.")
    L.append("")
    L.append("Table 1 (i.i.d.): N*")
    hdr = "  %-10s" % "" + "".join("%9s" % c for c in IIDC)
    L.append(hdr)
    for v in ("V-last", "P-LDPC", "P-BCH", "T-rep", "E-BCH", "E-LDPC", "E-LDPC*", "E-CAM", "T-vhdd"):
        if any((v, c) in NS for c in IIDC):
            L.append("  %-10s" % v + "".join("%9s" % cellf(v, c) for c in IIDC))
    # H1-P, H1-Pany, G
    rows = {}
    for name, store_names in (("H1-P", ("P-LDPC", "P-BCH")), ("H1-Pany", ("P-LDPC", "P-BCH", "V-last"))):
        ds = []
        L.append("")
        L.append("%s (Delta = steps(best table) - steps(%s); +: store fewer cells):" % (
            name, "P-best" if name == "H1-P" else "P-any"))
        for c in IIDC:
            sb, sv = best(store_names, c)
            tb, tv = best(tables, c)
            dlt = tb - sb
            if c != "e05":
                ds.append(dlt)
            L.append("    %-4s %-7s %6s | best table %-8s %6s | Delta %+d -> %s" % (
                c, sv, fmt(None if sb >= NG else sb), tv, fmt(None if tb >= NG else tb), dlt, label(dlt)))
            rows.setdefault(name, {})[c] = dict(store=sv, store_N=None if sb >= NG else GRID[sb], table=tv,
                                                table_N=None if tb >= NG else GRID[tb], delta=dlt)
        lab = h_label(ds)
        L.append("  %s: %s" % (name, lab))
        J[name] = dict(label=lab, rows=rows[name])
    L.append("")
    L.append("G (does coding help the store? steps(V-last) - steps(P-best); >= +2 helps, <= -2 hurts):")
    J["G"] = {}
    for c in IIDC:
        sb, sv = best(("P-LDPC", "P-BCH"), c)
        g = steps(NS.get(("V-last", c))) - sb
        lab = "coding helps" if g >= 2 else ("coding hurts" if g <= -2 else "tie")
        L.append("    %-4s V-last %6s  P-best %6s (%s)  G %+d -> %s" % (c, fmt(NS.get(("V-last", c))),
                                                                     fmt(None if sb >= NG else sb), sv, g, lab))
        J["G"][c] = dict(G=g, label=lab)
    # damage
    L.append("")
    L.append("Table 2 (localized damage; f = 2 eps of the matched i.i.d. condition): N*")
    names = ("V-last", "P-LDPC", "P-BCH", "T-rep", "E-BCH", "E-LDPC", "E-LDPC*", "E-CAM", "T-vhdd")
    L.append("  %-16s" % "" + "".join("%9s" % v for v in names) + "  | Delta P-best vs table | Delta V-last vs table")
    J["damage"] = {}
    p_dmg = any(("P-LDPC", c) in NS or ("P-BCH", c) in NS for c in DMG)
    if not p_dmg:
        L.append("  (part (a) damage conditions not run, compute notes 2 and 4: P columns empty, P labels not evaluated)")
    for lay in ("il", "nat"):
        dsP, dsV = [], []
        for g in ("lines", "clust"):
            for m in E.DAMAGE_EPS:
                c = "%s-%s-%s" % (g, lay, m)
                sb, _ = best(("P-LDPC", "P-BCH"), c)
                tb, tv = best(tables_dmg, c)
                vb = steps(NS.get(("V-last", c)))
                dsP.append(tb - sb)
                dsV.append(tb - vb)
                L.append("  %-16s" % c + "".join("%9s" % cellf(v, c) for v in names) + "  | %s | %+d %s" % (
                    ("%+d %s" % (tb - sb, label(tb - sb))) if p_dmg else "not run", tb - vb, label(tb - vb)))
                J["damage"][c] = dict(P_best=None if sb >= NG else GRID[sb], table=tv,
                                      table_N=None if tb >= NG else GRID[tb], V_last=None if vb >= NG else GRID[vb],
                                      delta_P=tb - sb, delta_V=tb - vb)
        L.append("  %s: P-best vs table %s; V-last vs table %s" % (lay, h_label(dsP) if p_dmg else "not run",
                                                                 h_label(dsV)))
        if p_dmg:
            J["damage"]["label_%s_P" % lay] = h_label(dsP)
        J["damage"]["label_%s_V" % lay] = h_label(dsV)
    # position independence
    L.append("")
    L.append("Position independence (|N*(damage) - N*(matched i.i.d.)| <= 1 grid step), lines + clusters:")
    J["posind"] = {}
    for v in ("V-last", "P-LDPC", "P-BCH", "P-best", "T-rep", "E-BCH", "E-LDPC", "E-LDPC*", "E-CAM", "T-vhdd"):
        for lay in ("il", "nat"):
            k = 0
            tot = 0
            for g in ("lines", "clust"):
                for m in E.DAMAGE_EPS:
                    c = "%s-%s-%s" % (g, lay, m)
                    if v == "P-best":
                        if not p_dmg:
                            continue
                        a, _ = best(("P-LDPC", "P-BCH"), c)
                        b, _ = best(("P-LDPC", "P-BCH"), m)
                    else:
                        vm = "E-LDPC" if v == "E-LDPC*" else v       # i.i.d. E-LDPC from B38 (compute note 5)
                        if (v, c) not in NS or (vm, m) not in NS:
                            continue
                        a, b = steps(NS[(v, c)]), steps(NS[(vm, m)])
                    tot += 1
                    k += abs(a - b) <= 1
            if tot:
                J["posind"]["%s|%s" % (v, lay)] = [k, tot]
        if "%s|il" % v in J["posind"]:
            L.append("    %-8s il %d/%d   nat %d/%d" % ((v,) + tuple(J["posind"]["%s|il" % v]) +
                                                       tuple(J["posind"].get("%s|nat" % v, [0, 0]))))
    # Fig. 5b rule
    L.append("")
    nat_lead = sum(1 for g in ("lines", "clust") for m in E.DAMAGE_EPS
                   if J["damage"]["%s-nat-%s" % (g, m)]["delta_V"] >= 2)
    il_ldpc = J["posind"].get("%s|il" % ldpc, [0, 0])
    h3 = J["damage"]["label_il_V"]
    holds = nat_lead >= 6 and il_ldpc[0] >= 7 and h3 == "TABLE WIN"
    L.append("Fig. 5b sentence rule: nat lead >= 2 steps in %d / 8 (need >= 6); %s interleaved within 1 step of i.i.d. "
             "in %d / %d (need >= 7); interleaved best store read vs best table: %s (need TABLE WIN) -> %s" % (
                 nat_lead, ldpc, il_ldpc[0], il_ldpc[1], h3, "HOLDS" if holds else "DOES NOT HOLD as worded"))
    J["fig5b"] = dict(nat_lead=nat_lead, il_ldpc=il_ldpc, h3=h3, holds=holds)
    # bootstrap of Delta (H1-P) and of P-best N*
    rng = np.random.default_rng(58)
    L.append("")
    L.append("Store-cluster bootstrap (2,000 resamples): 95% range of Delta (H1-P), of P-best N* and of the best "
             "table N*:")
    J["boot"] = {}
    for c in IIDC:
        dl, pb, tb_ = [], [], []
        for it in range(2000):
            draw = rng.integers(0, len(sids), len(sids))
            w = collections.Counter(sids[i] for i in draw)
            s_best = min(steps(nstar_counts(C[(v, c)], sids, w)[0]) for v in ("P-LDPC", "P-BCH") if (v, c) in C)
            t_best = min(steps(nstar_counts(C[(v, c)], sids, w)[0]) for v in tables if (v, c) in C)
            if s_best >= NG:
                pass
            dl.append(t_best - s_best)
            pb.append(s_best)
            tb_.append(t_best)

        def rg(a, f=lambda x: x):
            lo, hi = np.percentile(a, [2.5, 97.5])
            return f(int(round(lo))), f(int(round(hi)))
        g2 = lambda x: ">4096" if x >= NG else "%.0f" % GRID[x]
        L.append("    %-4s Delta [%+d, %+d]   P-best N* [%s, %s]   best table N* [%s, %s]" % (
            (c,) + rg(dl) + rg(pb, g2) + rg(tb_, g2)))
        J["boot"][c] = dict(delta=rg(dl), p_best=rg(pb, g2), table=rg(tb_, g2))
    part_b(C, sids, tables, L, J)
    part_c(L, J)
    predictions(J, NS, L)
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    tp = [p for p in sys.argv[1].split(",") if p]
    rp = [p for p in sys.argv[2].split(",") if p] if len(sys.argv) > 3 else []
    main(tp, rp, sys.argv[-1])
