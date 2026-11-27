"""B54 verdict (prereg/B54_mlc_cells_prereg.md).

python3 b54_verdict.py CHOICE.json TEST1.json[,TEST2.json ...] OUT.txt [OUT.json]

Binary (1 bit per cell) N* come from B38 (30 days, results/b38_verdict.txt Table 1) and B52 (c) (1 day,
results/b52c_verdict.json), except T-vhdd at 1 day, run here (b = 1). The noise-free requirement of a binary memory is
its N* at k1 (bit-error rate 0, B38); of an MLC memory its N* on ideal cells (no device noise) at the same b.
"""
import collections
import json
import math
import re
import sys

import b54_mlc as M

MEM = ("V-last", "T-rep", "E-BCH", "E-LDPC", "T-vhdd")
TABLES = ("T-rep", "E-BCH", "E-LDPC")                         # B38's non-HD tables run here (E-CAM never best)
KS = M.KS
DEV = False


def b38_binary():
    txt = open("results/b38_verdict.txt").read().split("Table 1:")[1].split("Bars")[0]
    cols = ["k1", "k2", "k4", "e05", "k8", "k16", "k32"]
    out = {}
    for line in txt.splitlines():
        p = line.split()
        if p and p[0] in MEM and len(p) == 8:
            out[p[0]] = {c: (None if v.startswith(">") else float(v)) for c, v in zip(cols, p[1:])}
    return out


def steps(a, b):
    """Grid steps of N* b minus N* a (None = never reached = one step beyond 4096)."""
    ga = 4 * math.log2(a if a else 4096 * 2 ** 0.25)
    gb = 4 * math.log2(b if b else 4096 * 2 ** 0.25)
    return int(round(gb - ga))


def label(d):
    return "VSA fewer cells" if d >= 2 else ("table fewer cells" if d <= -2 else "tie")


def fmt(x):
    return "%7s" % (">4096" if x is None else "%.0f" % x if x >= 10 else "%.1f" % x)


def ratio(a, b):
    return None if (a is None or b is None) else a / b


def fr(x):
    return "%7s" % ("-" if x is None else "%.2f" % x)


def main(choice_f, test_fs, out_f, out_json=None):
    res = []
    for f in test_fs.split(","):
        res += json.load(open(f))
    acc = M.aggregate(res)
    N = collections.defaultdict(dict)                         # N[(mem, b)][(k or 'ideal', t)] = cells per pair
    nq = None
    for (arm, b), dd in acc.items():
        for c in M.conditions(b):
            seen = collections.Counter(gi for dk in dd for gi in dd[dk][c])
            dup = [gi for gi, n in seen.items() if n > 1]
            if dup and not DEV:
                raise SystemExit("duplicate evaluation %s m%d %s %s" % (arm, b, c, dup))
            curve, _ = M.best_curve(dd, c)                    # test: one design per budget; dev: the best one
            for dk in dd:
                for gi, a in dd[dk][c].items():
                    nq = a[1]
            ns = M.nstar(curve) if curve else None
            pc = M.parse(c)
            key = ("ideal", None) if pc["ideal"] else (pc["k"], pc["t"])
            N[(arm, b)][key] = None if ns is None else M.GRID[ns]
            if curve and ns is None and max(curve) < len(M.GRID) - 1:
                print("WARNING: %s m%d %s not bracketed (top evaluated %.0f)" % (arm, b, c, M.GRID[max(curve)]))
    bb = b38_binary()
    b52 = json.load(open("results/b52c_verdict.json"))["nstar_1d"]
    for m in MEM:
        for k in KS:
            N[(m, 1)][(k, "30d")] = bb[m]["k%d" % k]
            if m != "T-vhdd":
                N[(m, 1)][(k, "1d")] = b52[m]["k%d" % k]
        N[(m, 1)][("ideal", None)] = bb[m]["k1"]
    L = []
    L.append("# B54 verdict (prereg/B54_mlc_cells_prereg.md): cells per stored pair for >= 99.5% exact reads of the")
    L.append("# probed RULER keys with 1, 2 or 3 bits per cell (multi-level PCM cells, B25's device model)")
    L.append("")
    L.append("Probed keys %s; binary (b = 1) rows: B38 (30 d) and B52 (c) (1 d); T-vhdd b = 1 at 1 d: this run." % nq)
    cal_rates = {c: M.ber(c) for b in M.BITS for c in M.conditions(b)[1:]}
    L.append("")
    L.append("Calibrated mean bit-error rate per stored bit (uniform levels, Gray labels, calibrated thresholds):")
    for t in M.TIMES:
        L.append("  %-4s" % t + "".join("   k%-2d" % k + " ".join(" m%d %.4f" % (b, cal_rates["m%d-k%d-%s" % (b, k, t)])
                                                         for b in M.BITS) + (" m1 %.4f" % (M.P1D[k] if t == "1d" else
                                                                                       __import__("b38_eq").P30[k]))
                                        for k in KS))
    for t in M.TIMES:
        L.append("")
        L.append("Table 1%s: N* (cells per stored pair, every stored bit counted; >4096 = not reached), %s" % (
            "a" if t == "30d" else "b", "30 days" if t == "30d" else "1 day"))
        L.append("  %-8s %3s %7s" % ("memory", "b", "ideal") + "".join("%7s" % ("k%d" % k) for k in KS))
        for m in MEM:
            for b in (1,) + M.BITS:
                row = N.get((m, b), {})
                L.append("  %-8s %3d %s" % (m, b, fmt(row.get(("ideal", None)))) +
                         "".join(fmt(row.get((k, t))) for k in KS))
    for t in M.TIMES:
        L.append("")
        L.append("Table 2%s: overhead relative to the memory's own noise-free requirement, N*(k) / N*(ideal, same b), %s"
                 % ("a" if t == "30d" else "b", "30 days" if t == "30d" else "1 day"))
        L.append("  %-8s %3s" % ("memory", "b") + "".join("%7s" % ("k%d" % k) for k in KS))
        for m in MEM:
            for b in (1,) + M.BITS:
                row = N.get((m, b), {})
                L.append("  %-8s %3d" % (m, b) + "".join(fr(ratio(row.get((k, t)), row.get(("ideal", None))))
                                                          for k in KS))
    best = {}
    for t in M.TIMES:
        L.append("")
        L.append("Table 3%s: best over b in {1, 2, 3} per memory (N*, b), and overhead relative to the best noise-free"
                 " requirement over b, %s" % ("a" if t == "30d" else "b", "30 days" if t == "30d" else "1 day"))
        L.append("  %-8s %10s" % ("memory", "ideal") + "".join("%14s" % ("k%d" % k) for k in KS))
        for m in MEM:
            def bo(key):
                c = [(N[(m, b)].get(key), b) for b in (1,) + M.BITS if N[(m, b)].get(key) is not None]
                return min(c) if c else (None, None)
            i0 = bo(("ideal", None))
            cells = []
            for k in KS:
                v, b = bo((k, t))
                best[(m, k, t)] = v
                cells.append("%14s" % ("%s (b%s) %s" % (fmt(v).strip(), b, fr(ratio(v, i0[0])).strip())
                                       if v else ">4096"))
            best[(m, "ideal")] = i0[0]
            L.append("  %-8s %10s" % (m, "%s (b%s)" % (fmt(i0[0]).strip(), i0[1])) + "".join(cells))
    # bars
    bars = {}
    L.append("")
    L.append("Bars (Delta = grid steps of the best table's N* minus the VSA store's; one step = 19%;")
    L.append("'VSA fewer cells' if Delta >= +2, 'table fewer cells' if <= -2, 'tie' within 1)")
    for t in M.TIMES:
        for name, bset in (("H1 binary (b = 1)", (1,)), ("H1-MLC b = 2", (2,)), ("H1-MLC b = 3", (3,)),
                           ("H1-best (each memory at its best b)", (1, 2, 3))):
            labs = []
            L.append("  %s, %s:" % (name, t))
            for k in KS:
                vs = [(N[("V-last", b)].get((k, t)) or 1e9, b) for b in bset]
                tb = [(N[(m, b)].get((k, t)) or 1e9, m, b) for m in TABLES for b in bset]
                v, tbest = min(vs), min(tb)
                d = steps(v[0] if v[0] < 1e9 else None, tbest[0] if tbest[0] < 1e9 else None)
                labs.append(label(d))
                L.append("    k%-2d VSA %s (b%d) | best table %s %s (b%d) | Delta %+d -> %s" % (
                    k, fmt(v[0] if v[0] < 1e9 else None), v[1], tbest[1], fmt(tbest[0] if tbest[0] < 1e9 else None),
                    tbest[2], d, label(d)))
            win = "VSA fewer cells" in labs
            lose = "table fewer cells" in labs
            verdict = "MIXED" if win and lose else "VSA WIN" if win else "TABLE WIN" if lose else "TIE"
            bars["%s|%s" % (name, t)] = verdict
            L.append("    -> %s" % verdict)
    # relative overhead ranking (referee M2b-d)
    L.append("")
    L.append("H4 (robustness vs storage efficiency): overhead N*(k) / N*(noise-free) per memory; is the VSA store's the")
    L.append("smallest? (k8, k16; per b and at each memory's best b)")
    h4 = {}
    for t in M.TIMES:
        for k in (8, 16):
            for b in (1,) + M.BITS + ("best",):
                ov = {}
                for m in MEM:
                    if b == "best":
                        ov[m] = ratio(best.get((m, k, t)), best.get((m, "ideal")))
                    else:
                        ov[m] = ratio(N[(m, b)].get((k, t)), N[(m, b)].get(("ideal", None)))
                known = {m: v for m, v in ov.items() if v is not None}
                least = min(known, key=known.get) if known else None
                h4["%s|k%d|b%s" % (t, k, b)] = dict(overhead=ov, least=least)
                L.append("  %-3s k%-2d b=%-4s " % (t, k, b) + "  ".join("%s %s" % (m, fr(v).strip()) for m, v in ov.items())
                         + "  -> least: %s" % least)
    out = dict(N={"%s|%d" % mb: {"%s|%s" % kk: v for kk, v in row.items()} for mb, row in N.items()}, bars=bars, h4=h4)
    open(out_f, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(out, open(out_json, "w"), indent=1)
    print("\n".join(L))
    return out


if __name__ == "__main__":
    if "--dev" in sys.argv:
        DEV = True
        sys.argv.remove("--dev")
    main(*sys.argv[1:])
