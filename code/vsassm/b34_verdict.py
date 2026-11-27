"""B34 verdict generator (prereg/B34_inmemory_directory_prereg.md): tables, gates, bars and the mechanical check of the
registered predictions, from the registered run's output only.

python3 b34_verdict.py results/b34_inmem.json results/b26_pcm_table.json > results/b34_verdict.txt
python3 b34_verdict.py DEV_FULL.json --dev          # the same tables for the synthetic dev run (no gates, no bars)
"""
import collections
import json
import sys

KS = (1, 2, 4, 8, 16, 32)
TIMES = ("60 s", "1 h", "1 d", "30 d", "1 y")
NONHD = ("T-i", "T-iih", "T-iis", "T-iix", "T-v")
HD = ("T-iiia", "T-iiib", "T-vhd", "T-vhdd")
ORDER = ("V-S", "V-Snt", "V-SC", "V-SCnt", "T-iv", "T-i", "T-iih", "T-iis", "T-iix", "T-v", "T-iiia", "T-iiib",
         "T-vhd", "T-vhdd")
TOT = tuple(a + "@tot" for a in NONHD)
PRIMARY_T = "30 d"


def share(tot, arm, k, tl):
    v = tot.get("%s|%s|x%g" % (arm, tl, k))
    return None if v is None else 100.0 * v[0] / v[1]


def best(tot, arms, k, tl):
    vals = [(share(tot, a, k, tl), a) for a in arms if share(tot, a, k, tl) is not None]
    return max(vals) if vals else (None, None)


def classify(deltas):
    """deltas: {k: arm - best table, points} for k >= 4."""
    ahead = [k for k, d in deltas.items() if d >= 5]
    behind = [k for k, d in deltas.items() if d <= -5]
    if all(abs(d) <= 2 for d in deltas.values()):
        return "TIE"
    if ahead and behind:
        return "MIXED"
    if ahead and not any(d < -2 for d in deltas.values()):
        return "VSA WIN" + (" (strict too)" if all(d >= 5 for d in deltas.values()) else " (not strict)")
    if behind and not any(d > 2 for d in deltas.values()):
        return "TABLE WIN"
    return "UNRESOLVED"


def bar(tot, arm, rivals, tl, label):
    ds, parts = {}, []
    for k in (4, 8, 16, 32):
        a = share(tot, arm, k, tl)
        b, who = best(tot, rivals, k, tl)
        ds[k] = a - b
        parts.append("x%g %s %.2f vs %s %.2f (%+.2f)" % (k, arm, a, who, b, a - b))
    return classify(ds), ds, "%s @ %s: %s" % (label, tl, "; ".join(parts))


def main():
    dev = "--dev" in sys.argv
    d = json.load(open(sys.argv[1]))
    tot = d["totals"]
    per = d["per_job"]
    n_keys = None
    print("# B34 %s (prereg/B34_inmemory_directory_prereg.md): the VSA store against tables whose directory is on the" % (
        "DEV RUN (synthetic stores)" if dev else "verdict"))
    print("# same simulated PCM devices, at equal device count; share of probed keys read exactly (%)")
    print()
    stores = sorted({r["store"] for r in per})
    for a in ORDER + TOT:
        if share(tot, a, 1, "60 s") is None:
            continue
        n_keys = tot["%s|60 s|x1" % a][1]
    print("Stores %d, probed keys %d. Choice: %s" % (len(stores), n_keys, json.dumps(d.get("choice"))))
    print()
    print("Table 1: share of the probed keys read exactly, by arm, noise multiplier k and time (budget C = the VSA")
    print("store's cells; @tot = equal total devices with V-SC, i.i.d. cell errors at the measured rate)")
    print("  %-10s %-4s %s" % ("arm", "k", "  ".join("%7s" % t for t in TIMES)))
    for a in ORDER + TOT:
        if share(tot, a, 1, "60 s") is None:
            continue
        for k in KS:
            print("  %-10s x%-3g %s" % (a, k, "  ".join("%7.2f" % share(tot, a, k, t) for t in TIMES)))
    print()
    # parameters per arm (cells used, repetition, widths), by store group
    print("Table 2: design parameters (median over the stores of each group; cells/C = share of the budget used)")
    groups = collections.defaultdict(list)
    for r in per:
        if r["k"] != 1:
            continue
        g = "/".join(r["store"].split("/")[:2]) if not dev else r["store"].rsplit("/", 1)[0]
        groups[g].append(r)
    for g, rs in sorted(groups.items()):
        r0 = rs[0]
        print("  %s (n stores %d; pairs %d, values %d, S %d, C %d cells)" % (g, len(rs), r0["n_pairs"], r0["n_values"],
                                                                           r0["S"], r0["C"]))
        for a, p in sorted(r0["params"].items()):
            print("     %-10s %s" % (a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()
                                        if k not in ("arm",)}))
    print()
    if dev:
        for tl in TIMES:
            for arm in ("V-S", "V-Snt"):
                print(bar(tot, arm, NONHD, tl, "%s vs best non-HD" % arm)[2])
        return
    # ---- gates
    print("Gates")
    g0 = []
    for a in ORDER + TOT:
        if share(tot, a, 1, "60 s") is None:
            continue
        lo = min(share(tot, a, 1, t) for t in TIMES)
        g0.append((lo, a))
    print("  G0 (k = 1, every arm >= 99.5%% at every time): %s (lowest %s %.2f)" % (
        "PASS" if min(g0)[0] >= 99.5 else "FAIL", min(g0)[1], min(g0)[0]))
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        b26 = {r["store"]: r["res"] for r in json.load(open(sys.argv[2]))["per_store"]}
        mism, checked = [], 0
        for r in per:
            for t in TIMES:
                for mine, theirs in (("V-S", "VSA"), ("T-iv", "Trep")):
                    a = r["res"].get("%s|%s" % (mine, t))
                    b = b26[r["store"]].get("%s | x%g | %s" % (theirs, r["k"], t))
                    if a is None or b is None:
                        continue
                    checked += 1
                    if a != b:
                        mism.append((r["store"], r["k"], t, mine, a, b))
        print("  G1 (V-S and T-iv reproduce B26b's banked counts per store, k, time): %s (%d checked, %d mismatches)%s" % (
            "PASS" if checked and not mism else "FAIL", checked, len(mism), (" e.g. %s" % mism[:3]) if mism else ""))
    print()
    # ---- bars
    print("Bars (30 days primary; Delta = arm - best rival, points; VSA ahead >= +5, tie within 2)")
    out = {}
    for name, arm, rivals, label in (("H1", "V-S", NONHD, "V-S vs best non-HD in-memory table (budget C)"),
                                     ("H2", "V-Snt", NONHD, "V-Snt vs best non-HD in-memory table (budget C)"),
                                     ("H3", "V-SC", TOT, "V-SC vs best non-HD table at equal total devices"),
                                     ("H3c", "V-SC", NONHD, "V-SC vs best non-HD table at budget C (favours V-SC)"),
                                     ("H4", "V-S", HD, "V-S vs best HD table"),
                                     ("H4nt", "V-Snt", HD, "V-Snt vs best HD table")):
        c, ds, txt = bar(tot, arm, rivals, PRIMARY_T, label)
        out[name] = (c, ds)
        print("  %s %s -> %s" % (name, txt, c))
        others = [(tl, bar(tot, arm, rivals, tl, label)[0]) for tl in TIMES if tl != PRIMARY_T]
        print("      other times: %s" % ", ".join("%s %s" % x for x in others))
    print()
    print("Per user-named design, V-S minus the design at 30 days (points), k = 4 / 8 / 16 / 32:")
    for a in ("T-i", "T-iih", "T-iis", "T-iix", "T-v", "T-iiia", "T-iiib", "T-vhd", "T-vhdd", "T-iv"):
        print("  %-7s %s" % (a, "  ".join("%+7.2f" % (share(tot, "V-S", k, PRIMARY_T) - share(tot, a, k, PRIMARY_T))
                                          for k in (4, 8, 16, 32))))
    print()
    predictions(tot, out)
    return tot, out


def predictions(tot, out):
    """The registered predictions (prereg, 'Predictions'), checked mechanically at 30 days unless stated."""
    S = lambda a, k, t=PRIMARY_T: share(tot, a, k, t)
    inr = lambda x, lo, hi: lo <= x <= hi
    best_nonhd = lambda k: best(tot, NONHD, k, PRIMARY_T)
    P = []
    P.append(("P1 T-i <= 10 at x4; T-iih <= 15 at x4", S("T-i", 4) <= 10 and S("T-iih", 4) <= 15,
              "%.2f, %.2f" % (S("T-i", 4), S("T-iih", 4))))
    P.append(("P2 T-iis >= 98 at x4, 80-95 at x8, 20-50 at x16, <= 10 at x32",
              S("T-iis", 4) >= 98 and inr(S("T-iis", 8), 80, 95) and inr(S("T-iis", 16), 20, 50) and S("T-iis", 32) <= 10,
              "%.2f %.2f %.2f %.2f" % tuple(S("T-iis", k) for k in (4, 8, 16, 32))))
    P.append(("P3 T-iix >= 99.5 at x4 and x8, 60-85 at x16, 5-20 at x32",
              S("T-iix", 4) >= 99.5 and S("T-iix", 8) >= 99.5 and inr(S("T-iix", 16), 60, 85) and inr(S("T-iix", 32), 5, 20),
              "%.2f %.2f %.2f %.2f" % tuple(S("T-iix", k) for k in (4, 8, 16, 32))))
    P.append(("P4 T-v >= 99.5 at x8, 85-97 at x16, 10-30 at x32; T-v the best non-HD table at x16 and x32",
              S("T-v", 8) >= 99.5 and inr(S("T-v", 16), 85, 97) and inr(S("T-v", 32), 10, 30)
              and best_nonhd(16)[1] == "T-v" and best_nonhd(32)[1] == "T-v",
              "%.2f %.2f %.2f; best x16 %s, x32 %s" % (S("T-v", 8), S("T-v", 16), S("T-v", 32), best_nonhd(16)[1],
                                                         best_nonhd(32)[1])))
    d1 = out["H1"][1]
    allt = all(bar(tot, "V-S", NONHD, tl, "")[0] == "TABLE WIN" for tl in TIMES)
    P.append(("P5 H1 TABLE WIN; |Delta| <= 2 at x4 and x8, Delta <= -80 at x16; TABLE WIN at every time",
              out["H1"][0] == "TABLE WIN" and abs(d1[4]) <= 2 and abs(d1[8]) <= 2 and d1[16] <= -80 and allt,
              "%s %s; every time %s" % (out["H1"][0], {k: round(x, 2) for k, x in d1.items()}, allt)))
    P.append(("P6 V-Snt >= 99.5 at x8, 97-100 at x16, 65-90 at x32",
              S("V-Snt", 8) >= 99.5 and inr(S("V-Snt", 16), 97, 100) and inr(S("V-Snt", 32), 65, 90),
              "%.2f %.2f %.2f" % (S("V-Snt", 8), S("V-Snt", 16), S("V-Snt", 32))))
    d2 = out["H2"][1]
    allt2 = all(bar(tot, "V-Snt", NONHD, tl, "")[0].startswith("VSA WIN") for tl in TIMES)
    P.append(("P7 H2 VSA WIN (not strict): ties at x4 and x8, Delta +2..+12 at x16, >= +30 at x32; at every time",
              out["H2"][0].startswith("VSA WIN") and abs(d2[4]) <= 2 and abs(d2[8]) <= 2 and inr(d2[16], 2, 12)
              and d2[32] >= 30 and allt2,
              "%s %s; every time %s" % (out["H2"][0], {k: round(x, 2) for k, x in d2.items()}, allt2)))
    P.append(("P8 T-iiib, T-vhd, T-vhdd >= 99.5 at x16; x32: T-iiib 35-60, T-vhd 45-70, T-vhdd 70-90; T-iiia 80-95 at x16",
              all(S(a, 16) >= 99.5 for a in ("T-iiib", "T-vhd", "T-vhdd")) and inr(S("T-iiib", 32), 35, 60)
              and inr(S("T-vhd", 32), 45, 70) and inr(S("T-vhdd", 32), 70, 90) and inr(S("T-iiia", 16), 80, 95),
              "x16 %.2f %.2f %.2f; x32 %.2f %.2f %.2f; T-iiia x16 %.2f" % (
                  S("T-iiib", 16), S("T-vhd", 16), S("T-vhdd", 16), S("T-iiib", 32), S("T-vhd", 32), S("T-vhdd", 32),
                  S("T-iiia", 16))))
    d4 = out["H4nt"][1]
    P.append(("P9 H4 TABLE WIN; H4nt: |Delta| <= 2 at x4, x8, x16 and V-Snt - T-vhdd in [-8, +2] at x32",
              out["H4"][0] == "TABLE WIN" and all(abs(d4[k]) <= 2 for k in (4, 8, 16))
              and inr(S("V-Snt", 32) - S("T-vhdd", 32), -8, 2),
              "H4 %s; H4nt %s, V-Snt - T-vhdd at x32 %+.2f" % (out["H4"][0], {k: round(x, 2) for k, x in d4.items()},
                                                               S("V-Snt", 32) - S("T-vhdd", 32))))
    P.append(("P10 V-SC <= 5 at x8; V-SCnt 95-100 at x8 and <= 10 at x16; H3 TABLE WIN; T-v@tot and T-iix@tot >= 99 at x32",
              S("V-SC", 8) <= 5 and inr(S("V-SCnt", 8), 95, 100) and S("V-SCnt", 16) <= 10 and out["H3"][0] == "TABLE WIN"
              and S("T-v@tot", 32) >= 99 and S("T-iix@tot", 32) >= 99,
              "%.2f; %.2f %.2f; %s; %.2f %.2f" % (S("V-SC", 8), S("V-SCnt", 8), S("V-SCnt", 16), out["H3"][0],
                                                 S("T-v@tot", 32), S("T-iix@tot", 32))))
    print("Predictions (registered; mechanical check, reporting only)")
    held = 0
    for name, ok, val in P:
        held += bool(ok)
        print("  %s: %s -> %s" % (name, val, "HELD" if ok else "WRONG"))
    print("  %d of %d held (P11 gates and P12 closed form: see the gate lines and results/b34_check.txt)" % (held, len(P)))
    print()


if __name__ == "__main__":
    main()
