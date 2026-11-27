"""B41 secondary report (after the test verdict): mechanical prediction checks P1-P11, accuracy at equal state per D,
the dead-mass split (start read correctly vs not), and the exploratory D = 2^18 runs (not in the bar).

  python b41_report.py --scores results/b41_test_verdict.json --runs /path/to/work/b41/runs \
      --runs_x /path/to/work/b41/runs_x --dev_head /path/to/work/b41/head/choice.json --out results/b41_report.txt
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_engine as EN                                                     # noqa: E402
import b41_verdict as V                                                     # noqa: E402

NS, HS = V.NS, V.HS
LV = ["low", "mid", "high"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--runs_x", default=None)
    ap.add_argument("--dev_head", default=None)
    ap.add_argument("--rhos", type=float, nargs=3, default=[0.03, 0.06, 0.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    S = json.load(open(a.scores))
    L = []
    P = {}
    k = lambda lv, h, N: "%s|h%d|N%d" % (lv, h, N)
    acc = lambda c, name: c["acc"][name][0]
    vsa_names = [n for n in V.VSAS]

    # P1 (dev naming statistics)
    if a.dev_head:
        st = json.load(open(a.dev_head))["dev_stats_at_T"]
        errs = [v["argmax_error"] for v in st.values()]
        share = [v["same_cluster_share_of_errors"] for v in st.values()]
        P["P1"] = (min(errs) <= 0.025 and max(errs) >= 0.25 and min(share) >= 0.6,
                   "dev error range %.3f-%.3f; in-cluster share of errors >= %.2f" % (min(errs), max(errs), min(share)))
    # P2
    a6 = [acc(S[k("high", 6, N)], "ARGMAX") for N in NS]
    a2 = [acc(S[k("low", 2, N)], "ARGMAX") for N in NS]
    P["P2"] = (max(a6) <= 0.10 and min(a2) >= 0.80, "ARGMAX high h6 %s; low h2 %s" % (np.round(a6, 3), np.round(a2, 3)))
    # P3
    g = [acc(S[k("high", h, N)], "EXACT") - acc(S[k("high", h, N)], "ARGMAX") for h in (4, 6) for N in NS]
    P["P3"] = (min(g) >= 0.30, "EXACT - ARGMAX at high h4/h6 over N: %s" % np.round(g, 3))
    # P4
    within, cheap = 0, []
    for c in S.values():
        if acc(c, "EXACT") - acc(c, "BEAM_k4_B16") <= 0.01:
            within += 1
    for lv in LV:
        for h in HS:
            for N in (3200, 32000, 100000):
                c = S[k(lv, h, N)]
                cheap.append(c["cost"]["BEAM_k4_B16"]["ops"] / c["cost"]["EXACT"]["ops"])
    P["P4"] = (within >= 40 and max(cheap) <= 0.01, "BEAM(4,16) within 0.01 of EXACT in %d/45 cells; ops ratio to "
                                                    "EXACT at N>=3200 max %.2e" % (within, max(cheap)))
    # P5
    end_v = [S[k(lv, 6, N)]["v_star"] for lv in LV for N in NS]
    endbad = []
    for lv in LV:
        for N in (3200, 32000, 100000):
            c = S[k(lv, 6, N)]
            be = max(acc(c, n) for n in vsa_names if n.startswith("VSA_END") and n in c["acc"])
            endbad.append(be <= acc(c, "ARGMAX"))
    P["P5"] = (not any("END" in v for v in end_v) and all(endbad),
               "v* modes at h6: %s; best END <= ARGMAX at N>=3200: %d/%d" % (
                   sorted(set(v.split("_")[1] for v in end_v)), sum(endbad), len(endbad)))
    # P6
    d6 = [acc(S[k("high", h, N)], S[k("high", h, N)]["v_acc"]) - acc(S[k("high", h, N)], "ARGMAX")
          for h in (4, 6) for N in (32, 320, 3200)]
    P["P6"] = (min(d6) >= 0.20, "most accurate VSA - ARGMAX at high, h4/h6, N<=3200: %s" % np.round(d6, 3))
    # P7
    over = []
    for key, c in S.items():
        bv = max(acc(c, n) for n in vsa_names if n in c["acc"])
        if bv > acc(c, "EXACT") + 0.02:
            over.append((key, round(bv - acc(c, "EXACT"), 3)))
    P["P7"] = (not over, "cells with a VSA config > EXACT + 0.02: %s" % (over or "none"))
    # P8
    wins = [(key, m) for key, c in S.items() for m in V.METRICS if c["verdict"][m] != "TABLE"]
    P["P8"] = (not wins, "VSA-WIN (cell, metric): %s" % (wins or "none"))
    # P9
    cr = [c["c_ratio_ops"] for c in S.values() if c["c_ratio_ops"] is not None]
    P["P9"] = (max(cr) <= 1e-3, "C-ratio range %.1e-%.1e over %d cells" % (min(cr), max(cr), len(cr)))
    # P10
    c = S[k("high", 6, 100000)]
    d10 = acc(c, "EXACT") - acc(c, c["v_acc"])
    P["P10"] = (d10 >= 0.20, "EXACT - most accurate VSA at high h6 N=100k: %.3f (EXACT %.3f)" % (d10, acc(c, "EXACT")))
    # P11
    dstar = []
    for N in NS:
        r = S[k("high", 6, N)]["recovery_by_D"]
        ok = [int(D) for D, x in r.items() if x["recovery"] is not None and x["recovery"] >= 0.9]
        dstar.append(min(ok) if ok else None)
    defined = [d for d in dstar if d is not None]
    P["P11"] = (len(defined) >= 2 and all(x <= y for x, y in zip(defined, defined[1:])),
                "smallest D with >= 90%% recovery at high h6 per N: %s (undefined = never on the grid)" % dstar)
    L.append("# B41 prediction checks (mechanical, from %s)" % a.scores)
    for p_, (ok, txt) in P.items():
        L.append("  %-4s %-5s %s" % (p_, "HELD" if ok else "FAIL", txt))

    # accuracy at equal state: best VSA at D vs best table with state <= D
    L.append("")
    L.append("# Accuracy at equal state width (words carried between hops): best VSA config at D vs best table config "
             "whose state <= D; high noise")
    for h in (4, 6):
        for N in NS:
            c = S[k("high", h, N)]
            row = []
            for D in EN.DS:
                vs = [n for n in vsa_names if "_D%d_" % D in n and n in c["acc"]]
                bv = max(acc(c, n) for n in vs)
                tb = [t for t in V.TABLES if t in c["cost"] and c["cost"][t]["state"] <= D]
                bt = max(acc(c, t) for t in tb) if tb else float("nan")
                row.append("D=%d VSA %.3f / table %.3f" % (D, bv, bt))
            L.append("  h=%d N=%-6d %s" % (h, N, "; ".join(row)))

    # dead-mass split and exploratory runs need the per-query files
    L.append("")
    L.append("# Start mention read correctly (argmax) vs not: accuracy of ARGMAX / BEAM(4,16) / EXACT / v*, h = 6")
    for lv, rho in zip(LV, a.rhos):
        for N in NS:
            cell, sid, metas = V.load_cell(a.runs, a.split, rho, N, a.seeds)
            if cell is None:
                continue
            ok0 = np.concatenate([np.array(m["diag"]["start_argmax_correct"]) for m in metas])
            vs = S[k(lv, 6, N)]["v_star"]
            parts = []
            for nm in ("ARGMAX", "BEAM_k4_B16", "EXACT", vs):
                cc = cell[nm]["correct"][:, 2]
                parts.append("%s %.3f/%.3f" % (nm.split("_")[0] if nm != vs else "v*", cc[ok0].mean(), cc[~ok0].mean()))
            L.append("  %-4s N=%-6d start right %.3f of queries | %s" % (lv, N, ok0.mean(), "; ".join(parts)))

    L.append("")
    L.append("# Failure types at h = 6 (share of failures): answer in the true answer's confusable cluster (near miss) / "
             "another entity / dead chain (no answer)")
    for lv, rho in zip(LV, a.rhos):
        for N in NS:
            cell, sid, metas = V.load_cell(a.runs, a.split, rho, N, a.seeds)
            if cell is None:
                continue
            gold = np.concatenate([np.load(os.path.join(a.runs, "%s_s%d_N%d_r%.3f.npz" % (a.split, s_, N, rho)))
                                   ["_answers"][:, 2] for s_ in a.seeds])
            vs = S[k(lv, 6, N)]["v_star"]
            parts = []
            for nm in ("ARGMAX", "BEAM_k4_B16", "EXACT", vs):
                ans = np.concatenate([np.load(os.path.join(a.runs, "%s_s%d_N%d_r%.3f.npz" % (a.split, s_, N, rho)))
                                      ["%s|answer" % nm][:, 2] for s_ in a.seeds])
                bad = ans != gold
                nb = max(1, int(bad.sum()))
                near = int((bad & (ans >= 0) & (ans // 4 == gold // 4)).sum())
                dead = int((bad & (ans < 0)).sum())
                parts.append("%s %d fails: %.2f/%.2f/%.2f" % ("v*" if nm == vs else nm.split("_")[0], int(bad.sum()),
                                                              near / nb, (nb - near - dead) / nb, dead / nb))
            L.append("  %-4s N=%-6d %s" % (lv, N, "; ".join(parts)))

    if a.runs_x:
        L.append("")
        L.append("# Exploratory E1 (registered in the dev record; not in the bar): VSA at D = 262,144 (2^18)")
        for lv, rho in zip(LV, a.rhos):
            for h_i, h in enumerate(HS):
                for N in (320, 3200, 32000, 100000):
                    files = [os.path.join(a.runs_x, "%s_s%d_N%d_r%.3f.npz" % (a.split, s, N, rho)) for s in a.seeds]
                    if not all(os.path.exists(f) for f in files):
                        L.append("  missing %s h%d N%d" % (lv, h, N)); continue
                    zs = [np.load(f) for f in files]
                    names = sorted(set(n.split("|")[0] for n in zs[0].files if n.startswith("VSA_")))
                    best = max(names, key=lambda n: np.concatenate([z["%s|correct" % n][:, h_i] for z in zs]).mean())
                    ab = np.concatenate([z["%s|correct" % best][:, h_i] for z in zs]).mean()
                    ops = np.concatenate([z["%s|ops" % best][:, h_i] for z in zs]).mean()
                    c = S[k(lv, h, N)]
                    L.append("  %-4s h=%d N=%-6d best 2^18 %-22s %.3f (ops %.1e) | D=65536 v* %.3f | ARGMAX %.3f "
                             "BEAM(4,16) %.3f EXACT %.3f" % (lv, h, N, best, ab, ops, acc(c, c["v_star"]),
                                                             acc(c, "ARGMAX"), acc(c, "BEAM_k4_B16"), acc(c, "EXACT")))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
