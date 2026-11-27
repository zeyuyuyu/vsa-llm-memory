"""B55 verdict (prereg/B55_argmax_read_prereg.md). Reads B55's prediction files and the comparison files, applies the
registered bars, prints the checks, and writes a hand-read sample.
usage: python b55_verdict.py <b55_preds> <b40_preds> <b50_preds> a800|pace [handread_out]
  a800: comparison arms = B40's files (THR = vsa_s*, DISC = disc_s*, R VSA = vsa_s*) and B50's discmsp_s* at rho 0
  pace: comparison arms = B55's same-GPU re-runs (vsathr, disc, vsa, discmsp at rho 0); B40/B50 files = reproduction
"""
import json, os, sys
import numpy as np
import b40_data as D
from b40_verdict import Arm, load, wilson, r2, fmt

B = 10000
RHOS = ("0.0", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0")
SEEDS = (0, 1, 2)


def boot(a, b, ids, rng):  # paired bootstrap over items of the seed-mean correctness (B40's convention)
    d = a.ok(ids).mean(0) - b.ok(ids).mean(0)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def label(t, sa, sb):
    d, lo, hi = t
    allw, alll = all(x > y for x, y in zip(sa, sb)), all(x < y for x, y in zip(sa, sb))
    lab = "WIN" if lo > 0 else "LOSE" if hi < 0 else ("TIE" if lo >= -0.02 and hi <= 0.02 else "NOT RESOLVED")
    return lab, allw, alll


def main():
    p55, p40, p50, where = sys.argv[1:5]
    hr_out = sys.argv[5] if len(sys.argv) > 5 else ""
    rng = np.random.default_rng(5555)
    out, js, hr = [], {}, []
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    f55 = lambda t, n, s, r: f"{p55}/{t}/{n}_s{s}_rho{r}.jsonl"
    f40 = lambda t, n, s, r: f"{p40}/{t}/{n}_s{s}.jsonl" if r == "0.0" else f"{p40}/{t}/{n}_s{s}_rho{r}.jsonl"

    def arm(files):
        fs = [f for f in files if os.path.exists(f)]
        return Arm(fs) if len(fs) == len(files) else None

    def comp(t, name40, name55, r):  # comparison arm under the registered plan
        if where == "a800":
            return arm([f40(t, name40, s, r) for s in SEEDS])
        return arm([f55(t, name55, s, r) for s in SEEDS])

    # ================================================================== task N
    rows = {r["id"]: r for r in D.gen("N", "test")}
    TR = [i for i, r in rows.items() if r["op"] in D.OPS_TRAIN]
    HO = [i for i, r in rows.items() if r["op"] in D.OPS_HELD]
    base = Arm([f"{p40}/N/base.jsonl"])
    P("===================== B55 TASK N (2,000 trained-type items TR; 500 held-out HO) =====================")
    P(f"BASE (B40): TR {base.acc(TR):.4f} HO {base.acc(HO):.4f}   comparison plan: {where}")
    js["N"] = {}
    for r in RHOS:
        A = arm([f55("N", "vsaarg", s, r) for s in SEEDS])
        T = comp("N", "vsa", "vsathr", r)
        Dd = comp("N", "disc", "disc", r)
        if where == "a800" and r == "0.0":
            M = arm([f"{p50}/N/discmsp_s{s}.jsonl" for s in SEEDS])
        else:
            M = arm([f55("N", "discmsp", s, r) for s in SEEDS])
        if A is None:
            P(f"rho {r}: ARG files missing"); continue
        P(f"--- rho {r}")
        row = {}
        for nm, X in (("ARG", A), ("THR", T), ("DISC", Dd), ("DISC-MSP", M)):
            if X is None:
                P(f"   {nm:8s} missing"); continue
            tr, ho = X.acc(TR), X.acc(HO)
            lo, hi = wilson(round(tr * len(TR)), len(TR))
            g_tr = np.mean([X.runs[k][i]["mod"]["gate"] for k in range(3) for i in TR])
            g_ho = np.mean([X.runs[k][i]["mod"]["gate"] for k in range(3) for i in HO])
            rex = np.mean([(lambda m, g: m["op"] == g["op"] and m["x"] == g["x"] and m["y"] == g["y"])(X.runs[k][i]["mod"], rows[i])
                           for k in range(3) for i in TR])
            P(f"   {nm:8s} TR {tr:.4f} [{lo:.3f}, {hi:.3f}] seeds {[round(x, 4) for x in X.seed_accs(TR)]}  HO {ho:.4f} "
              f"(add {X.acc([i for i in HO if rows[i]['op'] == 'addition']):.3f}, div {X.acc([i for i in HO if rows[i]['op'] == 'division']):.3f})"
              f"  read-exact TR {rex:.4f}  gate-open TR {g_tr:.4f} HO {g_ho:.4f}")
            row[nm] = {"TR": tr, "HO": ho, "seeds_TR": X.seed_accs(TR), "seeds_HO": X.seed_accs(HO), "read_exact_TR": float(rex),
                       "open_TR": float(g_tr), "open_HO": float(g_ho)}
        if Dd is not None:
            t = boot(A, Dd, TR, rng)
            lab, aw, al = label(t, A.seed_accs(TR), Dd.seed_accs(TR))
            l2 = r2(A, Dd, TR, np.random.default_rng(1))[0]
            P(f"   B55-N1 TR(ARG) - TR(DISC) = {fmt(t)} -> {lab} (ARG > DISC all seeds {aw}; < all seeds {al}); B40 R2 bar: {l2}")
            row["N1"] = {"delta": t, "label": lab, "all_seeds_win": aw, "all_seeds_lose": al, "R2": l2}
        if M is not None:
            t = boot(A, M, HO, rng)
            lab, aw, al = label(t, A.seed_accs(HO), M.seed_accs(HO))
            P(f"   B55-N2 HO(ARG) - HO(DISC-MSP) = {fmt(t)} -> {lab} (all seeds {aw} / {al}); "
              f"TR(ARG) - TR(DISC-MSP) = {fmt(boot(A, M, TR, rng))}")
            row["N2"] = {"delta": t, "label": lab, "all_seeds_win": aw, "all_seeds_lose": al}
        if Dd is not None:
            P(f"   HO(ARG) - HO(DISC-0.8) = {fmt(boot(A, Dd, HO, rng))}")
        if T is not None:
            P(f"   ARG - THR: TR {fmt(boot(A, T, TR, rng))}  HO {fmt(boot(A, T, HO, rng))}")
            row["ARG_minus_THR_TR"] = boot(A, T, TR, rng)
        P(f"   HO(ARG) - BASE = {fmt(boot(A, base, HO, rng))}")
        # checks: same noisy state read by ARG and THR (type similarity identical), and by DISC-MSP and DISC (op_p, x, y)
        if T is not None:
            mm = sum(A.runs[k][i]["mod"]["op_s"] != T.runs[k][i]["mod"]["op_s"] or A.runs[k][i]["mod"]["op"] != T.runs[k][i]["mod"]["op"]
                     for k in range(3) for i in rows)
            P(f"   check same-noise ARG vs THR: type read / op_s differ on {mm}/{3 * len(rows)}")
            row["chk_same_noise_thr"] = mm
        if M is not None and Dd is not None:
            mm = sum(any(M.runs[k][i]["mod"][f] != Dd.runs[k][i]["mod"][f] for f in ("op", "op_p", "x", "y"))
                     for k in range(3) for i in rows)
            both = [(k, i) for k in range(3) for i in rows if M.runs[k][i]["mod"]["gate"] and Dd.runs[k][i]["mod"]["gate"]]
            pm = sum(M.runs[k][i]["pred"] != Dd.runs[k][i]["pred"] for k, i in both)
            P(f"   check same-noise DISC-MSP vs DISC: reads differ on {mm}/{3 * len(rows)}; where both gates open, pred differs on {pm}/{len(both)}")
            row["chk_same_noise_msp"] = [mm, pm, len(both)]
        if M is not None and not (where == "a800" and r == "0.0"):
            tau = [json.load(open(f"{p50}/../out/thresholds.json"))["seeds"][str(s)]["tau"]["msp"] for s in SEEDS] \
                if os.path.exists(f"{p50}/../out/thresholds.json") else None
            if tau:
                bad = sum(M.runs[k][i]["mod"]["gate"] != (M.runs[k][i]["mod"]["msp"] >= tau[k]) for k in range(3) for i in rows)
                P(f"   check MSP gate re-derivation: {bad} mismatches")
        js["N"][r] = row
        if r in ("0.0", "0.25") and Dd is not None and hr_out:
            sub = rng.permutation([i for i in TR if A.runs[0][i]["correct"] != Dd.runs[0][i]["correct"]])[:12]
            hr.append(f"=== N rho {r}, seed 0: trained-type items where ARG and DISC disagree (12 of "
                      f"{sum(A.runs[0][i]['correct'] != Dd.runs[0][i]['correct'] for i in TR)})")
            for i in sub:
                g = rows[i]
                hr.append(f"{i} {g['op']} x={g['x']} y={g['y']} gold={A.runs[0][i]['gold']} | ARG {A.runs[0][i]['mod']} out={A.runs[0][i]['out']!r} "
                          f"ok={A.runs[0][i]['correct']} | DISC {Dd.runs[0][i]['mod']} out={Dd.runs[0][i]['out']!r} ok={Dd.runs[0][i]['correct']}"
                          + (f" | THR {T.runs[0][i]['mod']} out={T.runs[0][i]['out']!r}" if T is not None else ""))
    # rho-0 reproduction of B40 exploratory 1 (digit_thr 0) by the registered argmax arm (digit_thr -inf)
    A0, E1 = arm([f55("N", "vsaarg", s, "0.0") for s in SEEDS]), arm([f"{p40}/N/vsaargmax_s{s}.jsonl" for s in SEEDS])
    if A0 is not None and E1 is not None:
        dp = sum(A0.runs[k][i]["pred"] != E1.runs[k][i]["pred"] for k in range(3) for i in rows)
        dm = sum(A0.runs[k][i]["mod"] != E1.runs[k][i]["mod"] for k in range(3) for i in rows)
        P(f"check: ARG rho 0 vs B40 exploratory-1 files: pred differs on {dp}/7500, module record differs on {dm}/7500")
        js["N"]["chk_expl1"] = [dp, dm]
    if where == "pace":  # cross-GPU reproduction of the clean comparison arms
        for n55, n40 in (("vsathr", "vsa"), ("disc", "disc")):
            X, Y = arm([f55("N", n55, s, "0.0") for s in SEEDS]), arm([f"{p40}/N/{n40}_s{s}.jsonl" for s in SEEDS])
            if X is not None and Y is not None:
                P(f"check (pace vs B40 A800): N {n55} rho 0 pred differs on {sum(X.runs[k][i]['pred'] != Y.runs[k][i]['pred'] for k in range(3) for i in rows)}/7500")

    # ================================================================== task R
    rows = {r["id"]: r for r in D.gen("R", "test")}
    ids = list(rows)
    cells = [(f"{s}-d{d}", [i for i, r in rows.items() if r["size"] == s and r["depth"] == d]) for s in ("small", "large") for d in range(1, 6)]
    P("===================== B55 TASK R (all 2,000 items) =====================")
    js["R"] = {}
    for r in RHOS:
        A = arm([f55("R", "vsaarg", s, r) for s in SEEDS])
        V = comp("R", "vsa", "vsa", r)
        Dd = comp("R", "disc", "disc", r)
        if A is None:
            P(f"rho {r}: ARG files missing"); continue
        P(f"--- rho {r}")
        row = {}
        for nm, X in (("ARG", A), ("VSA-B40", V), ("DISC", Dd)):
            if X is None:
                P(f"   {nm:8s} missing"); continue
            acc = X.acc(ids)
            lo, hi = wilson(round(acc * len(ids)), len(ids))
            rok = np.mean([X.runs[k][i]["mod"].get("reads_ok", float("nan")) for k in range(3) for i in ids])
            P(f"   {nm:8s} acc {acc:.4f} [{lo:.3f}, {hi:.3f}] seeds {[round(x, 4) for x in X.seed_accs(ids)]}  reads-ok {rok:.4f}  "
              f"cells " + " ".join(f"{c}:{X.acc(ci):.3f}" for c, ci in cells))
            row[nm] = {"acc": acc, "seeds": X.seed_accs(ids), "reads_ok": float(rok), "cells": {c: X.acc(ci) for c, ci in cells}}
        if Dd is not None:
            t = boot(A, Dd, ids, rng)
            lab, aw, al = label(t, A.seed_accs(ids), Dd.seed_accs(ids))
            l2 = r2(A, Dd, ids, np.random.default_rng(1))[0]
            P(f"   B55-R1 ARG - DISC = {fmt(t)} -> {lab} (all seeds {aw} / {al}); B40 R2 bar: {l2}")
            row["R1"] = {"delta": t, "label": lab, "all_seeds_win": aw, "all_seeds_lose": al, "R2": l2}
        if V is not None:
            P(f"   ARG - VSA(B40, mentions uncleaned) = {fmt(boot(A, V, ids, rng))}")
            mm = sum(A.runs[k][i]["mod"]["reads_ok"] != V.runs[k][i]["mod"]["reads_ok"] for k in range(3) for i in ids)
            P(f"   check same-noise ARG vs VSA-B40: per-item reads_ok differs on {mm}/{3 * len(ids)}")
            row["chk_same_noise"] = mm
        js["R"][r] = row
        if r == "2.0" and Dd is not None and hr_out:
            sub = rng.permutation([i for i in ids if A.runs[0][i]["correct"] != Dd.runs[0][i]["correct"]])[:12]
            hr.append(f"=== R rho {r}, seed 0: items where ARG and DISC disagree (12 of {sum(A.runs[0][i]['correct'] != Dd.runs[0][i]['correct'] for i in ids)})")
            for i in sub:
                hr.append(f"{i} depth {rows[i]['depth']} {rows[i]['size']} gold={A.runs[0][i]['gold']} | ARG {A.runs[0][i]['mod']} out={A.runs[0][i]['out']!r} "
                          f"| DISC {Dd.runs[0][i]['mod']} out={Dd.runs[0][i]['out']!r}")
    print("\n".join(out))
    json.dump(js, open(os.environ.get("B55_JSON", "results/b55_scores.json"), "w"), indent=1)
    if hr_out:
        open(hr_out, "w").write("\n".join(hr) + "\n")


if __name__ == "__main__":
    main()
