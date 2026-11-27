"""V-151 FROZEN SCORER. Committed with the prereg, BEFORE the V-151 sweep. Run verbatim.

    PYTHONHASHSEED=0 python3 v151_score.py v151_logs/v151.json

Bars, transcribed once from v151_prereg.md and never edited:
  S0 HRR's interpolated k* lies in [v, 2v) for V-149's banked octave v = 32/64/128/256
  S1 for every model, the 95% bootstrap CI for alpha in k* ~ d^alpha contains 1.0
  S2 max(alpha) - min(alpha) across the four models < 0.25
  S3 k*(BSC)/k*(HRR) in [0.35, 0.65] at every interior d, and k*(FHRR)/k*(HRR) >= 1.4 at d=512
  S4 descriptive

Censored cells are excluded from every fit by this file, not by hand, and the exclusions are printed.
"""
import json, sys
import numpy as np

BOOT, BOOT_SEED, THRESH = 20000, 20261008, 0.70
V149_OCTAVE = {512: 32, 1024: 64, 2048: 128, 4096: 256}   # HRR, banked at fc874c8


def kstar(ks, accs):
    """Interpolate the 0.70 crossing in log k. Returns (k*, status)."""
    ks = list(ks)
    if accs[0] < THRESH:
        return None, "left-censored"
    if accs[-1] >= THRESH:
        return None, "right-censored"
    for i in range(len(ks) - 1):
        a0, a1 = accs[i], accs[i + 1]
        if a0 >= THRESH > a1:
            if a0 == a1:
                return float(ks[i]), "interior"
            f = (a0 - THRESH) / (a0 - a1)
            lk = np.log(ks[i]) + f * (np.log(ks[i + 1]) - np.log(ks[i]))
            return float(np.exp(lk)), "interior"
    return None, "no crossing found"


def main():
    D = json.load(open(sys.argv[1]))
    models = D["config"]["models"]
    dims = [int(x) for x in D["config"]["dims"]]
    ks = [int(x) for x in D["config"]["ks"]]
    C = D["capacity"]

    print("=" * 100)
    print("V-151  the capacity exponent at a resolution that can resolve it   n=%d, seeds %s"
          % (D["config"]["n"], D["config"]["cap_seeds"]))
    print("=" * 100)

    KS, status = {}, {}
    for m in models:
        KS[m], status[m] = {}, {}
        for d in dims:
            accs = [float(C[m][str(d)][str(k)]) for k in ks]
            v, st = kstar(ks, accs)
            KS[m][d] = v; status[m][d] = st

    print("")
    print("  interpolated k* (V-149's octave value in brackets)")
    print("  %-6s %s" % ("model", "  ".join("d=%-14d" % d for d in dims)))
    for m in models:
        cells = []
        for d in dims:
            if KS[m][d] is None:
                cells.append("%-16s" % status[m][d])
            else:
                oct_ = V149_OCTAVE[d] if m == "HRR" else None
                cells.append("%-16s" % ("%.1f%s" % (KS[m][d], " [%d]" % oct_ if oct_ else "")))
        print("  %-6s %s" % (m, "  ".join(cells)))
    cens = [(m, d, status[m][d]) for m in models for d in dims if KS[m][d] is None]
    print("  censored cells excluded from every fit: %s" % (cens if cens else "none"))

    # ---------------- S0 ----------------
    print("")
    print("  ---- S0 gate ----")
    ok0, det = True, []
    for d in dims:
        v = V149_OCTAVE[d]; k = KS["HRR"][d]
        good = k is not None and v <= k < 2 * v
        ok0 &= good
        det.append("d=%d: k*=%s in [%d,%d) %s" % (d, "%.1f" % k if k else "censored", v, 2 * v,
                                                  "ok" if good else "OUT"))
    print("  S0 HRR reproduces V-149's octave ladder at finer resolution: %s" % ok0)
    for x in det:
        print("     " + x)
    if not ok0:
        print("\n  >>> S0 FAILS. The protocol drifted; nothing is read.")
        print("V151-SCORED"); return

    # ---------------- S1 / S2 ----------------
    print("")
    print("  ---- S1 / S2 ----")
    rng = np.random.default_rng(BOOT_SEED)
    alphas, cis = {}, {}
    for m in models:
        pts = [(d, KS[m][d]) for d in dims if KS[m][d] is not None]
        if len(pts) < 3:
            alphas[m], cis[m] = None, None
            print("  %-5s too few interior cells (%d) to fit" % (m, len(pts)))
            continue
        x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
        a = float(np.polyfit(x, y, 1)[0])
        draws = []
        for _ in range(BOOT):
            idx = rng.integers(0, len(x), len(x))
            if len(set(x[idx])) < 2:
                continue
            draws.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        alphas[m], cis[m] = a, (lo, hi)
        print("  %-5s alpha %.3f  CI [%.3f, %.3f]  contains 1.0: %s"
              % (m, a, lo, hi, lo <= 1.0 <= hi))
    s1 = all(cis[m] is not None and cis[m][0] <= 1.0 <= cis[m][1] for m in models)
    print("  S1 every model's CI contains 1.0 => %s" % ("HOLDS" if s1 else "FAILS"))

    av = [alphas[m] for m in models if alphas[m] is not None]
    spread = (max(av) - min(av)) if len(av) == len(models) else float("nan")
    s2 = spread < 0.25
    print("  S2 alpha spread %.3f < 0.25 => %s" % (spread, "HOLDS" if s2 else "FAILS"))

    # ---------------- S3 ----------------
    print("")
    print("  ---- S3, the constants V-149 read off an octave grid ----")
    ratios, s3a = [], True
    for d in dims:
        if KS["BSC"][d] is None or KS["HRR"][d] is None:
            continue
        r = KS["BSC"][d] / KS["HRR"][d]; ratios.append((d, r))
        good = 0.35 <= r <= 0.65; s3a &= good
        print("     d=%-5d BSC/HRR %.3f in [0.35,0.65] %s" % (d, r, "ok" if good else "OUT"))
    rf = (KS["FHRR"][512] / KS["HRR"][512]) if (KS["FHRR"][512] and KS["HRR"][512]) else None
    s3b = rf is not None and rf >= 1.4
    print("     d=512  FHRR/HRR %s >= 1.4 %s" % ("%.3f" % rf if rf else "n/a", "ok" if s3b else "OUT"))
    s3 = s3a and s3b and bool(ratios)
    print("  S3 => %s" % ("HOLDS" if s3 else "FAILS"))

    # ---------------- S4 ----------------
    print("")
    print("  ---- S4 descriptive ----")
    for m in models:
        pts = [(d, KS[m][d]) for d in dims if KS[m][d] is not None]
        if len(pts) >= 2:
            x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
            a, c = np.polyfit(x, y, 1)
            print("    %-5s k* = exp(%.3f) * d^%.3f   -> k*/d at d=1024: %.4f"
                  % (m, c, a, KS[m][1024] / 1024 if KS[m][1024] else float("nan")))

    say = lambda v: "HOLDS" if v else "FAILS"
    print("")
    print("=" * 100)
    print("V-151 VERDICT: S1 %s | S2 %s | S3 %s" % (say(s1), say(s2), say(s3)))
    print("")
    if s1 and s2 and s3:
        print("  V-149's exponent failure was its grid, and all three of its readings survive at a")
        print("  resolution that can test them: one law, linear in d, with model-specific constants near")
        print("  the halves and doubles it reported.")
    elif s1 and s2 and not s3:
        print("  The law survives and the specific constants do not. V-149 read 'half' and 'twice' off an")
        print("  octave grid, where one step IS a factor of two by construction; at finer resolution the")
        print("  ratios are different. The manuscript must stop saying half and twice and quote these.")
    elif not s1:
        print("  The sub-linearity is REAL for at least one model and V-149's self-correction was itself")
        print("  wrong: a finer grid does not rescue the exponent. This is the outcome worth chasing.")
    print("  SCOPE: 2WikiMultihopQA validation slice, V-101 pooling, two-hop unbind-and-cleanup. Not")
    print("  analogical mapping -- V-146 and V-148/V-150 are different mechanisms and must not be pooled.")
    print("=" * 100)
    print("V151-SCORED")


if __name__ == "__main__":
    main()
