"""V-149 frozen scorer. Committed with the prereg BEFORE the sweep. Run verbatim on v149.json."""
import json, sys
import numpy as np

MODELS = ["HRR", "MAP", "BSC", "FHRR"]
V101_K128, V101_K256, V101_TOL = 0.7417, 0.2417, 0.05
REBASE_EM, REBASE_TOL = 0.9778, 0.01
P1_LEVEL, P1_MIN_MODELS = 0.90, 3
P2_MIN_SPREAD = 0.05
P3_LO, P3_HI = 0.85, 1.15
KSTAR_LEVEL = 0.70
BOOT, RNG = 10000, 20149


def kstar(byk, level=KSTAR_LEVEL):
    ok = [int(k) for k, v in byk.items() if v >= level]
    return max(ok) if ok else 0


def alpha(ks_by_d):
    ds = sorted(int(d) for d in ks_by_d)
    xs = [np.log(d) for d in ds if ks_by_d[str(d)] > 0]
    ys = [np.log(ks_by_d[str(d)]) for d in ds if ks_by_d[str(d)] > 0]
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    D = json.load(open(sys.argv[1]))
    cap, task = D["capacity"], D["task"]
    W = "=" * 96
    print(W)
    print("V-149  is the store result HRR's, or the VSA family's?   4 models, algebra swapped, all else fixed")
    print(W)

    print("\n  ---- M1 capacity grid (accuracy at pool size k; V-101 protocol) ----")
    ks = sorted(int(k) for k in next(iter(cap["HRR"].values())))
    print("  %-6s %-6s " % ("model", "d") + "".join("%9s" % ("k=%d" % k) for k in ks))
    kstars = {}
    for m in MODELS:
        kstars[m] = {}
        for d in sorted(cap[m], key=int):
            byk = cap[m][d]
            kstars[m][d] = kstar(byk)
            print("  %-6s %-6s " % (m, d) + "".join("%9.3f" % byk[str(k)] for k in ks))
    print("\n  k* (largest k at accuracy >= %.2f):" % KSTAR_LEVEL)
    for m in MODELS:
        print("    %-6s " % m + "  ".join("d%s=%d" % (d, kstars[m][d]) for d in sorted(kstars[m], key=int))
              + "   alpha = %.3f" % alpha(kstars[m]))

    print("\n  ---- M2 task readout (300 banked rows, d=2048, shard 64) ----")
    means = {m: task[m]["mean_em"] for m in MODELS}
    for m in MODELS:
        print("    %-6s EM %.4f   seeds %s   %-42s %s"
              % (m, means[m], " ".join("%.4f" % v for v in task[m]["per_seed"].values()),
                 task[m]["arith"], "INTEGER-ONLY" if task[m]["integer_only"] else ""))

    print("\n  ---- P0 gate ----")
    h2048 = cap["HRR"]["2048"]
    p0a = abs(h2048["128"] - V101_K128) <= V101_TOL and abs(h2048["256"] - V101_K256) <= V101_TOL
    p0b = abs(means["HRR"] - REBASE_EM) <= REBASE_TOL
    print("    P0a HRR reproduces V-101 (k128 %.4f vs %.4f, k256 %.4f vs %.4f, tol %.2f): %s"
          % (h2048["128"], V101_K128, h2048["256"], V101_K256, V101_TOL, p0a))
    print("    P0b HRR reproduces store_rebase (%.4f vs %.4f, tol %.3f): %s"
          % (means["HRR"], REBASE_EM, REBASE_TOL, p0b))
    if not (p0a and p0b):
        print("\n" + W)
        print("V-149 GATE FAILS: the transcribed HRR path is not the banked one. NO READING IS TAKEN.")
        print(W)
        print("V149-SCORED")
        return

    print("\n  ---- P1 / P2 / P3 ----")
    clear = [m for m in MODELS if means[m] >= P1_LEVEL]
    p1 = len(clear) >= P1_MIN_MODELS
    print("    P1 models at EM >= %.2f: %d of 4 (%s), bar >= %d => %s"
          % (P1_LEVEL, len(clear), ",".join(clear) or "none", P1_MIN_MODELS, "HOLDS" if p1 else "FAILS"))

    hi = max(MODELS, key=lambda m: means[m])
    lo = min(MODELS, key=lambda m: means[m])
    spread = means[hi] - means[lo]
    rng = np.random.default_rng(RNG)
    A = np.mean([task[hi]["per_row_hits"][s] for s in task[hi]["per_row_hits"]], axis=0)
    B = np.mean([task[lo]["per_row_hits"][s] for s in task[lo]["per_row_hits"]], axis=0)
    n = len(A)
    idx = rng.integers(0, n, size=(BOOT, n))
    boot = A[idx].mean(1) - B[idx].mean(1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    excl = ci[0] > 0 or ci[1] < 0
    p2 = spread >= P2_MIN_SPREAD and excl
    print("    P2 spread %s - %s = %+.4f  CI [%+.4f, %+.4f]  bar >= %.2f and CI excludes 0 => %s"
          % (hi, lo, spread, ci[0], ci[1], P2_MIN_SPREAD, "HOLDS" if p2 else "FAILS"))

    alphas = {m: alpha(kstars[m]) for m in MODELS}
    bad = [m for m in MODELS if not (P3_LO <= alphas[m] <= P3_HI)]
    p3 = not bad
    print("    P3 alpha in [%.2f,%.2f] for every model: %s => %s"
          % (P3_LO, P3_HI, ", ".join("%s=%.3f" % (m, alphas[m]) for m in MODELS),
             "HOLDS" if p3 else "FAILS (" + ",".join(bad) + ")"))

    print("\n  ---- P4 descriptive: the accuracy / arithmetic frontier ----")
    for m in sorted(MODELS, key=lambda m: -means[m]):
        t = D.get("timing", {}).get(m, {}).get("sec_per_query", float("nan"))
        print("    %-6s EM %.4f  %.4f s/query  %-42s %s"
              % (m, means[m], t, task[m]["arith"], "INTEGER-ONLY" if task[m]["integer_only"] else ""))

    print("\n" + W)
    print("V-149 VERDICT: P1 %s | P2 %s | P3 %s"
          % ("HOLDS" if p1 else "FAILS", "HOLDS" if p2 else "FAILS", "HOLDS" if p3 else "FAILS"))
    print("\n  SCOPE: one task, one identity scheme, one cleanup rule; four of the eleven models Survey")
    print("  Part I catalogues. The sparse families and the matrix binders are not tested.")
    print(W)
    print("V149-SCORED")


if __name__ == "__main__":
    main()
