"""B25 verdict (prereg/B25_pcm_device_model_prereg.md): tables, bars, prediction checks, the energy estimate.
Written before any B25 condition was run on the real stores.

python3 b25_verdict.py results/b25_pcm.json results/b25_verdict.txt
"""
import json
import sys

import numpy as np

TIMES = ["60 s", "1 h", "1 d", "30 d", "1 y"]
LOADS = [16, 32, 64]
ARMS = ["B1-S", "B1-C", "B1-SC", "ML-S", "ML-C", "ML-SC"]

# ---- energy constants (prereg; every number from the cited source) ----
N_MAX, D = 5993, 8192
E_KAR = 9.44e-9 / (22 * 10000)          # Karunaratne 2020 Table 1 (AM search, PCM) / Ext. Data Table 1 (c=22, d=10,000)
E_HER1 = 2 / 9.76e12                    # Le Gallo 2023: 9.76 TOPS/W (1-phase), 1 MAC = 2 OPs
E_HER4 = 2 / 2.48e12                    # Le Gallo 2023: 2.48 TOPS/W (4-phase)
E_SA = 9.8e-15                          # Karunaratne Ext. Data Table 2: energy per sense-amplifier read
TDP, INT8, FP16, FP32, BW = 400.0, 624e12, 312e12, 19.5e12, 2039e9   # NVIDIA A100 SXM datasheet

CITES = [
    "[J20] Joshi, V. et al. Accurate deep neural network inference using computational phase-change memory. Nat. Commun.",
    "  11, 2473 (2020). Main text: \"For all levels, we achieve a standard deviation less than 1.2 uS\" (programming noise,",
    "  Fig. 3b; median 0.94 uS over 11 levels, measured ~25 s after programming); iterative programming \"within the range",
    "  of 0-25 uS\"; \"In our PCM devices, nu ~ 0.06 on average\"; without compensation the accuracy \"quickly decreases",
    "  down to 10% (random guessing) within approximately 1000 s\"; Methods: \"on-chip 8-bit cyclic analog-to-digital",
    "  converter\", \"Global drift compensation (GDC) method\" (summed column currents under a calibration voltage divided",
    "  by the stored sum of the known programmed conductances). Supplementary Note 2: Eqs. 8-9 (mean and sd of nu vs target",
    "  conductance at 20 s: 0.055 / 0.0043 at 25 uS, 0.08 / 0.033 at 0 uS), Eqs. 10-11 (Q_s = 0.0388/G_T^0.5729 (t/1.089e5",
    "  s)^-nu_q), Eqs. 12-17 (programming noise at T0 = 27.36 s and T1 = 55.39 s; drift from T1; 1/f read noise with",
    "  gamma = 1.21, T_read = 250 ns, f_min = 1/(t + T_read)); Supplementary Fig. 16 (programming-noise sd ~1.07 uS at T0",
    "  and ~1.12 uS at T1 at 25 uS, ~0.15-0.2 uS at 0 uS; we use 1.2 and 0.2 uS per event). Data to 1e5 s on 10,000 devices.",
    "[LS20] Le Gallo, M. & Sebastian, A. An overview of phase-change memory device physics. J. Phys. D: Appl. Phys. 53,",
    "  213002 (2020), Sec. 4.2: \"The drift exponent nu_R, which typically has a value of 0.1 for the RESET state, exhibits",
    "  significant inter-device and intra-device variability ... the drift exponent is lower for the SET state (< 0.05)\";",
    "  \"Drift variability across different resistance states and devices is arguably the most significant challenge for",
    "  multi-level storage in PCM\".",
    "[K20] Karunaratne, G. et al. In-memory hyperdimensional computing. Nat. Electron. 3, 327-337 (2020). Binary",
    "  hypervectors (d = 10,000; \"Binary '1' components are programmed as crystalline states and binary '0' components",
    "  are programmed as amorphous states\") in PCM crossbars, up to 760,000 devices; Table 1: average energy per query of the",
    "  AM search 1,110 nJ (all-CMOS) vs 9.44 nJ (PCM crossbar), 117.5x; total 2,584 vs 430.3 nJ, 6.01x; exclusive AM",
    "  modules 1,104 vs 3.30 nJ; Extended Data Table 1: language task AM c = 22 classes, d = 10,000; Extended Data Table 2:",
    "  read voltage 0.1 V, 1 uA per conducting device, AM readout 100 ns, 12 pJ per ADC read, 9.8 fJ per sense-amp read.",
    "[LG23] Le Gallo, M. et al. A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural",
    "  network inference. Nat. Electron. 6, 680-693 (2023) (arXiv:2212.02872): 64 cores of 256 x 256 PCM unit cells, \"a",
    "  maximal throughput of 63.1 TOPS at an energy efficiency of 9.76 TOPS W-1 for 8-bit input/output matrix-vector",
    "  multiplications\"; \"2.48 - 9.76 TOPS/W in 4-phase (high-precision) - 1-phase (low-precision) read mode\";",
    "  Extended Data Table I note b: \"1 multiply-and-accumulate = 2 OPs\"; \"> 4M weights\" per chip.",
    "[A100] NVIDIA A100 Tensor Core GPU datasheet: FP32 19.5 TFLOPS; FP16 Tensor Core 312 TFLOPS (624 with sparsity);",
    "  INT8 Tensor Core 624 TOPS (1,248 with sparsity); 80GB SXM: HBM2e 2,039 GB/s, max TDP 400 W.",
    "[K21] Karunaratne, G. et al. Robust high-dimensional memory-augmented neural networks. Nat. Commun. 12, 2468 (2021)",
    "  (abstract): bipolar/binary high-dimensional key vectors in more than 256,000 PCM devices, few-shot Omniglot accuracy",
    "  close to 32-bit software.",
]


def main(src, out):
    J = json.load(open(src))
    T = {r["cond"]: r for r in J["totals"]}
    f = lambda n: T[n]["frac"]
    L = []
    w = L.append
    w("# B25 verdict (prereg/B25_pcm_device_model_prereg.md): the RULER vector stores (bipolar) on a measured PCM device")
    w("# model (Joshi et al. 2020), crossbar mapping after Karunaratne et al. 2020, with a cited energy estimate")
    w("")
    ideal16 = T["B1 ideal | 16/vec"]
    w("Stores: the 50 B22 stores, %d probed keys. Registered check: the ideal B1 path reproduces B22 amendment 2 on every"
      % ideal16["keys"])
    w("store (asserted per store in b25_pcm.py): B1 ideal at 16 pairs per vector reads %d / %d."
      % (ideal16["keys_exact"], ideal16["keys"]))
    w("")
    w("Ideal stores (no device): fraction of keys exact (stores fully exact of 50)")
    for var in ("B1", "ML"):
        w("  %-3s " % var + "  ".join("%d/vec %.4f (%d)" % (ld, f("%s ideal | %d/vec" % (var, ld)),
                                                         T["%s ideal | %d/vec" % (var, ld)]["stores_all_exact"]) for ld in LOADS))
    for comp, title in (("gdc", "WITH global drift compensation (GDC)"), ("none", "WITHOUT compensation")):
        w("")
        w("Table %s: fraction of the %d keys read exactly, %s" % ("1" if comp == "gdc" else "2", ideal16["keys"], title))
        w("  S = store crossbar on PCM (clean-up digital); C = clean-up crossbar on PCM (store digital); SC = both")
        w("  %-16s" % "arm | load" + "".join("%9s" % t for t in TIMES))
        for arm in ARMS:
            for ld in LOADS:
                w("  %-16s" % ("%s | %d" % (arm, ld)) +
                  "".join("%9.4f" % f("%s %s | %s | %d/vec" % (arm, comp, t, ld)) for t in TIMES))
    w("")
    w("Diagnostics (means over the 50 stores):")
    D_ = [r["diag"] for r in J["per_store"]]
    for t in TIMES:
        a = np.mean([d["alpha_cb | %s | x1" % t] for d in D_ if "alpha_cb | %s | x1" % t in d])
        be = np.mean([d["B1 store | %s | 16/vec | x1 | 8-bit" % t]["bit_err"] for d in D_])
        re = np.mean([d["ML store | %s | 16/vec | x1 | 8-bit" % t]["rel_err_gdc"] for d in D_])
        w("  %-5s clean-up GDC scale alpha %.3f | B1 store bit errors %.2e | ML store rms error / rms(m) %.3f (GDC)"
          % (t, a, be, re))
    w("")
    w("Descriptive extras (30 days, 16 pairs per vector, GDC; no bars):")
    for var in ("B1", "ML"):
        for st in ("S", "C", "SC"):
            w("  %-6s noise x2..x32: " % ("%s-%s" % (var, st)) + "  ".join(
                "x%d %.4f" % (k, f("X %s-%s gdc | 30 d | 16/vec | noise x%d" % (var, st, k))) for k in (2, 4, 8, 16, 32)))
        w("  %-6s ADC 6-bit %.4f  ADC 4-bit %.4f  read noise redrawn per MVM %.4f" % (
            var + "-SC", f("X %s-SC gdc | 30 d | 16/vec | ADC 6-bit" % var), f("X %s-SC gdc | 30 d | 16/vec | ADC 4-bit" % var),
            f("X %s-SC gdc | 30 d | 16/vec | read noise redrawn per MVM" % var)))
    be_k = {k: np.mean([d["B1 store | 30 d | 16/vec | x%d | 8-bit" % k]["bit_err"] for d in D_]) for k in (2, 4, 8, 16, 32)}
    al_k = {k: np.mean([d["alpha_cb | 30 d | x%d" % k] for d in D_ if "alpha_cb | 30 d | x%d" % k in d]) for k in (2, 4, 8, 16, 32)}
    w("  B1 store bit errors at x2..x32: " + "  ".join("x%d %.3f" % (k, v) for k, v in be_k.items()))
    w("  clean-up GDC alpha at x2..x32:  " + "  ".join("x%d %.3f" % (k, v) for k, v in al_k.items())
      + "  (x1: %.3f)" % np.mean([d["alpha_cb | 30 d | x1"] for d in D_ if "alpha_cb | 30 d | x1" in d]))

    # ---- bars ----
    w("")
    w("Bars:")
    h1 = f("B1-SC gdc | 1 d | 16/vec")
    h2 = f("B1-SC gdc | 30 d | 16/vec")
    h3 = f("ML-SC gdc | 1 d | 16/vec")
    h4 = [(ld, f("B1-SC gdc | 1 d | %d/vec" % ld), f("B1 ideal | %d/vec" % ld)) for ld in (32, 64)]
    P = lambda c: "PASS" if c else "FAIL"
    w("  H1 B1-SC GDC 16/vec 1 d >= 99.5%%: %s (%.4f)" % (P(h1 >= 0.995), h1))
    w("  H2 B1-SC GDC 16/vec 30 d >= 99.5%%: %s (%.4f)" % (P(h2 >= 0.995), h2))
    w("  H3 ML-SC GDC 16/vec 1 d >= 99.5%%: %s (%.4f)" % (P(h3 >= 0.995), h3))
    w("  H4 B1-SC GDC 1 d within 0.5 points of ideal at 32 and 64/vec: %s (%s)" % (
        P(all(ideal - x <= 0.005 for _, x, ideal in h4)),
        "; ".join("%d/vec %.4f vs ideal %.4f" % (ld, x, i) for ld, x, i in h4)))

    # ---- predictions ----
    w("")
    w("Predictions:")
    held = lambda c: "held" if c else "WRONG"
    p1 = [(a, ld, t, f("%s gdc | %s | %d/vec" % (a, t, ld))) for a in ARMS for ld in (16, 32) for t in TIMES]
    worst = min(p1, key=lambda x: x[3])
    w("  P1 GDC, every arm at 16 and 32/vec, every time >= 99.9%%: %s (lowest %s %d/vec %s: %.4f)"
      % (held(worst[3] >= 0.999), worst[0], worst[1], worst[2], worst[3]))
    p2 = [(a, t, f("%s gdc | %s | 64/vec" % (a, t)) - f("%s ideal | 64/vec" % a[:2])) for a in ARMS for t in TIMES]
    worst2 = max(p2, key=lambda x: abs(x[2]))
    w("  P2 GDC at 64/vec within 0.2 points of ideal: %s (largest gap %s %s: %+.4f); ideal B1 64/vec >= 99%%: %s (%.4f)"
      % (held(abs(worst2[2]) <= 0.002), worst2[0], worst2[1], worst2[2], held(f("B1 ideal | 64/vec") >= 0.99),
         f("B1 ideal | 64/vec")))
    n = lambda a, t: f("%s none | %s | 16/vec" % (a, t))
    sub = []
    for a in ("B1-C", "B1-SC"):
        sub.append(("%s none >= 99.5%% at 60 s and 1 h" % a, n(a, "60 s") >= 0.995 and n(a, "1 h") >= 0.995,
                    "%.4f, %.4f" % (n(a, "60 s"), n(a, "1 h"))))
        sub.append(("%s none < 99.5%% at 1 d" % a, n(a, "1 d") < 0.995, "%.4f" % n(a, "1 d")))
        sub.append(("%s none < 20%% at 30 d" % a, n(a, "30 d") < 0.20, "%.4f" % n(a, "30 d")))
    same = all(f("B1-S none | %s | %d/vec" % (t, ld)) == f("B1-S gdc | %s | %d/vec" % (t, ld)) for t in TIMES for ld in LOADS)
    sub.append(("B1-S unchanged by compensation", same, ""))
    for a in ("ML-S", "ML-C"):
        sub.append(("%s none >= 99%% at 1 d, < 80%% at 30 d, < 20%% at 1 y" % a,
                    n(a, "1 d") >= 0.99 and n(a, "30 d") < 0.80 and n(a, "1 y") < 0.20,
                    "%.4f, %.4f, %.4f" % (n(a, "1 d"), n(a, "30 d"), n(a, "1 y"))))
    sub.append(("ML-SC none < 99.5% at 1 h and < 5% at 1 d", n("ML-SC", "1 h") < 0.995 and n("ML-SC", "1 d") < 0.05,
                "%.4f, %.4f" % (n("ML-SC", "1 h"), n("ML-SC", "1 d"))))
    for txt, ok, val in sub:
        w("  P3 %s: %s %s" % (txt, held(ok), ("(" + val + ")") if val else ""))
    x = lambda var, st, k: f("X %s-%s gdc | 30 d | 16/vec | noise x%d" % (var, st, k))
    w("  P4 B1-SC >= 99.5%% up to x4, < 99.5%% at x8: %s (x4 %.4f, x8 %.4f)"
      % (held(x("B1", "SC", 2) >= .995 and x("B1", "SC", 4) >= .995 and x("B1", "SC", 8) < .995), x("B1", "SC", 4), x("B1", "SC", 8)))
    w("  P4 B1-S >= 99.5%% up to x4, > 90%% at x8, < 50%% at x16: %s (x4 %.4f, x8 %.4f, x16 %.4f)"
      % (held(all(x("B1", "S", k) >= .995 for k in (2, 4)) and x("B1", "S", 8) > 0.90 and x("B1", "S", 16) < 0.50),
         x("B1", "S", 4), x("B1", "S", 8), x("B1", "S", 16)))
    w("  P4 ML-SC >= 99.5%% at x2, < 99.5%% at x4: %s (%.4f, %.4f)"
      % (held(x("ML", "SC", 2) >= .995 and x("ML", "SC", 4) < .995), x("ML", "SC", 2), x("ML", "SC", 4)))
    a6 = [f("X %s-SC gdc | 30 d | 16/vec | ADC 6-bit" % v) for v in ("B1", "ML")]
    w("  P4 6-bit ADC >= 99.5%% for both forms: %s (B1 %.4f, ML %.4f)" % (held(min(a6) >= .995), a6[0], a6[1]))
    rd = [(f("X %s-SC gdc | 30 d | 16/vec | read noise redrawn per MVM" % v), f("%s-SC gdc | 30 d | 16/vec" % v)) for v in ("B1", "ML")]
    w("  P4 read noise redrawn per MVM within 0.1 points: %s (B1 %.4f vs %.4f, ML %.4f vs %.4f)"
      % (held(all(abs(a - b) <= 0.001 for a, b in rd)), rd[0][0], rd[0][1], rd[1][0], rd[1][1]))

    # ---- energy ----
    nd = N_MAX * D
    w("")
    w("Energy estimate (cited per-operation figures x operation counts; an estimate, not a measurement):")
    w("  One clean-up step = N x d MACs; largest store N = %d, d = %d -> %.1f M MACs; a one-value key takes 2 steps."
      % (N_MAX, D, nd / 1e6))
    w("  In-memory (PCM crossbar), per step:")
    w("    Karunaratne 2020, AM search 9.44 nJ / (22 x 10,000) = %.1f fJ/MAC -> %.2f uJ" % (E_KAR * 1e15, nd * E_KAR * 1e6))
    w("    Le Gallo 2023, 9.76 TOPS/W (1-phase, 8-bit I/O) = %.3f pJ/MAC -> %.1f uJ" % (E_HER1 * 1e12, nd * E_HER1 * 1e6))
    w("    Le Gallo 2023, 2.48 TOPS/W (4-phase)            = %.3f pJ/MAC -> %.1f uJ" % (E_HER4 * 1e12, nd * E_HER4 * 1e6))
    w("    store-row read (B1: d sense-amplifier reads x 9.8 fJ) = %.0f pJ (negligible); digital unbinding neglected"
      % (D * E_SA * 1e12))
    w("  GPU (NVIDIA A100 SXM, datasheet), per step:")
    for nm, peak in (("INT8 tensor 624 TOPS", INT8), ("FP16 tensor 312 TFLOPS", FP16), ("FP32 19.5 TFLOPS", FP32)):
        e = 2 * TDP / peak
        w("    compute at peak, %-22s: %.2f pJ/MAC -> %.0f uJ" % (nm, e * 1e12, nd * e * 1e6))
    eb = TDP / BW
    w("    one unbatched read, memory-bound (INT8 codebook at 2,039 GB/s, 400 W): %.0f pJ/MAC -> %.1f mJ" % (eb * 1e12, nd * eb * 1e3))
    w("  Ratio GPU / in-memory per step: best case for the GPU (INT8 at peak) vs Karunaratne %.0fx, vs Le Gallo 1-phase %.1fx,"
      % (2 * TDP / INT8 / E_KAR, 2 * TDP / INT8 / E_HER1))
    w("  vs Le Gallo 4-phase %.1fx; unbatched memory-bound GPU read vs Le Gallo 4-phase %.0fx." % (2 * TDP / INT8 / E_HER4, eb / E_HER4))
    w("  Footprint: 2 x N x d = %.1f M PCM devices for the largest codebook (0.2 um^2 each per Karunaratne: %.0f mm^2 of"
      % (2 * nd / 1e6, 2 * nd * 0.2e-6))
    w("  devices), %.1f times the 64 x 256 x 256 = 4.19 M weights of one Le Gallo 2023 chip." % (nd / (64 * 256 * 256)))
    w("  Sources: Karunaratne et al., Nat. Electron. 3, 327-337 (2020), Table 1 and Extended Data Tables 1-2; Le Gallo et al.,")
    w("  Nat. Electron. 6, 680-693 (2023), abstract/Sec. VI and Extended Data Table I footnote b (arXiv:2212.02872); NVIDIA A100")
    w("  Tensor Core GPU datasheet (SXM 80GB column).")
    w("")
    w("Citations and the exact numbers used (each checked against the paper itself):")
    for line in CITES:
        w("  " + line)
    if len(sys.argv) > 3:                     # interpretation and literature check, written after the results
        w("")
        w(open(sys.argv[3]).read().rstrip())
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
