"""B27 verdict (prereg/B27_arena_referee_baselines_prereg.md). Written before any test output exists.

Second implementation of the scores: every test accuracy is recomputed here from the saved per-instance scores
(LoRA: logit margin; A: logit margin; B/C: final cosine with the saved training threshold) and labels regenerated from
the arena generators, then compared with the number the run logged. Runs on the box (needs the tokenizer):

  python b27_verdict.py --root /path/to/work/b27 --out results_dir
"""
import argparse, glob, json, math, os, sys
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NAMING = {"A": [0.915, 0.913, 0.913], "B": [0.990, 1.000, 0.998], "C": [0.932, 0.932, 0.938]}
FLOOR = {"A": [0.508, 0.511, 0.508], "B": [0.794, 0.818, 0.832], "C": [0.512, 0.530, 0.504]}
SPANS = {"A": [0.511, 0.499, 0.493], "B": [0.502, 0.498, 0.522], "C": [0.472, 0.502, 0.476]}
LORA_PUB = {"A": [0.493, 0.508, 0.515], "B": [0.980, 0.895, 0.510], "C": [0.515, 0.5175, 0.515]}
SEEDS = [42, 123, 456]
FAMS = ["orth", "vic", "nce"]


def mean(x):
    return float(np.mean(x))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_mean(correct, B=10000, seed=0):
    """correct: (seeds, n) 0/1 matrix on the same test instances; resample instances, mean over seeds."""
    rng = np.random.default_rng(seed)
    C = np.asarray(correct, float); n = C.shape[1]
    idx = rng.integers(0, n, size=(B, n))
    m = C[:, idx].mean(axis=2).mean(axis=0)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def labels(arena):
    import b27_common as C
    M = C.load_arena_modules()
    tok = C.tokenizer()
    r = C.build_rows(M, tok, arena, "test")
    if arena == "A":
        return np.array([int(it["labels"]) for it in r.items])
    return np.array([int(x["label"]) for x in r])


def dev_acc(e):
    return e["dev"]["acc"] if isinstance(e["dev"], dict) else e["dev"]


def test_acc(e):
    return e["test"]["acc"] if isinstance(e["test"], dict) else e["test"]


def pick(epochs):
    """dev-selected epoch: highest dev accuracy, earliest on ties."""
    b, be = -1, None
    for e in epochs:
        if dev_acc(e) > b + 1e-12:
            b, be = dev_acc(e), e
    return be


def recompute(arena, e, y, kind):
    s = np.asarray(e["test_scores"], float)
    if kind == "lora" or arena == "A":
        pred = (s > 0).astype(int)
    else:
        pred = (s >= e["thr"]).astype(int)
    corr = (pred == y).astype(int)
    return corr, float(corr.mean()), float(pred.mean())


def summarize_runs(arena, runs, y, kind, log):
    """per seed: dev-selected epoch, recomputed test acc there, final epoch, best test epoch, train fit."""
    out = []
    for r in runs:
        ep = pick(r["epochs"])
        corr, acc, pos = recompute(arena, ep, y, kind)
        logged = test_acc(ep)
        if abs(acc - logged) > 1e-9:
            log.append("MISMATCH %s %s seed %d epoch %d: recomputed %.4f logged %.4f" % (kind, arena, r["seed"],
                                                                                      ep["epoch"], acc, logged))
        tr = (lambda e: e["train"]["acc"] if "train" in e else None) if kind == "lora" else (lambda e: e["train_acc"])
        trs = [tr(e) for e in r["epochs"] if tr(e) is not None]
        k = int(corr.sum()); n = len(corr)
        out.append({"seed": r["seed"], "epoch": ep["epoch"], "dev": dev_acc(ep), "test": acc, "k": k, "n": n,
                    "wilson": wilson(k, n), "pos_rate": pos, "final_test": test_acc(r["epochs"][-1]),
                    "best_test": max(test_acc(e) for e in r["epochs"]),
                    "train_at_epoch": tr(ep), "train_max": max(trs),
                    "train_final": tr(r["epochs"][-1]), "corr": corr,
                    "geom": ep.get("geom_test"), "geom_final": r["epochs"][-1].get("geom_test")})
    out.sort(key=lambda x: SEEDS.index(x["seed"]))
    return out


def load_runs(pattern):
    """Concatenate the runs of every output matching the pattern (one file per seed or one file for all seeds);
    the configuration must be identical across files apart from the seeds and the output path."""
    fs = sorted(glob.glob(pattern))
    if not fs:
        return None
    ds = [json.load(open(p)) for p in fs]
    key = lambda d: {k: v for k, v in d["args"].items() if k not in ("seeds", "out", "device")}
    assert all(key(d) == key(ds[0]) for d in ds), "configuration differs across %s" % fs
    runs = [r for d in ds for r in d["runs"]]
    assert sorted(r["seed"] for r in runs) == sorted(set(r["seed"] for r in runs)), "duplicate seeds %s" % fs
    return {"args": key(ds[0]), "runs": runs, "files": [os.path.basename(p) for p in fs]}


def fmt_seed(s):
    return "%.3f [%.3f-%.3f] ep%d" % (s["test"], s["wilson"][0], s["wilson"][1], s["epoch"])


def gpu_hours(root):
    p = os.path.join(root, "logs", "gpu_ledger.txt")
    starts, rows = {}, []
    for l in open(p):
        name, gpu, what, ts = l.split()[:4]
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        if what == "start":
            starts[(name, gpu)] = t
        elif (name, gpu) in starts:
            rows.append((name, gpu, starts.pop((name, gpu)), t))
    open_ = [(n, g, t) for (n, g), t in starts.items()]
    return rows, open_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/path/to/work/b27")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    R = a.root
    L, log, J = [], [], {"lf": {}, "lora": {}, "g0": {}}
    w = L.append
    w("# B27 verdict (prereg/B27_arena_referee_baselines_prereg.md): label-free anti-collapse projections and a tuned")
    w("# LoRA fine-tune in the three controlled arenas (referee baselines). Generated by b27_verdict.py.")
    w("")
    # ---------------- gates
    w("## Gates")
    for p in sorted(glob.glob(os.path.join(R, "out", "g0", "g0a_*.json"))):
        for r in json.load(open(p)):
            ok = abs(r["acc"] - r["banked"]) <= 0.004 + 1e-9
            J["g0"].setdefault("a", []).append({k: r[k] for k in ("arena", "arm", "seed", "acc", "banked")})
            w("  G0a %s %-17s seed %3d  acc %.4f  banked %.3f  %s" % (r["arena"], r["arm"], r["seed"], r["acc"],
                                                                     r["banked"], "PASS" if ok else "FAIL"))
    for p in sorted(glob.glob(os.path.join(R, "out", "g0", "g0b_*.json"))):
        d = json.load(open(p))
        for r in d["runs"]:
            ok = abs(r["g0_best"] - r["g0_banked"]) <= 0.004 + 1e-9
            J["g0"].setdefault("b", []).append({"arena": d["args"]["arena"], "seed": r["seed"],
                                                "acc": r["g0_best"], "banked": r["g0_banked"]})
            w("  G0b %s binding loss (b27_lf, R off) seed %3d  best test %.4f  banked %.3f  %s" % (
                d["args"]["arena"], r["seed"], r["g0_best"], r["g0_banked"], "PASS" if ok else "FAIL"))
    w("")
    # ---------------- selections
    sel_lf = json.load(open(os.path.join(R, "out", "select_lf.json")))
    sel_lora = json.load(open(os.path.join(R, "out", "select_lora.json")))
    w("## Dev selections (train + dev only)")
    for ar in "ABC":
        s = sel_lf["selected"].get(ar)
        if s:
            for f in FAMS:
                c = s["families"].get(f)
                if c:
                    w("  label-free %s %-4s lambda %-4g lr %.0e qc %-5s  dev %.3f (epoch %d)%s" % (
                        ar, f, c["lam"], c["lr"], c["qc"], c["dev"], c["epoch"], "  <- primary" if f == s["primary"] else ""))
        c = sel_lora["selected"].get(ar)
        if c:
            w("  LoRA       %s rank %d lr %.0e %s epochs %d  dev %.3f (epoch %d)  train max %.3f  fit %s" % (
                ar, c["rank"], c["lr"], c.get("sched", "cosine"), c["epochs"], c["dev"], c["epoch"], c["train_max"], c["fit"]))
    w("  LoRA configurations tried (seed 42): " + ", ".join(
        "%s r%d lr%.0e %s ep%d dev %.3f train %.3f" % (x["arena"], x["rank"], x["lr"], x.get("sched", "cosine"), x["epochs"],
                                                      x["dev"], x["train_max"])
        for x in sorted(sel_lora["configs"], key=lambda x: (x["arena"], x["epochs"], x["rank"], x.get("sched", ""), x["lr"]))))
    w("")
    # ---------------- results
    Y = {ar: labels(ar) for ar in "ABC"}
    pred = {}
    w("## Point 1: label-free projections (test at each seed's dev-selected epoch; per seed acc [Wilson 95%])")
    for ar in "ABC":
        F, N = mean(FLOOR[ar]), mean(NAMING[ar])
        w("  arena %s: floor %.3f, naming head %.3f (%s), binding loss as published %.3f" % (
            ar, F, N, "/".join("%.3f" % x for x in NAMING[ar]), mean(SPANS[ar])))
        for f in FAMS:
            d = load_runs(os.path.join(R, "out", "lf_final", "%s_%s*.json" % (ar, f)))
            if d is None:
                continue
            S = summarize_runs(ar, d["runs"], Y[ar], "lf", log)
            accs = [s["test"] for s in S]
            rec = [(x - F) / (N - F) for x in accs]
            m = mean(accs); mr = (m - F) / (N - F)
            lo, hi = boot_mean([s["corr"] for s in S])
            if mr >= 0.90 and all(r >= 0.80 for r in rec) and len(rec) == 3:
                v = "MATCH"
            elif mr >= 0.25:
                v = "PARTIAL"
            else:
                v = "FAIL"
            prim = sel_lf["selected"][ar]["primary"] == f
            J["lf"]["%s_%s" % (ar, f)] = {"primary": prim, "mean": m, "ci": [lo, hi], "recovery": mr, "verdict": v,
                                          "seeds": [{k: s[k] for k in s if k != "corr"} for s in S],
                                          "config": d["args"]}
            w("    %-4s%s mean %.3f [%.3f-%.3f]  recovery %.2f -> %s" % (f, "*" if prim else " ", m, lo, hi, mr, v))
            w("          seeds " + "; ".join(fmt_seed(s) for s in S))
            w("          final epoch " + "/".join("%.3f" % s["final_test"] for s in S) +
              "   best test epoch (published protocol) " + "/".join("%.3f" % s["best_test"] for s in S) +
              "   train acc at selected epoch " + "/".join("%.3f" % s["train_at_epoch"] for s in S))
            g = [s["geom"] for s in S if s["geom"]]
            if g:
                w("          cross-talk on test at the selected epoch: different-symbol %s, same-symbol %s" % (
                    "/".join("%.3f" % x["different"] for x in g), "/".join("%.3f" % x["same"] for x in g)))
        w("    (* = primary, chosen on dev)")
    d = load_runs(os.path.join(R, "out", "lf_explore", "A_nce_ep30_s*.json"))
    if d is not None:
        ar = "A"; F, N = mean(FLOOR[ar]), mean(NAMING[ar])
        S = summarize_runs(ar, d["runs"], Y[ar], "lf", log)
        accs = [s["test"] for s in S]; rec = [(x - F) / (N - F) for x in accs]
        m = mean(accs); mr = (m - F) / (N - F)
        lo, hi = boot_mean([s["corr"] for s in S])
        v = "MATCH" if (mr >= 0.90 and all(r >= 0.80 for r in rec) and len(rec) == 3) else ("PARTIAL" if mr >= 0.25 else "FAIL")
        J["lf"]["A_nce_ep30_exploratory"] = {"mean": m, "ci": [lo, hi], "recovery": mr, "verdict": v,
                                             "seeds": [{k: s[k] for k in s if k != "corr"} for s in S]}
        w("  EXPLORATORY (prereg/B27_exploratory1.md, not registered): A nce, same configuration, 30 epochs")
        w("    mean %.3f [%.3f-%.3f]  recovery %.2f -> %s" % (m, lo, hi, mr, v))
        w("          seeds " + "; ".join(fmt_seed(s) for s in S))
        w("          final epoch " + "/".join("%.3f" % s["final_test"] for s in S) +
          "   best test epoch " + "/".join("%.3f" % s["best_test"] for s in S) +
          "   train acc at selected epoch " + "/".join("%.3f" % s["train_at_epoch"] for s in S))
        g = [s["geom"] for s in S if s["geom"]]
        if g:
            w("          cross-talk on test at the selected epoch: different-symbol %s, same-symbol %s" % (
                "/".join("%.3f" % x["different"] for x in g), "/".join("%.3f" % x["same"] for x in g)))
    w("")
    w("## Point 2: tuned LoRA fine-tune (test at each seed's dev-selected epoch)")
    for ar in "ABC":
        d = load_runs(os.path.join(R, "out", "lora_final", "%s_*.json" % ar))
        if d is None:
            continue
        S = summarize_runs(ar, d["runs"], Y[ar], "lora", log)
        accs = [s["test"] for s in S]
        m = mean(accs); M = mean(NAMING[ar]) - m
        ms = [NAMING[ar][SEEDS.index(s["seed"])] - s["test"] for s in S]
        lo, hi = boot_mean([s["corr"] for s in S])
        fit_tune = sel_lora["selected"][ar]["fit"]
        fit_all = fit_tune and all(s["train_max"] >= 0.95 for s in S)
        v = "STANDS" if (M >= 0.10 and all(x >= 0.10 for x in ms)) else ("CLOSED" if M <= 0.03 else "NARROWED")
        J["lora"][ar] = {"mean": m, "ci": [lo, hi], "margin": M, "margins": ms, "verdict": v, "fitted": fit_all,
                         "seeds": [{k: s[k] for k in s if k != "corr"} for s in S], "config": d["args"]}
        w("  arena %s: LoRA mean %.3f [%.3f-%.3f]  (published %.3f)  naming head %.3f  margin %+.3f (published %+.3f) -> %s%s" % (
            ar, m, lo, hi, mean(LORA_PUB[ar]), mean(NAMING[ar]), M, mean(NAMING[ar]) - mean(LORA_PUB[ar]), v,
            "" if fit_all else "  [under-fitted baseline]"))
        w("    seeds " + "; ".join(fmt_seed(s) for s in S) + "   per-seed margins " + "/".join("%+.3f" % x for x in ms))
        w("    training accuracy: at selected epoch " + "/".join("-" if s["train_at_epoch"] is None else "%.3f" % s["train_at_epoch"] for s in S) +
          ", max " + "/".join("%.3f" % s["train_max"] for s in S) + ", final " + "/".join("%.3f" % s["train_final"] for s in S) +
          "   fitted (>= 0.95 in tuning and every seed): %s" % fit_all)
        w("    final epoch " + "/".join("%.3f" % s["final_test"] for s in S) + "   best test epoch " +
          "/".join("%.3f" % s["best_test"] for s in S) + "   predicted-positive rate " +
          "/".join("%.2f" % s["pos_rate"] for s in S))
    w("")
    # ---------------- predictions
    w("## Predictions")
    def P(text, ok):
        w("  %s: %s" % (text, "HELD" if ok else ("NOT HELD" if ok is not None else "n/a")))
    g0 = J["g0"]
    P("G0a and G0b pass in every cell", all(abs(x["acc"] - x["banked"]) <= 0.004 + 1e-9
                                             for k in ("a", "b") for x in g0.get(k, [])) if g0 else None)
    prim = {ar: J["lf"].get("%s_%s" % (ar, sel_lf["selected"][ar]["primary"])) for ar in "ABC" if ar in sel_lf["selected"]}
    want = {"A": "FAIL", "B": "PARTIAL", "C": "FAIL"}
    for ar in "ABC":
        if prim.get(ar):
            P("label-free primary arm %s = %s (measured %s)" % (ar, want[ar], prim[ar]["verdict"]),
              prim[ar]["verdict"] == want[ar])
    P("no label-free family reaches MATCH in any arena (registered runs)",
      not any(v["verdict"] == "MATCH" for k, v in J["lf"].items() if "exploratory" not in k))
    gC = [s["geom"] for k, v in J["lf"].items() if k in ("C_orth", "C_vic") for s in v["seeds"] if s["geom"]]
    if gC:
        P("C orth/vic: different-symbol cosine <= 0.25 and same-symbol <= 0.6 on test (every seed)",
          all(g["different"] <= 0.25 and g["same"] <= 0.6 for g in gC))
    for ar in "BC":
        if ar in sel_lf["selected"]:
            q = [sel_lf["selected"][ar]["families"][f]["qc"] for f in sel_lf["selected"][ar]["families"]]
            P("qc = proj selected on dev in %s (families: %s)" % (ar, ",".join(q)), all(x == "proj" for x in q))
    for ar in "ABC":
        if ar in J["lora"]:
            P("LoRA fits its training split in %s" % ar, J["lora"][ar]["fitted"])
    if "B" in J["lora"]:
        P("LoRA B mean >= 0.95 and every seed >= 0.90", J["lora"]["B"]["mean"] >= 0.95 and
          all(s["test"] >= 0.90 for s in J["lora"]["B"]["seeds"]))
    if "A" in J["lora"]:
        P("LoRA A mean <= 0.70", J["lora"]["A"]["mean"] <= 0.70)
    if "C" in J["lora"]:
        P("LoRA C mean <= 0.65", J["lora"]["C"]["mean"] <= 0.65)
    for ar, want2 in (("A", "STANDS"), ("C", "STANDS"), ("B", "CLOSED")):
        if ar in J["lora"]:
            P("Point 2 %s = %s (measured %s)" % (ar, want2, J["lora"][ar]["verdict"]), J["lora"][ar]["verdict"] == want2)
    if "C" in J["lora"]:
        s = J["lora"]["C"]["seeds"]
        P("LoRA C memorises: train >= 0.95 with dev <= 0.65 (every seed)",
          all(x["train_max"] >= 0.95 and x["dev"] <= 0.65 for x in s))
    w("")
    # ---------------- compute
    rows, open_ = gpu_hours(R)
    tot = sum((t1 - t0).total_seconds() for _, _, t0, t1 in rows) / 3600
    w("## Compute (GPU occupancy per B27 worker, logs/gpu_ledger.txt; A800 80 GB)")
    for n, g, t0, t1 in rows:
        w("  %-14s %s %s -> %s  %.2f h" % (n, g, t0.strftime("%m-%d %H:%M"), t1.strftime("%H:%M"),
                                          (t1 - t0).total_seconds() / 3600))
    for n, g, t in open_:
        w("  %-14s %s %s -> (still running)" % (n, g, t.strftime("%m-%d %H:%M")))
    w("  Total %.1f GPU-hours (cap 40)" % tot)
    J["gpu_hours"] = tot
    w("")
    w("## Second-implementation check")
    w("  " + ("; ".join(log) if log else "every test accuracy recomputed from saved scores and regenerated labels equals the logged value"))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "b27_verdict.txt"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(a.out, "b27_scores.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
