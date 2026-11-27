"""B44 verdict (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md). Reads every saved output (train / dev / test),
applies the registered selection and bars, and writes the verdict, the scores and the Fig. 2b,c data.

  python b44_verdict.py --root b44_out --out results/b44_verdict.txt --scores results/b44_scores.json \
      --fig results/b44_fig2bc.json
"""
import argparse, glob, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SEEDS = [42, 123, 456]
SIZES = [800, 8000, 80000]
B = 10000
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    L.append(s); print(s)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


RNG = np.random.default_rng(4444)
IDX = {}


def boot_idx(n):
    if n not in IDX:
        IDX[n] = RNG.integers(0, n, size=(B, n))
    return IDX[n]


def mean_ci(C):
    """C: (seeds, n) 0/1 correctness. Mean over seeds; bootstrap over instances (shared resample across seeds)."""
    C = np.asarray(C, dtype=float)
    I = boot_idx(C.shape[1])
    bs = C[:, I].mean(axis=(0, 2))
    return float(C.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def diff_ci(C1, C2):
    C1, C2 = np.asarray(C1, float), np.asarray(C2, float)
    I = boot_idx(C1.shape[1])
    bs = C1[:, I].mean(axis=(0, 2)) - C2[:, I].mean(axis=(0, 2))
    return float(C1.mean() - C2.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def f3(x):
    return "%.3f" % x if x is not None and not (isinstance(x, float) and math.isnan(x)) else "-"


# ------------------------------------------------------------------ loading

def labels(variant):
    import b44_indep as I
    import b44_data as BD
    return np.array([I.walk(s["text"])[0] for s in BD.split(variant, "test", struct=False)])


def lora_cell(root, task, variant, n, sel):
    """Per-seed correctness vectors of the selected configuration (seed 42 = the tuning run of that configuration)."""
    s = sel.get("%s_%s_n%d" % ("lora" if task == "cls" else "scratch", variant, n))
    if not s:
        return None
    out = {"lr": s["lr"], "rule": s.get("rule"), "C": [], "acc": [], "info": []}
    for seed in SEEDS:
        base = os.path.join(root, "lora", "%s_%s_n%d_r8_lr%s_s%d.json" % (task, variant, n, fmt_lr(s["lr"]), seed))
        if not (os.path.exists(base) and os.path.exists(base + ".test.json")):
            return dict(out, missing=os.path.basename(base))
        d = json.load(open(base)); t = json.load(open(base + ".test.json"))
        y = LAB[variant]
        if task == "cls":
            pred = np.array([int(x > 0) for x in t["test"]["scores"]])
        else:
            import b44_lora as BL
            pred = np.array([-1 if BL.parse_scratch(g)[0] is None else BL.parse_scratch(g)[0] for g in t["test"]["gens"]])
        c = (pred == y).astype(int)
        assert abs(c.mean() - t["test"]["acc"]) < 1e-6, (base, c.mean(), t["test"]["acc"])
        r = d["run"]
        out["C"].append(c); out["acc"].append(float(c.mean()))
        out["info"].append({"best_step": r["best_step"], "best_dev": r["best_dev"], "early_stop": r.get("early_stop"),
                            "train_sub_max": max(e["train_sub"]["acc"] for e in r["evals"]),
                            "instances_seen": r["evals"][-1]["instances"], "secs": r["secs"],
                            "test_extra": {k: v for k, v in t["test"].items() if k not in ("scores", "gens")}})
    return out


def fmt_lr(lr):
    return {1e-4: "1e-4", 3e-4: "3e-4"}[lr]


def mem_cell(root, arm, cell, enc="qwen8"):
    """name / exact (b31_arms.run outputs) or probe; per-seed correctness at the dev-selected epoch / step."""
    pat = re.compile(r"^%s_%s_%s(_s\d+)?\.json$" % (enc, arm, cell.replace(":", "_n")))
    fs = sorted(f for f in glob.glob(os.path.join(root, "name", "*.json")) if pat.match(os.path.basename(f)))
    if not fs:
        return None
    runs = []
    for f in fs:
        runs += json.load(open(f))["runs"]
    runs = sorted(runs, key=lambda r: SEEDS.index(r["seed"]))
    out = {"C": [], "acc": [], "geom": [], "epoch": [], "naming": [], "C_comm": []}
    v = cell.split(":")[0]
    y = LAB.get(v)
    for r in runs:
        if arm == "probe":
            pred = np.array([int(x > 0) for x in r["test_scores"]])
            out["C"].append((pred == y).astype(int) if y is not None else None); out["acc"].append(r["test_acc"])
            continue
        devs = [e["perm/named:dev"]["acc"] for e in r["epochs"]]
        i = max(range(len(devs)), key=lambda j: (devs[j], -j))
        te = r["epochs"][i]["perm/named:test"]
        pred = np.array([int(x >= te["thr"]) for x in te["scores"]])
        if y is not None:
            c = (pred == y).astype(int)
            assert abs(c.mean() - te["acc"]) < 1e-6
            out["C"].append(c)
        out["acc"].append(te["acc"]); out["geom"].append(te.get("geom")); out["epoch"].append(r["epochs"][i]["epoch"])
        out["naming"].append(te.get("naming_acc_mentions"))
        if "comm/named:test" in r["epochs"][i]:
            out["C_comm"].append(r["epochs"][i]["comm/named:test"]["acc"])
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="b44_out")
    ap.add_argument("--out", default="results/b44_verdict.txt")
    ap.add_argument("--scores", default="results/b44_scores.json")
    ap.add_argument("--fig", default="results/b44_fig2bc.json")
    a = ap.parse_args()
    global LAB
    LAB = {v: labels(v) for v in ("std", "ho")}
    sel = json.load(open(os.path.join(a.root, "select.json")))
    S = {}
    P("# B44 verdict (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md). Generated by b44_verdict.py on %s UTC."
      % __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").gmtime()))
    P("# Arena C, Qwen3-8B (frozen states for the memory arms; LoRA / scratchpad fine-tunes; prompting). Test: std = B31's")
    P("# arena-C test (500); ho = held-out-pair test (500). Accuracy per seed (42/123/456) [Wilson 95%], mean [bootstrap 95%].")
    P("")
    P("## Selections (train + dev only)")
    for k, v in sorted(sel.get("lora", {}).items()):
        P("  %-22s %s" % (k, json.dumps(v)))
    for k, v in sorted(sel.get("grids", {}).items()):
        P("  seed-42 runs %-12s %s" % (k, "; ".join("lr %s dev %.3f train_sub %.3f%s" % (fmt_lr(c["lr"]), c["dev"], c["train_sub_max"],
                                                                            " early-stop@%s" % c["early_stop"] if c.get("early_stop") else "")
                                                for c in sorted(v, key=lambda c: c["lr"]))))
    if "cot" in sel:
        P("  cot dev100: %s -> selected %s" % (json.dumps(sel["cot"]["dev100"]), sel["cot"]["selected"]))
    P("")
    cells = {}
    for v in ("std", "ho"):
        P("## Variant %s: test accuracy" % v)
        P("    dictionary (exact symbols, regex walk of the raw text)   mean 1.000  (labels = dictionary by construction; %d/500 positive)"
          % int(LAB[v].sum()))
        S["%s|dict" % v] = {"mean": 1.0}
        for n in SIZES:
            P("  n = %d training instances" % n)
            row = {}
            ex = mem_cell(a.root, "exact", "%s:%d" % (v, n))
            for key, lab, obj in (("exact", "exact symbols + VSA (perm)", ex),
                                  ("name", "naming head + VSA (perm, named)", mem_cell(a.root, "name", "%s:%d" % (v, n))),
                                  ("probe", "equalised probe", mem_cell(a.root, "probe", "%s:%d" % (v, n))),
                                  ("lora", "equalised LoRA", lora_cell(a.root, "cls", v, n, sel.get("lora", {}))),
                                  ("scratch", "scratchpad LoRA", lora_cell(a.root, "scratch", v, n, sel.get("lora", {})))):
                if obj is None or obj.get("missing") or len(obj.get("C", [])) < 3 or any(c is None for c in obj["C"]):
                    P("    %-56s not run / incomplete%s" % (lab, " (%s)" % obj["missing"] if obj and obj.get("missing") else ""))
                    continue
                m, ci = mean_ci(obj["C"])
                ws = ["%.3f [%.3f-%.3f]" % (x, *wilson(round(x * len(LAB[v])), len(LAB[v]))) for x in obj["acc"]]
                extra = ""
                if key == "exact" and obj.get("C_comm"):
                    extra = "  (comm %s)" % "/".join(f3(x) for x in obj["C_comm"])
                P("    %-56s mean %.3f [%.3f-%.3f]  seeds %s%s" % (lab, m, ci[0], ci[1], ", ".join(ws), extra))
                if key == "name":
                    P("        cross-talk different/same %s; mentions named correctly %s; dev-best epoch %s" % (
                        ", ".join("%s/%s" % (f3(g["different"]), f3(g["same"])) for g in obj["geom"] if g),
                        "/".join(f3(x) for x in obj["naming"]), "/".join(str(e) for e in obj["epoch"])))
                if key in ("lora", "scratch"):
                    P("        lr %s (%s); dev-best step %s; train-subset max %s; instances seen %s; early stop %s" % (
                        fmt_lr(obj["lr"]), obj["rule"], "/".join(str(i["best_step"]) for i in obj["info"]),
                        "/".join(f3(i["train_sub_max"]) for i in obj["info"]),
                        "/".join(str(i["instances_seen"]) for i in obj["info"]),
                        "/".join(str(i["early_stop"]) for i in obj["info"])))
                    if key == "scratch":
                        P("        test bridge / bridge-hex / answer-symbol accuracy, unparsed: %s" % ", ".join(
                            "%s/%s/%s, %s" % (f3(i["test_extra"]["bridge_acc"]), f3(i["test_extra"]["bridge_hex_acc"]),
                                              f3(i["test_extra"]["answer_sym_acc"]), f3(i["test_extra"]["unparsed"]))
                            for i in obj["info"]))
                row[key] = obj
                S["%s|n%d|%s" % (v, n, key)] = {"acc": obj["acc"], "mean": m, "ci": ci}
            cells[(v, n)] = row
        P("")
    # ---- bars 1 and 2
    P("## Bars 1 (std) and 2 (ho): naming head against the strongest equalised baseline at each n")
    lead = {}
    for v in ("std", "ho"):
        for n in SIZES:
            row = cells.get((v, n), {})
            if "name" not in row:
                P("  %s n=%-6d naming head not available" % (v, n)); continue
            base = [k for k in ("lora", "scratch", "probe") if k in row]
            if not base:
                P("  %s n=%-6d no baseline available" % (v, n)); continue
            bk = max(base, key=lambda k: np.mean(row[k]["acc"]))
            M, ci = diff_ci(row["name"]["C"], row[bk]["C"])
            ms = [a_ - b_ for a_, b_ in zip(row["name"]["acc"], row[bk]["acc"])]
            verdict = "STANDS" if M >= 0.10 and all(x >= 0.10 for x in ms) else ("CLOSED" if M <= 0.03 else "NARROWED")
            if not {"lora", "scratch"} <= set(base) and verdict != "CLOSED":
                verdict = "INCOMPLETE (no fine-tune cell finished; not evaluable)"
            lead[(v, n)] = verdict
            P("  %s n=%-6d naming head %.3f vs %s %.3f: lead %+.3f [%+.3f, %+.3f], per seed %s -> %s%s" % (
                v, n, np.mean(row["name"]["acc"]), bk, np.mean(row[bk]["acc"]), M, ci[0], ci[1],
                "/".join("%+.3f" % x for x in ms), verdict, "" if set(base) == {"lora", "scratch", "probe"} else
                "  (baselines available: %s)" % ",".join(base)))
            S["lead|%s|n%d" % (v, n)] = {"vs": bk, "M": M, "ci": ci, "per_seed": ms, "verdict": verdict}
    for v in ("std", "ho"):
        n800 = cells.get((v, 800), {}).get("name")
        if n800:
            target = np.mean(n800["acc"]) - 0.03
            nstar = [n for n in SIZES if any(k in cells.get((v, n), {}) and np.mean(cells[(v, n)][k]["acc"]) >= target
                                             for k in ("lora", "scratch"))]
            P("  %s: n* (smallest n with lora or scratch >= N(800) - 0.03 = %.3f): %s" % (v, target, nstar[0] if nstar else "none of the sizes run"))
    P("")
    P("## Held-out drop: arm(std, n) - arm(ho, n) (bootstrap over the two test sets, independent)")
    for n in SIZES:
        for k in ("name", "lora", "scratch", "probe"):
            a_, b_ = cells.get(("std", n), {}).get(k), cells.get(("ho", n), {}).get(k)
            if a_ and b_:
                d = np.mean(a_["acc"]) - np.mean(b_["acc"])
                I1, I2 = boot_idx(500), RNG.integers(0, 500, size=(B, 500))
                bs = np.asarray(a_["C"], float)[:, I1].mean(axis=(0, 2)) - np.asarray(b_["C"], float)[:, I2].mean(axis=(0, 2))
                P("  n=%-6d %-8s std %.3f  ho %.3f  drop %+.3f [%+.3f, %+.3f]" % (n, k, np.mean(a_["acc"]), np.mean(b_["acc"]), d,
                                                                            np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
                S["drop|%s|n%d" % (k, n)] = d
    P("")
    P("## Diagnostic: what the equalised LoRA learned (test accuracy by label and by whether the candidate is one of")
    P("## the 32 table-F (level-1) fillers; a shortcut 'yes iff the candidate is an F value' scores 1/1/0 on +F/-F-free/-F)")
    import b44_indep as I_
    import b44_data as BD_
    for v in ("std", "ho"):
        st_ = BD_.split(v, "test")
        inl1 = np.array([s_["mentions"][s_["cand"]][2] in [b for _, b in s_["levels"][1]] for s_ in st_])
        y_ = LAB[v]
        for n in SIZES:
            r = cells.get((v, n), {}).get("lora")
            if not r:
                continue
            for seed, c in zip(SEEDS, r["C"]):
                c = np.asarray(c)
                parts = []
                for lab_, f_, nm_ in ((1, True, "positive"), (0, False, "negative, cand not F"), (0, True, "negative, cand F")):
                    m_ = (y_ == lab_) & (inl1 == f_)
                    parts.append("%s %.3f (n %d)" % (nm_, c[m_].mean() if m_.any() else float("nan"), m_.sum()))
                P("  %s n=%-6d seed %-3d %s" % (v, n, seed, "; ".join(parts)))
    P("")
    P("## Bar 3: scratchpad - equalised LoRA (paired by seed and instance)")
    for v in ("std", "ho"):
        for n in SIZES:
            r = cells.get((v, n), {})
            if "scratch" in r and "lora" in r:
                d, ci = diff_ci(r["scratch"]["C"], r["lora"]["C"])
                P("  %s n=%-6d scratch %.3f lora %.3f: %+.3f [%+.3f, %+.3f] -> %s" % (
                    v, n, np.mean(r["scratch"]["acc"]), np.mean(r["lora"]["acc"]), d, ci[0], ci[1],
                    "HELPS" if d >= 0.05 and ci[0] > 0 else "NO CLEAR HELP"))
                S["scratch-lora|%s|n%d" % (v, n)] = {"d": d, "ci": ci}
    P("")
    # ---- CoT
    P("## Bar 4: CoT-prompted Qwen3-8B (no training)")
    cot = {}
    for f in sorted(glob.glob(os.path.join(a.root, "cot", "*.json"))):
        d = json.load(open(f)); ar = d["args"]
        if ar.get("n"):
            continue
        y = LAB[ar["variant"]]
        c = np.array([int(r["pred"] == yy) for r, yy in zip(d["rows"], y)])
        assert abs(c.mean() - d["acc"]) < 1e-6
        cot[(ar["arm"], ar["variant"])] = (c, d)
        m, ci = mean_ci([c])
        P("  %-11s %-3s test acc %.3f [%.3f-%.3f] (Wilson %.3f-%.3f); unparsed %.3f; hit token limit %.3f; mean new tokens %.0f" % (
            ar["arm"], ar["variant"], m, ci[0], ci[1], *wilson(int(c.sum()), len(c)), d["unparsed"], d["hit_limit"],
            d["mean_new_tokens"]))
        S["cot|%s|%s" % (ar["arm"], ar["variant"])] = {"acc": m, "ci": ci, "unparsed": d["unparsed"], "hit_limit": d["hit_limit"]}
    if "cot" in sel and (sel["cot"]["selected"], "std") in cot:
        cs = sel["cot"]["selected"]; c = cot[(cs, "std")][0]
        n800 = cells.get(("std", 800), {})
        if "name" in n800:
            Nm = np.mean(n800["name"]["acc"])
            d, ci = diff_ci([c, c, c], n800["name"]["C"])
            verdict = "MATCHES" if c.mean() >= Nm - 0.03 else ("FALLS SHORT" if c.mean() <= Nm - 0.10 else "IN BETWEEN")
            P("  selected %s: %.3f vs naming head (n=800) %.3f: %+.3f [%+.3f, %+.3f] -> %s" % (cs, c.mean(), Nm, d, ci[0], ci[1], verdict))
            S["cot_bar"] = {"selected": cs, "d": d, "ci": ci, "verdict": verdict}
        if "lora" in n800:
            d, ci = diff_ci([c, c, c], n800["lora"]["C"])
            P("  selected %s vs equalised LoRA (n=800) %.3f: %+.3f [%+.3f, %+.3f]" % (cs, np.mean(n800["lora"]["acc"]), d, ci[0], ci[1]))
    P("")
    # ---- sweep
    sweep_section(a, S)
    # ---- B42 quote
    P("## Trained binding-loss map, centred / whitened: B42 (bind+pc, bind+pz, cw), quoted verbatim, not rerun")
    b42s = [p_ for p_ in (os.path.join(HERE, "results", "b42_verdict.txt"), os.path.join(HERE, "results", "b42_verdict_C.txt"))
            if os.path.exists(p_)]
    if b42s:
        lines = open(b42s[-1]).read().splitlines()
        P("  source: %s (%s)" % (os.path.relpath(b42s[-1], HERE), lines[1].strip("# ") if len(lines) > 1 else ""))
        keep = False
        for l in lines:
            head = re.match(r"\s{4}(Phi-2|Qwen3-8B)\s", l)
            if head:
                keep = bool(re.search(r"post-hoc centring|post-hoc ZCA|centred / whitened \(trained\)|B31 binding-loss map \(perm", l))
            elif not l.startswith("              "):
                keep = False
            if keep or re.search(r"(post-hoc|centred / whitened).*->|ANSWER", l):
                P("  B42| " + l.rstrip())
    else:
        P("  B42 verdict not present at verdict time")
    P("")
    fig2(a, S)
    predictions(a, S, cells)
    json.dump(S, open(a.scores, "w"), indent=1, default=float)
    open(a.out, "w").write("\n".join(L) + "\n")


def sweep_section(a, S):
    p = os.path.join(HERE, "results", "b44_xtalk_sweep.json")
    P("## Bar 5: imposed cross-talk sweep (exact rows, B31 arena C; threshold refitted per level; 3 codebook seeds)")
    if not os.path.exists(p):
        P("  sweep not run"); return
    d = json.load(open(p))
    keys = ["raw/perm", "centred/perm", "whitened_a0.1/perm", "whitened_a0.5/perm", "raw/comm", "centred/comm",
            "whitened_a0.1/comm"]
    P("  accuracy (mean of 3 seeds) / AUROC; measured cross-talk of the bound codes in brackets")
    P("  %-5s " % "c" + " ".join("%-24s" % k for k in keys))
    for r in d["curve"]:
        P("  %.2f  " % r["c"] + " ".join("%-24s" % ("%.3f/%.3f (%+.3f)" % (r[k]["acc_mean"], r[k]["auroc_mean"], r[k]["crosstalk_mean"])) for k in keys))
    c0 = {k: d["curve"][0][k]["acc"] for k in keys}
    P("  G1 (c = 0, raw): perm %s, comm %s (B31: 1.000 x3; 0.932/0.932/0.938)" % (c0["raw/perm"], c0["raw/comm"]))
    rem = all(abs(r[k]["acc"][i] - c0["raw/perm"][i]) <= 0.01 for r in d["curve"] for k in ("centred/perm", "whitened_a0.1/perm")
              for i in range(3))
    P("  REMOVABLE (centred and whitened a=0.1 perm within 0.01 of c=0 raw at every c, 3/3 seeds): %s" % rem)
    S["sweep_removable"] = rem
    for floor_name, floor in (("Phi-2 probe 0.472", 0.472), ("Qwen3-8B probe 0.512", 0.512)):
        for k in ("raw/perm", "raw/comm"):
            ok = [r[k]["crosstalk_mean"] for r in d["curve"] if r[k]["acc_mean"] >= floor + 0.10]
            tol = max(ok) if ok else None
            P("  tolerance %s, floor %s + 0.10: largest measured cross-talk clearing it %s" % (k, floor_name, f3(tol)))
            S["tolerance|%s|%s" % (k, floor)] = tol
    ip = os.path.join(a.root, "xtalk_init.json")
    if os.path.exists(ip):
        P("  binding-loss map at initialisation (B31 seeds; perm/named; test accuracy with the training-fitted threshold):")
        I = json.load(open(ip))
        for k, v in sorted(I.items()):
            P("    %-14s raw: acc %.3f, different/same %.3f/%.3f | centred: acc %.3f, different/same %.3f/%.3f" % (
                k, v["raw"]["acc"], v["raw"]["geom"]["different"], v["raw"]["geom"]["same"], v["centred"]["acc"],
                v["centred"]["geom"]["different"], v["centred"]["geom"]["same"]))
        S["init"] = I
    P("")


def predictions(a, S, cells):
    P("## Predictions (registered)")
    m = lambda v, n, k: float(np.mean(cells[(v, n)][k]["acc"])) if k in cells.get((v, n), {}) else None
    def show(tag, txt, val):
        P("  %-4s %-100s %s" % (tag, txt, "not evaluable (cells missing)" if val is None else ("HELD" if val else "NOT HELD")))
    sd = json.load(open(os.path.join(a.root, "selftest_data.json"))) if os.path.exists(os.path.join(a.root, "selftest_data.json")) else {}
    sw = json.load(open(os.path.join(HERE, "results", "b44_xtalk_sweep.json")))["curve"][0] if os.path.exists(os.path.join(HERE, "results", "b44_xtalk_sweep.json")) else None
    g1 = None if sw is None else (sw["raw/perm"]["acc"] == [1.0, 1.0, 1.0] and sw["raw/comm"]["acc"] == [0.932, 0.932, 0.938])
    show("P1", "data self-test (B31 texts reproduced) and gate G1 pass",
         None if not sd or g1 is None else (sd.get("b31_C_test_identical") == 500 and sd.get("b31_C_train_identical") == 800 and g1))
    show("P2", "dictionary = 1.000 on std and ho test", bool(sd) and sd.get("std_test_dict_equals_label") == 500 and sd.get("ho_test_dict_equals_label") == 500)
    ex = [cells.get((v, 800), {}).get("exact") for v in ("std", "ho")]
    show("P3", "exact + VSA (perm) = 1.000 on std test and >= 0.99 on ho test, every seed",
         None if None in ex else (min(ex[0]["acc"]) == 1.0 and min(ex[1]["acc"]) >= 0.99))
    l = [m("std", n, "lora") for n in SIZES]
    show("P4", "lora std mean: 800 <= 0.75; 8,000 >= 0.85; 80,000 >= 0.95", None if None in l else (l[0] <= 0.75 and l[1] >= 0.85 and l[2] >= 0.95))
    l5 = S.get("lead|std|n80000", {}).get("verdict", "INCOMPLETE")
    show("P5", "(bar 1) lead CLOSED at 80,000 on std", None if l5.startswith("INCOMPLETE") else l5 == "CLOSED")
    lh, ls = m("ho", 80000, "lora"), m("std", 80000, "lora")
    show("P6", "lora ho at 80,000 >= 0.90 and within 0.05 of lora std at 80,000", None if None in (lh, ls) else (lh >= 0.90 and abs(lh - ls) <= 0.05))
    sc = [S.get("scratch-lora|std|n%d" % n) for n in (800, 8000)]
    s800 = m("std", 800, "scratch")
    show("P7", "scratch >= lora + 0.05 with CI > 0 at 800 and 8,000 (std), and scratch std 800 >= 0.80",
         None if None in sc or s800 is None else (all(x["d"] >= 0.05 and x["ci"][0] > 0 for x in sc) and s800 >= 0.80))
    nm = [m(v, n, "name") for v in ("std", "ho") for n in SIZES]
    show("P8", "naming head >= 0.97 at every n on std and ho; varies by <= 0.02 across n",
         None if None in nm else (min(nm) >= 0.97 and max(nm[:3]) - min(nm[:3]) <= 0.02 and max(nm[3:]) - min(nm[3:]) <= 0.02))
    c = {k: S.get("cot|%s|std" % k, {}).get("acc") for k in ("zs-nothink", "zs-think", "fs-think")}
    sel = S.get("cot_bar", {}).get("selected")
    show("P9", "CoT: zs-nothink <= 0.65; best thinking arm >= 0.80; the dev-selected arm is a thinking arm",
         None if None in c.values() or sel is None else (c["zs-nothink"] <= 0.65 and max(c["zs-think"], c["fs-think"]) >= 0.80 and "think" in sel and "nothink" not in sel))
    tol = S.get("tolerance|raw/perm|0.472")
    show("P10", "sweep REMOVABLE; raw perm tolerance (Phi-2 floor) between 0.15 and 0.30",
         None if tol is None else (S.get("sweep_removable") and 0.15 <= tol <= 0.30))
    I = S.get("init")
    show("P11", "binding-loss map at init: raw different >= 0.60 both encoders; centred <= 0.10; centred accuracy <= 0.70",
         None if not I else all(v["raw"]["geom"]["different"] >= 0.60 and v["centred"]["geom"]["different"] <= 0.10 and v["centred"]["acc"] <= 0.70 for v in I.values()))
    kc = json.load(open(a.fig))["c"] if os.path.exists(a.fig) else {}
    ok = None
    if kc.get("phi2") and kc.get("qwen8") and all("exact" in r and "name" in r for e in ("phi2", "qwen8") for r in kc[e]):
        ok = all(min(r["exact"]) == 1.0 for e in ("phi2", "qwen8") for r in kc[e]) and \
            all(np.mean(r["name"]) >= 0.98 for r in kc["phi2"]) and all(np.mean(r["name"]) >= 0.95 for r in kc["qwen8"])
    show("P12", "Fig. 2c (perm): exact = 1.000 at every k; Phi-2 naming head >= 0.98, Qwen3-8B >= 0.95 at every k", ok)
    ip = os.path.join(a.root, "indep.json")
    agree = None
    if os.path.exists(ip):
        I2 = json.load(open(ip))["cells"]
        vals = []
        for k, v in I2.items():
            for x in (v if isinstance(v, list) else [v]):
                # per-instance agreement where two parsers exist (scratch, CoT); elsewhere the same decision rule on the
                # saved scores, so agreement = accuracy equality against the independently derived labels (float32
                # logging: tolerance 1e-6)
                vals.append(min(x.get("parser_agreement", 1.0), 1.0 if abs(x["indep_acc"] - x["main_acc"]) < 1e-6 else 0.0))
        agree = min(vals) >= 0.99 if vals else None
    show("P13", "second scorer agrees with the main runs on >= 0.99 of instances in every cell", agree)
    P("  P14  total GPU-hours <= 80: see the compute ledger")
    P("")


def b31_lines(enc_label, arena="C"):
    """Verbatim result lines of results/b31_verdict.txt for an encoder in arena C."""
    lines = open(os.path.join(HERE, "results", "b31_verdict.txt")).read().splitlines()
    i0 = [i for i, l in enumerate(lines) if l.strip().startswith("arena %s (n =" % arena)][0]
    out = {}
    i = i0 + 1
    while i < len(lines) and not lines[i].strip().startswith("arena ") and not lines[i].startswith("##"):
        l = lines[i]
        m = re.match(r"\s+(\S+(?: \S+)?)\s{2,}(.+?)\s+mean ([0-9.]+) \[.*seeds (.*)$", l)
        if m and m.group(1).startswith(enc_label):
            arm = m.group(2).strip()
            rec = {"line": l, "mean": float(m.group(3)), "seeds": [float(x.split()[0]) for x in m.group(4).split(", ")]}
            j = i + 1
            while j < len(lines) and lines[j].startswith("              "):
                if "cross-talk different/same" in lines[j]:
                    rec["geom_line"] = lines[j]
                    pr = re.findall(r"(-?[0-9.]+)/(-?[0-9.]+)", lines[j].split("same", 1)[1])
                    rec["different"] = [float(x) for x, _ in pr]; rec["same"] = [float(y) for _, y in pr]
                j += 1
            out[arm] = rec
        i += 1
    return out


def fig2(a, S):
    P("## Fig. 2b,c data under the B31 protocol (results/b44_fig2bc.json; every number with its source line)")
    fig = {"b": {}, "c": {}}
    sp = os.path.join(HERE, "results", "b44_xtalk_sweep.json")
    if os.path.exists(sp):
        d = json.load(open(sp))
        for enc in ("perm", "comm"):
            for form in ("raw", "centred", "whitened_a0.1"):
                k = "%s/%s" % (form, enc)
                fig["b"]["sweep_" + k] = {"c": [r["c"] for r in d["curve"]], "crosstalk_mean": [r[k]["crosstalk_mean"] for r in d["curve"]],
                                          "acc_mean": [r[k]["acc_mean"] for r in d["curve"]], "acc_per_seed": [r[k]["acc"] for r in d["curve"]],
                                          "auroc_mean": [r[k]["auroc_mean"] for r in d["curve"]],
                                          "source": "results/b44_xtalk_sweep.json curve[]['%s']" % k}
        fig["b"]["floor_phi2"] = {"value": 0.472, "source": "results/b31_verdict.txt",
                                  "line": [l for l in open(os.path.join(HERE, "results", "b31_verdict.txt")).read().splitlines()
                                           if "Phi-2     equalised probe" in l][2]}
        fig["b"]["tolerance_perm_phi2_floor"] = S.get("tolerance|raw/perm|0.472")
        fig["b"]["tolerance_comm_phi2_floor"] = S.get("tolerance|raw/comm|0.472")
    for enc_label, enc in (("Phi-2", "phi2"), ("Qwen3-8B", "qwen8")):
        bl = b31_lines(enc_label)
        pts = {}
        for arm_txt, key in (("naming head + VSA (perm, named)", "naming_head"),
                             ("naming head read without snapping (perm)", "no_snapping"),
                             ("binding-loss map (perm, named)", "binding_loss_trained"),
                             ("binding-loss map, published form (comm, index)", "binding_loss_published_trained"),
                             ("label-free InfoNCE map (perm, named)", "infonce"),
                             ("equalised probe", "probe"), ("equalised LoRA", "lora")):
            r = bl.get(arm_txt)
            if r:
                pts[key] = {"acc_per_seed": r["seeds"], "acc_mean": r["mean"], "different_per_seed": r.get("different"),
                            "same_per_seed": r.get("same"), "source": "results/b31_verdict.txt",
                            "line": r["line"], "geom_line": r.get("geom_line")}
        ip = os.path.join(a.root, "xtalk_init.json")
        if os.path.exists(ip):
            I = json.load(open(ip))
            for form in ("raw", "centred"):
                v = [I["%s_seed%d" % (enc, s)][form] for s in SEEDS]
                pts["binding_loss_init_%s" % form] = {"acc_per_seed": [x["acc"] for x in v],
                                                     "different_per_seed": [x["geom"]["different"] for x in v],
                                                     "same_per_seed": [x["geom"]["same"] for x in v],
                                                     "source": "b44_out/xtalk_init.json %s_seed<seed>.%s" % (enc, form)}
        fig["b"]["points_" + enc] = pts
    for enc in ("phi2", "qwen8"):
        rows = []
        for k in (2, 4, 8, 16, 24, 32):
            r = {"k": k}
            for arm in ("exact", "name", "probe"):
                f = os.path.join(a.root, "k", "%s_k%d_%s.json" % (enc, k, arm))
                if not os.path.exists(f):
                    continue
                d = json.load(open(f))
                if arm == "probe":
                    r["probe"] = [x["test_acc"] for x in d["runs"]]
                else:
                    accs, comm, ep = [], [], []
                    for run in d["runs"]:
                        devs = [e["perm/named:dev"]["acc"] for e in run["epochs"]]
                        i = max(range(len(devs)), key=lambda j: (devs[j], -j))
                        accs.append(run["epochs"][i]["perm/named:test"]["acc"]); ep.append(run["epochs"][i]["epoch"])
                        if "comm/named:test" in run["epochs"][i]:
                            comm.append(run["epochs"][i]["comm/named:test"]["acc"])
                    r[arm] = accs
                    if comm:
                        r[arm + "_comm"] = comm
                    if arm == "name":
                        r["name_dev_epoch"] = ep
                r["source_" + arm] = "b44_out/k/%s_k%d_%s.json" % (enc, k, arm)
            rows.append(r)
            P("  2c %-5s k=%-2d exact %s | naming head %s | probe %s" % (enc, k, "/".join(f3(x) for x in r.get("exact", [])),
                                                                         "/".join(f3(x) for x in r.get("name", [])),
                                                                         "/".join(f3(x) for x in r.get("probe", []))))
        fig["c"][enc] = rows
    json.dump(fig, open(a.fig, "w"), indent=1)
    P("")


if __name__ == "__main__":
    main()
