"""B47 (2): hierarchical seed-by-question intervals for every arena margin (prereg/B47_phi2_finetunes_hierboot_
consistency_prereg.md). Written after the pre-registration (a1bf128) and before any B47 test output is read.

Per-instance test correctness is recomputed from the saved outputs exactly as each verdict does:
  B31 / B42 memory arms   saved final cosine >= saved training threshold, at each seed's dev-selected epoch (ties to
                          the earlier epoch), for the arm's reading
  B31 probe / LoRA        sign of the saved margin (> 0)
  B44 lora / name / probe as b44_verdict (lora: margin > 0; scratch: last yes/no of the generation; CoT: saved pred)
  B47 fine-tunes          margin > 0 (gen, fullgen), arg-max (fullcls), last yes/no of the generation (scratch)
Labels: b31_struct.build (B31 / B42 arenas A, B, C) and b44_data (B44 / B47). Every recomputed per-seed accuracy is
compared with the logged one.

Intervals (10,000 replicates, 95% percentile), for a difference X - Y of two arms scored on the same questions:
  hnest   3 seed slots drawn with replacement for each arm independently; for every slot one draw of the n questions
          (shared by X and Y in that slot); statistic = mean over slots of (mean_q X - mean_q Y)
  hcross  as hnest with ONE question draw per replicate shared by all slots (questions crossed with seeds)
  tseed   Student-t (2 df) over the seed-level differences of seed-sorted pairs
  inst    B31's instance-only bootstrap (seeds fixed and paired, one shared question draw; rng seed 1)
Recovery R = (A - F) / (N - F) (F probe, N naming head): the same draws, the ratio of the slot-averaged quantities;
tseed over per-seed R_s = (A_s - mean F) / (mean N - mean F).

  python b47_boot.py --out b47_out/b47_hierboot.json
"""
import argparse, glob, gzip, hashlib, importlib.machinery, json, math, os, re, sys, types
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import peft                                                            # noqa: F401
except ImportError:                    # the arena generators import peft at module level but never use it for labels
    _m = types.ModuleType("peft"); _m.__spec__ = importlib.machinery.ModuleSpec("peft", None)
    _m.LoraConfig = _m.get_peft_model = _m.TaskType = None; sys.modules["peft"] = _m

NB = 10000
SEEDS = [42, 123, 456]
T975_2DF = 4.302652729911275
FILES = {}


def md5(p):
    h = hashlib.md5(); h.update(open(p, "rb").read()); return h.hexdigest()


def load(p):
    for q in (p, p + ".gz"):
        if os.path.exists(q):
            FILES[os.path.relpath(q, HERE)] = md5(q)
            return json.load(gzip.open(q, "rt")) if q.endswith(".gz") else json.load(open(q))
    return None


# ------------------------------------------------------------------ labels

_LAB = {}


def lab31(ar):
    if ar not in _LAB:
        import b31_struct as S
        _LAB[ar] = np.array([s["label"] for s in S.build(ar, "test")])
    return _LAB[ar]


def lab44(variant):
    k = "b44_" + variant
    if k not in _LAB:
        import b44_data as BD
        _LAB[k] = np.array([BD.dict_answer(s["text"])["label"] for s in BD.split(variant, "test", struct=False)])
    return _LAB[k]


# ------------------------------------------------------------------ cells: dict(C=(S, n) 0/1, acc, logged, seeds)

def cell(C, logged, seeds, src, **kw):
    C = np.asarray(C, dtype=np.int8)
    acc = [float(c.mean()) for c in C]
    ok = all(abs(a - b) < 1e-6 for a, b in zip(acc, logged)) if logged else None   # logged values may be float32
    return dict(C=C, acc=acc, logged=logged, seeds=seeds, src=src, recomputed_equals_logged=ok, **kw)


def sel_epoch(run, reading):
    accs = [e["%s:dev" % reading]["acc"] for e in run["epochs"]]
    return max(range(len(accs)), key=lambda j: (accs[j], -j))


def mem(path, reading, y):
    d = load(path)
    if d is None:
        return None
    C, lg, sd, ep = [], [], [], []
    for run in sorted(d["runs"], key=lambda r: SEEDS.index(r["seed"])):
        j = sel_epoch(run, reading)
        e = run["epochs"][j]["%s:test" % reading]
        C.append(((np.array(e["scores"]) >= e["thr"]).astype(int) == y).astype(int))
        lg.append(e["acc"]); sd.append(run["seed"]); ep.append(run["epochs"][j]["epoch"])
    return cell(C, lg, sd, os.path.relpath(path, HERE), epoch=ep)


def b31_probe(R, enc, ar):
    d = load(os.path.join(R, "probe_final", "%s_%s.json" % (enc, ar)))
    y = lab31(ar)
    C = [(((np.array(r["test_scores"]) > 0).astype(int)) == y).astype(int) for r in d["runs"]]
    return cell(C, [r["test_acc"] for r in d["runs"]], [r["seed"] for r in d["runs"]],
                "b31 probe_final/%s_%s.json" % (enc, ar))


def b31_lora(R, enc, ar, idl):
    y = lab31(ar); C, lg = [], []
    for s in SEEDS:
        d = load(os.path.join(R, "lora_final", "%s_%s_id%d_s%d.json" % (enc, ar, idl, s)))
        r = d["runs"][0]
        C.append((((np.array(r["test_scores"]) > 0).astype(int)) == y).astype(int)); lg.append(r["test"]["acc"])
    return cell(C, lg, SEEDS, "b31 lora_final/%s_%s_id%d_s*.json" % (enc, ar, idl))


# ------------------------------------------------------------------ B44 cells (as b44_verdict)

def b44_lora(root, task, variant, n, sel):
    s = sel.get("%s_%s_n%d" % ("lora" if task == "cls" else "scratch", variant, n))
    if not s:
        return None
    y = lab44(variant); C, lg = [], []
    import b44_lora as BL
    for seed in SEEDS:
        base = os.path.join(root, "lora", "%s_%s_n%d_r8_lr%s_s%d.json" % (task, variant, n,
                                                                         {1e-4: "1e-4", 3e-4: "3e-4"}[s["lr"]], seed))
        t = load(base + ".test.json")
        if t is None or not os.path.exists(base):
            return None
        if task == "cls":
            pred = np.array([int(x > 0) for x in t["test"]["scores"]])
        else:
            pred = np.array([-1 if BL.parse_scratch(g)[0] is None else BL.parse_scratch(g)[0] for g in t["test"]["gens"]])
        C.append((pred == y).astype(int)); lg.append(t["test"]["acc"])
    return cell(C, lg, SEEDS, "b44 %s_%s_n%d lr %g" % (task, variant, n, s["lr"]))


def b44_mem(root, arm, variant, n):
    pat = re.compile(r"^qwen8_%s_%s_n%d(_s\d+)?\.json$" % (arm, variant, n))
    fs = sorted(f for f in glob.glob(os.path.join(root, "name", "*.json")) if pat.match(os.path.basename(f)))
    if not fs:
        return None
    runs = []
    for f in fs:
        runs += load(f)["runs"]
    runs = sorted(runs, key=lambda r: SEEDS.index(r["seed"]))
    if len(runs) != 3:
        return None
    y = lab44(variant); C, lg = [], []
    for r in runs:
        if arm == "probe":
            C.append((np.array([int(x > 0) for x in r["test_scores"]]) == y).astype(int)); lg.append(r["test_acc"])
            continue
        j = sel_epoch(r, "perm/named")
        te = r["epochs"][j]["perm/named:test"]
        C.append((np.array([int(x >= te["thr"]) for x in te["scores"]]) == y).astype(int)); lg.append(te["acc"])
    return cell(C, lg, [r["seed"] for r in runs], "b44 name/qwen8_%s_%s_n%d" % (arm, variant, n))


def b44_cot(root, arm, variant):
    d = load(os.path.join(root, "cot", "%s_%s_test.json" % (arm, variant)))
    if d is None:
        return None
    y = lab44(variant)
    C = [np.array([int(r["pred"] == yy) for r, yy in zip(d["rows"], y)])]
    return cell(C, [d["acc"]], [None], "b44 cot/%s_%s_test.json" % (arm, variant))


# ------------------------------------------------------------------ B47 cells

def b47_ft(root, task, variant, n, sel):
    s = sel.get("%s_%s_n%d" % (task, variant, n))
    if not s:
        return None
    y = lab44("std")                       # k32 test = std test = B31's arena-C test (checked in b47_tok)
    C, lg = [], []
    for seed in SEEDS:
        base = os.path.join(root, "%s_%s_n%d_r%d_lr%s_s%d.json" % (task, variant, n, s["rank"], "%g" % s["lr"], seed))
        t = load(base + ".test.json")
        if t is None:
            return None
        if task in ("gen", "fullgen", "fullcls"):
            pred = np.array([int(x > 0) for x in t["test"]["scores"]])
        else:
            import b44_lora as BL
            pred = np.array([-1 if BL.parse_scratch(g)[0] is None else BL.parse_scratch(g)[0] for g in t["test"]["gens"]])
        C.append((pred == y).astype(int)); lg.append(t["test"]["acc"])
    return cell(C, lg, SEEDS, "b47 %s_%s_n%d r%d lr %g" % (task, variant, n, s["rank"], s["lr"]))


# ------------------------------------------------------------------ bootstrap

def draws(cells, mode, rng_seed, B=NB, chunk=1000):
    """Per replicate, the slot-averaged accuracy of every cell; the question draws are shared by the cells.
    hnest: one question draw per slot (3 per replicate); hcross: one per replicate; inst: seeds fixed (all of them,
    paired), one question draw per replicate."""
    rng = np.random.default_rng(rng_seed)
    n = cells[0]["C"].shape[1]
    out = [[] for _ in cells]
    for b0 in range(0, B, chunk):
        b = min(chunk, B - b0)
        if mode == "inst":
            idx = rng.integers(0, n, size=(b, n))
            for i, c in enumerate(cells):
                out[i].append(c["C"][:, idx].mean(axis=2).mean(axis=0))
            continue
        idx = rng.integers(0, n, size=(b, 3, n)) if mode == "hnest" else rng.integers(0, n, size=(b, 1, n))
        for i, c in enumerate(cells):
            sl = rng.integers(0, c["C"].shape[0], size=(b, 3))
            out[i].append(c["C"][sl[:, :, None], idx].mean(axis=2).mean(axis=1))
    return [np.concatenate(o) for o in out]


def pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def tseed(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m = v.mean(); h = T975_2DF * v.std(ddof=1) / math.sqrt(len(v)) if len(v) == 3 else None
    return None if h is None else [float(m - h), float(m + h)]


def margin(X, Y, mid, src, rng_seed):
    pt = float(np.mean(X["acc"]) - np.mean(Y["acc"]))
    rec = {"id": mid, "source": src, "x": X["src"], "y": Y["src"], "point": pt,
           "acc_x": X["acc"], "acc_y": Y["acc"], "logged_ok": [X["recomputed_equals_logged"], Y["recomputed_equals_logged"]]}
    ci = {}
    for mode, off in (("hnest", 0), ("hcross", 1), ("inst", 2)):
        if mode == "inst":
            if len(X["acc"]) != len(Y["acc"]):
                continue
            rng = np.random.default_rng(1)                                  # B31's boot_diff
            n = X["C"].shape[1]
            idx = rng.integers(0, n, size=(NB, n))
            d = X["C"][:, idx].mean(axis=2).mean(axis=0) - Y["C"][:, idx].mean(axis=2).mean(axis=0)
            ci[mode] = pct(d)
            continue
        dx, dy = draws([X, Y], mode, rng_seed + off)
        ci[mode] = pct(dx - dy)
    sx = X["acc"] if len(X["acc"]) == 3 else X["acc"] * 3
    sy = Y["acc"] if len(Y["acc"]) == 3 else Y["acc"] * 3
    rec["seed_diffs"] = [a - b for a, b in zip(sx, sy)]
    ci["tseed"] = tseed(rec["seed_diffs"])
    rec["ci"] = ci
    ex = {k: (v[0] > 0 or v[1] < 0) for k, v in ci.items() if v}
    rec["excludes0"] = ex
    rec["excludes0_registered"] = bool(ex.get("hnest") and ex.get("hcross"))
    rec["flag"] = ("hnest/hcross disagree" if ex.get("hnest") != ex.get("hcross") else "") + \
                  ("; t-seed interval includes 0" if ci["tseed"] and not ex.get("tseed") and rec["excludes0_registered"]
                   else "")
    return rec


def recovery(A, Fc, Nc, mid, src, rng_seed):
    Fm, Nm, Am = np.mean(Fc["acc"]), np.mean(Nc["acc"]), np.mean(A["acc"])
    gap = Nm - Fm
    rec = {"id": mid, "source": src, "arm": A["src"], "floor": Fc["src"], "naming": Nc["src"], "acc": A["acc"],
           "F": float(Fm), "N": float(Nm), "gap": float(gap), "logged_ok": A["recomputed_equals_logged"]}
    if gap < 0.10:
        rec["point"] = None; rec["note"] = "gap < 0.10: recovery undefined (registered)"; return rec
    rec["point"] = float((Am - Fm) / gap)
    rec["per_seed"] = [float((a - Fm) / gap) for a in A["acc"]]
    ci = {}
    for mode, off in (("hnest", 0), ("hcross", 1), ("inst", 2)):
        a, f, nn_ = draws([A, Fc, Nc], mode, rng_seed + off)
        ci[mode] = pct((a - f) / (nn_ - f))
    ci["tseed"] = tseed(rec["per_seed"])
    rec["ci"] = ci
    return rec


def acc_ci(X, rng_seed):
    return {"mean": float(np.mean(X["acc"])), "per_seed": X["acc"], "src": X["src"],
            "hnest": pct(draws([X], "hnest", rng_seed)[0]), "hcross": pct(draws([X], "hcross", rng_seed + 1)[0]),
            "tseed": tseed(X["acc"]) if len(X["acc"]) == 3 else None,
            "recomputed_equals_logged": X["recomputed_equals_logged"]}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b31", default=os.path.join(HERE, "b47_out", "src_b31"), help="copy of /path/to/work/b31/out")
    ap.add_argument("--b42", default=os.path.join(HERE, "b42_out", "final"))
    ap.add_argument("--b44", default=os.path.join(HERE, "b44_out"))
    ap.add_argument("--b47", default=os.path.join(HERE, "b47_out", "ft"))
    ap.add_argument("--b47_select", default=os.path.join(HERE, "b47_out", "select.json"))
    ap.add_argument("--only", default="", help="comma list of groups: b31,ex1,rec,b44,b47 (default all)")
    ap.add_argument("--update", action="store_true", help="with --only: replace that group's entries in an existing --out")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups = set(a.only.split(",")) if a.only else {"b31", "ex1", "rec", "b44", "b47"}
    R = a.b31
    out = {"generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "script": "b47_boot.py",
           "prereg": "prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md", "replicates": NB,
           "methods": {"hnest": "seeds resampled per arm (3 slots), questions resampled within each slot (shared by the "
                                "two arms of a margin)", "hcross": "seeds resampled per arm, ONE question resample per "
                                "replicate shared by all slots and arms", "tseed": "Student-t, 2 df, over seed-level "
                                "differences (seed-sorted pairs)", "inst": "B31's instance-only bootstrap (rng seed 1)"},
           "margins": [], "recoveries": [], "accuracies": {}, "notes": []}
    cache = {}
    rs = [1000]

    def nxt():
        rs[0] += 10; return rs[0]

    def get(key, fn):
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    # ---------------- B31 (b), (c)
    for ar in "ABC":
        for enc in ("phi2", "qwen8"):
            y = lab31(ar)
            N = get(("name", enc, ar), lambda: mem(os.path.join(R, "arms_final", "%s_%s_name.json" % (enc, ar)), "perm/named", y))
            Pr = get(("probe", enc, ar), lambda: b31_probe(R, enc, ar))
            Lo = get(("lora", enc, ar), lambda: b31_lora(R, enc, ar, 0))
            Li = get(("loraid", enc, ar), lambda: b31_lora(R, enc, ar, 1))
            for k, c in (("name", N), ("probe", Pr), ("lora", Lo), ("loraid", Li)):
                out["accuracies"]["b31|%s|%s|%s" % (enc, ar, k)] = acc_ci(c, nxt())
            if "b31" in groups:
                Bm, bn = (Pr, "probe") if np.mean(Pr["acc"]) > np.mean(Lo["acc"]) else (Lo, "lora")
                out["margins"].append(margin(N, Bm, "b31(b)|%s|%s|name-%s" % (enc, ar, bn), "B31 (b)", nxt()))
                ob, on = (Lo, "lora") if bn == "probe" else (Pr, "probe")         # the other equalised baseline
                out["margins"].append(margin(N, ob, "b31|%s|%s|name-%s" % (enc, ar, on),
                                             "B31 (naming head - the other equalised baseline; the paper cites "
                                             "name - LoRA)", nxt()))
                out["margins"].append(margin(N, Li, "b31(c)|%s|%s|name-loraid" % (enc, ar), "B31 (c)", nxt()))
                out["margins"].append(margin(Li, Lo, "b31(c)|%s|%s|loraid-lora" % (enc, ar), "B31 (c) effect", nxt()))
    # ---------------- B31 exploratory 1
    if "ex1" in groups:
        for ar, L in (("A", 9), ("B", 9), ("C", 27)):
            p = os.path.join(R, "explore1", "qwen8L%d_%s_name.json" % (L, ar))
            Nx = mem(p, "perm/named", lab31(ar))
            if Nx is None:
                out["notes"].append("exploratory 1 output missing: %s" % p); continue
            out["accuracies"]["b31ex1|qwen8L%d|%s|name" % (L, ar)] = acc_ci(Nx, nxt())
            out["margins"].append(margin(Nx, cache[("lora", "qwen8", ar)], "b31ex1|qwen8L%d|%s|name-lora" % (L, ar),
                                         "B31 exploratory 1", nxt()))
    # ---------------- recoveries (B31 label-free maps, B42 interfaces), arenas C and A
    if "rec" in groups:
        for ar in "CA":
            y = lab31(ar)
            for enc in ("phi2", "qwen8"):
                Fc, Nc = cache[("probe", enc, ar)], cache[("name", enc, ar)]
                arms = [("b31 " + k, os.path.join(R, "arms_final", "%s_%s_%s.json" % (enc, ar, k)), rd)
                        for k, rd in (("bind", "perm/named"), ("bindpub", "comm/index"), ("nce", "perm/named"),
                                      ("soft", "perm/named"))]
                arms += [("b42 " + k, os.path.join(a.b42, "%s_%s_%s.json" % (enc, ar, k)), "perm/named")
                         for k in ("vq", "dkvb", "gumbel", "st", "cw")]
                arms += [("b42 bind+pc", os.path.join(a.b42, "%s_%s_bind.json" % (enc, ar)), "perm/named+pc"),
                         ("b42 bind+pz", os.path.join(a.b42, "%s_%s_bind.json" % (enc, ar)), "perm/named+pz")]
                for nm, p, rd in arms:
                    A = mem(p, rd, y)
                    if A is None:
                        out["notes"].append("missing %s" % p); continue
                    out["recoveries"].append(recovery(A, Fc, Nc, "rec|%s|%s|%s" % (enc, ar, nm), nm.split()[0].upper(),
                                                      nxt()))
    # ---------------- B44
    if "b44" in groups:
        sel = (load(os.path.join(a.b44, "select.json")) or {}).get("lora", {})
        for v in ("std", "ho"):
            for n in (800, 8000, 80000):
                Nm = b44_mem(a.b44, "name", v, n)
                base = {"lora": b44_lora(a.b44, "cls", v, n, sel), "scratch": b44_lora(a.b44, "scratch", v, n, sel),
                        "probe": b44_mem(a.b44, "probe", v, n)}
                for k, c in list(base.items()) + [("name", Nm)]:
                    if c is not None:
                        out["accuracies"]["b44|%s|n%d|%s" % (v, n, k)] = acc_ci(c, nxt())
                if Nm is None:
                    out["notes"].append("B44 %s n=%d naming head not finished" % (v, n)); continue
                avail = {k: c for k, c in base.items() if c is not None}
                missing = [k for k, c in base.items() if c is None]
                if missing:
                    out["notes"].append("B44 %s n=%d: %s not finished (lead computed over the finished baselines only)"
                                        % (v, n, "/".join(missing)))
                bk = max(avail, key=lambda k: np.mean(avail[k]["acc"]))
                out["margins"].append(margin(Nm, avail[bk], "b44|%s|n%d|name-%s" % (v, n, bk),
                                             "B44 lead (finished baselines: %s)" % "/".join(sorted(avail)), nxt()))
                if base["scratch"] is not None and base["lora"] is not None:
                    out["margins"].append(margin(base["scratch"], base["lora"], "b44|%s|n%d|scratch-lora" % (v, n),
                                                 "B44 bar 3", nxt()))
        cot = b44_cot(a.b44, "fs-nothink", "std")
        N8 = b44_mem(a.b44, "name", "std", 800)
        L8 = b44_lora(a.b44, "cls", "std", 800, sel)
        if cot and N8:
            out["accuracies"]["b44|std|cot fs-nothink"] = acc_ci(cot, nxt())
            out["margins"].append(margin(cot, N8, "b44|std|cot_fs-nothink-name800", "B44 bar 4", nxt()))
            if L8:
                out["margins"].append(margin(cot, L8, "b44|std|cot_fs-nothink-lora800", "B44 bar 4", nxt()))
    # ---------------- B47
    if "b47" in groups:
        sel47 = load(a.b47_select) or {}
        N = cache[("name", "phi2", "C")]
        for task in ("gen", "fullgen", "fullcls", "scratch"):
            c = b47_ft(a.b47, task, "k32", 800, sel47)
            if c is None:
                out["notes"].append("B47 %s n=800 not finished" % task); continue
            out["accuracies"]["b47|phi2|C|%s|n800" % task] = acc_ci(c, nxt())
            out["margins"].append(margin(N, c, "b47|phi2|C|n800|name-%s" % task, "B47 (1)", nxt()))
        N8k = None
        p8 = os.path.join(a.b47, "..", "name", "phi2_name_std_n8000.json")
        if os.path.exists(p8):
            N8k = mem(p8, "perm/named", lab44("std"))
        for task in ("gen", "fullgen"):
            c = b47_ft(a.b47, task, "std", 8000, sel47)
            if c is None:
                out["notes"].append("B47 %s n=8000 not finished" % task); continue
            out["accuracies"]["b47|phi2|C|%s|n8000" % task] = acc_ci(c, nxt())
            out["margins"].append(margin(N8k or N, c, "b47|phi2|C|n8000|name%s-%s" % ("8000" if N8k else "800", task),
                                         "B47 (1)", nxt()))
    out["files_read_md5"] = FILES
    if a.update and os.path.exists(a.out):
        old = json.load(open(a.out))
        pref = tuple(g for g in groups)
        keep = lambda i: not any(i.startswith(g) for g in pref) and not (("rec" in groups) and i.startswith("rec|"))
        for k in ("margins", "recoveries"):
            out[k] = [m for m in old[k] if keep(m["id"])] + out[k]
        acc = {k: v for k, v in old["accuracies"].items() if not any(k.startswith(g) for g in pref)}
        acc.update({k: v for k, v in out["accuracies"].items() if any(k.startswith(g) for g in pref)})
        out["accuracies"] = acc
        out["notes"] = [x for x in old["notes"] if not any(g.upper() in x for g in pref)] + out["notes"]
        old["files_read_md5"].update(FILES); out["files_read_md5"] = old["files_read_md5"]
        out["updated_groups"] = sorted(groups)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    for m in out["margins"]:
        print("%-42s %+.3f  hnest [%+.3f, %+.3f]  hcross [%+.3f, %+.3f]  inst %s  t %s %s" % (
            m["id"], m["point"], *m["ci"]["hnest"], *m["ci"]["hcross"],
            "[%+.3f, %+.3f]" % tuple(m["ci"]["inst"]) if "inst" in m["ci"] else "-",
            "[%+.3f, %+.3f]" % tuple(m["ci"]["tseed"]) if m["ci"]["tseed"] else "-", m["flag"]))
    for r in out["recoveries"]:
        if r["point"] is None:
            continue
        print("%-42s R %.3f  hnest [%.3f, %.3f]  hcross [%.3f, %.3f]  t %s" % (
            r["id"], r["point"], *r["ci"]["hnest"], *r["ci"]["hcross"],
            "[%.3f, %.3f]" % tuple(r["ci"]["tseed"]) if r["ci"]["tseed"] else "-"))
    bad = [k for k, v in out["accuracies"].items() if v["recomputed_equals_logged"] is False]
    print("accuracies recomputed != logged:", bad)
    print("notes:", out["notes"])


if __name__ == "__main__":
    main()
