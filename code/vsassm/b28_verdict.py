"""B28 verdict (prereg/B28_noncommutative_binding_prereg.md). Written before any test output exists.

Every accuracy is recomputed here from the saved per-question outputs (bAbI: predicted symbol against the gold
label of the dumped structure; arena C: saved final cosines against the saved training threshold and the gold
labels; LoRA: argmax of the saved logits against the dataset labels), independently of the numbers the runs
logged, and compared with them. Gates, bars, predictions and readings are applied exactly as registered.

  python b28_verdict.py --runs b28_out --indep results/b28_indep.json --ledger results/b28_gpu_ledger.txt \
      --out results/b28_verdict.txt --scores results/b28_scores.json
"""
import argparse, json, os, sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b28_common import wilson, boot_mean, boot_diff, ENCODINGS, SEEDS   # noqa: E402

BANK = {  # banked per-seed values (figdata / ngcls logs), exact symbols and naming head under comm
    "task4": {"parser": [0.674, 0.702, 0.696], "probe_argmax": [0.674, 0.702, 0.696], "lora": [1.0, 1.0, 1.0]},
    "task15": {"parser": [0.824, 0.848, 0.836], "probe_argmax": [0.824, 0.848, 0.836], "lora": [0.992, 0.992, 0.990]},
    "C": {"parser": [0.932, 0.932, 0.938], "probe_argmax": [0.932, 0.932, 0.938]},
}
FT_BANK = {"r8_e2_lr0.0001": [[0.8500, 0.6952380952380952, 0.8571428571428571],      # v50_task2_BANKED
                              [0.85, 0.6952380952380952, 0.8404761904761905]],       # V-131 rerun, same code
           "r16_e6_lr0.0001": [[0.9714285714285714, 0.9738095238095238, 0.9714285714285714]]}
TOL_EXACT, TOL_HEAD, TOL_FT = 0.002, 0.004, 0.03
REMOVE_BAR = 0.98
SETTINGS = ["task4", "task15", "C"]
NAMES = {"task4": "bAbI task 4", "task15": "bAbI task 15", "C": "arena C"}
ARMS = {"parser": "exact codebook rows", "probe_argmax": "naming head"}

L = []


def P(s=""):
    L.append(s); print(s)


def f3(x):
    return "%.3f" % x if x is not None and x == x else "n/a"


def load(p):
    """I/O only: reads p, or p.gz as committed in b28_out/ (added after the outputs existed; no scoring change)."""
    if os.path.exists(p):
        return json.load(open(p))
    if os.path.exists(p + ".gz"):
        import gzip
        return json.load(gzip.open(p + ".gz", "rt"))
    return None


# ------------------------------------------------------------------ per-question correctness, recomputed

def babi_matrix(d, arm, enc):
    """(seeds x n) correctness recomputed from the saved predictions and the gold labels; None if absent."""
    rows = []
    for s in SEEDS:
        k = "%s/%s/%d" % (arm, enc, s)
        if k not in d["runs"]:
            return None, None
        pq = d["runs"][k]["per_q"]
        rows.append([int(o["pred"] == g["label"]) for o, g in zip(pq, d["gold"])])
    logged = [d["runs"]["%s/%s/%d" % (arm, enc, s)]["acc_scored"] for s in SEEDS]
    return np.array(rows), logged


def c_matrix(dc):
    """arena C: correctness at the best test epoch (published protocol) and at the final epoch, recomputed from
    the saved cosines and thresholds; retrieval correctness at the final epoch."""
    y = np.array([g["label"] for g in dc["gold"]])
    best, final, ret, logged, named = [], [], [], [], []
    for r in sorted(dc["runs"], key=lambda r: SEEDS.index(r["seed"])):
        accs, cors = [], []
        for e in r["epochs"]:
            s = np.array(e["sims"])
            pred = (s >= e["thr"]).astype(int) if e["thr"] is not None else np.zeros_like(y)
            c = (pred == y).astype(int); cors.append(c); accs.append(c.mean())
        bi = int(np.argmax(accs))                       # earliest best epoch
        best.append(cors[bi]); final.append(cors[-1])
        hp = r["epochs"][-1]["hops"]
        ret.append([int(h[0] == g["s1"] and h[1] == g["s2"]) for h, g in zip(hp, dc["gold"])])
        logged.append(r["acc_best"])
        named.append(float(np.mean(r["epochs"][-1]["named_ok"])))
    return np.array(best), np.array(final), np.array(ret), logged, named


def seedline(M):
    n = M.shape[1]
    return ", ".join("%.3f [%.3f-%.3f]" % ((M[i].mean(),) + wilson(int(M[i].sum()), n)) for i in range(M.shape[0]))


def bar_reading(M, Mc):
    m, mc = M.mean(), Mc.mean()
    per = M.mean(axis=1)
    if all(per >= REMOVE_BAR):
        return "REMOVES THE LIMIT"
    lo, hi = boot_diff(M, Mc)
    if m - mc >= 0.05 and lo > 0:
        return "PARTIAL"
    if abs(m - mc) < 0.03:
        return "NO EFFECT"
    return "GRADED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--indep", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", required=True)
    a = ap.parse_args()
    S = {"gates": {}, "part1": {}, "mechanism": {}, "task2": {}, "predictions": {}}
    P("# B28 verdict (prereg/B28_noncommutative_binding_prereg.md): does an order-preserving VSA encoding remove")
    P("# the commutative-binding limit, and how do the bAbI task-2 routes compare on all 500 questions?")
    P("# Generated by b28_verdict.py on %s from the per-question outputs in %s." % (
        datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), a.runs))
    P()

    D = {"task4": load(os.path.join(a.runs, "b28_task4.json")),
         "task15": load(os.path.join(a.runs, "b28_task15.json"))}
    DC = {(arm, enc): load(os.path.join(a.runs, "b28_C_test_%s_%s.json" % (arm, enc)))
          for arm in ARMS for enc in ENCODINGS}
    M = {}           # (setting, arm, enc) -> (seeds x n) correctness (C: best epoch)
    MF, RET = {}, {}  # C final epoch, C retrieval
    logs = {}
    for st in ("task4", "task15"):
        if D[st] is None:
            continue
        for arm in ARMS:
            for enc in ENCODINGS:
                m_, lg = babi_matrix(D[st], arm, enc)
                if m_ is not None:
                    M[(st, arm, enc)] = m_; logs[(st, arm, enc)] = lg
                    RET[(st, arm, enc)] = m_
    named_C = {}
    for (arm, enc), dc in DC.items():
        if dc is None or len(dc["runs"]) < 3:
            continue
        b, f, r, lg, nm = c_matrix(dc)
        M[("C", arm, enc)] = b; MF[("C", arm, enc)] = f; RET[("C", arm, enc)] = r; logs[("C", arm, enc)] = lg
        named_C[(arm, enc)] = nm

    # ------------------------------------------------------------ gates
    P("## Gates")
    gate_ok = {}
    g0 = {"task4": load(os.path.join(a.runs, "g0_v58.json")), "task15": load(os.path.join(a.runs, "g0_v52.json"))}
    for st in ("task4", "task15"):
        for arm in ARMS:
            if g0[st] is None or arm not in g0[st].get("arms", {}):
                P("  G0 %-6s %-12s original script: MISSING" % (st, arm)); continue
            got = g0[st]["arms"][arm]
            ok = all(abs(x - y) <= TOL_EXACT for x, y in zip(got, BANK[st][arm]))
            S["gates"]["G0/%s/%s" % (st, arm)] = ok
            P("  G0 %-6s %-12s original script %s  banked %s  %s" % (st, arm, [round(x, 4) for x in got],
                                                                     BANK[st][arm], "PASS" if ok else "FAIL"))
    g37 = load(os.path.join(a.runs, "g0_v37b.json"))
    if g37 is not None:
        for arm in ("parser", "probe_argmax"):
            got = g37["arms"].get(arm, [])
            ok = len(got) == 3 and all(abs(x - 1.0) <= TOL_EXACT for x in got)
            S["gates"]["G0/task2/%s" % arm] = ok
            P("  G0 task2  %-12s original script %s on %s expressible  banked 1.000 x3  %s" % (
                arm, [round(x, 4) for x in got], g37.get("n_eval_expressible"), "PASS" if ok else "FAIL"))
    for st in SETTINGS:
        for arm in ARMS:
            k = (st, arm, "comm")
            if k not in M:
                P("  G1/G2 %-6s %-12s new code path, comm: MISSING" % (st, arm)); gate_ok[(st, arm)] = False; continue
            got = M[k].mean(axis=1)
            tol = TOL_EXACT if arm == "parser" else TOL_HEAD
            ok = all(abs(x - y) <= tol for x, y in zip(got, BANK[st][arm]))
            rec_ok = all(abs(x - y) < 1e-9 for x, y in zip(got, logs[k]))
            gate_ok[(st, arm)] = ok
            S["gates"]["G%s/%s/%s" % ("2" if st == "C" else "1", st, arm)] = ok
            P("  G%s %-6s %-12s new code path, comm %s  banked %s  %s  (recomputed = logged: %s)" % (
                "2" if st == "C" else "1", st, arm, [round(float(x), 4) for x in got], BANK[st][arm],
                "PASS" if ok else "FAIL", rec_ok))
    d2 = load(os.path.join(a.runs, "b28_task2.json"))
    if d2 is not None:
        ex = np.array(d2["expressible"])
        okc = d2["n_scored"] == 444 and int(ex.sum()) == 420
        P("  G3 task2  parse reads %d of %d test questions, %d expressible (published 444 / 420): %s" % (
            d2["n_scored"], d2["n_questions"], int(ex.sum()), "PASS" if okc else "FAIL"))
        S["gates"]["G3/counts"] = okc
        for arm in ARMS:
            accs = []
            for s in SEEDS:
                pq = d2["runs"]["%s/comm/%d" % (arm, s)]["per_q"]
                c = np.array([int(o["pred"] == g["label"]) for o, g in zip(pq, d2["gold"])])
                accs.append(float(c[ex == 1].mean()))
            ok = all(abs(x - 1.0) <= TOL_EXACT for x in accs)
            S["gates"]["G3/%s" % arm] = ok
            P("  G3 task2  %-12s new code path, comm, on the 420 expressible %s  banked 1.000 x3  %s" % (
                arm, [round(x, 4) for x in accs], "PASS" if ok else "FAIL"))
    P("  G4 (reported, not blocking): see the task-2 section")
    P()

    # ------------------------------------------------------------ part 1 and 2
    P("## Exact symbols and naming head under each encoding (per seed acc [Wilson 95%]; mean [bootstrap 95%];")
    P("## difference from comm [paired bootstrap 95%]; arena C at the best test epoch as published)")
    readings = {}
    for st in SETTINGS:
        P("  %s" % NAMES[st])
        for arm in ARMS:
            if not gate_ok.get((st, arm)):
                P("    %-20s gate failed or missing: not read" % ARMS[arm]); continue
            Mc = M[(st, arm, "comm")]
            for enc in ENCODINGS:
                k = (st, arm, enc)
                if k not in M:
                    P("    %-20s %-5s MISSING" % (ARMS[arm], enc)); continue
                X = M[k]; lo, hi = boot_mean(X)
                line = "    %-20s %-5s mean %.3f [%.3f-%.3f]  seeds %s" % (ARMS[arm], enc, X.mean(), lo, hi, seedline(X))
                if enc != "comm":
                    dl, dh = boot_diff(X, Mc)
                    rd = bar_reading(X, Mc)
                    readings[k] = rd
                    line += "  diff %+.3f [%+.3f, %+.3f] -> %s" % (X.mean() - Mc.mean(), dl, dh, rd)
                if st == "C":
                    line += "  | final epoch %.3f" % MF[k].mean()
                    if arm == "probe_argmax":
                        line += ", naming exact on %s of test instances" % "/".join("%.3f" % x for x in named_C[(arm, enc)])
                P(line)
                rec_ok = all(abs(x - y) < 1e-9 for x, y in zip(X.mean(axis=1), logs[k]))
                if not rec_ok:
                    P("      NOTE: recomputed %s differs from logged %s" % (
                        [round(float(x), 4) for x in X.mean(axis=1)], [round(float(x), 4) for x in logs[k]]))
                S["part1"]["%s/%s/%s" % (st, arm, enc)] = {
                    "per_seed": [float(x) for x in X.mean(axis=1)], "mean": float(X.mean()), "ci": [lo, hi],
                    "reading": readings.get(k)}
        P()
    P("## Bars (primary encoding perm; roles secondary; rtag and shift are controls registered to fail)")
    for st in SETTINGS:
        for arm in ARMS:
            P("  %-13s %-20s perm: %-18s roles: %-18s rtag: %-18s shift: %s" % (
                NAMES[st], ARMS[arm], readings.get((st, arm, "perm")), readings.get((st, arm, "roles")),
                readings.get((st, arm, "rtag")), readings.get((st, arm, "shift"))))
    overall = all(readings.get((st, "parser", "perm")) == "REMOVES THE LIMIT" for st in SETTINGS)
    P("  OVERALL (exact symbols, perm REMOVES in all three settings): %s" % (
        "the commutative-binding limit is a property of the encoding" if overall else "not established in all three"))
    S["overall_removes"] = overall
    P()

    # ------------------------------------------------------------ mechanism
    P("## Mechanism (second angle): accuracy split by the structural alias flag (bAbI: answer correct;")
    P("## arena C: both hops retrieve the true symbol, final epoch)")
    for st in SETTINGS:
        gold = D[st]["gold"] if st != "C" else (DC[("parser", "comm")] or {}).get("gold")
        if gold is None:
            continue
        al = np.array([g["alias"] for g in gold])
        P("  %s: alias share %.3f (%d of %d)" % (NAMES[st], al.mean(), al.sum(), len(al)))
        S["mechanism"]["%s/alias_share" % st] = float(al.mean())
        for arm in ARMS:
            for enc in ENCODINGS:
                k = (st, arm, enc)
                if k not in RET:
                    continue
                R = RET[k]
                a1 = [float(R[i][al == 1].mean()) if al.sum() else float("nan") for i in range(3)]
                a0 = [float(R[i][al == 0].mean()) if (1 - al).sum() else float("nan") for i in range(3)]
                S["mechanism"]["%s/%s/%s" % (st, arm, enc)] = {"alias": a1, "non_alias": a0}
                P("    %-20s %-5s alias %s | non-alias %s" % (ARMS[arm], enc, "/".join(f3(x) for x in a1),
                                                             "/".join(f3(x) for x in a0)))
    P()

    # ------------------------------------------------------------ memory vs fine-tune on 4 and 15
    P("## Memory (naming head) against the banked fine-tune on tasks 4 and 15")
    tie = {}
    for st in ("task4", "task15"):
        ft = BANK[st]["lora"]
        for enc in ("comm", "perm", "roles"):
            k = (st, "probe_argmax", enc)
            if k not in M:
                continue
            per = M[k].mean(axis=1); dmean = float(per.mean() - np.mean(ft))
            if abs(dmean) <= 0.02:
                rd = "TIE"
            elif all(f - h >= 0.05 for h, f in zip(per, ft)):
                rd = "FINE-TUNE LEADS"
            elif all(h - f >= 0.10 for h, f in zip(per, ft)):
                rd = "MEMORY LEADS"
            else:
                rd = "GRADED"
            tie[(st, enc)] = rd
            P("  %-12s %-5s naming head %s vs fine-tune %s: diff %+.3f -> %s" % (
                NAMES[st], enc, [round(float(x), 3) for x in per], ft, dmean, rd))
    P()

    # ------------------------------------------------------------ task 2 on all 500
    P("## bAbI task 2 on all 500 test questions")
    t2 = {}
    if d2 is not None:
        lab = np.array(d2["labels_all"]); n = len(lab)
        idx = np.array(d2["row_index"]); ex_all = np.zeros(n, int); ex_all[idx] = np.array(d2["expressible"])
        parsed = np.zeros(n, int); parsed[idx] = 1
        sub = {"all 500": np.ones(n, bool), "420 expressible": ex_all == 1,
               "24 parsed, not expressible": (parsed == 1) & (ex_all == 0), "56 not parsed": parsed == 0}

        def mem_matrix(dd, arm, enc):
            rows = []
            for s in SEEDS:
                k = "%s/%s/%d" % (arm, enc, s)
                if k not in dd["runs"]:
                    return None
                c = np.zeros(n, int)
                ii = np.array(dd["row_index"])
                pr = np.array([o["pred"] for o in dd["runs"][k]["per_q"]])
                c[ii] = (pr == lab[ii]).astype(int)
                rows.append(c)
            return np.array(rows)

        arms2 = {}
        for arm in ARMS:
            for enc in ENCODINGS:
                m_ = mem_matrix(d2, arm, enc)
                if m_ is not None:
                    arms2["MEM %s %s" % ("naming head" if arm == "probe_argmax" else "exact rows", enc)] = m_
        d2v = load(os.path.join(a.runs, "b28_task2_completed.json"))
        if d2v is not None:
            for arm in ARMS:
                m_ = mem_matrix(d2v, arm, "comm")
                if m_ is not None:
                    arms2["MEM-v %s comm (completed verbs)" % ("naming head" if arm == "probe_argmax" else "exact rows")] = m_
        ft = {}
        for tag in ("r8_e2_lr0.0001", "r16_e6_lr0.0001"):
            for trn in ("expr", "all"):
                dl = load(os.path.join(a.runs, "b28_lora2_%s_%s.json" % (tag, trn)))
                if dl is None or len(dl["runs"]) < 3:
                    continue
                kept = dl["kept"]; labs = np.array(dl["labels_all"])
                assert np.array_equal(labs, lab), "LoRA labels differ from the memory run's labels"
                rows, accA, accB_log = [], [], []
                for r in sorted(dl["runs"], key=lambda r: SEEDS.index(r["seed"])):
                    pB = np.array([kept[int(np.argmax(x))] for x in r["logitsB"]])
                    rows.append((pB == labs).astype(int))
                    pA = np.array([kept[int(np.argmax(x))] for x in r["logitsA"]])
                    accA.append(float((pA == labs[np.array(dl["passA_index"])]).mean()))
                    accB_log.append(r["accB"])
                name = "FT %s (%s, trained on %s)" % ("first config r8/2ep/1e-4" if tag.startswith("r8") else
                                                      "best of twelve r16/6ep/1e-4", tag, "775 expressible" if trn == "expr" else "all 900")
                arms2[name] = np.array(rows)
                ft[(tag, trn)] = (name, accA)
                rec_ok = all(abs(x - y) < 1e-9 for x, y in zip(np.array(rows).mean(axis=1), accB_log))
                if not rec_ok:
                    P("  NOTE: %s recomputed all-500 accuracy differs from logged" % name)
        P("  G4 fine-tune reruns on the 420 (pass A, v50 order) against the banked runs (tolerance %.2f):" % TOL_FT)
        for (tag, trn), (name, accA) in ft.items():
            if trn != "expr":
                continue
            banks = FT_BANK[tag]
            ok = all(any(abs(x - b[i]) <= TOL_FT for b in banks) for i, x in enumerate(accA))
            S["gates"]["G4/%s" % tag] = ok
            P("    %s: rerun %s  banked %s  %s" % (tag, [round(x, 4) for x in accA],
                                                  [[round(x, 4) for x in b] for b in banks], "PASS" if ok else "MISS"))
        P("  Accuracy by subset (mean over seeds; per seed in parentheses):")
        P("    %-62s %s" % ("arm", "  ".join("%-22s" % k for k in sub)))
        for name, X in arms2.items():
            cells = []
            for k, msk in sub.items():
                if msk.sum() == 0:
                    cells.append("%-22s" % "-"); continue
                per = X[:, msk].mean(axis=1)
                cells.append("%-22s" % ("%.3f (%s)" % (per.mean(), "/".join("%.3f" % x for x in per))))
            P("    %-62s %s" % (name, "  ".join(cells)))
            lo, hi = boot_mean(X)
            t2[name] = {"all500_per_seed": [float(x) for x in X.mean(axis=1)], "all500_mean": float(X.mean()),
                        "ci": [lo, hi], "subsets": {k: float(X[:, m_].mean()) for k, m_ in sub.items() if m_.sum()}}
        P("  subset sizes: %s" % ", ".join("%s %d" % (k, int(m_.sum())) for k, m_ in sub.items()))
        MEMk = "MEM naming head comm"
        guess = sub["56 not parsed"].sum() / 6.0 / n
        if MEMk in arms2:
            P("  sensitivity: a uniform guess over the six locations on the %d unparsed questions would add %.3f "
              "in expectation (%s -> %.3f)" % (int(sub["56 not parsed"].sum()), guess, f3(arms2[MEMk].mean()),
                                              arms2[MEMk].mean() + guess))
        P("  Registered contrasts (memory = naming head, comm, published parse and ordering rule):")
        for tag, label in (("r16_e6_lr0.0001", "best of twelve"), ("r8_e2_lr0.0001", "first config")):
            if MEMk not in arms2 or (tag, "expr") not in ft:
                continue
            F_ = arms2[ft[(tag, "expr")][0]]; Mm = arms2[MEMk]
            per = Mm.mean(axis=1) - F_.mean(axis=1)
            dl, dh = boot_diff(Mm, F_)
            if all(per >= 0.10):
                rd = "MEMORY LEADS"
            elif all(-per >= 0.02) and dh < 0:
                rd = "FINE-TUNE LEADS"
            else:
                rd = "NO CLEAR DIFFERENCE"
            t2["contrast/%s" % tag] = {"per_seed": [float(x) for x in per], "mean": float(per.mean()),
                                       "ci": [dl, dh], "reading": rd}
            P("    memory - fine-tune (%s), all 500: per seed %s, mean %+.3f [%+.3f, %+.3f] -> %s" % (
                label, "/".join("%+.3f" % x for x in per), per.mean(), dl, dh, rd))
            m420 = sub["420 expressible"]
            per4 = Mm[:, m420].mean(axis=1) - F_[:, m420].mean(axis=1)
            P("      (same contrast on the 420 the paper used: %s, mean %+.3f)" % (
                "/".join("%+.3f" % x for x in per4), per4.mean()))
    S["task2"] = t2
    P()

    # ------------------------------------------------------------ independent implementation
    P("## Second implementation (numpy float64, b28_indep.py)")
    ind = load(a.indep) if a.indep else None
    agree_ok = None
    if ind is not None:
        ag = [v.get("agree") for v in ind["same_codebook"].values() if v.get("agree") is not None]
        agree_ok = bool(ag) and min(ag) >= 0.99
        P("  per-question agreement with the main runs, same codebooks: min %.4f over %d cells" % (min(ag), len(ag)))
        for k, v in ind["same_codebook"].items():
            if v.get("agree") is not None and v["agree"] < 1.0:
                P("    %s agree %.4f (indep %.4f, main %.4f)" % (k, v["agree"], v["acc_indep"], v["acc_main"]))
        P("  arena C instances regenerated without the tokenizer match the main run: %s" % ind.get("arenaC_regen_check"))
        P("  fresh codebooks (20 each; mean, min-max):")
        for k, v in ind["fresh_codebooks"].items():
            P("    %-14s %.4f  (%.4f-%.4f)" % (k, np.mean(v), min(v), max(v)))
    P()

    # ------------------------------------------------------------ predictions
    P("## Predictions (registered)")
    pr = {}

    def H(name, val):
        pr[name] = val
        P("  %-100s %s" % (name, {True: "HELD", False: "NOT HELD", None: "NOT EVALUABLE"}[val]))

    gvals = [v for k, v in S["gates"].items() if not k.startswith("G4")]
    H("P1 gates G0-G3 pass in every cell", bool(gvals) and all(gvals))
    for st in SETTINGS:
        H("P2 perm, exact symbols, REMOVES on %s" % NAMES[st],
          readings.get((st, "parser", "perm")) == "REMOVES THE LIMIT" if (st, "parser", "perm") in readings else None)
    for st in SETTINGS:
        k = (st, "probe_argmax", "perm"); kp = (st, "parser", "perm")
        if k in M and kp in M:
            H("P3 perm, naming head, REMOVES on %s and equals exact rows within %.3f per seed" % (NAMES[st], TOL_HEAD),
              readings.get(k) == "REMOVES THE LIMIT" and
              all(abs(x - y) <= TOL_HEAD for x, y in zip(M[k].mean(axis=1), M[kp].mean(axis=1))))
    for st in SETTINGS:
        H("P4 roles, exact symbols, REMOVES on %s" % NAMES[st],
          readings.get((st, "parser", "roles")) == "REMOVES THE LIMIT" if (st, "parser", "roles") in readings else None)
    for st in SETTINGS:
        H("P5 rtag, exact symbols, does NOT remove on %s" % NAMES[st],
          readings.get((st, "parser", "rtag")) != "REMOVES THE LIMIT" if (st, "parser", "rtag") in readings else None)
    for st in SETTINGS:
        k, kc = (st, "parser", "shift"), (st, "parser", "comm")
        if k in RET and kc in RET:
            if st == "C":
                agr = [float((DC[("parser", "shift")]["runs"][i]["epochs"][-1]["pred"] ==
                              np.array(DC[("parser", "comm")]["runs"][i]["epochs"][-1]["pred"])).mean()) for i in range(3)]
            else:
                agr = [float(np.mean([int(x["pred"] == y["pred"]) for x, y in zip(
                    D[st]["runs"]["parser/shift/%d" % s]["per_q"], D[st]["runs"]["parser/comm/%d" % s]["per_q"])]))
                    for s in SEEDS]
            H("P6 shift equals comm per question (agreement >= 0.99) on %s [%s]" % (
                NAMES[st], "/".join("%.3f" % x for x in agr)), min(agr) >= 0.99)
    for st in SETTINGS:
        mc = S["mechanism"].get("%s/parser/comm" % st)
        if mc:
            H("P7 comm: non-alias >= 0.99 and alias in [0.30, 0.70], every seed, %s" % NAMES[st],
              all(x >= 0.99 for x in mc["non_alias"]) and all(0.30 <= x <= 0.70 for x in mc["alias"]))
    for st in SETTINGS:
        mp = S["mechanism"].get("%s/parser/perm" % st)
        if mp:
            H("P8 perm: >= 0.98 on alias and on non-alias instances, every seed, %s" % NAMES[st],
              all(x >= 0.98 for x in mp["alias"] + mp["non_alias"]))
    rng_ = {"task4": (0.50, 0.75), "task15": (0.20, 0.45), "C": (0.10, 0.25)}
    for st in SETTINGS:
        v = S["mechanism"].get("%s/alias_share" % st)
        if v is not None:
            H("P9 alias share of %s in [%.2f, %.2f] (measured %.3f)" % ((NAMES[st],) + rng_[st] + (v,)),
              rng_[st][0] <= v <= rng_[st][1])
    for st in ("task4", "task15"):
        if (st, "perm") in tie:
            H("P10 memory (naming head, perm) TIES the banked fine-tune on %s" % NAMES[st], tie[(st, "perm")] == "TIE")
    if "MEM naming head comm" in t2:
        per = t2["MEM naming head comm"]["all500_per_seed"]
        H("P11a task 2: memory on all 500 in [0.835, 0.850] every seed (%s)" % "/".join("%.3f" % x for x in per),
          all(0.835 <= x <= 0.850 for x in per))
    k12 = [k for k in t2 if k.startswith("FT best of twelve") and "775" in k]
    if k12:
        v = t2[k12[0]]["all500_mean"]
        H("P11b task 2: tuned fine-tune (best of twelve) on all 500, mean in [0.90, 0.96] (%.3f)" % v, 0.90 <= v <= 0.96)
    if "contrast/r16_e6_lr0.0001" in t2:
        H("P11c task 2: memory vs best of twelve on all 500 = FINE-TUNE LEADS",
          t2["contrast/r16_e6_lr0.0001"]["reading"] == "FINE-TUNE LEADS")
    if "contrast/r8_e2_lr0.0001" in t2:
        H("P12 task 2: memory vs first configuration on all 500 = NO CLEAR DIFFERENCE",
          t2["contrast/r8_e2_lr0.0001"]["reading"] == "NO CLEAR DIFFERENCE")
    if "MEM naming head comm" in t2:
        s_ = t2["MEM naming head comm"]["subsets"]
        oth = 1 - (420 / 500)
        m80 = (s_["all 500"] * 500 - s_["420 expressible"] * 420) / 80
        H("P13 task 2: memory 1.000 on the 420 and <= 0.05 on the other 80 (%.3f / %.3f)" % (s_["420 expressible"], m80),
          s_["420 expressible"] >= 0.999 and m80 <= 0.05)
    if ind is not None:
        H("P14a independent implementation agrees >= 0.99 per question in every cell", agree_ok)
        fr = ind["fresh_codebooks"]
        okp = all(min(fr[k]) >= 0.98 for k in fr if k.endswith("/perm"))
        H("P14b fresh codebooks: perm >= 0.98 in 20/20 for every setting", okp)
        pub = {"task4/comm": 0.691, "task15/comm": 0.836, "C/comm": 0.934}
        H("P14c fresh codebooks: comm mean within 0.03 of the published ceiling (task 4 .691, task 15 .836, C .934)",
          all(abs(np.mean(fr[k]) - v) <= 0.03 for k, v in pub.items() if k in fr))
    kv = [k for k in t2 if k.startswith("MEM-v naming head")]
    if kv:
        v = t2[kv[0]]["all500_mean"]
        H("P15 completed verb patterns: memory on all 500 in [0.90, 0.97] (%.3f)" % v, 0.90 <= v <= 0.97)
    ka = [k for k in t2 if k.startswith("FT best of twelve") and "900" in k]
    if ka and k12:
        H("P16 best of twelve trained on all 900 >= trained on the 775 (all-500 mean)",
          t2[ka[0]]["all500_mean"] >= t2[k12[0]]["all500_mean"])
    S["predictions"] = {k: v for k, v in pr.items()}
    P()

    # ------------------------------------------------------------ compute
    P("## Compute (GPU occupancy per B28 worker, A800 80 GB)")
    tot = 0.0
    if a.ledger and os.path.exists(a.ledger):
        starts = {}
        for line in open(a.ledger):
            p_ = line.split()
            if len(p_) >= 4 and p_[2] in ("start", "end"):
                t = datetime.strptime(p_[3], "%Y-%m-%dT%H:%M:%SZ")
                if p_[2] == "start":
                    starts[(p_[0], p_[1])] = t
                elif (p_[0], p_[1]) in starts:
                    t_s = starts.pop((p_[0], p_[1]))
                    h = (t - t_s).total_seconds() / 3600
                    tot += h
                    P("  %-14s %s %s -> %s  %.2f h" % (p_[0], p_[1], t_s.strftime("%m-%d %H:%M"),
                                                      t.strftime("%H:%M"), h))
        if starts:
            P("  still open: %s" % sorted(starts))
    P("  total %.2f GPU-hours (cap 40)" % tot)
    S["gpu_hours"] = tot
    H("P17 total GPU-hours <= 8", tot <= 8.0 if tot > 0 else None)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(S, open(a.scores, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
