"""B65 verdict (prereg/B65_tuned_lr_read_position_prereg.md, 'Decision rule' and 'Reported'). Reads prediction files and
the dev choices only.
Layout: <root>/preds/<pos>/<arm>_s<seed>.jsonl (pos in last, ops; arm in vsapub, vsabm, disc), <root>/out/choices.json.
usage: python b65_verdict.py <root>  -> results/b65_verdict.txt, results/b65_scores.json
"""
import json, math, os, sys
import numpy as np
import b40_data as D

POS = ("last", "ops")
ARMS = ("vsapub", "vsabm", "disc")
SEEDS = (0, 1, 2)
NB = 10000
EQ = 0.02  # the paper's equivalence margin, +/- 2 points


def load(path):
    return [json.loads(l) for l in open(path)]


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def label(lo, hi):
    if lo > 0:
        return "AHEAD"
    if hi < 0:
        return "BEHIND"
    if lo >= -EQ and hi <= EQ:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def twolevel(A, B, seed):
    """Two-level bootstrap of mean(A) - mean(B): seeds of each arm drawn with replacement (independent trainings, not
    paired), then items drawn with replacement once per draw and shared by both arms (the same test items)."""
    rng = np.random.default_rng(seed)
    sa, n = A.shape
    sb = B.shape[0]
    out = np.empty(NB)
    for b in range(NB):
        d = A[rng.integers(0, sa, sa)].mean(0) - B[rng.integers(0, sb, sb)].mean(0)
        out[b] = d[rng.integers(0, n, n)].mean()
    return float(A.mean() - B.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def onelevel(A, B, seed):
    """B40's convention: seed-mean correctness per item, paired bootstrap over items."""
    rng = np.random.default_rng(seed)
    d = A.mean(0) - B.mean(0)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(NB)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t):
    return f"{100 * t[0]:+.2f} [{100 * t[1]:+.2f}, {100 * t[2]:+.2f}]"


def read_exact(r):
    m = r["mod"]
    return m["op"] == r["op"] and m["x"] == r["x"] and m["y"] == r["y"]


def main():
    root = sys.argv[1]
    ch = json.load(open(f"{root}/out/choices.json"))
    P = {(p, a): [load(f"{root}/preds/{p}/{a}_s{s}.jsonl") for s in SEEDS
                  if os.path.exists(f"{root}/preds/{p}/{a}_s{s}.jsonl")] for p in POS for a in ARMS}
    P = {k: v for k, v in P.items() if v}
    ref = next(iter(P.values()))[0]
    ids = [r["id"] for r in ref]
    for k, runs in P.items():  # same items in the same order in every file
        for run in runs:
            assert [r["id"] for r in run] == ids, k
    TR = np.array([r["op"] in D.OPS_TRAIN for r in ref])
    HO = ~TR
    corr = {k: np.array([[float(r["correct"]) for r in run] for run in runs]) for k, runs in P.items()}
    out, js = [], {"choices": ch, "arms": {}, "primary": {}, "secondary": {}}
    pr = out.append
    pr("B65 verdict: task N, trained types (primary, %d items), held-out types (%d items, secondary)" % (TR.sum(), HO.sum()))
    pr("layers: %s" % ch["layer"])
    pr("")
    pr("arm accuracies (seed-mean over items; per seed; Wilson on the seed-mean; read exact / module answer right and gate "
       "open / followed given module right / held-out acc / held-out gate open; first-token CE on trained types)")
    for (p, a), runs in P.items():
        C_ = corr[(p, a)]
        acc = float(C_[:, TR].mean()); seeds = [float(x) for x in C_[:, TR].mean(1)]
        re_ = np.mean([[read_exact(r) for r, t in zip(run, TR) if t] for run in runs])
        modok = np.array([[str(r["mod"]["ans"]) == r["gold"] and r["mod"]["gate"] for r in run] for run in runs])
        fol = float(C_[:, TR][modok[:, TR]].mean())
        hoacc = float(C_[:, HO].mean()) if HO.any() else float("nan")
        hogate = float(np.mean([[r["mod"]["gate"] for r, h in zip(run, HO) if h] for run in runs])) if HO.any() else float("nan")
        ce = float(np.mean([[r["ce"] for r, t in zip(run, TR) if t] for run in runs]))
        lo, hi = wilson(acc, int(TR.sum()))
        js["arms"][f"{p}_{a}"] = {"acc": acc, "seeds": seeds, "wilson": [lo, hi], "read_exact": float(re_),
                                  "module_right_gate_open": float(modok[:, TR].mean()), "followed": fol,
                                  "ho_acc": hoacc, "ho_gate_open": hogate, "ce": ce, "n_seeds": len(runs)}
        pr(f"  {p:4s} {a:6s} acc {acc:.4f} [{lo:.3f}, {hi:.3f}] seeds {[round(s, 4) for s in seeds]} read-exact {re_:.4f} "
           f"module-right {modok[:, TR].mean():.4f} followed {fol:.4f} HO {hoacc:.4f} HO-gate {hogate:.3f} CE {ce:.3f}")
    pr("")
    pr("PRIMARY (decision rule: two-level 95% interval, seeds then items; AHEAD if lower > 0, BEHIND if upper < 0, "
       "EQUIVALENT if it contains 0 and lies within +/-2 points, else INCONCLUSIVE); points:")
    k = 0
    for p in POS:
        for a in ("vsapub", "vsabm"):
            if (p, a) not in P or (p, "disc") not in P:
                pr(f"  {p} {a} - disc: NOT RUN")
                continue
            A, B = corr[(p, a)][:, TR], corr[(p, "disc")][:, TR]
            t2 = twolevel(A, B, 6500 + k); t1 = onelevel(A, B, 6600 + k); k += 1
            sa, sb = A.mean(1), B.mean(1)
            alls = "all seeds A>B" if all(x > y for x, y in zip(sa, sb)) else ("all seeds A<B" if all(x < y for x, y in zip(sa, sb)) else "mixed seeds")
            lab = label(t2[1], t2[2])
            js["primary"][f"{p}_{a}-disc"] = {"diff": t2[0], "two_level": [t2[1], t2[2]], "label": lab,
                                             "one_level_items": [t1[1], t1[2]], "label_one_level": label(t1[1], t1[2]),
                                             "seeds": alls}
            pr(f"  {p:4s} {a} - disc: {fmt(t2)} -> {lab}   (one-level items {fmt(t1)} -> {label(t1[1], t1[2])}; {alls})")
    pr("")
    pr("SECONDARY (same two-level interval and labels; no headline)")
    sec = []
    for p in POS:
        sec.append((f"{p} vsabm - vsapub", (p, "vsabm"), (p, "vsapub")))
    for a in ARMS:
        sec.append((f"{a} ops - last", ("ops", a), ("last", a)))
    best = {}
    for a in ARMS:  # each arm at its dev-best read position (best dev write-back accuracy; ties -> last)
        dv = {p: ch["wb"][f"{p}_{a}"]["dev"][ch["wb"][f"{p}_{a}"]["lr"]] for p in POS if f"{p}_{a}" in ch.get("wb", {})}
        best[a] = max(dv, key=lambda p: (dv[p], p == "last")) if dv else None
    js["secondary"]["dev_best_position"] = best
    for a in ("vsapub", "vsabm"):
        if best[a] and best["disc"]:
            sec.append((f"{a}@{best[a]} - disc@{best['disc']} (each at its dev-best position)", (best[a], a), (best["disc"], "disc")))
    for name, ka, kb in sec:
        if ka not in P or kb not in P:
            pr(f"  {name}: NOT RUN"); continue
        A, B = corr[ka][:, TR], corr[kb][:, TR]
        t2 = twolevel(A, B, 6700 + k); k += 1
        js["secondary"][name] = {"diff": t2[0], "two_level": [t2[1], t2[2]], "label": label(t2[1], t2[2])}
        pr(f"  {name}: {fmt(t2)} -> {label(t2[1], t2[2])}")
    pr("")
    pr("per type (seed-mean accuracy)")
    ops = D.OPS_TRAIN + D.OPS_HELD
    pr("  " + "type".ljust(15) + "".join(f"{p[:4]}-{a}".rjust(13) for (p, a) in P))
    for o in ops:
        m = np.array([r["op"] == o for r in ref])
        pr("  " + o.ljust(15) + "".join(f"{corr[k_][:, m].mean():13.3f}" for k_ in P))
    # plain-loop re-scoring (independent of numpy): pred == gold
    worst = 0.0
    for (p, a), runs in P.items():
        for run, row in zip(runs, corr[(p, a)]):
            n_ok = 0; n = 0
            for r, t in zip(run, TR):
                if t:
                    n += 1
                    if r["pred"].strip() == r["gold"]:
                        n_ok += 1
            worst = max(worst, abs(n_ok / n - row[TR].mean()))
    pr("")
    pr(f"plain-loop scorer (pred == gold) vs numpy 'correct': max |diff| {worst:.2e}")
    js["plain_loop_maxdiff"] = worst
    os.makedirs("results", exist_ok=True)
    open("results/b65_verdict.txt", "w").write("\n".join(out) + "\n")
    json.dump(js, open("results/b65_scores.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
