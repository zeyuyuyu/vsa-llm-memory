"""B41 verdict (prereg/B41_superposed_uncertain_multihop_prereg.md). Reads the engine runs of one split, the dev
choice file and writes results/b41_<split>_verdict.txt and results/b41_<split>_scores.json.

Cell = (rho level, h, N). Per cell the queries of all seeds are pooled (bootstrap resamples queries within seed).
  A-gap(v; metric)   acc(v) - max{acc(t): t a table config with mean cost(t) <= mean cost(v)}   (WIN-ACC if CI lo > 0)
  A-gap10(v; metric) acc(v) - max{acc(t): mean cost(t) <= 10 * mean cost(v)}                     (WIN-COST if CI lo > -0.01)
  C-ratio(v)         mean ops of the cheapest table config with acc >= acc(v) - 0.01, divided by ops(v)
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_engine as EN                                                     # noqa: E402

NS = [32, 320, 3200, 32000, 100000]
HS = [2, 4, 6]
METRICS = ["ops", "traffic", "state"]
TABLES = ["ARGMAX"] + [EN.name_table(k, b) for (_, k, b) in EN.table_configs() if not (k == 1 and b == 1)] + \
         ["EXACT"] + ["ENUM_k%d" % k for k in EN.ENUM_K]
VSAS = ["VSA_%s_D%d_kb%d" % (m, D, kb) for D in EN.DS for kb in EN.KB for m in EN.MODES]
NB = 2000


def load_cell(root, split, rho, N, seeds):
    """{config: {'correct': (n,3) bool, 'ops','traffic','state': (n,3)}}, seed index array, meta list."""
    parts, metas = [], []
    for s in seeds:
        p = os.path.join(root, "%s_s%d_N%d_r%.3f.npz" % (split, s, N, rho))
        if not os.path.exists(p):
            return None, None, None
        parts.append(np.load(p)); metas.append(json.load(open(p.replace(".npz", ".json"))))
    out = {}
    for name in TABLES + VSAS:
        if "%s|correct" % name not in parts[0].files:
            continue
        out[name] = {k: np.concatenate([z["%s|%s" % (name, k)] for z in parts]) for k in ["correct"] + METRICS}
    sid = np.concatenate([np.full(len(z["_answers"]), i) for i, z in enumerate(parts)])
    out["_decoy"] = np.concatenate([z["_decoy"] for z in parts])
    return out, sid, metas


def boot_idx(sid, rng):
    idx = []
    for s in np.unique(sid):
        w = np.nonzero(sid == s)[0]
        idx.append(rng.choice(w, len(w), replace=True))
    return np.concatenate(idx)


def gap_stats(cell, sid, v, hi, metric, factor, rng, B=NB):
    cv = cell[v][metric][:, hi].mean()
    elig = [t for t in TABLES if t in cell and cell[t][metric][:, hi].mean() <= factor * cv]
    av = cell[v]["correct"][:, hi]
    if not elig:
        return {"point": float(av.mean()), "lo": float("nan"), "hi": float("nan"), "best_table": None,
                "n_eligible": 0}
    tc = np.stack([cell[t]["correct"][:, hi] for t in elig], 1).astype(float)
    point = av.mean() - tc.mean(0).max()
    bs = []
    for _ in range(B):
        i = boot_idx(sid, rng)
        bs.append(av[i].mean() - tc[i].mean(0).max())
    lo, hi_ = np.percentile(bs, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi_),
            "best_table": elig[int(tc.mean(0).argmax())], "n_eligible": len(elig)}


def acc_ci(c, sid, rng, B=NB):
    bs = [c[boot_idx(sid, rng)].mean() for _ in range(B)]
    return float(c.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def choose_dev(cell, sid, hi, rng):
    """Dev rule: v* = argmax dev A-gap under ops (ties: higher accuracy, then lower ops); also v_acc = most accurate."""
    best, vacc = None, None
    for v in VSAS:
        if v not in cell:
            continue
        cv = cell[v]["ops"][:, hi].mean()
        av = cell[v]["correct"][:, hi].mean()
        elig = [t for t in TABLES if cell[t]["ops"][:, hi].mean() <= cv]
        g = av - max(cell[t]["correct"][:, hi].mean() for t in elig) if elig else av
        key = (round(g, 6), round(av, 6), -cv)
        if best is None or key > best[0]:
            best = (key, v)
        k2 = (round(av, 6), -cv)
        if vacc is None or k2 > vacc[0]:
            vacc = (k2, v)
    return best[1], vacc[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rhos", type=float, nargs="+", required=True)
    ap.add_argument("--choice", default=None, help="dev choice json (test only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(41)
    levels = dict(zip(["low", "mid", "high"], a.rhos))
    choice = json.load(open(a.choice)) if a.choice else {}
    scores, lines, dev_choice = {}, [], {}
    for lv, rho in levels.items():
        for N in NS:
            cell, sid, metas = load_cell(a.root, a.split, rho, N, a.seeds)
            if cell is None:
                lines.append("MISSING %s N=%d" % (lv, N)); continue
            err = float(np.mean([m["diag"]["mention_argmax_error"] for m in metas]))
            for hi, h in enumerate(HS):
                key = "%s|h%d|N%d" % (lv, h, N)
                r = {"rho": rho, "mention_argmax_error": err, "n": int(len(sid)), "acc": {}, "cost": {}}
                for name in TABLES + VSAS:
                    if name in cell:
                        r["acc"][name] = acc_ci(cell[name]["correct"][:, hi].astype(float), sid, rng, B=200) \
                            if name in ("ARGMAX", "BEAM_k4_B16", "BEAM_k16_Binf", "EXACT") else \
                            (float(cell[name]["correct"][:, hi].mean()),)
                        r["cost"][name] = {m: float(cell[name][m][:, hi].mean()) for m in METRICS}
                vstar, vacc = choose_dev(cell, sid, hi, rng)
                if a.split == "dev":
                    dev_choice[key] = {"v_star": vstar, "v_acc": vacc}
                    r["v_star"], r["v_acc"] = vstar, vacc
                else:
                    vstar, vacc = choice[key]["v_star"], choice[key]["v_acc"]
                    r["v_star"], r["v_acc"] = vstar, vacc
                r["v_star_acc"] = acc_ci(cell[vstar]["correct"][:, hi].astype(float), sid, rng)
                r["v_acc_acc"] = acc_ci(cell[vacc]["correct"][:, hi].astype(float), sid, rng)
                for m in METRICS:
                    r["gap_%s" % m] = gap_stats(cell, sid, vstar, hi, m, 1.0, rng)
                    r["gap10_%s" % m] = gap_stats(cell, sid, vstar, hi, m, 10.0, rng)
                    r["vacc_gap_%s" % m] = gap_stats(cell, sid, vacc, hi, m, 1.0, rng, B=500)
                av = cell[vstar]["correct"][:, hi].mean()
                cheap = [cell[t]["ops"][:, hi].mean() for t in TABLES if cell[t]["correct"][:, hi].mean() >= av - 0.01]
                r["c_ratio_ops"] = float(min(cheap) / cell[vstar]["ops"][:, hi].mean()) if cheap else None
                verdicts = {}
                for m in METRICS:
                    win_acc = r["gap_%s" % m]["lo"] > 0
                    win_cost = r["gap10_%s" % m]["lo"] > -0.01
                    verdicts[m] = "VSA-WIN-ACC" if win_acc else ("VSA-WIN-COST" if win_cost else "TABLE")
                r["verdict"] = verdicts
                # value of carrying uncertainty and how much of it the VSA recovers, per D
                aA, aX = cell["ARGMAX"]["correct"][:, hi].mean(), cell["EXACT"]["correct"][:, hi].mean()
                r["recovery_by_D"] = {}
                for D in EN.DS:
                    vs = [v for v in VSAS if ("_D%d_" % D) in v and v in cell]
                    bv = max(vs, key=lambda v: cell[v]["correct"][:, hi].mean())
                    ab = cell[bv]["correct"][:, hi].mean()
                    r["recovery_by_D"][D] = {"best": bv, "acc": float(ab),
                                             "recovery": float((ab - aA) / (aX - aA)) if aX - aA > 0.02 else None}
                r["decoy_acc"] = {nm: [float(cell[nm]["correct"][cell["_decoy"], hi].mean()),
                                       float(cell[nm]["correct"][~cell["_decoy"], hi].mean())]
                                  for nm in ("ARGMAX", "BEAM_k4_B16", "EXACT", vstar)}
                scores[key] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(scores, open(a.out.replace(".txt", ".json"), "w"), indent=1)
    if a.split == "dev":
        json.dump(dev_choice, open(a.out.replace("_verdict.txt", "_choice.json"), "w"), indent=1)
    # ------------------------------------------------------------ text
    L = []
    L.append("# B41 %s verdict (b41_verdict.py). Cells pooled over seeds %s; accuracies with 95%% bootstrap CIs "
             "(queries resampled within seed)." % (a.split, a.seeds))
    L.append("# ops = per-query operations; traffic = 32-bit words read; state = words carried between hops.")
    for lv in levels:
        L.append("")
        L.append("## rho level %s (rho = %.3f)" % (lv, levels[lv]))
        for h in HS:
            L.append("  h = %d" % h)
            L.append("    %-7s %-6s %-22s %-22s %-22s %-22s | %-30s %-22s %s" % (
                "N", "err", "ARGMAX acc/ops", "BEAM(4,16) acc/ops", "BEAM(16,inf) acc/ops", "EXACT acc/ops",
                "VSA v* acc [CI] / ops", "A-gap ops [CI]", "verdict ops/traffic/state"))
            for N in NS:
                key = "%s|h%d|N%d" % (lv, h, N)
                if key not in scores:
                    continue
                r = scores[key]
                f = lambda nm: "%.3f/%.1e" % (r["acc"][nm][0], r["cost"][nm]["ops"])
                g = r["gap_ops"]
                L.append("    %-7d %-6.3f %-22s %-22s %-22s %-22s | %-30s %-22s %s" % (
                    N, r["mention_argmax_error"], f("ARGMAX"), f("BEAM_k4_B16"), f("BEAM_k16_Binf"), f("EXACT"),
                    "%.3f [%.3f,%.3f] / %.1e" % (r["v_star_acc"][0], r["v_star_acc"][1], r["v_star_acc"][2],
                                                 r["cost"][r["v_star"]]["ops"]),
                    "%+.3f [%+.3f,%+.3f]" % (g["point"], g["lo"], g["hi"]),
                    "/".join(r["verdict"][m] for m in METRICS)))
                L.append("            v* = %s; most accurate VSA %s %.3f; C-ratio %s; recovery by D %s" % (
                    r["v_star"], r["v_acc"], r["v_acc_acc"][0],
                    "%.1e" % r["c_ratio_ops"] if r["c_ratio_ops"] else "none",
                    ", ".join("%d:%s" % (D, "%.2f" % x["recovery"] if x["recovery"] is not None else "-")
                              for D, x in r["recovery_by_D"].items())))
    prim = [k for k in scores if k.startswith("high|") and ("|h4|" in k or "|h6|" in k)]
    L.append("")
    L.append("## Primary regime (rho high, h in {4, 6}, every N): %d cells" % len(prim))
    for m in METRICS:
        wins = [k for k in prim if scores[k]["verdict"][m] != "TABLE"]
        L.append("  %-8s VSA-WIN cells: %s" % (m, wins if wins else "none"))
    allw = {m: [k for k in scores if scores[k]["verdict"][m] != "TABLE"] for m in METRICS}
    L.append("## All cells: VSA-WIN under ops %s; traffic %s; state %s" % (allw["ops"] or "none",
                                                                          allw["traffic"] or "none",
                                                                          allw["state"] or "none"))
    any_prim = any(scores[k]["verdict"]["ops"] != "TABLE" for k in prim)
    L.append("## PRIMARY VERDICT: %s" % ("VSA REGIME FOUND" if any_prim else "TABLE DOMINATES"))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
