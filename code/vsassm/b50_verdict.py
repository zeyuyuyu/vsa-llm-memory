"""B50 verdict generator (prereg/B50_ood_gates_cot_budget_prereg.md, 'Bars and readings'). Reads prediction files only.

usage: python b50_verdict.py <b50_root> <b40_pred_root>   -> results/b50_verdict.txt, results/b50_scores.json
  <b50_root>/preds/N/disc<g>_s<s>.jsonl  g in open, p08, msp, energy, maha, mahalw, maxlogit
  <b50_root>/preds/{N,R}/cot2048_cot.jsonl (+ .timing.json), <b50_root>/out/thresholds.json, <b50_root>/logs/gpu_hours.txt
  <b40_pred_root>/{N,R}/{base,cot,vsa_s<s>,disc_s<s>,lora_s<s>,vsaargmax_s<s>}.jsonl  (B40, read-only)
"""
import json, math, os, sys
import numpy as np
import b40_data as D
from b40_verdict import Arm, boot_diff, r1, wilson, fmt, load

SEEDS = [0, 1, 2]
PRIMARY = ["msp", "energy", "maha", "mahalw"]
SECONDARY = ["maxlogit"]


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); pos = trained-type items (in-distribution), neg = held-out."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class Composed:
    """Per-item correctness of a gate applied to the forced-open runs: open -> forced-open correctness, closed ->
    BASE's correctness (B40). gate_fn(seed, item) -> bool."""

    def __init__(self, opens, base, gate_fn):
        self.runs = [{i: {"correct": (o[i]["correct"] if gate_fn(s, i) else base[i]["correct"])} for i in o}
                     for s, o in enumerate(opens)]

    ok = Arm.ok
    acc = Arm.acc
    seed_accs = Arm.seed_accs


def main():
    b50, b40 = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(5050)
    out, js = [], {}
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    rows = {r["id"]: r for r in D.gen("N", "test")}
    TR = [i for i, r in rows.items() if r["op"] in D.OPS_TRAIN]
    HO = [i for i, r in rows.items() if r["op"] in D.OPS_HELD]
    by_op = {op: [i for i, r in rows.items() if r["op"] == op] for op in D.OPS_HELD}
    base = Arm([f"{b40}/N/base.jsonl"])
    vsa = Arm([f"{b40}/N/vsa_s{s}.jsonl" for s in SEEDS])
    disc08 = Arm([f"{b40}/N/disc_s{s}.jsonl" for s in SEEDS])
    vsaam = Arm([f"{b40}/N/vsaargmax_s{s}.jsonl" for s in SEEDS])
    thr = json.load(open(f"{b50}/out/thresholds.json"))["seeds"]
    gfile = lambda g, s: f"{b50}/preds/N/disc{g}_s{s}.jsonl"
    have = lambda g: all(os.path.exists(gfile(g, s)) for s in SEEDS)
    arms = {g: Arm([gfile(g, s) for s in SEEDS]) for g in ["open", "p08"] + PRIMARY + SECONDARY if have(g)}
    tr08 = disc08.acc(TR)
    gate_rate = lambda arm, ids: float(np.mean([[float(r[i]["mod"]["gate"]) for i in ids] for r in arm.runs]))

    P("===================== B50 PART 1: OOD GATES FOR DISC (task N) =====================")
    P(f"trained-type test items {len(TR)}, held-out {len(HO)} (addition {len(by_op['addition'])}, division {len(by_op['division'])})")
    P(f"reference rows (B40): VSA TR {vsa.acc(TR):.4f} HO {vsa.acc(HO):.4f} (add {vsa.acc(by_op['addition']):.4f}, div "
      f"{vsa.acc(by_op['division']):.4f}) gate-open TR {gate_rate(vsa, TR):.4f} HO {gate_rate(vsa, HO):.4f}")
    P(f"                      DISC-0.8 TR {tr08:.4f} HO {disc08.acc(HO):.4f}; BASE TR {base.acc(TR):.4f} HO {base.acc(HO):.4f} "
      f"(add {base.acc(by_op['addition']):.4f}, div {base.acc(by_op['division']):.4f})")
    P("  dev matching: per seed, VSA dev open count k_s and DISC thresholds (open iff score >= tau):")
    for s in SEEDS:
        e = thr[str(s)]
        P(f"     seed {s}: k_s {e['vsa_dev_open']}/800; " + "; ".join(f"{g} {e['tau'][g]:+.5g} (dev {e['dev_open_at_tau'][g]})"
                                                                   for g in PRIMARY + SECONDARY))
    P("  per gate (seed-mean accuracy; gate-open rates on test):")
    P("     gate       TR      HO     add     div   open-TR  open-add  open-div  seeds-HO")
    res = {}
    for g in ["p08", "open"] + PRIMARY + SECONDARY:
        if g not in arms:
            P(f"     {g:8s} MISSING")
            continue
        a = arms[g]
        res[g] = {"TR": a.acc(TR), "HO": a.acc(HO), "add": a.acc(by_op["addition"]), "div": a.acc(by_op["division"]),
                  "open_TR": gate_rate(a, TR), "open_add": gate_rate(a, by_op["addition"]),
                  "open_div": gate_rate(a, by_op["division"]), "seeds_HO": a.seed_accs(HO), "seeds_TR": a.seed_accs(TR)}
        r = res[g]
        P(f"     {g:8s} {r['TR']:.4f}  {r['HO']:.4f}  {r['add']:.4f}  {r['div']:.4f}   {r['open_TR']:.4f}    {r['open_add']:.4f}    "
          f"{r['open_div']:.4f}   {[round(x, 4) for x in r['seeds_HO']]}")
    js["part1"] = {"gates": res, "vsa": {"TR": vsa.acc(TR), "HO": vsa.acc(HO)}, "disc08_TR": tr08}
    # ---------------- primary bar
    elig = {g: res[g]["TR"] >= tr08 - 0.02 for g in PRIMARY if g in res}
    elig_alt = {g: res[g]["TR"] >= vsa.acc(TR) - 0.02 for g in PRIMARY if g in res}
    P(f"  eligibility (TR >= TR(DISC-0.8) - 0.02 = {tr08 - 0.02:.4f}): {elig}")
    P(f"  (alternative reading, TR >= TR(VSA) - 0.02 = {vsa.acc(TR) - 0.02:.4f}: {elig_alt})")
    el = [g for g in PRIMARY if elig.get(g)]
    if len(elig) < len(PRIMARY):
        P("  PRIMARY BAR NOT EVALUATED: missing gate files")
    elif not el:
        P("  B50-1 = NO ELIGIBLE GATE")
        js["part1"]["B50_1"] = {"label": "NO ELIGIBLE GATE"}
    else:
        best = max(el, key=lambda g: res[g]["HO"])
        d, lo, hi = boot_diff(vsa, arms[best], HO, rng)
        sv, sb = vsa.seed_accs(HO), arms[best].seed_accs(HO)
        allseeds = all(a_ - b_ >= 0.10 for a_, b_ in zip(sv, sb))
        if d >= 0.10 and lo > 0:
            lab = "SURVIVES"
        elif hi < 0:
            lab = "DOES NOT SURVIVE (DISC-best ahead)"
        elif lo > 0:
            lab = "DOES NOT SURVIVE (VSA ahead by less than 0.10)"
        else:
            lab = "DOES NOT SURVIVE (no VSA advantage)"
        P(f"  DISC-best = {best} (HO {res[best]['HO']:.4f}, TR {res[best]['TR']:.4f})")
        P(f"  B50-1: HO(VSA) - HO(DISC-{best}) = {fmt((d, lo, hi))}; seeds VSA {[round(x, 4) for x in sv]} "
          f"DISC-{best} {[round(x, 4) for x in sb]}; per-seed >= +0.10: {allseeds} -> {lab}")
        tdiff = boot_diff(vsa, arms[best], TR, rng)
        P(f"  trained types: TR(VSA) - TR(DISC-{best}) = {fmt(tdiff)} (alternative reading of 'within 0.02')")
        js["part1"]["B50_1"] = {"label": lab, "best": best, "delta": [d, lo, hi], "seeds_vsa": sv, "seeds_best": sb,
                                "tr_diff": list(tdiff)}
        for g in PRIMARY + SECONDARY:
            if g in arms:
                t = boot_diff(vsa, arms[g], HO, rng)
                ta = boot_diff(vsaam, arms[g], HO, rng)
                P(f"     HO(VSA) - HO(DISC-{g:8s}) = {fmt(t)};  exploratory HO(VSA-argmax) - HO(DISC-{g}) = {fmt(ta)}")
                js["part1"].setdefault("deltas", {})[g] = {"vsa": list(t), "vsa_argmax": list(ta)}
    # ---------------- S1 hurts per held-out type
    P("  S1 held-out types: arm - BASE [CI] (HURTS if <= -0.05 and CI upper < 0):")
    for op in D.OPS_HELD:
        for name, a in [("vsa", vsa), ("disc-0.8", disc08)] + [(f"disc-{g}", arms[g]) for g in PRIMARY + SECONDARY if g in arms]:
            t = boot_diff(a, base, by_op[op], rng)
            P(f"     {op:9s} {name:14s} {fmt(t)}{' HURTS' if t[0] <= -0.05 and t[2] < 0 else ''}")
    # ---------------- S2 AUROC and matched false-open operating point
    if "open" in arms:
        P("  S2 AUROC (trained-type vs held-out test items; higher score = in-distribution), per seed:")
        opens = arms["open"].runs
        au = {}
        for g in PRIMARY + SECONDARY:
            au[g] = [float(auroc([o[i]["mod"][g] for i in TR], [o[i]["mod"][g] for i in HO])) for o in opens]
        au["vsa_op_s"] = [float(auroc([r[i]["mod"]["op_s"] for i in TR], [r[i]["mod"]["op_s"] for i in HO])) for r in vsa.runs]
        for k, v in au.items():
            P(f"     {k:9s} {[round(x, 4) for x in v]}")
        js["part1"]["auroc"] = au
        P("  S2b 'matched false-open' operating point (threshold set on TEST held-out scores so that DISC's held-out open")
        P("      count equals the VSA seed's; a description of the ROC, not a usable gate):")
        js["part1"]["matched_fo"] = {}
        b0 = base.runs[0]
        for g in PRIMARY + SECONDARY:
            taus = []
            for s, o in enumerate(opens):
                m = sum(vsa.runs[s][i]["mod"]["gate"] for i in HO)
                hs = np.sort([o[i]["mod"][g] for i in HO])[::-1]
                taus.append(hs[m - 1] if m > 0 else np.nextafter(hs[0], np.inf))
            comp = Composed(opens, b0, lambda s, i, g=g, taus=taus: opens[s][i]["mod"][g] >= taus[s])
            otr = np.mean([[float(opens[s][i]["mod"][g] >= taus[s]) for i in TR] for s in SEEDS])
            oho = np.mean([[float(opens[s][i]["mod"][g] >= taus[s]) for i in HO] for s in SEEDS])
            P(f"     {g:8s} open-TR {otr:.4f} open-HO {oho:.4f}  composed TR {comp.acc(TR):.4f} HO {comp.acc(HO):.4f}")
            js["part1"]["matched_fo"][g] = {"open_TR": float(otr), "open_HO": float(oho), "TR": comp.acc(TR), "HO": comp.acc(HO)}
        # ---------------- S3 sensitivity (dev open rates 0.99, 0.95), composed
        P("  S3 sensitivity: thresholds at dev open rates 0.99 / 0.95 (composed from the forced-open runs and BASE):")
        js["part1"]["sens"] = {}
        for rate in ("0.99", "0.95"):
            for g in PRIMARY + SECONDARY:
                tt = [thr[str(s)]["tau_sens"][g][rate] for s in SEEDS]
                comp = Composed(opens, b0, lambda s, i, g=g, tt=tt: opens[s][i]["mod"][g] >= tt[s])
                otr = np.mean([[float(opens[s][i]["mod"][g] >= tt[s]) for i in TR] for s in SEEDS])
                oho = np.mean([[float(opens[s][i]["mod"][g] >= tt[s]) for i in HO] for s in SEEDS])
                d = boot_diff(vsa, comp, HO, rng)
                P(f"     dev-rate {rate} {g:8s} open-TR {otr:.4f} open-HO {oho:.4f} TR {comp.acc(TR):.4f} HO {comp.acc(HO):.4f} "
                  f"HO(VSA)-HO {fmt(d)}")
                js["part1"]["sens"][f"{rate}_{g}"] = {"open_TR": float(otr), "open_HO": float(oho), "TR": comp.acc(TR),
                                                       "HO": comp.acc(HO), "delta": list(d)}
        # ---------------- checks: gate re-derivation, composition, B40 reproduction
        P("  CHECKS:")
        for g in PRIMARY + SECONDARY:
            if g not in arms:
                continue
            mis = sum(int(r[i]["mod"]["gate"] != (r[i]["mod"][g] >= thr[str(s)]["tau"][g])) for s, r in enumerate(arms[g].runs) for i in r)
            comp = sum(int(r[i]["correct"] != (opens[s][i]["correct"] if r[i]["mod"]["gate"] else b0[i]["correct"]))
                       for s, r in enumerate(arms[g].runs) for i in r)
            sc = max(abs(r[i]["mod"][g] - opens[s][i]["mod"][g]) for s, r in enumerate(arms[g].runs) for i in r)
            P(f"     {g:8s} gate re-derivation mismatches {mis}/7500; composition mismatches {comp}/7500; "
              f"max |score - forced-open score| {sc:.3g}")
            js["part1"].setdefault("checks", {})[g] = {"gate_mismatch": mis, "composition_mismatch": comp, "score_maxdiff": sc}
        if "p08" in arms:
            dp = sum(int(r[i]["pred"] != disc08.runs[s][i]["pred"]) for s, r in enumerate(arms["p08"].runs) for i in r)
            dg = sum(int(r[i]["mod"]["gate"] != disc08.runs[s][i]["mod"]["gate"]) for s, r in enumerate(arms["p08"].runs) for i in r)
            P(f"     p08 vs B40 DISC: pred differs on {dp}/7500, gate differs on {dg}/7500")
            js["part1"].setdefault("checks", {})["p08_vs_b40"] = {"pred": dp, "gate": dg}
            comp = sum(int(r[i]["correct"] != (opens[s][i]["correct"] if r[i]["mod"]["gate"] else b0[i]["correct"]))
                       for s, r in enumerate(arms["p08"].runs) for i in r)
            P(f"     p08 composition mismatches {comp}/7500")

    # ======================================================================= PART 2: CoT 2048
    P("===================== B50 PART 2: CoT WITH A 2,048-TOKEN BUDGET =====================")
    js["part2"] = {}
    for task in ("N", "R"):
        f = f"{b50}/preds/{task}/cot2048_cot.jsonl"
        if not os.path.exists(f):
            P(f"  task {task}: MISSING")
            continue
        rws = {r["id"]: r for r in D.gen(task, "test")}
        prim = [i for i, r in rws.items() if r["op"] in D.OPS_TRAIN] if task == "N" else list(rws)
        c2 = Arm([f]); c5 = Arm([f"{b40}/{task}/cot.jsonl"])
        v = Arm([f"{b40}/{task}/vsa_s{s}.jsonl" for s in SEEDS])
        lora = Arm([f"{b40}/{task}/lora_s{s}.jsonl" for s in SEEDS])
        bse = Arm([f"{b40}/{task}/base.jsonl"])
        r2048, r512 = c2.runs[0], c5.runs[0]
        n = len(prim)
        k = sum(r2048[i]["correct"] for i in prim)
        lo, hi = wilson(k, n)
        hit = [i for i in r2048 if r2048[i]["hit_limit"]]
        hitp = [i for i in prim if r2048[i]["hit_limit"]]
        ub = (k + sum(1 for i in hitp if not r2048[i]["correct"])) / n
        P(f"  task {task}: COT2048 {k / n:.4f} Wilson [{lo:.3f}, {hi:.3f}] (B40 COT512 {c5.acc(prim):.4f}) on {n} primary items")
        P(f"     still at 2,048 tokens without an end token: {len(hitp)}/{n} primary ({len(hit)}/{len(r2048)} all); "
          f"upper bound if all counted correct {ub:.4f}")
        trunc512 = [i for i in r512 if "final answer" not in r512[i]["out"].lower()]
        P(f"     B40 outputs without 'Final Answer' at 512: {len(trunc512)}; of these, now ending before 2,048: "
          f"{sum(1 for i in trunc512 if not r2048[i]['hit_limit'])}, now correct: {sum(1 for i in trunc512 if r2048[i]['correct'])}")
        ng = np.array([r2048[i]["n_gen"] for i in r2048])
        P(f"     generated tokens: mean {ng.mean():.1f} median {np.median(ng):.0f} p95 {np.percentile(ng, 95):.0f} max {ng.max()} "
          f"total {ng.sum()}")
        tm = json.load(open(f + ".timing.json"))
        P(f"     generation wall time {tm['gen_s'] / 3600:.3f} GPU-h on {tm['gpu']} ({tm['gen_s'] / len(r2048):.2f} s/item); "
          f"module arms: 6 new tokens")
        P("  R1 re-read with COT2048 (VSA - X >= +0.10, CI > 0, all seeds):")
        r1s = {}
        for name, x in (("base", bse), ("cot2048", c2), ("lora", lora)):
            ok, t, alls = r1(v, x, prim, rng)
            r1s[name] = ok
            P(f"     vs {name:8s}: {fmt(t)} all seeds {alls} -> {'PASS' if ok else 'fail'}")
        npass = sum(r1s.values())
        P(f"  R1 = {'REPLICATED' if npass == 3 else 'PARTLY' if npass else 'NOT REPLICATED'}")
        cells = ([(op, [i for i, r in rws.items() if r["op"] == op]) for op in D.OPS_TRAIN + D.OPS_HELD] if task == "N" else
                 [(f"{s}-d{dd}", [i for i, r in rws.items() if r["size"] == s and r["depth"] == dd])
                  for s in ("small", "large") for dd in (1, 2, 3, 4, 5)])
        P("     cell            cot512  cot2048  at-2048")
        cj = {}
        for c, ids in cells:
            cj[c] = {"cot512": c5.acc(ids), "cot2048": c2.acc(ids), "hit": sum(r2048[i]["hit_limit"] for i in ids)}
            P(f"     {c:14s} {cj[c]['cot512']:.3f}   {cj[c]['cot2048']:.3f}    {cj[c]['hit']}")
        # reproducibility vs B40's 512-token run
        short = [i for i in r2048 if r2048[i]["n_gen"] < 512]
        same = sum(1 for i in short if r2048[i]["out"] == r512[i]["out"])
        longi = [i for i in r2048 if r2048[i]["n_gen"] >= 512]
        pref = sum(1 for i in longi if r2048[i]["out"].startswith(r512[i]["out"][:max(0, len(r512[i]["out"]) - 8)]))
        P(f"     reproducibility: items ending before 512 tokens with text identical to B40: {same}/{len(short)}; "
          f"longer items whose text starts with B40's 512-token text: {pref}/{len(longi)}")
        js["part2"][task] = {"acc": k / n, "wilson": [lo, hi], "cot512": c5.acc(prim), "hit_primary": len(hitp),
                             "hit_all": len(hit), "ub": ub, "tokens_mean": float(ng.mean()), "tokens_total": int(ng.sum()),
                             "gen_gpu_h": tm["gen_s"] / 3600, "R1": r1s, "cells": cj, "repro_same": [same, len(short)],
                             "repro_prefix": [pref, len(longi)],
                             "heldout": ({op: c2.acc([i for i, r in rws.items() if r["op"] == op]) for op in D.OPS_HELD}
                                         if task == "N" else None)}
    # ---------------- compute
    gh = 0.0
    if os.path.exists(f"{b50}/logs/gpu_hours.txt"):
        gh = sum(json.loads(l)["gpu_h"] for l in open(f"{b50}/logs/gpu_hours.txt"))
    P(f"===================== COMPUTE: {gh:.2f} GPU-hours (process wall time, logs/gpu_hours.txt) =====================")
    js["gpu_h"] = gh
    os.makedirs("results", exist_ok=True)
    open("results/b50_verdict.txt", "w").write("\n".join(out) + "\n")
    json.dump(js, open("results/b50_scores.json", "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    main()
