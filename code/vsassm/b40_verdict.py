"""B40 verdict generator (prereg/B40_inforward_vsa_prereg.md, 'Bars and readings'). Reads prediction files only.

Layout: <pred_root>/<task>/<name>.jsonl with name in
  base, cot, <arm>_s<seed> (arm in vsa, disc, lora, lora1l, e2e_vsa, e2e_tab, e2e_const),
  <arm>_s<seed>_rho<r> (noise), vsasoft_s<seed>[_rho<r>], <arm>_s<seed>_shuf (pairings destroyed)
usage: python b40_verdict.py <pred_root> [<train_root for param counts>]  -> results/b40_verdict.txt, b40_scores.json
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
import b40_data as D

SEEDS = [0, 1, 2]
B = 10000
RHOS = ["0.25", "0.5", "1.0", "2.0", "4.0", "8.0"]


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


class Arm:
    def __init__(self, files):
        self.runs = [load(f) for f in files]

    def ok(self, ids):  # [n_seeds, n_items]
        return np.array([[float(r[i]["correct"]) for i in ids] for r in self.runs])

    def acc(self, ids):
        return float(self.ok(ids).mean(0).mean()) if self.runs else float("nan")

    def seed_accs(self, ids):
        return [float(x) for x in self.ok(ids).mean(1)]


def boot_diff(a, b, ids, rng):
    da, db = a.ok(ids).mean(0), b.ok(ids).mean(0)
    d = da - db
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def r1(vsa, x, ids, rng):
    d, lo, hi = boot_diff(vsa, x, ids, rng)
    sv, sx = vsa.seed_accs(ids), x.seed_accs(ids)
    allseeds = all(v > sx[0] for v in sv) if len(sx) == 1 else all(a > b for a, b in zip(sv, sx))
    ok = d >= 0.10 and lo > 0 and allseeds
    return ok, (d, lo, hi), allseeds


def r2(vsa, disc, ids, rng):
    d, lo, hi = boot_diff(vsa, disc, ids, rng)
    sv, sd = vsa.seed_accs(ids), disc.seed_accs(ids)
    win = d >= 0.03 and lo > 0 and all(a > b for a, b in zip(sv, sd))
    lose = d <= -0.03 and hi < 0 and all(a < b for a, b in zip(sv, sd))
    tie = abs(d) <= 0.02 and lo >= -0.03 and hi <= 0.03
    lab = "VSA WIN" if win else "DISC WIN" if lose else "TIE" if tie else "UNRESOLVED"
    return lab, (d, lo, hi), sv, sd


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main():
    root = sys.argv[1]
    rng = np.random.default_rng(4040)
    out, js = [], {}
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    for task in ("N", "R"):
        d = f"{root}/{task}"
        if not os.path.isdir(d):
            continue
        rows = {r["id"]: r for r in D.gen(task, "test")}
        if task == "N":
            prim = [i for i, r in rows.items() if r["op"] in D.OPS_TRAIN]
        else:
            prim = list(rows)
        arms = {}
        for a in ("vsa", "disc", "lora", "lora1l", "e2e_vsa", "e2e_tab", "e2e_const", "e2e3_vsa", "e2e3_tab", "e2e3_const", "vsasoft"):
            fs = [f"{d}/{a}_s{s}.jsonl" for s in SEEDS if os.path.exists(f"{d}/{a}_s{s}.jsonl")]
            if fs:
                arms[a] = Arm(fs)
        for a in ("base", "cot"):
            if os.path.exists(f"{d}/{a}.jsonl"):
                arms[a] = Arm([f"{d}/{a}.jsonl"])
        P(f"===================== TASK {task} =====================")
        P(f"primary set: {len(prim)} items; runs per arm:", {k: len(v.runs) for k, v in arms.items()})
        res = {}
        for k, v in arms.items():
            if k.startswith("e2e"):
                continue
            ids = [i for i in prim if all(i in r for r in v.runs)]
            if len(ids) != len(prim):
                P(f"  WARNING {k}: {len(ids)}/{len(prim)} primary items present")
            res[k] = {"acc": v.acc(prim), "seeds": v.seed_accs(prim)}
            n = len(prim)
            lo, hi = wilson(round(res[k]["acc"] * n), n)
            P(f"  {k:8s} acc {res[k]['acc']:.4f}  Wilson [{lo:.3f}, {hi:.3f}]  seeds {[round(x, 4) for x in res[k]['seeds']]}")
        js[task] = {"acc": res}
        if "base" in res:
            P(f"  context: LLM alone fails (BASE <= 0.5): {res['base']['acc'] <= 0.5}")
        if "vsa" in arms:
            P("  R1 (VSA - X >= +0.10, CI > 0, all seeds):")
            r1s = {}
            for x in ("base", "cot", "lora"):
                if x in arms:
                    ok, t, alls = r1(arms["vsa"], arms[x], prim, rng)
                    r1s[x] = ok
                    P(f"     vs {x:5s}: {fmt(t)}  all seeds {alls}  -> {'PASS' if ok else 'fail'}")
            npass = sum(r1s.values())
            lab = "REPLICATED" if npass == 3 else "PARTLY" if npass else "NOT REPLICATED"
            P(f"  R1 = {lab}")
            js[task]["R1"] = {"label": lab, **{k: bool(v) for k, v in r1s.items()}}
        if "vsa" in arms and "disc" in arms:
            lab, t, sv, sd = r2(arms["vsa"], arms["disc"], prim, rng)
            P(f"  R2 VSA - DISC = {fmt(t)}; seeds VSA {[round(x, 4) for x in sv]} DISC {[round(x, 4) for x in sd]} -> {lab}")
            js[task]["R2"] = {"label": lab, "delta": t}
        # ---------------- S1 per type / cell
        P("  S1 per type/cell (seed-mean accuracy):")
        if task == "N":
            cells = [(op, [i for i, r in rows.items() if r["op"] == op]) for op in D.OPS_TRAIN + D.OPS_HELD]
        else:
            cells = [(f"{s}-d{dd}", [i for i, r in rows.items() if r["size"] == s and r["depth"] == dd])
                     for s in ("small", "large") for dd in (1, 2, 3, 4, 5)]
        names = [k for k in ("base", "cot", "lora", "lora1l", "vsa", "disc") if k in arms]
        P("     " + "cell".ljust(14) + "".join(n.rjust(9) for n in names))
        js[task]["cells"] = {}
        for c, ids in cells:
            vals = [arms[n].acc(ids) for n in names]
            js[task]["cells"][c] = dict(zip(names, vals))
            P("     " + c.ljust(14) + "".join(f"{v:9.3f}" for v in vals))
        # ---------------- S2 held-out (task N)
        if task == "N":
            P("  S2 held-out types: arm - BASE [CI]; gate-open rate")
            for op in D.OPS_HELD:
                ids = [i for i, r in rows.items() if r["op"] == op]
                for n in ("vsa", "disc", "lora", "lora1l"):
                    if n in arms and "base" in arms:
                        t = boot_diff(arms[n], arms["base"], ids, rng)
                        hurt = t[0] <= -0.05 and t[2] < 0
                        gate = ""
                        if n in ("vsa", "disc"):
                            g = np.mean([[float(r[i]["mod"]["gate"]) for i in ids] for r in arms[n].runs])
                            gate = f" gate-open {g:.3f}"
                        P(f"     {op:9s} {n:6s} {fmt(t)}{' HURTS' if hurt else ''}{gate}")
            for n in ("vsa", "disc"):
                if n in arms:
                    g = np.mean([[float(r[i]["mod"]["gate"]) for i in prim] for r in arms[n].runs])
                    P(f"     trained types gate-open {n}: {g:.4f}")
        # ---------------- S3 reads and CE
        P("  S3 module reads / module answer / first-token CE:")
        for n in ("vsa", "disc", "vsasoft"):
            if n not in arms:
                continue
            ra, ma = [], []
            for r in arms[n].runs:
                for i in prim:
                    m, g = r[i]["mod"], rows[i]
                    if task == "N":
                        ra.append(float(m["op"] == g["op"] and m["x"] == g["x"] and m["y"] == g["y"]))
                        ma.append(float(m["ans"] == g["ans"]))
                    else:
                        ra.append(float(m["reads_ok"] == 1.0))
                        ma.append(float(m["ans"] == g["ans"]))
            P(f"     {n:7s} read-exact {np.mean(ra):.4f}  module-answer-correct {np.mean(ma):.4f}")
            js[task].setdefault("reads", {})[n] = {"read_exact": float(np.mean(ra)), "module_answer": float(np.mean(ma))}
        for n in ("base", "lora", "lora1l", "vsa", "disc"):
            if n in arms:
                ce = np.mean([[r[i]["ce"] for i in prim] for r in arms[n].runs])
                P(f"     CE {n:7s} {ce:.4f}")
                js[task].setdefault("ce", {})[n] = float(ce)
        if "vsa" in js[task].get("ce", {}) and "base" in js[task].get("ce", {}):
            P(f"     VSA CE reduction vs BASE: {1 - js[task]['ce']['vsa'] / js[task]['ce']['base']:.3f}")
        # ---------------- S4 noise
        P("  S4 noisy reads (rho = noise norm / state norm), seed-mean accuracy; R2 bars VSA - DISC:")
        js[task]["noise"] = {}
        for rho in RHOS:
            line = {}
            for n in ("vsa", "disc", "vsasoft"):
                fs = [f"{d}/{n}_s{s}_rho{rho}.jsonl" for s in SEEDS if os.path.exists(f"{d}/{n}_s{s}_rho{rho}.jsonl")]
                if fs:
                    line[n] = Arm(fs)
            if "vsa" in line and "disc" in line:
                lab, t, sv, sd = r2(line["vsa"], line["disc"], prim, rng)
                extra = ""
                if "vsasoft" in line:
                    lab2, t2, _, _ = r2(line["vsasoft"], line["disc"], prim, rng)
                    extra = f"; soft {line['vsasoft'].acc(prim):.4f} soft-DISC {fmt(t2)} {lab2}"
                P(f"     rho {rho}: VSA {line['vsa'].acc(prim):.4f} DISC {line['disc'].acc(prim):.4f} "
                  f"VSA-DISC {fmt(t)} {lab}{extra}")
                js[task]["noise"][rho] = {"vsa": line["vsa"].acc(prim), "disc": line["disc"].acc(prim), "label": lab,
                                          **({"soft": line["vsasoft"].acc(prim)} if "vsasoft" in line else {})}
        if "vsasoft" in arms and "vsa" in arms:
            t = boot_diff(arms["vsasoft"], arms["vsa"], prim, rng)
            P(f"     clean: VSA-soft - VSA-hard {fmt(t)}")
        # ---------------- exploratory 1 (task N): the trained VSA seeds with argmax digit clean-up (no 0.5 threshold)
        fs = [f"{d}/vsaargmax_s{s}.jsonl" for s in SEEDS if os.path.exists(f"{d}/vsaargmax_s{s}.jsonl")]
        if task == "N" and fs and "disc" in arms:
            xa = Arm(fs)
            lab, t, sv, sd = r2(xa, arms["disc"], prim, rng)
            ra = np.mean([[float(r[i]["mod"]["op"] == rows[i]["op"] and r[i]["mod"]["x"] == rows[i]["x"] and r[i]["mod"]["y"] == rows[i]["y"]) for i in prim] for r in xa.runs])
            ma = np.mean([[float(r[i]["mod"]["ans"] == rows[i]["ans"]) for i in prim] for r in xa.runs])
            P(f"  EXPLORATORY 1 (argmax clean-up, same trained VSA seeds, {len(fs)} runs): acc {xa.acc(prim):.4f} seeds "
              f"{[round(x, 4) for x in xa.seed_accs(prim)]}; read-exact {ra:.4f} module-answer {ma:.4f}; "
              f"VSA-argmax - DISC {fmt(t)} -> {lab} (exploratory)")
            for op in D.OPS_HELD:
                ids = [i for i, r in rows.items() if r["op"] == op]
                P(f"     held-out {op}: {xa.acc(ids):.4f}")
            js[task]["exploratory1"] = {"acc": xa.acc(prim), "label": lab, "delta": t, "read_exact": float(ra)}
        # ---------------- S7 (task R): VSA - LORA per cell
        if task == "R" and "vsa" in arms and "lora" in arms:
            P("  S7 VSA - LORA per cell [CI]:")
            for c, ids in cells:
                P(f"     {c:10s} {fmt(boot_diff(arms['vsa'], arms['lora'], ids, rng))}")
        # ---------------- S6 label-free (depth 2 as registered; depth 3 from amendment 2)
        for pre, dep in (("e2e", 2), ("e2e3", 3)):
            if task != "R" or f"{pre}_const" not in arms:
                continue
            ids = [i for i, r in rows.items() if r["depth"] == dep]
            P(f"  S6 label-free, {dep} fixed hops (depth-{dep} test cells, n = {len(ids)}):")
            fl = arms[f"{pre}_const"]
            P(f"     {pre}_const acc {fl.acc(ids):.4f} seeds {[round(x, 4) for x in fl.seed_accs(ids)]}")
            js[task][pre] = {"const": fl.acc(ids)}
            for n in (f"{pre}_vsa", f"{pre}_tab"):
                if n not in arms:
                    continue
                t = boot_diff(arms[n], fl, ids, rng)
                seeds_ok = [a >= b + 0.10 for a, b in zip(arms[n].seed_accs(ids), fl.seed_accs(ids))]
                learns = t[0] >= 0.10 and t[1] > 0 and all(seeds_ok)
                sh = [f"{d}/{n}_s{s}_shuf.jsonl" for s in SEEDS if os.path.exists(f"{d}/{n}_s{s}_shuf.jsonl")]
                use = ""
                if sh:
                    sa = Arm(sh)
                    tt = boot_diff(arms[n], sa, ids, rng)
                    use = f"; intact - shuffled {fmt(tt)} -> {'USES MODULE' if tt[0] >= 0.10 else 'no module use'}"
                P(f"     {n:9s} acc {arms[n].acc(ids):.4f} seeds {[round(x, 4) for x in arms[n].seed_accs(ids)]} "
                  f"- const {fmt(t)} seeds>=+0.10 {seeds_ok} -> {'LEARNS' if learns else 'does not learn'}{use}")
                js[task][pre][n] = {"acc": arms[n].acc(ids), "learns": bool(learns), "seeds_ok": seeds_ok}
            if f"{pre}_vsa" in arms and f"{pre}_tab" in arms:
                lab, t, sv, sd = r2(arms[f"{pre}_vsa"], arms[f"{pre}_tab"], ids, rng)
                P(f"     {pre}-VSA - {pre}-TAB {fmt(t)} -> {lab.replace('DISC', 'TAB')}")
    os.makedirs("results", exist_ok=True)
    open("results/b40_verdict.txt", "w").write("\n".join(out) + "\n")
    json.dump(js, open("results/b40_scores.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
