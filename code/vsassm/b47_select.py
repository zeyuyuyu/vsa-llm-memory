"""B47 selection (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md), B31's rule on train + dev only:
"fits" = training-subset accuracy >= 0.95 at some evaluation point; among configurations that fit, the highest dev
accuracy at the best evaluation point; if none fits, the highest dev overall; ties within 0.004 to the smaller rank,
then the smaller rate. Reads only the tuning runs' JSON (no test file).

  python b47_select.py --root out/ft --out out/select.json
"""
import argparse, glob, json, os, re

GRID = {("gen", "k32", 800): [(r, lr) for r in (8, 64) for lr in (3e-5, 1e-4, 3e-4)],
        ("fullgen", "k32", 800): [(8, lr) for lr in (3e-6, 1e-5, 3e-5)],
        ("fullcls", "k32", 800): [(8, lr) for lr in (3e-6, 1e-5, 3e-5)],
        ("scratch", "k32", 800): [(8, lr) for lr in (1e-4, 3e-4)],
        ("gen", "std", 8000): [(None, lr) for lr in (3e-5, 1e-4, 3e-4)],       # rank = the one selected at 800
        ("fullgen", "std", 8000): [(8, lr) for lr in (3e-6, 1e-5, 3e-5)]}


def fname(task, variant, n, rank, lr, seed):
    return "%s_%s_n%d_r%d_lr%s_s%d.json" % (task, variant, n, rank, "%g" % lr, seed)


def select(root, task, variant, n, rank_override=None):
    rows = []
    for rank, lr in GRID[(task, variant, n)]:
        rank = rank if rank is not None else rank_override
        p = os.path.join(root, fname(task, variant, n, rank, lr, 42))
        if not os.path.exists(p):
            return None, "missing %s" % os.path.basename(p)
        r = json.load(open(p))["run"]
        tmax = max(e["train_sub"]["acc"] for e in r["evals"])
        rows.append({"rank": rank, "lr": lr, "dev": r["best_dev"], "best_step": r["best_step"], "train_sub_max": tmax,
                     "fit": tmax >= 0.95, "early_stop": r.get("early_stop"), "file": os.path.basename(p)})
    pool = [x for x in rows if x["fit"]] or rows
    best = max(x["dev"] for x in pool)
    tied = [x for x in pool if x["dev"] >= best - 0.004 - 1e-12]
    s = sorted(tied, key=lambda x: (x["rank"], x["lr"]))[0]
    return dict(s, any_fit=any(x["fit"] for x in rows), grid=rows), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/ft")
    ap.add_argument("--out", default="out/select.json")
    a = ap.parse_args()
    sel = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for (task, variant, n) in GRID:
        key = "%s_%s_n%d" % (task, variant, n)
        if key in sel:
            continue                                    # a written selection is never changed
        ro = None
        if (task, n) == ("gen", 8000):
            if "gen_k32_n800" not in sel:
                continue
            ro = sel["gen_k32_n800"]["rank"]
        s, why = select(a.root, task, variant, n, ro)
        if s is None:
            print(key, why); continue
        sel[key] = s
        print(key, json.dumps({k: v for k, v in s.items() if k != "grid"}))
    json.dump(sel, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
