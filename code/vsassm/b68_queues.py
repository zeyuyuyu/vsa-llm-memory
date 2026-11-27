"""B68 (prereg/B68_early_layer_labelfree_prereg.md): queue lines, dev selection, release of the finals and the seed merge
for the seven B42 label-free interfaces (+ B31's binding-loss map) and the same-layer naming head on the input
embeddings (L0) and early layers (L2, L4) of Phi-2 and Qwen3-8B, arenas A, B, C.

Cells in the registered priority order (the cut order is its reverse): arena A (Qwen3-8B L0, L4, L2; Phi-2 L0, L4, L2),
then arena B (same), then arena C (same).

Grids: B42's (b42_queues.grid) for vq, dkvb, gumbel, st; cw = B42's cw grid at B42's rate for the model and arena
(as B48: Phi-2 3e-4; Qwen3-8B A and C 1e-3, B 3e-4); binding-loss map lr in {3e-4, 1e-3} chosen on the cell's dev
split (B52's rule). G1 (labelled positive control, dev): gumbel and st with the identity loss, hard clean-up, lr 3e-4.
Selection per family: highest best-epoch dev accuracy (perm/named), ties within 0.002 to the configuration listed
first (b42_select rule). Finals: the selected configuration, seeds 42, 123, 456, one process per seed
(out/final_seed/), merged into out/final/<cell>_<fam>.json (B42 layout). Naming head (B31 explore1 arguments, 4 epochs,
3 seeds) per cell as the reference.

  python b68_queues.py tune                     -> q/tune (static, cell order)
  python b68_queues.py select                   -> out/select.json (cells whose grids are complete; others listed)
  python b68_queues.py final --batch A          -> q/final (finals + naming head of the batch's cells; only after the
                                                   batch's dev choices are committed)
  python b68_queues.py merge                    (per-seed finals -> out/final/)
"""
import argparse, fcntl, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b42_queues import grid as b42_grid, cfg_name                           # noqa: E402

FEATS = "/data0/path/to/HOME/b68/feats"
ORDER = [(e + "L%d" % L, ar) for ar in "ABC" for e in ("qwen8", "phi2") for L in (0, 4, 2)]
FAMS = ["vq", "dkvb", "gumbel", "st", "cw"]
BIND_LRS = [3e-4, 1e-3]
CW_LR = {"phi2": {"A": 3e-4, "B": 3e-4, "C": 3e-4}, "qwen8": {"A": 1e-3, "B": 3e-4, "C": 1e-3}}   # B42 / B48
SEEDS = [42, 123, 456]
PRIMARY = "perm/named"
NAME_ARGS = ("--arm name --mode final --enc perm --conv named --readings perm/index --idset all --readout thr "
             "--lr 0.0003 --lam 0.0 --tau 0.1 --epochs 4 --fit_epochs 6 --seeds 42 123 456")   # = B31 explore1 / B52


def key(enc, ar):
    return "%s_%s" % (enc, ar)


def model(enc):
    return enc.split("L")[0]


def grid(fam, enc, ar):
    if fam == "cw":
        lr = CW_LR[model(enc)][ar]
        return [dict(cleanup=c, inw=i, code=k, lr=lr) for c in ("hard", "st") for i in (0, 1)
                for k in ("center", "inst", "zca")]
    return b42_grid(fam, model(enc), ar)


def cmd(enc, ar, fam, mode, c, out, seeds="42"):
    s = "b68_run.py b42_arms.py --enc_model %s --arena %s --fam %s --mode %s --seeds %s" % (enc, ar, fam, mode, seeds)
    for k, v in sorted(c.items()):
        s += " --%s %s" % (k, ("%g" % v) if isinstance(v, float) else v)
    return s + " --feats %s --out %s" % (FEATS, out)


def append(qfile, lines):
    os.makedirs(os.path.dirname(qfile) or ".", exist_ok=True)
    with open(qfile + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(qfile, "a") as f:
            f.write("".join(l + "\n" for l in lines))
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(qfile, "+%d" % len(lines), flush=True)


def load(p):
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except json.JSONDecodeError:
        return None


def best_dev(d):
    accs = [e["%s:dev" % PRIMARY]["acc"] for e in d["runs"][0]["epochs"]]
    i = max(range(len(accs)), key=lambda j: (accs[j], -j))
    return accs[i], i + 1


def pick(rows):
    top = max(r["dev"] for r in rows)
    return [r for r in rows if r["dev"] >= top - 0.002][0]


def bind_file(enc, ar, lr):
    return "out/tune/%s_bind_lr%g.json" % (key(enc, ar), lr)


def tune_files(enc, ar):
    k = key(enc, ar)
    fs = [(None, {"lr": lr}, bind_file(enc, ar, lr)) for lr in BIND_LRS]
    fs += [("g1", fam, "out/g1/%s_%s_g1.json" % (k, fam)) for fam in ("gumbel", "st")]
    fs += [(fam, c, "out/tune/%s_%s.json" % (k, cfg_name(fam, c))) for fam in FAMS for c in grid(fam, enc, ar)]
    return fs


def tune_lines():
    L = []
    for enc, ar in ORDER:
        for fam, c, out in tune_files(enc, ar):
            if fam is None:
                L.append(cmd(enc, ar, "bind", "tune", c, out))
            elif fam == "g1":
                L.append(cmd(enc, ar, c, "tune", {"lr": 3e-4, "idlam": 1.0, "cleanup": "hard"}, out))
            else:
                L.append(cmd(enc, ar, fam, "tune", c, out))
    return L


def select():
    sel = {"bind": {}, "cells_complete": [], "cells_incomplete": {}}
    for f in FAMS:
        sel[f] = {}
    sel["grids"] = {f: {} for f in FAMS}
    for enc, ar in ORDER:
        k = key(enc, ar)
        missing = [out for _, _, out in tune_files(enc, ar) if load(out) is None]
        if missing:
            sel["cells_incomplete"][k] = len(missing)
            continue
        rows = []
        for lr in BIND_LRS:
            acc, ep = best_dev(load(bind_file(enc, ar, lr)))
            rows.append({"cfg": {"lr": lr}, "dev": acc, "epoch": ep, "file": os.path.basename(bind_file(enc, ar, lr))})
        sel["bind"][k] = {"pick": pick(rows), "rows": rows}
        for fam in FAMS:
            rr = []
            for c in grid(fam, enc, ar):
                p = "out/tune/%s_%s.json" % (k, cfg_name(fam, c))
                acc, ep = best_dev(load(p))
                rr.append({"cfg": c, "dev": acc, "epoch": ep, "file": os.path.basename(p)})
            sel[fam][k] = pick(rr)
            sel["grids"][fam][k] = rr
        sel["cells_complete"].append(k)
        print(k, "bind lr", sel["bind"][k]["pick"]["cfg"]["lr"], "|", " | ".join(
            "%s %s %.3f" % (f, json.dumps(sel[f][k]["cfg"]), sel[f][k]["dev"]) for f in FAMS), flush=True)
    json.dump(sel, open("out/select.json.tmp", "w"), indent=1)
    os.replace("out/select.json.tmp", "out/select.json")
    print("complete", len(sel["cells_complete"]), "incomplete", json.dumps(sel["cells_incomplete"]))


def final_lines(batch, cut=(), only=()):
    sel = load("out/select.json")
    L = []
    for enc, ar in ORDER:
        k = key(enc, ar)
        if ar != batch or k in cut:             # compute note 2: cut cells get no finals
            continue
        if only and k not in only:              # release a committed sub-batch only (dev-choices record)
            continue
        assert k in sel["cells_complete"], "dev grid of %s incomplete" % k
        for fam in FAMS + ["bind"]:
            c = sel[fam][k]["cfg"] if fam != "bind" else sel["bind"][k]["pick"]["cfg"]
            for sd in SEEDS:
                L.append(cmd(enc, ar, fam, "final", c, "out/final_seed/%s_%s_s%d.json" % (k, fam, sd), str(sd)))
        L.append("b68_run.py b31_arms.py --enc_model %s --arena %s %s --feats %s --out out/name/%s_name.json"
                 % (enc, ar, NAME_ARGS, FEATS, k))
    return L


def merge():
    import torch
    os.makedirs("out/final", exist_ok=True)
    for enc, ar in ORDER:
        cell = key(enc, ar)
        for fam in FAMS + ["bind"]:
            ps = [os.path.join("out", "final_seed", "%s_%s_s%d.json" % (cell, fam, s)) for s in SEEDS]
            dst = os.path.join("out", "final", "%s_%s.json" % (cell, fam))
            if not all(os.path.exists(p) for p in ps) or os.path.exists(dst):
                continue
            ds = [json.load(open(p)) for p in ps]
            out = dict(ds[0])
            out["runs"] = [d["runs"][0] for d in ds]
            assert [r["seed"] for r in out["runs"]] == SEEDS
            out["args"] = dict(ds[0]["args"], seeds=SEEDS, out=dst)
            out["merged_from"] = ps
            json.dump(out, open(dst, "w"))
            sides = [p.replace(".json", ".side.pt") for p in ps]
            if all(os.path.exists(p) for p in sides):
                side = {}
                for p in sides:
                    side.update(torch.load(p, weights_only=False))
                torch.save(side, dst.replace(".json", ".side.pt"))
            print("merged", cell, fam)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "select", "final", "merge"])
    ap.add_argument("--batch", choices=list("ABC"))
    ap.add_argument("--cut", default="", help="cells cut by a compute note (comma-separated)")
    ap.add_argument("--only", default="", help="release only these cells (comma-separated)")
    a = ap.parse_args()
    if a.phase == "tune":
        append("q/tune", tune_lines())
    elif a.phase == "select":
        select()
    elif a.phase == "final":
        append("q/final", final_lines(a.batch, set(x for x in a.cut.split(",") if x), set(x for x in a.only.split(",") if x)))
    else:
        merge()
