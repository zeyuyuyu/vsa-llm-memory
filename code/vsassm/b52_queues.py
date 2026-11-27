"""B52 part (a) (prereg/B52_robustness_layers_maps_1day_prereg.md): queue lines, dev selection and the orchestrator for
the seven B42 label-free interfaces on Qwen3-8B's B31-exploratory layer and input embeddings.

Cells (encoder feature set x arena), in priority order: qwen8L27 x C (B31 exploratory 1 chose layer 27 for C on dev),
qwen8L0 x C (input embeddings), qwen8L9 x A (layer 9 chosen for A), qwen8L0 x A.

Grids: B42's (b42_queues.grid) for vq, dkvb, gumbel, st; cw = B42's cw grid at the binding-loss lr selected on this
cell's dev split (B31 / B42 rule: lr in {3e-4, 1e-3}, highest best-epoch dev, ties within 0.002 to 3e-4), because no
B31 dev choice exists for these layers. Selection per family: highest best-epoch dev accuracy (perm/named), ties within
0.002 to the configuration listed first (b42_select rule). Finals: seeds 42 123 456.

  python b52_queues.py tune                  -> q/tune  (bind lr runs first, G1, vq/dkvb/gumbel/st grids, per cell)
  python b52_queues.py name                  -> q/final (the naming head on the input embeddings, A and C)
  python b52_queues.py orch                  (loop: bind lr choice -> cw grid; full grid -> select -> finals)
  python b52_queues.py select                (out/select.json from whatever tune outputs exist)
"""
import argparse, fcntl, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b42_queues import grid as b42_grid, cfg_name                           # noqa: E402

ALL_CELLS = [("qwen8L27", "C"), ("qwen8L0", "C"), ("qwen8L9", "A"), ("qwen8L0", "A")]
# compute note 2 (deadline): B52_CELLS="qwen8L27_C" restricts the orchestrator to the listed cells
CELLS = [c for c in ALL_CELLS if not os.environ.get("B52_CELLS") or "%s_%s" % c in os.environ["B52_CELLS"].split(",")]
FAMS = ["vq", "dkvb", "gumbel", "st", "cw"]
BIND_LRS = [3e-4, 1e-3]
SEEDS = "42 123 456"
PRIMARY = "perm/named"
NAME_ARGS = ("--arm name --mode final --enc perm --conv named --readings perm/index --idset all --readout thr "
             "--lr 0.0003 --lam 0.0 --tau 0.1 --epochs 4 --fit_epochs 6 --seeds 42 123 456")   # = B31 explore1 args


def key(enc, ar):
    return "%s_%s" % (enc, ar)


def grid(fam, enc, ar, bind_lr=None):
    if fam == "cw":
        assert bind_lr is not None
        return [dict(cleanup=c, inw=i, code=k, lr=bind_lr) for c in ("hard", "st") for i in (0, 1)
                for k in ("center", "inst", "zca")]
    return b42_grid(fam, enc, ar)


def cmd(enc, ar, fam, mode, c, out, seeds="42"):
    s = "b52_run.py b42_arms.py --enc_model %s --arena %s --fam %s --mode %s --seeds %s" % (enc, ar, fam, mode, seeds)
    for k, v in sorted(c.items()):
        s += " --%s %s" % (k, ("%g" % v) if isinstance(v, float) else v)
    return s + " --out " + out


def append(qfile, lines):
    os.makedirs(os.path.dirname(qfile) or ".", exist_ok=True)
    with open(qfile + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(qfile, "a") as f:
            f.write("".join(l + "\n" for l in lines))
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(qfile, "+%d" % len(lines), flush=True)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def best_dev(d):
    accs = [e["%s:dev" % PRIMARY]["acc"] for e in d["runs"][0]["epochs"]]
    i = max(range(len(accs)), key=lambda j: (accs[j], -j))
    return accs[i], i + 1


def pick(rows):
    top = max(r["dev"] for r in rows)
    return [r for r in rows if r["dev"] >= top - 0.002][0]


def bind_file(enc, ar, lr):
    return "out/tune/%s_bind_lr%g.json" % (key(enc, ar), lr)


def tune_lines():
    L = []
    for enc, ar in CELLS:                       # binding-loss lr runs first: the cw grid waits for their choice
        for lr in BIND_LRS:
            L.append(cmd(enc, ar, "bind", "tune", {"lr": lr}, bind_file(enc, ar, lr)))
    for enc, ar in CELLS:
        k = key(enc, ar)
        for fam in ("gumbel", "st"):            # G1: the labelled positive control
            L.append(cmd(enc, ar, fam, "tune", {"lr": 3e-4, "idlam": 1.0, "cleanup": "hard"}, "out/g1/%s_%s_g1.json" % (k, fam)))
        for fam in ("vq", "dkvb", "gumbel", "st"):
            for c in grid(fam, enc, ar):
                L.append(cmd(enc, ar, fam, "tune", c, "out/tune/%s_%s.json" % (k, cfg_name(fam, c))))
    return L


def name_lines():
    return ["b52_run.py b31_arms.py --enc_model qwen8L0 --arena %s %s --out out/name/qwen8L0_%s_name.json" % (ar, NAME_ARGS, ar)
            for ar in ("C", "A")]


def bind_choice(enc, ar):
    rows = []
    for lr in BIND_LRS:
        d = load(bind_file(enc, ar, lr))
        if d is None:
            return None
        acc, ep = best_dev(d)
        rows.append({"cfg": {"lr": lr}, "dev": acc, "epoch": ep, "file": os.path.basename(bind_file(enc, ar, lr))})
    return {"pick": pick(rows), "rows": rows}


def select_cell(enc, ar, bind_lr):
    out, grids, missing = {}, {}, []
    k = key(enc, ar)
    for fam in FAMS:
        rows = []
        for c in grid(fam, enc, ar, bind_lr):
            p = "out/tune/%s_%s.json" % (k, cfg_name(fam, c))
            d = load(p)
            if d is None:
                missing.append(p); continue
            acc, ep = best_dev(d)
            rows.append({"cfg": c, "dev": acc, "epoch": ep, "file": os.path.basename(p)})
        grids[fam] = rows
        if rows:
            out[fam] = pick(rows)
    return out, grids, missing


def write_select(state):
    sel = {f: {} for f in FAMS}
    sel["grids"] = {f: {} for f in FAMS}
    sel["bind"] = {}
    for k, v in state.items():
        if "bind" in v:
            sel["bind"][k] = v["bind"]
        for f in FAMS:
            if f in v.get("sel", {}) and f in v.get("grids", {}):
                sel[f][k] = v["sel"][f]
                sel["grids"][f][k] = v["grids"][f]
    json.dump(sel, open("out/select.json.tmp", "w"), indent=1)
    os.replace("out/select.json.tmp", "out/select.json")


def orch(poll=60):
    state = load("out/orch_state.json") or {}
    while True:
        done = 0
        for enc, ar in CELLS:
            k = key(enc, ar)
            s = state.setdefault(k, {})
            if "bind" not in s:
                b = bind_choice(enc, ar)
                if b is None:
                    continue
                s["bind"] = b
                lr = b["pick"]["cfg"]["lr"]
                # cw tuning goes to the PRIORITY queue (q/final is served first), so a cell's grid is not held up behind
                # later cells' tuning
                append("q/final", [cmd(enc, ar, "cw", "tune", c, "out/tune/%s_%s.json" % (k, cfg_name("cw", c)))
                                  for c in grid("cw", enc, ar, lr)])
                print(time.strftime("%H:%M:%S"), k, "bind lr", lr, json.dumps(b["rows"]), flush=True)
            # compute note 4 (deadline): finals are queued per family as soon as that family's grid is complete
            # (bind at once); the selection rule per family is unchanged
            s.setdefault("sel", {}); s.setdefault("grids", {}); s.setdefault("queued", [])
            lr = s["bind"]["pick"]["cfg"]["lr"]
            sel, grids, missing = select_cell(enc, ar, lr)
            for fam in FAMS + ["bind"]:
                if fam in s["queued"]:
                    continue
                if fam != "bind":
                    if any(("_%s_" % fam) in os.path.basename(p) for p in missing) or fam not in sel:
                        continue
                    s["sel"][fam], s["grids"][fam] = sel[fam], grids[fam]
                    c = sel[fam]["cfg"]
                else:
                    c = {"lr": lr}
                append("q/final", [cmd(enc, ar, fam, "final", c, "out/final_seed/%s_%s_s%s.json" % (k, fam, sd), sd)
                                   for sd in SEEDS.split()])
                s["queued"].append(fam)
                print(time.strftime("%H:%M:%S"), k, "queued finals", fam, json.dumps(c), flush=True)
            if len(s["queued"]) < len(FAMS) + 1:
                continue
            done += 1
        json.dump(state, open("out/orch_state.json", "w"), indent=1)
        write_select(state)
        if done == len(CELLS):
            print("all finals queued", flush=True)
            return
        time.sleep(poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "name", "orch", "select"])
    a = ap.parse_args()
    if a.phase == "tune":
        append("q/tune", tune_lines())
    elif a.phase == "name":
        append("q/final", name_lines())
    elif a.phase == "orch":
        orch()
    else:
        st = load("out/orch_state.json") or {}
        write_select(st)
