"""B65 queues and dev choices (prereg/B65_tuned_lr_read_position_prereg.md, 'Dev selection'). Run in $B65_ROOT on the
box; every choice is mechanical and reads dev files only (out/scan, out/rd, out/dev). Writes q/<phase> queue files and
out/choices.json.
usage: python b65_plan.py <phase> [position]
  S         self-test queue (training rows only)
  D1        dev caches + layer scan (ops) + reader grid (last)
  L         choose the ops layer; reader grid (ops)
  R <pos>   choose readers (per arm) at a position; write-back grid (4 learning rates x 3 arms) -> q/R_wb_<pos>
  W <pos>   choose write-back learning rates at a position; extension queue -> q/W_ext_<pos> for every arm whose
            choice sits strictly on the grid's edge (re-run after the extensions finish, until none is added)
  F <pos>   (all choices at the position interior or extensions exhausted) seeds 1, 2: readers, then write-back
  T         test queue (18 evaluations, once each) + determinism re-runs (seed 0, three arms, position last)
"""
import json, math, os, sys

R = "b65_run.py"
ARMS = ("vsapub", "vsabm", "disc")
POS = ("last", "ops")
LAYERS = (8, 12, 14, 16, 17, 18, 20, 24)
LAST_LAYER = 20  # B40's dev choice for task N (same scan, out/dev_choices.json); not re-scanned
READER_LR = ("1e-3", "1e-4")
EPOCHS = {"vsa": (100, 300, 1000), "disc": (30, 100, 300)}
WB_GRID = ("3e-4", "1e-3", "3e-3", "1e-2")
LADDER = ("3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1")
MAX_EXT = 3
CH = "out/choices.json"


def w(name, lines):
    os.makedirs("q", exist_ok=True)
    open(f"q/{name}", "w").write("\n".join(lines) + ("\n" if lines else ""))
    print(name, len(lines))


def ch():
    return json.load(open(CH)) if os.path.exists(CH) else {}


def save(c):
    os.makedirs("out", exist_ok=True)
    json.dump(c, open(CH, "w"), indent=1, sort_keys=True)


def rarm(arm):
    return "disc" if arm == "disc" else "vsa"


def rkey(arm):
    return {"vsapub": "pub", "vsabm": "bm", "disc": "disc"}[arm]


def rdir(pos, ra, lr, s):
    return f"out/rd/{pos}_{ra}_lr{lr}_s{s}"


def wdir(pos, arm, lr, s):
    return f"out/dev/{pos}_{arm}_lr{lr}_s{s}" if s == 0 else f"out/fin/{pos}_{arm}_s{s}"


def reader_line(pos, ra, L, lr, s, epochs):
    return f"{R} reader --pos {pos} --arm {ra} --layer {L} --lr {lr} --epochs {','.join(map(str, epochs))} --seed {s} --out {rdir(pos, ra, lr, s)}"


def train_line(pos, arm, L, rd, lr, s):
    return f"{R} train --pos {pos} --arm {arm} --layer {L} --reader {rd} --lr {lr} --steps 1000 --eval_every 250 --seed {s} --out {wdir(pos, arm, lr, s)}"


def choose_wb(pos, arm, grid):
    """Highest best-dev first-token accuracy (steps 250-1000); ties -> lr nearest 1e-3 (|log10 ratio|), then smaller."""
    res = {}
    for lr in grid:
        f = f"{wdir(pos, arm, lr, 0)}/train.json"
        if os.path.exists(f):
            res[lr] = json.load(open(f))["best_dev"]
    best = max(res.values())
    tied = [lr for lr in res if res[lr] == best]
    pick = sorted(tied, key=lambda lr: (abs(math.log10(float(lr) / 1e-3)), float(lr)))[0]
    return pick, res


ph = sys.argv[1]
c = ch()

if ph == "S":  # self-test on TRAINING rows only (smoke/ outputs; dev / test never touched)
    L = [f"{R} selftest --split train",
         f"{R} cache --split train --layers {','.join(map(str, LAYERS))} --bs 64"]
    w("S1", L)
    L = []
    for pos, lay in (("last", 20), ("ops", 17)):
        for ra in ("vsa", "disc"):
            L.append(f"{R} reader --pos {pos} --arm {ra} --layer {lay} --lr 1e-3 --epochs 1,2 --seed 0 --dev_split train --dev_limit 512 --out smoke/rd_{pos}_{ra}")
    w("S2", L)
    L = []
    for pos, lay in (("last", 20), ("ops", 17)):
        for arm in ARMS:
            L.append(f"{R} train --pos {pos} --arm {arm} --layer {lay} --reader smoke/rd_{pos}_{rarm(arm)}/e2 --lr 1e-3 --steps 20 --eval_every 10 --recon_epochs 2 --seed 0 --dev_split train --dev_limit 32 --out smoke/wb_{pos}_{arm}")
            L.append(f"{R} eval --pos {pos} --arm {arm} --layer {lay} --reader smoke/rd_{pos}_{rarm(arm)}/e2 --ckpt smoke/wb_{pos}_{arm} --split train --limit 32 --bs 32 --pred smoke/pred_{pos}_{arm}.jsonl")
    w("S3", L)

if ph == "D1":
    w("D1_cache", [f"{R} cache --split dev --layers {','.join(map(str, LAYERS))} --bs 64"])
    w("D1_scan", [f"{R} reader --pos ops --arm vsa --layer {L} --lr 1e-3 --epochs 100 --seed 0 --out out/scan/ops_L{L}" for L in LAYERS])
    w("D1_rdlast", [reader_line("last", ra, LAST_LAYER, lr, 0, EPOCHS[ra]) for ra in ("vsa", "disc") for lr in READER_LR])

if ph == "L":  # B40's layer rule on the ops reads: highest dev read exact (published read, VSA encoder, 100 epochs);
    # ties within 0.005 -> the layer nearest 17 (DE25's), then the lower layer
    acc = {L: json.load(open(f"out/scan/ops_L{L}/reader.json"))["hist"][-1]["pub"] for L in LAYERS}
    best = max(acc.values())
    pick = sorted([L for L in LAYERS if acc[L] >= best - 0.005], key=lambda L: (abs(L - 17), L))[0]
    c["layer"] = {"last": LAST_LAYER, "ops": pick}
    c["scan_ops"] = acc
    save(c)
    print("ops layer", pick, acc)
    w("L_rdops", [reader_line("ops", ra, pick, lr, 0, EPOCHS[ra]) for ra in ("vsa", "disc") for lr in READER_LR])

SEL = (sys.argv[2],) if len(sys.argv) > 2 else POS

if ph == "R":  # reader per arm and position: highest dev read exact under the arm's own read rule; ties within 0.002
    # -> lr 1e-3 (DE25 / B40) first, then fewer epochs
    c.setdefault("reader", {}); c.setdefault("reader_dev", {})
    for pos in SEL:
        for arm in ARMS:
            ra, k = rarm(arm), rkey(arm)
            cand = []
            for lr in READER_LR:
                for h in json.load(open(f"{rdir(pos, ra, lr, 0)}/reader.json"))["hist"]:
                    cand.append((lr, h["epoch"], h[k]))
            best = max(x[2] for x in cand)
            pick = sorted([x for x in cand if x[2] >= best - 0.002], key=lambda x: (READER_LR.index(x[0]), x[1]))[0]
            c["reader"][f"{pos}_{arm}"] = {"lr": pick[0], "epochs": pick[1], "dev_read": pick[2]}
            c["reader_dev"][f"{pos}_{arm}"] = [list(x) for x in cand]
    save(c)
    L = []
    for pos in SEL:
        for lr in WB_GRID:
            for arm in ARMS:
                r_ = c["reader"][f"{pos}_{arm}"]
                rd = f"{rdir(pos, rarm(arm), r_['lr'], 0)}/e{r_['epochs']}"
                L.append(train_line(pos, arm, c["layer"][pos], rd, lr, 0))
    w("R_wb_" + "_".join(SEL), L)

if ph == "W":  # write-back learning rate per arm and position, with the edge-extension rule
    c.setdefault("wb_grid", {})
    ext = []
    for pos in SEL:
        for arm in ARMS:
            key = f"{pos}_{arm}"
            grid = c["wb_grid"].get(key, list(WB_GRID))
            pick, res = choose_wb(pos, arm, grid)
            lo, hi = min(grid, key=float), max(grid, key=float)
            nlo = LADDER.index(lo); nhi = LADDER.index(hi)
            n_ext_lo = LADDER.index("3e-4") - nlo; n_ext_hi = nhi - LADDER.index("1e-2")
            status = "interior"
            new = None
            if pick == lo and len(res) == len(grid):
                status = "edge-low"
                if n_ext_lo < MAX_EXT:
                    new = LADDER[nlo - 1]
            elif pick == hi and len(res) == len(grid):
                status = "edge-high"
                if n_ext_hi < MAX_EXT:
                    new = LADDER[nhi + 1]
            if len(res) < len(grid):
                status = "incomplete"
            if new:
                grid = grid + [new]
                r_ = c["reader"][key]
                ext.append(train_line(pos, arm, c["layer"][pos], f"{rdir(pos, rarm(arm), r_['lr'], 0)}/e{r_['epochs']}", new, 0))
            c["wb_grid"][key] = grid
            c.setdefault("wb", {})[key] = {"lr": pick, "status": status if not new else status + "->extend " + new,
                                           "dev": res}
            print(key, pick, status, res, "extend" if new else "")
    save(c)
    w("W_ext_" + "_".join(SEL), ext)

if ph == "F":  # seeds 1, 2 at the chosen settings: reader (chosen lr, all epochs snapshots kept; the chosen epoch
    # count is used) then write-back (chosen lr); own best dev step
    for pos in SEL:
        for arm in ARMS:
            v = c["wb"][f"{pos}_{arm}"]
            assert v["status"] in ("interior", "edge-low", "edge-high"), (pos, arm, v)
    rd_lines, wb_lines, done = [], [], set()
    for pos in SEL:
        for arm in ARMS:
            r_ = c["reader"][f"{pos}_{arm}"]
            for s in (1, 2):
                tag = (pos, rarm(arm), r_["lr"], s)
                if tag not in done:
                    done.add(tag)
                    rd_lines.append(reader_line(pos, rarm(arm), c["layer"][pos], r_["lr"], s, EPOCHS[rarm(arm)]))
                wb_lines.append(train_line(pos, arm, c["layer"][pos], f"{rdir(pos, rarm(arm), r_['lr'], s)}/e{r_['epochs']}",
                                           c["wb"][f"{pos}_{arm}"]["lr"], s))
    w("F_rd_" + "_".join(SEL), rd_lines)
    w("F_wb_" + "_".join(SEL), wb_lines)

if ph == "T":
    L, Dt = [], []
    for pos in SEL:
        for arm in ARMS:
            r_ = c["reader"][f"{pos}_{arm}"]
            lr = c["wb"][f"{pos}_{arm}"]["lr"]
            for s in (0, 1, 2):
                rd = f"{rdir(pos, rarm(arm), r_['lr'], s)}/e{r_['epochs']}"
                line = (f"{R} eval --pos {pos} --arm {arm} --layer {c['layer'][pos]} --reader {rd} --ckpt {wdir(pos, arm, lr, s)} "
                        f"--split test --bs 32 --pred preds/{pos}/{arm}_s{s}.jsonl")
                L.append(line)
                if pos == "last" and s == 0:
                    Dt.append(line.replace(f"preds/{pos}/", "preds_rerun/"))
    w("T_test", L)
    w("T_rerun", Dt)
