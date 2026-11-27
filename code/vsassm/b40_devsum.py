"""B40 dev summary: reads out/scan/*/reader.json, out/rd/*/reader.json, out/dev/*/train.json and applies the prereg's
dev-selection rules (layer; reader length; write-back / LoRA / E2E configuration). Prints a table and the choices.
usage: python b40_devsum.py <b40_root>"""
import glob, json, os, sys

root = sys.argv[1]


def J(p):
    return json.load(open(p))


def pick_layer(task):
    res = {}
    for p in glob.glob(f"{root}/out/scan/{task}_L*/reader.json"):
        L = int(p.split("_L")[1].split("/")[0])
        res[L] = J(p)["final"]["exact"]
    if not res:
        return None, res
    best = max(res.values())
    cands = [L for L, v in res.items() if v >= best - 0.005]
    return min(cands, key=lambda L: (abs(L - 17), L)), res


def pick_reader(task, arm, grid):
    res = {}
    for e in grid:
        p = f"{root}/out/rd/{task}_{arm}_e{e}_s0/reader.json"
        if os.path.exists(p):
            res[e] = J(p)["final"]["exact"]
    if len(res) < len(grid):
        return None, res
    best = max(res.values())
    return min(e for e, v in res.items() if v >= best - 0.002), res


def pick_train(prefix, opts):
    """opts: ordered list of run-name suffixes (first listed wins ties)."""
    res = {}
    for o in opts:
        p = f"{root}/out/dev/{prefix}{o}_s0/train.json"
        if os.path.exists(p):
            j = J(p)
            res[o] = (j["best_dev"], j["best_step"], j["trainable_params"], [h["dev"] for h in j["hist"]])
    if len(res) < len(opts):
        return None, res
    best = max(v[0] for v in res.values())
    return next(o for o in opts if res[o][0] >= best), res


out = {}
for task in ("N", "R"):
    L, res = pick_layer(task)
    print(f"[{task}] layer scan (VSA encoder, 100 epochs, dev read exact):", {k: round(v, 4) for k, v in sorted(res.items())}, "-> L =", L)
    out[f"{task}_layer"] = L
    for arm, grid in (("vsa", (100, 300, 1000)), ("disc", (30, 100, 300))):
        e, r = pick_reader(task, arm, grid)
        print(f"[{task}] reader {arm} epochs:", {k: round(v, 4) for k, v in r.items()}, "-> epochs", e)
        out[f"{task}_{arm}_epochs"] = e
opts = {
    "N_vsa_": ["recon_lr1e-3", "zero_lr1e-3", "recon_lr3e-4", "zero_lr3e-4"],
    "N_disc_": ["lr1e-3", "lr3e-4"], "N_lora1l_": ["lr1e-3", "lr3e-4"],
    "R_vsa_": ["lr1e-3", "lr3e-4"], "R_disc_": ["lr1e-3", "lr3e-4"],
    "N_lora_": ["lr1e-4", "lr3e-4", "lr1e-3"], "R_lora_": ["lr1e-4", "lr3e-4", "lr1e-3"],
    "R_e2e_vsa_": ["lr1e-3", "lr3e-3"], "R_e2e_tab_": ["lr1e-3", "lr3e-3"], "R_e2e_const_": ["lr1e-3", "lr3e-3"],
    "R3_e2e_vsa_": ["lr1e-3", "lr3e-3"], "R3_e2e_tab_": ["lr1e-3", "lr3e-3"], "R3_e2e_const_": ["lr1e-3", "lr3e-3"],
}
for pre, o in opts.items():
    c, r = pick_train(pre, o)
    print(f"{pre:14s}", {k: (round(v[0], 4), v[1], v[2], [round(x, 3) for x in v[3]]) for k, v in r.items()}, "-> choice", c)
    out[pre] = c
json.dump(out, open(f"{root}/out/dev_choices.json", "w"), indent=1)
soft = {}
for b in (10, 30, 100):
    p = f"{root}/out/devpred/R_vsasoft_b{b}.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        soft[b] = sum(r["correct"] for r in rows) / len(rows)
if soft:
    best = max(soft.values())
    out["soft_beta"] = next(b for b in (10, 30, 100) if b in soft and soft[b] >= best)
    print("VSA-soft beta (dev accuracy):", soft, "-> beta", out["soft_beta"])
    json.dump(out, open(f"{root}/out/dev_choices.json", "w"), indent=1)
