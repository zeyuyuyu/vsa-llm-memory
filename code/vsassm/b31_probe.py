"""B31 equalised probe (prereg/B31_encoder_equalised_baselines_prereg.md): a read-out of the frozen states that is given
the same question positions the memory arms get.

Features (every block standardised with training-split mean and standard deviation):
  last  the state at the last non-pad token (the B27 standardised diagnostic probe)
  eq    [query span mean, candidate span mean, last-token state, mean over the input]
  eqx   eq plus the elementwise product of the query and candidate span means
Models: logistic regression (L2 weight lam in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}; Adam 1e-2, full batch, 2,000 steps) and
a one-hidden-layer MLP (512 ReLU units, dropout 0.1, AdamW 1e-3, weight decay in {1e-4, 1e-2, 1e-1}, full batch, 1,000
steps). Dev accuracy is read every 50 steps; the (feature set, model, hyper-parameter, step) with the highest dev
accuracy is selected (ties: logistic before MLP, then the smaller feature set, then the larger penalty, then the earlier
step). Tune mode reads train + dev only; final mode trains the selected configuration with seeds 42, 123, 456 and
reads the test split once, at the selected step.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b31_arms import layout                                                  # noqa: E402

FSETS = ["last", "eq", "eqx"]
LAMS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
WDS = [1e-4, 1e-2, 1e-1]


def feats(root, enc, arena, split):
    o = torch.load(os.path.join(root, "%s_%s_%s.pt" % (enc, arena, split)), weights_only=False)
    _, _, q, c, _ = layout(arena)
    return {"q": o["F"][:, q], "c": o["F"][:, c], "last": o["last"], "mean": o["mean"], "y": o["labels"]}


def design(fs, d, stats):
    blocks = {"last": ["last"], "eq": ["q", "c", "last", "mean"], "eqx": ["q", "c", "last", "mean", "qc"]}[fs]
    out = []
    for b in blocks:
        x = d["q"] * d["c"] if b == "qc" else d[b]
        mu, sd = stats[b]
        out.append((x - mu) / sd)
    return torch.cat(out, 1)


def make_stats(tr):
    st = {}
    for b in ("q", "c", "last", "mean", "qc"):
        x = tr["q"] * tr["c"] if b == "qc" else tr[b]
        st[b] = (x.mean(0), x.std(0) + 1e-6)
    return st


def train_eval(model, hyper, X, y, Xd, yd, Xt, seed, dev, steps_at=None):
    torch.manual_seed(seed)
    X, y, Xd, yd = X.to(dev), y.to(dev), Xd.to(dev), yd.to(dev)
    if model == "logistic":
        net = nn.Linear(X.size(1), 2).to(dev)
        nn.init.zeros_(net.weight); nn.init.zeros_(net.bias)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        n_steps, lam = 2000, hyper
    else:
        net = nn.Sequential(nn.Dropout(0.1), nn.Linear(X.size(1), 512), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(512, 2)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=hyper)
        n_steps, lam = 1000, 0.0
    curve, test_s = [], None
    for st in range(1, n_steps + 1):
        net.train()
        loss = F.cross_entropy(net(X), y)
        if lam:
            loss = loss + 0.5 * lam * net.weight.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 50 == 0:
            net.eval()
            with torch.no_grad():
                tra = float((net(X).argmax(1) == y).float().mean())
                da = float((net(Xd).argmax(1) == yd).float().mean())
                curve.append((st, tra, da))
                if steps_at is not None and st == steps_at and Xt is not None:
                    lg = net(Xt.to(dev)); test_s = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    return curve, test_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, choices=["phi2", "qwen8"])
    ap.add_argument("--arena", required=True, choices=["A", "B", "C"])
    ap.add_argument("--mode", required=True, choices=["tune", "final"])
    ap.add_argument("--select", default=None, help="final: JSON of the selected configuration")
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    tr = feats(a.feats, a.enc, a.arena, "train"); dv = feats(a.feats, a.enc, a.arena, "dev")
    stats = make_stats(tr)
    res = {"args": vars(a), "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.mode == "tune":
        grid = []
        for fs in FSETS:
            X, Xd = design(fs, tr, stats), design(fs, dv, stats)
            for model, hs in (("logistic", LAMS), ("mlp", WDS)):
                for hyp in hs:
                    curve, _ = train_eval(model, hyp, X, tr["y"], Xd, dv["y"], None, 42, a.device)
                    bst = max(curve, key=lambda t: (t[2], -t[0]))
                    grid.append({"fset": fs, "model": model, "hyper": hyp, "curve": curve, "best_step": bst[0],
                                 "best_dev": bst[2], "train_at_best": bst[1]})
                    print("  %s %s %-4s %-8s %-7g dev %.4f @%d (train %.4f)" % (a.enc, a.arena, fs, model, hyp,
                                                                                bst[2], bst[0], bst[1]), flush=True)
        rank = {"logistic": 0, "mlp": 1}
        sel = max(grid, key=lambda g: (round(g["best_dev"], 6), -rank[g["model"]], -FSETS.index(g["fset"]),
                                       g["hyper"], -g["best_step"]))
        res["grid"] = grid
        res["selected"] = {k: sel[k] for k in ("fset", "model", "hyper", "best_step", "best_dev", "train_at_best")}
        print("  SELECTED", json.dumps(res["selected"]), flush=True)
    else:
        sel = json.load(open(a.select))
        te = feats(a.feats, a.enc, a.arena, "test")
        X, Xd, Xt = design(sel["fset"], tr, stats), design(sel["fset"], dv, stats), design(sel["fset"], te, stats)
        res["selected"] = sel; res["runs"] = []
        for seed in (42, 123, 456):
            curve, ts = train_eval(sel["model"], sel["hyper"], X, tr["y"], Xd, dv["y"], Xt, seed, a.device,
                                   steps_at=sel["best_step"])
            yt = te["y"].numpy()
            acc = float(((ts > 0).astype(int) == yt).mean())
            res["runs"].append({"seed": seed, "test_acc": acc, "test_scores": [round(float(x), 5) for x in ts],
                                "curve": curve})
            print("  TEST %s %s seed %d %.4f" % (a.enc, a.arena, seed, acc), flush=True)
    res["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
