"""B41 naming head (box, one GPU): a linear head over the 4,096-name codebook on the span-mean frozen Qwen3-8B state of
a mention (the B31 recipe, retrained for this codebook), trained with identity labels on the train bank.

  fit    for each layer in b41_feats.LAYERS: standardise with train statistics, train the head (cross-entropy, AdamW,
         epoch chosen by pooled dev NLL); pick the layer with the lowest pooled dev NLL; fit one temperature T on the
         pooled dev bank (NLL); write head_L<L>.pt, choice.json (layer, T, dev error / NLL / top-k mass per rho)
  logits write <split>_logits.npy (fp16, rows in bank order, raw logits / T) for dev or test with the chosen head
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_names as NM                                                      # noqa: E402

LAYERS = [9, 18, 27, 36]


def load(feats, split, L):
    return torch.from_numpy(np.load(os.path.join(feats, "%s_L%d.npy" % (split, L))).astype(np.float32))


def train_head(Xtr, ytr, Xdv, ydv, dev, epochs=30, lr=1e-3, wd=1e-4, bs=2048, seed=41):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    Xtr_ = ((Xtr - mu) / sd).to(dev); Xdv_ = ((Xdv - mu) / sd).to(dev)
    ytr_, ydv_ = ytr.to(dev), ydv.to(dev)
    W = torch.nn.Linear(Xtr.shape[1], NM.E).to(dev)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=wd)
    best = (1e9, None, -1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(ytr_), generator=g).to(dev)
        for b in range(0, len(perm), bs):
            i = perm[b:b + bs]
            loss = F.cross_entropy(W(Xtr_[i]), ytr_[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nll = F.cross_entropy(W(Xdv_), ydv_).item()
        if nll < best[0]:
            best = (nll, {k: v.detach().cpu().clone() for k, v in W.state_dict().items()}, ep)
    return {"mu": mu, "sd": sd, "state": best[1], "dev_nll": best[0], "epoch": best[2]}


def apply(head, X, dev, bs=8192):
    W = torch.nn.Linear(X.shape[1], NM.E).to(dev)
    W.load_state_dict(head["state"])
    out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            out.append(W(((X[b:b + bs] - head["mu"]) / head["sd"]).to(dev)).cpu())
    return torch.cat(out)


def fit_T(logits, y):
    lt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)

    def clo():
        opt.zero_grad()
        l = F.cross_entropy(logits / lt.exp(), y)
        l.backward()
        return l
    opt.step(clo)
    return float(lt.exp())


def stats(logits, y, rhos_rows, T):
    p = torch.softmax(logits / T, -1)
    out = {}
    for r in sorted(set(rhos_rows.tolist())):
        m = rhos_rows == r
        pr, yr = p[m], y[m]
        am = pr.argmax(-1)
        top = torch.topk(pr, 16, -1)
        out["%.3f" % r] = {
            "n": int(m.sum()), "argmax_error": float((am != yr).float().mean()),
            "nll": float(F.nll_loss(torch.log(pr + 1e-30), yr)),
            "same_cluster_share_of_errors": float(((am // 4 == yr // 4) & (am != yr)).float().sum() /
                                                  max(1, int((am != yr).sum()))),
            "true_in_top4": float((torch.topk(pr, 4, -1).indices == yr[:, None]).any(-1).float().mean()),
            "true_in_top16": float((top.indices == yr[:, None]).any(-1).float().mean()),
            "mass_outside_top1": float(1 - top.values[:, 0].mean()),
            "mass_outside_top4": float(1 - top.values[:, :4].sum(-1).mean()),
            "mass_outside_top16": float(1 - top.values.sum(-1).mean()),
            "mean_p_true": float(pr.gather(1, yr[:, None]).mean()),
            "ece_top1": ece(pr, yr)}
    return out


def ece(p, y, nb=15):
    conf, am = p.max(-1)
    acc = (am == y).float()
    e = 0.0
    for i in range(nb):
        m = (conf > i / nb) & (conf <= (i + 1) / nb)
        if m.any():
            e += float(m.float().mean()) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "logits"])
    ap.add_argument("--feats", default="/path/to/work/b41/feats")
    ap.add_argument("--out", default="/path/to/work/b41/head")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    tr = NM.bank("train"); dv = NM.bank("dev")
    ytr = torch.tensor([r[0] for r in tr]); ydv = torch.tensor([r[0] for r in dv])
    rdv = torch.tensor([r[1] for r in dv])
    if a.cmd == "fit":
        res = {}
        for L in LAYERS:
            t0 = time.time()
            h = train_head(load(a.feats, "train", L), ytr, load(a.feats, "dev", L), ydv, dev)
            torch.save(h, os.path.join(a.out, "head_L%d.pt" % L))
            lg = apply(h, load(a.feats, "dev", L), dev)
            res[L] = {"dev_nll_T1": h["dev_nll"], "epoch": h["epoch"],
                      "dev_err_pooled": float((lg.argmax(-1) != ydv).float().mean()), "sec": time.time() - t0}
            print(L, res[L], flush=True)
        Lb = min(LAYERS, key=lambda L: res[L]["dev_nll_T1"])
        h = torch.load(os.path.join(a.out, "head_L%d.pt" % Lb), weights_only=False)
        lg = apply(h, load(a.feats, "dev", Lb), dev)
        T = fit_T(lg, ydv)
        st = stats(lg, ydv, rdv, T)
        json.dump({"layers": res, "layer": Lb, "T": T, "dev_stats_at_T": st,
                   "dev_stats_at_T1": stats(lg, ydv, rdv, 1.0)}, open(os.path.join(a.out, "choice.json"), "w"),
                  indent=1)
        print("layer", Lb, "T", T, json.dumps(st, indent=1), flush=True)
    else:
        ch = json.load(open(os.path.join(a.out, "choice.json")))
        Lb, T = ch["layer"], ch["T"]
        h = torch.load(os.path.join(a.out, "head_L%d.pt" % Lb), weights_only=False)
        lg = apply(h, load(a.feats, a.split, Lb), dev) / T
        np.save(os.path.join(a.out, "%s_logits.npy" % a.split), lg.half().numpy())
        meta = json.load(open(os.path.join(a.feats, "%s_meta.json" % a.split)))
        rows = NM.bank(a.split, rhos=meta["rhos"])
        assert NM.digest(rows) == meta["digest"]
        json.dump({"split": a.split, "rhos": meta["rhos"], "digest": meta["digest"], "layer": Lb, "T": T},
                  open(os.path.join(a.out, "%s_logits_meta.json" % a.split), "w"))
        print("wrote", a.split, lg.shape, flush=True)


if __name__ == "__main__":
    main()
