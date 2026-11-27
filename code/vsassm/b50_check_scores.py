"""B50 independent score implementation (second angle on the gate scores; prereg 'Checks from other angles').

Refits the two Mahalanobis models with sklearn's EmpiricalCovariance and LedoitWolf CLASSES (not the torch code in
b50_run.fitmaha), recomputes MSP log-odds / energy / max-logit from DISC's saved head weights in numpy, on cached
layer-20 states (B40's dev cache; B50's test cache), and compares with the scores logged live in the forward pass
(preds/<split>/discopen_s<s>.jsonl). Reports Spearman correlation, median relative difference, and agreement of the
gate decision at each registered threshold.
usage: python b50_check_scores.py <b40_root> <b50_root> <split: dev|test>
"""
import json, sys
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import b40_data as D

SEEDS = (0, 1, 2)
GATES = ("msp", "energy", "maha", "mahalw", "maxlogit")


def main():
    b40, b50, split = sys.argv[1:4]
    H = torch.load(f"{b40}/cache/N_train_L20.pt").float().numpy().astype(np.float64)
    y = np.array([D.OPS_TRAIN.index(o) for o in json.load(open(f"{b40}/cache/N_train_meta.json"))["op"]])
    mu = np.stack([H[y == c].mean(0) for c in range(len(D.OPS_TRAIN))])
    X = H - mu[y]
    P = {"maha": EmpiricalCovariance(assume_centered=False).fit(X).precision_,
         "mahalw": LedoitWolf(assume_centered=False).fit(X).precision_}
    cache = f"{b40}/cache/N_dev_L20.pt" if split == "dev" else f"{b50}/cache/N_test_L20.pt"
    Z = torch.load(cache).float().numpy().astype(np.float64)
    off = {}
    for k, Pk in P.items():
        d = np.stack([(((Z - mu[c]) @ Pk) * (Z - mu[c])).sum(1) for c in range(len(mu))], 1)
        off[k] = -d.min(1)
    thr = json.load(open(f"{b50}/out/thresholds.json"))["seeds"]
    rep = {}
    for s in SEEDS:
        sd = torch.load(f"{b40}/out/rd/N_disc_e300_s{s}/reader.pt")
        W, b = sd["head.weight"].double().numpy()[:8], sd["head.bias"].double().numpy()[:8]
        z = Z @ W.T + b
        top = z.max(1)
        rest = z.copy()
        rest[np.arange(len(z)), z.argmax(1)] = -np.inf
        o = dict(off, msp=top - logsumexp(rest, 1), energy=logsumexp(z, 1), maxlogit=top)
        live = [json.loads(l) for l in open(f"{b50}/preds/{'dev' if split == 'dev' else 'N'}/discopen_s{s}.jsonl")]  # test files live in preds/N
        assert len(live) == len(Z)
        rep[s] = {}
        for g in GATES:
            lv = np.array([r["mod"][g] for r in live])
            rho = spearmanr(lv, o[g]).correlation
            rel = np.median(np.abs(lv - o[g]) / (np.abs(o[g]) + 1e-9))
            t = thr[str(s)]["tau"][g]
            agree = float(np.mean((lv >= t) == (o[g] >= t)))
            rep[s][g] = {"spearman": float(rho), "median_rel_diff": float(rel), "gate_agree_at_tau": agree,
                         "n_disagree": int(np.sum((lv >= t) != (o[g] >= t)))}
            print(f"{split} seed {s} {g:8s} spearman {rho:.5f} median rel diff {rel:.2e} gate agreement at tau "
                  f"{agree:.4f} ({rep[s][g]['n_disagree']} items differ)")
    json.dump(rep, open(f"{b50}/out/check_scores_{split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
