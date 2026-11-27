"""V-40: hashing inside the unsupervised principal subspace (prereg 8c7082f, bars registered before this file).

V-39's failure was attributed to a random hyperplane's sign being dominated by the ~2,524 symbol-irrelevant
directions of a 2,560-dimensional state. The consequence is to restrict the hash to the principal subspace.
PCA uses no labels, so the route stays label-free, and it is fitted on a HELD-OUT arena (seed + 9000) so no
fitting touches an evaluated row.

Registered in advance: V-24's .9999 was SUPERVISED decodability from 36 components and does not imply
unsupervised separability inside them.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402

PCA_ARMS = ("pca_hash_36", "pca_hash_128")
_Base = v13.BindArm
_FIT = {}


def fit_pca(tok, llm, device, seed, n=200):
    """Unsupervised fit on a held-out arena: span representations only, never a symbol id."""
    ar = v13.Arena(tok, n, seed + 9000)
    Hs = v13.cache(ar.rows, llm, device)
    X = []
    with torch.no_grad():
        for b, r in enumerate(ar.rows):
            H = Hs[b].to(device).float()
            for L in range(v13.HOPS):
                for aa, bb, spr, spf in r["ctx"][L]:
                    for sp in (spr, spf):
                        X.append(H[sp[0]:sp[1]].mean(0) if sp[1] > sp[0] else H[sp[0]])
    X = torch.stack(X, 0)
    mu = X.mean(0)
    Xc = X - mu
    U, S, V = torch.pca_lowrank(Xc, q=min(160, Xc.size(0) - 1, Xc.size(1)))
    print("    PCA fitted on %d held-out span vectors; top-36 variance share %.3f, top-128 %.3f"
          % (X.size(0), float((S[:36] ** 2).sum() / (S ** 2).sum()),
             float((S[:128] ** 2).sum() / (S ** 2).sum())), flush=True)
    return mu, V, S


class PCAHashArm(_Base):
    def __init__(self, mode, seed):
        super().__init__(mode if mode not in PCA_ARMS else "parser", seed)
        self.mode = mode
        if mode in PCA_ARMS:
            k = int(mode.rsplit("_", 1)[1])
            mu, V, S = _FIT[seed]
            self.register_buffer("mu", mu.cpu())
            self.register_buffer("W", (V[:, :k] / S[:k].clamp(min=1e-6)).cpu())   # project + whiten
            g = torch.Generator().manual_seed(seed + 777)
            self.register_buffer("R", torch.randn(v13.VSA_DIM, k, generator=g) / (k ** 0.5))

    def vec(self, H, sid, sp, device):
        if self.mode not in PCA_ARMS:
            return super().vec(H, sid, sp, device)
        h = self.sf(H, sp).to(device).float()
        z = (h - self.mu.to(device)) @ self.W.to(device)
        return F.normalize(torch.sign(self.R.to(device) @ z), dim=-1)

    def q_vec(self, sid, device):
        if self.mode not in PCA_ARMS:
            return super().q_vec(sid, device)
        raise RuntimeError("q_vec must not be reached for a label-free arm")

    def sims(self, H, rows, device):
        if self.mode not in PCA_ARMS:
            return super().sims(H, rows, device)
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            key = self.vec(H[b], None, r["q_span"], device)
            for L in range(v13.HOPS):
                mem = torch.stack([v13.bind(self.vec(H[b], bb, spf, device),
                                            self.vec(H[b], aa, spr, device))
                                   for aa, bb, spr, spf in r["ctx"][L]], 0).sum(0)
                hat = F.normalize(v13.unbind(mem, key), dim=-1)
                cands = torch.stack([self.vec(H[b], bb, spf, device)
                                     for aa, bb, spr, spf in r["ctx"][L]], 0)
                key = cands[int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())]
            out[b] = torch.dot(key, self.vec(H[b], None, r["cand_span"], device)).clamp(-1, 1)
        return out


def within(mode, seed, tok, llm, device, n=150, idh=None, sym=None):
    """Within-instance same/different-symbol code cosine, as in V-39b."""
    ar = v13.Arena(tok, n, seed)
    Hs = v13.cache(ar.rows, llm, device)
    arm = PCAHashArm(mode, seed).to(device) if mode in PCA_ARMS else None
    same, diff = [], []
    with torch.no_grad():
        for b, r in enumerate(ar.rows):
            H = Hs[b].to(device)
            by = {}
            for L in range(v13.HOPS):
                for aa, bb, spr, spf in r["ctx"][L]:
                    for sid, sp in ((aa, spr), (bb, spf)):
                        if arm is not None:
                            v = arm.vec(H, sid, sp, device)
                        else:
                            f = H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()
                            v = sym[int(idh(f).argmax().item())]
                        by.setdefault(sid, []).append(v)
            for v in [x for x in by.values() if len(x) > 1]:
                V = torch.stack(v[:6], 0)
                C = V @ V.T
                same.append(float((C.sum() - C.diag().sum()) / max(1, C.numel() - C.size(0))))
            ks = list(by.keys())
            for i in range(min(12, len(ks))):
                for j in range(i + 1, min(12, len(ks))):
                    diff.append(float(torch.dot(by[ks[i]][0], by[ks[j]][0])))
    return float(np.mean(same)), float(np.mean(diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["floor", "parser", "probe_argmax", "pca_hash_36", "pca_hash_128"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v40_logs/v40.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    print("===== V-40  label-free hashing INSIDE the principal subspace =====", flush=True)
    for s in a.seeds:
        _FIT[s] = fit_pca(tok, llm, a.device, s)
    v13.BindArm = PCAHashArm
    out = {"arms": {}, "diagnostic": {}}
    for arm in a.arms:
        accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                        a.epochs, a.batch_size, a.lr)["acc"] for s in a.seeds]
        out["arms"][arm] = accs
        print("  %-13s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    for arm in PCA_ARMS:
        if arm in out["arms"]:
            s_, d_ = within(arm, a.seeds[0], tok, llm, a.device)
            out["diagnostic"][arm] = {"same_within": s_, "diff_within": d_, "gap": s_ - d_}
            print("  DIAG %-13s within-instance same=%.4f diff=%.4f gap=%.4f" % (arm, s_, d_, s_ - d_),
                  flush=True)

    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    fl, pa = m["floor"], m["parser"]
    Hd = pa - fl
    out.update({"floor": fl, "parser": pa, "headroom": Hd})
    print("\n===== V-40 SCORING (bars registered at 8c7082f) =====", flush=True)
    print("  floor=%.4f parser=%.4f H=%.4f  works >= %.4f partial >= %.4f fails <= %.4f"
          % (fl, pa, Hd, fl + 0.90 * Hd, fl + 0.50 * Hd, fl + 0.10 * Hd), flush=True)
    if Hd < 0.20:
        v = "VOID (admission failed: parser - floor = %.4f)" % Hd
    else:
        best = max((x for x in PCA_ARMS if x in out["arms"]), key=lambda x: m[x])
        acc = out["arms"][best]
        if all(x >= fl + 0.90 * Hd for x in acc):
            v = "LABEL-FREE CODE WORKS via %s (the supervision limitation is removed, not measured)" % best
        elif all(x >= fl + 0.50 * Hd for x in acc):
            v = "PARTIAL via %s (the limitation stays, narrowed)" % best
        elif all(all(x <= fl + 0.10 * Hd for x in out["arms"][arm]) for arm in PCA_ARMS if arm in out["arms"]):
            v = ("FAILS (three independent label-free routes have now failed -- k-means, raw hashing, subspace "
                 "hashing -- so the limitation is a measured property of the frozen geometry)")
        else:
            v = "GRADED (report as measured; license neither reading)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-40 VERDICT: %s" % v, flush=True)
    print("V40-DONE", flush=True)


if __name__ == "__main__":
    main()
