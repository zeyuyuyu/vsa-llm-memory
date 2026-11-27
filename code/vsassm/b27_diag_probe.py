"""B27 diagnostic (train + dev only, CPU): can a linear head on the frozen final state of the last non-pad token fit the
training split, and how large is the shared component of that state? Explains the LoRA plateau; no test data."""
import sys, json, torch, numpy as np
sys.path.insert(0, "/path/to/work/b27")
import b27_common as C
M = C.load_arena_modules()
out = {}
for ar in sys.argv[1:]:
    def last(split):
        obj = torch.load(C.feats_path("/path/to/work/b27/feats", ar, split), weights_only=False)
        if ar == "A":
            ids, am, H, y = obj
        else:
            tok = C.tokenizer(); rows = C.build_rows(M, tok, ar, split); H = obj
            am = torch.stack([r["am"] for r in rows]); y = torch.tensor([r["label"] for r in rows])
        idx = am.sum(1).long() - 1
        return H[torch.arange(H.size(0)), idx].float(), y
    Xtr, ytr = last("train"); Xdv, ydv = last("dev")
    cos = torch.nn.functional.normalize(Xtr, dim=1); mc = float((cos @ cos.t()).mean())
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    res = {"mean_pairwise_cos_last_state": mc, "norm": float(Xtr.norm(dim=1).mean())}
    for name, (A, B) in {"raw": (Xtr, Xdv), "standardised": ((Xtr - mu) / sd, (Xdv - mu) / sd)}.items():
        w = torch.zeros(A.size(1), 2, requires_grad=True); b = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2)
        for it in range(2000):
            loss = torch.nn.functional.cross_entropy(A @ w + b, ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            res[name] = {"train_acc": float(((A @ w + b).argmax(1) == ytr).float().mean()),
                         "dev_acc": float(((B @ w + b).argmax(1) == ydv).float().mean()), "loss": float(loss)}
    out[ar] = res
    print(ar, json.dumps(res), flush=True)
