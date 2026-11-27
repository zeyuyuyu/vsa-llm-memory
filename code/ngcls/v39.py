"""V-39: a label-free code from injectivity alone (prereg 47fc3ce, bars registered before this file existed).

V-36 narrowed the requirement from consistency to injectivity. Any DETERMINISTIC map from state to code is
consistent for free, so the open question is whether an unsupervised map can be injective enough. V-23's k-means
failed, but clustering demands compact clusters and injectivity does not; V-24 measured the geometry as linearly
separable (.9999 from 36 PCs) and not compact (purity .31), which is the regime where hashing should work.

Arms are installed by subclassing v13.BindArm and swapping it into v13, so the evaluation path -- arena
construction, caching, two-hop retrieval, cleanup, threshold sweep -- is byte-identical to the mechanism table's.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402

LSH_ARMS = ("lsh_raw", "lsh_inst")
_Base = v13.BindArm


class LSHArm(_Base):
    def __init__(self, mode, seed):
        super().__init__(mode if mode not in LSH_ARMS else "parser", seed)
        self.mode = mode
        if mode in LSH_ARMS:
            g = torch.Generator().manual_seed(seed + 777)
            self.register_buffer("R", torch.randn(v13.VSA_DIM, v13.HID, generator=g) / (v13.HID ** 0.5))

    def vec(self, H, sid, sp, device):
        if self.mode not in LSH_ARMS:
            return super().vec(H, sid, sp, device)
        h = self.sf(H, sp)
        if self.mode == "lsh_inst":
            m = H.float()
            h = (h - m.mean(0)) / m.std(0).clamp(min=1e-6)
        return F.normalize(torch.sign(self.R.to(device) @ h.to(device)), dim=-1)

    def q_vec(self, sid, device):
        if self.mode not in LSH_ARMS:
            return super().q_vec(sid, device)
        raise RuntimeError("q_vec must not be reached for a label-free arm: it would read the generator's id")

    def sims(self, H, rows, device):
        if self.mode not in LSH_ARMS:
            return super().sims(H, rows, device)
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            # the query key and the candidate are derived from THEIR OWN spans, never from an id
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


def diagnostic(mode, seed, tok, llm, device, n=120):
    """Same-symbol and different-symbol code cosine, so a failure is attributable."""
    v13.N_SYM, v13.N_CTX, v13.HOPS = v13.N_SYM, v13.N_CTX, v13.HOPS
    ar = v13.Arena(tok, n, seed)
    Hs = v13.cache(ar.rows, llm, device)
    arm = LSHArm(mode, seed).to(device)
    by = {}
    with torch.no_grad():
        for b, r in enumerate(ar.rows):
            H = Hs[b].to(device)
            for L in range(v13.HOPS):
                for aa, bb, spr, spf in r["ctx"][L]:
                    for sid, sp in ((aa, spr), (bb, spf)):
                        by.setdefault(sid, []).append(arm.vec(H, sid, sp, device))
    same, diff = [], []
    ks = [k for k, v in by.items() if len(v) > 1][:40]
    for k in ks:
        V = torch.stack(by[k][:8], 0)
        C = V @ V.T
        same.append(float((C.sum() - C.diag().sum()) / max(1, C.numel() - C.size(0))))
    for i in range(min(30, len(ks))):
        for j in range(i + 1, min(30, len(ks))):
            a_, b_ = by[ks[i]][0], by[ks[j]][0]
            diff.append(float(torch.dot(a_, b_)))
    return float(np.mean(same)) if same else float("nan"), float(np.mean(diff)) if diff else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["floor", "parser", "probe_argmax", "lsh_raw", "lsh_inst"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v39_logs/v39.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    v13.BindArm = LSHArm                      # identical evaluation path, one new arm family
    print("===== V-39  label-free codes by hashing: arena C, N_SYM=%d N_CTX=%d HOPS=%d ====="
          % (v13.N_SYM, v13.N_CTX, v13.HOPS), flush=True)
    out = {"arms": {}, "diagnostic": {}}
    for arm in a.arms:
        accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                        a.epochs, a.batch_size, a.lr)["acc"] for s in a.seeds]
        out["arms"][arm] = accs
        print("  %-12s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    for arm in LSH_ARMS:
        if arm in out["arms"]:
            s_, d_ = diagnostic(arm, a.seeds[0], tok, llm, a.device)
            out["diagnostic"][arm] = {"same_symbol_cos": s_, "diff_symbol_cos": d_}
            print("  DIAG %-9s same-symbol cos=%.4f  different-symbol cos=%.4f" % (arm, s_, d_), flush=True)

    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    fl, pa = m["floor"], m["parser"]
    Hd = pa - fl
    out.update({"floor": fl, "parser": pa, "headroom": Hd})
    print("\n===== V-39 SCORING (bars registered at 47fc3ce) =====", flush=True)
    print("  floor=%.4f parser=%.4f H=%.4f  works >= %.4f  partial >= %.4f  fails <= %.4f"
          % (fl, pa, Hd, fl + 0.90 * Hd, fl + 0.50 * Hd, fl + 0.10 * Hd), flush=True)
    if Hd < 0.20:
        v = "VOID (admission failed: parser - floor = %.4f)" % Hd
    else:
        best = max((arm for arm in LSH_ARMS if arm in out["arms"]), key=lambda x: m[x])
        acc = out["arms"][best]
        if all(x >= fl + 0.90 * Hd for x in acc):
            v = ("LABEL-FREE CODE WORKS via %s (an injective deterministic code suffices and need not be "
                 "learned; the supervision limitation is removed, not measured)" % best)
        elif all(x >= fl + 0.50 * Hd for x in acc):
            v = "PARTIAL via %s (the limitation stays, narrowed)" % best
        elif all(all(x <= fl + 0.10 * Hd for x in out["arms"][arm]) for arm in LSH_ARMS if arm in out["arms"]):
            v = ("FAILS (V-23's conclusion stands for a second independent reason; the diagnostic says which "
                 "geometric property is missing)")
        else:
            v = "GRADED (report as measured; license neither reading)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-39 VERDICT: %s" % v, flush=True)
    print("V39-DONE", flush=True)


if __name__ == "__main__":
    main()
