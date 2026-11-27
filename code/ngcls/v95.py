"""V-95 (prereg 477dcbb): the cross-talk threshold for two-hop binding.

No training anywhere. The arena's orthonormal codebook rows are replaced by rows with cross-talk dialled to a
target,
    v_s = sqrt(1-c) * sym_s + sqrt(c) * u        (L2-normalised, u one fixed common direction)
which mimics the narrow cone that makes LLM states anisotropic. Everything else -- binding, unbinding,
cleanup, the two-hop chain -- is v13's, untouched. So cross-talk is the only variable, and this is a causal
manipulation rather than another property measured on a trained arm.

Reference points from V-93/V-94: supplied codebook -0.0001, a perfect classifier's continuous logits 0.3904,
raw span-pooled states 0.9149. Arena C floor 0.515, codebook arm 0.9244.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13


class ConeArm(v13.BindArm):
    """The parser arm with a cross-talk-controlled codebook. `vec` and `q_vec` both read the same rows, so
    the arm stays internally consistent -- only the geometry of the rows changes."""

    def __init__(self, seed, c):
        super().__init__("parser", seed)
        g = torch.Generator().manual_seed(seed + 4242)
        u = F.normalize(torch.randn(v13.VSA_DIM, generator=g), dim=-1)
        rows = np.sqrt(max(0.0, 1.0 - c)) * self.sym + np.sqrt(max(0.0, c)) * u.unsqueeze(0)
        self.sym.data = F.normalize(rows, dim=-1)

    def realised_crosstalk(self):
        S = self.sym @ self.sym.t()
        off = S[~torch.eye(S.shape[0], dtype=torch.bool, device=S.device)]
        return float(off.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_te", type=int, default=600)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="v95.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== V-95 cross-talk sweep (prereg 477dcbb) =====", flush=True)
    print("  references: codebook -0.0001 (acc 0.9244) | perfect-classifier logits 0.3904 | "
          "raw states 0.9149 | floor 0.515", flush=True)
    print("  P3: the largest c still clearing floor+.10 = 0.615 lies strictly BELOW 0.39", flush=True)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    out = {"prereg": "477dcbb", "curve": []}
    for c in a.cs:
        accs, xts = [], []
        for s in a.seeds:
            torch.manual_seed(s); np.random.seed(s)
            te = v13.Arena(tok, a.n_te, 5000 + s)
            H = v13.cache(te.rows, llm, dev)
            m = ConeArm(s, c).to(dev)
            xts.append(m.realised_crosstalk())
            with torch.no_grad():
                preds, ys = [], []
                for i in range(0, len(te.rows), a.bs):
                    rows = te.rows[i:i + a.bs]
                    preds.append(m(H[i:i + a.bs].to(dev), None, rows, dev).argmax(-1).cpu())
                    ys.append(torch.tensor([r["label"] for r in rows]))
                accs.append((torch.cat(preds) == torch.cat(ys)).float().mean().item())
        rec = {"target_c": c, "crosstalk": float(np.mean(xts)), "acc": float(np.mean(accs)),
               "acc_per_seed": accs}
        out["curve"].append(rec)
        print("  target c=%.2f  realised crosstalk %.4f  acc %.4f  %s"
              % (c, rec["crosstalk"], rec["acc"], [round(x, 4) for x in accs]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    # ---- scoring ----
    cur = out["curve"]
    print("", flush=True)
    print("===== V-95 SCORING =====", flush=True)
    a0 = cur[0]["acc"]
    p1 = abs(a0 - 0.9244) <= 0.03
    notes = ["P1 gate: c=0 reproduces the codebook arm (%.4f vs 0.9244): %s" % (a0, p1)]
    if not p1:
        notes.append("GATE FAILS: this sweep harness is not the one that produced 0.9244, so no reading is "
                     "taken from the curve.")
        out["verdict"] = " ; ".join(notes)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  >>> %s" % out["verdict"], flush=True)
        print("V95-DONE", flush=True)
        return

    accs = [r["acc"] for r in cur]
    viol = [(cur[i]["crosstalk"], cur[i + 1]["crosstalk"])
            for i in range(len(accs) - 1) if accs[i + 1] > accs[i] + 0.03]
    p2 = not viol
    notes.append("P2 monotone non-increasing in cross-talk (0.03 tolerance): %s%s"
                 % (p2, "" if p2 else " violations at %s" % viol))

    BAR = 0.615
    passing = [r for r in cur if r["acc"] >= BAR]
    thr = max((r["crosstalk"] for r in passing), default=float("nan"))
    thr_c = max((r["target_c"] for r in passing), default=float("nan"))
    notes.append("threshold: largest cross-talk still clearing %.3f is %.4f (target c=%.2f)"
                 % (BAR, thr, thr_c))
    p3 = thr < 0.39
    notes.append("P3 threshold < 0.39 (a perfect classifier's cosine): %s" % p3)
    if p3:
        notes.append("QUANTITATIVE JUSTIFICATION FOR THE HANDOVER: two-hop binding over 32 superposed pairs "
                     "survives only below cross-talk %.4f; the best continuous route available reaches "
                     "0.3904, which is on the wrong side. The discrete lookup is a requirement, not a "
                     "convenience." % thr)
    else:
        notes.append("P3 FAILS: cross-talk up to %.4f is tolerable, so codes at 0.3904 would suffice and "
                     "cross-talk does NOT explain why every continuous arm sat at the floor. The isotropy "
                     "mechanism from V-93/V-94 must be retracted, as the alignment story was in V-92, and "
                     "the reason continuous routes fail returns to unknown." % thr)
    out["verdict"] = " ; ".join(notes)
    out["threshold_crosstalk"] = thr
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-95 VERDICT: %s" % out["verdict"], flush=True)
    print("V95-DONE", flush=True)


if __name__ == "__main__":
    main()
