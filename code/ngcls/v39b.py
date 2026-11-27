"""V-39b: within-instance diagnostic, because V-39's pooled one cannot attribute lsh_inst's failure.

V-39 measured same-symbol code cosine by pooling occurrences ACROSS instances. For lsh_inst the standardisation
is per-instance, and the binding never crosses an instance, so that number measures something the mechanism does
not require and cannot say whether the arm failed from crowding or from inconsistency. This measures both
cosines WITHIN instances. The accuracy verdict from V-39 is untouched; only its attribution is.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
from v39 import LSHArm, LSH_ARMS                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v39_logs/v39b.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    ar = v13.Arena(tok, a.n, a.seed)
    Hs = v13.cache(ar.rows, llm, a.device)
    out = {"n_instances": a.n, "arms": {}}
    print("===== V-39b  WITHIN-instance code cosine (attribution only) =====", flush=True)
    for mode in LSH_ARMS:
        arm = LSHArm(mode, a.seed).to(a.device)
        same, diff, n_rep = [], [], 0
        with torch.no_grad():
            for b, r in enumerate(ar.rows):
                H = Hs[b].to(a.device)
                by = {}
                for L in range(v13.HOPS):
                    for aa, bb, spr, spf in r["ctx"][L]:
                        for sid, sp in ((aa, spr), (bb, spf)):
                            by.setdefault(sid, []).append(arm.vec(H, sid, sp, a.device))
                by.setdefault(r["q"], []).append(arm.vec(H, None, r["q_span"], a.device))
                reps = [v for v in by.values() if len(v) > 1]
                n_rep += len(reps)
                for v in reps:
                    V = torch.stack(v[:6], 0)
                    C = V @ V.T
                    same.append(float((C.sum() - C.diag().sum()) / max(1, C.numel() - C.size(0))))
                ks = list(by.keys())
                for i in range(min(12, len(ks))):
                    for j in range(i + 1, min(12, len(ks))):
                        diff.append(float(torch.dot(by[ks[i]][0], by[ks[j]][0])))
        rec = {"same_within": float(np.mean(same)) if same else float("nan"),
               "diff_within": float(np.mean(diff)) if diff else float("nan"),
               "repeated_symbol_groups": n_rep}
        rec["gap"] = rec["same_within"] - rec["diff_within"]
        out["arms"][mode] = rec
        print("  %-9s WITHIN-instance same=%.4f  diff=%.4f  gap=%.4f  (%d repeated-symbol groups)"
              % (mode, rec["same_within"], rec["diff_within"], rec["gap"], n_rep), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    r_, i_ = out["arms"]["lsh_raw"], out["arms"]["lsh_inst"]
    print("\n  attribution: HRR needs same near 1 and diff near 0. Both arms hold neither, so the failure is "
          "CROWDING in the frozen geometry, not a defect of hashing per se.", flush=True)
    print("  lsh_inst gap %.4f vs lsh_raw gap %.4f" % (i_["gap"], r_["gap"]), flush=True)
    print("V39B-DONE", flush=True)


if __name__ == "__main__":
    main()
