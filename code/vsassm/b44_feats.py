"""B44 frozen-state features (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md), B31's recipe (b31_prep.feats):
the span mean of every mention's final-layer state (Qwen3-8B: AutoModel, bf16, last_hidden_state after the final norm;
Phi-2: fp16 last_hidden_state), the state at the last token and the mean over the input. Span means are computed in
fp32 as H[s:e].float().mean(0) and STORED IN FP16 (N x M x d memmap, .npy) because the 80,000-instance sets would need
170 GB in fp32; last / mean are stored in fp32 (.pt). Inputs come from the b44_tok cache (no truncation, no padding;
instances are batched by length with right padding and an attention mask).

  python b44_feats.py --enc qwen8 --items std:train std:dev std:test ho:train ho:dev ho:test
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_prep as BP                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="qwen8", choices=["qwen8", "phi2"])
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--feats", default=os.path.join(HERE, "feats"))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=None, help="with --nchunks: compute only this contiguous chunk")
    ap.add_argument("--nchunks", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.feats, exist_ok=True)
    from transformers import AutoModel
    if a.enc == "qwen8":
        llm = AutoModel.from_pretrained(BP.QWEN8, torch_dtype=torch.bfloat16).to(a.device).eval()
    else:
        llm = AutoModel.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    pad = 0
    for item in a.items:
        tag = "%s_%s" % (a.enc, item.replace(":", "_"))
        T = torch.load(os.path.join(a.data, tag + ".pt"), weights_only=False)
        N0, M = T["n"], len(T["spans"][0])
        lo, hi = 0, N0
        ctag = tag
        if a.nchunks:                     # contiguous chunk [lo, hi) of the split, written to its own files
            lo, hi = a.chunk * N0 // a.nchunks, (a.chunk + 1) * N0 // a.nchunks
            ctag = "%s.c%dof%d" % (tag, a.chunk, a.nchunks)
            for k_ in ("ids", "spans", "tgt", "chain"):
                T[k_] = T[k_][lo:hi]
            T["label"] = T["label"][lo:hi]; T["sym"] = T["sym"][lo:hi]
        fp, sp = os.path.join(a.feats, ctag + ".F16.npy"), os.path.join(a.feats, ctag + ".pt")
        if os.path.exists(sp):
            print("exists", sp); continue
        N = hi - lo
        d = llm.config.hidden_size
        Fm = np.lib.format.open_memmap(fp + ".tmp", mode="w+", dtype=np.float16, shape=(N, M, d))
        last = torch.zeros(N, d); mean = torch.zeros(N, d); lens = torch.zeros(N, dtype=torch.long)
        order = sorted(range(N), key=lambda i: len(T["ids"][i]))
        t0 = time.time()
        with torch.no_grad():
            for k in range(0, N, a.bs):
                idx = order[k:k + a.bs]
                L = max(len(T["ids"][i]) for i in idx)
                ids = torch.full((len(idx), L), pad, dtype=torch.long); am = torch.zeros(len(idx), L, dtype=torch.long)
                for b, i in enumerate(idx):
                    n = len(T["ids"][i]); ids[b, :n] = T["ids"][i].long(); am[b, :n] = 1
                H = llm(input_ids=ids.to(a.device), attention_mask=am.to(a.device)).last_hidden_state
                W = torch.zeros(len(idx), M, L)
                for b, i in enumerate(idx):
                    for j, (s, e) in enumerate(T["spans"][i].tolist()):
                        W[b, j, s:max(e, s + 1)] = 1.0 / max(e - s, 1)
                P = torch.bmm(W.to(a.device), H.float())                  # span means, fp32
                Ph = P.half().cpu().numpy()
                for b, i in enumerate(idx):
                    n = len(T["ids"][i])
                    Fm[i] = Ph[b]
                    last[i] = H[b, n - 1].float().cpu(); mean[i] = H[b, :n].float().mean(0).cpu(); lens[i] = n
                if k % (a.bs * 500) == 0:
                    print("  %s %d/%d %.0fs" % (tag, k, N, time.time() - t0), flush=True)
        Fm.flush(); del Fm
        os.replace(fp + ".tmp", fp)
        obj = {"last": last, "mean": mean, "len": lens, "labels": T["label"], "sym": T["sym"], "text_md5": T["text_md5"],
               "spans": T["spans"], "enc": a.enc, "item": item, "n": N, "M": M, "range": [lo, hi], "n_split": N0}
        torch.save(obj, sp + ".tmp"); os.replace(sp + ".tmp", sp)
        print("feats", ctag, (N, M, d), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
