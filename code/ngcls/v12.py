"""V-12 (prereg 64dfd34): replace my single-pass bundling strawman with the CANONICAL retrained HD
classifier, so the comparator is a method this literature actually uses.

Why this exists: my `hd` arm bundled class prototypes once and never retrained. That is the weakest HD
classifier there is, and it lost 16pp to a 5.1K linear probe -- which is what that strawman should do. The
standard pipeline in this literature (Imani et al.; Rahimi et al.) retrains: encode, bundle, then correct
prototypes on misclassified samples. OnlineHD (Hernandez-Cano et al., 2021) scales each correction by
(1 - cosine similarity) so confident errors move the prototype more. Omitting retraining understates HD, and
the omission favoured my own conclusion, which is the wrong direction for an error to point.

Configurations, all on the SAME frozen last-token features as every other arm:
  encoding  raw          the pooled LLM state used directly (what my original arm did)
            rp_sign      random projection to D dims followed by sign -- the standard HD encoding for
                         dense features, giving a bipolar hypervector
  training  single       bundle once, no correction (the original arm)
            retrain      N passes: for each misclassified sample add it to the correct prototype and
                         subtract it from the wrongly predicted one
            online       the same, scaled by (1 - cos) per OnlineHD
The best HD configuration becomes the comparator reported in the manuscript.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

from v7 import get_arena, featurize, Probe, train_pooled     # noqa: E402

ARENAS = ["variable_tracking", 1, 2, 16]          # the four V-7 admitted arenas
HID = 2560


def encode(X, mode, D, seed, device):
    if mode == "raw":
        return F.normalize(X.double(), dim=1)
    g = torch.Generator(device="cpu"); g.manual_seed(1000 + seed)
    W = torch.randn(X.size(1), D, generator=g).double().to(device)
    return F.normalize(torch.sign(X.double() @ W), dim=1)


def hd_fit(Htr, ytr, n_class, mode, passes, device):
    """bundle, then optionally correct. mode in {single, retrain, online}."""
    P = torch.zeros(n_class, Htr.size(1), dtype=torch.float64, device=device)
    for c in range(n_class):
        m = (ytr == c)
        if m.any():
            P[c] = Htr[m].sum(0)
    if mode == "single":
        return F.normalize(P, dim=1)
    for _ in range(passes):
        Pn = F.normalize(P, dim=1)
        sims = Htr @ Pn.t()
        pred = sims.argmax(1)
        wrong = (pred != ytr).nonzero(as_tuple=True)[0]
        if wrong.numel() == 0:
            break
        for i in wrong.tolist():
            h, yt, yp = Htr[i], int(ytr[i]), int(pred[i])
            if mode == "online":
                a_t = float(1.0 - sims[i, yt].item())
                a_p = float(1.0 - sims[i, yp].item())
                P[yt] += a_t * h
                P[yp] -= a_p * h
            else:
                P[yt] += h
                P[yp] -= h
    return F.normalize(P, dim=1)


def hd_eval(P, H, y):
    return float(((H @ P.t()).argmax(1) == y).double().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--dim", type=int, default=10000)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v12_logs/v12.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for name in ARENAS:
        texts, ys, n_class, maj, label = get_arena(name, a.n_train, a.n_val, a.n_test)
        llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                        torch_dtype=torch.float16).to(a.device).eval()
        for p in llm.parameters():
            p.requires_grad_(False)
        feats = {s: featurize(texts[s], ys[s], tok, llm, a.device, "last")
                 for s in ["train", "val", "test"]}
        del llm; torch.cuda.empty_cache()
        print("\n=== %s === n_class=%d majority=%.4f" % (label, n_class, maj), flush=True)

        rec = {"arena": label, "n_class": n_class, "majority": maj, "probe": [], "hd": {}}
        for s in a.seeds:
            r = train_pooled(Probe(n_class), feats, a.device, s)
            rec["probe"].append(r["test"])
        print("  probe            %s mean=%.4f" % ([round(x, 4) for x in rec["probe"]],
                                                   float(np.mean(rec["probe"]))), flush=True)

        Xtr = feats["train"]["pooled"].to(a.device)
        Xte = feats["test"]["pooled"].to(a.device)
        ytr = feats["train"]["y"].to(a.device)
        yte = feats["test"]["y"].to(a.device)
        for enc in ["raw", "rp_sign"]:
            for mode in ["single", "retrain", "online"]:
                accs = []
                for s in a.seeds:
                    Htr = encode(Xtr, enc, a.dim, s, a.device)
                    Hte = encode(Xte, enc, a.dim, s, a.device)
                    P = hd_fit(Htr, ytr, n_class, mode, a.passes, a.device)
                    accs.append(hd_eval(P, Hte, yte))
                    if enc == "raw" and mode == "single":
                        break            # deterministic: no seed dependence
                key = "%s/%s" % (enc, mode)
                rec["hd"][key] = accs
                print("  hd %-16s %s mean=%.4f" % (key, [round(x, 4) for x in accs],
                                                   float(np.mean(accs))), flush=True)
        rows.append(rec)
        del feats; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    # scorer, per prereg 64dfd34, written before the run
    print("\n=== V-12 SCORING ===", flush=True)
    keys = list(rows[0]["hd"].keys())

    def arena_mean(getter):
        return float(np.mean([getter(r) for r in rows]))

    probe_m = [arena_mean(lambda r, i=i: r["probe"][i]) for i in range(len(a.seeds))]
    print("  probe admitted-mean per seed: %s" % [round(x, 4) for x in probe_m], flush=True)
    best_key, best_gap = None, -9.9
    for k in keys:
        hm = [arena_mean(lambda r, i=i, k=k: r["hd"][k][min(i, len(r["hd"][k]) - 1)])
              for i in range(len(a.seeds))]
        gaps = [(hm[i] - probe_m[i]) * 100 for i in range(len(a.seeds))]
        print("  %-16s mean/seed %s   gap vs probe %s pp"
              % (k, [round(x, 4) for x in hm], [round(g, 2) for g in gaps]), flush=True)
        if min(gaps) > best_gap:
            best_gap, best_key = min(gaps), k
    print("\n  best HD configuration: %s (worst-seed gap %.2f pp)" % (best_key, best_gap), flush=True)
    if best_gap >= -2:
        v = "HD IS COMPETITIVE"
    elif best_gap <= -5:
        v = "HD REMAINS BEHIND"
    else:
        v = "INTERMEDIATE"
    print("  >>> V-12 VERDICT: %s" % v, flush=True)
    print("V12-DONE", flush=True)


if __name__ == "__main__":
    main()
