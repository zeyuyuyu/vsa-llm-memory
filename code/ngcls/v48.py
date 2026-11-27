"""V-48: recompute the bAbI task-2 floor with the PAPER'S estimator (prereg 7724eb3, bars registered first).

The manuscript quotes .203 for the task-2 floor. That came from v37.py's masked-mean pooling; the paper's
convention, in v13.Floor and v30.py, is the LAST non-pad token. On bAbI the two differ by forty points. Only the
floor is recomputed here -- probe_argmax, parser, the subset sizes and the expressiveness ceiling are read back
as banked from v37_logs/v37b.json and never touched.
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
import v37b                                                           # noqa: E402
from v5 import load_babi                                              # noqa: E402

QUOTED = 0.2032


def last_tok(H, r, device):
    """The paper's estimator, as v13.Floor and v30.py compute it."""
    idx = max(int(r["am"].sum()) - 1, 0)
    return H[idx].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--banked", default="v37_logs/v37b.json")
    ap.add_argument("--out", default="v48_logs/v48.json")
    a = ap.parse_args()

    banked = json.load(open(a.banked))
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    d = load_babi(2, a.n_train, 200, a.n_test)
    locs = {l: i for i, l in enumerate(d["vocab"])}
    persons, objs = {}, {}
    for text, _ in d["train"]:
        for _, k, s_, o, _, _ in v37b.events(text):
            persons.setdefault(s_, len(persons))
            if k != "move":
                objs.setdefault(o, len(objs))
    tr, _ = v37b.build(d["train"], tok, persons, locs, objs)
    te, _ = v37b.build(d["test"], tok, persons, locs, objs)
    trE = [r for r in tr if r["expressible"]]
    teE = [r for r in te if r["expressible"]]
    print("===== V-48  bAbI task-2 floor under the paper's estimator (last non-pad token) =====", flush=True)
    print("  expressible subsets: %d train / %d eval (banked: %s / %s)"
          % (len(trE), len(teE), banked.get("n_train_expressible"), banked.get("n_eval_expressible")),
          flush=True)
    Htr, Hte = v13.cache(trE, llm, a.device), v13.cache(teE, llm, a.device)
    ytr = [r["label"] for r in trE]
    yte = [r["label"] for r in teE]
    maj = Counter(yte).most_common(1)[0][1] / len(yte)

    accs, trains = [], []
    for seed in a.seeds:
        torch.manual_seed(seed)
        pr = nn.Linear(v13.HID, len(locs)).to(a.device)
        opt = torch.optim.AdamW(pr.parameters(), lr=1e-3)
        X = torch.stack([last_tok(Htr[b].to(a.device), r, a.device) for b, r in enumerate(trE)])
        Y = torch.tensor(ytr).to(a.device)
        for _ in range(a.probe_epochs):
            p = torch.randperm(X.size(0), device=a.device)
            for i in range(0, p.numel(), 256):
                s = p[i:i + 256]
                loss = F.cross_entropy(pr(X[s]), Y[s])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            trains.append(float((pr(X).argmax(1).cpu() == torch.tensor(ytr)).double().mean()))
            Xt = torch.stack([last_tok(Hte[b].to(a.device), r, a.device) for b, r in enumerate(teE)])
            accs.append(float((pr(Xt).argmax(1).cpu() == torch.tensor(yte)).double().mean()))
        print("  seed %d  floor(last-token)=%.4f  (train %.4f)" % (seed, accs[-1], trains[-1]), flush=True)

    pa = banked["arms"]["parser"]
    pq = banked["arms"]["probe_argmax"]
    fl_m, pa_m = float(np.mean(accs)), float(np.mean(pa))
    H = pa_m - fl_m
    out = {"quoted": QUOTED, "estimator": "last-non-pad-token (paper, as v13.Floor)",
           "floor_last_token": accs, "floor_train_acc": trains, "majority_eval": maj,
           "banked_parser": pa, "banked_probe_argmax": pq,
           "n_train_expressible": len(trE), "n_eval_expressible": len(teE),
           "floor_mean": fl_m, "parser_mean": pa_m, "headroom": H}
    print("\n===== V-48 SCORING (bars registered at 7724eb3) =====", flush=True)
    print("  quoted floor %.4f | last-token floor %s mean=%.4f | majority %.4f"
          % (QUOTED, [round(x, 4) for x in accs], fl_m, maj), flush=True)
    print("  parser (banked) %.4f -> headroom %.4f  (admission needs >= 0.20)" % (pa_m, H), flush=True)
    stands = all(abs(x - QUOTED) <= 0.05 for x in accs)
    wrong = all(x - QUOTED > 0.10 for x in accs)
    adm = H >= 0.20
    print("  (a) quoted number stands -> %s" % stands, flush=True)
    print("  (b) quoted number is wrong -> %s" % wrong, flush=True)
    print("  (c) admission survives -> %s" % adm, flush=True)
    if stands:
        v = "THE QUOTED NUMBER STANDS (the two estimators agree here; no manuscript change)"
    elif wrong and adm:
        v = ("THE QUOTED NUMBER IS WRONG, ADMISSION SURVIVES (correct the sentence to the paper's own "
             "convention and restate the headroom; the depth-two result itself holds)")
    elif wrong and not adm:
        v = ("THE QUOTED NUMBER IS WRONG AND ADMISSION FAILS (the depth-two natural-text comparison is VOID by "
             "the paper's own gate and the paragraph must say so)")
    else:
        v = "GRADED (report both and reconcile before editing)"
    out["verdict"] = v
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-48 VERDICT: %s" % v, flush=True)
    print("V48-DONE", flush=True)


if __name__ == "__main__":
    main()
