"""V-5 (prereg 8706a5b): the public-benchmark gate and the HD/VSA-native comparator.

bAbI tasks 1 / 2 / 3 / 16, STANDARD multi-class protocol (predict the answer token over the task's answer
vocabulary). The binary recast used by the old cross-dataset table is not reused: it produced a CFQ number
below chance for a two-way task and was uninterpretable.

Four arms on identical frozen Phi-2 features, identical splits, seeds 42/123/456:
  probe  5.1K linear read-out on the pooled state                       -- the floor
  mlp    2-layer, parameter-matched to the VSA module                   -- separates structure/capacity
  hd     ZERO TRAINED PARAMETERS: bundle the pooled train features of each answer class into a prototype
         hypervector, classify by cosine cleanup                        -- the HD/VSA-native comparator
  vsa    the manuscript's own module: HybridModel(use_vsa=True), with its apply_vsa_reasoning and
         classifier driven from cached hidden states

FIDELITY NOTE on the vsa arm: HybridModel.forward is frozen-LLM -> apply_vsa_reasoning -> masked mean ->
classifier. Because the LLM is frozen, caching its hidden states once and then calling the SAME
apply_vsa_reasoning and classifier is exact, not an approximation, and about 8x cheaper. Two adaptations
are recorded: the LLM is deleted after featurisation to free memory, and the classifier head is resized
from 2 outputs to the task's answer-vocabulary size, which any multi-class use requires. The VSA module
itself is untouched.

ADMISSION GATE, applied per task before any arm comparison is read:
  majority + 3pp <= probe <= 0.95 in 3/3 seeds.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

TASKS = [1, 2, 3, 16]
MAX_LEN = 128
HID = 2560


# ------------------------------------------------------------------ data

def load_babi(task, n_train, n_val, n_test):
    from datasets import load_dataset
    d = load_dataset("Muennighoff/babi")

    def take(split, n):
        rows = [r for r in d[split] if r["task"] == task]
        return rows[:n]

    tr, va, te = take("train", n_train), take("validation", n_val), take("test", n_test)
    vocab = sorted({r["answer"] for r in tr})
    idx = {a: i for i, a in enumerate(vocab)}

    def prep(rows):
        keep = [(r["passage"].strip() + "\n" + r["question"].strip(), idx[r["answer"]])
                for r in rows if r["answer"] in idx]
        oov = len(rows) - len(keep)
        return keep, oov

    (Xtr, oov_tr), (Xva, oov_va), (Xte, oov_te) = prep(tr), prep(va), prep(te)
    return {"train": Xtr, "val": Xva, "test": Xte, "vocab": vocab,
            "oov": {"train": oov_tr, "val": oov_va, "test": oov_te}}


def majority_of(rows, n_class):
    c = np.bincount([y for _, y in rows], minlength=n_class)
    return float(c.max() / c.sum())


# ------------------------------------------------------------------ featurisation

@torch.no_grad()
def featurize(rows, tok, llm, device, bs=32, keep_full=False):
    """Returns pooled features (float32, on CPU) and, if asked, the full hidden states in fp16 on CPU."""
    pooled, full, masks, ys = [], [], [], []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        e = tok([t for t, _ in chunk], truncation=True, max_length=MAX_LEN,
                padding="max_length", return_tensors="pt")
        ids, am = e["input_ids"].to(device), e["attention_mask"].to(device)
        h = llm(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        pooled.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
        if keep_full:
            full.append(h.half().cpu()); masks.append(am.cpu())
        ys += [y for _, y in chunk]
    out = {"pooled": torch.cat(pooled), "y": torch.tensor(ys)}
    if keep_full:
        out["full"] = torch.cat(full); out["mask"] = torch.cat(masks)
    return out


# ------------------------------------------------------------------ arms

class Probe(nn.Module):
    def __init__(self, n_class):
        super().__init__(); self.fc = nn.Linear(HID, n_class)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, n_class, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(HID, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_class))

    def forward(self, x):
        return self.net(x)


def train_pooled(model, feats, device, seed, epochs=60, bs=64, lr=1e-3, wd=1e-2):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr, ytr = feats["train"]["pooled"].to(device), feats["train"]["y"].to(device)

    def ev(s):
        model.eval()
        X, y = feats[s]["pooled"].to(device), feats[s]["y"].to(device)
        with torch.no_grad():
            return float((model(X).argmax(-1) == y).float().mean().item())

    g = torch.Generator(); g.manual_seed(seed)
    best_v, best_sd, best_ep = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.size(0), generator=g).to(device)
        for k in range(0, Xtr.size(0), bs):
            i = perm[k:k + bs]
            loss = F.cross_entropy(model(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        v = ev("val")
        if v > best_v:
            best_v, best_ep = v, ep + 1
            best_sd = {k2: t.detach().clone() for k2, t in model.state_dict().items()}
    model.load_state_dict(best_sd)
    return {"val": best_v, "val_epoch": best_ep, "test": ev("test"),
            "trainable": sum(p.numel() for p in model.parameters())}


def hd_readout(feats, n_class, device):
    """Textbook HD: bundle per class, classify by cosine cleanup. No trained parameters, no seed
    dependence beyond the data itself."""
    Xtr, ytr = feats["train"]["pooled"].to(device).double(), feats["train"]["y"].to(device)
    protos = torch.zeros(n_class, Xtr.size(1), dtype=torch.float64, device=device)
    for c in range(n_class):
        m = (ytr == c)
        if m.any():
            protos[c] = Xtr[m].sum(0)
    protos = F.normalize(protos, dim=1)

    def ev(s):
        X = F.normalize(feats[s]["pooled"].to(device).double(), dim=1)
        y = feats[s]["y"].to(device)
        return float(((X @ protos.t()).argmax(1) == y).double().mean().item())

    return {"val": ev("val"), "test": ev("test"), "trainable": 0, "val_epoch": 0}


def train_vsa(hyb, feats, device, seed, epochs=20, bs=16, lr=2e-4):
    """Drive the manuscript's own apply_vsa_reasoning + classifier from cached hidden states."""
    torch.manual_seed(seed)
    params = [p for p in hyb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    def fwd(H, M):
        h, _ = hyb.apply_vsa_reasoning(H, M)
        h = h.float()
        m = M.unsqueeze(-1).float()
        return hyb.classifier((h * m).sum(1) / m.sum(1).clamp(min=1))

    def ev(s):
        hyb.eval(); c = t = 0
        with torch.no_grad():
            F_, M_, y_ = feats[s]["full"], feats[s]["mask"], feats[s]["y"]
            for k in range(0, F_.size(0), bs):
                H = F_[k:k + bs].to(device).to(torch.float16)
                M = M_[k:k + bs].to(device)
                c += (fwd(H, M).argmax(-1).cpu() == y_[k:k + bs]).sum().item()
                t += M.size(0)
        return c / t

    Ftr, Mtr, ytr = feats["train"]["full"], feats["train"]["mask"], feats["train"]["y"]
    g = torch.Generator(); g.manual_seed(seed)
    best_v, best_sd, best_ep = -1.0, None, 0
    for ep in range(epochs):
        hyb.train()
        perm = torch.randperm(Ftr.size(0), generator=g)
        for k in range(0, Ftr.size(0), bs):
            i = perm[k:k + bs]
            H, M = Ftr[i].to(device).to(torch.float16), Mtr[i].to(device)
            loss = F.cross_entropy(fwd(H, M), ytr[i].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        v = ev("val")
        if v > best_v:
            best_v, best_ep = v, ep + 1
            best_sd = {n: p.detach().clone() for n, p in hyb.named_parameters() if p.requires_grad}
    if best_sd:
        with torch.no_grad():
            for n, p in hyb.named_parameters():
                if n in best_sd:
                    p.copy_(best_sd[n])
    return {"val": best_v, "val_epoch": best_ep, "test": ev("test"),
            "trainable": sum(p.numel() for p in params)}


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mlp_hidden", type=int, default=4707)
    ap.add_argument("--out", default="v5_logs/v5.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    from model import HybridModel
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for task in TASKS:
        d = load_babi(task, a.n_train, a.n_val, a.n_test)
        n_class = len(d["vocab"])
        maj = majority_of(d["test"], n_class)
        print("\n=== bAbI task %d ===  |vocab|=%d  n=%d/%d/%d  majority(test)=%.4f  oov=%s"
              % (task, n_class, len(d["train"]), len(d["val"]), len(d["test"]), maj, d["oov"]),
              flush=True)

        llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                        torch_dtype=torch.float16).to(a.device).eval()
        for p in llm.parameters():
            p.requires_grad_(False)
        feats = {s: featurize(d[s], tok, llm, a.device, keep_full=True) for s in ["train", "val", "test"]}
        del llm; torch.cuda.empty_cache()

        rec = {"task": task, "n_class": n_class, "majority": maj, "arms": {}}
        for s in a.seeds:
            r = train_pooled(Probe(n_class), feats, a.device, s)
            rec["arms"].setdefault("probe", []).append(r)
            print("  probe sd%-4d test=%.4f (majority %.4f) trainable=%s"
                  % (s, r["test"], maj, format(r["trainable"], ",")), flush=True)
        probe_acc = [r["test"] for r in rec["arms"]["probe"]]
        admitted = all(maj + 0.03 <= x <= 0.95 for x in probe_acc)
        rec["admitted"] = bool(admitted)
        print("  ADMITTED=%s (probe %s)" % (admitted, [round(x, 4) for x in probe_acc]), flush=True)

        if admitted:
            rec["arms"]["hd"] = [hd_readout(feats, n_class, a.device)]
            print("  hd (0 params) test=%.4f" % rec["arms"]["hd"][0]["test"], flush=True)
            for s in a.seeds:
                r = train_pooled(MLP(n_class, a.mlp_hidden), feats, a.device, s)
                rec["arms"].setdefault("mlp", []).append(r)
                print("  mlp sd%-4d test=%.4f trainable=%s"
                      % (s, r["test"], format(r["trainable"], ",")), flush=True)
            for s in a.seeds:
                torch.manual_seed(s)
                hyb = HybridModel(model_name="microsoft/phi-2", vsa_dim=1024, num_role_vectors=16,
                                  max_bindings=3, use_vsa=True, freeze_llm=True)
                in_f = hyb.classifier.in_features if hasattr(hyb.classifier, "in_features") else HID
                hyb.classifier = nn.Linear(in_f, n_class)
                if hasattr(hyb, "llm"):
                    del hyb.llm                      # features are cached; the LLM is dead weight
                hyb = hyb.to(a.device); torch.cuda.empty_cache()
                r = train_vsa(hyb, feats, a.device, s)
                rec["arms"].setdefault("vsa", []).append(r)
                print("  vsa sd%-4d test=%.4f trainable=%s"
                      % (s, r["test"], format(r["trainable"], ",")), flush=True)
                del hyb; torch.cuda.empty_cache()
        rows.append(rec)
        del feats; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    # scorer, per prereg 8706a5b, written before the run
    adm = [r for r in rows if r["admitted"]]
    print("\n  ADMITTED TASKS: %s" % [r["task"] for r in adm], flush=True)
    if len(adm) < 2:
        print("  >>> VOID: fewer than two bAbI tasks admitted; the public-benchmark gate yields no "
              "verdict on B1, per the prereg.", flush=True)
        print("V5-DONE", flush=True); return

    def mean_over_tasks(arm, si):
        vals = []
        for r in adm:
            lst = r["arms"].get(arm, [])
            vals.append(lst[min(si, len(lst) - 1)]["test"])
        return float(np.mean(vals))

    gaps = []
    for si in range(len(a.seeds)):
        v = mean_over_tasks("vsa", si)
        best = max(mean_over_tasks(k, si) for k in ["probe", "mlp", "hd"])
        gaps.append((v - best) * 100)
    print("  B1 gap vsa - best{probe,mlp,hd}, per seed (pp): %s" % [round(g, 2) for g in gaps],
          flush=True)
    b1 = "SURVIVES" if all(g >= 3 for g in gaps) else ("COLLAPSES" if all(g <= 1 for g in gaps)
                                                       else "INTERMEDIATE")
    print("  >>> B1 VERDICT: %s" % b1, flush=True)
    hdgap = [(mean_over_tasks("hd", si) - mean_over_tasks("probe", si)) * 100
             for si in range(len(a.seeds))]
    print("  B2 gap hd - probe, per seed (pp): %s" % [round(g, 2) for g in hdgap], flush=True)
    print("  >>> B2 VERDICT: %s" % ("HD-COMPETITIVE" if all(g >= -2 for g in hdgap) else "NOT"),
          flush=True)
    print("V5-DONE", flush=True)


if __name__ == "__main__":
    main()
