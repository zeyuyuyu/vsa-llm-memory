"""V-7 (prereg 7e03dd6): re-run the admission gate and every arm under the pooling V-6 selected.

V-6 measured, on identical frozen states, that reading at the FINAL NON-PAD TOKEN instead of taking a
masked mean adds 13.1pp on variable_tracking and 39.5pp on bAbI task 1. So every frozen-feature number in
this line was pooling-limited, and the "frozen representations are the bottleneck" reading was withdrawn.
This file redoes the comparison properly.

DISCLOSED MODIFICATION: HybridModel pools with a masked mean internally. Here the VSA arm is pooled at the
final token like every other arm, because a matched comparison requires all arms to be read the same way.
Leaving the VSA arm on the weaker reader while its baselines improve would manufacture the mirror image of
the error this project found in the published version.

Arenas: the V-3 generator's four tasks, and bAbI 1/2/3/16.
Arms: probe / parameter-matched MLP / HD bundling+cleanup (zero parameters) / the manuscript's VSA module.
Gate: majority + 3pp <= probe <= 0.95 in 3/3 seeds, recomputed under this pooling.
Bars: VSA SURVIVES iff >= +3pp over the best comparator on the admitted-task mean in 3/3; COLLAPSES iff
<= +1pp in 3/3. HD keeps its own bar against the probe.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

from v3 import make_splits, majority_baseline          # noqa: E402
from v5 import load_babi                               # noqa: E402

HID = 2560
SYNTH = ["variable_tracking", "rule_priority", "state_update", "binding_test"]
BABI = [1, 2, 3, 16]


def pool_states(h, am, mode):
    m = am.unsqueeze(-1).to(h.dtype)
    if mode == "mean":
        return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float()
    if mode == "last":
        idx = am.sum(1).long().clamp(min=1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx].float()
    raise ValueError(mode)


@torch.no_grad()
def featurize(texts, ys, tok, llm, device, mode, bs=32, max_len=256, keep_full=False):
    pooled, full, masks = [], [], []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt")
        ids, am = e["input_ids"].to(device), e["attention_mask"].to(device)
        h = llm(input_ids=ids, attention_mask=am).last_hidden_state
        pooled.append(pool_states(h, am, mode).cpu())
        if keep_full:
            full.append(h.half().cpu()); masks.append(am.cpu())
    out = {"pooled": torch.cat(pooled), "y": ys}
    if keep_full:
        out["full"] = torch.cat(full); out["mask"] = torch.cat(masks)
    return out


class Probe(nn.Module):
    def __init__(self, n):
        super().__init__(); self.fc = nn.Linear(HID, n)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(HID, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n))

    def forward(self, x):
        return self.net(x)


def train_pooled(model, feats, device, seed, epochs=60, bs=64, lr=1e-3, wd=1e-2):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr, ytr = feats["train"]["pooled"].to(device), feats["train"]["y"].to(device)

    def ev(s):
        model.eval()
        with torch.no_grad():
            X, y = feats[s]["pooled"].to(device), feats[s]["y"].to(device)
            return float((model(X).argmax(-1) == y).float().mean().item())

    g = torch.Generator(); g.manual_seed(seed)
    best_v, best_sd = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.size(0), generator=g).to(device)
        for k in range(0, Xtr.size(0), bs):
            i = perm[k:k + bs]
            loss = F.cross_entropy(model(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        v = ev("val")
        if v > best_v:
            best_v = v
            best_sd = {k2: t.detach().clone() for k2, t in model.state_dict().items()}
    model.load_state_dict(best_sd)
    return {"val": best_v, "test": ev("test"),
            "trainable": sum(p.numel() for p in model.parameters())}


def hd_readout(feats, n_class, device):
    Xtr = feats["train"]["pooled"].to(device).double()
    ytr = feats["train"]["y"].to(device)
    protos = torch.zeros(n_class, Xtr.size(1), dtype=torch.float64, device=device)
    for c in range(n_class):
        m = (ytr == c)
        if m.any():
            protos[c] = Xtr[m].sum(0)
    protos = F.normalize(protos, dim=1)

    def ev(s):
        X = F.normalize(feats[s]["pooled"].to(device).double(), dim=1)
        return float(((X @ protos.t()).argmax(1) == feats[s]["y"].to(device)).double().mean().item())

    return {"val": ev("val"), "test": ev("test"), "trainable": 0}


def train_vsa(hyb, feats, device, seed, mode, epochs=20, bs=16, lr=2e-4):
    torch.manual_seed(seed)
    params = [p for p in hyb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    def fwd(H, M):
        h, _ = hyb.apply_vsa_reasoning(H, M)
        return hyb.classifier(pool_states(h, M, mode))     # SAME pooling as every other arm

    def ev(s):
        hyb.eval(); c = t = 0
        with torch.no_grad():
            Fs, Ms, ys = feats[s]["full"], feats[s]["mask"], feats[s]["y"]
            for k in range(0, Fs.size(0), bs):
                H = Fs[k:k + bs].to(device).to(torch.float16)
                M = Ms[k:k + bs].to(device)
                c += (fwd(H, M).argmax(-1).cpu() == ys[k:k + bs]).sum().item(); t += M.size(0)
        return c / t

    Ftr, Mtr, ytr = feats["train"]["full"], feats["train"]["mask"], feats["train"]["y"]
    g = torch.Generator(); g.manual_seed(seed)
    best_v, best_sd = -1.0, None
    for _ in range(epochs):
        hyb.train()
        perm = torch.randperm(Ftr.size(0), generator=g)
        for k in range(0, Ftr.size(0), bs):
            i = perm[k:k + bs]
            H, M = Ftr[i].to(device).to(torch.float16), Mtr[i].to(device)
            loss = F.cross_entropy(fwd(H, M), ytr[i].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        v = ev("val")
        if v > best_v:
            best_v = v
            best_sd = {n: p.detach().clone() for n, p in hyb.named_parameters() if p.requires_grad}
    if best_sd:
        with torch.no_grad():
            for n, p in hyb.named_parameters():
                if n in best_sd:
                    p.copy_(best_sd[n])
    return {"val": best_v, "test": ev("test"),
            "trainable": sum(p.numel() for p in params)}


def get_arena(name, n_tr, n_va, n_te):
    if isinstance(name, int):
        d = load_babi(name, n_tr, n_va, n_te)
        texts = {k: [t for t, _ in d[k]] for k in ["train", "val", "test"]}
        ys = {k: torch.tensor([y for _, y in d[k]]) for k in ["train", "val", "test"]}
        n_class = len(d["vocab"])
        c = np.bincount(ys["test"].numpy(), minlength=n_class)
        return texts, ys, n_class, float(c.max() / c.sum()), "babi%d" % name
    sp = make_splits(name, n_tr, n_va, n_te, seed=0)
    texts = {k: [r["text"] for r in sp[k]] for k in ["train", "val", "test"]}
    ys = {k: torch.tensor([r["label"] for r in sp[k]]) for k in ["train", "val", "test"]}
    maj, _ = majority_baseline(sp["test"])
    return texts, ys, 2, maj, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="last", choices=["last", "mean"])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--mlp_hidden", type=int, default=4707)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="v7_logs/v7.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    from model import HybridModel
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for name in SYNTH + BABI:
        texts, ys, n_class, maj, label = get_arena(name, a.n_train, a.n_val, a.n_test)
        llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                        torch_dtype=torch.float16).to(a.device).eval()
        for p in llm.parameters():
            p.requires_grad_(False)
        feats = {s: featurize(texts[s], ys[s], tok, llm, a.device, a.pool, keep_full=True)
                 for s in ["train", "val", "test"]}
        del llm; torch.cuda.empty_cache()

        rec = {"arena": label, "pool": a.pool, "n_class": n_class, "majority": maj, "arms": {}}
        for s in a.seeds:
            r = train_pooled(Probe(n_class), feats, a.device, s)
            rec["arms"].setdefault("probe", []).append(r)
        pa = [r["test"] for r in rec["arms"]["probe"]]
        rec["admitted"] = bool(all(maj + 0.03 <= x <= 0.95 for x in pa))
        print("\n=== %s (pool=%s) === n_class=%d majority=%.4f  probe=%s  ADMITTED=%s"
              % (label, a.pool, n_class, maj, [round(x, 4) for x in pa], rec["admitted"]), flush=True)

        if rec["admitted"]:
            rec["arms"]["hd"] = [hd_readout(feats, n_class, a.device)]
            print("  hd (0 params) test=%.4f" % rec["arms"]["hd"][0]["test"], flush=True)
            for s in a.seeds:
                r = train_pooled(MLP(n_class, a.mlp_hidden), feats, a.device, s)
                rec["arms"].setdefault("mlp", []).append(r)
                print("  mlp sd%-4d test=%.4f trainable=%s" % (s, r["test"],
                      format(r["trainable"], ",")), flush=True)
            for s in a.seeds:
                torch.manual_seed(s)
                hyb = HybridModel(model_name="microsoft/phi-2", vsa_dim=1024, num_role_vectors=16,
                                  max_bindings=3, use_vsa=True, freeze_llm=True)
                in_f = getattr(hyb.classifier, "in_features", HID)
                hyb.classifier = nn.Linear(in_f, n_class)
                if hasattr(hyb, "llm"):
                    del hyb.llm
                hyb = hyb.to(a.device); torch.cuda.empty_cache()
                r = train_vsa(hyb, feats, a.device, s, a.pool)
                rec["arms"].setdefault("vsa", []).append(r)
                print("  vsa sd%-4d test=%.4f trainable=%s" % (s, r["test"],
                      format(r["trainable"], ",")), flush=True)
                del hyb; torch.cuda.empty_cache()
        rows.append(rec)
        del feats; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    adm = [r for r in rows if r["admitted"]]
    print("\n=== V-7 SCORING (pool=%s) ===" % a.pool, flush=True)
    print("  ADMITTED: %s   (of %d arenas)" % ([r["arena"] for r in adm], len(rows)), flush=True)
    if len(adm) < 2:
        print("  >>> VOID: fewer than two arenas admitted under this pooling.", flush=True)
        print("V7-DONE", flush=True); return

    def m(arm, si):
        return float(np.mean([r["arms"][arm][min(si, len(r["arms"][arm]) - 1)]["test"] for r in adm]))

    gaps = [(m("vsa", i) - max(m("probe", i), m("mlp", i), m("hd", i))) * 100
            for i in range(len(a.seeds))]
    print("  vsa - best{probe,mlp,hd} per seed (pp): %s" % [round(g, 2) for g in gaps], flush=True)
    v = "SURVIVES" if all(g >= 3 for g in gaps) else ("COLLAPSES" if all(g <= 1 for g in gaps)
                                                      else "INTERMEDIATE")
    print("  >>> VSA VERDICT: %s" % v, flush=True)
    hg = [(m("hd", i) - m("probe", i)) * 100 for i in range(len(a.seeds))]
    print("  hd - probe per seed (pp): %s  -> %s"
          % ([round(g, 2) for g in hg], "HD-COMPETITIVE" if all(g >= -2 for g in hg) else "NOT"),
          flush=True)
    print("V7-DONE", flush=True)


if __name__ == "__main__":
    main()
