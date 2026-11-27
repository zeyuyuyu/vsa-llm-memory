"""B27 LoRA fine-tune of Phi-2 on the arenas (prereg/B27_arena_referee_baselines_prereg.md).

Same instances and split sizes as every other arm (A 2000/1000, B 1000/500, C 800/500 train/test) plus a 500-instance
dev split; LoRA on all linear layers of Phi-2 (q, k, v, dense, fc1, fc2), alpha = 2 x rank, dropout 0.05; a linear
head on the final state of the last non-pad token (the readout of the published B/C fine-tune and of HF sequence
classification); bf16 base, fp32 adapters and head; AdamW, weight decay 0, linear warm-up over 6% of the steps then
cosine decay to 0 at the last epoch; batch 8; gradient clipping 1.0. Every epoch scores the whole training split
(eval mode), the dev split and, in final mode only, the test split.

Inputs are the arena token ids exactly; each batch is cut to its longest unpadded row (the arenas pad on the right
and Phi-2 is causal, so this changes no state that is read).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b27_common as C                                                  # noqa: E402

M = C.load_arena_modules()
TARGETS = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]


def rows_for(arena, split, tok, n=None):
    r = C.build_rows(M, tok, arena, split)
    if arena == "A":
        out = [{"ids": it["input_ids"], "am": it["attention_mask"], "label": int(it["labels"])} for it in r.items]
    else:
        out = [{"ids": x["ids"], "am": x["am"], "label": int(x["label"])} for x in r]
    return out[:n] if n else out


class LoRAClf(nn.Module):
    def __init__(self, rank, dropout=0.05, readout="cls", dtype=torch.bfloat16, targets=None):
        super().__init__()
        from transformers import AutoModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        self.readout = readout
        if readout == "lm":
            # Amendment 2 option: answer-token readout through the pretrained LM head (logits of " no" and " yes" at
            # the last non-pad position), the standard causal-LM fine-tune; the "head" below is then unused.
            base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=dtype)
            tok = C.tokenizer()
            ids = [tok.encode(w, add_special_tokens=False) for w in (" no", " yes")]
            assert all(len(i) == 1 for i in ids), ids
            self.register_buffer("ans_ids", torch.tensor([i[0] for i in ids]))
        else:
            base = AutoModel.from_pretrained("microsoft/phi-2", torch_dtype=dtype)
        base.config.use_cache = False
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=dropout, bias="none",
                         target_modules=targets or TARGETS)
        self.llm = get_peft_model(base, cfg)
        self.head = nn.Linear(base.config.hidden_size, 2)
        # Amendment 1: fixed standardisation of the head input (training-split mean and std of the initial model's
        # last-token state; not trained). Off unless fit_std() is called.
        self.register_buffer("mu", torch.zeros(base.config.hidden_size))
        self.register_buffer("sd", torch.ones(base.config.hidden_size))
        # Amendment 2 option: non-affine batch normalisation of the head input (batch statistics in training, running
        # statistics in eval), which follows the drift of the shared component as LoRA trains.
        self.bn = None

    def state(self, ids, am):
        if self.readout == "lm":
            h = self.llm.get_base_model().model(input_ids=ids, attention_mask=am).last_hidden_state
        else:
            h = self.llm(input_ids=ids, attention_mask=am).last_hidden_state
        idx = am.sum(1).long().clamp(min=1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx].float()

    def forward(self, ids, am):
        x = self.state(ids, am)
        if self.readout == "lm":
            W = self.llm.get_base_model().lm_head
            return F.linear(x.to(W.weight.dtype), W.weight[self.ans_ids], W.bias[self.ans_ids]).float()
        x = self.bn(x) if self.bn is not None else (x - self.mu) / self.sd
        return self.head(x)

    def linear_probe(self, X, y, steps=2000, lr=1e-2):
        """LP-FT option: fit the head on the standardised frozen states first (full batch, Adam), as in the B27
        diagnostic probe; the fine-tune then starts from this head."""
        Z = (X - self.mu) / self.sd
        opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        with torch.enable_grad():
            for _ in range(steps):
                loss = F.cross_entropy(self.head(Z), y)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return float((self.head(Z).argmax(1) == y).float().mean())

    @torch.no_grad()
    def fit_std(self, rows, dev, bs=16):
        self.eval(); X = []
        for k in range(0, len(rows), bs):
            ids, am, _ = batch(rows[k:k + bs], dev)
            X.append(self.state(ids, am))
        X = torch.cat(X)
        self.mu.copy_(X.mean(0)); self.sd.copy_(X.std(0) + 1e-4)
        cos = F.normalize(X, dim=1); n = X.size(0)
        self._X = X
        return {"mean_pairwise_cos": float(((cos @ cos.t()).sum() - n) / (n * (n - 1))),
                "mean_norm": float(X.norm(dim=1).mean())}


def batch(rows, dev):
    am = torch.stack([r["am"] for r in rows])
    L = int(am.sum(1).max())
    ids = torch.stack([r["ids"] for r in rows])[:, :L]
    return ids.to(dev), am[:, :L].to(dev), torch.tensor([r["label"] for r in rows], device=dev)


@torch.no_grad()
def evaluate(m, rows, dev, bs=16):
    m.eval(); S, Y, L = [], [], []
    for k in range(0, len(rows), bs):
        ids, am, y = batch(rows[k:k + bs], dev)
        lg = m(ids, am).float()
        L.append(F.cross_entropy(lg, y, reduction="none").cpu())
        S.append((lg[:, 1] - lg[:, 0]).cpu()); Y.append(y.cpu())
    s, y, l = torch.cat(S).numpy(), torch.cat(Y).numpy(), torch.cat(L).numpy()
    pred = (s > 0).astype(int)
    return {"acc": float((pred == y).mean()), "pos_rate": float(pred.mean()), "loss": float(l.mean())}, s


def run(arena, seed, a, D, dev):
    torch.manual_seed(seed)
    m = LoRAClf(a.rank, readout=a.readout, dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype],
                targets=None if a.targets == "all" else ["q_proj", "k_proj", "v_proj", "dense"]).to(dev)
    if a.readout == "lm":
        for p in m.head.parameters():
            p.requires_grad_(False)
    std_info = None
    if a.head_norm in ("fixed", "bn") or a.lp_steps:
        std_info = m.fit_std(D["train"], dev)                            # eval mode, no RNG drawn
    if a.head_norm == "bn":
        m.bn = nn.BatchNorm1d(m.head.in_features, affine=False).to(dev)
    if a.lp_steps:
        y = torch.tensor([r["label"] for r in D["train"]], device=dev)
        std_info["lp_train_acc"] = m.linear_probe(m._X, y, steps=a.lp_steps)
    m._X = None
    params = [p for p in m.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    hp = list(m.head.parameters()); hid = set(id(p) for p in hp)
    rest = [p for p in params if id(p) not in hid]
    hp = [p for p in hp if p.requires_grad]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": hp, "lr": a.head_lr or a.lr}] if hp else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    tr = D["train"]
    steps_ep = math.ceil(len(tr) / a.bs)
    total = steps_ep * a.epochs
    warm = max(1, int(0.06 * total))
    if a.sched == "const":       # the published fine-tune (v15): constant rate, no warm-up
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    g = torch.Generator(); g.manual_seed(seed)
    rec = {"seed": seed, "trainable": n_par, "epochs": [], "head_std": std_info}
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        tl, nb = 0.0, 0
        for k in range(0, len(perm), a.bs):
            ids, am, y = batch([tr[i] for i in perm[k:k + a.bs]], dev)
            lg = m(ids, am)
            loss = F.cross_entropy(lg.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            tl += float(loss); nb += 1
            if not math.isfinite(float(loss)):
                break
        e = {"epoch": ep + 1, "train_loss_running": tl / max(nb, 1)}
        if (ep + 1) % a.score_train_every == 0 or ep + 1 == a.epochs:
            e["train"], _ = evaluate(m, tr, dev)
        for split in [s for s in ("dev", "test") if s in D]:
            e[split], s = evaluate(m, D[split], dev)
            if a.mode == "final":
                e[split + "_scores"] = [round(float(x), 5) for x in s]
        e["secs"] = round(time.time() - t0, 1)
        rec["epochs"].append(e)
        print("  %s r%d lr%.0e sd%d ep%d loss %.4f train %s %s (%.0fs)" % (
            arena, a.rank, a.lr, seed, ep + 1, e["train_loss_running"],
            "%.4f" % e["train"]["acc"] if "train" in e else "-",
            " ".join("%s %.4f" % (s, e[s]["acc"]) for s in ("dev", "test") if s in e), e["secs"]), flush=True)
        if not math.isfinite(e["train_loss_running"]):
            rec["diverged"] = True
            break
    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    del m, opt; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True, choices=["A", "B", "C"])
    ap.add_argument("--mode", required=True, choices=["tune", "final", "smoke"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n_train", type=int, default=None, help="smoke only")
    ap.add_argument("--head_std", action="store_true", help="amendment 1 (alias of --head_norm fixed)")
    ap.add_argument("--head_norm", default="none", choices=["none", "fixed", "bn"])
    ap.add_argument("--readout", default="cls", choices=["cls", "lm"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--targets", default="all", choices=["all", "attn"])
    ap.add_argument("--sched", default="cosine", choices=["cosine", "const"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--head_lr", type=float, default=None, help="head learning rate (default: --lr)")
    ap.add_argument("--lp_steps", type=int, default=0, help="LP-FT: full-batch head fit before fine-tuning")
    ap.add_argument("--score_train_every", type=int, default=1, help="amendment 2: score the training split every N "
                    "epochs and at the last epoch")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.head_std:
        a.head_norm = "fixed"
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    if a.n_train and a.mode != "smoke":
        raise SystemExit("--n_train is for smoke runs only")
    tok = C.tokenizer()
    D = {"train": rows_for(a.arena, "train", tok, a.n_train)}
    D["dev"] = rows_for(a.arena, "dev", tok)
    if a.mode == "final":
        D["test"] = rows_for(a.arena, "test", tok)
    out = {"args": vars(a), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(a.device),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": {k: len(v) for k, v in D.items()},
           "max_len": {k: int(max(int(r["am"].sum()) for r in v)) for k, v in D.items()},
           "runs": []}
    print("B27 LoRA", json.dumps({k: out[k] for k in ("n", "max_len", "gpu")}), flush=True)
    for s in a.seeds:
        out["runs"].append(run(a.arena, s, a, D, a.device))
        json.dump(out, open(a.out + ".part", "w"))
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(out, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".part"):
        os.remove(a.out + ".part")


if __name__ == "__main__":
    main()
