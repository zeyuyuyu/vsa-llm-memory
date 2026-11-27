"""B31 LoRA fine-tunes with an equalised read-out (prereg/B31_encoder_equalised_baselines_prereg.md).

The fine-tune is given the same positions the memory arms get for the question: its classification head reads the
span-mean final states at the query span and at the candidate span (the spans the memory arms use) and the state at the
last non-pad token, each standardised with fixed training-split statistics of the initial model (no learned scale),
concatenated, then one linear layer to the two classes.

  --id_lambda 0   equalised LoRA (no identity labels)
  --id_lambda 1   LoRA + identity auxiliary loss: an identity head (one linear layer, R mentions over the 64 roles)
                  on the span-mean final state of EVERY symbol mention, trained with cross-entropy on the same
                  identity labels the naming head gets (all mentions), added to the task loss with weight 1

LoRA on every linear layer of the encoder (Phi-2: q, k, v, dense, fc1, fc2; Qwen3-8B: q, k, v, o, gate, up, down),
alpha = 2 x rank, dropout 0.05, bf16 base, fp32 adapters and heads; AdamW, weight decay 0, linear warm-up over 6% of the
steps then cosine decay to 0 at the last epoch; batch 8 (micro-batches with gradient accumulation if --mbs < 8);
gradient clipping 1.0. Every epoch scores a fixed training subset (the first 500 training instances, eval mode) and the
dev split; the state with the best dev accuracy is kept and, in final mode, scored once on the test split.
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402
import b31_prep as BP                                                       # noqa: E402
from b31_arms import layout, NCLS, NROLE                                    # noqa: E402

TARGETS = {"phi2": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
           "qwen8": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
NSUB = 500


def rows(enc, arena, split, tok):
    st = S.build(arena, split)
    out = []
    for s in st:
        ids, am, spans = S.tokenize(tok, s)
        out.append({"ids": ids, "spans": spans, "label": s["label"], "sym": [m[2] for m in s["mentions"]]})
    return out


class Clf(nn.Module):
    def __init__(self, enc, arena, rank, id_lambda, dtype=torch.bfloat16, grad_ckpt=False):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        path = "microsoft/phi-2" if enc == "phi2" else BP.QWEN8
        base = AutoModel.from_pretrained(path, torch_dtype=dtype)
        base.config.use_cache = False
        if grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none", target_modules=TARGETS[enc])
        self.llm = get_peft_model(base, cfg)
        hid = base.config.hidden_size
        self.arena = arena
        self.kinds, _, self.q, self.c, _ = layout(arena)
        self.head = nn.Linear(3 * hid, 2)
        self.id_lambda = id_lambda
        if id_lambda > 0:
            self.id_head = nn.Linear(hid, NCLS[arena])
        self.register_buffer("mu", torch.zeros(3 * hid))
        self.register_buffer("sd", torch.ones(3 * hid))
        self.register_buffer("is_r", torch.tensor([k == "R" for k in self.kinds]))

    def states(self, batch, dev, all_mentions=False):
        T = max(len(r["ids"]) for r in batch)
        ids = torch.zeros(len(batch), T, dtype=torch.long)
        am = torch.zeros(len(batch), T, dtype=torch.long)
        npos = len(self.kinds) if all_mentions else 2
        W = torch.zeros(len(batch), npos, T)
        for b, r in enumerate(batch):
            n = len(r["ids"]); ids[b, :n] = torch.tensor(r["ids"]); am[b, :n] = 1
            sp = r["spans"] if all_mentions else [r["spans"][self.q], r["spans"][self.c]]
            for j, (lo, hi) in enumerate(sp):
                W[b, j, lo:hi] = 1.0 / (hi - lo)
        h = self.llm(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state.float()
        P = torch.bmm(W.to(dev), h)                                         # span means
        last = h[torch.arange(h.size(0), device=dev), am.sum(1).to(dev) - 1]
        if all_mentions:
            qc = P[:, [self.q, self.c]]
        else:
            qc = P
        x = torch.cat([qc[:, 0], qc[:, 1], last], dim=-1)
        return x, (P if all_mentions else None)

    def forward(self, batch, dev):
        x, P = self.states(batch, dev, all_mentions=self.id_lambda > 0)
        return self.head((x - self.mu) / self.sd), P

    def id_logits(self, P):
        lg = self.id_head(P)
        if self.arena in ("A", "B"):
            lg = torch.where(self.is_r[None, :, None] & (torch.arange(lg.size(-1), device=lg.device) >= NROLE),
                             torch.full_like(lg, -float("inf")), lg)
        return lg


@torch.no_grad()
def fit_std(m, tr, dev, bs):
    m.eval(); X = []
    for k in range(0, len(tr), bs):
        X.append(m.states(tr[k:k + bs], dev)[0].float().cpu())
    X = torch.cat(X)
    m.mu.copy_(X.mean(0).to(dev)); m.sd.copy_((X.std(0) + 1e-4).to(dev))
    cos = F.normalize(X[:, -X.size(1) // 3:], dim=1); n = X.size(0)
    return {"last_state_mean_pairwise_cos": float(((cos @ cos.t()).sum() - n) / (n * (n - 1)))}


@torch.no_grad()
def evaluate(m, rs, dev, bs=16, id_acc=False):
    m.eval(); S_, Y, L, nid, nok = [], [], [], 0, 0
    for k in range(0, len(rs), bs):
        b = rs[k:k + bs]
        lg, P = m(b, dev)
        y = torch.tensor([r["label"] for r in b], device=dev)
        L.append(F.cross_entropy(lg.float(), y, reduction="none").cpu())
        S_.append((lg[:, 1] - lg[:, 0]).float().cpu()); Y.append(y.cpu())
        if id_acc and P is not None:
            pred = m.id_logits(P).argmax(-1).cpu()
            gold = torch.tensor([r["sym"] for r in b])
            nok += int((pred == gold).sum()); nid += gold.numel()
    s, y, l = torch.cat(S_).numpy(), torch.cat(Y).numpy(), torch.cat(L).numpy()
    pred = (s > 0).astype(int)
    out = {"acc": float((pred == y).mean()), "pos_rate": float(pred.mean()), "loss": float(l.mean())}
    if nid:
        out["id_acc_mentions"] = nok / nid
    return out, s


def run(a, seed, D_, dev):
    torch.manual_seed(seed)
    m = Clf(a.enc, a.arena, a.rank, a.id_lambda, grad_ckpt=a.grad_ckpt).to(dev)
    std_info = fit_std(m, D_["train"], dev, a.ebs)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    tr = D_["train"]
    steps_ep = math.ceil(len(tr) / a.bs)
    total = steps_ep * a.epochs
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    g = torch.Generator(); g.manual_seed(seed)
    rec = {"seed": seed, "trainable": sum(p.numel() for p in params), "epochs": [], "std": std_info}
    best, best_state = -1.0, None
    t0 = time.time()
    sub = tr[:NSUB]
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        tl, il, nb = 0.0, 0.0, 0
        for k in range(0, len(perm), a.bs):
            bidx = perm[k:k + a.bs]
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(bidx), a.mbs):
                b = [tr[i] for i in bidx[j:j + a.mbs]]
                lg, P = m(b, dev)
                y = torch.tensor([r["label"] for r in b], device=dev)
                loss = F.cross_entropy(lg.float(), y)
                tl += float(loss) * len(b) / len(bidx)
                if a.id_lambda > 0:
                    lgi = m.id_logits(P)
                    gold = torch.tensor([r["sym"] for r in b], device=dev)
                    li = F.cross_entropy(lgi.reshape(-1, lgi.size(-1)), gold.reshape(-1))
                    il += float(li) * len(b) / len(bidx)
                    loss = loss + a.id_lambda * li
                (loss * len(b) / len(bidx)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            nb += 1
        e = {"epoch": ep + 1, "train_loss_running": tl / nb, "id_loss_running": il / nb if a.id_lambda else None}
        e["train_sub"], _ = evaluate(m, sub, dev, a.ebs)
        e["dev"], sdev = evaluate(m, D_["dev"], dev, a.ebs, id_acc=a.id_lambda > 0)
        e["dev_scores"] = [round(float(x), 4) for x in sdev]
        e["secs"] = round(time.time() - t0, 1)
        rec["epochs"].append(e)
        print("  %s %s r%d lr%.0e id%g sd%d ep%d loss %.4f train_sub %.4f dev %.4f%s (%.0fs)" % (
            a.enc, a.arena, a.rank, a.lr, a.id_lambda, seed, ep + 1, e["train_loss_running"], e["train_sub"]["acc"],
            e["dev"]["acc"], " id %.4f" % e["dev"]["id_acc_mentions"] if "id_acc_mentions" in e["dev"] else "",
            e["secs"]), flush=True)
        if e["dev"]["acc"] > best:
            best = e["dev"]["acc"]; rec["best_epoch"] = ep + 1
            if a.mode == "final":
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()
                              if "lora_" in k or k.startswith("head") or k.startswith("id_head")}
        if not math.isfinite(e["train_loss_running"]):
            rec["diverged"] = True
            break
    if a.mode == "final":
        m.load_state_dict(best_state, strict=False)
        rec["test"], st = evaluate(m, D_["test"], dev, a.ebs, id_acc=a.id_lambda > 0)
        rec["test_scores"] = [round(float(x), 5) for x in st]
        rec["train_sub_at_best"] = rec["epochs"][rec["best_epoch"] - 1]["train_sub"]
        print("  TEST %s %s seed %d at dev-best epoch %d: %.4f" % (a.enc, a.arena, seed, rec["best_epoch"],
                                                                  rec["test"]["acc"]), flush=True)
    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    del m, opt; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, choices=["phi2", "qwen8"])
    ap.add_argument("--arena", required=True, choices=["A", "B", "C"])
    ap.add_argument("--mode", required=True, choices=["tune", "final", "smoke"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--id_lambda", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--ebs", type=int, default=16)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--n_train", type=int, default=None, help="smoke only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    if a.n_train and a.mode != "smoke":
        raise SystemExit("--n_train is for smoke runs only")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = BP.tokenizer(a.enc)
    D_ = {"train": rows(a.enc, a.arena, "train", tok)}
    if a.n_train:
        D_["train"] = D_["train"][:a.n_train]
    D_["dev"] = rows(a.enc, a.arena, "dev", tok) if a.mode != "smoke" else D_["train"][:64]
    if a.mode == "final":
        D_["test"] = rows(a.enc, a.arena, "test", tok)
    out = {"args": vars(a), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(a.device),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": {k: len(v) for k, v in D_.items()}, "max_len": {k: max(len(r["ids"]) for r in v) for k, v in D_.items()},
           "runs": []}
    print("B31 LoRA", json.dumps({k: out[k] for k in ("n", "max_len", "gpu")}), flush=True)
    for s in a.seeds:
        out["runs"].append(run(a, s, D_, a.device))
        json.dump(out, open(a.out + ".part", "w"))
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".part"):
        os.remove(a.out + ".part")


if __name__ == "__main__":
    main()
