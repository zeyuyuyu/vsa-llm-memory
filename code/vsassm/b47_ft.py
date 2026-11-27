"""B47 Phi-2 fine-tunes on arena C (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md).

  --task gen      generative answer-token LoRA: the B31 LoRA (every linear layer of Phi-2: q, k, v, dense, fc1, fc2;
                  alpha 2r, dropout 0.05, bf16 base, fp32 adapters) on the causal LM; input = the arena text +
                  "\\nAnswer:" (B44's scratchpad input); target = the single token " yes" or " no"; loss = cross-entropy
                  over the full vocabulary at the answer position; score = logit(" yes") - logit(" no") there
                  (prediction: score > 0). The free arg-max over the vocabulary is logged beside it.
  --task fullgen  the same read-out with ALL weights trained (no LoRA): fp32 master weights, bf16 autocast, gradient
                  checkpointing.
  --task fullcls  B31's equalised classifier read-out (one linear layer on [query-span mean, candidate-span mean,
                  last-token state] of the final layer, standardised with the initial model's statistics on the first
                  min(n, 2,000) training instances) with ALL weights trained (secondary).
  --task scratch  B44's scratchpad LoRA (b44_lora.Scratch and b44_lora.run, unchanged) on Phi-2 (secondary).

Recipe (B31 / B44): AdamW, weight decay 0, warm-up 6% then cosine to 0 over the planned steps, batch 8, clip 1.0.
Planned budget (B44): 800 -> 8 epochs, 8,000 -> 4 epochs; 8 evaluation points evenly spaced; each scores the fixed
training subset (first 500 training instances) and the dev split; the dev-best state is kept. Early stop (B44): dev >=
0.995 at two consecutive evaluation points. --final scores the test split once from the dev-best state after training;
--score_test scores a finished run's saved dev-best state once (the seed-42 tuning run of a selected configuration).

Data (b47_tok.py, b44_tok format): variant k32 = B31's arena C (train seed 42, 800; dev 777; test 4242); variant std =
B44's std stream (train = first 8,000 of seed 44042; dev / test = B31's dev / test).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b44_lora as L                                                       # noqa: E402
import b31_prep as BP                                                      # noqa: E402

PHI2 = "microsoft/phi-2"
EPOCHS = {800: 8, 8000: 4}
NSUB = 500
SUFFIX = "\nAnswer:"
START = time.time()


def load_rows(root, variant, split, n=None):
    return L.load_rows(root, "phi2", variant, split, n)


# ------------------------------------------------------------------ models

class Gen(nn.Module):
    """Answer-token read-out through the LM head; LoRA (full=False) or all weights (full=True)."""
    def __init__(self, rank, tok, full=False):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.full = full
        if full:
            base = AutoModelForCausalLM.from_pretrained(PHI2, torch_dtype=torch.float32)
            base.config.use_cache = False
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.llm = base
            self.inner = base
        else:
            from peft import LoraConfig, get_peft_model
            base = AutoModelForCausalLM.from_pretrained(PHI2, torch_dtype=torch.bfloat16)
            base.config.use_cache = False
            self.llm = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                                       target_modules=L.TARGETS["phi2"]))
            self.inner = self.llm.base_model.model
        self.suffix = tok(SUFFIX, add_special_tokens=False)["input_ids"]
        y = tok(" yes", add_special_tokens=False)["input_ids"]; n = tok(" no", add_special_tokens=False)["input_ids"]
        assert len(y) == 1 and len(n) == 1, (y, n)
        self.yes, self.no = y[0], n[0]
        self.pad = tok.eos_token_id

    def last_logits(self, batch, dev):
        seqs = [r["ids"] + self.suffix for r in batch]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad, dtype=torch.long); am = torch.zeros(len(seqs), T, dtype=torch.long)
        for b, s in enumerate(seqs):                                   # right padding; the answer is read at the
            ids[b, :len(s)] = torch.tensor(s); am[b, :len(s)] = 1       # last real position
        ids, am = ids.to(dev), am.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.full):
            h = self.inner.model(input_ids=ids, attention_mask=am).last_hidden_state
            last = h[torch.arange(h.size(0), device=dev), am.sum(1) - 1]
            lg = self.inner.lm_head(last)
        return lg.float()

    def loss(self, batch, dev):
        tgt = torch.tensor([self.yes if r["label"] == 1 else self.no for r in batch], device=dev)
        return F.cross_entropy(self.last_logits(batch, dev), tgt)


class FullCls(nn.Module):
    """B31's equalised classifier read-out (b44_lora.Cls / b31_lora.Clf for arena C) with all weights trained."""
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        base = AutoModel.from_pretrained(PHI2, torch_dtype=torch.float32)
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.llm = base
        hid = base.config.hidden_size
        self.head = nn.Linear(3 * hid, 2)
        self.register_buffer("mu", torch.zeros(3 * hid))
        self.register_buffer("sd", torch.ones(3 * hid))

    def states(self, batch, dev):
        T = max(len(r["ids"]) for r in batch)
        ids = torch.zeros(len(batch), T, dtype=torch.long); am = torch.zeros(len(batch), T, dtype=torch.long)
        W = torch.zeros(len(batch), 2, T)
        for b, r in enumerate(batch):
            n = len(r["ids"]); ids[b, :n] = torch.tensor(r["ids"]); am[b, :n] = 1
            for j, (lo, hi) in enumerate([r["spans"][L.QPOS], r["spans"][L.CPOS]]):
                W[b, j, lo:hi] = 1.0 / (hi - lo)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.llm(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
        h = h.float()
        P = torch.bmm(W.to(dev), h)
        last = h[torch.arange(h.size(0), device=dev), am.sum(1).to(dev) - 1]
        return torch.cat([P[:, 0], P[:, 1], last], dim=-1)

    def forward(self, batch, dev):
        return self.head((self.states(batch, dev) - self.mu) / self.sd)

    def loss(self, batch, dev):
        y = torch.tensor([r["label"] for r in batch], device=dev)
        return F.cross_entropy(self(batch, dev).float(), y)


def build(a, tok):
    if a.task == "gen":
        return Gen(a.rank, tok, full=False)
    if a.task == "fullgen":
        return Gen(a.rank, tok, full=True)
    if a.task == "fullcls":
        return FullCls()
    raise ValueError(a.task)


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate(m, rs, dev, task, bs=16, keep=False):
    m.eval()
    if task == "fullcls":
        return L.evaluate(m, rs, dev, "cls", bs, keep)
    S_, A_ = [], []
    for k in range(0, len(rs), bs):
        lg = m.last_logits(rs[k:k + bs], dev)
        S_.append((lg[:, m.yes] - lg[:, m.no]).cpu()); A_.append(lg.argmax(-1).cpu())
    s = torch.cat(S_); am = torch.cat(A_)
    y = torch.tensor([r["label"] for r in rs])
    pred = (s > 0).long()
    yn = (am == m.yes) | (am == m.no)
    free_ok = ((am == m.yes) & (y == 1)) | ((am == m.no) & (y == 0))
    out = {"acc": float((pred == y).float().mean()), "pos_rate": float(pred.float().mean()),
           "free_yesno_rate": float(yn.float().mean()), "free_acc": float(free_ok.float().mean())}
    if keep:
        out["scores"] = [round(float(x), 5) for x in s]
        out["free_argmax"] = am.tolist()
    return out


# ------------------------------------------------------------------ training

def trainable_state(m, full):
    if full:
        return {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()}
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if "lora_" in k}


def run(a, tok, D_, dev):
    torch.manual_seed(a.seed)
    full = a.task in ("fullgen", "fullcls")
    m = build(a, tok).to(dev)
    tr = D_["train"]
    n = len(tr)
    epochs = EPOCHS[n]
    steps_ep = math.ceil(n / a.bs)
    total = steps_ep * epochs
    eval_every = max(1, total // 8)
    warm = max(1, int(0.06 * total))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ck = a.out + ".ckpt"
    best_state = None
    if os.path.exists(ck):
        C = torch.load(ck, weights_only=False)
        m.load_state_dict(C["model"], strict=False); opt.load_state_dict(C["opt"]); sched.load_state_dict(C["sched"])
        torch.set_rng_state(C["rng"]); torch.cuda.set_rng_state(C["cuda_rng"])
        rec, step, ep, perm = C["rec"], C["step"], C["ep"], C["perm"]
        del C
        print("  resumed at step %d / %d" % (step, total), flush=True)
    else:
        rec = {"seed": a.seed, "n_train": n, "epochs_planned": epochs, "total_steps": total, "eval_every": eval_every,
               "evals": [], "best_dev": -1.0, "best_step": None, "secs": 0.0, "segments": 0,
               "n_trainable": sum(p.numel() for p in params)}
        if a.task == "fullcls":
            rec["std"] = L.fit_std(m, tr[:min(n, 2000)], dev, a.ebs)
        step, ep, perm = 0, 0, None
    rec["segments"] += 1
    t0 = time.time(); seg_t0 = time.time(); seg_secs = None
    sub = tr[:NSUB]
    while step < total:
        if perm is None or step >= (ep + 1) * steps_ep:
            if perm is not None:
                ep += 1
            g = torch.Generator(); g.manual_seed(a.seed * 1000 + ep)
            perm = torch.randperm(n, generator=g).tolist()
        m.train()
        k = (step - ep * steps_ep) * a.bs
        bidx = perm[k:k + a.bs]
        opt.zero_grad(set_to_none=True)
        lsum = 0.0
        for j in range(0, len(bidx), a.mbs):
            b = [tr[i] for i in bidx[j:j + a.mbs]]
            loss = m.loss(b, dev)
            (loss * len(b) / len(bidx)).backward()
            lsum += float(loss) * len(b) / len(bidx)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        step += 1
        if step % 50 == 0:
            rec.setdefault("loss_trace", []).append(round(lsum, 5))
        if step % eval_every == 0 or step == total:
            e = {"step": step, "instances": step * a.bs, "epoch_float": round(step / steps_ep, 3),
                 "train_sub": evaluate(m, sub, dev, a.task, a.ebs), "dev": evaluate(m, D_["dev"], dev, a.task, a.ebs),
                 "secs": round(rec["secs"] + time.time() - t0, 1)}
            rec["evals"].append(e)
            print("  %s phi2 %s n%d r%d lr%.0e sd%d step %d/%d (ep %.2f) loss %.4f train_sub %.4f dev %.4f%s (%.0fs)" % (
                a.task, a.variant, n, a.rank, a.lr, a.seed, step, total, e["epoch_float"], lsum,
                e["train_sub"]["acc"], e["dev"]["acc"],
                " free %.3f/%.3f" % (e["dev"]["free_yesno_rate"], e["dev"]["free_acc"]) if a.task != "fullcls" else "",
                e["secs"]), flush=True)
            if e["dev"]["acc"] > rec["best_dev"]:
                rec["best_dev"], rec["best_step"] = e["dev"]["acc"], step
                if full:
                    best_state = trainable_state(m, True)          # kept in RAM; written at the end / at exit 3
                else:
                    torch.save(trainable_state(m, False), a.out + ".best.pt")
            if not math.isfinite(lsum):
                rec["diverged"] = True; break
            if len(rec["evals"]) >= 2 and all(x["dev"]["acc"] >= 0.995 for x in rec["evals"][-2:]):
                rec["early_stop"] = step; break
            rec["secs"] += time.time() - t0; t0 = time.time()
            seg = time.time() - seg_t0; seg_t0 = time.time()
            seg_secs = seg if seg_secs is None else max(seg_secs, seg)
            stop = a.max_hours and (time.time() - START) + 1.25 * seg_secs > a.max_hours * 3600 and step < total
            if not full or stop:                                       # LoRA: checkpoint at every evaluation point;
                if full and best_state is not None:                    # full: only when the segment ends
                    torch.save(best_state, a.out + ".best.pt")
                torch.save({"model": trainable_state(m, full), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "rec": rec,
                            "step": step, "ep": ep, "perm": perm}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
            if stop:
                print("  time budget reached at step %d; exit 3 for resume" % step, flush=True)
                return rec, 3
    rec["secs"] += time.time() - t0
    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    if full and best_state is not None:
        torch.save(best_state, a.out + ".best.pt")
    return rec, 0


def score_test(a, tok, D_, dev, rec_path):
    m = build(a, tok).to(dev)
    st = torch.load(a.out + ".best.pt", weights_only=False)
    missing = [k for k in st if k not in m.state_dict()]
    assert not missing, missing[:5]
    m.load_state_dict(st, strict=False)
    del st
    res = {"test": evaluate(m, D_["test"], dev, a.task, a.ebs, keep=True),
           "dev_recheck": evaluate(m, D_["dev"], dev, a.task, a.ebs)}
    rec = json.load(open(rec_path))
    res["dev_logged"] = rec["run"]["best_dev"]
    print("  TEST %s phi2 %s n%d r%d lr%.0e sd%d: test %.4f (dev recheck %.4f, logged %.4f)" % (
        a.task, a.variant, a.n, a.rank, a.lr, a.seed, res["test"]["acc"], res["dev_recheck"]["acc"], res["dev_logged"]),
        flush=True)
    return res


# ------------------------------------------------------------------ scratchpad (B44 code, unchanged)

def scratch_args(a):
    from types import SimpleNamespace
    return SimpleNamespace(task="scratch", enc="phi2", variant=a.variant, n=a.n, rank=a.rank, lr=a.lr, seed=a.seed,
                           bs=a.bs, mbs=a.mbs, ebs=a.ebs, max_hours=a.max_hours, out=a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gen", "fullgen", "fullcls", "scratch"])
    ap.add_argument("--variant", required=True, choices=["k32", "std"])
    ap.add_argument("--n", type=int, required=True, choices=[800, 8000])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--ebs", type=int, default=16)
    ap.add_argument("--max_hours", type=float, default=0.0)
    ap.add_argument("--score_test", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="train on the first N instances, dev = first 64 train; no "
                    "dev or test split is read")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", default="task", choices=["task", "candF", "lastpar"],
                    help="candF = EXPLORATORY positive control (prereg/B47_exploratory1.md): the label is replaced by "
                         "'the candidate is one of table F's fillers'; train and dev only, never with --final/--score_test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert (a.variant, a.n) in (("k32", 800), ("std", 8000)), "k32 = B31's 800; std = B44's first 8,000"
    assert a.label == "task" or not (a.final or a.score_test), "the positive control never reads the test split"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = BP.tokenizer("phi2")
    if a.task == "scratch":
        L.EPOCHS.update(EPOCHS)

    def test_dev():
        return {"test": load_rows(a.data, a.variant, "test")[0], "dev": load_rows(a.data, a.variant, "dev")[0]}

    def do_score():
        D2 = test_dev()
        if a.task == "scratch":
            res = L.score_test(scratch_args(a), tok, D2, a.device, a.out)
        else:
            res = score_test(a, tok, D2, a.device, a.out)
        res["args"] = vars(a); res["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tp = a.out + ".test.json"
        json.dump(res, open(tp + ".tmp", "w")); os.replace(tp + ".tmp", tp)

    if a.score_test:
        if os.path.exists(a.out + ".test.json"):
            print("exists, skip", a.out + ".test.json"); return
        assert os.path.exists(a.out), "run not finished"
        do_score(); return
    if os.path.exists(a.out):
        if a.final and not os.path.exists(a.out + ".test.json"):
            do_score(); return
        print("exists, skip", a.out); return
    tr, md5 = load_rows(a.data, a.variant, "train", a.n)
    assert len(tr) == a.n
    D_ = {"train": tr}
    if a.smoke:
        D_["train"] = tr[:a.smoke]; D_["dev"] = tr[:64]
        EPOCHS[a.smoke] = 1; L.EPOCHS[a.smoke] = 1
    else:
        D_["dev"] = load_rows(a.data, a.variant, "dev")[0]
    if a.label in ("candF", "lastpar"):
        import re as _re
        import b44_data as BD
        for sp in ("train", "dev"):
            st = BD.ksplit(32, sp, struct=False) if a.variant == "k32" else BD.split("std", sp, 8000 if sp == "train" else None, struct=False)
            assert len(st) == len(D_[sp])
            for r, s_ in zip(D_[sp], st):
                t = s_["text"]
                fv = {int(b) for _a, b in _re.findall(r"\[F([0-9A-F]{2})\]\{(\d+)\}", t)}
                cand = int(_re.search(r"Q \[R[0-9A-F]{2}\] \{(\d+)\} \?", t).group(1))
                r["label"] = int(cand in fv) if a.label == "candF" else int(int(_re.findall(r"\{(\d+)\}", t)[-1]) % 2 == 0)
        print("candF positive rate train %.3f dev %.3f" % (sum(r["label"] for r in D_["train"]) / len(D_["train"]),
                                                          sum(r["label"] for r in D_["dev"]) / len(D_["dev"])), flush=True)
    out = {"args": vars(a), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(a.device),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "train_md5_full": md5,
           "n": {k: len(v) for k, v in D_.items()}, "max_len": {k: max(len(r["ids"]) for r in v) for k, v in D_.items()}}
    print("B47 FT", json.dumps({k: out[k] for k in ("n", "max_len", "gpu")}), flush=True)
    if a.task == "scratch":
        rec, code = L.run(scratch_args(a), tok, D_, a.device)
    else:
        rec, code = run(a, tok, D_, a.device)
    if code == 3:
        sys.exit(3)
    out["run"] = rec
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".ckpt"):
        os.remove(a.out + ".ckpt")
    if a.final and not a.smoke:
        torch.cuda.empty_cache()
        do_score()


if __name__ == "__main__":
    main()
