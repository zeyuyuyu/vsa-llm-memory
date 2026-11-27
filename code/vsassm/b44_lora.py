"""B44 LoRA fine-tunes on arena C at 800 / 8,000 / 80,000 training instances (prereg/B44_data_scaling_heldout_cot_
crosstalk_prereg.md).

  --task cls      the B31 equalised LoRA (b31_lora.Clf, unchanged): LoRA on every linear layer, head on [query-span
                  mean, candidate-span mean, last-token state] of the final layer, standardised with fixed statistics of
                  the initial model (here: the first min(n, 2,000) training instances; = B31 at n = 800)
  --task scratch  scratchpad-supervised LoRA: the same LoRA on the causal LM (Qwen3ForCausalLM); input = the arena text
                  + "\\nAnswer:"; target = " <bridge decimal> F<bridge hex> <answer decimal> <yes|no>" + EOS; loss =
                  token cross-entropy on the target only; scored by greedy generation (max 16 new tokens), the label
                  read from the last yes/no of the generated text (none -> wrong)

Recipe (B31): rank r, alpha 2r, dropout 0.05, bf16 base, fp32 adapters / heads; AdamW, weight decay 0; warm-up 6% then
cosine to 0 over the planned steps; batch 8; clip 1.0. Planned budget: 800 -> 8 epochs, 8,000 -> 4 epochs, 80,000 -> 1
epoch; 8 evaluation points evenly spaced (every epoch / half epoch / 10,000 instances). Each evaluation scores a fixed
training subset (the first 500 training instances) and the dev split; the dev-best state is kept (saved to disk). Early
stop: dev >= 0.995 at two consecutive evaluation points. Resumable: a checkpoint (adapters, heads, optimiser,
scheduler, epoch permutation, position, RNG) is written at every evaluation point; with --max_hours the run exits with
code 3 when the next segment would not fit, and a rerun of the same command resumes.

  --score_test    load the saved dev-best state of a finished run and score the test split once (written to
                  <out>.test.json); used for the seed-42 tuning run of the selected configuration and for every final.
"""
import argparse, json, math, os, re, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_prep as BP                                                       # noqa: E402

TARGETS = {"phi2": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
           "qwen8": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
EPOCHS = {800: 8, 8000: 4, 80000: 1}
NSUB = 500
QPOS, CPOS = 128, 129                                     # arena C layout: query R and candidate F mentions
YESNO = re.compile(r"\b(yes|no)\b")


def load_rows(root, enc, variant, split, n=None):
    obj = torch.load(os.path.join(root, "%s_%s_%s.pt" % (enc, variant, split)), weights_only=False)
    N = n or obj["n"]
    rows = []
    for i in range(N):
        rows.append({"ids": obj["ids"][i].tolist(), "spans": obj["spans"][i].tolist(), "label": int(obj["label"][i]),
                     "tgt": obj["tgt"][i], "chain": obj["chain"][i]})
    return rows, obj["text_md5"]


# ------------------------------------------------------------------ models

class Cls(nn.Module):
    """b31_lora.Clf for arena C (id_lambda 0)."""
    def __init__(self, enc, rank, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        base = AutoModel.from_pretrained("microsoft/phi-2" if enc == "phi2" else BP.QWEN8, torch_dtype=dtype)
        base.config.use_cache = False
        self.llm = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                                   target_modules=TARGETS[enc]))
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
            for j, (lo, hi) in enumerate([r["spans"][QPOS], r["spans"][CPOS]]):
                W[b, j, lo:hi] = 1.0 / (hi - lo)
        h = self.llm(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state.float()
        P = torch.bmm(W.to(dev), h)
        last = h[torch.arange(h.size(0), device=dev), am.sum(1).to(dev) - 1]
        return torch.cat([P[:, 0], P[:, 1], last], dim=-1)

    def forward(self, batch, dev):
        return self.head((self.states(batch, dev) - self.mu) / self.sd)

    def loss(self, batch, dev):
        y = torch.tensor([r["label"] for r in batch], device=dev)
        return F.cross_entropy(self(batch, dev).float(), y)


class Scratch(nn.Module):
    def __init__(self, enc, rank, tok, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2" if enc == "phi2" else BP.QWEN8,
                                                    torch_dtype=dtype)
        base.config.use_cache = False
        self.llm = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                                   target_modules=TARGETS[enc]))
        self.tok = tok
        self.suffix = tok(BP_SUFFIX, add_special_tokens=False)["input_ids"]
        self.eos = tok.eos_token_id
        self.cache = {}

    def tgt_ids(self, r):
        k = r["tgt"]
        if k not in self.cache:
            self.cache[k] = self.tok(k, add_special_tokens=False)["input_ids"] + [self.eos]
        return self.cache[k]

    def loss(self, batch, dev):
        seqs, tpos = [], []
        for r in batch:
            p = r["ids"] + self.suffix
            t = self.tgt_ids(r)
            seqs.append(p + t); tpos.append((len(p), len(t)))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.eos, dtype=torch.long); am = torch.zeros(len(seqs), T, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s); am[b, :len(s)] = 1
        inner = self.llm.base_model.model
        h = inner.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
        sel_b, sel_t, gold = [], [], []
        for b, (p0, n) in enumerate(tpos):
            for j in range(n):                       # the state at position p0 + j - 1 predicts token p0 + j
                sel_b.append(b); sel_t.append(p0 + j - 1); gold.append(seqs[b][p0 + j])
        logits = inner.lm_head(h[torch.tensor(sel_b, device=dev), torch.tensor(sel_t, device=dev)]).float()
        return F.cross_entropy(logits, torch.tensor(gold, device=dev))

    @torch.no_grad()
    def generate(self, batch, dev, max_new=16):
        P = [r["ids"] + self.suffix for r in batch]
        T = max(len(p) for p in P)
        ids = torch.full((len(P), T), self.eos, dtype=torch.long); am = torch.zeros(len(P), T, dtype=torch.long)
        for b, p in enumerate(P):                      # left padding for generation
            ids[b, T - len(p):] = torch.tensor(p); am[b, T - len(p):] = 1
        out = self.llm.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=max_new,
                                do_sample=False, use_cache=True, pad_token_id=self.eos, eos_token_id=self.eos)
        return [self.tok.decode(o[T:], skip_special_tokens=True) for o in out.cpu()]


BP_SUFFIX = "\nAnswer:"


def parse_scratch(txt):
    """(predicted label or None, bridge, bridge_hex, answer) from a generated scratchpad."""
    yn = YESNO.findall(txt)
    lab = None if not yn else int(yn[-1] == "yes")
    m = re.match(r"\s*(\d+)\s+F([0-9A-F]{2})\s+(\d+)", txt)
    return lab, (int(m.group(1)) if m else None), (m.group(2) if m else None), (int(m.group(3)) if m else None)


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def fit_std(m, tr, dev, bs):
    m.eval(); X = []
    for k in range(0, len(tr), bs):
        X.append(m.states(tr[k:k + bs], dev).float().cpu())
    X = torch.cat(X)
    m.mu.copy_(X.mean(0).to(dev)); m.sd.copy_((X.std(0) + 1e-4).to(dev))
    cos = F.normalize(X[:, -X.size(1) // 3:], dim=1); n = X.size(0)
    return {"n": n, "last_state_mean_pairwise_cos": float(((cos @ cos.t()).sum() - n) / (n * (n - 1)))}


@torch.no_grad()
def evaluate(m, rs, dev, task, bs=16, keep=False):
    m.eval()
    if task == "cls":
        S_ = []
        for k in range(0, len(rs), bs):
            lg = m(rs[k:k + bs], dev)
            S_.append((lg[:, 1] - lg[:, 0]).float().cpu())
        s = torch.cat(S_)
        y = torch.tensor([r["label"] for r in rs])
        pred = (s > 0).long()
        out = {"acc": float((pred == y).float().mean()), "pos_rate": float(pred.float().mean())}
        if keep:
            out["scores"] = [round(float(x), 5) for x in s]
        return out
    gens = []
    for k in range(0, len(rs), bs):
        gens += m.generate(rs[k:k + bs], dev)
    ok = nb = nbh = na = unp = 0
    for g, r in zip(gens, rs):
        lab, b, bh, ans = parse_scratch(g)
        unp += lab is None
        ok += lab is not None and lab == r["label"]
        nb += b == r["chain"][1]; nbh += bh == "%02X" % r["chain"][1]; na += ans == r["chain"][2]
    n = len(rs)
    out = {"acc": ok / n, "unparsed": unp / n, "bridge_acc": nb / n, "bridge_hex_acc": nbh / n, "answer_sym_acc": na / n}
    if keep:
        out["gens"] = gens
    return out


# ------------------------------------------------------------------ training with checkpoints

def trainable_state(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()
            if "lora_" in k or k.startswith("head") or k in ("mu", "sd")}


def run(a, tok, D_, dev):
    torch.manual_seed(a.seed)
    m = (Cls(a.enc, a.rank) if a.task == "cls" else Scratch(a.enc, a.rank, tok)).to(dev)
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
    if os.path.exists(ck):
        C = torch.load(ck, weights_only=False)
        m.load_state_dict(C["model"], strict=False); opt.load_state_dict(C["opt"]); sched.load_state_dict(C["sched"])
        torch.set_rng_state(C["rng"]); torch.cuda.set_rng_state(C["cuda_rng"])
        rec, step, ep, perm = C["rec"], C["step"], C["ep"], C["perm"]
        print("  resumed at step %d / %d" % (step, total), flush=True)
    else:
        rec = {"seed": a.seed, "n_train": n, "epochs_planned": epochs, "total_steps": total, "eval_every": eval_every,
               "evals": [], "best_dev": -1.0, "best_step": None, "secs": 0.0, "segments": 0}
        if a.task == "cls":
            rec["std"] = fit_std(m, tr[:min(n, 2000)], dev, a.ebs)
        step, ep, perm = 0, 0, None
    rec["segments"] += 1
    g = torch.Generator(); g.manual_seed(a.seed * 1000 + ep)
    t0 = time.time(); seg_t0 = time.time()
    sub = tr[:NSUB]
    seg_secs = None
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
        rec.setdefault("loss_trace", []).append(round(lsum, 5)) if step % 50 == 0 else None
        if step % eval_every == 0 or step == total:
            e = {"step": step, "instances": step * a.bs, "epoch_float": round(step / steps_ep, 3),
                 "train_sub": evaluate(m, sub, dev, a.task, a.ebs), "dev": evaluate(m, D_["dev"], dev, a.task, a.ebs),
                 "secs": round(rec["secs"] + time.time() - t0, 1)}
            rec["evals"].append(e)
            print("  %s %s %s n%d r%d lr%.0e sd%d step %d/%d (ep %.2f) loss %.4f train_sub %.4f dev %.4f%s (%.0fs)" % (
                a.task, a.enc, a.variant, n, a.rank, a.lr, a.seed, step, total, e["epoch_float"], lsum,
                e["train_sub"]["acc"], e["dev"]["acc"],
                " bridge %.3f unparsed %.3f" % (e["dev"]["bridge_acc"], e["dev"]["unparsed"]) if a.task == "scratch"
                else "", e["secs"]), flush=True)
            if e["dev"]["acc"] > rec["best_dev"]:
                rec["best_dev"], rec["best_step"] = e["dev"]["acc"], step
                torch.save(trainable_state(m), a.out + ".best.pt")
            if not math.isfinite(lsum):
                rec["diverged"] = True; break
            if len(rec["evals"]) >= 2 and all(x["dev"]["acc"] >= 0.995 for x in rec["evals"][-2:]):
                rec["early_stop"] = step; break
            rec["secs"] += time.time() - t0; t0 = time.time()
            torch.save({"model": trainable_state(m), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "rec": rec, "step": step,
                        "ep": ep, "perm": perm}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            seg = time.time() - seg_t0; seg_t0 = time.time()
            seg_secs = seg if seg_secs is None else max(seg_secs, seg)
            if a.max_hours and (time.time() - START) + 1.15 * seg_secs > a.max_hours * 3600 and step < total:
                print("  time budget reached at step %d; exit 3 for resume" % step, flush=True)
                return rec, 3
    rec["secs"] += time.time() - t0
    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    return rec, 0


def score_test(a, tok, D_, dev, rec_path):
    m = (Cls(a.enc, a.rank) if a.task == "cls" else Scratch(a.enc, a.rank, tok)).to(dev)
    st = torch.load(a.out + ".best.pt", weights_only=False)
    missing = [k for k in st if k not in m.state_dict()]
    assert not missing, missing[:5]
    m.load_state_dict(st, strict=False)
    res = {"test": evaluate(m, D_["test"], dev, a.task, a.ebs, keep=True),
           "dev_recheck": evaluate(m, D_["dev"], dev, a.task, a.ebs)}
    rec = json.load(open(rec_path))
    res["dev_logged"] = rec["run"]["best_dev"]
    print("  TEST %s %s %s n%d r%d lr%.0e sd%d: test %.4f (dev recheck %.4f, logged %.4f)" % (
        a.task, a.enc, a.variant, a.n, a.rank, a.lr, a.seed, res["test"]["acc"], res["dev_recheck"]["acc"],
        res["dev_logged"]), flush=True)
    return res


START = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cls", "scratch"])
    ap.add_argument("--enc", default="qwen8", choices=["qwen8", "phi2"])
    ap.add_argument("--variant", required=True, choices=["std", "ho"])
    ap.add_argument("--n", type=int, required=True, choices=[800, 8000, 80000])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--ebs", type=int, default=16)
    ap.add_argument("--max_hours", type=float, default=0.0)
    ap.add_argument("--score_test", action="store_true")
    ap.add_argument("--final", action="store_true", help="a final run (configuration already selected): score the test "
                    "split once from the dev-best state right after training")
    ap.add_argument("--smoke", type=int, default=0, help="train on the first N instances, dev = first 64 train")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = BP.tokenizer(a.enc)
    if a.score_test:
        tp = a.out + ".test.json"
        if os.path.exists(tp):
            print("exists, skip", tp); return
        assert os.path.exists(a.out), "run not finished"
        D_ = {"test": load_rows(a.data, a.enc, a.variant, "test")[0], "dev": load_rows(a.data, a.enc, a.variant, "dev")[0]}
        res = score_test(a, tok, D_, a.device, a.out)
        res["args"] = vars(a); res["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(res, open(tp + ".tmp", "w")); os.replace(tp + ".tmp", tp)
        return
    if os.path.exists(a.out):
        if a.final and not os.path.exists(a.out + ".test.json"):
            D2 = {"test": load_rows(a.data, a.enc, a.variant, "test")[0], "dev": load_rows(a.data, a.enc, a.variant, "dev")[0]}
            res = score_test(a, tok, D2, a.device, a.out)
            res["args"] = vars(a); res["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            json.dump(res, open(a.out + ".test.json.tmp", "w")); os.replace(a.out + ".test.json.tmp", a.out + ".test.json")
            return
        print("exists, skip", a.out); return
    tr, md5 = load_rows(a.data, a.enc, a.variant, "train", a.n)
    D_ = {"train": tr}
    if a.smoke:
        D_["train"] = tr[:a.smoke]; D_["dev"] = tr[:64]
        EPOCHS[a.smoke] = 1
    else:
        D_["dev"] = load_rows(a.data, a.enc, a.variant, "dev")[0]
    out = {"args": vars(a), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(a.device),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "train_md5_full": md5,
           "n": {k: len(v) for k, v in D_.items()}, "max_len": {k: max(len(r["ids"]) for r in v) for k, v in D_.items()}}
    print("B44 LoRA", json.dumps({k: out[k] for k in ("n", "max_len", "gpu")}), flush=True)
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
        a.score_test = True
        D2 = {"test": load_rows(a.data, a.enc, a.variant, "test")[0], "dev": D_["dev"]}
        res = score_test(a, tok, D2, a.device, a.out)
        res["args"] = vars(a); res["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tp = a.out + ".test.json"
        json.dump(res, open(tp + ".tmp", "w")); os.replace(tp + ".tmp", tp)


if __name__ == "__main__":
    main()
