"""B28 bAbI task 2: the published LoRA fine-tune configurations rerun and scored on ALL 500 test questions
(prereg/B28_noncommutative_binding_prereg.md).

Training is the lora_direct arm of the verbatim v50.py (LoRA rank r, alpha 2r, dropout 0, on q/k/v/dense of Phi-2
in bf16; AdamW; batch 2; clip 1.0; cross-entropy at the last token over the six location tokens), with the same
instance set as published: v50.build_rows(2, ...) -> the 775 expressible training stories (--train expr) -- or,
secondary, all 900 training stories (--train all). torch.manual_seed(seed) before the model is built and the
seeded batch order are as in v50.

Two scoring passes per seed, with per-question logits saved:
  A  the 420 expressible test questions, in v50's order and batching (reproducibility gate against the banked runs)
  B  all 500 test questions, in dataset order (the registered comparison)
--smoke: trains on the first 64 training rows for one epoch and scores 32 TRAINING rows (no test row is built).
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b28_common as C                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--train", default="expr", choices=["expr", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    M = C.load_ngcls()
    v50, v5, v37b = M["v50"], M["v5"], M["v37b"]
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = C.tokenizer()
    t0 = time.time()

    tr, te, vocab, n_p = v50.build_rows(2, tok, 2000, 500)          # verbatim: expressible train / test rows
    d = v5.load_babi(2, 2000, 200, 500)
    ans_ids, kept = [], []
    for i, w in enumerate(vocab):
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ans_ids.append(t[0]); kept.append(i)
    remap = {g: j for j, g in enumerate(kept)}
    assert len(kept) == len(vocab), "a location is not a single token"
    tr = [r for r in tr if r["label"] in remap]
    te = [r for r in te if r["label"] in remap]

    def raw_rows(split):
        rows = []
        for text, y in split:
            enc = tok(text, truncation=True, max_length=768, padding="max_length", return_tensors="pt")
            rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                         "label": int(y)})
        return rows

    if a.train == "all":
        tr = raw_rows(d["train"])
    if a.smoke:
        tr = tr[:64]
        te, all500 = tr[:32], tr[:32]
        teidx = list(range(len(te)))
    else:
        all500 = raw_rows(d["test"])
        # map the 420 expressible rows onto the 500 by token ids (order preserving)
        teidx, j = [], 0
        for i, r in enumerate(all500):
            if j < len(te) and torch.equal(r["ids"], te[j]["ids"]):
                teidx.append(i); j += 1
        assert j == len(te), "expressible rows not found among the 500 (%d of %d)" % (j, len(te))
    print("B28 LoRA task 2: rank %d, epochs %d, lr %g, train=%s (%d rows); pass A %d rows, pass B %d rows"
          % (a.rank, a.epochs, a.lr, a.train, len(tr), len(te), len(all500)), flush=True)

    def fresh():
        m = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(a.device)
        cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "dense"], task_type="CAUSAL_LM")
        m = get_peft_model(m, cfg)
        m.train()
        return m

    def batch(rows):
        return (torch.stack([r["ids"] for r in rows]).to(a.device),
                torch.stack([r["am"] for r in rows]).to(a.device))

    def score(model, rows):
        logits = []
        with torch.no_grad():
            for k in range(0, len(rows), a.bs):
                ids, am = batch(rows[k:k + a.bs])
                lg = model(input_ids=ids, attention_mask=am).logits
                last = am.sum(-1) - 1
                sel = lg[torch.arange(lg.size(0), device=a.device), last].float()[:, ans_ids]
                logits += sel.cpu().tolist()
        pred = [int(np.argmax(x)) for x in logits]
        return logits, pred

    tag = "r%d_e%d_lr%g_%s%s" % (a.rank, a.epochs, a.lr, a.train, "_smoke" if a.smoke else "")
    fn = os.path.join(a.out, "b28_lora2_%s.json" % tag)
    os.makedirs(a.out, exist_ok=True)
    res = {"rank": a.rank, "epochs": a.epochs, "lr": a.lr, "train": a.train, "n_train": len(tr),
           "kept": kept, "vocab": vocab, "passA_index": teidx, "labels_all": [r["label"] for r in all500],
           "runs": []}
    for seed in a.seeds:
        torch.manual_seed(seed)
        model = fresh()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr)
        g = torch.Generator().manual_seed(seed)
        losses = []
        for ep in range(1 if a.smoke else a.epochs):
            perm = torch.randperm(len(tr), generator=g).tolist()
            tot, nb = 0.0, 0
            for k in range(0, len(perm), a.bs):
                rows = [tr[i] for i in perm[k:k + a.bs]]
                ids, am = batch(rows)
                lg = model(input_ids=ids, attention_mask=am).logits
                last = am.sum(-1) - 1
                sel = lg[torch.arange(lg.size(0), device=a.device), last].float()
                y = torch.tensor([remap[r["label"]] for r in rows], device=a.device)
                loss = F.cross_entropy(sel[:, ans_ids], y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                tot += float(loss.item()); nb += 1
            losses.append(tot / max(1, nb))
            print("    seed %d ep%d loss=%.4f (%.0fs)" % (seed, ep + 1, losses[-1], time.time() - t0), flush=True)
        model.eval()
        lgA, pA = score(model, te)
        lgB, pB = score(model, all500)
        accA = float(np.mean([int(p == remap[r["label"]]) for p, r in zip(pA, te)]))
        accB = float(np.mean([int(p == remap[r["label"]]) for p, r in zip(pB, all500)]))
        print("  seed %d: pass A (expressible %d) %.4f | pass B (all %d) %.4f" % (seed, len(te), accA,
                                                                              len(all500), accB), flush=True)
        res["runs"].append({"seed": seed, "losses": losses, "accA": accA, "accB": accB,
                            "logitsA": lgA, "predA": pA, "logitsB": lgB, "predB": pB})
        json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
        del model, opt
        torch.cuda.empty_cache()
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print("B28-LORA2-DONE %s (%.0fs)" % (tag, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
