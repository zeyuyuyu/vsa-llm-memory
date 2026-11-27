"""B33 fine-tune arms on bAbI task 2 (prereg/B33_fresh_task2_prereg.md).

Exactly B28's rerun of the published LoRA fine-tune (b28_lora2.py = V-50's lora_direct arm: LoRA rank r, alpha 2r,
dropout 0, on q/k/v/dense of Phi-2 in bf16; AdamW; batch 2; clip 1.0; cross-entropy at the last token over the six
location tokens; torch.manual_seed(seed) before the model is built; seeded batch order), trained on the same rows
(--train expr: the 775 stories expressible under the published parse, as published; --train all: all 900, secondary),
and scored, with per-question logits saved, on the B33 evaluation sets (raw rows in dataset order, max length 768,
as in B28's pass B). Configurations: best of twelve (rank 16, 6 epochs, lr 1e-4) and the first configuration (rank 8,
2 epochs, lr 1e-4). Nothing is tuned on any evaluation set.
--smoke: trains on 64 training rows for one epoch and scores 32 training rows (no evaluation set is built).
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b28_common as C                                                # noqa: E402
import b33_common as B                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--train", default="expr", choices=["expr", "all"])
    ap.add_argument("--sets", nargs="+", default=B.SETS)
    ap.add_argument("--fresh_file", default="b33_data/b33_fresh.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    M = C.load_ngcls()
    v50, v5 = M["v50"], M["v5"]
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = C.tokenizer()
    t0 = time.time()

    tr, _, vocab, n_p = v50.build_rows(2, tok, 2000, 500)          # verbatim: expressible train rows (775)
    d, vocab2, sets = B.load_sets(v5, a.fresh_file, [] if a.smoke else a.sets)
    assert vocab2 == vocab
    ans_ids, kept = [], []
    for i, w in enumerate(vocab):
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ans_ids.append(t[0]); kept.append(i)
    remap = {g: j for j, g in enumerate(kept)}
    assert len(kept) == len(vocab), "a location is not a single token"
    tr = [r for r in tr if r["label"] in remap]

    def raw_rows(split):
        rows = []
        for text, y in split:
            enc = tok(text, truncation=True, max_length=768, padding="max_length", return_tensors="pt")
            rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                         "label": int(y), "truncated": int(len(tok(text)["input_ids"]) > 768)})
        return rows

    if a.train == "all":
        tr = raw_rows(d["train"])
    if a.smoke:
        tr = tr[:64]
        evals = {"smoke_train32": tr[:32]}
    else:
        evals = {name: raw_rows(split) for name, split in sets.items()}
    print("B33 LoRA task 2: rank %d, epochs %d, lr %g, train=%s (%d rows); sets %s"
          % (a.rank, a.epochs, a.lr, a.train, len(tr), {k: len(v) for k, v in evals.items()}), flush=True)

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
        return logits, [int(np.argmax(x)) for x in logits]

    tag = "r%d_e%d_lr%g_%s%s%s" % (a.rank, a.epochs, a.lr, a.train,
                                   "" if a.seeds == C.SEEDS else "_s" + "-".join(map(str, a.seeds)),
                                   "_smoke" if a.smoke else "")
    fn = os.path.join(a.out, "b33_lora_%s.json" % tag)
    os.makedirs(a.out, exist_ok=True)
    res = {"rank": a.rank, "epochs": a.epochs, "lr": a.lr, "train": a.train, "n_train": len(tr), "kept": kept,
           "vocab": vocab, "sets": {k: {"labels_all": [r["label"] for r in v],
                                        "truncated": [r.get("truncated", 0) for r in v]} for k, v in evals.items()},
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
        run = {"seed": seed, "losses": losses, "sets": {}}
        for name, rows in evals.items():
            lg, pr = score(model, rows)
            acc = float(np.mean([int(p == remap[r["label"]]) for p, r in zip(pr, rows)]))
            run["sets"][name] = {"acc": acc, "logits": lg, "pred": pr}
            print("  seed %d: %-8s (%d) %.4f" % (seed, name, len(rows), acc), flush=True)
        res["runs"].append(run)
        json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
        del model, opt
        torch.cuda.empty_cache()
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print("B33-LORA-DONE %s (%.0fs)" % (tag, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
