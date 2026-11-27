"""V-2 (prereg bdadbb9): regenerate the VSA-LLM headline contrast under a protocol that cannot produce
the two defects the main-table audit verified.

D1 removed: the existing runners report max-over-epochs accuracy on the single set they also report
   (run_compositional.py lines 229-230; train_v2/train_v3 likewise, which is where "Best Accuracy for
   hybrid: 0.8940" comes from against a final epoch of 0.868). Here the epoch is chosen on a VALIDATION
   split and TEST is evaluated exactly once, at that epoch.
D2 removed: CompositionalDataset seeds with `42 if split == 'train' else 123`, so any non-train split
   name yields identical data and a third split cannot be obtained by naming it. Here a single
   1500-sample evaluation pool is generated once and split deterministically: [:500] validation,
   [500:] test (1000 samples).

The two arms differ only by HybridModel(use_vsa=True/False), which is already matched; that part of the
original design is sound and is kept unchanged.
Scorer is in this file, written before any run.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_compositional import CompositionalDataset, set_seed
from model import HybridModel

TASKS = ["variable_tracking", "rule_priority", "state_update", "binding_test"]


def evaluate(model, loader, device="cuda"):
    model.eval()
    correct = tot = 0
    per = {t: [0, 0] for t in TASKS}
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=am)
            logits = out["logits"] if isinstance(out, dict) else out
            pred = logits.argmax(-1)
            ok = (pred == y)
            correct += ok.sum().item(); tot += y.numel()
            for i, t in enumerate(batch.get("task", [None] * y.numel())):
                if t in per:
                    per[t][0] += int(ok[i].item()); per[t][1] += 1
    acc = correct / max(tot, 1)
    pt = {t: (c / n if n else float("nan")) for t, (c, n) in per.items()}
    return acc, pt


def run_arm(use_vsa, seed, epochs, device, bs=8, lr=2e-4):
    set_seed(seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = CompositionalDataset(tok, num_samples=2000, split="train")
    pool = CompositionalDataset(tok, num_samples=1500, split="eval")
    val_ds = Subset(pool, list(range(0, 500)))          # epoch selection only
    test_ds = Subset(pool, list(range(500, len(pool))))  # evaluated once, at the val-selected epoch

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = HybridModel(model_name="microsoft/phi-2", vsa_dim=1024, num_role_vectors=16,
                        max_bindings=3, use_vsa=use_vsa, freeze_llm=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_val, best_state, hist = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for batch in train_ld:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=am)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); tl += loss.item()
        va, _ = evaluate(model, val_ld, device)
        hist.append({"epoch": ep + 1, "loss": tl / max(len(train_ld), 1), "val_acc": va})
        print("    ep%d loss=%.4f val=%.4f" % (ep + 1, tl / max(len(train_ld), 1), va), flush=True)
        if va > best_val:
            best_val = va
            best_epoch = ep + 1
            # only the trainable parameters: the LLM is frozen, so cloning its 2.7B weights every
            # improving epoch would be pointless and would not fit
            best_state = {n: p.detach().clone() for n, p in model.named_parameters()
                          if p.requires_grad}
    if best_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n])
    test_acc, test_per = evaluate(model, test_ld, device)
    del model; torch.cuda.empty_cache()
    return {"arm": "hybrid" if use_vsa else "llm_only", "seed": seed, "trainable": trainable,
            "val_best": best_val, "val_best_epoch": best_epoch, "test_acc": test_acc,
            "test_per_task": test_per, "history": hist, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="v2_logs/v2.json")
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        for use_vsa in [False, True]:
            print("  === %s seed %d ===" % ("hybrid" if use_vsa else "llm_only", seed), flush=True)
            r = run_arm(use_vsa, seed, a.epochs, a.device)
            print("  %-9s sd%-4d val_best=%.4f (ep%d) TEST=%.4f trainable=%s %ss"
                  % (r["arm"], seed, r["val_best"], r["val_best_epoch"], r["test_acc"],
                     format(r["trainable"], ","), r["secs"]), flush=True)
            print("      per-task: " + "  ".join("%s=%.4f" % (t, r["test_per_task"][t]) for t in TASKS),
                  flush=True)
            rows.append(r)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    # scorer, exactly as pre-registered at bdadbb9
    def col(arm):
        return [r["test_acc"] for r in rows if r["arm"] == arm]
    h, l = col("hybrid"), col("llm_only")
    if len(h) == len(l) == len(a.seeds):
        gaps = [(x - y) * 100 for x, y in zip(h, l)]
        print("\n  TEST gap hybrid - llm_only, per seed (pp): %s" % [round(g, 2) for g in gaps],
              flush=True)
        v = "SURVIVES" if all(g >= 5.0 for g in gaps) else (
            "COLLAPSES" if all(g <= 2.0 for g in gaps) else "INTERMEDIATE")
        print("  registered bar: SURVIVES >= +5pp in 3/3 ; COLLAPSES <= +2pp in 3/3", flush=True)
        print("  >>> V-2 VERDICT: %s   (paper claims +29.2pp)" % v, flush=True)


if __name__ == "__main__":
    main()
