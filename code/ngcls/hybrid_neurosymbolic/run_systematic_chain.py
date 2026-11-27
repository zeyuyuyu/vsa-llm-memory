#!/usr/bin/env python3
import json
import time
import warnings
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.stats import ttest_rel

from systematic_chain_dataset import SystematicChainDataset, ChainConfig
from explicit_chain_models import SymbolicVSAChainClassifier, FrozenLLMClassifier, LoRAClassifier, LoRAQPoolClassifier

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456]


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def train_one(model, train_loader, eval_loader, epochs: int, lr: float, device: str):
    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train ep {ep+1}/{epochs}"):
            opt.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)
        log(f"ep {ep+1}: loss={avg_loss:.4f} eval_acc={acc:.4f}")
        if acc > best:
            best = acc

    return best


def run(seed: int, cfg: ChainConfig, train_samples: int, eval_samples: int, batch_size: int, epochs: int, lr_vsa: float, lr_other: float, vsa_dim: int, lora_rank: int, qpool_window: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"seed={seed} device={device}")

    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = SystematicChainDataset(tok, num_samples=train_samples, split="train", cfg=cfg)
    eval_ds = SystematicChainDataset(tok, num_samples=eval_samples, split="eval", cfg=cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    results = {}

    log("model=FrozenLLM")
    m0 = FrozenLLMClassifier()
    acc0 = train_one(m0, train_loader, eval_loader, epochs=epochs, lr=lr_other, device=device)
    results["frozen_llm"] = acc0
    del m0
    torch.cuda.empty_cache()

    log("model=LoRA")
    m1 = LoRAClassifier(lora_rank=lora_rank)
    acc1 = train_one(m1, train_loader, eval_loader, epochs=epochs, lr=lr_other, device=device)
    results[f"lora_r{lora_rank}"] = acc1
    del m1
    torch.cuda.empty_cache()

    log("model=LoRA_QPool")
    m1b = LoRAQPoolClassifier(lora_rank=lora_rank, window=qpool_window)
    acc1b = train_one(m1b, train_loader, eval_loader, epochs=epochs, lr=lr_other, device=device)
    results[f"lora_qpool_r{lora_rank}"] = acc1b
    del m1b
    torch.cuda.empty_cache()

    log("model=VSAChain")
    m2 = SymbolicVSAChainClassifier(vsa_dim=vsa_dim, num_roles=cfg.num_roles, vocab_size=cfg.vocab_size)
    acc2 = train_one(m2, train_loader, eval_loader, epochs=epochs, lr=lr_vsa, device=device)
    results[f"vsa_chain_symbolic_dim{vsa_dim}"] = acc2
    del m2
    torch.cuda.empty_cache()

    return results


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_samples", type=int, default=12000)
    ap.add_argument("--eval_samples", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr_vsa", type=float, default=3e-4)
    ap.add_argument("--lr_other", type=float, default=1e-4)
    ap.add_argument("--vsa_dim", type=int, default=2048)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--num_pairs", type=int, default=ChainConfig().num_pairs)
    ap.add_argument("--max_length", type=int, default=ChainConfig().max_length)
    ap.add_argument("--p_positive", type=float, default=ChainConfig().p_positive)
    ap.add_argument("--qpool_window", type=int, default=8)
    ap.add_argument("--out", type=str, default="systematic_chain_results.json")
    args = ap.parse_args()
    cfg = ChainConfig(num_pairs=args.num_pairs, max_length=args.max_length, p_positive=args.p_positive)
    log("=" * 60)
    log("Systematic Chain (3-hop) Experiment")
    log("=" * 60)
    log(f"cfg={cfg}")
    log(f"args={vars(args)}")

    all_results = {
        "cfg": asdict(cfg),
        "args": vars(args),
        "seeds": SEEDS,
        "per_seed": {},
        "summary": {},
    }

    # allow overriding seeds from CLI
    seeds = SEEDS
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except Exception:
        seeds = SEEDS
    all_results["seeds"] = seeds

    start = time.time()
    for seed in seeds:
        res = run(
            seed,
            cfg,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_vsa=args.lr_vsa,
            lr_other=args.lr_other,
            vsa_dim=args.vsa_dim,
            lora_rank=args.lora_rank,
            qpool_window=args.qpool_window,
        )
        all_results["per_seed"][str(seed)] = res
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"saved interim -> {args.out}")

    keys = sorted(list(next(iter(all_results["per_seed"].values())).keys()))
    for k in keys:
        vals = [all_results["per_seed"][str(s)][k] for s in seeds]
        all_results["summary"][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    vsa_key = f"vsa_chain_symbolic_dim{args.vsa_dim}"
    lora_key = f"lora_r{args.lora_rank}"
    if vsa_key in keys and lora_key in keys:
        v = np.array([all_results["per_seed"][str(s)][vsa_key] for s in seeds])
        l = np.array([all_results["per_seed"][str(s)][lora_key] for s in seeds])
        t, p = ttest_rel(v, l)
        all_results["summary"]["paired_ttest_vsa_vs_lora"] = {"t": float(t), "p": float(p)}
        all_results["summary"]["gap_vsa_minus_lora_pp"] = float((v.mean() - l.mean()) * 100.0)

    all_results["elapsed_sec"] = time.time() - start

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    log("=" * 60)
    for k, st in all_results["summary"].items():
        if isinstance(st, dict) and "mean" in st:
            log("{}: {:.2f}% ± {:.2f}%".format(k, st["mean"] * 100.0, st["std"] * 100.0))
    if "gap_vsa_minus_lora_pp" in all_results["summary"]:
        log("VSA - LoRA gap: {:.2f} pp".format(all_results["summary"]["gap_vsa_minus_lora_pp"]))
        tt = all_results["summary"].get("paired_ttest_vsa_vs_lora")
        if tt:
            log("paired t-test: t={:.3f} p={:.3g}".format(tt["t"], tt["p"]))
    log("saved -> {}".format(args.out))
    log("=" * 60)


if __name__ == "__main__":
    main()
