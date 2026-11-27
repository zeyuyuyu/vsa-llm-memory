"""V-1 (prereg a94c26e): does VSA-over-LLM-hidden-states still beat LoRA on SysChain once the
hand-written marker parser is removed?

Five arms, identical data, identical seeds, MAIN-TABLE budget (2000/1000, 2 epochs) — not the
256-sample appendix budget that produced the uninformative 47.8 rows.
  frozen_llm   linear probe on frozen hidden states                (floor)
  lora         LoRAClassifier r=16                                  (paper arm)
  lora_qpool   LoRAQPoolClassifier r=16, window 8                   (paper arm, stronger)
  vsa_hidden   VSAChainClassifier — HRR over LEARNED projections of LLM hidden states  (THE TEST)
  vsa_parser   SymbolicVSAChainClassifier — the paper's arm, kept as upper reference
"""
import argparse, json, os, sys, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systematic_chain_dataset import SystematicChainDataset, ChainConfig
from explicit_chain_models import (FrozenLLMClassifier, LoRAClassifier, LoRAQPoolClassifier,
                                   VSAChainClassifier, SymbolicVSAChainClassifier)
from run_systematic_chain import train_one


def build(arm, cfg, vsa_dim, lora_rank, qpool_window, seed):
    if arm == "frozen_llm":
        return FrozenLLMClassifier()
    if arm == "lora":
        return LoRAClassifier(lora_rank=lora_rank)
    if arm == "lora_qpool":
        return LoRAQPoolClassifier(lora_rank=lora_rank, window=qpool_window)
    if arm == "vsa_hidden":
        return VSAChainClassifier(vsa_dim=vsa_dim, num_roles=cfg.num_roles)
    if arm == "vsa_parser":
        return SymbolicVSAChainClassifier(vsa_dim=vsa_dim, num_roles=cfg.num_roles,
                                          vocab_size=cfg.vocab_size, seed=seed)
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["frozen_llm", "lora", "lora_qpool", "vsa_hidden", "vsa_parser"])
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--eval_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--vsa_dim", type=int, default=2048)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--qpool_window", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = ChainConfig(num_roles=64, vocab_size=256, num_pairs=16, max_length=512, p_positive=0.5)
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        torch.manual_seed(s)
        tr = SystematicChainDataset(tok, num_samples=a.train_samples, split="train", cfg=cfg)
        ev = SystematicChainDataset(tok, num_samples=a.eval_samples, split="eval", cfg=cfg)
        trl = DataLoader(tr, batch_size=a.batch_size, shuffle=True)
        evl = DataLoader(ev, batch_size=a.batch_size, shuffle=False)
        m = build(a.arm, cfg, a.vsa_dim, a.lora_rank, a.qpool_window, s)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        lr = 3e-4 if a.arm.startswith("vsa") else 1e-4
        t0 = time.time()
        acc = train_one(m, trl, evl, epochs=a.epochs, lr=lr, device=a.device)
        r = {"arm": a.arm, "seed": s, "acc": float(acc), "trainable_params": int(n_train),
             "lr": lr, "secs": round(time.time() - t0, 1)}
        print(f"  {a.arm:<11} sd{s} acc={r['acc']:.4f} trainable={n_train:,} {r['secs']}s", flush=True)
        rows.append(r)
        del m
        torch.cuda.empty_cache()
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
