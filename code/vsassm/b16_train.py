"""B16 step 3: LoRA-distil the teacher's outputs into Qwen3-1.7B (loss on the output tokens only).
python3 b16_train.py --model <Qwen3-1.7B> --data b16_data/labels_*.jsonl --seed 0 --out b16_models/student_s0
"""
import argparse
import glob
import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)  # glob
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--holdout", type=int, default=500)  # teacher-labelled examples kept out (agreement check)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ex = [json.loads(l) for p in sorted(glob.glob(args.data)) for l in open(p)]
    rng.shuffle(ex)
    hold, ex = ex[:args.holdout], ex[args.holdout:]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))

    def encode(e):
        head = tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        h = tok(head, add_special_tokens=False).input_ids
        o = tok(e["output"].strip() + "<|im_end|>", add_special_tokens=False).input_ids
        return h + o, [-100] * len(h) + o

    data = [encode(e) for e in ex]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * (len(data) // args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    model.train()
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        rng.shuffle(data)
        for k in range(0, len(data) - args.bs + 1, args.bs):
            batch = data[k:k + args.bs]
            L = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device="cuda")
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device="cuda")
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0 or step == steps:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item():.4f} t={time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(args.out)
    json.dump(hold, open(f"{args.out}/holdout.json", "w"))
    print("saved", args.out, len(data), "training examples")


if __name__ == "__main__":
    main()
