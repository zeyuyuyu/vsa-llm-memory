# NSR1 copy of b6_qwen.py (repo HEAD), changed in ONE place only: device_map="auto" instead of "cuda", so Qwen3-32B is
# spread over the visible GPUs. b6_qwen.py puts the whole model on one GPU, which fits an H200 (141 GB) but leaves an
# A800 (80 GB) under 15 GB for the KV cache, and the NSR1 runs went out of memory (2026-11-07). Prompts, chat template,
# thinking mode, greedy decoding, 8,192 new tokens, output format and resume logic are unchanged.
"""B6: Qwen3 thinking-mode baseline (prereg/B6_prereg.md + amendment 1).

One job = (model, task): 5 N x 40 problems, enable_thinking=True, greedy (or the conditional sampling arm), 8,192 new
tokens. Dumps one JSON line per problem with the raw generated text; scoring is done by score_b6.py. Resumes.
"""
import argparse
import glob
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # local snapshot path
    ap.add_argument("--name", required=True)  # e.g. qwen3-8b
    ap.add_argument("--task", required=True)
    ap.add_argument("--problems", default="b6_problems.jsonl")
    ap.add_argument("--out", default="b6_out")
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--ns", default="")  # optional shard: comma-separated N values; output file gets an _n<..> tag
    ap.add_argument("--no_think", action="store_true")  # B7 reference: enable_thinking=False
    ap.add_argument("--sample", action="store_true")  # conditional robustness arm: T 0.6, top-p 0.95, top-k 20
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ns = {int(x) for x in args.ns.split(",")} if args.ns else None
    tag = ("_sample" if args.sample else "") + ("_nothink" if args.no_think else "") + \
        (f"_n{args.ns.replace(',', '-')}" if ns else "")
    path = f"{args.out}/{args.name}_{args.task}{tag}.jsonl"
    # done = problems answered in ANY output file of this model, task and decoding mode, so N-shards and the
    # original whole-task run never redo each other's problems
    done = set()
    for fp in glob.glob(f"{args.out}/{args.name}_{args.task}*.jsonl"):
        for r in map(json.loads, open(fp)):
            if r["sample"] == args.sample and r.get("no_think", False) == args.no_think:
                done.add((r["n"], r["idx"]))
    probs = [p for p in map(json.loads, open(args.problems))
             if p["task"] == args.task and (p["n"], p["idx"]) not in done and (ns is None or p["n"] in ns)]
    if not probs:
        print("nothing to do", path)
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa").eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    think_end = tok.convert_tokens_to_ids("</think>")
    if args.sample:
        gc = GenerationConfig(max_new_tokens=args.max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                              eos_token_id=list(eos), pad_token_id=tok.pad_token_id)
    else:
        gc = GenerationConfig(max_new_tokens=args.max_new, do_sample=False, eos_token_id=list(eos),
                              pad_token_id=tok.pad_token_id)

    # transformers >= 4.50 silently replaces a passed config's default-valued fields (do_sample=False) with the
    # model's own defaults (Qwen3: do_sample=True, T=0.6), so the config is installed on the model itself.
    model.generation_config = gc
    print("decoding:", {k: getattr(gc, k) for k in ("do_sample", "temperature", "top_p", "top_k")}, flush=True)

    with open(path, "a") as f:
        for s in range(0, len(probs), args.bs):
            batch = probs[s:s + args.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=not args.no_think) for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            torch.manual_seed(0)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc)
            dt = time.time() - t0
            new = out[:, enc.input_ids.shape[1]:].tolist()
            for p, ids, m in zip(batch, new, enc.attention_mask.sum(1).tolist()):
                n_gen = next((i + 1 for i, t in enumerate(ids) if t in eos), len(ids))
                ids = ids[:n_gen]
                stopped = ids[-1] in eos
                te = ids.index(think_end) + 1 if think_end in ids else (0 if args.no_think else None)
                f.write(json.dumps({
                    "model": args.name, "gpu": torch.cuda.get_device_name(0), "task": p["task"], "n": p["n"], "idx": p["idx"], "sample": args.sample, "no_think": args.no_think,
                    "prompt_tokens": m, "gen_tokens": n_gen, "think_tokens": te if te is not None else n_gen,
                    "closed_think": te is not None, "hit_limit": not stopped and n_gen >= args.max_new,
                    "thinking": tok.decode(ids[:te] if te is not None else ids, skip_special_tokens=True),
                    "answer_text": tok.decode(ids[te:], skip_special_tokens=True) if te is not None else "",
                }) + "\n")
            f.flush()
            print(f"{args.name} {args.task} n={batch[0]['n']} batch {s // args.bs}: {dt:.0f}s, "
                  f"max gen {max(len(x) for x in new)}", flush=True)


if __name__ == "__main__":
    main()
