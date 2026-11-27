"""Budget forcing for the LLM reference runs: every answer that hit the 8,192-token reasoning limit is given its own
reasoning back, closed with Qwen's documented phrase, and allowed 64 tokens to answer. The page reports the strict
score (a run that hit the limit gave no answer) and the forced one.

  python3 apps/real/force_answer.py --model <snapshot> --name qwen3-8b --problems apps/real/llm_problems_test2_*.jsonl \
      --runs 'apps/real/out/test2/qwen3-8b_REAL_n*.jsonl' --out apps/real/out/test2/forced_qwen3-8b.jsonl
"""
import argparse
import glob
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--runs", required=True)  # glob of b6_qwen.py outputs for this model
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    prompts = {}
    for p in args.problems:
        for r in map(json.loads, open(p)):
            prompts[(r["n"], r["idx"])] = r["prompt"]
    done = {(r["n"], r["idx"]) for r in map(json.loads, open(args.out))} if os.path.exists(args.out) else set()
    todo = [r for fp in sorted(glob.glob(args.runs)) for r in map(json.loads, open(fp))
            if r["hit_limit"] and (r["n"], r["idx"]) in prompts and (r["n"], r["idx"]) not in done]
    print(f"{len(todo)} runs to force", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    eos = model.generation_config.eos_token_id
    eos = list(eos if isinstance(eos, list) else [eos])
    with open(args.out, "a") as f:
        for s in range(0, len(todo), args.bs):
            batch = todo[s:s + args.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": prompts[(r["n"], r["idx"])]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True) + r["thinking"] + CLOSE
                     for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None, top_p=None,
                                     top_k=None, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for r, ids in zip(batch, out[:, enc.input_ids.shape[1]:].tolist()):
                f.write(json.dumps({"model": args.name, "n": r["n"], "idx": r["idx"],
                                    "forced_answer": tok.decode(ids, skip_special_tokens=True)}) + "\n")
            f.flush()
            print(f"forced {s + len(batch)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
