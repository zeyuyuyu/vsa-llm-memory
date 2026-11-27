"""NSR1 LLM arms with vLLM (deviation from the apps/real runner, recorded in the NSR1 verdict): on the A800 box HF
generate needs about an hour per 8 problems for Qwen3-32B with ~10k-token prompts, because transformers falls back from
native GQA to materialised repeat_kv whenever a padding mask is present (batched runs). Everything the protocol fixes is
kept: the same chat template with thinking on, greedy decoding (temperature 0), 8,192 new tokens, and b6_qwen.py's
output record (n, idx, thinking, answer_text, hit_limit, ...), so nsr1_llm.py parse and force_answer.py read it unchanged.

python3 nsr/nsr1_vllm.py --model <snap> --name qwen3-32b --problems <file> --out <dir> --ns 0,1,... --tp 2
"""
import argparse
import glob
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", default="NSR1")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="")
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=24576)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ns = {int(x) for x in a.ns.split(",")} if a.ns else None
    path = f"{a.out}/{a.name}_{a.task}" + (f"_n{a.ns.replace(',', '-')}" if ns else "") + "_vllm.jsonl"
    done = set()
    for fp in glob.glob(f"{a.out}/{a.name}_{a.task}*.jsonl"):
        for r in map(json.loads, open(fp)):
            done.add((r["n"], r["idx"]))
    probs = [p for p in map(json.loads, open(a.problems))
             if p["task"] == a.task and (p["n"], p["idx"]) not in done and (ns is None or p["n"] in ns)]
    print(f"{len(probs)} problems to run ({len(done)} already done)", flush=True)
    if not probs:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    think_end = tok.convert_tokens_to_ids("</think>")
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for p in probs]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=0.90, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, skip_special_tokens=True)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    import torch
    gpu = torch.cuda.get_device_name(0)
    with open(path, "a") as f:
        for p, o, text in zip(probs, outs, texts):
            ids = list(o.outputs[0].token_ids)
            te = ids.index(think_end) + 1 if think_end in ids else None
            f.write(json.dumps({
                "model": a.name, "gpu": gpu, "engine": "vllm-0.11.0 greedy", "task": p["task"], "n": p["n"],
                "idx": p["idx"], "sample": False, "no_think": False, "prompt_tokens": len(o.prompt_token_ids),
                "gen_tokens": len(ids), "think_tokens": te if te is not None else len(ids), "closed_think": te is not None,
                "hit_limit": o.outputs[0].finish_reason == "length",
                "thinking": tok.decode(ids[:te] if te is not None else ids, skip_special_tokens=True),
                "answer_text": tok.decode(ids[te:], skip_special_tokens=True) if te is not None else "",
            }) + "\n")
    print(f"{path}: {len(probs)} problems in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
