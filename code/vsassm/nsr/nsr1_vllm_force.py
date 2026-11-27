"""NSR1 budget forcing with vLLM, same rule as apps/real/force_answer.py (707351b): every run that hit the 8,192-token
limit gets its own reasoning back, closed with the identical CLOSE phrase, and 64 greedy tokens to answer. Same output
record ({"model", "n", "idx", "forced_answer"}). vLLM keeps the NSR1 LLM arm on one decoding engine (see nsr1_vllm.py).

python3 nsr/nsr1_vllm_force.py --model <snap> --name qwen3-32b --problems <file> --runs '<dir>/qwen3-32b_NSR1_n*.jsonl'
    --out <dir>/forced_qwen3-32b.jsonl --tp 2
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_707351b"))
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=64)  # NSR1 amendment 2: 512 (set answers need more than 64)
    a = ap.parse_args()
    prompts = {(r["n"], r["idx"]): r["prompt"] for r in map(json.loads, open(a.problems))}
    done = {(r["n"], r["idx"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    todo = [r for fp in sorted(glob.glob(a.runs)) for r in map(json.loads, open(fp))
            if r["hit_limit"] and (r["n"], r["idx"]) in prompts and (r["n"], r["idx"]) not in done]
    print(f"{len(todo)} runs to force", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompts[(r["n"], r["idx"])]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) + r["thinking"] + CLOSE
             for r in todo]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=24576,
              gpu_memory_utilization=0.90, seed=0)
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens, skip_special_tokens=True))
    with open(a.out, "a") as f:
        for r, o in zip(todo, outs):
            f.write(json.dumps({"model": a.name, "n": r["n"], "idx": r["idx"], "forced_answer": o.outputs[0].text}) + "\n")
    print(f"forced {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
