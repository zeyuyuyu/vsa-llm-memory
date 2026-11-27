"""B26a write side: re-read the varied needle chunks with B21's reader (prereg/B26_vector_form_prereg.md).

python3 b26_read.py MODEL_DIR PROMPTS.jsonl OUT.jsonl

A standalone copy of the generation path that B21 used (ruler_rm.CachedReader -> b14.Reader, B17 off): Qwen3-8B, bf16,
sdpa attention, chat template with thinking off, left padding, greedy decoding, at most 400 new tokens, prompts sorted
by length and batched up to 64 at a time under a budget of 64 x 2,400 characters; the output is decoded up to and
including the first end-of-turn token. Resumes: ids already in OUT are skipped.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MAX_NEW, BS, BUDGET = 400, 64, 64 * 2400


def main(model_dir, prompts_path, out_path):
    todo = [json.loads(l) for l in open(prompts_path)]
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    todo = [x for x in todo if x["id"] not in done]
    print("prompts to read:", len(todo), "already done:", len(done), flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation="sdpa").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation="sdpa").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    model.generation_config = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos,
                                               pad_token_id=tok.pad_token_id)
    eos = set(eos)
    print("gpu:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    order = sorted(range(len(todo)), key=lambda i: len(todo[i]["prompt"]))
    k, t0, ntok = 0, time.time(), 0
    with open(out_path, "a") as f:
        while k < len(order):
            n = 1
            while k + n < len(order) and n < BS and len(todo[order[k + n]]["prompt"]) * (n + 1) <= BUDGET:
                n += 1
            batch = [todo[i] for i in order[k:k + n]]
            texts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for x in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc)[:, enc.input_ids.shape[1]:].tolist()
            for x, ids in zip(batch, gen):
                m = next((i + 1 for i, t in enumerate(ids) if t in eos), len(ids))
                ntok += m
                f.write(json.dumps({"id": x["id"], "text": tok.decode(ids[:m], skip_special_tokens=True), "ntok": m}) + "\n")
            f.flush()
            k += n
            if (k // n) % 20 == 0:
                print("read %d / %d  %.0f s  %d tokens" % (k, len(order), time.time() - t0, ntok), flush=True)
    print("done %d prompts in %.0f s, %d generated tokens" % (len(order), time.time() - t0, ntok), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
