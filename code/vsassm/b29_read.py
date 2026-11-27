"""B29 reader replica (prereg/B29_same_tooling_prereg.md): B21's generation path, with the model as a parameter.

python3 b29_read.py MODEL_DIR PROMPTS.jsonl OUT.jsonl [--bs 64]

A copy of b26_read.py (itself a standalone copy of ruler_rm.CachedReader -> b14.Reader with B17 off): bf16, sdpa
attention, chat template with thinking off, left padding, greedy decoding, at most 400 new tokens, prompts sorted by
length and batched up to 64 at a time under a budget of 64 x 2,400 characters; the output is decoded up to and including
the first end-of-turn token. Two additions, neither of which changes a prompt or a decoding setting:
  - a batch that runs out of GPU memory is split in half and retried (counted in the log and in OUT.meta.json);
  - the GPU name, torch and transformers versions, wall time and generated tokens go to OUT.meta.json.
Resumes: ids already in OUT are skipped.
"""
import json
import os
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MAX_NEW, BUDGET_PER = 400, 2400


def main(model_dir, prompts_path, out_path, bs=64):
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
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    model.generation_config = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos,
                                               pad_token_id=tok.pad_token_id)
    eos = set(eos)
    gpu = torch.cuda.get_device_name(0)
    print("gpu:", gpu, "torch", torch.__version__, "transformers", transformers.__version__, flush=True)
    budget = bs * BUDGET_PER
    order = sorted(range(len(todo)), key=lambda i: len(todo[i]["prompt"]))
    k, t0, ntok, splits = 0, time.time(), 0, 0

    def run(batch):
        nonlocal splits
        texts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for x in batch]
        try:
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                return model.generate(**enc)[:, enc.input_ids.shape[1]:].tolist()
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            splits += 1
            torch.cuda.empty_cache()
            print("OOM at batch %d: split" % len(batch), flush=True)
            h = len(batch) // 2
            return run(batch[:h]) + run(batch[h:])

    with open(out_path, "a") as f:
        nb = 0
        while k < len(order):
            n = 1
            while k + n < len(order) and n < bs and len(todo[order[k + n]]["prompt"]) * (n + 1) <= budget:
                n += 1
            batch = [todo[i] for i in order[k:k + n]]
            gen = run(batch)
            for x, ids in zip(batch, gen):
                m = next((i + 1 for i, t in enumerate(ids) if t in eos), len(ids))
                ntok += m
                f.write(json.dumps({"id": x["id"], "text": tok.decode(ids[:m], skip_special_tokens=True), "ntok": m}) + "\n")
            f.flush()
            k += n
            nb += 1
            if nb % 20 == 0:
                print("read %d / %d  %.0f s  %d tokens" % (k, len(order), time.time() - t0, ntok), flush=True)
    wall = time.time() - t0
    print("done %d prompts in %.0f s, %d generated tokens, %d OOM splits" % (len(order), wall, ntok, splits), flush=True)
    meta = {"model": model_dir, "prompts": prompts_path, "n": len(order), "wall_s": round(wall, 1), "gen_tokens": ntok,
            "oom_splits": splits, "gpu": gpu, "torch": torch.__version__, "transformers": transformers.__version__,
            "bs": bs, "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out_path + ".meta.json", "a") as f:
        f.write(json.dumps(meta) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    bs = 64
    if "--bs" in a:
        i = a.index("--bs")
        bs = int(a[i + 1])
        a = a[:i] + a[i + 2:]
    main(a[0], a[1], a[2], bs)
