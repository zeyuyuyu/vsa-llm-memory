"""B29 full-read reader for the store arm (prereg/B29_same_tooling_prereg.md): the same prompts, chat template (thinking
off) and greedy decoding with at most 400 new tokens as b29_read.py, decoded with vLLM instead of HF generate, because
the needle-dense RULER tasks need about 300,000 chunk reads at 128k (HF generate: about 35-40 A800 GPU-hours for a
32B reader). The dict and the store arms of the full-read path answer from the SAME outputs, so the store-vs-dict
comparison has no engine difference inside it; the engine difference to the HF path is measured, not assumed
(prereg: cross-engine agreement of the dict answers).

python3 b29_vllm.py MODEL_DIR PROMPTS.jsonl OUT.jsonl [--tp 2] [--chunk 20000]
Resumes: ids already in OUT are skipped. Output records as b29_read.py: {"id", "text", "ntok"}.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm
    todo = [json.loads(l) for l in open(a.prompts)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [x for x in todo if x["id"] not in done]
    print(f"prompts to read: {len(todo)}, already done: {len(done)}", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0,
              disable_custom_all_reduce=True)   # PCIe A800s: the custom all-reduce kernel fails (dev pilot 1)
    sp = SamplingParams(temperature=0.0, max_tokens=400, skip_special_tokens=True)
    t0, ntok = time.time(), 0
    for k in range(0, len(todo), a.chunk):
        part = todo[k:k + a.chunk]
        texts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for x in part]
        outs = llm.generate(texts, sp)
        with open(a.out, "a") as f:
            for x, o in zip(part, outs):
                c = o.outputs[0]
                n = len(c.token_ids)
                ntok += n
                f.write(json.dumps({"id": x["id"], "text": c.text, "ntok": n}) + "\n")
        print(f"read {k + len(part)} / {len(todo)}  {time.time() - t0:.0f} s  {ntok} tokens", flush=True)
    meta = {"model": a.model, "prompts": a.prompts, "n": len(todo), "wall_s": round(time.time() - t0, 1),
            "gen_tokens": ntok, "engine": "vllm " + vllm.__version__, "tp": a.tp,
            "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out + ".meta.json", "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
