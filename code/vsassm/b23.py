"""B23 (prereg/B23_babilong_finetune_baseline_prereg.md): the referee baseline. Qwen3-1.7B, the model of the B18
reader, fine-tuned end to end (LoRA) to answer BABILong questions directly from the whole text, with no memory.

python3 b23.py prep  --model M --out b23_data                       (CPU) training + dev examples, label sets, token counts
python3 b23.py testlen --model M --out b23_data/testlen.json        (CPU) prompt token counts of the test inputs (no labels)
python3 b23.py train --arm N|Y --model M --data b23_data --seed 0 --out b23_models/Y_s0
python3 b23.py eval  --arm N|Y --model M --adapter DIR --split dev|test --out b23_out/test_Y_s0.jsonl
python3 b23.py gate  --dev b23_out/dev_Y_s0.jsonl                   exit 0 iff dev QA1-5 mean at 0k >= 0.90 (bar 0)
python3 b23.py score ...                                            (see score())
"""
import argparse
import glob
import json
import math
import os
import random
import time

BENCH = os.environ.get("BENCH_ROOT", "/path/to/pace-scratch/bench")
TRAIN_DIRS = [f"{BENCH}/RMT-team__babilong-train-5k-samples/data", f"{BENCH}/b23_train/data"]
TEST_DIR = f"{BENCH}/RMT-team__babilong/data"
TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]
# Training indices per length (the same index holds the same bAbI story at every length, so the ranges are disjoint
# and no story is seen twice). qa3 and qa5 have no 1k file in the training split, and qa4's has only 999 samples:
# the 1k slot of these three tasks is read from 2k.
TRAIN = {"0k": (0, 2000), "1k": (2000, 1000), "2k": (3000, 500), "4k": (3500, 500), "8k": (4000, 300),
         "16k": (4300, 200), "32k": (4500, 100), "64k": (4600, 50)}
NO_1K = {"qa3", "qa4", "qa5"}
DEV = (4900, 20)
DEV_LENGTHS = ["0k", "4k", "16k", "32k", "64k"]
TEST_LENGTHS = ["0k", "4k", "16k", "32k", "64k", "128k"]
NATIVE = 32768  # Qwen3 native window; YaRN factor 4 on top of it, as the Qwen3 references (B19, B20)
MAX_POS = 40960  # Qwen3-1.7B max_position_embeddings
MAX_NEW = 32
DEVICE = os.environ.get("B23_DEVICE", "cuda")  # "cpu" only for the local smoke test with a tiny random model
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # = B18


def prompt(text, question):
    """The direct-answer prompt of the B18 Qwen3 and frontier references (b18_llm.prep), verbatim."""
    return (f"{text}\n\nAnswer the question about the people and objects in the text above. "
            f"Question: {question}\nGive only the answer: a single word (a place, a name, an object, a "
            f"number word, yes / no / maybe), or a comma-separated list for 'what is ... carrying' (or 'nothing').")


def train_file(task, length):
    for d in TRAIN_DIRS:
        f = f"{d}/{task}/{length}.json"
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"{task}/{length}")


def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def rope_kwargs(factor):
    return {} if not factor else {"rope_scaling": {"rope_type": "yarn", "factor": float(factor),
                                                   "original_max_position_embeddings": NATIVE}}


def rope_for(arm, length, longest):
    """Prereg window rule. Arm N: native up to 32k (if the longest prompt + output fits in max_position_embeddings),
    YaRN beyond; arm Y: YaRN at every length. YaRN factor 4, raised to ceil((longest + 32) / 32768) if needed."""
    need = longest + MAX_NEW
    f = max(4, math.ceil(need / NATIVE))
    if arm == "N" and length in ("0k", "1k", "2k", "4k", "8k", "16k", "32k") and need <= MAX_POS:
        return 0
    return f


# ----------------------------------------------------------------------------------------------------------- prep
def prep(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(args.out, exist_ok=True)
    stats, n_tr = {}, 0
    with open(f"{args.out}/train.jsonl", "w") as ft, open(f"{args.out}/dev.jsonl", "w") as fd:
        for task in TASKS:
            cache = {}

            def rows(length):
                if length not in cache:
                    cache.clear()  # one length in memory at a time (64k files are 1.3 GB)
                    cache[length] = json.load(open(train_file(task, length)))
                return cache[length]

            for length, (a, n) in TRAIN.items():
                src = "2k" if (length == "1k" and task in NO_1K) else length
                rs = rows(src)
                stats.setdefault("rows_per_file", {})[f"{task}/{src}"] = len(rs)
                assert len(rs) >= DEV[0] + DEV[1], (task, src, len(rs))
                for i in range(a, a + n):
                    r = rs[i]
                    rec = {"task": task, "length": length, "src": src, "i": i, "prompt": prompt(r["input"], r["question"]),
                           "target": str(r["target"]).strip()}
                    ft.write(json.dumps(rec) + "\n")
                    n_tr += 1
                    if i < a + 40:  # token counts on the first 40 per (task, length)
                        stats.setdefault(f"train/{length}", []).append(len(tok(chat(tok, rec["prompt"])).input_ids))
                if length in DEV_LENGTHS:
                    for i in range(DEV[0], DEV[0] + DEV[1]):
                        r = rs[i]
                        rec = {"split": "dev", "task": task, "length": length, "i": i,
                               "prompt": prompt(r["input"], r["question"]), "question": r["question"],
                               "target": str(r["target"]).strip()}
                        fd.write(json.dumps(rec) + "\n")
                        stats.setdefault(f"dev/{length}", []).append(len(tok(chat(tok, rec["prompt"])).input_ids))
                print(task, length, "from", src, flush=True)
    files = stats.pop("rows_per_file")
    print("rows per file:", files)
    summ = {k: {"n": len(v), "mean": round(sum(v) / len(v)), "max": max(v), "min": min(v)} for k, v in stats.items()}
    for k, v in summ.items():
        print(k, v)
    # label sets of our scorer (b18.labels: answers seen in the training split, as in B18 -- 0k/4k/16k files)
    import b18
    labs = {t: sorted(b18.labels(t)) for t in TASKS}
    import babilong_metrics_official as off
    for t in TASKS:
        print(t, "labels", labs[t], "official", off.TASK_LABELS[t],
              "SAME" if set(labs[t]) == {x.lower() for x in off.TASK_LABELS[t]} else "DIFFERENT")
    json.dump(labs, open(f"{args.out}/labels.json", "w"), indent=1)
    json.dump({"train_examples": n_tr, "rows_per_file": files, "tokens": summ}, open(f"{args.out}/stats.json", "w"), indent=1)
    print("train examples", n_tr)


def testlen(args):
    """Prompt token counts of the test inputs (lengths only; needed for the prereg window rule)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    out = {}
    for length in TEST_LENGTHS:
        ns = []
        for task in TASKS:
            for r in json.load(open(f"{TEST_DIR}/{task}/{length}.json"))[:100]:
                ns.append(len(tok(chat(tok, prompt(r["input"], r["question"]))).input_ids))
        out[length] = {"n": len(ns), "mean": round(sum(ns) / len(ns)), "max": max(ns), "min": min(ns),
                       "rope_N": rope_for("N", length, max(ns)), "rope_Y": rope_for("Y", length, max(ns))}
        print(length, out[length], flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------- train
def gpu_mem():
    import torch
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def load_model(path, factor, adapter=None, train=False):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEVICE,
                                                 attn_implementation="sdpa", **rope_kwargs(factor))
    rot = getattr(getattr(model, "model", None), "rotary_emb", None)
    print("rope:", getattr(model.config, "rope_scaling", None), "attention_scaling",
          getattr(rot, "attention_scaling", None), flush=True)  # YaRN factor 4 -> 0.1 ln 4 + 1 = 1.139
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model if train else model.eval()


def train(args):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    factor = 4 if args.arm == "Y" else 0
    ex = [json.loads(l) for l in open(f"{args.data}/train.jsonl")]
    if args.arm == "N":
        ex = [e for e in ex if e["length"] != "64k"]  # arm N: lengths inside the native window only
    rng.shuffle(ex)
    if args.limit:  # smoke tests only
        ex = ex[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    data = []
    for e in ex:
        h = tok(chat(tok, e["prompt"]), add_special_tokens=False).input_ids
        o = tok(e["target"] + "<|im_end|>", add_special_tokens=False).input_ids
        data.append((h + o, len(o)))
    lens = [len(x) for x, _ in data]
    dropped = 0
    if args.arm == "N":  # native RoPE: never train past max_position_embeddings (prereg; expected to drop none)
        keep = [d for d in data if len(d[0]) <= MAX_POS]
        dropped, data = len(data) - len(keep), keep
    print(f"arm {args.arm}: {len(data)} examples ({dropped} dropped), {sum(lens) / 1e6:.1f}M tokens, longest "
          f"{max(lens)}, tokenized in {time.time() - t0:.0f}s", flush=True)
    model = load_model(args.model, factor, train=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**LORA))
    model.train()  # from_pretrained returns eval mode, where gradient checkpointing is skipped (OOM at 66k tokens)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = len(data) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    os.makedirs(args.out, exist_ok=True)
    ck = f"{args.out}/ckpt"
    step = 0
    if os.path.exists(f"{ck}/state.pt"):  # resume (data order is fixed by the seed)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(f"{ck}/adapter_model.safetensors"))
        st = torch.load(f"{ck}/state.pt", map_location=DEVICE, weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        print("resumed at step", step, flush=True)

    def fwd_bwd(ids, m):
        x = torch.tensor([ids], device=DEVICE)
        out = model(input_ids=x, logits_to_keep=m + 1)
        logits = out.logits[0, :-1].float()  # the m positions that predict the m target tokens
        loss = F.cross_entropy(logits, x[0, -m:])
        (loss / args.accum).backward()
        return loss.item()

    if step == 0:  # preflight: the longest example first, so an out-of-memory error shows in minutes
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        j = max(range(len(data)), key=lambda k: len(data[k][0]))
        tp = time.time()
        fwd_bwd(*data[j])
        opt.zero_grad(set_to_none=True)
        print(f"preflight: {len(data[j][0])} tokens fwd+bwd {time.time() - tp:.1f}s, peak "
              f"{gpu_mem():.1f} GiB on {gpu_name()}", flush=True)
    t0, run_loss, run_tok = time.time(), [], 0
    while step < steps:
        batch = data[step * args.accum:(step + 1) * args.accum]
        for ids, m in batch:
            run_loss.append(fwd_bwd(ids, m))
            run_tok += len(ids)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0 or step == steps:
            el = time.time() - t0
            print(f"step {step}/{steps} loss {sum(run_loss) / len(run_loss):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{run_tok / el:.0f} tok/s elapsed {el:.0f}s peak {gpu_mem():.1f} GiB",
                  flush=True)
            run_loss = []
        if step % 100 == 0 and step < steps:
            model.save_pretrained(ck)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, f"{ck}/state.pt.tmp")
            os.replace(f"{ck}/state.pt.tmp", f"{ck}/state.pt")
    model.save_pretrained(args.out)
    json.dump({"arm": args.arm, "seed": args.seed, "examples": len(data), "steps": steps, "lr": args.lr,
               "accum": args.accum, "yarn_factor": factor, "lora": LORA, "secs": round(time.time() - t0)},
              open(f"{args.out}/b23_train.json", "w"), indent=1)
    print("saved", args.out, flush=True)


# ----------------------------------------------------------------------------------------------------------- eval
def eval_rows(split, lengths, tasks, data_dir):
    if split == "dev":
        rs = [json.loads(l) for l in open(f"{data_dir}/dev.jsonl")]
        return [r for r in rs if r["length"] in lengths and r["task"] in tasks]
    out = []
    for length in lengths:
        for task in tasks:
            for i, r in enumerate(json.load(open(f"{TEST_DIR}/{task}/{length}.json"))[:100]):
                out.append({"split": "test", "task": task, "length": length, "i": i,
                            "prompt": prompt(r["input"], r["question"]), "question": r["question"],
                            "target": str(r["target"]).strip()})
    return out


def evaluate(args):
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(args.model)
    lengths = args.lengths.split(",") if args.lengths else (DEV_LENGTHS if args.split == "dev" else TEST_LENGTHS)
    done = set()
    if os.path.exists(args.out):
        done = {(r["task"], r["length"], r["i"]) for r in map(json.loads, open(args.out))}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model, cur = None, None
    for length in lengths:
        rows = eval_rows(args.split, [length], args.tasks.split(","), args.data)
        enc = [tok(chat(tok, r["prompt"]), add_special_tokens=False).input_ids for r in rows]
        factor = rope_for(args.arm, length, max(len(e) for e in enc))
        todo = [(r, e) for r, e in zip(rows, enc) if (r["task"], r["length"], r["i"]) not in done]
        print(f"{args.split} {length}: {len(rows)} rows, {len(todo)} to do, longest {max(len(e) for e in enc)} "
              f"tokens, yarn factor {factor or 'none (native)'}", flush=True)
        if not todo:
            continue
        if model is None or cur != factor:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_model(args.model, factor, adapter=args.adapter)
            eos = [151645, 151643]  # <|im_end|>, <|endoftext|>
            model.generation_config = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos,
                                                       pad_token_id=tok.pad_token_id)  # installed (see b6_qwen.py)
            cur = factor
        t0 = time.time()
        with open(args.out, "a") as f, torch.no_grad():
            for k, (r, ids) in enumerate(todo):  # one sample at a time: no padding, no batch numerics
                ts = time.time()
                x = torch.tensor([ids], device=DEVICE)
                g = model.generate(input_ids=x, attention_mask=torch.ones_like(x))[0, len(ids):].tolist()
                n = next((j + 1 for j, t in enumerate(g) if t in (151645, 151643)), len(g))
                f.write(json.dumps({"split": r["split"], "arm": args.arm, "adapter": args.adapter, "task": r["task"],
                                    "length": r["length"], "i": r["i"], "question": r["question"],
                                    "target": r["target"], "output": tok.decode(g[:n], skip_special_tokens=True),
                                    "gen_tokens": n, "hit_limit": n >= MAX_NEW and g[n - 1] not in (151645, 151643),
                                    "prompt_tokens": len(ids), "yarn_factor": factor,
                                    "secs": round(time.time() - ts, 2)}) + "\n")
                f.flush()
                if (k + 1) % 50 == 0:
                    print(f"  {length} {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        print(f"{args.split} {length} done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------- score
_LABELS_SET = False


def set_labels(path):
    """Our scorer (b18.correct_official) with the label sets it computes from the training split, cached by prep."""
    global _LABELS_SET
    import b18
    if not _LABELS_SET:
        for t, v in json.load(open(path)).items():
            b18._LABELS[t] = set(v)
        _LABELS_SET = True
    return b18


def ok_ours(r, labels_path):
    return bool(set_labels(labels_path).correct_official(r["output"], r["target"], r["question"], r["task"]))


def ok_official(r):
    import babilong_metrics_official as off
    return bool(off.compare_answers(r["target"], r["output"], r["question"], off.TASK_LABELS[r["task"]]))


def gate(args):
    rs = [json.loads(l) for l in open(args.dev)]
    rs = [r for r in rs if r["length"] == "0k"]
    acc = sum(ok_ours(r, args.labels) for r in rs) / max(1, len(rs))
    print(f"bar 0 (dev QA1-5 mean at 0k >= 0.90): {acc:.3f} on {len(rs)} -> {'PASS' if acc >= 0.90 else 'FAIL'}")
    raise SystemExit(0 if len(rs) == 100 and acc >= 0.90 else 3)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p: binomial test of b vs c discordant pairs at 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model", default="")
    ap.add_argument("--arm", default="Y", choices=["N", "Y"])
    ap.add_argument("--data", default="b23_data")
    ap.add_argument("--labels", default="b23_data/labels.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--lengths", default="")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--dev", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"prep": prep, "testlen": testlen, "train": train, "eval": evaluate, "gate": gate}[a.cmd](a)
