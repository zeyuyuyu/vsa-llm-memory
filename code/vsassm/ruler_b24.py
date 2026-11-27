"""B24 (prereg/B24_ruler_store_ablation_prereg.md): does the VSA store matter beyond the reader's extraction?

Two baselines on the 9 RULER tasks that use the memory, with the same Qwen3-8B, the same test samples and the official
scorer (string_match_all). Nothing is trained.

  Arm F, facts in the prompt: the grounded pairs that B21's store received (saved in the B21 output; no re-reading),
         written as "key: value" lines in reading order in place of the text; Qwen3-8B answers the RULER question.
  Arm R, standard RAG: BM25 over the ~250-character chunks of the text (ruler_rm.chunks), query = the question; the
         top K chunks, in document order, in place of the text; Qwen3-8B answers.

python3 ruler_b24.py armf OUT.jsonl --model M --root RULER_DIR --length L --tasks t1,t2 --b21 B21_DIR [--n 100]
python3 ruler_b24.py armr OUT.jsonl --model M --root RULER_DIR --length L --tasks t1,t2 [--k 20] [--n 100]
python3 ruler_b24.py promptlen OUT.json --model M --root RULER_DIR --length L --tasks t1,t2 --b21 B21_DIR
python3 ruler_b24.py selftest [--root DEV_DIR]

Decoding is B19 direct's (ruler_rm.run_direct): chat template, thinking off, RULER's answer prefix starts the model's
turn, greedy, at most 2 x RULER's tokens_to_generate, one sample at a time (no batching, no padding). A prompt longer
than 32,768 tokens (template and answer prefix included) is run with YaRN factor 4 (original 32,768), as B19 direct
ran 64k / 128k; every other prompt runs natively. Output rows record the prompt length and whether YaRN was on.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import subprocess

import ruler_rm as R

K_DEFAULT = 20
K1, B = 1.2, 0.75
NATIVE = 32768
FACT_NOTE = ('The text has been replaced by the key-value facts read from it, in reading order, one per line as '
             '"key: value" (an assignment VAR X = VAR Y is the line "X: Y").')
RAG_NOTE = 'The text has been replaced by the {n} passages of it that best match the question, in their original order.'
TOK = re.compile(r"\w+")


# ---------------------------------------------------------------- prompt parts
def parts(s):
    """(prefix, body, question) of a RULER input. prefix = everything up to and including the last occurrence of the
    task instruction (the input's first line) and the newlines after it: for NIAH the instruction line, for VT the
    in-context example plus the instruction. body = the text (haystack) the model is asked about."""
    ctx, q = R.split(s)
    head = ctx.split("\n", 1)[0]
    j = ctx.rindex(head) + len(head)
    while j < len(ctx) and ctx[j] == "\n":
        j += 1
    prefix, body = ctx[:j], ctx[j:]
    assert prefix + body + "\n" + q == s["input"].rstrip(), "input does not split into prefix / body / question"
    return prefix, body, q


def gen_len(task):
    return 2 * R.GEN["niah" if task.startswith("niah") else task.split("_")[0]]


# ---------------------------------------------------------------- arm F
def replay_cased(ctx, raw, pairs):
    """The B21 pairs with each key in the case the reader wrote it. Replays B21's grounded_pairs over the saved raw
    reader outputs (no model); the replay must give back the saved pairs exactly once keys are lower-cased."""
    cs = R.chunks(ctx)
    assert len(cs) == len(raw), (len(cs), len(raw))
    seen, out, check = collections.defaultdict(list), [], []
    for c, txt in zip(cs, raw):
        for k, v in R.PAIR.findall(txt):
            k, v = re.sub(r"^VAR\s+", "", k.strip()), re.sub(r"^VAR\s+", "", v.strip().rstrip("."))
            if k.lower() in c.lower() and v.lower() in c.lower() and k.lower() != v.lower():
                if v not in seen[k.lower()]:
                    seen[k.lower()].append(v)
                    out.append((k, v))
                    check.append([k.lower(), v, len(seen[k.lower()]) - 1])
    assert check == [list(p) for p in pairs], "replay differs from the saved B21 pairs"
    return out


def prompt_f(s, b21row):
    prefix, body, q = parts(s)
    ctx, _ = R.split(s)
    facts = replay_cased(ctx, b21row["raw"], b21row["pairs"])
    lines = "\n".join(f"{k}: {v}" for k, v in facts)
    return prefix + FACT_NOTE + "\n" + lines + "\n" + q, {"n_facts": len(facts)}


def load_b21(d, length, task):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, f"{length}_*.jsonl"))) for l in open(f)]
    return [r for r in rows if r["task"] == task and int(r["length"]) == int(length)]


# ---------------------------------------------------------------- arm R
def bm25(docs, query, k1=K1, b=B):
    """Okapi BM25 scores of each doc for the query. Tokens: lower-cased \\w+ runs, no stemming, no stop list; each
    distinct query term counted once; idf = ln(1 + (N - df + 0.5) / (df + 0.5))."""
    toks = [TOK.findall(d.lower()) for d in docs]
    n = len(toks)
    avg = sum(map(len, toks)) / max(1, n)
    df = collections.Counter(t for d in toks for t in set(d))
    qt = list(dict.fromkeys(TOK.findall(query.lower())))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in qt}
    out = []
    for d in toks:
        tf = collections.Counter(d)
        norm = k1 * (1 - b + b * len(d) / avg) if avg else k1
        out.append(sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in qt if tf[t]))
    return out


def retrieve(body, q, k):
    cs = R.chunks(body)
    sc = bm25(cs, q)
    top = sorted(range(len(cs)), key=lambda i: (-sc[i], i))[:k]   # ties broken by document order
    return cs, sorted(top), sc


def prompt_r(s, k):
    prefix, body, q = parts(s)
    cs, keep, sc = retrieve(body, q, k)
    txt = "\n".join(cs[i] for i in keep)
    return prefix + RAG_NOTE.format(n=len(keep)) + "\n" + txt + "\n" + q, {"n_chunks": len(cs), "kept": keep,
                                                                             "min_kept_score": round(min(sc[i] for i in keep), 3) if keep else None}


# ---------------------------------------------------------------- model
def chat_text(tok, prompt, s):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return text + s.get("answer_prefix", "")


def load_model(path, yarn):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": NATIVE}} if yarn else {}
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa", **kw).eval()


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                               os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def items(args, arm):
    """Every (row meta, prompt) of this job, in data order; pos = position of the sample in RULER's file."""
    out = []
    for task in args.tasks.split(","):
        samples = R.load(args.root, args.length, task, args.n)
        b21 = load_b21(args.b21, args.length, task) if arm == "f" else None
        if b21 is not None:
            assert len(b21) == len(samples), (task, len(b21), len(samples))
        for pos, s in enumerate(samples):
            if arm == "f":
                row = b21[pos]
                assert row["index"] == s["index"] and row["outputs"] == s["outputs"], (task, pos)
                p, extra = prompt_f(s, row)
            else:
                p, extra = prompt_r(s, args.k)
            out.append(({"task": task, "length": int(args.length), "index": s["index"], "pos": pos,
                         "outputs": s["outputs"], **extra}, p, s))
    return out


def run(args, arm):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    done = set()
    if os.path.exists(args.out):   # resume after a pre-emption: skip finished samples
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["task"], r["pos"]))
    todo = []
    for meta, p, s in items(args, arm):
        if (meta["task"], meta["pos"]) in done:
            continue
        text = chat_text(tok, p, s)
        ids = tok(text, return_tensors="pt").input_ids
        todo.append((meta, ids, ids.shape[1] > NATIVE))
    gpu = gpu_name()
    print(f"{arm} {args.length}: {len(todo)} to run ({len(done)} done), yarn {sum(t[2] for t in todo)}, "
          f"max prompt {max([t[1].shape[1] for t in todo], default=0)} tokens, gpu {gpu}", flush=True)
    with open(args.out, "a") as f:
        for yarn in (False, True):
            group = [t for t in todo if t[2] == yarn]
            if not group:
                continue
            model = load_model(args.model, yarn)
            for meta, ids, _ in group:
                ids = ids.to("cuda")
                with torch.no_grad():
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=gen_len(meta["task"]), do_sample=False)
                pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({**meta, "arm": arm, "pred": pred, "prompt_tokens": int(ids.shape[1]),
                                    "yarn": yarn, "gpu": gpu}) + "\n")
                f.flush()
            del model
            torch.cuda.empty_cache()
            print(f"{arm} {args.length} yarn={yarn}: {len(group)} done", flush=True)


def promptlen(args):
    """Prompt token counts of both arms (no model, no answers read)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    res = {}
    for arm in ("f", "r"):
        for meta, p, s in items(args, arm):
            n = len(tok(chat_text(tok, p, s)).input_ids)
            res.setdefault(f"{arm}/{meta['task']}", []).append(n)
    out = {k: {"n": len(v), "max": max(v), "median": sorted(v)[len(v) // 2], "over_32768": sum(x > NATIVE for x in v),
               "sum": sum(v)} for k, v in res.items()}
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in out.items():
        print(args.length, k, v, flush=True)


# ---------------------------------------------------------------- self-test (CPU, dev data or synthetic)
def selftest(args):
    docs = ["the cat sat on the mat", "a special magic number for blue-harbor is 5512", "the dog", "blue sky"]
    sc = bm25(docs, "What is the special magic number for blue-harbor?")
    assert max(range(4), key=lambda i: sc[i]) == 1 and sc[2] == 0.0 or sc[2] < sc[1], sc
    # BM25 against a direct formula on a two-document corpus
    sc2 = bm25(["x y", "y y z"], "y")
    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    avg = 2.5
    want = [idf * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / avg)), idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / avg))]
    assert all(abs(a - b) < 1e-9 for a, b in zip(sc2, want)), (sc2, want)
    print("selftest bm25: ok")
    if not args.root:
        return
    for f in sorted(glob.glob(os.path.join(args.root, "*.jsonl"))):
        for l in list(open(f))[:5]:
            s = json.loads(l)
            prefix, body, q = parts(s)
            p, extra = prompt_r(s, K_DEFAULT)
            assert len(extra["kept"]) == min(K_DEFAULT, extra["n_chunks"])
        print(os.path.basename(f), "parts ok; prefix", repr(prefix[-90:]), "| q", repr(q[:70]), "| chunks",
              extra["n_chunks"], "kept", extra["kept"][:6], flush=True)
    print("selftest parts / retrieval on", args.root, ": ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["armf", "armr", "promptlen", "selftest"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--model")
    ap.add_argument("--root")
    ap.add_argument("--length")
    ap.add_argument("--tasks", default=",".join(R.NIAH + ["vt"]))
    ap.add_argument("--b21")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    a = ap.parse_args()
    if a.mode in ("armf", "armr", "promptlen"):
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.mode == "armf":
        run(a, "f")
    elif a.mode == "armr":
        run(a, "r")
    elif a.mode == "promptlen":
        promptlen(a)
    else:
        selftest(a)
