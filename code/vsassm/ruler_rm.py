"""B19 (prereg/B19_ruler_prereg.md): RULER with the readable memory, and Qwen3-8B answering directly.

python3 ruler_rm.py rm     --model <Qwen3-8B> --root RULER_DIR --length 4096 --tasks niah_single_1,... --out OUT.jsonl
python3 ruler_rm.py direct --model <Qwen3-8B> --root RULER_DIR --length 131072 --tasks ... --out OUT.jsonl [--yarn]
python3 ruler_rm.py score  OUT_DIR
RULER_DIR/<length>/<task>/validation.jsonl as written by RULER's prepare.py.
"""
import argparse
import collections
import glob
import json
import os
import re

FACT_PROMPT = """Extract key-value facts. Copy keys and values exactly; for "VAR B = VAR A" the value is A. Output a JSON list of pairs.

Passage: The sky is blue. VAR KLM = VAR XYZ The sun is yellow.
JSON: [["KLM", "XYZ"]]

Passage: We walked home. The cabin code for blue-harbor is: 5512. The locker code for quiet-lamp is: 9031.
JSON: [["blue-harbor", "5512"], ["quiet-lamp", "9031"]]

Passage: One of the special secret ids for 1a2b3c4d-0000-4e5f-8a9b-123456789abc is: 9f8e7d6c-1111-4b2a-9c3d-abcdefabcdef. Here we go.
JSON: [["1a2b3c4d-0000-4e5f-8a9b-123456789abc", "9f8e7d6c-1111-4b2a-9c3d-abcdefabcdef"]]

Passage: It rained all day and nobody came.
JSON: []

Passage: {chunk}
JSON:"""

REL_PROMPT = """Question: {q}

Document:
{doc}

Would this document help answer the question? Reply with only YES or NO."""

ANSWER_PROMPT = """Answer the question based on the given documents. Only give me the answer and do not output any other words.

The following are given documents.

{docs}

Answer the question based on the given documents. Only give me the answer and do not output any other words.

Question: {q}"""

PAIR = re.compile(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]')
NIAH = [f"niah_{t}" for t in ["single_1", "single_2", "single_3", "multikey_1", "multikey_2", "multikey_3",
                              "multivalue", "multiquery"]]
TASKS = NIAH + ["vt", "cwe", "fwe", "qa_1", "qa_2"]
GEN = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}  # RULER's tokens_to_generate


def load(root, length, task, n):
    return [json.loads(l) for l in open(f"{root}/{length}/{task}/validation.jsonl")][:n]


def split(sample):
    """(context, question): the question is the input's last line."""
    ctx, q = sample["input"].rstrip().rsplit("\n", 1)
    return ctx, q.strip()


def chunks(text, size=250):
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip()
        if not s:
            continue
        if cur and len(cur) + len(s) + 1 > size:
            out.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


class CachedReader:
    """b14.Reader with a cache: identical prompts (repeated haystack chunks) are decoded once (greedy, deterministic)."""

    def __init__(self, path):
        import b14
        self.r = b14.Reader(path)
        self.cache, self.calls, self.unique, self.tokens = {}, 0, 0, 0

    def __call__(self, prompts, max_new):
        self.r.model.generation_config.max_new_tokens = max_new
        todo = sorted({p for p in prompts if p not in self.cache})
        if todo:
            for p, (txt, n) in zip(todo, self.r(todo, bs=64, budget=64 * 2400)):
                self.cache[p] = txt
                self.tokens += n
        self.calls += len(prompts)
        self.unique += len(todo)
        return [self.cache[p] for p in prompts]


def facts_of(reader, ctx):
    """Fact table: key (lower case) -> values in reading order. A pair is kept only if key and value both occur in
    the chunk (grounding, as for BABILong)."""
    cs = chunks(ctx)
    table = collections.defaultdict(list)
    for c, txt in zip(cs, reader([FACT_PROMPT.format(chunk=c) for c in cs], 400)):
        for k, v in PAIR.findall(txt):
            k, v = re.sub(r"^VAR\s+", "", k.strip()), re.sub(r"^VAR\s+", "", v.strip().rstrip("."))
            if k.lower() in c.lower() and v.lower() in c.lower() and k.lower() != v.lower():
                if v not in table[k.lower()]:
                    table[k.lower()].append(v)
    return table


def answer_niah(table, q):
    m = re.search(r"for (.+?) mentioned", q)
    keys = re.split(r",\s*and\s+|,\s*|\s+and\s+", m.group(1)) if m else []
    return " ".join(v for k in keys for v in table.get(k.strip().lower(), []))


def answer_vt(table, q):
    m = re.search(r"assigned the value (\S+?)\s+in", q)
    target = m.group(1) if m else ""

    def resolve(x, seen=()):
        vals = table.get(x.lower(), [])
        if not vals or x in seen:
            return x
        return resolve(vals[-1], seen + (x,))
    return " ".join(k.upper() for k in table if resolve(k) == target)


def answer_count(ctx, q, task):
    """Counter memory (no model): CWE counts the numbered list's words, FWE the coded words (dots ignored)."""
    if task == "cwe":
        words = re.findall(r"\d+\.\s+(\S+)", ctx)
        k = 10
    else:
        words = [w for w in ctx.split() if not re.fullmatch(r"\.+", w)]
        k = 3
    m = re.search(r"(\d+|three|ten) most", q)
    if m:
        k = {"three": 3, "ten": 10}.get(m.group(1), None) or int(m.group(1))
    return " ".join(w for w, _ in collections.Counter(words).most_common(k))


def answer_qa(reader, ctx, q, cap=20):
    query = q.split("Question:", 1)[-1].strip()
    docs = re.split(r"Document \d+:\n", ctx)[1:]
    docs = [d.strip() for d in docs]
    rel = reader([REL_PROMPT.format(q=query, doc=d) for d in docs], 3)
    keep = [d for d, r in zip(docs, rel) if r.strip().upper().startswith("YES")][:cap]
    body = "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(keep))
    return reader([ANSWER_PROMPT.format(docs=body, q=query)], 32)[0].strip(), len(keep), len(docs)


def run_rm(args):
    reader = CachedReader(args.model)
    with open(args.out, "a") as f:
        for task in args.tasks.split(","):
            for s in load(args.root, args.length, task, args.n):
                ctx, q = split(s)
                extra = {}
                if task.startswith("niah"):
                    t = facts_of(reader, ctx)
                    pred = answer_niah(t, q)
                    extra["n_keys"] = len(t)
                elif task == "vt":
                    pred = answer_vt(facts_of(reader, ctx), q)
                elif task in ("cwe", "fwe"):
                    pred = answer_count(ctx, q, task)
                else:
                    pred, kept, nd = answer_qa(reader, ctx, q)
                    extra.update(relevant=kept, docs=nd)
                f.write(json.dumps({"task": task, "length": args.length, "index": s["index"], "pred": pred,
                                    "outputs": s["outputs"], **extra}) + "\n")
                f.flush()
            print(f"{task} {args.length}: calls {reader.calls} unique {reader.unique} gen tokens {reader.tokens}",
                  flush=True)


def run_direct(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {}
    if args.yarn:
        kw["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", **kw).eval()
    with open(args.out, "a") as f:
        for task in args.tasks.split(","):
            gen = GEN["niah" if task.startswith("niah") else task.split("_")[0]]
            for s in load(args.root, args.length, task, args.n):
                text = tok.apply_chat_template([{"role": "user", "content": s["input"]}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                text += s.get("answer_prefix", "")  # RULER's protocol: the answer prefix starts the model's turn
                enc = tok(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=gen * 2, do_sample=False)
                pred = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"task": task, "length": args.length, "index": s["index"], "pred": pred,
                                    "outputs": s["outputs"]}) + "\n")
                f.flush()
            print(task, args.length, "done", flush=True)


def string_match_all(pred, refs):
    return sum(r.lower() in pred.lower() for r in refs) / len(refs)


def string_match_part(pred, refs):
    return max(float(r.lower() in pred.lower()) for r in refs)


def score(args):
    rows = [json.loads(l) for f in glob.glob(f"{args.out}/*.jsonl") for l in open(f)]
    seen, acc = set(), collections.defaultdict(list)
    for r in rows:
        key = (r["task"], r["length"], r["index"], json.dumps(r["outputs"]))  # RULER repeats some indices
        if key in seen:
            continue
        seen.add(key)
        fn = string_match_part if r["task"].startswith("qa") else string_match_all
        acc[(r["task"], r["length"])].append(100 * fn(r["pred"], r["outputs"]))
    lengths = sorted({k[1] for k in acc}, key=int)
    print("task".ljust(16) + "".join(f"{int(l) // 1024}k".rjust(8) for l in lengths))
    for t in TASKS:
        print(t.ljust(16) + "".join((f"{sum(v) / len(v):8.1f}" if (v := acc.get((t, l))) else "       -")
                                     for l in lengths))
    for name, ts in [("avg13", TASKS), ("avg w/o cwe,fwe", [t for t in TASKS if t not in ("cwe", "fwe")]),
                     ("niah+vt", NIAH + ["vt"]), ("qa", ["qa_1", "qa_2"])]:
        cells = []
        for l in lengths:
            ms = [sum(acc[(t, l)]) / len(acc[(t, l)]) for t in ts if acc.get((t, l))]
            cells.append(f"{sum(ms) / len(ms):8.1f}" if len(ms) == len(ts) else "       -")
        print(name.ljust(16) + "".join(cells))
    print("n per cell:", sorted({len(v) for v in acc.values()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rm", "direct", "score"])
    ap.add_argument("out")
    ap.add_argument("--model")
    ap.add_argument("--root")
    ap.add_argument("--length")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--yarn", action="store_true")
    a = ap.parse_args()
    if a.mode != "score":
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    {"rm": run_rm, "direct": run_direct, "score": score}[a.mode](a)
