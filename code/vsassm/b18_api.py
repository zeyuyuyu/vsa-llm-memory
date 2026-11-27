"""B18 amendment 6: frontier API models answering BABILong questions directly (zero-shot), on the official test
samples our system was scored on (b18_api_subset.jsonl.gz, built on PACE from RMT-team/babilong + training labels).

python3 b18_api.py count                      # Opus token counts -> which samples fit the 1M window
python3 b18_api.py opus-submit                # Anthropic Message Batches (effort high), batch ids -> b18_api/opus_batches.json
python3 b18_api.py opus-collect               # poll + write b18_api/opus.jsonl
python3 b18_api.py run --model openai/gpt-6.1-sol --provider OpenAI --name gpt61 --cap 200   # via OpenRouter
python3 b18_api.py score b18_api/opus.jsonl   # official metric, QA1-5 means per length
"""
import argparse
import collections
import concurrent.futures as cf
import gzip
import json
import os
import re
import time

SUBSET = "b18_api_subset.jsonl.gz"
OUT = "b18_api"
OPUS = "claude-opus-5-5"
GPT = "openai/gpt-6.1-sol"
MAX_OUT = 16000
OPUS_CTX = 1_000_000
LENGTHS = ["0k", "4k", "16k", "64k", "128k", "512k", "1M"]


def load():
    labels, rows = {}, []
    for r in map(json.loads, gzip.open(SUBSET, "rt")):
        if "labels" in r:
            labels[r["task"]] = set(r["labels"])
        else:
            r["id"] = f"{r['task']}-{r['length']}-{r['i']}"
            rows.append(r)
    return labels, rows


def prompt(r):  # identical to b18_llm.prep (the Qwen3 references)
    return (f"{r['input']}\n\nAnswer the question about the people and objects in the text above. "
            f"Question: {r['question']}\nGive only the answer: a single word (a place, a name, an object, a "
            f"number word, yes / no / maybe), or a comma-separated list for 'what is ... carrying' (or 'nothing').")


def correct_official(pred, target, question, labels):  # = b18.correct_official with the task's labels passed in
    if pred is None:
        return False
    out = str(pred).lower().split(".")[0]
    q = question.lower()
    if labels is None:
        return False
    labs = {l for l in labels if not re.search(r"\b" + re.escape(l) + r"\b", q)}
    found = {l for l in labs if re.search(r"\b" + re.escape(l) + r"\b", out)}
    tgt = {x.strip() for x in str(target).lower().split(",")}
    return tgt <= found and not (found - tgt) if len(tgt) > 1 else found == tgt


def anthropic_client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for line in open(os.path.expanduser("~/clerk/.env")):
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
    return anthropic.Anthropic(max_retries=5)


def count(_):
    """Opus input tokens of every sample (count_tokens is free); 0k-64k are estimated from one sample per length."""
    client = anthropic_client()
    _, rows = load()
    path = f"{OUT}/opus_tokens.json"
    toks = json.load(open(path)) if os.path.exists(path) else {}
    for r in rows:
        if r["id"] in toks:
            continue
        n = client.messages.count_tokens(model=OPUS, messages=[{"role": "user", "content": prompt(r)}]).input_tokens
        toks[r["id"]] = n
        json.dump(toks, open(path, "w"))
        print(r["id"], n, flush=True)
    for length in LENGTHS:
        v = [toks[r["id"]] for r in rows if r["length"] == length]
        print(length, f"mean {sum(v) / len(v):.0f} max {max(v)} over-window {sum(x + MAX_OUT > OPUS_CTX for x in v)}")


def opus_submit(_):
    client = anthropic_client()
    _, rows = load()
    toks = json.load(open(f"{OUT}/opus_tokens.json"))
    ok = [r for r in rows if toks[r["id"]] + MAX_OUT <= OPUS_CTX]
    print(len(ok), "requests,", len(rows) - len(ok), "exceed the context window (not sent)")
    batches, cur, size = [], [], 0
    for r in ok:  # keep each batch well under the 256 MB request limit
        req = {"custom_id": r["id"], "params": {"model": OPUS, "max_tokens": MAX_OUT,
                                                "output_config": {"effort": "high"},
                                                "messages": [{"role": "user", "content": prompt(r)}]}}
        s = len(json.dumps(req))
        if cur and size + s > 150_000_000:
            batches.append(cur)
            cur, size = [], 0
        cur.append(req)
        size += s
    batches.append(cur)
    ids = []
    for b in batches:
        mb = client.messages.batches.create(requests=b)
        ids.append(mb.id)
        print("batch", mb.id, len(b), flush=True)
    json.dump(ids, open(f"{OUT}/opus_batches.json", "w"))


def opus_collect(_):
    client = anthropic_client()
    ids = json.load(open(f"{OUT}/opus_batches.json"))
    while True:
        st = [client.messages.batches.retrieve(i) for i in ids]
        print(time.strftime("%H:%M"), [(b.processing_status, b.request_counts.succeeded, b.request_counts.errored)
                                       for b in st], flush=True)
        if all(b.processing_status == "ended" for b in st):
            break
        time.sleep(300)
    with open(f"{OUT}/opus.jsonl", "w") as f:
        for i in ids:
            for res in client.messages.batches.results(i):
                rec = {"id": res.custom_id, "type": res.result.type}
                if res.result.type == "succeeded":
                    m = res.result.message
                    rec.update(text="".join(b.text for b in m.content if b.type == "text"), stop=m.stop_reason,
                               input_tokens=m.usage.input_tokens, output_tokens=m.usage.output_tokens)
                else:
                    rec["error"] = str(getattr(res.result, "error", ""))[:500]
                f.write(json.dumps(rec) + "\n")


GATEWAYS = {"openrouter": ("https://openrouter.ai/api/v1", "~/.openrouter_key"),
            "commonstack": ("https://api.commonstack.ai/v1", "~/.cskey")}


def run(args):
    """One model through an OpenAI-compatible gateway (OpenRouter: provider pinned, no fallbacks), resumable; stops
    before a length whose estimated cost would pass the $ cap. Cost = the gateway's billed cost when it reports one."""
    import openai
    base, key = GATEWAYS[args.gateway]
    client = openai.OpenAI(base_url=base, api_key=open(os.path.expanduser(key)).read().strip(), max_retries=3,
                           timeout=1800)
    _, rows = load()
    path = f"{OUT}/{args.name}.jsonl"
    done = {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}
    spent = sum(r.get("cost", 0) for r in done.values())
    todo = [r for r in rows if r["id"] not in done or done[r["id"]]["type"] == "errored"]
    if args.lengths:
        todo = [r for r in todo if r["length"] in args.lengths.split(",")]
    if args.per_task:
        todo = [r for r in todo if r["i"] < args.per_task]
    extra = {"reasoning": {"effort": "high"}, "usage": {"include": True}}
    if args.gateway == "openrouter":
        extra["provider"] = {"order": [args.provider], "allow_fallbacks": False}

    def one(r):
        try:
            c = client.chat.completions.create(model=args.model, max_tokens=MAX_OUT, extra_body=extra,
                                               messages=[{"role": "user", "content": prompt(r)}])
            u = c.usage
            if u is None:  # OpenRouter returns some provider errors as a 200 body with "error"
                err = str(getattr(c, "error", None) or c.model_extra.get("error", ""))
                kind = "exceeds_context" if re.search(r"context", err, re.I) else "errored"
                return {"id": r["id"], "type": kind, "error": err[:500]}
            cost = getattr(u, "cost", None)
            if cost is None:  # list price, long-context step for OpenAI models
                pin, pout = args.price
                cost = u.prompt_tokens * pin + u.completion_tokens * pout
            return {"id": r["id"], "type": "succeeded", "text": c.choices[0].message.content or "",
                    "stop": c.choices[0].finish_reason, "input_tokens": u.prompt_tokens,
                    "output_tokens": u.completion_tokens, "cost": cost,
                    "provider": getattr(c, "provider", None)}
        except openai.BadRequestError as e:
            kind = "exceeds_context" if re.search(r"context|too long|maximum|too many tokens", str(e), re.I) else "errored"
            return {"id": r["id"], "type": kind, "error": str(e)[:500]}
        except Exception as e:  # noqa: BLE001 - recorded, retried on the next run
            return {"id": r["id"], "type": "errored", "error": str(e)[:500]}

    with open(path, "a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        for length in LENGTHS:
            batch = [r for r in todo if r["length"] == length]
            if not batch:
                continue
            est = sum(len(prompt(r)) / 3.5 for r in batch) * args.price[0] * (args.long_mult if length in ("512k", "1M") else 1)
            if spent + est > args.cap:
                print(f"stop before {length}: spent ${spent:.2f} + est ${est:.2f} > cap ${args.cap}", flush=True)
                break
            for rec in ex.map(one, batch):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                spent += rec.get("cost", 0) or 0
            kinds = collections.Counter(r["type"] for r in map(json.loads, open(path)) if r["id"].split("-")[1] == length)
            print(f"{length}: {dict(kinds)} spent ${spent:.2f}", flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (max(0.0, c - h), min(1.0, c + h))


def judge(g, r, labels, parse="full"):
    """One fixed reading (primary = "full": the official BABILong rule on the whole output, as on the leaderboard).
    API errors and truncated outputs are wrong; "exceeds_context" is reported separately (None)."""
    if g is None:
        return "missing"
    if g["type"] == "errored":
        return False
    if g["type"] == "exceeds_context":
        return None
    if g.get("stop") in ("length", "max_tokens"):
        return False
    t = g["text"].strip()
    lines = [x.strip().strip("*") for x in t.split("\n") if x.strip()]
    if not lines:
        return False
    if parse == "official":  # second scorer: the official babilong code, verbatim, on the raw output
        import babilong_metrics_official as bo
        return bo.compare_answers(r["target"], g["text"], r["question"], bo.TASK_LABELS[r["task"]])
    cand = {"full": t.replace("*", ""), "first": lines[0], "last": lines[-1]}[parse]
    return correct_official(cand, r["target"], r["question"], labels.get(r["task"]))


def score(args):
    labels, rows = load()
    got = {}
    for r in map(json.loads, open(args.file)):
        if r["id"] in got and r["type"] == "errored" and got[r["id"]]["type"] != "errored":
            continue  # a failed retry never replaces an answer
        if r["id"] in got and r["type"] == "errored" and got[r["id"]]["type"] == "exceeds_context":
            continue
        got[r["id"]] = r
    print(f"{args.file}  parse={args.parse}  (errors and truncations count as wrong)")
    print("length  " + "  ".join(f"{t:>5}" for t in ["qa1", "qa2", "qa3", "qa4", "qa5"])
          + "  QA1-5 [95% CI]      n  exceeds-ctx  errors  missing")
    for length in LENGTHS:
        accs, K, N, ctx, err, miss = [], 0, 0, 0, 0, 0
        for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
            k = n = 0
            for r in rows:
                if r["task"] != t or r["length"] != length:
                    continue
                g = got.get(r["id"])
                miss += g is None
                err += g is not None and g["type"] == "errored"
                v = judge(g, r, labels, args.parse)
                if v == "missing":
                    continue
                if v is None:
                    ctx += 1
                    continue
                k += v
                n += 1
            accs.append(k / n if n else None)
            K, N = K + k, N + n
        if miss == sum(1 for r in rows if r["length"] == length):
            continue
        lo, hi = wilson(K, N)
        cells = "  ".join(f"{a:5.2f}" if a is not None else "    -" for a in accs)
        mean = f"{100 * K / N:5.1f} [{100 * lo:4.1f}-{100 * hi:5.1f}]" if N else "  exceeds context  "
        print(f"{length:6s}  {cells}  {mean}  {N:4d}  {ctx:5d}  {err:5d}  {miss:5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--parse", default="full", choices=["full", "first", "last", "official"])
    ap.add_argument("--cap", type=float, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="openai/gpt-6.1-sol")
    ap.add_argument("--provider", default="OpenAI")
    ap.add_argument("--gateway", default="openrouter")
    ap.add_argument("--name", default="gpt61")
    ap.add_argument("--price", type=float, nargs=2, default=[2e-6, 10e-6])
    ap.add_argument("--lengths", default="")
    ap.add_argument("--per_task", type=int, default=0)
    ap.add_argument("--long_mult", type=float, default=2.0)  # OpenAI: input price x2 above 272k; Anthropic: 1
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"count": count, "opus-submit": opus_submit, "opus-collect": opus_collect, "run": run, "score": score}[a.cmd](a)
