"""Frontier-model arms on OpenAI's own API (never a reseller): A4 direct, A5 with code, A3 translator.

Streamed chat completions; usage and list-price cost logged per call; a hard USD cap over the whole ledger
(runs/spend_ledger.jsonl) stops the run. Resumable: an (id, arm) with a valid record is skipped.

    python3 runs/llm_api.py --arm direct --items data/sre_dev.jsonl --out out/a4_dev.jsonl --max_calls 15
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import translate as TR  # noqa: E402
import translate_zebra as TZ  # noqa: E402
import translate_chunked as TC  # noqa: E402

BASE = "https://api.openai.com/v1"
PRICE = {"in": 2.0, "cached": 0.10, "out": 10.0}          # USD per million tokens, gpt-6.1-sol list price
LEDGER = os.path.join(HERE, "spend_ledger.jsonl")
CAP_USD = 150.0
LOCK = threading.Lock()
UA = "nsengine-client/1.0"   # a generic client name (the vexcloud gateway's Cloudflare rejects Python-urllib)
ENGINE_LOCK = threading.Lock()


def key(path):
    kv = dict(l.strip().split("=", 1) for l in open(os.path.expanduser(path)) if "=" in l)
    if not kv.get("OPENAI_API_KEY"):
        raise SystemExit("no OPENAI_API_KEY in " + path)
    return kv["OPENAI_API_KEY"]


def cost(usage):
    if not usage:
        return 0.0
    pt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    return ((pt - cached) * PRICE["in"] + cached * PRICE["cached"] + ct * PRICE["out"]) / 1e6


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l)["usd"] for l in open(LEDGER))


def chat(model, messages, max_tokens, effort):
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if effort:
        body["reasoning_effort"] = effort
    for attempt in range(4):
        try:
            t0 = time.time()
            r = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json",
                                                "User-Agent": UA})
            text, finish, usage = [], None, None
            with urllib.request.urlopen(r, timeout=1800) as f:
                for raw in f:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ch = json.loads(data)
                    if ch.get("usage"):
                        usage = ch["usage"]
                    for c in ch.get("choices") or []:
                        text.append((c.get("delta") or {}).get("content") or "")
                        finish = c.get("finish_reason") or finish
            if finish is None and usage is None:
                raise ConnectionError("stream closed early")
            return {"text": "".join(text), "finish_reason": finish, "usage": usage, "seconds": round(time.time() - t0, 1)}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": "HTTP %d: %r" % (e.code, e.read()[:200])}
        except Exception as e:
            if attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": repr(e)[:200]}


def log_spend(arm, rid, r):
    usd = cost(r.get("usage"))
    with LOCK, open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.time(), "arm": arm, "id": rid, "usd": usd}) + "\n")
    return usd


# ------------------------------------------------------------------------------------------------ prompts
def body(it):
    return "<runbook>\n%s\n</runbook>\n\n%s\n\nQuestion: %s" % (it["text"], it["rules"], it["question"])


def prompt_direct(it):
    if it.get("task") == "zebra":
        return ("Solve this logic puzzle.\n\n" + it["text"] + "\n\nWork it out carefully. End your reply with one line "
                "of the form\n" + it["format"] + "\nwith every ___ filled in.")
    return ("You are given a production runbook and a question about it.\n\n" + body(it) +
            "\n\nWork it out carefully. End your reply with one line of the form\n" + it["format"])


def prompt_code(it):
    if it.get("task") == "zebra":
        return ("Solve this logic puzzle.\n\n" + it["text"] + "\n\nWrite one Python 3 program that computes the solution. "
                "You may use the standard library and z3 (the z3-solver package, `import z3`). The program must print, as "
                "its last line, exactly one line of the form\n" + it["format"] + "\nwith every ___ filled in. Reply with the "
                "program only, in a single ```python code block.")
    return ("You are given a production runbook and a question about it.\n\n" + body(it) +
            "\n\nWrite one Python 3 program that computes the answer. Hard-code the facts you need from the runbook. "
            "You may use the standard library and z3 (the z3-solver package, `import z3`). The program must print, as its "
            "last line, exactly one line of the form\n" + it["format"] +
            "\nReply with the program only, in a single ```python code block.")


def run_sandbox(code, timeout=60):
    """Run model-written code in a throwaway container: no network, 2 CPUs, 2 GB, read-only except /tmp."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "prog.py"), "w").write(code)
        cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "2g", "--read-only",
               "--tmpfs", "/tmp", "-v", d + ":/w:ro", "nsengine-sandbox:z3", "timeout", str(timeout),
               "python3", "-I", "/w/prog.py"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
            return p.returncode, p.stdout[-4000:], p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return m[-1] if m else None


# ------------------------------------------------------------------------------------------------ arms
def do_direct(it, a):
    r = chat(a.model, [{"role": "user", "content": prompt_direct(it)}], a.max_tokens, a.effort)
    r["usd"] = log_spend("direct", it["id"], r) if "error" not in r else 0.0
    return r


def do_code(it, a):
    msgs = [{"role": "user", "content": prompt_code(it)}]
    r = chat(a.model, msgs, a.max_tokens, a.effort)
    if "error" in r:
        return r
    usd = log_spend("code", it["id"], r)
    rec = {"attempts": []}
    for attempt in range(2):
        code = extract_code(r["text"])
        rc, out, err = run_sandbox(code) if code else (-1, "", "no python code block in the reply")
        rec["attempts"].append({"code": code, "rc": rc, "stdout": out, "stderr": err, "usage": r.get("usage"),
                                "finish_reason": r.get("finish_reason"), "seconds": r.get("seconds")})
        ok = rc == 0 and "ANSWER:" in out.upper()
        if ok or attempt == 1:
            break
        msgs += [{"role": "assistant", "content": r["text"]},
                 {"role": "user", "content": "Running the program failed (exit code %s). Last output:\n%s\n%s\n"
                  "Reply with a corrected program only, in a single ```python code block." % (rc, out[-1500:], err[-1500:])}]
        r = chat(a.model, msgs, a.max_tokens, a.effort)
        if "error" in r:
            rec["error"] = r["error"]
            break
        usd += log_spend("code", it["id"], r)
    rec["text"] = rec["attempts"][-1]["stdout"]
    rec["usd"] = usd
    return rec


def do_translate(it, a):
    zebra = it.get("task") == "zebra"
    prompt = TZ.prompt_translate(it, ZSHOTS) if zebra else TR.prompt_translate(it, SHOTS)
    r = chat(a.model, [{"role": "user", "content": prompt}], a.max_tokens, a.effort)
    if "error" in r:
        return r
    r["usd"] = log_spend("translate", it["id"], r)
    dsl = TR.parse_json(r["text"])
    with ENGINE_LOCK:                                   # z3's default context is not thread-safe
        ans = TZ.answer_from(dsl) if zebra else TR.answer_from(dsl)
    return {**r, "dsl": dsl, **ans}


def do_tchunk(it, a):
    """Chunked translation (NSR6b): paragraphs in batches of 20, question separately, merged, then the engine."""
    parts, usd, secs, n_calls = [], 0.0, 0.0, 0
    for pr in TC.chunk_prompts(it, size=20, shot=CHUNK_SHOT) + [TC.query_prompt(it)]:
        r = chat(a.model, [{"role": "user", "content": pr}], a.max_tokens, a.effort)
        if "error" in r:
            return {"error": r["error"]}
        usd += log_spend("tchunk", it["id"], r)
        secs += r.get("seconds") or 0
        n_calls += 1
        parts.append(TR.parse_json(r["text"]))
    query = parts.pop()
    dsl, why = TC.merge(parts, query)
    if dsl is None:
        return {"abstain": True, "reason": why, "usd": usd, "calls": n_calls, "seconds": round(secs, 1)}
    with ENGINE_LOCK:
        ans = TR.answer_from(dsl)
    return {"dsl": dsl, **ans, "usd": usd, "calls": n_calls, "seconds": round(secs, 1)}


ARMS = {"direct": do_direct, "code": do_code, "translate": do_translate, "tchunk": do_tchunk}
CHUNK_SHOT = None
SHOTS = None
ZSHOTS = None


def valid(r):
    return "error" not in r


def main():
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--max_calls", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", default="")
    ap.add_argument("--keyfile", default="~/.openai_key_nsr")
    a = ap.parse_args()
    KEY = key(a.keyfile)
    global SHOTS, ZSHOTS, CHUNK_SHOT
    CHUNK_SHOT = TC.shot_from()
    SHOTS = TR.load_shots()
    ZSHOTS = TZ.load_shots()
    items = [json.loads(l) for l in open(a.items)]
    if a.ids:
        want = set(a.ids.split(","))
        items = [it for it in items if it["id"] in want]
    done = set()
    if os.path.exists(a.out):
        done = {r["id"] for r in map(json.loads, open(a.out)) if valid(r)}
    todo = [it for it in items if it["id"] not in done][:a.max_calls]
    print("%d calls (%d done); ledger USD %.2f of cap %.0f" % (len(todo), len(done), spent(), CAP_USD), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with cf.ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        futs = {}
        for it in todo:
            if spent() >= CAP_USD:
                print("spend cap reached: not submitting more", flush=True)
                break
            futs[ex.submit(ARMS[a.arm], it, a)] = it
        for k, fu in enumerate(cf.as_completed(futs)):
            it, r = futs[fu], fu.result()
            rec = {"id": it["id"], "arm": a.arm, "model": a.model, "effort": a.effort, "base": BASE, **r}
            ct = ((r.get("usage") or {}).get("completion_tokens") or 0)
            rec["hit_limit"] = r.get("finish_reason") == "length" or ct >= a.max_tokens
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print("%d/%d done; ledger USD %.2f" % (k + 1, len(futs), spent()), flush=True)
    print("finished; ledger USD %.2f" % spent(), flush=True)


if __name__ == "__main__":
    main()
