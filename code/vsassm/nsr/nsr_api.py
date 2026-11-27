"""Frontier-model arm through the Sidrune gateway (OpenAI-compatible, https://sidrune.ai/v1). Same prompts as the
open-model arms; one chat call per problem, default reasoning; answer text parsed later by nsr1_llm.resolve.

Spend control: the wallet balance (GET /v1/usage) is read before and after; --max_calls caps the calls; the run stops if
the balance falls below --floor. Resumable (skips (n, idx) already in the output file). Key read from ~/.sidrune_key.

python3 nsr/nsr_api.py --model gpt-5.6-terra --problems P --out F --ids 0:10 --max_calls 10
OpenAI direct (no balance endpoint: --max_calls is the only cap; usage is logged per call):
python3 nsr/nsr_api.py --base https://api.openai.com/v1 --keyfile ~/.openai_key_nsr --model gpt-6.1-sol ...
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://sidrune.ai/v1"


def key(path="~/.sidrune_key"):
    kv = dict(l.strip().split("=", 1) for l in open(os.path.expanduser(path)) if "=" in l)
    for name in ("SIDRUNE_API_KEY", "OPENAI_API_KEY"):  # ~/.sidrune_key also holds an OpenRouter key: never take that
        if kv.get(name):
            return kv[name]
    raise SystemExit("no key")


def req(path, body=None, timeout=900):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def balance():
    """Sidrune wallet; other gateways have no balance endpoint (NaN: the floor checks then never fire)."""
    return float(req("/usage", timeout=60)["balance"]) if "sidrune.ai" in BASE else float("nan")


def call(model, prompt, max_tokens, effort=None):
    """One streamed chat call (streaming keeps the connection alive through long reasoning; without it Cloudflare cuts
    the request at ~100 s with HTTP 524). Collects the answer text, finish_reason and the final usage chunk."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    if effort:
        body["reasoning_effort"] = effort
    for attempt in range(4):
        try:
            t0 = time.time()
            r = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            text, finish, usage, done = [], None, None, False
            with urllib.request.urlopen(r, timeout=1800) as f:
                for raw in f:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    ch = json.loads(data)
                    if ch.get("usage"):
                        usage = ch["usage"]
                    for c in ch.get("choices") or []:
                        text.append((c.get("delta") or {}).get("content") or "")
                        finish = c.get("finish_reason") or finish
            if finish is None and usage is None:
                # the gateway sometimes closes the stream early with no error, no finish_reason and no usage (seen
                # 11-11: 1/230 NSR1 calls, 8/210 NSR3, 4/18 NSR5); that is a transport failure, not a model answer
                raise ConnectionError(f"stream closed early (done={done}, {len(''.join(text))} chars)")
            return {"answer_text": "".join(text), "finish_reason": finish, "usage": usage,
                    "seconds": round(time.time() - t0, 1), "reasoning_effort": effort}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 524) and attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": f"HTTP {e.code}: {e.read()[:200]!r}"}
        except Exception as e:  # timeouts / connection resets
            if attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": repr(e)[:200]}


def valid(r):
    """A record that holds a model answer (possibly empty): no error and the stream finished."""
    return "error" not in r and not (r.get("finish_reason") is None and r.get("usage") is None)


def final(path):
    """The answer of record per (n, idx): the last valid record (re-runs of dropped streams append to the file)."""
    out = {}
    for r in map(json.loads, open(path)):
        if valid(r):
            out[(r["n"], r["idx"])] = r
    return list(out.values())


def main():
    global KEY, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")  # "a:b" slice of the problem list, or comma list of problem ids
    ap.add_argument("--max_calls", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--floor", type=float, default=10.0)
    ap.add_argument("--effort", default="")  # reasoning_effort (e.g. xhigh); empty = gateway default
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--keyfile", default="~/.sidrune_key")
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    KEY = key(a.keyfile)
    probs = [json.loads(l) for l in open(a.problems)]
    if ":" in a.ids:
        s, e = (int(x) for x in a.ids.split(":"))
        probs = probs[s:e]
    elif a.ids:
        want = set(a.ids.split(","))
        probs = [p for p in probs if p["id"] in want]
    done = set()
    if os.path.exists(a.out):
        done = {(r["n"], r["idx"]) for r in map(json.loads, open(a.out)) if valid(r)}
    todo = [p for p in probs if (p["n"], p["idx"]) not in done][:a.max_calls]
    b0 = balance()
    print(f"{len(todo)} calls ({len(done)} done); balance before {b0:.4f}", flush=True)
    if b0 < a.floor:
        raise SystemExit("balance below floor")
    with cf.ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        futs = {ex.submit(call, a.model, p["prompt"], a.max_tokens, a.effort or None): p for p in todo}
        for k, fu in enumerate(cf.as_completed(futs)):
            p, r = futs[fu], fu.result()
            rec = {"model": a.model, "base": BASE, "n": p["n"], "idx": p["idx"], "id": p["id"], "sample": False, "no_think": False, **r}
            # the gateway reports "stop" even when reasoning used the whole budget and no answer came out
            ct = ((r.get("usage") or {}).get("completion_tokens") or 0)
            rec["hit_limit"] = r.get("finish_reason") == "length" or (
                not (r.get("answer_text") or "").strip() and ct >= a.max_tokens)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                bal = balance()
                print(f"{k + 1}/{len(todo)} done; balance {bal:.4f}", flush=True)
                if bal < a.floor:
                    print("balance below floor: stopping", flush=True)
                    ex.shutdown(cancel_futures=True)
                    break
    b1 = balance()
    print(f"balance after {b1:.4f}; spent {b0 - b1:.4f} USD on {len(todo)} calls", flush=True)


if __name__ == "__main__":
    main()
