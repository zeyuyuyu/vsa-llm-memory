"""Frontier arm through the user-supplied Responses-API gateway client (~/Downloads/api_call_1.py, given 2026-11-12).

The client holds the gateway key and base URLs, so this wrapper only shells out to it (`call --json --allow-incomplete`,
prompt on stdin). The key never enters this repo. Records use the same format as nsr_api.py (answer_text,
finish_reason, usage with prompt/completion_tokens, hit_limit), so final()/valid() and the parsers work unchanged.

python3 nsr/nsr_gw.py --models                      # list what the gateway exposes
python3 nsr/nsr_gw.py --model M --effort xhigh --problems P --out F --ids a,b --max_tokens 64000 [--base_url U]
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time

CLIENT = os.path.expanduser("~/Downloads/api_call_1.py")


def client(args, stdin=None, base_url="", timeout=1800):
    cmd = ["python3", "-I", CLIENT, "--timeout", str(timeout)] + (["--base-url", base_url] if base_url else []) + args
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout + 120)
    return p.returncode, p.stdout, p.stderr.strip()


def text_of(resp):
    if isinstance(resp.get("output_text"), str) and resp["output_text"]:
        return resp["output_text"]
    out = []
    for item in resp.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            out += [c["text"] for c in item.get("content") or [] if c.get("type") == "output_text"]
    return "\n".join(out)


def call(model, prompt, max_tokens, effort, base_url):
    t0 = time.time()
    for attempt in range(4):
        args = ["call", "--prompt-file", "-", "--model", model, "--max-output-tokens", str(max_tokens), "--json",
                "--allow-incomplete"] + (["--reasoning-effort", effort] if effort else [])
        try:
            rc, out, err = client(args, stdin=prompt, base_url=base_url)
        except subprocess.TimeoutExpired:
            rc, out, err = 1, "", "client timeout"
        if rc == 0:
            resp = json.loads(out)
            u = resp.get("usage") or {}
            st = resp.get("status")
            why = (resp.get("incomplete_details") or {}).get("reason")
            finish = "stop" if st == "completed" else ("length" if why == "max_output_tokens" else st)
            usage = {"prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens"),
                     "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"), "raw": u}
            return {"answer_text": text_of(resp), "finish_reason": finish, "usage": usage, "status": st,
                    "seconds": round(time.time() - t0, 1), "reasoning_effort": effort, "response_id": resp.get("id")}
        retryable = any(s in err for s in ("HTTP 429", "HTTP 50", "HTTP 408", "Unable to reach", "timed out",
                                           "client timeout"))
        if retryable and attempt < 3:
            time.sleep(30 * (attempt + 1))
            continue
        return {"error": err[-300:], "seconds": round(time.time() - t0, 1)}


def valid(r):
    return "error" not in r and r.get("finish_reason") is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--base_url", default="")
    ap.add_argument("--model")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--problems")
    ap.add_argument("--out")
    ap.add_argument("--ids", default="")  # "a:b" slice or comma list of problem ids
    ap.add_argument("--max_calls", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.models:
        rc, out, err = client(["models"], base_url=a.base_url, timeout=60)
        if rc:
            raise SystemExit(err)
        for m in json.loads(out)["data"]:
            print(m["id"], m.get("api"), m.get("reasoning_efforts"))
        return
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
    print(f"{len(todo)} calls ({len(done)} done) on {a.model} effort {a.effort}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        futs = {ex.submit(call, a.model, p["prompt"], a.max_tokens, a.effort, a.base_url): p for p in todo}
        for k, fu in enumerate(cf.as_completed(futs)):
            p, r = futs[fu], fu.result()
            rec = {"model": a.model, "gateway": "api_call_1", "n": p["n"], "idx": p["idx"], "id": p["id"],
                   "sample": False, "no_think": False, **r}
            ct = (r.get("usage") or {}).get("completion_tokens") or 0
            rec["hit_limit"] = r.get("finish_reason") == "length" or (
                valid(r) and not (r.get("answer_text") or "").strip() and ct >= a.max_tokens)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{k + 1}/{len(todo)} {p['id']}: {r.get('finish_reason') or r.get('error', '')[:80]} "
                  f"{r.get('seconds')} s", flush=True)


if __name__ == "__main__":
    main()
