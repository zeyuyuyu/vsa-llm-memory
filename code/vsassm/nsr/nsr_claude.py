"""Second frontier arm: Claude (Anthropic API, official SDK), same prompt text as the gpt-6.1-sol arm (nsr_api.py).

The prompt is sent as two text blocks, split just before "\n\nQuestion:", so the shared log prefix of a site can be
cached (cache_control on the first block). The model sees the same characters as the other arms. Within a site the first
question runs alone (it writes the cache) and the rest run concurrently (they read it).

Spend control: no balance endpoint, so the cost of every call is computed from its usage at list prices (conservative
cache-read price) and appended to a ledger (nsr/out/api/claude_ledger.jsonl) shared by all runs. A call is not started
if ledger total + a worst-case estimate for it would exceed --budget. Resumable: skips (n, idx) with a valid record.
Key read from ~/.anthropic_key_nsr (ANTHROPIC_API_KEY=...). No refusal fallback: a declined call is recorded as a
refusal (no answer), so that one model answers the whole arm.

python3 nsr/nsr_claude.py --problems P --out F [--model claude-fable-5-1] [--effort xhigh] [--budget 180]
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import threading
import time

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "out", "api", "claude_ledger.jsonl")
# USD per million tokens: input, output, cache write (5 min), cache read. Cache read priced at 0.1 x input
# (conservative; the published Fable 5.1 cache-read price is lower).
PRICE = {"claude-fable-5-1": (10.0, 50.0, 12.5, 1.0), "claude-opus-5-5": (4.0, 20.0, 5.0, 0.4)}
LOCK = threading.Lock()


def key():
    for line in open(os.path.expanduser("~/.anthropic_key_nsr")):
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def cost(model, u):
    pi, po, pw, pr = PRICE[model]
    return (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po
            + u.get("cache_creation_input_tokens", 0) * pw + u.get("cache_read_input_tokens", 0) * pr) / 1e6


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l)["usd"] for l in open(LEDGER))


def split(prompt):
    i = prompt.rfind("\n\nQuestion:")
    return (prompt[:i], prompt[i:]) if i > 0 else (prompt, "")


def call(client, model, prompt, max_tokens, effort):
    head, tail = split(prompt)
    content = [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}}]
    if tail:
        content.append({"type": "text", "text": tail})
    t0 = time.time()
    for attempt in range(4):
        try:
            with client.messages.stream(model=model, max_tokens=max_tokens, output_config={"effort": effort},
                                        messages=[{"role": "user", "content": content}]) as s:
                m = s.get_final_message()
            text = "\n".join(b.text for b in m.content if b.type == "text")  # keep block breaks (amendment 2)
            u = m.usage.to_dict() if hasattr(m.usage, "to_dict") else dict(m.usage)
            finish = {"end_turn": "stop", "max_tokens": "length"}.get(m.stop_reason, m.stop_reason)
            return {"answer_text": text, "finish_reason": finish, "usage": u, "seconds": round(time.time() - t0, 1),
                    "reasoning_effort": effort, "request_id": getattr(m, "_request_id", None)}
        except anthropic.BadRequestError as e:  # not retryable
            return {"error": f"HTTP 400: {str(e.message)[:200]}"}
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            if attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": repr(e)[:200]}
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": f"HTTP {e.status_code}: {str(e.message)[:200]}", "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # stream cut mid-body (httpx.RemoteProtocolError etc.): billed but no usage returned
            return {"error": repr(e)[:200], "seconds": round(time.time() - t0, 1)}


def valid(r):
    return "error" not in r and r.get("finish_reason") is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=64000)
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--sites", type=int, default=2)  # sites in parallel
    ap.add_argument("--inner", type=int, default=6)  # concurrent questions within a site after the first
    ap.add_argument("--max_calls", type=int, default=10 ** 6)
    ap.add_argument("--stagger", type=float, default=60.0)  # seconds between a site's first call and the rest
    a = ap.parse_args()
    client = anthropic.Anthropic(api_key=key(), max_retries=2, timeout=3600)
    probs = [json.loads(l) for l in open(a.problems)]
    done = set()
    if os.path.exists(a.out):
        done = {(r["n"], r["idx"]) for r in map(json.loads, open(a.out)) if valid(r)}
    todo = [p for p in probs if (p["n"], p["idx"]) not in done][:a.max_calls]
    by0 = collections.OrderedDict()
    for p in todo:
        by0.setdefault(p["n"], []).append(p)
    # sites round-robin over scenarios, so that a budget stop leaves a balanced sample
    scen = collections.OrderedDict()
    for n, ps_ in by0.items():
        scen.setdefault(ps_[0]["id"].split("-")[0], []).append((n, ps_))
    by = collections.OrderedDict()
    while any(scen.values()):
        for k in list(scen):
            if scen[k]:
                n, ps_ = scen[k].pop(0)
                by[n] = ps_
    pi, po, pw, _ = PRICE[a.model]
    worst = lambda p: (len(p["prompt"]) / 3 * pw + a.max_tokens * po) / 1e6  # noqa: E731
    print(f"{len(todo)} calls in {len(by)} sites ({len(done)} done); ledger {spent():.2f} USD of {a.budget}", flush=True)
    f = open(a.out, "a")
    stop = threading.Event()
    n_done = [0]
    reserved = [0.0]

    def one(p):
        if stop.is_set():
            return
        w = worst(p)
        with LOCK:  # calls in flight are counted at their worst case
            if spent() + reserved[0] + w > a.budget:
                stop.set()
                print(f"budget: next call could pass {a.budget} USD, stopping", flush=True)
                return
            reserved[0] += w
        try:
            r = call(client, a.model, p["prompt"], a.max_tokens, a.effort)
        finally:
            with LOCK:
                reserved[0] -= w
        rec = {"model": a.model, "n": p["n"], "idx": p["idx"], "id": p["id"], "sample": False, "no_think": False, **r}
        rec["hit_limit"] = r.get("finish_reason") == "length"
        if r.get("usage"):
            usd = cost(a.model, r["usage"])
        else:  # failed call: book a conservative estimate (prompt written to cache + ~110 output tokens/s while it ran)
            usd = (len(p["prompt"]) / 3 * pw + min(a.max_tokens, 110 * r.get("seconds", 0)) * po) / 1e6
            rec["usd_estimated"] = True
        rec["usd"] = round(usd, 4)
        with LOCK:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            with open(LEDGER, "a") as lg:
                lg.write(json.dumps({"t": time.time(), "model": a.model, "out": os.path.basename(a.out),
                                     "id": p["id"], "usd": round(usd, 5)}) + "\n")
            n_done[0] += 1
            if n_done[0] % 10 == 0:
                print(f"{n_done[0]}/{len(todo)} done; ledger {spent():.2f} USD", flush=True)

    def guarded(p):
        try:
            one(p)
        except Exception as e:  # keep the other threads going
            print(f"call failed outside the API wrapper: {p['id']}: {e!r}"[:300], flush=True)

    def site(ps):
        # the 5-minute cache TTL runs from the last use, and one call thinks for 5-10 min, so the others must start
        # while the first is still running: start it, wait until its prefill has written the cache, then start the rest
        with cf.ThreadPoolExecutor(1 + a.inner) as ex:
            futs = [ex.submit(guarded, ps[0])]
            if len(ps) > 1:
                time.sleep(a.stagger)
                futs += [ex.submit(guarded, p) for p in ps[1:]]
            for fu in futs:
                fu.result()

    with cf.ThreadPoolExecutor(a.sites) as ex:
        list(ex.map(site, by.values()))
    f.close()
    print(f"finished {n_done[0]} calls; ledger {spent():.2f} USD", flush=True)


if __name__ == "__main__":
    main()
