"""Claude Opus 5.5 (and other models) through the user's Anthropic-compatible endpoint (user 2026-11-15).
Credentials: ~/.zeyu_anthropic_key (mode 600; ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN lines). Never printed.
Spend is logged to runs/spend_ledger_claude.jsonl at list prices (Opus 5.5: $4 in / $20 out per MTok).
No server-side fallbacks on purpose: every answer in an experiment arm must come from the named model; a refusal is an
abstention, not a silent switch to another model.
"""
import json
import os
import threading
import time

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "spend_ledger_claude.jsonl")
PRICE = {"claude-opus-5-5": (4.0, 20.0)}          # USD per MTok in / out; cache reads 0.05x, cache writes 1.25x input
LOCK = threading.Lock()
_client = None


def client():
    global _client
    if _client is None:
        kv = dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.zeyu_anthropic_key")) if "=" in l)
        base = kv["ANTHROPIC_BASE_URL"]
        base = base if base.startswith("http") else "https://" + base
        _client = anthropic.Anthropic(base_url=base, auth_token=kv["ANTHROPIC_AUTH_TOKEN"], max_retries=3,
                                      timeout=1800.0)
    return _client


def spent():
    return sum(json.loads(l)["usd"] for l in open(LEDGER)) if os.path.exists(LEDGER) else 0.0


def chat(model, messages, max_tokens=32000, effort="high", tag="", rid=""):
    """One streamed request -> {"text", "stop_reason", "usage", "seconds"} or {"error"}."""
    t0 = time.time()
    try:
        kw = {"output_config": {"effort": effort}} if effort else {}
        with client().messages.stream(model=model, max_tokens=max_tokens, messages=messages, **kw) as s:
            m = s.get_final_message()
    except anthropic.BadRequestError as e:
        return {"error": "400: %s" % str(e)[:300]}
    except anthropic.RateLimitError as e:
        return {"error": "429: %s" % str(e)[:200]}
    except anthropic.APIStatusError as e:
        return {"error": "%s: %s" % (e.status_code, str(e)[:200])}
    except anthropic.APIConnectionError as e:
        return {"error": "connection: %s" % str(e)[:200]}
    text = "".join(b.text for b in m.content if b.type == "text")
    u = {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens,
         "cache_read": getattr(m.usage, "cache_read_input_tokens", 0) or 0,
         "cache_write": getattr(m.usage, "cache_creation_input_tokens", 0) or 0}
    pin, pout = PRICE.get(model, (0.0, 0.0))
    usd = ((u["input_tokens"] + 1.25 * u["cache_write"] + 0.05 * u["cache_read"]) * pin + u["output_tokens"] * pout) / 1e6
    with LOCK, open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.time(), "model": model, "tag": tag, "id": rid, "usd": usd, **u}) + "\n")
    return {"text": text, "stop_reason": m.stop_reason, "usage": u, "usd": usd, "model": m.model,
            "seconds": round(time.time() - t0, 1)}
