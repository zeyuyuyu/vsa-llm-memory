"""Frontier references (B19 amendment 2, B20 amendment 1): run prompts through OpenRouter, pinned provider, effort high.

python3 frontier_api.py PROMPTS.jsonl OUT.jsonl --model openai/gpt-6.1-sol --provider OpenAI --cap 70
PROMPTS.jsonl rows: {"id": ..., "prompt": ...}. Resumable; stops when the billed total passes the cap.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re

import openai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--spent_before", type=float, default=0.0)  # spend on this model from other files
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=4000)
    a = ap.parse_args()
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", max_retries=3, timeout=1800,
                           api_key=open(os.path.expanduser("~/.openrouter_key")).read().strip())
    rows = [json.loads(l) for l in open(a.prompts)]
    done = {}
    if os.path.exists(a.out):
        for r in map(json.loads, open(a.out)):
            if r["type"] != "errored" or r["id"] not in done:
                done[r["id"]] = r
    spent = a.spent_before + sum(r.get("cost", 0) or 0 for r in done.values())
    todo = [r for r in rows if r["id"] not in done or done[r["id"]]["type"] == "errored"]
    extra = {"reasoning": {"effort": "high"}, "usage": {"include": True},
             "provider": {"order": [a.provider], "allow_fallbacks": False}}

    def one(r):
        try:
            c = client.chat.completions.create(model=a.model, max_tokens=a.max_tokens, extra_body=extra,
                                               messages=[{"role": "user", "content": r["prompt"]}])
            if c.usage is None:
                err = str(getattr(c, "error", None) or (c.model_extra or {}).get("error", ""))
                return {"id": r["id"], "type": "exceeds_context" if re.search("context", err, re.I) else "errored",
                        "error": err[:400]}
            return {"id": r["id"], "type": "succeeded", "text": c.choices[0].message.content or "",
                    "stop": c.choices[0].finish_reason, "input_tokens": c.usage.prompt_tokens,
                    "output_tokens": c.usage.completion_tokens, "cost": getattr(c.usage, "cost", 0) or 0}
        except openai.BadRequestError as e:
            kind = "exceeds_context" if re.search(r"context|too long|maximum", str(e), re.I) else "errored"
            return {"id": r["id"], "type": kind, "error": str(e)[:400]}
        except Exception as e:  # noqa: BLE001 - recorded; retried on the next run
            return {"id": r["id"], "type": "errored", "error": str(e)[:400]}

    with open(a.out, "a") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k in range(0, len(todo), a.workers * 2):
            if spent >= a.cap:
                print(f"cap reached: ${spent:.2f}", flush=True)
                break
            for rec in ex.map(one, todo[k:k + a.workers * 2]):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                spent += rec.get("cost", 0) or 0
            print(f"{min(k + a.workers * 2, len(todo))}/{len(todo)} spent ${spent:.2f}", flush=True)


if __name__ == "__main__":
    main()
