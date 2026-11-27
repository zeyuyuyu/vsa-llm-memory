"""NSR8 runner (prereg/NSR8_prereg.md): the general engine on held-out families (bench/heldout/).

The systems see only the question view of an item (id, family, level, text, answer_keys, kind); scoring uses the full
item and the family's own check(). Arms:
  E  engine: gpt-6.1-sol (effort medium) writes K = 2 CPMpy formalisations with runs/gtranslate.py (one repair round on a
     crash), certified by engine/gcert.py with sentence coverage (missing only); otherwise abstain.
  P  frontier + code: E's first formalisation alone, solved, no checks (no extra calls).
  D  frontier direct: gpt-6.1-sol, effort high, at most 32,000 completion tokens, answers in the slot format.
    python3 runs/nsr8_run.py --arm E --questions data/nsr8_q.jsonl --out out/nsr8_E.jsonl
    python3 runs/nsr8_run.py --arm D --questions data/nsr8_q.jsonl --out out/nsr8_D.jsonl
    python3 runs/nsr8_run.py --score --items data/nsr8_items.jsonl          (writes out/nsr8_score.json)
"""
import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "bench"),
                os.path.join(HERE, "..", "bench", "heldout")]
import llm_api as L  # noqa: E402
import gtranslate as GT  # noqa: E402
import gcert  # noqa: E402
import gm  # noqa: E402
import g1_pilot as G1  # noqa: E402
import claude_api as C  # noqa: E402

MODEL = "gpt-6.1-sol"
PROVIDER = "openai"
CLAUDE_CAP = 80.0         # NSR9 amendment 1: stop launching Claude calls above this ledger total


def _claude_chat(model, messages, max_tokens, effort):
    """llm_api.chat-shaped wrapper around claude_api.chat (spend is logged by claude_api itself)."""
    r = C.chat(MODEL, messages, max_tokens=max_tokens, effort=effort, tag="nsr9a1")
    if "error" in r:
        return r
    return {"text": r["text"], "finish_reason": r["stop_reason"], "usage": r["usage"], "seconds": r["seconds"],
            "usd": r["usd"]}

CAP_STOP = 148.0          # stop launching new calls above this ledger total (OpenAI ledger cap 150; the vex ledger uses the same stop)
TIME_LIMIT = None         # solver limit per primitive (s); NSR9 sets 300 with --time-limit
PROC_TIMEOUT = 300
KIND = {"check": "find", "find": "find", "unique": "unique", "optimize": "optimize"}


def run_E(q):
    text, keys, kind = q["text"], q["answer_keys"], KIND[q["kind"]]
    G1.task = lambda it: (it["text"], it["answer_keys"], KIND[it["kind"]])  # the generic adapter, for formalise()
    forms = [G1.formalise(q, j, MODEL, "medium") for j in range(2)]
    codes = [f["code"] for f in forms if f["code"]]
    rec = {"id": q["id"], "arm": "E", "forms": forms}
    if codes:
        op = {"op": "solve", "time_limit": TIME_LIMIT} if TIME_LIMIT else {"op": "solve"}
        r = gm.run(codes[0], op, trusted=False, timeout=PROC_TIMEOUT)
        rec["P"] = ({"impossible": True} if r.get("status") == "unsat" else r.get("answer")
                    if r.get("status") == "sat" and (kind != "optimize" or r.get("optimal")) else None)
        rec["P_status"] = r.get("status") or r.get("error")
    if len(codes) == 2:
        n = len(GT.number_sentences(text)[1])
        v, ans, cert = gcert.certify(codes, kind, trusted=False, timeout=PROC_TIMEOUT, n_sentences=n, strict_coverage=False,
                                     time_limit=TIME_LIMIT)
        rec["E"] = ans if v == "certified" else None
        rec["why"] = cert.get("why")
    return rec


def direct_prompt(q):
    return (q["text"] + "\n\nGive your final answer as one JSON object on the last line, with these slots: " +
            q["answer_keys"] + ' If no answer exists, reply {"impossible": true} instead.')


D_MAX = 32000             # direct arm output limit (registered); exploratory runs may raise it with --d-max


def run_D(q):
    r = L.chat(MODEL, [{"role": "user", "content": direct_prompt(q)}], D_MAX, "high")
    usd = (r.get("usd", 0.0) if PROVIDER == "claude" else L.log_spend("nsr8_D", q["id"], r)) if "error" not in r else 0.0
    ans = None
    for m in reversed(re.findall(r"\{.*\}", r.get("text") or "", re.S)):
        try:
            ans = json.loads(m)
            break
        except json.JSONDecodeError:
            continue
    return {"id": q["id"], "arm": "D", "D": ans, "usd": usd, "finish": r.get("finish_reason"), "error": r.get("error"),
            "usage": r.get("usage")}


def score(items_path, tag="nsr8"):
    import families as FAM
    fams = FAM.all_families()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path)}
    out = {}
    for arm, path in (("E", "out/%s_E.jsonl" % tag), ("D", "out/%s_D.jsonl" % tag)):
        if not os.path.exists(path):
            continue
        recs = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
        for a in ((arm, "P") if arm == "E" else (arm,)):
            C = collections.defaultdict(collections.Counter)
            for iid, it in items.items():
                ans = (recs.get(iid) or {}).get(a)
                o = "abstain" if ans is None else ("correct" if fams[it["family"]].check(it, ans) else "wrong")
                for key in ("all", it["family"], "L%d" % it["level"]):
                    C[key][o] += 1
            out[a] = {k: dict(v) for k, v in C.items()}
    json.dump(out, open("out/%s_score.json" % tag, "w"), indent=1)
    for a, v in out.items():
        print(a, v["all"], {k: v[k] for k in sorted(v) if k.startswith("L")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["E", "D"])
    ap.add_argument("--questions")
    ap.add_argument("--out")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--items")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--keyfile", default="~/.openai_key_nsr")
    ap.add_argument("--tag", default="nsr8")
    ap.add_argument("--time-limit", type=float, default=0)
    ap.add_argument("--d-max", type=int, default=32000)
    ap.add_argument("--provider", default="openai", choices=["openai", "vex", "claude"],
                    help="vex = api.vexcloud.ai gateway (user-given 11-15 after OpenAI spend was stopped); own ledger")
    a = ap.parse_args()
    if a.provider == "vex":                              # same model name and request body, other gateway, own ledger
        L.BASE = "https://api.vexcloud.ai/v1"
        L.LEDGER = os.path.join(HERE, "spend_ledger_vex.jsonl")
        a.keyfile = "~/.vexcloud_key"
    global TIME_LIMIT, PROC_TIMEOUT, D_MAX
    D_MAX = a.d_max
    if a.time_limit:
        TIME_LIMIT, PROC_TIMEOUT = a.time_limit, int(2 * a.time_limit + 120)
    if a.score:
        score(a.items, a.tag)
        return
    global MODEL, PROVIDER
    PROVIDER = a.provider
    if a.provider == "openai":
        raise SystemExit("OpenAI calls are stopped (user 11-15); use --provider vex or claude")
    if a.provider == "claude":                           # the user's endpoint; Claude Opus 5.5 (NSR9 amendment 1)
        MODEL = "claude-opus-5-5"
        L.chat = _claude_chat                            # used by run_D and by g1_pilot.formalise
        L.log_spend = lambda arm, rid, r: r.get("usd", 0.0)
        L.spent = C.spent
    else:
        L.KEY = open(os.path.expanduser(a.keyfile)).read().strip()
    qs = [json.loads(l) for l in open(a.questions)]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    qs = [q for q in qs if q["id"] not in done]
    fn = run_E if a.arm == "E" else run_D
    print("%d to do; ledger %.2f" % (len(qs), L.spent()), flush=True)

    def guarded(q):
        if L.spent() >= (CLAUDE_CAP if PROVIDER == "claude" else CAP_STOP):
            return None
        return fn(q)
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for rec in ex.map(guarded, qs):
            if rec is None:
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["id"], "E" if a.arm == "E" else "D", "ok" if rec.get(a.arm) is not None else "none",
                  rec.get("why") or rec.get("finish") or "", flush=True)
    print("ledger %.2f" % L.spent())


if __name__ == "__main__":
    main()
