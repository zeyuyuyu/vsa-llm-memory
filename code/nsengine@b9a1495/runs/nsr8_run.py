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

CAP_STOP = 148.0          # stop launching new calls above this ledger total (the shared cap is 150)
KIND = {"check": "find", "find": "find", "unique": "unique", "optimize": "optimize"}


def run_E(q):
    text, keys, kind = q["text"], q["answer_keys"], KIND[q["kind"]]
    G1.task = lambda it: (it["text"], it["answer_keys"], KIND[it["kind"]])  # the generic adapter, for formalise()
    forms = [G1.formalise(q, j, "gpt-6.1-sol", "medium") for j in range(2)]
    codes = [f["code"] for f in forms if f["code"]]
    rec = {"id": q["id"], "arm": "E", "forms": forms}
    if codes:
        r = gm.run(codes[0], {"op": "solve"}, trusted=False, timeout=300)
        rec["P"] = ({"impossible": True} if r.get("status") == "unsat" else r.get("answer")
                    if r.get("status") == "sat" and (kind != "optimize" or r.get("optimal")) else None)
        rec["P_status"] = r.get("status") or r.get("error")
    if len(codes) == 2:
        n = len(GT.number_sentences(text)[1])
        v, ans, cert = gcert.certify(codes, kind, trusted=False, timeout=300, n_sentences=n, strict_coverage=False)
        rec["E"] = ans if v == "certified" else None
        rec["why"] = cert.get("why")
    return rec


def direct_prompt(q):
    return (q["text"] + "\n\nGive your final answer as one JSON object on the last line, with these slots: " +
            q["answer_keys"] + ' If no answer exists, reply {"impossible": true} instead.')


def run_D(q):
    r = L.chat("gpt-6.1-sol", [{"role": "user", "content": direct_prompt(q)}], 32000, "high")
    usd = L.log_spend("nsr8_D", q["id"], r) if "error" not in r else 0.0
    ans = None
    for m in reversed(re.findall(r"\{.*\}", r.get("text") or "", re.S)):
        try:
            ans = json.loads(m)
            break
        except json.JSONDecodeError:
            continue
    return {"id": q["id"], "arm": "D", "D": ans, "usd": usd, "finish": r.get("finish_reason"), "error": r.get("error"),
            "usage": r.get("usage")}


def score(items_path):
    import families as FAM
    fams = FAM.all_families()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path)}
    out = {}
    for arm, path in (("E", "out/nsr8_E.jsonl"), ("D", "out/nsr8_D.jsonl")):
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
    json.dump(out, open("out/nsr8_score.json", "w"), indent=1)
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
    a = ap.parse_args()
    if a.score:
        score(a.items)
        return
    L.KEY = L.key(a.keyfile)
    qs = [json.loads(l) for l in open(a.questions)]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    qs = [q for q in qs if q["id"] not in done]
    fn = run_E if a.arm == "E" else run_D
    print("%d to do; ledger %.2f" % (len(qs), L.spent()), flush=True)

    def guarded(q):
        if L.spent() >= CAP_STOP:
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
