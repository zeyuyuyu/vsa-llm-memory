"""G1 pilot (development, burned items): the generic translator prompt with a frontier model, K formalisations per
item, then the general harness and the certification layer. Logs spend to the shared ledger (cap in llm_api).
    python3 runs/g1_pilot.py data/sre_nsr7.jsonl --n 10 --k 2 --out out/g1_pilot_sre.jsonl
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "bench")]
import llm_api as L  # noqa: E402
import gtranslate as GT  # noqa: E402
import gcert  # noqa: E402
import gm  # noqa: E402
import sre_score as S  # noqa: E402
import zebra_score as Z  # noqa: E402
from g0_cert_selftest import to_pred  # noqa: E402


def task(it):
    return GT.zebra_task(it) if it.get("task") == "zebra" else GT.sre_task(it)


def judge(it, verdict, ans):
    """'abstain' / 'correct' / 'WRONG' for an engine answer, any family."""
    if verdict != "certified" or ans is None:
        return "abstain"
    if it.get("task") == "zebra":
        if ans.get("impossible"):
            return "WRONG"
        ok, n = Z.cells(it, GT.zebra_grid(it, ans))
        return "correct" if ok == n else "WRONG"
    p = to_pred(it, verdict, ans)
    return "abstain" if p is None else ("correct" if S.correct(it, p) else "WRONG")


REPAIR = ("Running your code failed with this error (the harness appends the solve step itself):\n%s\n"
          "Reply with the corrected complete code, in one ```python block.")


def formalise(it, j, model, effort, repairs=1):
    """One formalisation; a program that crashes (not one that is merely slow or unsatisfiable) is sent back with its
    error, at most `repairs` times. The repair sees only the error text, never an answer."""
    text, keys, kind = task(it)
    msgs = [{"role": "user", "content": GT.prompt(text, keys)}]
    usd, rep, code, r = 0.0, 0, None, {}
    for attempt in range(repairs + 1):
        r = L.chat(model, msgs, 32000, effort)
        usd += L.log_spend("g1_pilot", "%s#%d.%d" % (it["id"], j, attempt), r) if "error" not in r else 0.0
        code = GT.extract_code(r.get("text"))
        if not code or attempt == repairs:
            break
        t = gm.run(code, {"op": "solve"}, trusted=False, timeout=300)
        if "error" not in t or t.get("error") == "timeout":
            break
        msgs += [{"role": "assistant", "content": r.get("text") or ""},
                 {"role": "user", "content": REPAIR % (t.get("stderr") or t.get("error"))[-1500:]}]
        rep += 1
    return {"j": j, "code": code, "usd": usd, "finish": r.get("finish_reason"), "error": r.get("error"),
            "seconds": r.get("seconds"), "repairs": rep}


def one(it, k, model, effort, prev=None):
    text, keys, kind = task(it)
    forms = prev["forms"] if prev else [formalise(it, j, model, effort) for j in range(k)]
    codes = [f["code"] for f in forms if f["code"]]
    rec = {"id": it["id"], "kind": kind, "forms": forms}
    if codes:                                            # the primary alone (frontier + code, no checks)
        r = gm.run(codes[0], {"op": "solve"}, trusted=False, timeout=300)
        ans, v = None, "abstain"
        if r.get("status") == "sat":
            ans, v = r["answer"], "certified"
            if kind == "all_opt":
                e = gm.run(codes[0], {"op": "enum", "at_optimum": True, "cap": 200}, trusted=False, timeout=300)
                ans = {"optimum": e.get("objective"), "answers": e.get("answers", [])}
        elif r.get("status") == "unsat":
            ans, v = {"impossible": True}, "certified"
        rec["primary"] = judge(it, v, ans)
        rec["primary_status"] = r.get("status") or r.get("error")
    if len(codes) == k:
        n = len(GT.number_sentences(text)[1])
        for name, kw in (("certified", {}), ("cert_cov", {"n_sentences": n, "strict_coverage": False}),
                         ("cert_cov_strict", {"n_sentences": n, "strict_coverage": True})):
            v, ans, cert = gcert.certify(codes, kind, trusted=False, timeout=300, **kw)
            rec[name] = judge(it, v, ans)
            rec[name + "_why"] = cert.get("why")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keyfile", default="~/.openai_key_nsr")
    ap.add_argument("--ids", default="", help="file with one item id per line (instead of --n random items)")
    ap.add_argument("--rescore", default="", help="re-run the checks on the formalisations saved in this file")
    a = ap.parse_args()
    L.KEY = L.key(a.keyfile)
    items = [json.loads(l) for f in a.items.split(",") for l in open(f)]   # one or more item files, comma-separated
    if a.ids:
        want = [l.strip() for l in open(a.ids) if l.strip()]
        byid = {it["id"]: it for it in items}
        items = [byid[i] for i in want]
    else:
        random.Random(a.seed).shuffle(items)
        items = items[:a.n]
    prev = {json.loads(l)["id"]: json.loads(l) for l in open(a.rescore)} if a.rescore else {}
    print("ledger before: %.2f" % L.spent(), flush=True)
    with ThreadPoolExecutor(5) as ex, open(a.out, "a") as f:
        for rec in ex.map(lambda it: one(it, a.k, a.model, a.effort, prev.get(it["id"])), items):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["id"], rec.get("primary"), rec.get("certified"), rec.get("cert_cov"), rec.get("cert_cov_strict"),
                  rec.get("cert_cov_strict_why"), "$%.3f" % sum(x["usd"] for x in rec["forms"]), flush=True)
    print("ledger after: %.2f" % L.spent())


if __name__ == "__main__":
    main()
