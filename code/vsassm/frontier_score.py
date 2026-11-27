"""Score the frontier references (B19 amendment 2, B20 amendment 1) next to our systems on the same samples.

python3 frontier_score.py ruler      # GPT-6.1 Sol, Opus 5.5 and ours on the 130 RULER-128k samples
python3 frontier_score.py lme_export # frontier outputs -> LongMemEval hypothesis files (then lme_rm.py judge)
"""
import collections
import glob
import json
import sys

import ruler_rm as R


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return 100 * max(0, c - h), 100 * min(1, c + h)


def ruler():
    prompts = [json.loads(l) for l in open("frontier/ruler128k_prompts.jsonl")]
    ours = {}
    for f in glob.glob("b19_test/rm/131072_*.jsonl"):
        for r in map(json.loads, open(f)):
            ours[f"{r['task']}|{r['index']}|{json.dumps(r['outputs'])}"] = r["pred"]
    systems = {"ours": {k: {"type": "succeeded", "text": v} for k, v in ours.items()}}
    for name in ["gpt61", "opus55"]:
        rows = {}
        for r in map(json.loads, open(f"frontier/ruler_{name}.jsonl")):
            if r["id"] not in rows or r["type"] == "succeeded":
                rows[r["id"]] = r
        systems[name] = rows
    print("task".ljust(16) + "".join(n.rjust(10) for n in systems))
    tot = {n: [] for n in systems}
    for t in R.TASKS:
        cells = []
        for n, rows in systems.items():
            sc = []
            for p in prompts:
                if p["task"] != t:
                    continue
                g = rows.get(p["id"])
                if g is None:
                    continue
                ok = g["type"] == "succeeded" and g.get("stop") not in ("length",)
                fn = R.string_match_part if t.startswith("qa") else R.string_match_all
                sc.append(100 * fn(g["text"], p["outputs"]) if ok else 0.0)
            tot[n].append(sum(sc) / len(sc) if sc else float("nan"))
            cells.append(f"{sum(sc) / len(sc):10.1f}" if sc else "         -")
        print(t.ljust(16) + "".join(cells))
    print("avg13".ljust(16) + "".join(f"{sum(v) / len(v):10.1f}" for v in tot.values()))
    drop = [i for i, t in enumerate(R.TASKS) if t not in ("cwe", "fwe")]
    print("w/o cwe,fwe".ljust(16) + "".join(f"{sum(v[i] for i in drop) / len(drop):10.1f}" for v in tot.values()))
    for n, rows in systems.items():
        if n != "ours":
            print(n, collections.Counter(r["type"] for r in rows.values()),
                  f"cost ${sum(r.get('cost', 0) or 0 for r in rows.values()):.2f}")


def lme_export():
    for name in ["gpt61", "opus55"]:
        rows = {}
        for r in map(json.loads, open(f"frontier/lme_{name}.jsonl")):
            if r["id"] not in rows or r["type"] == "succeeded":
                rows[r["id"]] = r
        with open(f"lme/out/{name}_sub100.jsonl", "w") as f:
            for q, r in rows.items():
                hyp = r["text"] if r["type"] == "succeeded" else f"[API {r['type']}]"
                f.write(json.dumps({"question_id": q, "hypothesis": hyp}) + "\n")
        print(name, collections.Counter(r["type"] for r in rows.values()),
              f"cost ${sum(r.get('cost', 0) or 0 for r in rows.values()):.2f}")


if __name__ == "__main__":
    {"ruler": ruler, "lme_export": lme_export}[sys.argv[1]]()
