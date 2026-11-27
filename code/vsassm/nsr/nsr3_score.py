"""NSR3 independent scorer: replays each log's events itself, takes the union of the truth states over the window,
and scores every arm's answer file (one JSON per line: {"id": "log|qtype|target|i|j", "pred": [...]}). Imports only
nsr1_score.site_facts (catalog constants), not the question generator or the engines.

python3 nsr/nsr3_score.py --split nsr2 --logs nsr/data --arms history final sor llm_qwen3-32b_forced ...
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nsr1_score import site_facts  # noqa: E402


def states(log):
    st, out = {}, []
    for line in log["lines"]:
        for ev in line.get("events") or []:
            if ev[0] == "move":
                st[ev[1]] = ev[2]
            elif ev[0] == "swap":
                a, b = ev[1], ev[2]
                moved = {o: (b if p == a else a) for o, p in st.items() if p in (a, b)}
                st.update(moved)
        out.append(dict(st))
    return out


def affected(log, st, qtype, target):
    part, needs = site_facts(log)

    def reaches(x):
        seen = set()
        while x not in seen:
            seen.add(x)
            x = st[x] if x in st else part.get(x)
            if x is None:
                return False
            if x == target:
                return True
        return False
    if qtype == "host_down":
        return {s for s, d in needs.items() if st.get(d) == target}
    keep = {"freezer_down": lambda x: x.startswith("S-"), "region_down": lambda x: not x.endswith("primary")}.get(
        qtype, lambda x: True)
    return {x for x in st if keep(x) and reaches(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="nsr2")
    ap.add_argument("--logs", default=os.path.join(HERE, "data"))
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    logs = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for line in open(os.path.join(a.logs, f"{sc}_{a.split}.jsonl")):
            d = json.loads(line)
            logs[d["id"]] = d
    S = {lid: states(log) for lid, log in logs.items()}
    out, keys = {}, None
    for arm in a.arms:
        recs = [json.loads(l) for l in open(os.path.join(HERE, "out", f"nsr3_{a.split}_{arm}.jsonl"))]
        k = sorted(r["id"] for r in recs)
        assert keys is None or k == keys, f"{arm}: different question set"
        keys = k
        per = collections.defaultdict(list)
        for r in recs:
            lid, qt, tg, i, j = r["id"].split("|")
            i, j = int(i), int(j)
            g = set()
            for st in S[lid][max(i - 1, 0):j + 1]:
                g |= affected(logs[lid], st, qt, tg)
            ok = r["pred"] is not None and set(r["pred"]) == g
            per[logs[lid]["scenario"]].append(ok)
            per["ALL"].append(ok)
        out[arm] = {s: round(sum(v) / len(v), 4) for s, v in per.items()}
    out["n_questions"] = len(keys)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
