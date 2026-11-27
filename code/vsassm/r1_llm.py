"""R1 references: Qwen3 answering CLUTRR questions directly (thinking), via b6_qwen.py.
python3 r1_llm.py prep DATA SPLIT N OUT.jsonl      -> problems (task R1)
python3 r1_llm.py score DATA SPLIT N GEN.jsonl     -> accuracy by k (answer = last relation word named in the answer)
"""
import json
import re
import sys

from r1 import RELS, load


def prep(data, split, n, out):
    with open(out, "w") as f:
        for i, r in enumerate(load(f"{data}/{split}.csv")[:n]):
            a, b = r["query"]
            p = (f"Story: {r['story']}\n\nHow is {b} related to {a}? {b} is {a}'s ... Answer with exactly one of: "
                 f"{', '.join(RELS)}.")
            f.write(json.dumps({"task": "R1", "n": r["k"], "idx": i, "prompt": p}) + "\n")


def score(data, split, n, gen):
    rows = load(f"{data}/{split}.csv")[:n]
    got = {r["idx"]: r for r in map(json.loads, open(gen))}
    by_k = {}
    pat = re.compile(r"\b(" + "|".join(sorted(RELS, key=len, reverse=True)) + r")\b")
    missing = sum(1 for i in range(len(rows)) if i not in got)
    if missing:
        print(f"WARNING {gen}: {missing} of {len(rows)} questions have no answer (run incomplete); scored on answered only")
    for i, r in enumerate(rows):
        g = got.get(i)
        if g is None:
            continue
        found = pat.findall(g["answer_text"].lower()) if g and not g["hit_limit"] else []
        ok = bool(found) and found[-1] == r["target"]
        b = by_k.setdefault(r["k"], [0, 0, 0])
        b[0] += ok
        b[1] += 1
        b[2] += g["think_tokens"] if g else 0
    tot = [sum(v[0] for v in by_k.values()), sum(v[1] for v in by_k.values())]
    print(f"{gen}: " + "  ".join(f"k={k}: {v[0] / v[1]:.3f} ({v[1]}, {v[2] / v[1]:.0f} tok)" for k, v in sorted(by_k.items()))
          + f"  all {tot[0] / tot[1]:.3f}")


if __name__ == "__main__":
    a = sys.argv
    prep(a[2], a[3], int(a[4]), a[5]) if a[1] == "prep" else score(a[2], a[3], int(a[4]), a[5])
