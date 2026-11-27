"""B18 references: Qwen3 answering BABILong questions directly from the full context (thinking), via b6_qwen.py.
python3 b18_llm.py prep SPLIT LENGTHS TASKS N OUT.jsonl
python3 b18_llm.py score SPLIT LENGTHS TASKS N GEN.jsonl
"""
import json
import sys

import pandas as pd

import b18


def rows(split, lengths, tasks, n):
    out = []
    for length in lengths.split(","):
        for task in tasks.split(","):
            if split == "test100":
                d = pd.DataFrame(json.load(open(f"{b18.ROOT['test_long']}/data/{task}/{length}.json"))[:n])
            else:
                import glob
                d = pd.read_parquet(glob.glob(f"{b18.ROOT[split]}/{length}/{task}-*.parquet")[0]).head(n)
            for i, r in d.iterrows():
                out.append({"length": length, "task": task, "i": i, "input": r["input"], "question": r["question"],
                            "target": r["target"]})
    return out


def prep(split, lengths, tasks, n, out):
    with open(out, "w") as f:
        for j, r in enumerate(rows(split, lengths, tasks, n)):
            p = (f"{r['input']}\n\nAnswer the question about the people and objects in the text above. "
                 f"Question: {r['question']}\nGive only the answer: a single word (a place, a name, an object, a "
                 f"number word, yes / no / maybe), or a comma-separated list for 'what is ... carrying' (or 'nothing').")
            f.write(json.dumps({"task": f"B18_{lengths}", "n": j, "idx": 0, "prompt": p}) + "\n")


def score(split, lengths, tasks, n, gen):
    rs = rows(split, lengths, tasks, n)
    got = {r["n"]: r for r in map(json.loads, open(gen))}
    acc = {}
    for j, r in enumerate(rs):
        g = got.get(j)
        pred = g["answer_text"].strip().strip(".").lower() if g and not g["hit_limit"] else None
        ok = pred is not None and b18.correct_official(pred.split("\n")[-1].strip(), r["target"], r["question"], r["task"])
        a = acc.setdefault((r["length"], r["task"]), [0, 0, 0])
        a[0] += ok
        a[1] += 1
        a[2] += g["think_tokens"] if g else 0
    for (length, task), (k, m, t) in sorted(acc.items()):
        print(f"{length} {task}: {k / m:.3f} ({m}), think tokens {t / m:.0f}")


if __name__ == "__main__":
    a = sys.argv
    (prep if a[1] == "prep" else score)(a[2], a[3], a[4], int(a[5]), a[6])
