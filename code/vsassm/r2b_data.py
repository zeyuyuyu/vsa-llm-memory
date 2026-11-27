"""R2b (prereg/R2b_prereg.md): R2's training data plus long stories built from the TRAIN split only.

Each long story concatenates 2-4 training stories whose name sets are disjoint, in random order; the reader target is
the union of their gold facts (both directions, as R2), the direct target is one component story's question and answer.

python3 r2b_data.py r1_data r2b_data
"""
import ast
import csv
import json
import os
import random
import sys

from r1 import PROMPT, RELS
from r2_data import DIRECT, gold_facts


def names(r):
    return {g.split(":")[0] for g in r["genders"].split(",")}


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(f"{src}/train.csv")))
    rng = random.Random(0)
    rels = ", ".join(RELS)
    clean = lambda s: s.replace("[", "").replace("]", "")
    reader, direct = [], []
    for r in rows:  # R2's examples, unchanged
        a, b = ast.literal_eval(r["query"])
        reader.append({"prompt": PROMPT.format(story=clean(r["story"]), rels=rels),
                       "output": json.dumps({"facts": gold_facts(r)})})
        direct.append({"prompt": DIRECT.format(rels=rels, story=clean(r["story"]), a=a, b=b), "output": r["target_text"]})
    made = 0
    while made < len(rows):
        m = rng.choice([2, 3, 4])
        group, used = [], set()
        for r in rng.sample(rows, 40):
            if not (names(r) & used):
                group.append(r)
                used |= names(r)
            if len(group) == m:
                break
        if len(group) < m:
            continue
        story = " ".join(clean(r["story"]).strip() for r in group)
        facts = [f for r in group for f in gold_facts(r)]
        reader.append({"prompt": PROMPT.format(story=story, rels=rels), "output": json.dumps({"facts": facts})})
        q = rng.choice(group)
        a, b = ast.literal_eval(q["query"])
        direct.append({"prompt": DIRECT.format(rels=rels, story=story, a=a, b=b), "output": q["target_text"]})
        made += 1
    for name, data in (("reader", reader), ("direct", direct)):
        rng.shuffle(data)
        with open(f"{out}/{name}.jsonl", "w") as f:
            for e in data:
                f.write(json.dumps(e) + "\n")
    print(len(rows), "original +", made, "long stories ->", out)


if __name__ == "__main__":
    main()
