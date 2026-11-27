"""Second, independently written implementation of NSR0's table arm T (prereg/NSR0_prereg.md, verification 2).

Shares no code with nsr0.py. Learns a vote table from a list of training stories (k = 2 votes, then three closure
rounds over k = 3 stories against the table as it stood at the start of the round), and answers each test chain by
memoised recursion over intervals (every split point). Compares its per-story answers with the T records in
nsr/out/nsr0_preds.jsonl.gz.

python3 nsr/nsr0_table2.py
"""
import csv
import functools
import gzip
import json


def majority(counter):
    best = max(counter.values())
    return sorted(v for v, n in counter.items() if n == best)[0]


def fit(stories):
    votes = {}

    def vote(pair, value):
        votes.setdefault(pair, {})
        votes[pair][value] = votes[pair].get(value, 0) + 1

    for st in stories:
        r = st["rels"]
        if len(r) == 2:
            vote((r[0], r[1]), st["target"])
    for _round in range(3):
        frozen = {pair: majority(c) for pair, c in votes.items()}
        for st in stories:
            r = st["rels"]
            if len(r) != 3:
                continue
            if (r[0], r[1]) in frozen and (frozen[(r[0], r[1])], r[2]) not in frozen:
                vote((frozen[(r[0], r[1])], r[2]), st["target"])
            if (r[1], r[2]) in frozen and (r[0], frozen[(r[1], r[2])]) not in frozen:
                vote((r[0], frozen[(r[1], r[2])]), st["target"])
    return {pair: majority(c) for pair, c in votes.items()}


def answer(chain, rules):
    chain = tuple(chain)

    @functools.lru_cache(maxsize=None)
    def span(lo, hi):  # all relations derivable for chain[lo:hi]
        if hi - lo == 1:
            return frozenset([chain[lo]])
        got = set()
        for cut in range(lo + 1, hi):
            for left in span(lo, cut):
                for right in span(cut, hi):
                    if (left, right) in rules:
                        got.add(rules[(left, right)])
        return frozenset(got)

    res = span(0, len(chain))
    if not res:
        return "UNKNOWN"
    return next(iter(res)) if len(res) == 1 else "AMBIG"


def main():
    train = [{"rels": r["f_comb"].split("-"), "target": r["target_text"]}
             for r in csv.DictReader(open("r1_data/train.csv", encoding="utf-8-sig"))]
    test = {r["id"]: r["f_comb"].split("-") for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}
    sets = {("S_all", 0): train}
    for line in open("nsr/out/nsr0_trainsets.jsonl"):
        d = json.loads(line)
        sets[(d["cell"], d["seed"])] = d["stories"]
    recs = {(d["cell"], d["seed"]): d for d in map(json.loads, gzip.open("nsr/out/nsr0_preds.jsonl.gz", "rt"))
            if d["arm"] == "T"}
    for key, stories in sets.items():
        rules = fit(stories)
        rec = recs[key]
        mine = [answer(test[i], rules) for i in rec["ids"]]
        diff = sum(a != b for a, b in zip(mine, rec["pred"]))
        print(f"{key}: {len(rules)} rules; {diff} of {len(mine)} answers differ from nsr0.py's T")


if __name__ == "__main__":
    main()
