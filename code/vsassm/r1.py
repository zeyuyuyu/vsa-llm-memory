"""R1 (prereg/R1_prereg.md): Readable Reasoner on CLUTRR = a reader (prompted Qwen3) that writes kinship facts, a
fact graph, and the VSA rule memory (rules learned from the training split only) that folds the path between the two
people of the question.

python3 r1.py --model <Qwen3> --data DIR --split validation --n 200 --out r1_out/val_8B.jsonl
"""
import argparse
import ast
import collections
import csv
import json
import os
import re

from r1_oracle import RuleMemory, compose

RELS = ["father", "mother", "son", "daughter", "husband", "wife", "brother", "sister", "grandfather", "grandmother",
        "grandson", "granddaughter", "uncle", "aunt", "nephew", "niece", "father-in-law", "mother-in-law",
        "son-in-law", "daughter-in-law"]

PROMPT = """Read the story and list every family relation it states between two people.

Story: {story}

Write each relation as {{"who": A, "is_the": relation, "of": B}}, read as "A is the relation of B", using only these
relations: {rels}.
Give every relation in both directions (for "Ann is the mother of Bob": {{"who": "Ann", "is_the": "mother", "of": "Bob"}}
and {{"who": "Bob", "is_the": "son", "of": "Ann"}}); use the people's names and work out genders from names and pronouns.
Only write a relation when the story names that relation for that exact pair. Never work out a relation yourself from
other relations.

Examples
Story: Joe and his brother, Lewis, went to visit Joe's grandfather, Jason.
{{"facts": [{{"who": "Lewis", "is_the": "brother", "of": "Joe"}}, {{"who": "Joe", "is_the": "brother", "of": "Lewis"}},
{{"who": "Jason", "is_the": "grandfather", "of": "Joe"}}, {{"who": "Joe", "is_the": "grandson", "of": "Jason"}}]}}
(No relation between Lewis and Jason: the story does not state it.)

Story: Michelle called her mother Dorothy. Dorothy was yelling at her father Jason.
{{"facts": [{{"who": "Dorothy", "is_the": "mother", "of": "Michelle"}}, {{"who": "Michelle", "is_the": "daughter", "of": "Dorothy"}},
{{"who": "Jason", "is_the": "father", "of": "Dorothy"}}, {{"who": "Dorothy", "is_the": "daughter", "of": "Jason"}}]}}
("her father" in the second sentence is Dorothy's father.)

Return only JSON: {{"facts": [...]}}"""

FACT = re.compile(r'\{\s*"who"\s*:\s*"([^"]+)"\s*,\s*"is_the"\s*:\s*"([^"]+)"\s*,\s*"of"\s*:\s*"([^"]+)"\s*\}')


def learn_rules(train_rows):
    votes = collections.defaultdict(collections.Counter)
    for r in train_rows:
        if len(r["rels"]) == 2:
            votes[tuple(r["rels"])][r["target"]] += 1
    rules = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    for _ in range(3):
        new = collections.defaultdict(collections.Counter)
        for r in train_rows:
            if len(r["rels"]) == 3:
                a, b, c = r["rels"]
                if (a, b) in rules and (rules[(a, b)], c) not in rules:
                    new[(rules[(a, b)], c)][r["target"]] += 1
                if (b, c) in rules and (a, rules[(b, c)]) not in rules:
                    new[(a, rules[(b, c)])][r["target"]] += 1
        rules.update({k: c.most_common(1)[0][0] for k, c in new.items()})
    return rules


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append({"story": r["story"].replace("[", "").replace("]", ""), "query": ast.literal_eval(r["query"]),
                    "target": r["target_text"], "rels": r["f_comb"].split("-"), "k": len(r["f_comb"].split("-"))})
    return out


def all_paths(graph, src, dst, max_len=12, limit=200):
    out, stack = [], [(src, [], {src})]
    while stack and len(out) < limit:
        u, rels, seen = stack.pop()
        for v, rel in graph[u]:
            if v == dst:
                out.append(rels + [rel])
            elif v not in seen and len(rels) + 1 < max_len:
                stack.append((v, rels + [rel], seen | {v}))
    return out


def answer_vote(facts, query, rules, mem):
    """Every simple path from query[0] to query[1] (up to a limit) is folded in the VSA; the most common answer wins
    (ties: the shortest path's answer). One wrong fact then rarely decides the answer."""
    graph = collections.defaultdict(list)
    for a, rel, b in set(facts):
        graph[a].append((b, rel))
    votes, first = collections.Counter(), {}
    for path in sorted(all_paths(graph, *query), key=len):
        res = compose(path, lambda a, b: (a, b) in rules, lambda a, b: mem.read(a, b)[0])
        if isinstance(res, str):
            votes[res] += 1
            first.setdefault(res, path)
    if not votes:
        return None, []
    top = max(votes.values())
    best = min((r for r, c in votes.items() if c == top), key=lambda r: len(first[r]))
    return best, first[best]


def answer(facts, query, rules, mem):
    """Shortest path from query[0] to query[1] over the fact graph; fold its relations (any bracketing) in the VSA."""
    graph = collections.defaultdict(list)
    for a, rel, b in facts:
        graph[a].append((b, rel))
    src, dst = query
    prev, frontier = {src: None}, [src]
    while frontier and dst not in prev:
        nxt = []
        for u in frontier:
            for v, rel in graph[u]:
                if v not in prev:
                    prev[v] = (u, rel)
                    nxt.append(v)
        frontier = nxt
    if dst not in prev:
        return None, []
    path, cur = [], dst
    while prev[cur]:
        u, rel = prev[cur]
        path.append(rel)
        cur = u
    path.reverse()
    res = compose(path, lambda a, b: (a, b) in rules, lambda a, b: mem.read(a, b)[0])
    return (res if isinstance(res, str) else None), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--vote", action="store_true")  # dev note 3: vote over all simple paths
    ap.add_argument("--think", action="store_true")  # dev note 4: reader in thinking mode
    args = ap.parse_args()
    train = load(f"{args.data}/train.csv")
    rules = learn_rules(train)
    mem = RuleMemory()
    for (a, b), c in rules.items():
        mem.write(a, b, c)
    rows = load(f"{args.data}/{args.split}.csv")[:args.n]
    import b14
    reader = b14.Reader(args.model, args.adapter or None)
    reader.model.generation_config.max_new_tokens = 600 if not args.think else 6000
    if args.think:  # dev note 4: the reader may think once per story (cost reported)
        reader.think = True
    outs = reader([PROMPT.format(story=r["story"], rels=", ".join(RELS)) for r in rows], bs=32 if not args.think else 16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    by_k = collections.defaultdict(lambda: [0, 0])
    with open(args.out, "w") as f:
        for r, (txt, n) in zip(rows, outs):
            # reader writes "A is B's relation"; the graph edge is (B, relation, A): A is B's relation (CLUTRR order)
            final = txt[txt.rfind('"facts"'):] if '"facts"' in txt else txt  # thinking drafts are not facts
            facts = [(b.strip(), rel.strip().lower(), a.strip()) for a, rel, b in FACT.findall(final)]
            pred, path = (answer_vote if args.vote else answer)(facts, r["query"], rules, mem)
            ok = pred == r["target"]
            by_k[r["k"]][0] += ok
            by_k[r["k"]][1] += 1
            f.write(json.dumps({"story": r["story"], "query": r["query"], "target": r["target"], "pred": pred,
                                "path": path, "gold_path": r["rels"], "facts": facts, "ok": ok, "tokens": n}) + "\n")
    tot = [sum(v[0] for v in by_k.values()), sum(v[1] for v in by_k.values())]
    print(f"{args.out}: " + "  ".join(f"k={k}: {v[0] / v[1]:.3f} ({v[1]})" for k, v in sorted(by_k.items()))
          + f"  all {tot[0] / tot[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
