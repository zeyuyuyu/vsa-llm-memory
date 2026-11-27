"""R2 (prereg/R2_prereg.md): training data for a fine-tuned CLUTRR reader and for the direct-answer control.

Reader targets come from CLUTRR's own annotations, never from the test split: story_edges (a, b) with edge type r mean
"node b is the r of node a"; genders lists the nodes in order, so node i's name is genders[i]. Every edge is written in
both directions, in r1.PROMPT's format ("A is the relation of B"), so r1.py reads the adapter's output unchanged.

python3 r2_data.py r1_data r2_data
"""
import ast
import csv
import json
import os
import random
import sys

from r1 import PROMPT, RELS

INV = {"father": ("son", "daughter"), "mother": ("son", "daughter"), "son": ("father", "mother"),
       "daughter": ("father", "mother"), "husband": ("husband", "wife"), "wife": ("husband", "wife"),
       "brother": ("brother", "sister"), "sister": ("brother", "sister"),
       "grandfather": ("grandson", "granddaughter"), "grandmother": ("grandson", "granddaughter"),
       "grandson": ("grandfather", "grandmother"), "granddaughter": ("grandfather", "grandmother"),
       "uncle": ("nephew", "niece"), "aunt": ("nephew", "niece"), "nephew": ("uncle", "aunt"),
       "niece": ("uncle", "aunt"), "father-in-law": ("son-in-law", "daughter-in-law"),
       "mother-in-law": ("son-in-law", "daughter-in-law"), "son-in-law": ("father-in-law", "mother-in-law"),
       "daughter-in-law": ("father-in-law", "mother-in-law")}
DIRECT = "Read the story and answer with one word from this list: {rels}.\n\nStory: {story}\n\nHow is {b} related to {a}? {b} is {a}'s"


def gold_facts(r):
    nodes = [g.split(":") for g in r["genders"].split(",")]
    names, sex = [n for n, _ in nodes], {n: s for n, s in nodes}
    facts = []
    for (a, b), rel in zip(ast.literal_eval(r["story_edges"]), ast.literal_eval(r["edge_types"])):
        A, B = names[a], names[b]
        facts.append({"who": B, "is_the": rel, "of": A})
        facts.append({"who": A, "is_the": INV[rel][0 if sex[A] == "male" else 1], "of": B})
    return facts


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(f"{src}/train.csv")))
    random.Random(0).shuffle(rows)
    rels = ", ".join(RELS)
    with open(f"{out}/reader.jsonl", "w") as fr, open(f"{out}/direct.jsonl", "w") as fd:
        for r in rows:
            story = r["story"].replace("[", "").replace("]", "")
            facts = gold_facts(r)
            fr.write(json.dumps({"prompt": PROMPT.format(story=story, rels=rels),
                                 "output": json.dumps({"facts": facts})}) + "\n")
            a, b = ast.literal_eval(r["query"])
            fd.write(json.dumps({"prompt": DIRECT.format(rels=rels, story=story, a=a, b=b),
                                 "output": r["target_text"]}) + "\n")
    print(len(rows), "training stories ->", out)


if __name__ == "__main__":
    main()
