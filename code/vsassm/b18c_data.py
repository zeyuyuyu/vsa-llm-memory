"""B18c (prereg/B18c_prereg.md): relabel the bAbI-original qa11-13 training sentences with the story's own coreference
rule (pronouns = subjects of the previous sentence) instead of the Qwen3-8B teacher; everything else as B18b.

python3 b18c_data.py b18_data/train_b18b.jsonl b18_data/qa111213_v2.jsonl b18_data/train_b18c.jsonl
"""
import json
import re
import sys

from b18c_coref import resolve

MOVE = re.compile(r"\bto the ([a-z]+)\.?$")


def rule_label(prompt):
    ctx = prompt.split("Context: ")[1].split("\nSentence: ")[0]
    sent = prompt.split("Sentence: ")[1].split("\nEvents:")[0]
    sents = ([x for x in re.split(r"(?<=\.)\s+", ctx) if x.strip()] if ctx != "(none)" else []) + [sent]
    m = MOVE.search(sent.lower().strip())
    subj = resolve(sents)[-1]
    if not m or not subj:
        return None
    return json.dumps({"events": [["go", p, m.group(1)] for p in subj]})


def main():
    base, babi, out = sys.argv[1:4]
    targets = {json.loads(l)["prompt"] for l in open(babi)}
    n = changed = kept = 0
    with open(out, "w") as f:
        for l in open(base):
            r = json.loads(l)
            if r["prompt"] in targets:
                new = rule_label(r["prompt"])
                if new is None:
                    kept += 1
                else:
                    changed += new != r["output"]
                    r["output"] = new
            n += 1
            f.write(json.dumps(r) + "\n")
    print(f"{n} examples; qa11-13 prompts relabelled, {changed} outputs changed, {kept} left as the teacher wrote them")


if __name__ == "__main__":
    main()
