"""R2 arm B: the direct fine-tune answers "How is B related to A?" with no memory (prereg/R2_prereg.md).

python3 r2_direct.py --model <Qwen3-1.7B> --adapter r2_models/direct_s0 --data r1_data --split test --out r2_out/direct_test.jsonl
"""
import argparse
import collections
import csv
import json
import os

from r1 import RELS
from r2_data import DIRECT

BY_LEN = sorted(RELS, key=len, reverse=True)  # "father-in-law" is matched before "father"


def first_rel(text):
    t = text.lower()
    hits = [(t.find(r), -len(r), r) for r in BY_LEN if r in t]
    return min(hits)[2] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(f"{args.data}/{args.split}.csv")))
    import ast
    import b14
    reader = b14.Reader(args.model, args.adapter)
    reader.model.generation_config.max_new_tokens = 16
    rels = ", ".join(RELS)
    prompts = []
    for r in rows:
        a, b = ast.literal_eval(r["query"])
        prompts.append(DIRECT.format(rels=rels, story=r["story"].replace("[", "").replace("]", ""), a=a, b=b))
    outs = reader(prompts, bs=64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    by_k = collections.defaultdict(lambda: [0, 0])
    with open(args.out, "w") as f:
        for r, (txt, n) in zip(rows, outs):
            k = len(r["f_comb"].split("-"))
            pred = first_rel(txt)
            ok = pred == r["target_text"]
            by_k[k][0] += ok
            by_k[k][1] += 1
            f.write(json.dumps({"query": r["query"], "target": r["target_text"], "pred": pred, "raw": txt, "ok": ok,
                                "k": k}) + "\n")
    tot = [sum(v[0] for v in by_k.values()), sum(v[1] for v in by_k.values())]
    print(f"{args.out}: " + "  ".join(f"k={k}: {v[0] / v[1]:.3f} ({v[1]})" for k, v in sorted(by_k.items()))
          + f"  all {tot[0] / tot[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
