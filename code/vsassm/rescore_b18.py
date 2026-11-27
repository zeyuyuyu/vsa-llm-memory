"""Re-score saved B18 predictions with the official metric (after the qa11-qa13 label fix)."""
import glob
import json
import sys

import b18

for d in glob.glob(f"{sys.argv[1]}/*/"):
    sp = d + "summary.json"
    try:
        summ = json.load(open(sp))
    except FileNotFoundError:
        continue
    for k in list(summ):
        length, task = k.split("/")
        rows = [json.loads(l) for l in open(f"{d}{length}_{task}.jsonl")]
        ok = [b18.correct_official(r["pred"], r["target"], r["question"], task) for r in rows]
        summ[k]["acc"] = round(sum(ok) / len(ok), 3)
    json.dump(summ, open(sp, "w"), indent=1)
print("rescored")
