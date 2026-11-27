"""B30a: merge the per-length test outputs (one process per length, b30a_out/test_<L>/) into b30a_out/b30a_rows.jsonl
and b30a_out/b30a_summary.json, the inputs of `b30a_floor.py analyze`. Checks that every run used the committed code."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "b30a_out")
LENGTHS = ["0k", "4k", "16k", "64k", "128k", "512k", "1M", "10M"]
rows, summary, codes = [], {}, set()
for L in LENGTHS:
    d = os.path.join(OUT, f"test_{L}")
    s = json.load(open(os.path.join(d, "b30a_summary.json")))
    codes.add(s["code"])
    summary.update(s["summary"])
    rows += [json.loads(l) for l in open(os.path.join(d, "b30a_rows.jsonl"))]
assert len(codes) == 1, codes
with open(os.path.join(OUT, "b30a_rows.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
json.dump({"code": codes.pop(), "summary": summary}, open(os.path.join(OUT, "b30a_summary.json"), "w"), indent=1)
print(f"{len(rows)} rows, {len(summary)} cells")
