"""B30a descriptive analysis (referee item 3, registered without a bar): can B18's codebooks (256 codes each) have
overflowed? The reader's own events were not logged per sample below 10M; what is banked is the number of events
the memory applied, per sample at 10M (b18_10M/*.partial.jsonl) and summed over each task's samples at 0k-1M (the
PACE summary.json files of the B18 test run, copied to b30a_out/pace_b18_test_summaries.txt on 2026-11-18).

Each applied event writes one key and one value, so a story's codebooks hold at most (names of its own facts) +
(events applied beyond its own facts). The story's own facts and names come from the floor's parse (F1 rows). If
the reader applied every fact of each story, the events beyond the story's own facts in any one sample are at most
the task's total minus the stories' facts, which gives the bound printed here.

python3 b30a_names.py
"""
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "b30a_out", "b30a_rows.jsonl"))]
f1 = collections.defaultdict(list)
for r in rows:
    if r["arm"] == "F1" and r["in_b18"]:
        f1[(r["length"], r["task"])].append(r)
tot = {}
for line in open(os.path.join(HERE, "b30a_out", "pace_b18_test_summaries.txt")):
    m = re.match(r"(\S+) (\{.*\})\s*$", line)
    if not m:
        continue
    for k, v in json.loads(m.group(2)).items():
        tot[tuple(k.split("/"))] = v
print("length task   n  reader events  story facts (F1)  beyond story  story names max  bound on codes per sample")
worst = 0
for length in ["0k", "4k", "16k", "64k", "128k", "512k", "1M"]:
    for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
        v, rr = tot.get((length, t)), f1.get((length, t), [])
        if not v or not rr:
            continue
        assert v["n"] == len(rr), (length, t, v["n"], len(rr))
        story = sum(r["applied"] for r in rr)
        extra = max(0, v["events"] - story)
        names = max(max(r["n_keys"], r["n_values"]) for r in rr)
        bound = names + extra + 1  # +1: the "unknown" value code
        worst = max(worst, bound)
        print(f"{length:6s} {t:4s} {v['n']:4d}  {v['events']:13d}  {story:16d}  {extra:12d}  {names:15d}  {bound:6d}"
              + ("  (> 256)" if bound > 256 else ""))
print("10M, per sample (b18_10M):")
for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
    b = [json.loads(l) for l in open(os.path.join(HERE, "b18_10M", f"10M_{t}.partial.jsonl"))]
    rr = f1[("10M", t)]
    for i, x in enumerate(b):
        g = rr[i]
        bound = max(g["n_keys"], g["n_values"]) + max(0, x["events"] - g["applied"]) + 1
        worst = max(worst, bound)
        print(f"10M    {t} sample {i}: reader events {x['events']:3d}, story facts {g['applied']:3d}, story names "
              f"{max(g['n_keys'], g['n_values'])}, bound {bound}" + ("  (> 256)" if bound > 256 else "")
              + ("; events alone <= 255, so under the cap without assumption" if x["events"] <= 255 else ""))
print(f"largest bound over all lengths: {worst}")
