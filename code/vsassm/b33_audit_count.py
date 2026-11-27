"""B33 audit (box, CPU): rows per split/task of Muennighoff/babi, story boundaries of the task-2 test split, overlap of
test rows 500-999 with rows 0-499 and with train. Output: b33_data/audit_count_output.txt."""
import os, json, collections
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset
d = load_dataset("Muennighoff/babi")
for sp in d:
    c = collections.Counter(r["task"] for r in d[sp])
    print(sp, len(d[sp]), "task2:", c.get(2), "per-task:", dict(sorted(c.items())))
print(d["test"].features)
te = [r for r in d["test"] if r["task"] == 2]
print("task-2 test rows", len(te))
# story boundaries: a new story starts when the passage is not an extension of the previous one
starts = []
for i, r in enumerate(te):
    if i == 0 or not r["passage"].strip().startswith(te[i-1]["passage"].strip()[:40]):
        starts.append(i)
print("n stories (heuristic)", len(starts), "first starts", starts[:12])
print("starts near 500:", [s for s in starts if 480 <= s <= 520])
for i in (497, 498, 499, 500, 501):
    print("----", i, repr(te[i]["passage"][:200]), "|", te[i]["question"], "|", te[i]["answer"])
tr = [r for r in d["train"] if r["task"] == 2]
va = [r for r in d["validation"] if r["task"] == 2]
print("train", len(tr), "val", len(va))
# exact duplicates of passages+question across splits
key = lambda r: (r["passage"].strip(), r["question"].strip())
s_tr, s_te500, s_rest = set(map(key, tr)), set(map(key, te[:500])), set(map(key, te[500:]))
print("dup rest vs first500", len(s_rest & s_te500), "rest vs train", len(s_rest & s_tr))
