"""B33 audit (box, CPU): export the official task-2 TRAINING rows (900 train + 100 validation) used to build and check
the generator. Output: b33_data/official_task2_trainval.jsonl."""
import os, json
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset
d = load_dataset("Muennighoff/babi")
with open("tools/official_task2_trainval.jsonl", "w") as f:
    for sp in ("train", "validation"):
        for i, r in enumerate(x for x in d[sp] if x["task"] == 2):
            f.write(json.dumps({"split": sp, "i": i, "passage": r["passage"], "question": r["question"], "answer": r["answer"]}) + "\n")
print("ok")
