"""Infra probe only (no scientific reading): which mainstream multi-hop corpora load here, and
which of them ship STRUCTURED per-hop evidence that the V-65 store arm can consume (a relation
chain, not just supporting sentences). Availability probing, per V-41 precedent."""
import os, json
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset

CANDS = [
    ("dgslibisey/MuSiQue", None, "validation"),
    ("bdsaglam/musique", "answerable", "validation"),
    ("hotpotqa/hotpot_qa", "distractor", "validation"),
    ("rmanluo/RoG-webqsp", None, "test"),
    ("rmanluo/RoG-cwq", None, "test"),
]
for name, cfg, split in CANDS:
    try:
        ds = load_dataset(name, cfg, split=split, streaming=True) if cfg else \
             load_dataset(name, split=split, streaming=True)
        ex = next(iter(ds))
        keys = list(ex.keys())
        print("\n=== OK %s cfg=%s split=%s" % (name, cfg, split), flush=True)
        print("    keys: %s" % keys, flush=True)
        for k in ("question_decomposition", "decomposition", "graph", "evidences",
                  "supporting_facts", "answer", "question"):
            if k in ex:
                s = json.dumps(ex[k], ensure_ascii=False, default=str)
                print("    %-22s %s" % (k + ":", s[:420]), flush=True)
    except Exception as e:
        print("\n=== FAIL %s cfg=%s : %s %s" % (name, cfg, type(e).__name__, str(e)[:160]), flush=True)
print("\nPROBE-DONE", flush=True)
