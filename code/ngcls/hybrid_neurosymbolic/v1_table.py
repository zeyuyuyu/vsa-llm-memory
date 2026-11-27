import json, os
import numpy as np
ARMS = ["frozen_llm", "lora", "lora_qpool", "vsa_hidden", "vsa_parser"]
print("=== V-1 FINAL (SysChain, seeds 42/123/456, 2000 train / 1000 eval, 2 epochs) ===")
print("%-12s %-26s %-8s %14s" % ("arm", "per-seed", "mean", "trainable"))
rows = {}
for a in ARMS:
    p = "v1_logs/%s.json" % a
    if not os.path.exists(p):
        print("%-12s (still running)" % a); continue
    r = json.load(open(p))
    acc = np.array([x["acc"] for x in r])
    rows[a] = acc
    print("%-12s %-26s %-8.4f %14s" % (a, str(np.round(acc, 4)), acc.mean(),
                                       format(r[0]["trainable_params"], ",")))
if "vsa_hidden" in rows:
    vh = rows["vsa_hidden"]
    inc = [rows[k] for k in ("lora", "lora_qpool") if k in rows]
    if inc:
        best = np.maximum.reduce(inc)
        gap = (vh - best) * 100
        print("\nregistered bar: vsa_hidden must beat best LoRA by >= 10pp in 3/3")
        print("  gap per seed (pp): %s" % np.round(gap, 2))
        v = "SURVIVES" if np.all(gap >= 10) else ("COLLAPSES" if np.all(gap <= 5) else "INTERMEDIATE")
        print("  >>> V-1 VERDICT: %s" % v)
    if "frozen_llm" in rows:
        print("  vsa_hidden vs frozen-LLM floor (pp): %s"
              % np.round((vh - rows["frozen_llm"]) * 100, 2))
