"""Bank the per-chain-length counts of the CLUTRR run (Fig. 4b, "ours") from the per-story file.

    python3 figdata_v2/r1_logs/bank_ours_by_k.py [/path/to/vsassm/r1_out/test_8B_think.jsonl]

Reads the per-story output of the frozen Qwen3-8B reader + rule memory (one JSON object per line, field `ok`),
groups the stories by chain length k = len(gold_path) and writes ours_by_k.json next to this file. Nothing is typed:
the JSON is the count table, and make_fig6_benchmarks.py divides correct by n.
"""
import collections, hashlib, json, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/path/to/vsassm/r1_out/test_8B_think.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ours_by_k.json")

raw = open(SRC, "rb").read()
rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
by = collections.OrderedDict()
for r in rows:
    k = len(r["gold_path"])
    c = by.setdefault(k, {"n": 0, "correct": 0})
    c["n"] += 1
    c["correct"] += int(bool(r["ok"]))
by = {str(k): by[k] for k in sorted(by)}
out = {
    "source": "vsassm/r1_out/test_8B_think.jsonl (CLUTRR gen_train23_test2to10, test split; frozen Qwen3-8B reader, "
              "reasoning on, + rule memory); field `ok`, grouped by len(gold_path)",
    "source_md5": hashlib.md5(raw).hexdigest(),
    "n_rows": len(rows),
    "correct": sum(c["correct"] for c in by.values()),
    "by_k": by,
}
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
for k, c in by.items():
    print("  k=%-2s n=%-4d correct=%-4d acc=%.4f" % (k, c["n"], c["correct"], c["correct"] / c["n"]))
print("  all  n=%d correct=%d acc=%.4f" % (out["n_rows"], out["correct"], out["correct"] / out["n_rows"]))
