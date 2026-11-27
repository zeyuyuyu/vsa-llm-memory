"""Generate held-out items as JSONL.

    python3 make.py --seeds 1000:1100 --out items.jsonl
    python3 make.py --seeds 1000:1100 --families vans,project --levels 4,5 \
        --out items.jsonl --questions questions.jsonl

--seeds A:B generates seeds A..B-1, i.e. N = B-A items per (family, level).
The --out file holds full items (text, answer_keys, kind, gold, optimum, plus the
structured "data" and "seed" that check() needs).  Only "id", "family", "level",
"text", "answer_keys" and "kind" are meant for the system under test; the
optional --questions file contains just those fields.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import FAMILY_NAMES, LEVELS, load  # noqa: E402

PUBLIC = ("id", "family", "level", "text", "answer_keys", "kind")


def _one(args):
    fam, level, seed = args
    t0 = time.time()
    item = load(fam).generate(seed, level)
    return item, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="A:B, generates seeds A..B-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default=None,
                    help="optional JSONL with only the fields shown to the system")
    ap.add_argument("--families", default=",".join(FAMILY_NAMES))
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(":"))
    fams = [f for f in a.families.split(",") if f]
    for f in fams:
        if f not in FAMILY_NAMES:
            sys.exit(f"unknown family {f}")
    levels = [int(x) for x in a.levels.split(",")]
    jobs = [(f, L, s) for f in fams for L in levels for s in range(lo, hi)]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        results = pool.map(_one, jobs, chunksize=1)
    with open(a.out, "w") as fo:
        for item, _ in results:
            fo.write(json.dumps(item) + "\n")
    if a.questions:
        with open(a.questions, "w") as fq:
            for item, _ in results:
                fq.write(json.dumps({k: item[k] for k in PUBLIC}) + "\n")
    slow = max(results, key=lambda r: r[1])
    print(f"wrote {len(results)} items to {a.out} in {time.time() - t0:.1f}s "
          f"(slowest item {slow[0]['id']}: {slow[1]:.1f}s)")


if __name__ == "__main__":
    main()
