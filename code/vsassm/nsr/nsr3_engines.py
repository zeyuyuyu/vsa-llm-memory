"""NSR3 engine arms on one split: history engine, final state only, system-of-record history, truth sanity.
Writes nsr/out/nsr3_<split>_<arm>.jsonl.

python3 nsr/nsr3_engines.py --split nsr2
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_catalog as C  # noqa: E402
import nsr2 as M  # noqa: E402
import nsr3_questions as Q3  # noqa: E402

M.SPLITS.setdefault("nsr2", ("nsr/data", "nsr/out/held/reader_nsr2_verify_v4.jsonl"))

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="nsr2")
a = ap.parse_args()
logs, rd = M.load(a.split)
outs = {k: [] for k in ("history", "final", "sor", "sanity")}
for lid, log in sorted(logs.items()):
    cat = C.catalog(log)
    hist = Q3.read_states(rd[lid])
    sor = Q3.truth_states(log, sor=True)
    truth = Q3.truth_states(log)
    for q in Q3.questions(log):
        key = f"{q['log']}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"
        outs["history"].append({"id": key, "pred": Q3.union_answer(q["qtype"], q["target"], hist, q["i"], q["j"], cat)})
        outs["final"].append({"id": key, "pred": sorted(C.answer(q["qtype"], q["target"], hist[-1], cat))})
        outs["sor"].append({"id": key, "pred": Q3.union_answer(q["qtype"], q["target"], sor, q["i"], q["j"], cat)})
        outs["sanity"].append({"id": key, "pred": Q3.union_answer(q["qtype"], q["target"], truth, q["i"], q["j"], cat)})
for arm, recs in outs.items():
    with open(os.path.join(HERE, "out", f"nsr3_{a.split}_{arm}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
print({k: len(v) for k, v in outs.items()})
