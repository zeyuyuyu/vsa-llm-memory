"""NSR3 LLM arm: prompts for the temporal questions (same intro, asset list and catalog blocks as NSR1) and parsing.

python3 nsr/nsr3_llm.py prep --split nsr2 --logs nsr/data
python3 nsr/nsr3_llm.py parse --split nsr2 --logs nsr/data --problems P --runs 'DIR/qwen3-32b_NSR3_n*.jsonl'
    --forced DIR/forced512_qwen3-32b.jsonl --arm llm_qwen3-32b [--normalize]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_llm as L  # noqa: E402
import nsr3_questions as Q3  # noqa: E402

SC = ["warehouse", "biobank", "home", "ops"]


def qid(q):
    return f"{q['log']}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "parse"])
    ap.add_argument("--split", default="nsr2")
    ap.add_argument("--logs", default=os.path.join(HERE, "data"))
    ap.add_argument("--problems")
    ap.add_argument("--runs")
    ap.add_argument("--forced")
    ap.add_argument("--arm", default="llm_qwen3-32b")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--repair", action="store_true")  # frontier amendment 2
    a = ap.parse_args()
    L.NORMALIZE = a.normalize
    L.REPAIR = a.repair
    logs = L.load(a.split, a.logs)
    if a.cmd == "prep":
        path = os.path.join(HERE, "out", f"nsr3_llm_problems_{a.split}.jsonl")
        with open(path, "w") as f:
            for log in logs:
                n = SC.index(log["scenario"]) * 100 + int(log["id"].rsplit("-", 1)[1])
                for idx, q in enumerate(Q3.questions(log)):
                    p = (f"{L.INTRO[log['scenario']]} Answer the question about the record.\n\n{L.assets_block(log)}\n\n"
                         f"{L.catalog_block(log)}\n\n" + "\n".join(log["text_lines"]) + f"\n\nQuestion: {q['q']}\n{L.ASK}")
                    f.write(json.dumps({"task": "NSR3", "n": n, "idx": idx, "id": qid(q), "prompt": p}) + "\n")
        print(path)
        return
    byid = {l["id"]: l for l in logs}
    pid = {(r["n"], r["idx"]): r["id"] for r in map(json.loads, open(a.problems))}
    runs = {pid[(r["n"], r["idx"])]: r for fp in sorted(glob.glob(a.runs)) for r in map(json.loads, open(fp))}
    forced = {pid[(r["n"], r["idx"])]: r["forced_answer"] for r in map(json.loads, open(a.forced))} if a.forced else {}
    for mode in ("strict", "forced"):
        path = os.path.join(HERE, "out", f"nsr3_{a.split}_{a.arm}{'n' if a.normalize else ''}_{mode}.jsonl")
        with open(path, "w") as f:
            for log in logs:
                for q in Q3.questions(log):
                    r = runs.get(qid(q))
                    txt = None if r is None else (r["answer_text"] if not r["hit_limit"] else
                                                  (forced.get(qid(q)) if mode == "forced" else None))
                    f.write(json.dumps({"id": qid(q), "pred": L.resolve(txt, byid[log["id"]])}) + "\n")
        print(path)


if __name__ == "__main__":
    main()
