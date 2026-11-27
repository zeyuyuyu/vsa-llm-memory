"""NSR4 run (prereg/NSR4_prereg.md): engine arms E0 / E1 (R1+R3) / E2 (R1+R2+R3) on one split.

Writes answer files for the independent scorers (impact: nsr/out/nsr1_<split>_nsr4_<arm>.jsonl for nsr1_score.py;
temporal: nsr/out/nsr3_<split>_nsr4_<arm>.jsonl for nsr3_score.py) and nsr/out/nsr4_<split>_facts.json with repaired /
broken fact counts (final state and every line's state) and rule firings.

python3 nsr/nsr4_run.py --eval nsr3 [--train dev,test,nsr1]
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_catalog as C  # noqa: E402
import nsr2 as M  # noqa: E402
import nsr2_types as T  # noqa: E402
import nsr3_questions as Q3  # noqa: E402
import nsr4 as E  # noqa: E402

M.SPLITS.setdefault("nsr2", ("nsr/data", "nsr/out/held/reader_nsr2_verify_v4.jsonl"))
M.SPLITS.setdefault("nsr3", ("nsr/data", "nsr/out/held/reader_nsr3_verify_v4.jsonl"))
ARMS = {"E0": (), "E1": ("R1", "R3"), "E2": ("R1", "R2", "R3")}


def states_for(log, rec, table, rules, stats):
    st, out = {}, []
    for i, evs in enumerate(rec["events"]):
        evs = [list(e) for e in (evs or [])]
        if rules:
            kind = log["lines"][i]["kind"] if "R2" in rules else "struct"  # R2 only acts on human lines
            evs = E.repair_line(evs, log["text_lines"][i], log, kind, table if "R3" in rules else None, stats)
        for ev in evs:
            Q3.apply(st, ev)
        out.append(dict(st))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dev,test,nsr1")
    ap.add_argument("--eval", default="nsr3")
    a = ap.parse_args()
    table = set()
    for sp in a.train.split(","):
        table |= T.learn_table(M.load(sp)[0].values())
    logs, rd = M.load(a.eval)
    answers = {arm: ([], []) for arm in ARMS}
    facts = {arm: collections.Counter() for arm in ARMS}
    for lid, log in sorted(logs.items()):
        cat = C.catalog(log)
        truth = Q3.truth_states(log)
        S = {}
        for arm, rules in ARMS.items():
            stats = collections.Counter()
            S[arm] = states_for(log, rd[lid], table, rules, stats)
            facts[arm].update({f"fires_{k}": v for k, v in stats.items()})
        for arm in ARMS:
            for q in C.questions(log):
                answers[arm][0].append({"log": lid, "scenario": log["scenario"], "qtype": q["qtype"], "target": q["target"],
                                        "pred": sorted(C.answer(q["qtype"], q["target"], S[arm][-1], cat))})
            for q in Q3.questions(log):
                key = f"{q['log']}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"
                answers[arm][1].append({"id": key, "pred": Q3.union_answer(q["qtype"], q["target"], S[arm], q["i"], q["j"], cat)})
            if arm == "E0":
                continue
            for x, p in log["truth"].items():
                b, r = S["E0"][-1].get(x) == p, S[arm][-1].get(x) == p
                facts[arm]["final_repaired"] += r and not b
                facts[arm]["final_broken"] += b and not r
            for i in range(len(truth)):
                for x, p in truth[i].items():
                    b, r = S["E0"][i].get(x) == p, S[arm][i].get(x) == p
                    facts[arm]["history_repaired"] += r and not b
                    facts[arm]["history_broken"] += b and not r
        for arm in ARMS:
            for x, p in log["truth"].items():
                facts[arm]["final_facts"] += 1
                facts[arm]["final_correct"] += S[arm][-1].get(x) == p
    for arm, (imp, tmp) in answers.items():
        with open(os.path.join(HERE, "out", f"nsr1_{a.eval}_nsr4_{arm}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in imp)
        with open(os.path.join(HERE, "out", f"nsr3_{a.eval}_nsr4_{arm}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in tmp)
    json.dump({arm: dict(c) for arm, c in facts.items()}, open(os.path.join(HERE, "out", f"nsr4_{a.eval}_facts.json"), "w"),
              indent=1)
    print(json.dumps({arm: dict(c) for arm, c in facts.items()}, indent=1))


if __name__ == "__main__":
    main()
