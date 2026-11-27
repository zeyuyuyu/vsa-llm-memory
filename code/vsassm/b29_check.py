"""B29 second scorer (prereg/B29_same_tooling_prereg.md): recomputes every accuracy of results/b29_scores.json from the
saved per-sample / per-question outputs with separate code, and reports any difference.

  RULER: its own string match (every reference, lower case, contained in the prediction; mean over references), its
         own reading of B21's files (file order within each length and task), its own 9-task means.
  Logs:  gold from a different derivation than the registered scorers: impact = nsr1_catalog.answer (the hand-written
         graph query) on the simulation's TRUE state (registered: nsr1_score.gold); window = nsr3_questions.union_answer
         over nsr3_questions.truth_states (registered: nsr3_score's own replay and affected()); location = the
         question's stored answer compared by its own rule (registered: run_real.ok).

python3 -I -B b29_check.py [--scores results/b29_scores.json]
"""
import argparse
import collections
import glob
import json
import os
import site
import sys

if os.path.isdir(site.getusersitepackages()) and site.getusersitepackages() not in sys.path:
    sys.path.append(site.getusersitepackages())

ROOT = os.path.dirname(os.path.abspath(__file__))
RO = os.path.join(ROOT, "b29_out", "ruler")
LO = os.path.join(ROOT, "b29_out", "logs")
L9 = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
      "niah_multivalue", "niah_multiquery", "vt"]
LENS = [4096, 8192, 16384, 32768, 65536, 131072]


def sm(pred, refs):
    p = pred.lower()
    return sum(1 for r in refs if r.lower() in p) / len(refs)


def ruler_means(rows, field):
    cell = collections.defaultdict(list)
    for r in rows:
        if field in r and not r.get("missing"):
            cell[(int(r["length"]), r["task"])].append(sm(r[field], r["outputs"]))
    out = {}
    for L in LENS:
        if all(len(cell[(L, t)]) == 100 for t in L9):
            out[str(L)] = round(100 * sum(sum(cell[(L, t)]) for t in L9) / 900, 2)
        else:
            out[str(L)] = None
    return out


def b21_means():
    cell = collections.defaultdict(list)
    for fp in glob.glob(os.path.join(ROOT, "b21_dl", "b21_test", "*.jsonl")):
        for l in open(fp):
            r = json.loads(l)
            cell[(int(r["length"]), r["task"])].append(sm(r["pred_table"], r["outputs"]))
    return {str(L): round(100 * sum(sum(cell[(L, t)]) for t in L9) / 900, 2) for L in LENS}


def logs_of(ddir, split):
    out = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for l in open(os.path.join(ROOT, ddir, f"{sc}_{split}.jsonl")):
            d = json.loads(l)
            out[d["id"]] = d
    return out


def gold2():
    sys.path.insert(0, os.path.join(ROOT, "nsr"))
    import nsr1_catalog as C  # noqa: E402
    import nsr3_questions as Q3  # noqa: E402
    g = {"impact": {}, "window": {}, "where": {}}
    for lid, log in logs_of("nsr/data", "nsr1").items():
        cat = C.catalog(log)
        for q in C.questions(log):
            g["impact"][f"{lid}|{q['qtype']}|{q['target']}"] = sorted(C.answer(q["qtype"], q["target"], log["truth"], cat))
    for lid, log in logs_of("nsr/data", "nsr2").items():
        cat = C.catalog(log)
        tr = Q3.truth_states(log)
        for q in Q3.questions(log):
            g["window"][f"{lid}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"] = sorted(
                Q3.union_answer(q["qtype"], q["target"], tr, q["i"], q["j"], cat))
    for lid, log in logs_of("apps/real/data", "test2").items():
        for qi, q in enumerate(log["questions"]):
            g["where"][f"{lid}|{qi}"] = (q["kind"], q["a"])
    return g


def right(kind, pred, gold):
    if kind == "where":
        k, a = gold
        if pred is None:
            return False
        if k == "where":
            return pred == a
        if k == "what_in":
            return isinstance(pred, list) and collections.Counter(pred) == collections.Counter(a)
        return isinstance(pred, list) and pred == a
    return pred is not None and sorted(set(pred)) == gold and len(set(pred)) == len(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(ROOT, "results", "b29_scores.json"))
    a = ap.parse_args()
    J = json.load(open(a.scores))
    bad = 0
    print("# B29 second scorer\n\n## RULER 9-task means (dict answers), recomputed")
    mine = {"8B (B21, banked)": b21_means()}
    for name, f in {"8B replay (gate G1)": "test_rel_q8.jsonl", "32B": "test_rel_q32.jsonl", "1.7B": "test_rel_q17.jsonl",
                    "32B full read (vLLM)": "test_full_q32.jsonl"}.items():
        p = os.path.join(RO, f)
        if os.path.exists(p):
            mine[name] = ruler_means([json.loads(l) for l in open(p)], "pred_dict")
            if "full" in f:
                mine[name + " store"] = ruler_means([json.loads(l) for l in open(p)], "pred_store")
    for name, m in mine.items():
        v = J["ruler"]["means"].get(name)
        same = v is None or all((v.get(L) is None and m.get(L) is None) or
                                (v.get(L) is not None and m.get(L) is not None and abs(v[L] - m[L]) < 0.006)
                                for L in m) if v else None
        if v is not None and not same:
            bad += 1
        print(f"  {name:28s} " + " ".join(f"{L}:{m[L]}" for L in m) +
              (f"   verdict {'agrees' if same else 'DIFFERS: ' + json.dumps(v)}" if v is not None else "   (not in verdict)"))
    print("\n## Logs, accuracy with an independently derived gold")
    g = gold2()
    for kind, split in (("where", "test2"), ("impact", "nsr1"), ("window", "nsr2")):
        for tag, name in (("banked", "1.7B fine-tuned (banked)"), ("ft17", "1.7B fine-tuned (re-run, gate G2)"),
                          ("q8", "Qwen3-8B prompted"), ("q32", "Qwen3-32B prompted"),
                          ("gpt61low", "GPT-6.1 Sol (low) prompted"), ("opus_low", "Claude Opus 5.5 (low) prompted")):
            p = os.path.join(LO, f"ans_{split}_{tag}.jsonl")
            if not os.path.exists(p):
                continue
            recs = [json.loads(l) for l in open(p)]
            ok = [right(kind, r["pred"], g[kind][r["id"]]) for r in recs]
            acc = sum(ok) / len(ok)
            v = J["logs"]["acc"].get(kind, {}).get(name)
            same = v is not None and abs(v - acc) < 5e-5
            bad += v is not None and not same
            line = f"  {kind:6s} {name:36s} {acc:.4f} ({sum(ok)}/{len(ok)})" + (
                f"   verdict {v:.4f} {'agrees' if same else 'DIFFERS'}" if v is not None else "")
            dp = os.path.join(LO, f"ans_{split}_{tag}_dict.jsonl")
            if os.path.exists(dp):
                rd = [json.loads(l) for l in open(dp)]
                okd = [right(kind, r["pred"], g[kind][r["id"]]) for r in rd]
                line += f"   dict {sum(okd) / len(okd):.4f}"
            print(line)
    print(f"\n{bad} disagreements with the verdict")


if __name__ == "__main__":
    main()
