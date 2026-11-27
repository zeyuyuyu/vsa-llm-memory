"""B33 gate G-FRESH (prereg/B33_fresh_task2_prereg.md), run once after the fresh set is generated and before any arm
reads it: size, second scorer of the labels (b33_track, independent of the generator's world and of the memory's
parse), no (story, question) row shared with any official task-2 split, input lengths, and story grouping.

  python b33_fresh_check.py --fresh b33_data/b33_fresh.jsonl --out results/b33_fresh_check.json
"""
import argparse, collections, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b33_common as B                                                # noqa: E402
from b33_track import answer_question                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default="b33_data/b33_fresh.jsonl")
    ap.add_argument("--out", default="results/b33_fresh_check.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.fresh)]
    agree = [answer_question(r["passage"], r["question"]) == r["answer"] for r in rows]
    from datasets import load_dataset
    d = load_dataset("Muennighoff/babi")
    key = lambda p, q: (p.strip(), q.strip())
    off = set()
    for sp in d:
        for r in d[sp]:
            if r["task"] == 2:
                off.add(key(r["passage"], r["question"]))
    shared = sum(key(r["passage"], r["question"]) in off for r in rows)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    texts = [r["passage"].strip() + "\n" + r["question"].strip() for r in rows]
    ntok = [len(tok(t)["input_ids"]) for t in texts]
    sid = B.story_ids([(t, 0) for t in texts])
    res = {"n_questions": len(rows), "n_stories": len({r["story"] for r in rows}),
           "labels_agree": bool(all(agree)), "n_label_disagree": int(len(agree) - sum(agree)),
           "shared_with_official": int(shared), "max_tokens": max(ntok), "n_over_768": sum(n > 768 for n in ntok),
           "story_ids_match": sid == [r["story"] for r in rows],
           "answers": dict(collections.Counter(r["answer"] for r in rows)),
           "sha256": hashlib.sha256(open(a.fresh, "rb").read()).hexdigest()}
    # inputs longer than 768 tokens are not a gate failure: they are kept, scored as the published code scores them
    # (the memory's parse loses a truncated question, the fine-tune sees the truncated text) and reported
    res["gate"] = bool(res["n_questions"] == B.FRESH_STORIES * 5 and res["labels_agree"] and shared == 0
                       and res["story_ids_match"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("G-FRESH %s" % ("PASS" if res["gate"] else "FAIL"))


if __name__ == "__main__":
    main()
