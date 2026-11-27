"""B18c step 0 (diagnosis, no test data): where does the B18b student fail on qa11 / qa13?

bAbI v1.2 en-10k TRAINING stories outside the 300 used for distillation (b18_distill.babi_original takes the first 300),
read by the B18b student with the unchanged World; every sentence's raw output and parsed events are recorded so the
errors can be read by hand.

python3 b18c_diag.py --model <Qwen3-1.7B> --adapter b18_models/student_b18b_s0 --tasks 11,13 --start 1000 --n 200 \
  --out b18c_diag
"""
import argparse
import glob
import json
import os

import b18


def babi_questions(task_num, root, start, n):
    f = glob.glob(f"{root}/tasks_1-20_v1-2/en-10k/qa{task_num}_*_train.txt")[0]
    out, cur = [], []
    for line in open(f):
        num, text = line.strip().split(" ", 1)
        if num == "1":
            cur = []
        if "\t" in text:
            q, a, _ = text.split("\t")
            out.append({"sents": list(cur), "question": q, "target": a})
        else:
            cur.append(text)
    return out[start:start + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", default="11,13")
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--babi_root", default=os.environ.get("BENCH_ROOT", ".") + "/babi")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import b14
    reader = b14.Reader(args.model, args.adapter)
    tpl = "Context: {context}\nSentence: {sentence}\nEvents:"
    os.makedirs(args.out, exist_ok=True)
    for t in args.tasks.split(","):
        samples = babi_questions(int(t), args.babi_root, args.start, args.n)
        rec = []
        worlds, stats = b18.run(lambda ps: reader(ps, bs=256), samples, record=rec, template=tpl)
        # record is filled sentence-major (sentence i of every live sample); rebuild it per sample
        per, k = [[] for _ in samples], 0
        for i in range(max(len(s["sents"]) for s in samples)):
            for j, s in enumerate(samples):
                if i < len(s["sents"]):
                    per[j].append({"sentence": s["sents"][i], "output": rec[k]["output"]})
                    k += 1
        ok = 0
        with open(f"{args.out}/qa{t}.jsonl", "w") as f:
            for s, w, p in zip(samples, worlds, per):
                pred = w.answer(s["question"])
                good = b18.correct(pred, s["target"])
                ok += good
                f.write(json.dumps({"question": s["question"], "target": s["target"], "pred": pred, "ok": good,
                                    "trace": p}) + "\n")
        print(f"qa{t}: {ok}/{len(samples)} correct on bAbI train stories {args.start}-{args.start + args.n - 1}",
              flush=True)


if __name__ == "__main__":
    main()
