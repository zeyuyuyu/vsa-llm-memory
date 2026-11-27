"""B33 hand-reading helper (prereg/B33_fresh_task2_prereg.md, "Checks"): prints, for chosen questions of a set, the
raw story, the question, the gold answer, the text tracker's answer and every arm's prediction, so that each memory
error (all of them) and samples of fine-tune and prompting errors can be read in the raw story.

  python b33_handread.py --runs b33_out --set fresh --pick memory_errors
  python b33_handread.py ... --pick ft12_errors --n 10 --seed 0
"""
import argparse, gzip, json, os, random, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b33_track import answer_question                                 # noqa: E402


def load(p):
    if os.path.exists(p):
        return json.load(open(p))
    return json.load(gzip.open(p + ".gz", "rt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="b33_out")
    ap.add_argument("--set", default="fresh")
    ap.add_argument("--pick", default="memory_errors", choices=["memory_errors", "ft12_errors", "prompt_errors"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mc = load(os.path.join(a.runs, "b33_mem_completed.json"))
    S = mc["sets"][a.set]
    vocab = mc["vocab"]
    n = len(S["labels_all"])
    texts = S["texts"]
    mem = {}
    for k, v in S["runs"].items():
        if k.startswith("probe_argmax/comm/"):
            p = [None] * n
            for j, o in zip(S["row_index"], v["per_q"]):
                p[j] = vocab[o["pred"]]
            mem[k.split("/")[-1]] = p
    ftp = {}
    for s in (42, 123, 456):                       # one file per seed, or all seeds in one file
        for fn in ("b33_lora_r16_e6_lr0.0001_expr_s%d.json" % s, "b33_lora_r16_e6_lr0.0001_expr.json"):
            p = os.path.join(a.runs, fn)
            if os.path.exists(p) or os.path.exists(p + ".gz"):
                ft = load(p)
                for r in ft["runs"]:
                    if r["seed"] == s:
                        ftp[s] = [vocab[ft["kept"][int(np.argmax(x))]] for x in r["sets"][a.set]["logits"]]
                break
    if not ftp:
        ftp = {0: ["-"] * n}
    pr = load(os.path.join(a.runs, "b33_prompt.json"))
    prp = [vocab[pr["kept"][j]] for j in pr["sets"][a.set]["pred"]]
    gold = [vocab[y] for y in S["labels_all"]]
    if a.pick == "memory_errors":
        idx = sorted({i for p in mem.values() for i in range(n) if p[i] != gold[i]})
    elif a.pick == "ft12_errors":
        cand = sorted({i for p in ftp.values() for i in range(n) if p[i] != gold[i]})
        idx = sorted(random.Random(a.seed).sample(cand, min(a.n, len(cand))))
    else:
        cand = [i for i in range(n) if prp[i] != gold[i]]
        idx = sorted(random.Random(a.seed).sample(cand, min(a.n, len(cand))))
    print("%s on %s: %d questions listed" % (a.pick, a.set, len(idx)))
    for i in idx:
        lines = texts[i].split("\n")
        print("=" * 100)
        print("question %d (story %d) gold %s | tracker %s | memory %s | FT12 %s | prompting %s" % (
            i, S["story_id"][i], gold[i], answer_question("\n".join(lines[:-1]), lines[-1]),
            "/".join(str(mem[s][i]) for s in sorted(mem)), "/".join(ftp[s][i] for s in sorted(ftp)), prp[i]))
        for j, l in enumerate(lines):
            print("  %2d %s" % (j + 1, l))


if __name__ == "__main__":
    main()
