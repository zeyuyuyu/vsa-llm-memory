"""B33 additional check (added after registration, before the verdict; reporting only): B28's separately written numpy
float64 implementation of the exact-symbol task-2 route (b28_indep.f2 / codes / torch_rows, unchanged) run on the
gold structures that b33_babi.py dumped for every evaluation set, (1) with the main runs' codebooks: per-question
agreement with the main run's exact-rows predictions under all five encodings; (2) with 20 fresh numpy codebooks
(comm): accuracy over all questions of the set (an unparsed question counts as wrong).

  python b33_indep.py --mem b33_out/b33_mem_completed.json --out results/b33_indep.json
"""
import argparse, gzip, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b28_indep import f2, codes, torch_rows, rownorm, ENC, D    # noqa: E402


def load(p):
    return json.load(open(p)) if os.path.exists(p) else json.load(gzip.open(p + ".gz", "rt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", default="b33_out/b33_mem_completed.json")
    ap.add_argument("--out", default="results/b33_indep.json")
    ap.add_argument("--fresh", type=int, default=20)
    a = ap.parse_args()
    d = load(a.mem)
    n_p, n_l = len(d["persons"]), len(d["vocab"])
    TOT = d["codebook"]
    R = {}
    for name, S in d["sets"].items():
        gold, n = S["gold"], S["n_questions"]
        for enc in ENC:
            for s in (42, 123, 456):
                key = "parser/%s/%d" % (enc, s)
                if key not in S["runs"]:
                    continue
                K, V = codes(enc, torch_rows(TOT, s + 11), seed=s)
                pred = [f2(g, K, V, 0, n_p, n_p + n_l) for g in gold]
                main = [o["pred"] for o in S["runs"][key]["per_q"]]
                R["%s/%s/%d" % (name, enc, s)] = {
                    "agree": float(np.mean([x == y for x, y in zip(pred, main)])),
                    "acc_all_indep": sum(int(p == g["label"]) for p, g in zip(pred, gold)) / n,
                    "acc_all_main": S["runs"][key]["acc_all"]}
        accs = []
        for j in range(a.fresh):
            rng = np.random.default_rng(330_000 + j)
            K, V = codes("comm", rownorm(rng.standard_normal((TOT, D))), rng=rng)
            accs.append(sum(int(f2(g, K, V, 0, n_p, n_p + n_l) == g["label"]) for g in gold) / n)
        R["%s/fresh_codebooks_comm" % name] = accs
        ag = [v["agree"] for k, v in R.items() if k.startswith(name + "/") and isinstance(v, dict)]
        print("%-8s same codebooks: min agreement %.4f over %d cells; fresh codebooks (comm) mean %.4f min %.4f"
              % (name, min(ag), len(ag), np.mean(accs), min(accs)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
