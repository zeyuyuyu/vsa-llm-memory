"""B41 second scorer: regenerates every test instance from its seed (b41_data.make), takes the gold answer e_h from
the regenerated chain (not from the engine's stored answers), rescores every configuration's stored answers and
compares with the engine's per-run accuracy (meta json) and with the verdict's pooled cell accuracy.

  python b41_rescore.py --runs /path/to/work/b41/runs --scores /path/to/work/b41/results/b41_test_verdict.json
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_data as DA                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    S = json.load(open(a.scores))
    lv = {0.03: "low", 0.06: "mid", 0.2: "high"}
    pooled = {}
    n_runs = mism_gold = mism_acc = 0
    for f in sorted(glob.glob(os.path.join(a.runs, "%s_s*_N*_r*.npz" % a.split))):
        meta = json.load(open(f.replace(".npz", ".json")))
        d = DA.make(meta["seed"], meta["N"], meta["nq"], meta["split"])
        gold = d["q_ents"][:, [2, 4, 6]]
        Z = np.load(f)
        if not np.array_equal(gold, Z["_answers"]):
            mism_gold += 1
        names = sorted(set(k.split("|")[0] for k in Z.files if "|answer" in k))
        for nm in names:
            acc = (Z["%s|answer" % nm] == gold).mean(0)
            if np.abs(acc - np.array(meta["acc"][nm])).max() > 1e-3:
                mism_acc += 1
            key = (meta["rho"], meta["N"], nm)
            pooled.setdefault(key, []).append((Z["%s|answer" % nm] == gold))
        n_runs += 1
    cell_bad = 0
    n_cells = 0
    for (rho, N, nm), parts in pooled.items():
        c = np.concatenate(parts).mean(0)
        for hi, h in enumerate((2, 4, 6)):
            k = "%s|h%d|N%d" % (lv[round(rho, 2)], h, N)
            if nm in S[k]["acc"]:
                n_cells += 1
                if abs(S[k]["acc"][nm][0] - c[hi]) > 1e-9:
                    cell_bad += 1
    print("runs rescored %d; gold answers regenerated != stored: %d runs; per-run accuracy mismatches (>0.001): %d; "
          "pooled cell-config accuracies compared %d, mismatches %d" % (n_runs, mism_gold, mism_acc, n_cells, cell_bad))


if __name__ == "__main__":
    main()
