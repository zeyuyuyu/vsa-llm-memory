"""B41 gate G0 job list (box CPU): the second implementation on every test run (prereg gate G0).
N <= 3,200: first 150 queries, tables + VSA (D <= 4,096, kb in {1, 4}, END/HOP1/HOP4); N >= 32,000: 100 queries, tables.

  python b41_indep_jobs.py --split test --rhos 0.03 0.06 0.2 --seeds 1 2 3 | xargs -P 12 -I{} sh -c "{}"
"""
import argparse

PY = "/path/to/HOME/miniconda3/envs/dino_wm/bin/python"
ROOT = "/path/to/work/b41"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--rhos", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    for N in [32, 320, 3200, 32000, 100000]:
        for s in a.seeds:
            for r in a.rhos:
                run = "%s/runs/%s_s%d_N%d_r%.3f.npz" % (ROOT, a.split, s, N, r)
                out = run.replace("/runs/", "/indep/").replace(".npz", "_indep.json")
                extra = "--nsub 150" if N <= 3200 else "--nsub 100 --no_vsa"
                print("[ -f %s ] || OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 %s %s/code/b41_indep.py "
                      "--run %s --logits %s/head/%s_logits.npy %s --out %s > %s.log 2>&1" % (
                          out, PY, ROOT, run, ROOT, a.split, extra, out, out))


if __name__ == "__main__":
    main()
