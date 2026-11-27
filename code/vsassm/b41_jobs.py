"""B41 job lists for the engine (box). Prints one shell line per job; a worker runs its share sequentially on one GPU.

  python b41_jobs.py --split dev --rhos 0.06 0.15 0.3 --seeds 101 --nq 500 --worker 0 --nworkers 3 | sh
"""
import argparse

NS = [100000, 32000, 3200, 320, 32]
PY = "/path/to/HOME/miniconda3/envs/dino_wm/bin/python"
ROOT = "/path/to/work/b41"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--rhos", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--nq", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    jobs = [(s, N, r) for N in NS for s in a.seeds for r in a.rhos]
    for i, (s, N, r) in enumerate(jobs):
        if i % a.nworkers != a.worker:
            continue
        out = "%s/runs/%s_s%d_N%d_r%.3f.npz" % (ROOT, a.split, s, N, r)
        print("[ -f %s ] || %s %s/code/b41_engine.py --split %s --seed %d --N %d --rho %.3f --nq %d "
              "--logits %s/head/%s_logits.npy --out %s" % (out, PY, ROOT, a.split, s, N, r, a.nq, ROOT, a.split, out))


if __name__ == "__main__":
    main()
