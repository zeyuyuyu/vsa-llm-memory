"""B41 gate G1 (dev): with one-hot (noise-free) readings, VSA-HOP1 at D = 4,096 must answer >= 0.99 at N = 320;
also prints the other modes / D and the noise-free tables (all must be 1.000) as a code check.

  python b41_g1.py --logits /path/to/work/b41/head/dev_logits.npy --out /path/to/work/b41/runs/g1.json
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_engine as EN                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, nargs="*", default=[320, 3200])
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {}
    for N in a.N:
        R = EN.Run("dev", 101, N, 0.03, 500, a.logits, "cuda:0")
        oh = lambda e: torch.nn.functional.one_hot(torch.from_numpy(e.astype(np.int64)), EN.E).double().cuda()
        R.Ps, R.Po, R.P0 = oh(R.d["f_s"]), oh(R.d["f_o"]), oh(R.d["q_ents"][:, 0])
        r = {"ARGMAX": R.run_table(1, 1)["correct"].mean(0).tolist(), "EXACT": R.run_exact()["correct"].mean(0).tolist()}
        for D in (1024, 4096, 16384):
            PHI, P, Pinv = R.codebook(D)
            Mem = R.build_memory(PHI, P, 1)
            for m, x in R.run_vsa(D, 1, PHI, P, Pinv, Mem).items():
                r["VSA_%s_D%d" % (m, D)] = x["correct"].mean(0).tolist()
        res[N] = r
        print(N, json.dumps(r), flush=True)
    g1 = min(res[320]["VSA_HOP1_D4096"]) >= 0.99 if 320 in res else None
    res["G1_pass"] = g1
    json.dump(res, open(a.out, "w"), indent=1)
    print("G1", "PASS" if g1 else "FAIL")


if __name__ == "__main__":
    main()
