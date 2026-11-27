"""B31 diagnostic (exploratory, not registered; reading only): where does the naming head misname mentions?

From the saved per-instance misnamed lists of the naming-head finals (test split, each seed's dev-selected epoch,
perm/named reading), the error rate per mention kind and position class (context, query, candidate), and, for
misnamed decimal fillers and hex markers, the share whose predicted name is a reordering of the gold digits (e.g. 123
named 132 or 213; 3A named A3), the signature of an order-blind span mean over digit-split tokens.

  python b31_namediag.py --root /path/to/work/b31/out > out/namediag.txt
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402
from b31_arms import layout                                                 # noqa: E402


def sel_epoch(run):
    accs = [e["perm/named:dev"]["acc"] for e in run["epochs"]]
    return max(range(len(accs)), key=lambda j: (accs[j], -j))


def digits(kind, v, arena):
    if kind in "RXY" or (arena in "BC" and kind == "R"):
        return "%02X" % v
    return "%d" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/path/to/work/b31/out")
    a = ap.parse_args()
    for enc in ("phi2", "qwen8"):
        for ar in "ABC":
            p = os.path.join(a.root, "arms_final", "%s_%s_name.json" % (enc, ar))
            if not os.path.exists(p):
                continue
            st = S.build(ar, "test")
            kinds, hops, q, c, _ = layout(ar)
            d = json.load(open(p))
            tot, err, perm_err = Counter(), Counter(), Counter()
            for run in d["runs"]:
                e = run["epochs"][sel_epoch(run)]["perm/named:test"]
                for i, mis in enumerate(e["misnamed"]):
                    for j in range(len(kinds)):
                        cls = "query" if j == q else ("candidate" if j == c else "context")
                        tot[(kinds[j], cls)] += 1
                    for j, pr in mis:
                        cls = "query" if j == q else ("candidate" if j == c else "context")
                        err[(kinds[j], cls)] += 1
                        g = st[i]["mentions"][j][2]
                        dg, dp = digits(kinds[j], g, ar), digits(kinds[j], pr, ar)
                        if sorted(dg) == sorted(dp) and dg != dp:
                            perm_err[(kinds[j], cls)] += 1
            print("%s arena %s (3 seeds pooled, test, dev-selected epoch)" % (enc, ar))
            for k in sorted(tot):
                if err[k] or k[1] != "context":
                    print("   %-2s %-9s error rate %.4f (%d of %d); of the errors, digit reorderings %d" % (
                        k[0], k[1], err[k] / tot[k], err[k], tot[k], perm_err[k]))


if __name__ == "__main__":
    main()
