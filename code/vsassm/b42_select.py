"""B42 selection (prereg/B42_label_free_discrete_prereg.md); train + dev outputs only.

Per family, encoder and arena: the configuration of the registered grid (b42_queues.grid, seed 42) with the highest
best-epoch dev accuracy (reading perm/named); ties within 0.002 go to the configuration listed first in the grid.

  python b42_select.py --root out --out out/select.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b42_queues import ENCS, FAMS, grid, cfg_name                          # noqa: E402

PRIMARY = "perm/named"


def best_dev(d):
    accs = [e["%s:dev" % PRIMARY]["acc"] for e in d["runs"][0]["epochs"]]
    i = max(range(len(accs)), key=lambda j: (accs[j], -j))
    return accs[i], i + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--out", default="out/select.json")
    ap.add_argument("--arenas", default="ABC")
    a = ap.parse_args()
    sel = {f: {} for f in FAMS}
    sel["grids"] = {f: {} for f in FAMS}
    sel["missing"] = []
    for ar in a.arenas:
        for enc in ENCS:
            k = "%s_%s" % (enc, ar)
            for fam in FAMS:
                rows = []
                for c in grid(fam, enc, ar):
                    p = os.path.join(a.root, "tune", "%s_%s.json" % (k, cfg_name(fam, c)))
                    if not os.path.exists(p):
                        sel["missing"].append(p); continue
                    acc, ep = best_dev(json.load(open(p)))
                    rows.append({"cfg": c, "dev": acc, "epoch": ep, "file": os.path.basename(p)})
                if not rows:
                    continue
                top = max(r["dev"] for r in rows)
                sel[fam][k] = [r for r in rows if r["dev"] >= top - 0.002][0]
                sel["grids"][fam][k] = rows
                print(fam, k, json.dumps(sel[fam][k]), "(%d configurations)" % len(rows))
    json.dump(sel, open(a.out, "w"), indent=1)
    if sel["missing"]:
        print("MISSING", len(sel["missing"]))


if __name__ == "__main__":
    main()
