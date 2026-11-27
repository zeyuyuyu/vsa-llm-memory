"""B48 part A selection (prereg/B48_one_form_and_alias_naming_prereg.md); train + dev outputs only.

  families (vq, dkvb, gumbel, st, cw): b42_select's rule on the C1 tuning runs (highest best-epoch dev accuracy,
           reading perm/named; ties within 0.002 to the configuration listed first in B42's grid)
  bind     lr in {3e-4, 1e-3}: the highest best-epoch dev accuracy; ties within 0.002 to 3e-4 (B31's rule)
  probe    the selection made inside b31_probe.py (tune mode)

  python b48_select.py --root out --out out/select.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b42_queues import FAMS, grid, cfg_name                                 # noqa: E402
from b48_queues import ENCS, BINDLR                                         # noqa: E402

PRIMARY = "perm/named"


def best_dev(d):
    accs = [e["%s:dev" % PRIMARY]["acc"] for e in d["runs"][0]["epochs"]]
    i = max(range(len(accs)), key=lambda j: (accs[j], -j))
    return accs[i], i + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--out", default="out/select.json")
    a = ap.parse_args()
    sel = {f: {} for f in FAMS + ["bind", "probe"]}
    sel["grids"] = {f: {} for f in FAMS + ["bind"]}
    sel["missing"] = []
    for enc in ENCS:
        k = "%s_C1" % enc
        for fam in FAMS:
            rows = []
            for c in grid(fam, enc, "C"):
                if fam == "cw":
                    c = dict(c, lr=BINDLR[enc])
                p = os.path.join(a.root, "tune", "%s_%s.json" % (k, cfg_name(fam, c)))
                if not os.path.exists(p):
                    sel["missing"].append(p); continue
                acc, ep = best_dev(json.load(open(p)))
                rows.append({"cfg": c, "dev": acc, "epoch": ep, "file": os.path.basename(p)})
            if rows:
                top = max(r["dev"] for r in rows)
                sel[fam][k] = [r for r in rows if r["dev"] >= top - 0.002][0]
                sel["grids"][fam][k] = rows
                print(fam, k, json.dumps(sel[fam][k]), "(%d configurations)" % len(rows))
        rows = []
        for lr in (3e-4, 1e-3):
            p = os.path.join(a.root, "bindtune", "%s_bind_lr%g.json" % (k, lr))
            if not os.path.exists(p):
                sel["missing"].append(p); continue
            acc, ep = best_dev(json.load(open(p)))
            rows.append({"lr": lr, "dev": acc, "epoch": ep})
        if rows:
            top = max(r["dev"] for r in rows)
            sel["bind"][k] = [r for r in rows if r["dev"] >= top - 0.002][0]
            sel["grids"]["bind"][k] = rows
            print("bind", k, json.dumps(sel["bind"][k]))
        p = os.path.join(a.root, "probe_tune", "%s.json" % k)
        if os.path.exists(p):
            sel["probe"][k] = json.load(open(p))["selected"]
            print("probe", k, json.dumps(sel["probe"][k]))
        else:
            sel["missing"].append(p)
    json.dump(sel, open(a.out, "w"), indent=1)
    if sel["missing"]:
        print("MISSING", len(sel["missing"]), sel["missing"][:5])


if __name__ == "__main__":
    main()
