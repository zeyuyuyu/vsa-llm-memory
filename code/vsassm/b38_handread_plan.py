"""B38 hand reading (prereg: at least three failed keys per primary arm at its N* minus one step at k8 and k16, and in
the natural-layout lines condition at k8), re-simulated with the run's own seeds by b38_check.handread.

python3 b38_handread_plan.py STORES.json.gz CHOICE.json TEST1.json[,TEST2...] > results/b38_handread.txt
"""
import contextlib
import io
import json
import sys

import b38_check as K
import b38_design as Dz
import b38_eq as E

ARMS = ("V-last", "V-Snt", "E-LDPC", "E-BCH", "T-rep", "T-vhdd")


def main(stores_f, choice_f, test_fs):
    choice = json.load(open(choice_f))
    acc, results = Dz.test_curves(test_fs.split(","))
    for cond in ("k8", "k16", "lines-nat-k8"):
        for v in ARMS:
            cur, ns = Dz.test_nstar(acc, choice, v, cond)
            m = E.parse_cond(cond)["match"]
            if cond == "lines-nat-k8":                       # the arm's own i.i.d. N* budget, damaged
                _, ns_iid = Dz.test_nstar(acc, choice, v, m)
                gi = ns_iid
            else:
                gi = None if ns is None else ns - 1
            if gi is None or gi < 0:
                print("== %s %s: no budget to read (N* %s)" % (v, cond, ns))
                continue
            B = E.GRID[gi]
            fam = Dz.FAMILY[v]
            if v in ("E-BCH", "E-CAM"):
                design = "auto"
            elif v in ("V-last", "T-rep"):
                design = None
            elif fam == "V-plain":
                design = None
            else:
                design = json.loads(Dz.design_for(choice, v, m, gi))
            rule = {"V-Snt": "snt", "V-last": "last"}.get(v, "")
            shown = 0
            print("\n######## %s at %s, B = %.0f (share %.4f)" % (v, cond, B, cur.get(gi, float("nan"))))
            for r in results:
                if r["arm"] != fam or shown >= 3:
                    continue
                for x in r["res"]:
                    if "cond" not in x or x["cond"] != cond or Dz.gidx(x["B"]) != gi or not x.get("failed"):
                        continue
                    if fam == "V-plain" and x["rule"] != rule:
                        continue
                    if v in ("E-BCH", "E-CAM") and (x.get("dkey") or "auto") != "auto":
                        continue
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        K.handread(stores_f, r["sid"], fam, x["B"], json.dumps(design), cond,
                                   min(3 - shown, len(x["failed"])))
                    out = buf.getvalue().splitlines()
                    keep = [l for l in out if l.startswith("==") or ("rule" in l and (not rule or rule in l))
                            or l.startswith("    key")]
                    if fam == "V-plain":                      # only the variant's own rule block
                        sel, on = [], False
                        for l in out:
                            if l.startswith("=="):
                                sel.append(l)
                            elif l.strip().startswith("rule"):
                                on = l.split()[1] == rule
                                if on:
                                    sel.append(l)
                            elif on:
                                sel.append(l)
                        keep = sel
                    print("\n".join(keep))
                    shown += min(3 - shown, len(x["failed"]))
                    break


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
