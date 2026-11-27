"""B68 sample reading by the AI-assisted process (prereg/B68_early_layer_labelfree_prereg.md): arena A, one source
(default the input embeddings), the label-free reader with the highest selected dev accuracy (out/select.json; bind
readings excluded, as in b42_handread), seed 42 at its dev-selected epoch: n correct and n wrong test instances, with
the query, the gold chain R -> X -> Y -> F, the candidate, the clean-up winners of hops 1 and 2, and (discrete readers)
the code of each chain mention and the other symbols of the instance that share its code. Reading only; it changes no
number.

  python b68_sample.py --cell qwen8L0_A --n 6 > out/sample_qwen8L0_A.txt
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402

PRIMARY = "perm/named"
LF = ["vq", "dkvb", "gumbel", "st", "cw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--fam", default=None)
    ap.add_argument("--root", default="out")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    sel = json.load(open(os.path.join(a.root, "select.json")))
    k = a.cell
    fam = a.fam or max(LF, key=lambda f: (sel[f][k]["dev"], -LF.index(f)))
    d = json.load(open(os.path.join(a.root, "final", "%s_%s.json" % (k, fam))))
    run = d["runs"][0]
    j = run["dev_selected_epoch"] - 1
    e = run["epochs"][j]["%s:test" % PRIMARY]
    side_p = os.path.join(a.root, "final", "%s_%s.side.pt" % (k, fam))
    asg = torch.load(side_p, weights_only=False)[run["seed"]]["assign_test"].numpy() if os.path.exists(side_p) else None
    st = S.build("A", "test")
    y = np.array([s["label"] for s in st])
    pred = (np.array(e["scores"]) >= e["thr"]).astype(int)
    W = np.array(e["wins"])
    right = [i for i in range(len(st)) if pred[i] == y[i]]
    wrong = [i for i in range(len(st)) if pred[i] != y[i]]
    print("%s, reader %s (selected dev %.3f, config %s), seed %d epoch %d: right %d, wrong %d of %d" % (
        k, fam, sel[fam][k]["dev"], json.dumps(sel[fam][k]["cfg"]), run["seed"], j + 1, len(right), len(wrong), len(st)))
    for tag, ids in (("RIGHT", right[:a.n]), ("WRONG", wrong[:a.n])):
        for i in ids:
            s = st[i]
            sym = [m[2] for m in s["mentions"]]
            kinds = [m[1] for m in s["mentions"]]
            txt = lambda p: s["text"][s["mentions"][p][3]:s["mentions"][p][4]]
            r0 = [r for r, x in s["secA"] if sym[r] == sym[s["q"]]][0]
            x0 = dict(s["secA"])[r0]
            xk = [x for x, yy in s["secB"] if sym[x] == sym[x0]][0]
            y0 = dict(s["secB"])[xk]
            yk = [yy for yy, f in s["secC"] if sym[yy] == sym[y0]][0]
            f0 = dict(s["secC"])[yk]
            print("\n[%s] #%d label %d, memory says %d (score %.3f, threshold %.3f)" % (tag, i, s["label"], pred[i], e["scores"][i], e["thr"]))
            print("  chain %s -> %s -> %s -> %s, candidate %s" % (txt(s["q"]), txt(x0), txt(y0), txt(f0), txt(s["cand"])))
            print("  hop-1 winner %s (%s), hop-2 winner %s (%s)" % (
                txt(W[i, 0]), "right" if sym[W[i, 0]] == sym[x0] else "WRONG", txt(W[i, 1]),
                "right" if sym[W[i, 1]] == sym[y0] else "WRONG"))
            if asg is not None:
                def code(p):
                    v = asg[i, p]
                    return tuple(v.tolist()) if np.ndim(v) else int(v)
                for name, p in (("query R", s["q"]), ("context R", r0), ("X value", x0), ("X key", xk), ("Y value", y0),
                                ("Y key", yk), ("F value", f0), ("candidate F", s["cand"])):
                    c = code(p)
                    share = sorted(set(txt(q) for q in range(len(sym)) if kinds[q] == kinds[p] and sym[q] != sym[p] and code(q) == c))
                    print("    %-12s %-8s code %-24s shared with other symbols: %s" % (name, txt(p), str(c), ", ".join(share) or "none"))


if __name__ == "__main__":
    main()
