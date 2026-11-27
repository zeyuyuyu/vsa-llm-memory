"""B42 hand-reading helper (prereg/B42_label_free_discrete_prereg.md): for an encoder and arena C, the label-free family
with the highest selected dev accuracy (out/select.json), seed 42 at its dev-selected epoch: test instances where it is
wrong, with the raw text, the gold chain, the clean-up winners and which hop failed, and (discrete families) the code
each chain mention was given: the query, the level-0 role and filler, the level-1 role written in hex for the same
symbol as the level-0 filler (the alias), the level-1 filler and the candidate. Reading only; it changes no number.

  python b42_handread.py --enc qwen8 --n 12 > out/handread_qwen8_C.txt
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
    ap.add_argument("--enc", required=True)
    ap.add_argument("--fam", default=None)
    ap.add_argument("--root", default="out")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    sel = json.load(open(os.path.join(a.root, "select.json")))
    k = "%s_C" % a.enc
    fam = a.fam or max(LF, key=lambda f: sel[f][k]["dev"])
    d = json.load(open(os.path.join(a.root, "final", "%s_%s.json" % (k, fam))))
    run = d["runs"][0]
    j = run["dev_selected_epoch"] - 1
    e = run["epochs"][j]["%s:test" % PRIMARY]
    side_p = os.path.join(a.root, "final", "%s_%s.side.pt" % (k, fam))
    asg = None
    if os.path.exists(side_p):
        asg = torch.load(side_p, weights_only=False)[run["seed"]]["assign_test"].numpy()
    st = S.build("C", "test")
    y = np.array([s["label"] for s in st])
    pred = (np.array(e["scores"]) >= e["thr"]).astype(int)
    W = np.array(e["wins"])
    wrong = [i for i in range(len(st)) if pred[i] != y[i]]
    print("%s arena C, family %s (selected dev %.3f, config %s), seed %d epoch %d: wrong on %d of %d" % (
        a.enc, fam, sel[fam][k]["dev"], json.dumps(sel[fam][k]["cfg"]), run["seed"], j + 1, len(wrong), len(st)))
    for i in wrong[:a.n]:
        s = st[i]
        sym = [m[2] for m in s["mentions"]]
        ch = s["chain"]
        lv0, lv1 = s["lv"]
        r0 = [r for r, f in lv0 if sym[r] == ch[0]][0]; f0 = dict(lv0)[r0]
        r1 = [r for r, f in lv1 if sym[r] == ch[1]][0]; f1 = dict(lv1)[r1]
        h1 = sym[W[i, 0]] == ch[1]; h2 = sym[W[i, 1]] == ch[2]
        print("\n#%d label %d, chain %d -> %d -> %d, candidate %d; memory says %d (score %.3f, threshold %.3f)" % (
            i, s["label"], ch[0], ch[1], ch[2], sym[s["cand"]], pred[i], e["scores"][i], e["thr"]))
        print("  text: %s ..." % s["text"][:300])
        print("  hop 1 winner symbol %d (%s), hop 2 winner symbol %d (%s)" % (
            sym[W[i, 0]], "right" if h1 else "WRONG", sym[W[i, 1]], "right" if h2 else "WRONG"))
        if asg is not None:
            def c(p):
                v = asg[i, p]
                return str(tuple(v.tolist())) if np.ndim(v) else str(int(v))
            for name, p in (("query (hex)", s["q"]), ("level-0 role (hex)", r0), ("level-0 filler (decimal)", f0),
                            ("level-1 role, same symbol (hex)", r1), ("level-1 filler (decimal)", f1),
                            ("candidate (decimal)", s["cand"])):
                print("    %-34s %-8r code %s" % (name, s["text"][s["mentions"][p][3]:s["mentions"][p][4]], c(p)))


if __name__ == "__main__":
    main()
