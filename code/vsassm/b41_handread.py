"""B41 hand-reading helper: prints failures of chosen arms in one engine run with the mention texts, their top-3
readings, the chain, the decoy flag and the arm's answer, for the notes in results/b41_handread_notes.txt.

  python b41_handread.py --run <engine .npz> --logits <split>_logits.npy --arms ARGMAX EXACT ... --h 6 --n 15
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_names as NM                                                      # noqa: E402
import b41_data as DA                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    meta = json.load(open(a.run.replace(".npz", ".json")))
    lmeta = json.load(open(a.logits.replace("_logits.npy", "_logits_meta.json")))
    d = DA.make(meta["seed"], meta["N"], meta["nq"], meta["split"])
    names = NM.entity_names()
    rows = NM.bank(meta["split"], rhos=lmeta["rhos"], names=names)
    idx = DA.row_index(meta["split"], lmeta["rhos"])
    lg = np.load(a.logits, mmap_mode="r")
    Z = np.load(a.run)
    hi = (2, 4, 6).index(a.h)
    rho = meta["rho"]
    slot = {}
    for f in range(len(d["f_s"])):
        slot.setdefault((int(d["f_ctx"][f]), int(d["f_rel"][f])), []).append(f)

    def reading(e, j):
        r = idx(int(e), rho, int(j))
        x = lg[r].astype(np.float64); p = np.exp(x - x.max()); p /= p.sum()
        top = np.argsort(-p)[:3]
        return "%r -> %s" % (rows[r][3], ", ".join("%s%s %.2f" % (names[t], "*" if t == e else "", p[t])
                                                   for t in top))
    for arm in a.arms:
        cor = Z["%s|correct" % arm][:, hi]
        ans = Z["%s|answer" % arm][:, hi]
        fails = np.nonzero(~cor)[0][:a.n]
        print("=" * 100)
        print("%s: %d failures of %d at h=%d (N=%d, rho=%.3f, seed %d)" % (arm, int((~cor).sum()), len(cor), a.h,
                                                                           meta["N"], rho, meta["seed"]))
        for q in fails:
            ents, rels, c = d["q_ents"][q], d["q_rels"][q], int(d["q_ctx"][q])
            print("-- q%d decoy=%s answer=%s got=%s" % (q, bool(d["q_decoy"][q]), names[ents[a.h]],
                                                        names[ans[q]] if ans[q] >= 0 else "DEAD"))
            print("   start  %s" % reading(ents[0], d["q_j0"][q]))
            for t in range(a.h):
                fl = slot.get((c, int(rels[t])), [])
                tf = [f for f in fl if d["f_s"][f] == ents[t] and d["f_o"][f] == ents[t + 1]]
                if tf:
                    f = tf[0]
                    print("   hop%d %-10s s %s" % (t + 1, NM.RELATIONS[rels[t]], reading(d["f_s"][f], d["f_js"][f])))
                    print("   %-15s o %s" % ("", reading(d["f_o"][f], d["f_jo"][f])))
                print("   %-15s (%d facts with this relation in context)" % ("", len(fl)))


if __name__ == "__main__":
    main()
