"""B31 hand-reading helper (prereg/B31_encoder_equalised_baselines_prereg.md): prints, for a given encoder and arena,
test instances where the naming head (perm/named, dev-selected epoch, seed 42) is wrong, and instances where it and
the equalised LoRA (seed 42) disagree, with the raw text, the gold chain, the head's names for the mentions on the
chain, the memory's clean-up winners and both decisions. Reading only; it changes no number.

  python b31_handread.py --enc qwen8 --arena C --n 10 > out/handread_qwen8_C.txt
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402
from b31_arms import layout                                                 # noqa: E402

PRIMARY = "perm/named"


def sel_epoch(run):
    accs = [e["%s:dev" % PRIMARY]["acc"] for e in run["epochs"]]
    return max(range(len(accs)), key=lambda j: (accs[j], -j))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--root", default="/path/to/work/b31/out")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    st = S.build(a.arena, "test")
    y = np.array([s["label"] for s in st])
    nm = json.load(open(os.path.join(a.root, "arms_final", "%s_%s_name.json" % (a.enc, a.arena))))["runs"][0]
    e = nm["epochs"][sel_epoch(nm)]["%s:test" % PRIMARY]
    pm = (np.array(e["scores"]) >= e["thr"]).astype(int)
    lp = os.path.join(a.root, "lora_final", "%s_%s_id0.json" % (a.enc, a.arena))
    pl = None
    if os.path.exists(lp):
        lr = json.load(open(lp))["runs"][0]
        pl = (np.array(lr["test_scores"]) > 0).astype(int)
    kinds, hops, q, c, _ = layout(a.arena)
    wrong = [i for i in range(len(st)) if pm[i] != y[i]]
    dis = [i for i in range(len(st)) if pl is not None and pm[i] != pl[i]]
    print("%s arena %s: naming head wrong on %d of %d; naming head vs LoRA disagree on %d" % (
        a.enc, a.arena, len(wrong), len(st), len(dis)))
    for title, idx in (("NAMING HEAD WRONG", wrong[:a.n]), ("DISAGREE WITH LoRA", [i for i in dis if i not in wrong][:a.n])):
        print("\n=== %s ===" % title)
        for i in idx:
            s = st[i]
            sym = [m[2] for m in s["mentions"]]
            ans = S.dict_answer(a.arena, s)[1]
            print("\n#%d label %d (dictionary answer %d, candidate %d) memory says %d%s" % (
                i, s["label"], ans, sym[c], pm[i], "" if pl is None else ", LoRA says %d" % pl[i]))
            print("  text:", s["text"][:600] + (" ..." if len(s["text"]) > 600 else ""))
            print("  clean-up winners (mention positions):", e["wins"][i], "gold symbols", [sym[w] for w in e["wins"][i]])
            print("  all mentions named correctly:", bool(e["naming_all_ok"][i]))
            for j, pr in e.get("misnamed", [[]] * len(st))[i]:
                m = s["mentions"][j]
                tag = "QUERY" if j == q else ("CANDIDATE" if j == c else "context")
                print("    misnamed %s mention %d %r (kind %s, gold %d) named %d" % (tag, j, s["text"][m[3]:m[4]], m[1],
                                                                                   m[2], pr))


if __name__ == "__main__":
    main()
