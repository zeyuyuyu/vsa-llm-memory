"""B48 hand-reading helper (prereg/B48_one_form_and_alias_naming_prereg.md). Reading only; it changes no number.

  A  C1: the label-free family with the highest selected dev accuracy for an encoder, seed 42 at its dev-selected
     epoch: wrong test instances with the gold chain, the clean-up winners, which hop failed and (discrete families)
     the code of the query, its context role, the level-0 filler, the level-1 role of the same symbol (now the same
     written form), the level-1 filler and the candidate.
  B  alias stories: hard mentions (W alias / P pronoun) misnamed by the head (seed 42) and by the lm, and random
     correctly named ones, each shown with its line, the gold listed name and what each arm chose / wrote.

  python b48_handread.py A --enc qwen8 --n 12 > out/handread_A_qwen8.txt
  python b48_handread.py B --kind W --n 20 > out/handread_B_W.txt
"""
import argparse, json, os, random, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b48_struct as S48                                                    # noqa: E402

PRIMARY = "perm/named"
LF = ["vq", "dkvb", "gumbel", "st", "cw"]


def part_a(a):
    sel = json.load(open(os.path.join(a.root, "select.json")))
    k = "%s_C1" % a.enc
    fam = a.fam or max(LF, key=lambda f: sel[f][k]["dev"])
    d = json.load(open(os.path.join(a.root, "final", "%s_%s.json" % (k, fam))))
    run = d["runs"][0]
    j = run["dev_selected_epoch"] - 1
    e = run["epochs"][j]["%s:test" % PRIMARY]
    side_p = os.path.join(a.root, "final", "%s_%s.side.pt" % (k, fam))
    asg = torch.load(side_p, weights_only=False)[run["seed"]]["assign_test"].numpy() if os.path.exists(side_p) else None
    st = S48.build_C1("test")
    y = np.array([s["label"] for s in st])
    pred = (np.array(e["scores"]) >= e["thr"]).astype(int)
    W = np.array(e["wins"])
    wrong = [i for i in range(len(st)) if pred[i] != y[i]]
    print("%s arena C1, family %s (selected dev %.3f, config %s), seed %d epoch %d: wrong on %d of %d" % (
        a.enc, fam, sel[fam][k]["dev"], json.dumps(sel[fam][k]["cfg"]), run["seed"], j + 1, len(wrong), len(st)))
    n1 = n2 = 0
    for i in wrong:
        s = st[i]; sym = [m[2] for m in s["mentions"]]; ch = s["chain"]
        n1 += sym[W[i, 0]] != ch[1]; n2 += sym[W[i, 0]] == ch[1] and sym[W[i, 1]] != ch[2]
    print("failures: hop-1 clean-up wrong in %d, hop 1 right but hop 2 wrong in %d, both right in %d" % (
        n1, n2, len(wrong) - n1 - n2))
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
        print("  hop 1 winner symbol %d (%s), hop 2 winner symbol %d (%s)" % (
            sym[W[i, 0]], "right" if h1 else "WRONG", sym[W[i, 1]], "right" if h2 else "WRONG"))
        if asg is not None:
            def c(p):
                v = asg[i, p]
                return str(tuple(v.tolist())) if np.ndim(v) else str(int(v))
            for name, p in (("query", s["q"]), ("level-0 role", r0), ("level-0 filler", f0),
                            ("level-1 role, same symbol", r1), ("level-1 filler", f1), ("candidate", s["cand"])):
                ctx = s["text"][max(0, s["mentions"][p][3] - 3):s["mentions"][p][4] + 2]
                print("    %-27s %-10r code %s" % (name, ctx, c(p)))


def part_b(a):
    st = S48.build_B(a.kind, "test")
    d = json.load(open(os.path.join(a.root, "b", "scores_%s_test.json" % a.kind)))
    hd = json.load(open(os.path.join(a.root, "b", "final", "%s_head.json" % a.kind)))
    run = [r for r in hd["runs"] if r["seed"] == 42][0]
    pred = run["epochs"][run["dev_selected_epoch"] - 1]["test"]["pred"]
    lw = d.get("lm_written")
    hard = "alias" if a.kind == "W" else "pron"
    rows = []
    for i, s in enumerate(st):
        for mi, (kd, e, cs, ce, mt) in enumerate(s["mentions"]):
            if mt != hard:
                continue
            ls = s["text"].rfind("\n", 0, cs) + 1
            line = s["text"][ls:s["text"].find("\n", ce)]
            gold = s["ents"][e]["name"]
            hpick = s["ents"][pred[i][mi]]["name"]
            lpick = lw[i].get(str(mi)) if lw else None
            rows.append((i, mi, line, s["text"][cs:ce], gold, hpick, lpick))
    norm = S48.norm
    hw = [r for r in rows if r[5] != r[4]]
    lwr = [r for r in rows if lw and (r[6] is None or norm(r[6]) != norm(r[4]))]
    print("%s test: %d %s mentions; head (seed 42) wrong on %d, lm wrong on %d; both wrong on %d" % (
        a.kind, len(rows), hard, len(hw), len(lwr), len(set((r[0], r[1]) for r in hw) & set((r[0], r[1]) for r in lwr))))
    rng = random.Random(0)
    for title, sub in (("HEAD WRONG", hw), ("LM WRONG", lwr),
                       ("HEAD RIGHT (random)", [r for r in rows if r not in hw]),
                       ("LM RIGHT (random)", [r for r in rows if r not in lwr])):
        pick = rng.sample(sub, min(a.n if "WRONG" in title else 10, len(sub)))
        print("\n## %s (%d shown of %d)" % (title, len(pick), len(sub)))
        for i, mi, line, surf, gold, hp, lp in pick:
            print("  story %d: %r\n     written %r -> gold %r | head %r | lm %r" % (i, line, surf, gold, hp, lp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["A", "B"])
    ap.add_argument("--enc", default="qwen8")
    ap.add_argument("--fam", default=None)
    ap.add_argument("--kind", default="W")
    ap.add_argument("--root", default="out")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    part_a(a) if a.part == "A" else part_b(a)


if __name__ == "__main__":
    main()
