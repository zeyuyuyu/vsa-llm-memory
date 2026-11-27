"""B44 hand-reading (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md): prints test failures of a saved output
with the gold chain read from the raw text, for reading by hand.

  python b44_handread.py --kind lora   --file b44_out/lora/cls_std_n80000_r8_lr1e-4_s42.json.test.json --k 12
  python b44_handread.py --kind scratch --file b44_out/lora/scratch_std_n8000_..._s42.json.test.json
  python b44_handread.py --kind cot    --file b44_out/cot/zs-think_std_test.json
  python b44_handread.py --kind name   --file b44_out/name/qwen8_name_std_n800.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b44_data as BD                                                       # noqa: E402
import b44_indep as I                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["lora", "scratch", "cot", "name"])
    ap.add_argument("--file", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--tail", type=int, default=700)
    a = ap.parse_args()
    d = json.load(open(a.file))
    v = a.variant or d["args"].get("variant") or d["args"].get("cell", "std:").split(":")[0]
    st = BD.split(v, "test")
    gold = [I.walk(s["text"]) for s in st]
    fails = []
    if a.kind == "lora":
        for i, s_ in enumerate(d["test"]["scores"]):
            if int(s_ > 0) != gold[i][0]:
                fails.append((i, "margin %+.3f" % s_))
    elif a.kind == "scratch":
        import b44_lora as BL
        for i, g in enumerate(d["test"]["gens"]):
            lab = BL.parse_scratch(g)[0]
            if lab != gold[i][0]:
                fails.append((i, "generated %r" % g))
    elif a.kind == "cot":
        for r in d["rows"]:
            if r["pred"] != gold[r["i"]][0]:
                fails.append((r["i"], "pred %s (%s), %d new tokens; reply tail:\n%s" % (
                    r["pred"], r["parse"], r["new_tokens"], r["reply"][-a.tail:])))
    else:
        r = d["runs"][0]
        devs = [e["perm/named:dev"]["acc"] for e in r["epochs"]]
        j = max(range(len(devs)), key=lambda x: (devs[x], -x))
        te = r["epochs"][j]["perm/named:test"]
        for i, s_ in enumerate(te["scores"]):
            if int(s_ >= te["thr"]) != gold[i][0]:
                fails.append((i, "cos %.3f thr %.3f; misnamed mentions (pos, predicted symbol): %s; hop winners %s" % (
                    s_, te["thr"], te["misnamed"][i], te["wins"][i])))
    print("%s: %d failures of %d; showing %d" % (a.file, len(fails), len(st), min(a.k, len(fails))))
    for i, msg in fails[:a.k]:
        s = st[i]; lab, v1, w = gold[i]
        c = s["chain"]; cand = s["mentions"][s["cand"]][2]
        print("-" * 100)
        print("test #%d  label %d  query R%s -> bridge %d (= F%s) -> answer %d ; candidate %d" % (
            i, lab, BD.h2(c[0]), v1, BD.h2(v1), w, cand))
        # where the chain pairs sit in the text, and confusable pairs (same filler as the bridge, alias roles)
        lv0, lv1 = s["levels"]
        pos0 = [p for p, (a_, b_) in enumerate(lv0) if a_ == c[0]][0]
        pos1 = [p for p, (a_, b_) in enumerate(lv1) if a_ == v1][0]
        dup_b = [BD.h2(a_) for a_, b_ in lv0 if b_ == v1 and a_ != c[0]]
        print("  level-0 pair at position %d/32, level-1 pair at %d/32; other level-0 roles with the same filler %s; "
              "candidate also a level-1 filler: %s" % (pos0, pos1, dup_b, cand in [b_ for _, b_ in lv1]))
        print("  " + msg)


if __name__ == "__main__":
    main()
