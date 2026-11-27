"""B42 queue files (prereg/B42_label_free_discrete_prereg.md): plain command lines for b42_worker.sh in q/<name>.

  python b42_queues.py tune  --arenas C       gates G0 / G1 and the tuning grids (seed 42, train + dev only)
  python b42_queues.py final --arenas C       after b42_select.py: the selected configurations, seeds 42 123 456,
                                              plus the fixed arms (bind with post-hoc readings; dkvbor diagnostic)
"""
import argparse, json, os

SEEDS = "42 123 456"
ENCS = ["phi2", "qwen8"]
BINDLR = {"phi2_A": 3e-4, "phi2_B": 3e-4, "phi2_C": 3e-4, "qwen8_A": 1e-3, "qwen8_B": 3e-4, "qwen8_C": 1e-3}  # B31 dev


def grid(fam, enc, ar):
    """Registered grids, in tie-break order (earlier wins ties within 0.002 dev)."""
    if fam == "vq":
        return [dict(cleanup=c, mult=m, inw=i, lr=lr) for c in ("hard", "st") for m in (1, 4) for i in (0, 1)
                for lr in (3e-4, 1e-3)]
    if fam == "dkvb":
        return [dict(cleanup=c, mult=m, heads=h, lr=1e-3) for c in ("hard", "st") for m in (1, 4) for h in (1, 8)]
    if fam in ("gumbel", "st"):
        return [dict(cleanup=c, div=d, lr=lr) for c in ("hard", "st") for d in (0.0, 0.1, 1.0) for lr in (3e-4, 1e-3)]
    if fam == "cw":
        return [dict(cleanup=c, inw=i, code=k, lr=BINDLR["%s_%s" % (enc, ar)]) for c in ("hard", "st") for i in (0, 1)
                for k in ("center", "inst", "zca")]
    raise KeyError(fam)


FAMS = ["vq", "dkvb", "gumbel", "st", "cw"]


def cfg_name(fam, c):
    return fam + "_" + "_".join("%s%s" % (k, ("%g" % v) if isinstance(v, float) else v) for k, v in sorted(c.items()))


def cmd(enc, ar, fam, mode, c, out, seeds="42"):
    s = "b42_arms.py --enc_model %s --arena %s --fam %s --mode %s --seeds %s" % (enc, ar, fam, mode, seeds)
    for k, v in sorted(c.items()):
        s += " --%s %s" % (k, ("%g" % v) if isinstance(v, float) else v)
    return s + " --out " + out


def w(name, lines):
    os.makedirs("q", exist_ok=True)
    open(os.path.join("q", name), "w").write("".join(l + "\n" for l in lines))
    print(name, len(lines))


def tune(arenas):
    L = []
    for ar in arenas:
        for enc in ENCS:
            k = "%s_%s" % (enc, ar)
            L.append(cmd(enc, ar, "bind", "tune", {"lr": BINDLR[k]}, "out/g0/%s_bind.json" % k))
            for fam in ("gumbel", "st"):           # G1: the labelled positive control (identity loss on)
                L.append(cmd(enc, ar, fam, "tune", {"lr": 3e-4, "idlam": 1.0, "cleanup": "hard"},
                             "out/g1/%s_%s_g1.json" % (k, fam)))
            for fam in FAMS:
                for c in grid(fam, enc, ar):
                    L.append(cmd(enc, ar, fam, "tune", c, "out/tune/%s_%s.json" % (k, cfg_name(fam, c))))
    w("tune_" + "".join(arenas), L)


def final(arenas):
    sel = json.load(open("out/select.json"))
    L = []
    for ar in arenas:
        for enc in ENCS:
            k = "%s_%s" % (enc, ar)
            for fam in FAMS:
                c = sel[fam][k]["cfg"]
                L.append(cmd(enc, ar, fam, "final", c, "out/final/%s_%s.json" % (k, fam), SEEDS))
            L.append(cmd(enc, ar, "bind", "final", {"lr": BINDLR[k]}, "out/final/%s_bind.json" % k, SEEDS))
            if ar == "C":
                for cl in ("hard", "st"):
                    L.append(cmd(enc, ar, "dkvbor", "final", {"lr": 1e-3, "cleanup": cl},
                                 "out/final/%s_dkvbor_%s.json" % (k, cl), SEEDS))
    w("final_" + "".join(arenas), L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "final"])
    ap.add_argument("--arenas", default="C")
    a = ap.parse_args()
    {"tune": tune, "final": final}[a.phase](list(a.arenas))
