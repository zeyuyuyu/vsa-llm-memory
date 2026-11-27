"""B48 queue files (prereg/B48_one_form_and_alias_naming_prereg.md): plain command lines in q/<name>, run by
b48_pace.sbatch (or b48_worker.sh on the A800 box), relative to the B48 work directory.

  python b48_queues.py tune    part A gates (G0, G1), binding-loss lr, probe grid and the seven B42 grids on C1 (seed
                               42, train + dev); part B naming-head grid (seed 42, train + dev)
  python b48_queues.py final   after b48_select.py / b48_alias.py select: part A finals (3 seeds, test) and part B
                               naming-head finals (3 seeds, test)
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b42_queues import FAMS, grid, cfg_name                                 # noqa: E402  (B42's registered grids)

SEEDS = "42 123 456"
ENCS = ["phi2", "qwen8"]
BINDLR = {"phi2": 3e-4, "qwen8": 1e-3}         # B42's arena-C rate (B31 dev-selected): the cw grid and gates G0
R = "b48_run.py "


def c42(enc, fam, mode, c, out, seeds="42"):
    s = R + "b42_arms.py --enc_model %s --arena C --fam %s --mode %s --seeds %s --feats feats" % (enc, fam, mode, seeds)
    for k, v in sorted(c.items()):
        s += " --%s %s" % (k, ("%g" % v) if isinstance(v, float) else v)
    return s + " --out " + out


def w(name, lines):
    os.makedirs("q", exist_ok=True)
    open(os.path.join("q", name), "w").write("".join(l + "\n" for l in lines))
    print(name, len(lines))


def tune():
    L = []
    for enc in ENCS:
        L.append(R + "b31_arms.py --enc_model %s --arena C --arm bind --mode tune --lr %g --epochs 10 --feats feats "
                 "--out out/g0/%s_C1_bind_b31.json" % (enc, BINDLR[enc], enc))
        for lr in (3e-4, 1e-3):
            L.append(c42(enc, "bind", "tune", {"lr": lr}, "out/bindtune/%s_C1_bind_lr%g.json" % (enc, lr)))
        for fam in ("gumbel", "st"):
            L.append(c42(enc, fam, "tune", {"lr": 3e-4, "idlam": 1.0, "cleanup": "hard"},
                         "out/g1/%s_C1_%s_g1.json" % (enc, fam)))
        L.append(R + "b31_probe.py --enc %s --arena C --mode tune --feats feats --out out/probe_tune/%s_C1.json"
                 % (enc, enc))
        L.append(R + "b31_arms.py --enc_model %s --arena C --arm name --mode tune --epochs 4 --feats feats "
                 "--out out/nametune/%s_C1_name.json" % (enc, enc))          # the naming head's dev (gate G1)
        for fam in FAMS:
            for c in grid(fam, enc, "C"):
                if fam == "cw":
                    c = dict(c, lr=BINDLR[enc])
                L.append(c42(enc, fam, "tune", c, "out/tune/%s_C1_%s.json" % (enc, cfg_name(fam, c))))
    w("tuneA", L)
    B = ["b48_alias.py tune --kind %s --layer %d --lr %g --feats feats --out out/b/tune/%s_L%d_lr%g.json"
         % (k, l, lr, k, l, lr) for k in "WP" for l in (9, 18, 27, 36) for lr in (3e-4, 1e-3)]
    w("tuneB", B)


def final():
    sel = json.load(open("out/select.json"))
    L = []
    for enc in ENCS:
        k = "%s_C1" % enc
        for fam in FAMS:
            L.append(c42(enc, fam, "final", sel[fam][k]["cfg"], "out/final/%s_%s.json" % (k, fam), SEEDS))
        L.append(c42(enc, "bind", "final", {"lr": sel["bind"][k]["lr"]}, "out/final/%s_bind.json" % k, SEEDS))
        L.append(R + "b31_arms.py --enc_model %s --arena C --arm name --mode final --epochs 4 --readings perm/index "
                 "comm/named comm/index --seeds %s --feats feats --out out/final/%s_name.json" % (enc, SEEDS, k))
        json.dump(sel["probe"][k], open("out/probe_sel_%s.json" % k, "w"))
        L.append(R + "b31_probe.py --enc %s --arena C --mode final --select out/probe_sel_%s.json --feats feats "
                 "--out out/final/%s_probe.json" % (enc, k, k))
    L.append(R + "b31_arms.py --enc_model phi2 --arena C --arm exact --mode final --enc perm --readings comm/named "
             "--seeds %s --feats feats --out out/final/X_C1_exact.json" % SEEDS)
    w("finalA", L)
    w("finalB", ["b48_alias.py final --kind %s --select out/b/select.json --feats feats --out out/b/final/%s_head.json"
                 % (k, k) for k in "WP"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "final"])
    a = ap.parse_args()
    {"tune": tune, "final": final}[a.phase]()
