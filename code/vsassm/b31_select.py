"""B31 selection rules (prereg/B31_encoder_equalised_baselines_prereg.md); train + dev outputs only.

  nce   per encoder and arena: the (lambda, lr) whose best dev accuracy (primary reading perm/named) is highest; ties
        within 0.002 go to the smaller lambda, then lr 3e-4.
  bind  per encoder and arena: the lr with the highest best dev accuracy; ties within 0.002 to lr 3e-4.
  lora  (stage 1, and stage 2 = LoRA + identity loss, separately) per encoder and arena: among the configurations that fit (accuracy >= 0.95 on the fixed 500-instance
        training subset at some epoch), the one with the highest dev accuracy at its best epoch; if none fits, the
        highest dev accuracy overall. Ties within 0.004 (2 of 500): the smaller rank, then the smaller rate.
  stage 3 the 24-epoch run of the configuration closest to fitting (lora_tune3/): selected if it fits.
  probe the selection is made inside b31_probe.py (tune mode) and copied here.

  python b31_select.py --root /path/to/work/b31/out --out /path/to/work/b31/out/select.json
"""
import argparse, glob, json, os

PRIMARY = "perm/named"


def best_dev_arms(d):
    accs = [e["%s:dev" % PRIMARY]["acc"] for e in d["runs"][0]["epochs"]]
    i = max(range(len(accs)), key=lambda j: (accs[j], -j))
    return accs[i], i + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/path/to/work/b31/out")
    ap.add_argument("--out", default="/path/to/work/b31/out/select.json")
    a = ap.parse_args()
    sel = {"bind": {}, "nce": {}, "lora": {}, "loraid": {}, "probe": {}, "stage3": {},
           "grids": {"bind": {}, "nce": {}, "lora": {}, "loraid": {}}}
    for enc in ("phi2", "qwen8"):
        for ar in "ABC":
            cands = []
            for p in sorted(glob.glob(os.path.join(a.root, "arms_tune", "%s_%s_nce_*.json" % (enc, ar)))):
                d = json.load(open(p))
                acc, ep = best_dev_arms(d)
                cands.append({"lam": d["args"]["lam"], "lr": d["args"]["lr"], "dev": acc, "epoch": ep,
                              "file": os.path.basename(p)})
            if cands:
                top = max(c["dev"] for c in cands)
                tied = [c for c in cands if c["dev"] >= top - 0.002]
                tied.sort(key=lambda c: (c["lam"], abs(c["lr"] - 3e-4)))
                sel["nce"]["%s_%s" % (enc, ar)] = tied[0]
                sel["grids"]["nce"]["%s_%s" % (enc, ar)] = cands
            bc = []
            for p in sorted(glob.glob(os.path.join(a.root, "arms_tune", "%s_%s_bind_*.json" % (enc, ar)))):
                d = json.load(open(p))
                acc, ep = best_dev_arms(d)
                bc.append({"lr": d["args"]["lr"], "dev": acc, "epoch": ep, "file": os.path.basename(p)})
            if bc:
                top = max(c["dev"] for c in bc)
                sel["bind"]["%s_%s" % (enc, ar)] = sorted([c for c in bc if c["dev"] >= top - 0.002],
                                                          key=lambda c: abs(c["lr"] - 3e-4))[0]
                sel["grids"]["bind"]["%s_%s" % (enc, ar)] = bc
            for idl, key in ((0, "lora"), (1, "loraid")):
                lc = lora_grid(a.root, "lora_tune", enc, ar, idl)
                if lc:
                    pool = [c for c in lc if c["fit"]] or lc
                    top = max(c["dev"] for c in pool)
                    tied = sorted([c for c in pool if c["dev"] >= top - 0.004], key=lambda c: (c["rank"], c["lr"]))
                    s = dict(tied[0]); s["any_fit"] = any(c["fit"] for c in lc)
                    sel[key]["%s_%s" % (enc, ar)] = s
                    sel["grids"][key]["%s_%s" % (enc, ar)] = lc
            s3 = lora_grid(a.root, "lora_tune3", enc, ar, 0)
            if s3:
                sel["stage3"]["%s_%s" % (enc, ar)] = s3[0]
                if s3[0]["fit"]:                       # stage 3 fits: it becomes the selected configuration
                    sel["lora"]["%s_%s" % (enc, ar)] = dict(s3[0], any_fit=True, stage=3)
            pp = os.path.join(a.root, "probe_tune", "%s_%s.json" % (enc, ar))
            if os.path.exists(pp):
                sel["probe"]["%s_%s" % (enc, ar)] = json.load(open(pp))["selected"]
    json.dump(sel, open(a.out, "w"), indent=1)
    for k in ("bind", "nce", "probe", "lora", "loraid", "stage3"):
        for kk, v in sel[k].items():
            print(k, kk, json.dumps(v))


def lora_grid(root, sub, enc, ar, idl):
    lc = []
    for p in sorted(glob.glob(os.path.join(root, sub, "%s_%s_r*_lr*_id%d.json" % (enc, ar, idl)))):
        d = json.load(open(p))
        r = d["runs"][0]
        devs = [e["dev"]["acc"] for e in r["epochs"]]
        trs = [e["train_sub"]["acc"] for e in r["epochs"]]
        lc.append({"rank": d["args"]["rank"], "lr": d["args"]["lr"], "epochs": d["args"]["epochs"], "dev": max(devs),
                   "epoch": devs.index(max(devs)) + 1, "train_max": max(trs), "fit": max(trs) >= 0.95,
                   "file": os.path.basename(p)})
    return lc


if __name__ == "__main__":
    main()
