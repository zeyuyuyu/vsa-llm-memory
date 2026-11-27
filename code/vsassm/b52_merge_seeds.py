"""B52 compute note 3: merge the per-seed final runs (out/final_seed/<cell>_<fam>_s<seed>.json and .side.pt) into the
B42 layout (out/final/<cell>_<fam>.json with runs in seed order 42, 123, 456; side file keyed by seed). No number changes.

  python b52_merge_seeds.py --root out --cell qwen8L27_C
"""
import argparse, json, os
import torch

SEEDS = [42, 123, 456]
FAMS = ["vq", "dkvb", "gumbel", "st", "cw", "bind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--cell", required=True)
    a = ap.parse_args()
    for fam in FAMS:
        ps = [os.path.join(a.root, "final_seed", "%s_%s_s%d.json" % (a.cell, fam, s)) for s in SEEDS]
        if not all(os.path.exists(p) for p in ps):
            print("incomplete", fam, [os.path.exists(p) for p in ps]); continue
        ds = [json.load(open(p)) for p in ps]
        out = dict(ds[0])
        out["runs"] = [d["runs"][0] for d in ds]
        assert [r["seed"] for r in out["runs"]] == SEEDS, [r["seed"] for r in out["runs"]]
        out["args"] = dict(ds[0]["args"], seeds=SEEDS, out="out/final/%s_%s.json" % (a.cell, fam))
        out["merged_from"] = ps
        dst = os.path.join(a.root, "final", "%s_%s.json" % (a.cell, fam))
        json.dump(out, open(dst, "w"))
        sides = [p.replace(".json", ".side.pt") for p in ps]
        if all(os.path.exists(p) for p in sides):
            side = {}
            for p in sides:
                side.update(torch.load(p, weights_only=False))
            torch.save(side, dst.replace(".json", ".side.pt"))
        print("merged", fam, [r["seed"] for r in out["runs"]])


if __name__ == "__main__":
    main()
