"""B44 selection (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md); reads train + dev outputs only.

LoRA arms (lora = --task cls, scratch = --task scratch): per arm, variant and n, among the seed-42 runs of the grid
(rank 8, lr 1e-4 / 3e-4), the highest dev accuracy at the best evaluation point; ties within 0.004 to the smaller rate.
Where the prereg fixes the rate from another cell (lora ho and scratch std at 80,000 <- the same arm and variant at
8,000; scratch ho <- scratch std at the same n), the rule is applied to that cell and the rate copied.
CoT: the arm with the highest accuracy on the first 100 std dev instances; ties: thinking off before on, then zero-shot
before few-shot.

  python b44_select.py --root b44_out --out b44_out/select.json
"""
import argparse, glob, json, os

ORDER = ["zs-nothink", "fs-nothink", "zs-think", "fs-think"]          # tie-break order for CoT


def lora_cells(root):
    cells = {}
    for p in glob.glob(os.path.join(root, "lora", "*_s42.json")):
        d = json.load(open(p))
        a = d["args"]
        key = "%s_%s_n%d" % ("lora" if a["task"] == "cls" else "scratch", a["variant"], a["n"])
        r = d["run"]
        best = max(e["dev"]["acc"] for e in r["evals"])
        tr = max(e["train_sub"]["acc"] for e in r["evals"])
        cells.setdefault(key, []).append({"lr": a["lr"], "rank": a["rank"], "dev": best, "best_step": r["best_step"],
                                          "train_sub_max": tr, "file": os.path.basename(p),
                                          "early_stop": r.get("early_stop")})
    return cells


def pick(cands):
    top = max(c["dev"] for c in cands)
    return sorted([c for c in cands if c["dev"] >= top - 0.004], key=lambda c: c["lr"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="b44_out")
    ap.add_argument("--out", default="b44_out/select.json")
    a = ap.parse_args()
    cells = lora_cells(a.root)
    sel = {"grids": cells, "lora": {}}
    for key, cands in sorted(cells.items()):
        if len(cands) >= 2:
            sel["lora"][key] = dict(pick(cands), rule="grid")
    for arm, v in (("lora", "ho"), ("scratch", "std")):
        k8 = "%s_%s_n8000" % (arm, v)
        if k8 in sel["lora"]:
            sel["lora"]["%s_%s_n80000" % (arm, v)] = {"lr": sel["lora"][k8]["lr"], "rank": 8, "rule": "from " + k8}
    for n in (800, 8000, 80000):
        ks = "scratch_std_n%d" % n
        if ks in sel["lora"]:
            sel["lora"]["scratch_ho_n%d" % n] = {"lr": sel["lora"][ks]["lr"], "rank": 8, "rule": "from " + ks}
    cot = {}
    for p in glob.glob(os.path.join(a.root, "cot", "*_std_dev100.json")):
        d = json.load(open(p))
        cot[d["args"]["arm"]] = {"acc": d["acc"], "unparsed": d["unparsed"], "mean_new_tokens": d["mean_new_tokens"]}
    if len(cot) == 4:
        top = max(v["acc"] for v in cot.values())
        sel["cot"] = {"selected": [k for k in ORDER if cot[k]["acc"] >= top - 1e-9][0], "dev100": cot}
    json.dump(sel, open(a.out, "w"), indent=1)
    for k, v in sorted(sel["lora"].items()):
        print(k, json.dumps(v))
    if "cot" in sel:
        print("cot", json.dumps(sel["cot"]))


if __name__ == "__main__":
    main()
