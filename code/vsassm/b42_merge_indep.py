"""B42: merge the second scorer's shards (out/indep/*.json) into out/indep42.json, and (compute note 2) check that each
rerun in out/final_rerun/ reproduces its original final in out/final/ (every seed, every epoch: test scores, dev and
test accuracy). For the rerun arms the shard scored on the rerun's side files replaces the one scored on the original's.

  python b42_merge_indep.py --root out
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    a = ap.parse_args()
    res = {}
    for p in sorted(glob.glob(os.path.join(a.root, "indep", "*.json"))):
        if os.path.basename(p).startswith("rerun_"):
            continue
        res.update(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(a.root, "indep", "rerun_*.json"))):
        for k, v in json.load(open(p)).items():
            v = dict(v); v["scored_on"] = "rerun side file (compute note 2)"
            res[k] = v
    for p in sorted(glob.glob(os.path.join(a.root, "final_rerun", "*.json"))):
        name = os.path.basename(p)
        o = json.load(open(os.path.join(a.root, "final", name)))
        r = json.load(open(p))
        diff, same_acc = 0.0, True
        for ro, rr in zip(o["runs"], r["runs"]):
            for eo, er in zip(ro["epochs"], rr["epochs"]):
                for k in eo:
                    if k.endswith(":test") or k.endswith(":dev"):
                        same_acc &= eo[k]["acc"] == er[k]["acc"]
                        if "scores" in eo[k]:
                            diff = max(diff, max(abs(x - y) for x, y in zip(eo[k]["scores"], er[k]["scores"])))
        res["rerun reproduces %s" % name[:-5]] = {"identical_dev_test_acc_every_epoch": same_acc,
                                                  "max_abs_test_score_diff": diff,
                                                  "same_dev_selected_epochs": [x["dev_selected_epoch"] for x in o["runs"]] ==
                                                  [x["dev_selected_epoch"] for x in r["runs"]]}
        print(name, res["rerun reproduces %s" % name[:-5]])
    json.dump(res, open(os.path.join(a.root, "indep42.json"), "w"), indent=1)
    print(len(res), "entries")


if __name__ == "__main__":
    main()
