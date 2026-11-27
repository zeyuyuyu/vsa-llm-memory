"""B44 second scorer (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md), written separately from the main code:
labels from its own regex walk of the raw arena text (not b44_data.dict_answer, not the generator's records), the
held-out property re-checked from the raw texts, and every saved test output re-scored:
  lora / scratch  <out>.test.json: margins (sign) / generations (its own parser: the last word of the reply that is
                  'yes' or 'no' after stripping punctuation)
  name / exact    the saved per-epoch test scores and training thresholds of b31_arms.run, at the dev-best epoch
  probe           saved test margins (sign)
  cot             saved replies (its own parser: the text after the last '</think>', the last line containing 'yes'
                  or 'no' as a word, whichever of the two comes last on that line)

  python b44_indep.py --root b44_out --out b44_out/indep.json
"""
import argparse, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAIR = re.compile(r"\[([RF])([0-9A-F]{2})\]\{(\d+)\}")
QRY = re.compile(r"Q \[R([0-9A-F]{2})\] \{(\d+)\} \?")


def walk(text):
    tabs = {"R": {}, "F": {}}
    for t, a, b in PAIR.findall(text):
        tabs[t][int(a, 16)] = int(b)
    q, cand = QRY.search(text).groups()
    v = tabs["R"][int(q, 16)]
    w = tabs["F"][v]
    return int(w == int(cand)), v, w


def pairs(text):
    return {(int(a, 16), int(b)) for _, a, b in PAIR.findall(text)}


def yn_last_word(txt):
    words = [w.strip(".,:;!?*()[]{}\"'").lower() for w in txt.split()]
    words = [w for w in words if w in ("yes", "no")]
    return None if not words else int(words[-1] == "yes")


def cot_parse(reply):
    if "<think>" in reply and "</think>" not in reply:
        return None
    final = reply.rsplit("</think>", 1)[-1]
    for line in reversed([l for l in final.splitlines() if l.strip()]):
        ws = re.findall(r"\b(yes|no)\b", line, re.I)
        if ws:
            return int(ws[-1].lower() == "yes")
    return None


def texts(variant, split):
    import b44_data as BD                       # the generator only produces the text; labels are re-derived here
    return [s["text"] for s in BD.split(variant, split, struct=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="b44_out")
    ap.add_argument("--out", default="b44_out/indep.json")
    ap.add_argument("--skip_ho_train", action="store_true")
    a = ap.parse_args()
    res = {"labels": {}, "cells": {}}
    lab = {}
    for v in ("std", "ho"):
        for sp in ("dev", "test"):
            T = texts(v, sp)
            lab[(v, sp)] = [walk(t)[0] for t in T]
            res["labels"]["%s_%s_pos" % (v, sp)] = sum(lab[(v, sp)])
    if not a.skip_ho_train:
        trp = set()
        for t in texts("ho", "train"):
            trp |= pairs(t)
        tep = set().union(*[pairs(t) for t in texts("ho", "test")])
        res["ho_test_pairs_in_train"] = len(tep & trp)
        res["ho_train_distinct_pairs"] = len(trp)
    # lora / scratch
    for p in sorted(glob.glob(os.path.join(a.root, "lora", "*.test.json"))):
        d = json.load(open(p)); ar = d["args"]; y = lab[(ar["variant"], "test")]
        t = d["test"]
        if ar["task"] == "cls":
            pred = [int(s > 0) for s in t["scores"]]
            main_pred = pred                                   # the decision rule is the sign; labels are independent
        else:
            pred = [yn_last_word(g) for g in t["gens"]]
            import b44_lora as BL                              # the main parser, for per-instance agreement only
            main_pred = [BL.parse_scratch(g)[0] for g in t["gens"]]
        acc = sum(int(p_ == y_) for p_, y_ in zip(pred, y)) / len(y)
        agree = sum(int(a_ == b_) for a_, b_ in zip(pred, main_pred)) / len(pred)
        res["cells"][os.path.basename(p)] = {"indep_acc": acc, "main_acc": t["acc"], "same": abs(acc - t["acc"]) < 1e-6,
                                             "parser_agreement": agree, "pred": pred}
    # naming head / exact / probe
    for p in sorted(glob.glob(os.path.join(a.root, "name", "*.json"))):
        d = json.load(open(p)); ar = d["args"]
        cell = ar["cell"]
        if cell.startswith("k"):
            continue
        y = lab[(cell.split(":")[0], "test")]
        out = []
        for r in d["runs"]:
            if ar["arm"] == "probe":
                pred = [int(s > 0) for s in r["test_scores"]]; main = r["test_acc"]
            else:
                devs = [e["perm/named:dev"]["acc"] for e in r["epochs"]]
                i = max(range(len(devs)), key=lambda j: (devs[j], -j))
                te = r["epochs"][i]["perm/named:test"]
                pred = [int(s >= te["thr"]) for s in te["scores"]]; main = te["acc"]
            acc = sum(int(p_ == y_) for p_, y_ in zip(pred, y)) / len(y)
            out.append({"seed": r["seed"], "indep_acc": acc, "main_acc": main, "same": abs(acc - main) < 1e-6,
                        "pred": pred})
        res["cells"][os.path.basename(p)] = out
    # CoT
    for p in sorted(glob.glob(os.path.join(a.root, "cot", "*.json"))):
        d = json.load(open(p)); ar = d["args"]
        if ar["split"] != "test":
            continue
        y = lab[(ar["variant"], "test")]
        pred = [cot_parse(r["reply"]) for r in d["rows"]]
        acc = sum(int(p_ == y_) for p_, y_ in zip(pred, y)) / len(pred)
        agree = sum(int(p_ == r["pred"]) for p_, r in zip(pred, d["rows"])) / len(pred)
        res["cells"][os.path.basename(p)] = {"indep_acc": acc, "main_acc": d["acc"], "parser_agreement": agree,
                                             "pred": pred}
    json.dump(res, open(a.out, "w"))
    for k, v in res["cells"].items():
        if isinstance(v, list):
            print(k, [(x["seed"], round(x["indep_acc"], 4), round(x["main_acc"], 4)) for x in v])
        else:
            print(k, round(v["indep_acc"], 4), round(v["main_acc"], 4), v.get("parser_agreement", ""))
    print("held-out pairs shared:", res.get("ho_test_pairs_in_train"))


if __name__ == "__main__":
    main()
