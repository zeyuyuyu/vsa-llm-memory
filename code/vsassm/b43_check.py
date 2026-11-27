"""B43 second scorer (prereg/B43_natural_text_and_chunked_baseline_prereg.md, Checks). Written separately from
b43_verdict.py / b43_propara.py and imports neither (nor ruler_rm's or b18's scorers).

python3 b43_check.py ruler COA_DIR B21_DIR RULER_DIR       re-score store / dict / CoA, 9-task means per length
python3 b43_check.py bab COA_DIR ASSEMBLED_128K TRAIN_ROOT  re-score BABILong CoA answers with the official rule
python3 b43_check.py propara SPLIT PP_OUTDIR GRIDS_DIR      state accuracy (nltk Porter stemmer) and grid -> action rows
"""
import collections
import glob
import json
import re
import sys

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt"]


def sm_all(pred, refs):
    p = pred.lower()
    return sum(1.0 for r in refs if r.lower() in p) / len(refs)


def ruler(coa_dir, b21_dir, root):
    for L in (65536, 131072):
        coa = collections.defaultdict(list)
        for f in glob.glob(f"{coa_dir}/coa_ruler_{L}_*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                coa[r["task"]].append(r)
        if not coa:
            continue
        b21 = collections.defaultdict(list)
        for f in glob.glob(f"{b21_dir}/{L}_*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                b21[r["task"]].append(r)
        means = collections.defaultdict(list)
        for t in TASKS:
            gold = [json.loads(l) for l in open(f"{root}/{L}/{t}/validation.jsonl")][:100]
            # match by (index, gold outputs) - a different alignment path from the verdict's file position
            ck = {(r["index"], json.dumps(r["outputs"])): r for r in coa[t]}
            bk = {(r["index"], json.dumps(r["outputs"])): r for r in b21[t]}
            sc = collections.defaultdict(list)
            for g in gold:
                key = (g["index"], json.dumps(g["outputs"]))
                if key not in ck:  # sample not (yet) run: skipped and counted
                    continue
                sc["coa"].append(sm_all(ck[key]["pred"], g["outputs"]))
                sc["store"].append(sm_all(bk[key]["pred_vsa"], g["outputs"]))
                sc["dict"].append(sm_all(bk[key]["pred_table"], g["outputs"]))
            print(L, t, "n", len(sc["coa"]), {k: round(100 * sum(v) / len(v), 2) for k, v in sc.items() if v})
            if len(sc["coa"]) == 100:
                for k, v in sc.items():
                    means[k].append(100 * sum(v) / len(v))
        print(L, "mean over complete tasks", {k: round(sum(v) / len(v), 3) for k, v in means.items()},
              "complete tasks", len(means["coa"]))


def bab(coa_dir, assembled, train_root):
    labels = {}

    def labs(task):
        if task not in labels:
            s = set()
            for f in glob.glob(f"{train_root}/data/{task}/*.json"):
                for r in json.load(open(f)):
                    s.update(x.strip().lower() for x in str(r["target"]).split(","))
            labels[task] = s
        return labels[task]

    def official(pred, target, question, task):  # babilong/metrics.py compare_answers, re-written
        out = pred.lower().split(".")[0]
        ls = {l for l in labs(task) if not re.search(r"\b" + re.escape(l) + r"\b", question.lower())}
        found = {l for l in ls if re.search(r"\b" + re.escape(l) + r"\b", out)}
        tg = {x.strip() for x in str(target).lower().split(",")}
        return (tg <= found and not (found - tg)) if len(tg) > 1 else found == tg

    mem = {(r["task"], r["i"]): r["ok_official"] for r in map(json.loads, open(assembled))}
    acc = collections.defaultdict(list)
    for f in glob.glob(f"{coa_dir}/coa_bab_128k*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            acc[r["task"]].append((official(r["pred"].strip(), r["target"], r["q"], r["task"]), mem[(r["task"], r["index"])]))
    for t in sorted(acc):
        print(t, len(acc[t]), "coa", round(100 * sum(a for a, _ in acc[t]) / len(acc[t]), 1),
              "memory", round(100 * sum(b for _, b in acc[t]) / len(acc[t]), 1))
    if acc:
        print("mean coa", round(sum(100 * sum(a for a, _ in v) / len(v) for v in acc.values()) / len(acc), 2))


def propara(split, outdir, gridsdir):
    from nltk.stem.porter import PorterStemmer
    st = PorterStemmer()
    arts = ("a ", "an ", "the ", "your ", "his ", "their ", "my ", "another ", "other ", "this ", "that ")

    def norm(x):
        x = x.lower().strip()
        for a in arts:
            if x.startswith(a):
                x = x[len(a):]
                break
        return st.stem(x).strip()

    rows = [json.loads(l) for l in open(f"{gridsdir}/grids.v1.{split}.json")]
    for arm in ("direct_think", "direct_nothink", "store_vsa", "store_dict"):
        try:
            g = json.load(open(f"{outdir}/{arm}_grids.json"))
            tsv = [l.rstrip("\n").split("\t") for l in open(f"{outdir}/{arm}.tsv")]
        except FileNotFoundError:
            continue
        ok = n = 0
        for r in rows:
            for P, gold in zip(r["participants"], r["states"]):
                for a, b in zip(gold, g[r["para_id"]][P]):
                    n += 1
                    if a in ("-", "?") or b in ("-", "?", None):
                        ok += a == b
                    else:
                        ok += norm(a) == norm(b)
        # grid -> action rows, re-derived
        mism = 0
        tsvk = {(x[0], x[2], x[1]): x[3:] for x in tsv}
        for r in rows:
            for P in r["participants"]:
                s = [x if x is not None else "?" for x in g[r["para_id"]][P]]
                for t in range(1, len(s)):
                    b, a = s[t - 1], s[t]
                    act = ("CREATE" if b == "-" and a != "-" else "DESTROY" if b != "-" and a == "-" else
                           "MOVE" if b != a else "NONE")
                    mism += tsvk[(r["para_id"], P, str(t))] != [act, b, a]
        print(f"{arm:15s} state acc {100 * ok / n:.2f} ({n} cells); action rows differing from the .tsv: {mism}")


if __name__ == "__main__":
    {"ruler": ruler, "bab": bab, "propara": propara}[sys.argv[1]](*sys.argv[2:])
