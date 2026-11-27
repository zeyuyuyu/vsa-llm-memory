"""B28 hand-reading helper: prints discordant cases in their raw bAbI stories, for reading by eye
(prereg "Checks from other angles": >= 10 discordant cases per bAbI setting and >= 10 task-2 disagreements per
subset). Runs on the box (needs the cached bAbI test split and the tokenizer).

  python b28_handread.py --runs out --out out/b28_handread.txt
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b28_common as C                                                # noqa: E402
from b28_babi import index_rows                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    M = C.load_ngcls(); v5 = M["v5"]
    tok = C.tokenizer()
    lines = []
    W = lines.append
    for task, ml in ((4, 192), (15, 256)):
        d = json.load(open(os.path.join(a.runs, "b28_task%d.json" % task)))
        raw = v5.load_babi(task, 2000, 200, 500)["test"]
        voc = d["vocab"]
        # rebuild the index mapping (tasks 4 and 15 keep every question, checked here)
        assert d["n_scored"] == len(raw) == 500
        W("=" * 100); W("TASK %d  (vocab %s)" % (task, voc))
        for s in C.SEEDS:
            pc = d["runs"]["parser/comm/%d" % s]["per_q"]; pp = d["runs"]["parser/perm/%d" % s]["per_q"]
            disc = [i for i in range(500) if pc[i]["pred"] != d["gold"][i]["label"] and pp[i]["pred"] == d["gold"][i]["label"]]
            perr = [i for i in range(500) if pp[i]["pred"] != d["gold"][i]["label"]]
            W("-- seed %d: comm wrong & perm right %d; perm errors %d; comm errors on non-alias %d" % (
                s, len(disc), len(perr),
                sum(1 for i in range(500) if pc[i]["pred"] != d["gold"][i]["label"] and not d["gold"][i]["alias"])))
            show = (disc[:a.n] if s == 42 else []) + perr
            for i in show:
                g = d["gold"][i]
                W("   [q%d] alias=%d gold=%s comm=%s perm=%s  (comm cos top %.3f other %s)" % (
                    i, g["alias"], voc[g["label"]],
                    voc[pc[i]["pred"]], voc[pp[i]["pred"]], pc[i]["cos_top"],
                    "%.3f" % pc[i]["cos_other"] if pc[i]["cos_other"] is not None else "-"))
                for ln in raw[i][0].split("\n"):
                    W("        " + ln)
    # task 2: memory (naming head, comm) vs best-of-twelve fine-tune, seed 42
    d = json.load(open(os.path.join(a.runs, "b28_task2.json")))
    ft = json.load(open(os.path.join(a.runs, "b28_lora2_r16_e6_lr0.0001_expr.json")))
    raw = v5.load_babi(2, 2000, 200, 500)["test"]; voc = d["vocab"]
    lab = d["labels_all"]; idx = d["row_index"]
    mem = {}
    for j, o in zip(idx, d["runs"]["probe_argmax/comm/42"]["per_q"]):
        mem[j] = o["pred"]
    exp = {j: e for j, e in zip(idx, d["expressible"])}
    run = [r for r in ft["runs"] if r["seed"] == 42][0]
    kept = ft["kept"]
    fpred = [kept[max(range(len(x)), key=lambda k: x[k])] for x in run["logitsB"]]
    W("=" * 100); W("TASK 2 seed 42: memory (naming head, comm) vs fine-tune best of twelve")
    for name, cond in (("420 expressible", lambda j: exp.get(j) == 1), ("24 parsed, not expressible", lambda j: exp.get(j) == 0),
                       ("56 not parsed", lambda j: j not in exp)):
        cases = [j for j in range(500) if cond(j) and (mem.get(j) != fpred[j])]
        W("-- %s: %d disagreements (memory right %d, fine-tune right %d)" % (
            name, len(cases), sum(mem.get(j) == lab[j] for j in cases), sum(fpred[j] == lab[j] for j in cases)))
        for j in cases[:a.n]:
            W("   [q%d] gold=%s memory=%s fine-tune=%s" % (j, voc[lab[j]], voc[mem[j]] if j in mem else "(no answer)",
                                                         voc[fpred[j]]))
            for ln in raw[j][0].split("\n"):
                W("        " + ln)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("B28-HANDREAD-DONE -> %s (%d lines)" % (a.out, len(lines)))


if __name__ == "__main__":
    main()
