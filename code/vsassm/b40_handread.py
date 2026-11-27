"""B40 hand-read helper: 20 test errors per trained arm and task (seed 0), sampled with a fixed seed, printed with the
question, the gold answer, the model's output and the module's own record, for manual classification
(read error / computation path / write-back not followed / format). Also lists every VSA-vs-DISC disagreement on
task N with its source (module read wrong vs module right but the LLM said something else).
usage: python b40_handread.py <pred_root> > results/b40_handread_raw.txt"""
import json, os, sys
import numpy as np
import b40_data as D

root = sys.argv[1]
rng = np.random.default_rng(4041)


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else None


def show(task, row, x):
    if task == "N":
        q = D.op_question(row["op"], row["x"], row["y"])
    else:
        story, qtext, _ = D.story_text(row)
        q = f"{story} | {qtext} (depth {row['depth']}, {row['size']})"
    m = x.get("mod")
    return f"  {x['id']}: {q}\n     gold={x['gold']} out={x['out'][-160:]!r} pred={x['pred']!r} module={m}"


for task in ("N", "R"):
    rows = {r["id"]: r for r in D.gen(task, "test")}
    for arm in ("vsa", "disc", "lora", "lora1l", "base", "cot"):
        p = f"{root}/{task}/{arm}_s0.jsonl" if arm not in ("base", "cot") else f"{root}/{task}/{arm}.jsonl"
        P = load(p)
        if P is None:
            continue
        errs = [i for i, x in P.items() if not x["correct"]]
        pick = sorted(rng.choice(len(errs), min(20, len(errs)), replace=False)) if errs else []
        print(f"===== {task} {arm}: {len(errs)} errors of {len(P)}; showing {len(pick)}")
        for k in pick:
            print(show(task, rows[errs[k]], P[errs[k]]))
    V, Dd = load(f"{root}/{task}/vsa_s0.jsonl"), load(f"{root}/{task}/disc_s0.jsonl")
    if V and Dd:
        dis = [i for i in V if V[i]["correct"] != Dd[i]["correct"]]
        print(f"===== {task} VSA-vs-DISC disagreements (seed 0): {len(dis)}")
        src = {"vsa_wrong_read": 0, "vsa_right_read_llm_wrong": 0, "disc_wrong_read": 0, "disc_right_read_llm_wrong": 0}
        for i in dis:
            g = rows[i]
            for nm, P in (("vsa", V), ("disc", Dd)):
                if P[i]["correct"]:
                    continue
                m = P[i].get("mod") or {}
                if m.get("ans") == g["ans"]:
                    src[f"{nm}_right_read_llm_wrong"] += 1
                else:
                    src[f"{nm}_wrong_read"] += 1
        print("   sources:", src)
        for i in dis[:30]:
            print(show(task, rows[i], V[i]))
            print("     DISC:", Dd[i]["pred"], Dd[i].get("mod"))
