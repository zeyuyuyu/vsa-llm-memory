"""B65 sample reading (prereg 'Checks'): for each primary cell, up to 12 seed-0 trained-type test items where the vector
arm and DISC disagree (fixed seed 65), with both modules' reads, answers, gates and outputs, and a mechanical tag of the
losing arm's failure (read error: gate open, module answer wrong; gate: closed; not followed: module answer right and
gate open, output wrong). The tags are then read item by item (results/b65_handread.txt is the raw sheet).
usage: python b65_handread.py <root>
"""
import json, sys
import numpy as np
import b40_data as D


def load(p):
    return [json.loads(l) for l in open(p)]


def tag(r):
    m = r["mod"]
    if not m["gate"]:
        return "gate closed"
    if str(m["ans"]) != r["gold"]:
        wrong = [k for k in ("op", "x", "y") if m[k] != r[k]]
        return "read error (" + ",".join(wrong) + ")" if wrong else "module answer wrong with exact read"
    return "write-back not followed"


def main():
    root = sys.argv[1]
    rng = np.random.default_rng(65)
    out = []
    for pos in ("last", "ops"):
        dc = load(f"{root}/preds/{pos}/disc_s0.jsonl")
        for arm in ("vsapub", "vsabm"):
            vc = load(f"{root}/preds/{pos}/{arm}_s0.jsonl")
            dis = [i for i, (a, b) in enumerate(zip(vc, dc)) if a["op"] in D.OPS_TRAIN and a["correct"] != b["correct"]]
            vw = [i for i in dis if vc[i]["correct"]]
            dw = [i for i in dis if dc[i]["correct"]]
            out.append(f"=== {pos} {arm} vs disc (seed 0): {len(dis)} disagreements; vector right {len(vw)}, disc right {len(dw)}")
            counts = {}
            for side, lst, loser, winner in (("vector right, disc wrong", vw, dc, vc), ("disc right, vector wrong", dw, vc, dc)):
                pick = sorted(rng.choice(lst, min(6, len(lst)), replace=False).tolist()) if lst else []
                full = {}
                for i in lst:
                    t = tag(loser[i]).split(" (")[0]
                    full[t] = full.get(t, 0) + 1
                out.append(f" -- {side}: all {len(lst)} tagged {full}")
                for i in pick:
                    a, b = vc[i], dc[i]
                    out.append(f"   {a['id']} {a['op']} x={a['x']} y={a['y']} gold={a['gold']} | loser: {tag(loser[i])}")
                    out.append(f"      vector mod={a['mod']} out={a['out']!r} correct={a['correct']}")
                    out.append(f"      disc   mod={b['mod']} out={b['out']!r} correct={b['correct']}")
    open("results/b65_handread.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
