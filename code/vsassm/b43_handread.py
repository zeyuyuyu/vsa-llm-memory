"""B43 hand-reading helper (prereg Checks): prints CoA failures on RULER at a length (up to 40, sampled with numpy seed 0
if more) with the gold values, where each gold value appeared along the chain, the final CU and the manager's answer,
and a mechanical class; the human reading and notes go to results/b43_handread.txt.

python3 b43_handread.py ruler COA_DIR 131072 [--max 40]
python3 b43_handread.py propara PP_OUTDIR SPLIT [--k 10]
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))


def sm_all(pred, refs):
    return sum(r.lower() in pred.lower() for r in refs) / len(refs)


def ruler(coa_dir, L, mx=40):
    fails = []
    for f in sorted(glob.glob(f"{coa_dir}/coa_ruler_{L}_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if sm_all(r["pred"], r["outputs"]) < 1:
                fails.append(r)
    print(f"{len(fails)} CoA failures at {L}")
    if len(fails) > mx:
        idx = sorted(np.random.default_rng(0).choice(len(fails), mx, replace=False))
        fails = [fails[i] for i in idx]
    for r in fails:
        gold = r["outputs"]
        seen = {g: [i for i, w in enumerate(r["workers"]) if g.lower() in w["cu"].lower()] for g in gold}
        final = r["workers"][-1]["cu"]
        if all(g.lower() in final.lower() for g in gold):
            cls = "in final CU, manager failed"
        elif all(not v for v in seen.values()):
            cls = "never in any CU"
        elif any(v and g.lower() not in final.lower() for g, v in seen.items()):
            cls = "lost along the chain"
        else:
            cls = "partly never in any CU"
        print("=" * 100)
        print(f"{r['task']} pos {r['pos']} index {r['index']} windows {r['n_windows']}  class: {cls}")
        print("Q:", r["q"])
        print("gold:", gold, " CUs containing each:", seen)
        print("pred:", repr(r["pred"][:300]))
        print("final CU:", repr(final[:1200]))


def propara(outdir, split, k=10):
    import b43_propara as B
    rows = {r["para_id"]: r for r in B.load(split)}
    gd = {a: json.load(open(f"{outdir}/{a}_grids.json")) for a in ("direct_think", "store_vsa")}
    ev = json.load(open(f"{outdir}/events.json"))
    diffs = []
    for pid, r in rows.items():
        acc = {}
        for a in gd:
            c = [B.cell_correct(g, p) for P in r["participants"] for g, p in zip(dict(zip(r["participants"], r["states"]))[P], gd[a][pid][P])]
            acc[a] = sum(c) / len(c)
        diffs.append((abs(acc["store_vsa"] - acc["direct_think"]), pid, acc))
    for _, pid, acc in sorted(diffs, reverse=True)[:k]:
        r = rows[pid]
        print("=" * 100)
        print(f"paragraph {pid}  state acc store {acc['store_vsa']:.2f} direct {acc['direct_think']:.2f}")
        for i, s in enumerate(r["sentence_texts"]):
            print(f"  {i + 1}. {s}   events: {ev[pid].get(str(i + 1), [])}")
        for P, st in zip(r["participants"], r["states"]):
            print(f"  {P}\n    gold   {st}\n    store  {gd['store_vsa'][pid][P]}\n    direct {gd['direct_think'][pid][P]}")


if __name__ == "__main__":
    if sys.argv[1] == "ruler":
        ruler(sys.argv[2], int(sys.argv[3]))
    else:
        propara(sys.argv[2], sys.argv[3])
