"""Second angle on NSR9 arm D_c's wrong answers: (1) did the parser get a full answer (every slot), and how far is it
from the gold; (2) do the engine's own formalisations (E_c, both models) reject it (fix op)?"""
import json, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = ["bench/heldout", "engine"]
import families as F, gm
fams = F.all_families()
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/nsr9_items.jsonl")}
E = {json.loads(l)["id"]: json.loads(l) for l in open("out/nsr9c_E.jsonl")}
D = {json.loads(l)["id"]: json.loads(l) for l in open("out/nsr9c_D.jsonl")}
def norm(k): return ":".join(p.strip().lower() for p in str(k).split(":"))
wrong = [i for i, r in D.items() if r.get("D") is not None and not fams[items[i]["family"]].check(items[i], r["D"])]
def one(iid):
    it, d = items[iid], D[iid]["D"]
    gold = {norm(k): v for k, v in it["gold"].items()} if not it["gold"].get("impossible") else {}
    got = {norm(k): v for k, v in d.items()}
    miss = len(set(gold) - set(got)); diff = sum(1 for k in gold if k in got and str(got[k]) != str(gold[k]))
    verdicts = []
    for f in E[iid]["forms"]:
        if not f["code"]: verdicts.append("no model"); continue
        r = gm.run(f["code"], {"op": "fix", "assign": d}, timeout=600)
        verdicts.append(r.get("status") or ("error: " + str(r.get("error"))[:40]))
    return iid, len(gold), miss, diff, verdicts
with ThreadPoolExecutor(6) as ex:
    for iid, n, miss, diff, v in ex.map(one, wrong):
        print("%-22s slots %4d  missing %3d  differ from gold %3d  engine models: %s" % (iid, n, miss, diff, v))
