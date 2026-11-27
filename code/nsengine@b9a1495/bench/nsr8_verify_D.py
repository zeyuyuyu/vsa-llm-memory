"""Second angle on NSR8 arm D: is each level 4-5 direct answer accepted by the engine's own formalisations (fix op;
for optimisation, reaching that model's optimum)? Independent of the family's check()."""
import json, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = ["/path/to/nsengine/engine"]
import gm
items = {json.loads(l)["id"]: json.loads(l) for l in open("/path/to/nsengine/data/nsr8_items.jsonl")}
E = {json.loads(l)["id"]: json.loads(l) for l in open("/path/to/nsengine/out/nsr8_E.jsonl")}
D = {json.loads(l)["id"]: json.loads(l) for l in open("/path/to/nsengine/out/nsr8_D.jsonl")}
def one(iid):
    it, d = items[iid], D[iid]["D"]
    out = []
    for f in E[iid]["forms"]:
        if not f["code"]:
            out.append("no model"); continue
        if d.get("impossible"):
            r = gm.run(f["code"], {"op": "solve"}, timeout=300); out.append("agrees impossible" if r.get("status") == "unsat" else "model says possible")
            continue
        r = gm.run(f["code"], {"op": "fix", "assign": d}, timeout=300)
        if r.get("status") != "sat":
            out.append("rejects (%s)" % (r.get("status") or r.get("error", "")[:60])); continue
        if "objective_model" in r:
            out.append("accepts, optimal" if r["objective_model"] == r["objective_fixed"] else "accepts, NOT optimal")
        else:
            out.append("accepts")
    return iid, out
ids = [i for i, it in items.items() if it["level"] >= 4]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, ids))
C = collections.Counter()
for iid, o in res:
    C[(items[iid]["family"], tuple(sorted(set(o))))] += 1
for k, v in sorted(C.items()): print(k, v)
