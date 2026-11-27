"""NSR9 post-hoc lenient scoring of direct answers (disclosed; not the registered scorer): when a family's answer
slots have a single prefix (islanders "knight:", vans "van:", project "start:") and the answer uses bare entity
names as keys, the prefix is added before the family's check(). Strict (registered) and lenient are both reported."""
import json, re, sys, collections
sys.path[:0] = ["bench/heldout"]
import families as F
fams = F.all_families()
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/nsr9_items.jsonl")}
def prefixes(it): return sorted(set(re.findall(r"(\w+):<", it["answer_keys"])))
def lenient(it, ans):
    if not isinstance(ans, dict) or ans.get("impossible"): return ans
    ps = prefixes(it)
    if len(ps) != 1: return ans
    return {(k if ":" in str(k) else "%s:%s" % (ps[0], k)): v for k, v in ans.items()}
def score(path, arm):
    R = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    C = collections.defaultdict(collections.Counter)
    for iid, it in items.items():
        a = (R.get(iid) or {}).get(arm)
        for mode, x in (("strict", a), ("lenient", lenient(it, a) if a is not None else None)):
            o = "abstain" if x is None else ("correct" if fams[it["family"]].check(it, x) else "wrong")
            C[(mode, "all")][o] += 1; C[(mode, it["family"])][o] += 1; C[(mode, "L%d" % it["level"])][o] += 1
    return C
for path, arm in (("out/nsr9c_D.jsonl", "D"), ("out/nsr9c_E.jsonl", "E")):
    C = score(path, arm)
    for mode in ("strict", "lenient"):
        print(arm, mode, dict(C[(mode, "all")]), {k[1]: dict(v) for k, v in C.items() if k[0] == mode and k[1] != "all"})
