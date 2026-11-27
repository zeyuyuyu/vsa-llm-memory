"""Refactor check: run two copies of run_real.py on a split with the same deterministic fake reader (true events,
sometimes with the slots swapped; hashed A/B choices and yes/no checks) and compare their reader outputs.

  python3 apps/real/tools/check_reader_equivalence.py <run_real copy inside apps/real> <tag> <split> [large=2000 ...]

Extra key=value arguments set run_reader options (only, large, list_limit, out); with large, this is the oracle-reader
run of the large-site condition.
"""
import hashlib, importlib.util, json, re, sys, types
sys.path.insert(0, "apps/real"); sys.path.insert(0, ".")
import b14
path, tag, split = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("run_real", path); rr = importlib.util.module_from_spec(spec)
sys.modules["run_real"] = rr; spec.loader.exec_module(rr)
truth = {}
for l in rr.load(split):
    for x in l["lines"]:
        if x["kind"] == "human":
            truth[rr.message(l, x)] = x["events"]
h = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)
class Fake:
    def __init__(self, *a): pass
    def __call__(self, prompts):
        out = []
        for p in prompts:
            if p.startswith("Message:") and "Choose one:" in p:
                out.append(("AB"[h(p) % 2], 1))
            elif p.startswith("Message:"):
                out.append(("yes" if h(p) % 5 else "no", 1))
            else:
                sent = re.findall(r"\nSentence: ([^\n]*)", p)[-1]
                evs = truth.get(sent, [])
                if h(p) % 7 == 0 and evs:  # sometimes swap the slots, to exercise orient / grounding
                    evs = [[e[0], e[2], e[1]] for e in evs]
                out.append((json.dumps({"events": evs}), 3))
        return out
b14.Reader = Fake
opts = dict(only="", large=0, list_limit=None, out="")
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    opts[k] = int(v) if k in ("large", "list_limit") else v
rr.run_reader(types.SimpleNamespace(split=split, suffix="", model=None, adapter=None, verify=True, tag=tag, **opts))
