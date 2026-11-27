"""The SDK (readable_memory.SiteMemory.observe, one line at a time) must give the evaluation's answers: run one test2
log per scenario through it with the same fake reader as check_reader_equivalence.py and compare with that run.
"""
import hashlib, json, re, sys
sys.path.insert(0, "apps/real"); sys.path.insert(0, ".")
import run_real as rr
from readable_memory import SiteMemory
truth = {}
logs = rr.load("test2")
for l in logs:
    for x in l["lines"]:
        if x["kind"] == "human": truth[rr.message(l, x)] = x["events"]
h = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)
def fake(prompts):
    out = []
    for p in prompts:
        if p.startswith("Message:") and "Choose one:" in p: out.append(("AB"[h(p) % 2], 1))
        elif p.startswith("Message:"): out.append(("yes" if h(p) % 5 else "no", 1))
        else:
            sent = re.findall(r"\nSentence: ([^\n]*)", p)[-1]; evs = truth.get(sent, [])
            if h(p) % 7 == 0 and evs: evs = [[e[0], e[2], e[1]] for e in evs]
            out.append((json.dumps({"events": evs}), 3))
    return out
ref = {r["id"]: r for r in map(json.loads, open("apps/real/out/reader_test2_verify_eqnew.jsonl"))}
for sc in rr.SCEN:
    l = next(x for x in logs if x["scenario"] == sc)
    mem = SiteMemory(registry=l["registry"], domain=l["domain"], connector=l["scenario"], reader=fake)
    for t, x in zip(l["text_lines"], l["lines"]):
        if x["kind"] == "human": mem.observe(rr.message(l, x), channel=x["src"])
        else: mem.observe(t)
    got = [mem.site.ask(q["q"], q["kind"])[0] for q in l["questions"]]
    exp = [a["pred"] for a in ref[l["id"]]["answers"]]
    print(sc, "SDK == evaluation:", got == exp, f"({sum(g == e for g, e in zip(got, exp))}/{len(exp)})")
