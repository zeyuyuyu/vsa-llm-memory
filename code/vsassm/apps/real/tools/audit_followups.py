import json, sys, collections
sys.path.insert(0, "apps/real"); sys.path.insert(0, ".")
from run_real import load
from product import Registry
split = sys.argv[1]
rd = {r["id"]: r for r in map(json.loads, open(f"apps/real/out/reader_{split}_verify.jsonl"))}
for sc in ["warehouse", "biobank", "home", "ops"]:
    c = collections.Counter()
    for l in [x for x in load(split) if x["scenario"] == sc]:
        reg = Registry(l); r = rd[l["id"]]
        for i, x in enumerate(l["lines"]):
            if x["kind"] != "human" or x["spec"]["mode"] != "follow_up": continue
            ant = x["events"][0][1]; c["n"] += 1
            prev = [(k, y) for k, y in enumerate(l["lines"][:i]) if y["kind"] == "human" and y["src"] == x["src"]]
            # the report that set the antecedent: last same-channel line whose events move ant
            k_ant = max((k for k, y in prev if any(e[1] == ant for e in y["events"])), default=None)
            if k_ant is None: c["no_antecedent_line"] += 1; continue
            between = [y for k, y in prev if k > k_ant]
            dist = len(between) + 1
            c["in_ctx3" if dist <= 3 else "beyond_ctx3"] += 1
            other = [y for y in between if (reg.find_in(y["text"]) or ant) != ant and y["spec"]["mode"] in ("plan", "not_found", "noise", "report", "swap")]
            if other: c["other_item_mentioned_between"] += 1
            # what the reader had applied last in this channel before line i
            last = None
            for k, y in prev:
                for e in r["events"][k]:
                    if e[0] == "move": last = e[1]
            c["last_applied_is_ant" if last == ant else "last_applied_other"] += 1
            got = r["events"][i]
            c["right" if [tuple(e) for e in got] == [tuple(e) for e in x["events"]] else "wrong"] += 1
            c["names_item" if reg.find_in(x["text"]) else "names_no_item"] += 1
    print(sc, dict(c))
