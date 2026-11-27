"""NSR1 site catalogs and impact questions (design on dev / retired test; prereg/NSR1_prereg.md).

A catalog holds the static facts of a site that the logs never change (which aisle a rack is in, which room a place is
in, which region a cluster is in, which database a service needs). Every arm is given the same catalog. An impact
question composes the site's current state (where each thing is, read from the log) with catalog facts.

Facts are triples (x, rel, y): state facts are (thing, "in", place); catalog facts use "part_of" (place in a larger
area) and "needs" (service needs a database's primary).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "apps", "real")

ROOMS = {"kitchen table": "kitchen", "kitchen counter": "kitchen", "nightstand": "bedroom", "sofa": "living room",
         "recliner": "living room", "bookshelf": "living room", "desk": "study", "hall table": "hallway",
         "key bowl": "hallway", "coat pocket": "hallway", "bathroom shelf": "bathroom", "medicine cabinet": "bathroom",
         "garden bench": "garden", "car": "garage"}
REGIONS = {"prod-use1-a": "us-east-1", "prod-use1-b": "us-east-1", "canary": "us-east-1", "prod-euw1-a": "eu-west-1",
           "staging": "us-west-2"}
NEEDS = {"payments-api": "ledger-db", "auth-service": "users-db", "search-indexer": "orders-db",
         "checkout-web": "orders-db", "notification-worker": "users-db", "fraud-scorer": "ledger-db",
         "image-resizer": None, "billing-cron": "ledger-db", "recs-model": "orders-db", "chat-gateway": "sessions-redis"}


def catalog(log):
    """Static facts for one log, from its registry (no state)."""
    sc = log["scenario"]
    items = [x["id"] for x in log["registry"]["items"]]
    places = [x["id"] for x in log["registry"]["places"]]
    facts = []
    if sc == "warehouse":
        for p in places:
            if p[:1] in "ABCD" and p[1:2] == "-":
                facts.append((p, "part_of", f"aisle {p[0]}"))
    elif sc == "home":
        facts += [(p, "part_of", ROOMS[p]) for p in places if p in ROOMS]
    elif sc == "ops":
        facts += [(c, "part_of", REGIONS[c]) for c in places if c in REGIONS]
        for s in items:
            if not s.endswith("primary") and NEEDS.get(s):
                db = f"{NEEDS[s]} primary"
                if db in items:
                    facts.append((s, "needs", db))
    return facts


def state_facts(state):
    """state: thing -> place."""
    return [(x, "in", p) for x, p in sorted(state.items())]


def answer(qtype, target, state, cat):
    """Reference graph query (rules written by hand): the true answer of an impact question on a given state."""
    up = dict(state)

    def chain(x):  # containment chain upward through state, then catalog part_of
        part = {a: b for a, r, b in cat if r == "part_of"}
        out, seen = [], {x}
        while True:
            nxt = up.get(x, part.get(x))
            if nxt is None or nxt in seen:
                return out
            out.append(nxt)
            seen.add(nxt)
            x = nxt

    if qtype in ("freezer_down", "aisle_closed", "room_cleared", "region_down"):
        return sorted(x for x in state if target in chain(x) and _is_item(qtype, x))
    if qtype == "host_down":
        dbs = {d for d, p in state.items() if d.endswith("primary") and p == target}
        return sorted(s for s, r, d in cat if r == "needs" and d in dbs)
    raise ValueError(qtype)


def _is_item(qtype, x):
    if qtype == "freezer_down":
        return x.startswith("S-")  # specimens only (boxes and racks are containers)
    if qtype == "region_down":
        return not x.endswith("primary")
    return True


QTEXT = {
    "freezer_down": "{t} has defrosted. Which specimens are affected?",
    "aisle_closed": "{t} is closed for a forklift repair. Which pallets are blocked in it?",
    "room_cleared": "The {t} is being repainted today. Which of Grandpa's things have to be moved out of it?",
    "region_down": "If {t} goes down, which services lose their running instance?",
    "host_down": "If host {t} fails, which services lose the database primary they need?",
}


def questions(log):
    sc = log["scenario"]
    cat = catalog(log)
    places = [x["id"] for x in log["registry"]["places"]]
    if sc == "biobank":
        qs = [("freezer_down", f) for f in places if f.startswith("FRZ-")]
    elif sc == "warehouse":
        qs = [("aisle_closed", a) for a in sorted({b for _, r, b in cat if r == "part_of"})]
    elif sc == "home":
        qs = [("room_cleared", r) for r in sorted({b for _, r, b in cat if r == "part_of"})]
    else:
        qs = [("region_down", r) for r in sorted({b for _, r, b in cat if r == "part_of"})]
        qs += [("host_down", h) for h in places if h.startswith("db-")]
    out = []
    for qt, t in qs:
        out.append({"log": log["id"], "scenario": sc, "qtype": qt, "target": t, "q": QTEXT[qt].format(t=t),
                    "a": answer(qt, t, log["truth"], cat), "sor": answer(qt, t, log["sor"], cat)})
    return out


def load_logs(split):
    logs = []
    for sc in ("warehouse", "biobank", "home", "ops"):
        p = os.path.join(REAL, "data", f"{sc}_{split}.jsonl")
        logs += [json.loads(l) for l in open(p)]
    return logs


if __name__ == "__main__":
    import collections
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    stats = collections.defaultdict(lambda: [0, 0, 0, 0])
    for log in load_logs(split):
        for q in questions(log):
            s = stats[q["qtype"]]
            s[0] += 1
            s[1] += len(q["a"]) > 0
            s[2] += len(q["a"])
            s[3] += q["sor"] == q["a"]
    for k, (n, ne, tot, sor) in stats.items():
        print(f"{k:13s} questions {n:3d}  non-empty {ne:3d}  mean |answer| {tot / n:.1f}  SoR exact {sor / n:.2f}")
