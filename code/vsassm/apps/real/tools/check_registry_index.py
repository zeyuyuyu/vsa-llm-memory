"""The indexed Registry must answer as the scanning one did (commit e377030's product.py, passed as the argument):
resolve / find_in / mentions / said_as on every message, question and reader output of the splits, and on a 3,000-item
registry with colliding aliases (40 pallets per customer, 4-digit tails that repeat).

    git show e377030:apps/real/product.py > /tmp/product_old.py
    python3 apps/real/tools/check_registry_index.py /tmp/product_old.py
"""
import importlib.util, json, random, sys
sys.path.insert(0, "apps/real"); sys.path.insert(0, ".")
import b14
import product
import run_real as rr
spec = importlib.util.spec_from_file_location("product_old", sys.argv[1]); old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def compare(log, names, texts, questions):
    A, B = old.Registry(log), product.Registry(log)
    n, bad = 0, []
    for x in names:
        for pool in (None, "items", "places"):
            pa, pb = (getattr(A, pool) if pool else None), (getattr(B, pool) if pool else None)
            n += 1
            if A.resolve(x, pa) != B.resolve(x, pb):
                bad.append(("resolve", x, pool))
    for q in questions:
        n += 1
        if A.find_in(q) != B.find_in(q):
            bad.append(("find_in", q))
    ids = A.items + A.places
    for t in texts:
        n += 3
        if A.find_in(t, A.items, message=True) != B.find_in(t, B.items, message=True):
            bad.append(("find_in message", t))
        if [i for i in ids if A.mentions(t, i)] != [i for i in ids if B.mentions(t, i)]:
            bad.append(("mentions", t))
        if [A.said_as(t, i) for i in ids] != [B.said_as(t, i) for i in ids]:
            bad.append(("said_as", t))
    return n, bad


total, bads = 0, []
for split in ("dev", "test", "test2"):
    raws = {}
    for f in (f"apps/real/out/reader_{split}_verify_v4.jsonl",):
        for r in map(json.loads, open(f)):
            raws[r["id"]] = r["raw"]
    for l in rr.load(split):
        texts = [rr.message(l, x) for x in l["lines"] if x["kind"] == "human"] + list(l["text_lines"])
        names = [str(s) for txt in raws.get(l["id"], {}).values() for ev in (b14.parse(txt) or []) for s in ev[1:3]]
        names += [n for e in l["registry"]["items"] + l["registry"]["places"] for n in [e["id"]] + e["aliases"]]
        names += [w for t in texts[:40] for w in t.split()[:6]]
        n, bad = compare(l, names, texts, [q["q"] for q in l["questions"]])
        total += n
        bads += bad
    print(f"{split}: {total} lookups so far, {len(bads)} differ")

r = random.Random(7)
custs = ["Bosch", "Dyson", "Nestle", "Lego", "Sony", "Weber", "Yeti", "Brita"] * 2 + ["Hydro Flask", "Kitchen Aid"]
items = [{"id": f"LPN {r.randint(40, 49):03d}{r.randint(1000, 9999)}", "aliases": []} for _ in range(3000)]
for e in items:
    c = r.choice(custs)
    e["aliases"] = [e["id"][-4:], f"the {c} pallet", f"{c} {r.randint(1, 40)}"]
places = [{"id": f"{a}-{b:02d}-{c:02d}", "aliases": [f"{a}{b:02d}-{c:02d}", f"aisle {a} bay {b}"]}
          for a in "ABCDEFGH" for b in range(1, 30) for c in range(1, 5)][:900]
log = {"registry": {"items": items, "places": places}, "domain": "locations"}
names, texts, qs = [], [], []
for _ in range(400):
    e, p = r.choice(items), r.choice(places)
    nm, pn = r.choice([e["id"]] + e["aliases"]), r.choice([p["id"]] + p["aliases"])
    names += [nm, pn, nm.split()[-1], "the " + nm, nm + " pallet", pn + " shelf"]
    texts.append(r.choice([f"moved {nm} to {pn}, it was blocking", f"{nm} -> {pn}", f"anyone seen {nm}? not at {pn}",
                           f"put the {r.choice(custs)} stuff in {pn}", f"#{e['id'][-3:]} is at {pn} now"]))
    qs.append(r.choice([f"Where is {nm}?", f"Where's {nm} now?", f"What is in {pn}?"]))
n, bad = compare(log, names, texts[:150], qs)
total += n
bads += bad
print(f"large registry ({len(items)} items, {len(places)} places): {total} lookups in all, {len(bads)} differ")
for b in bads[:10]:
    print("  ", b)
