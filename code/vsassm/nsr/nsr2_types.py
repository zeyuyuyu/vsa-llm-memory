"""NSR2 development probe: entity types and a containment-type table learned from truth histories, used to flag
reader-written state facts that no observed truth ever had (e.g. a rack inside a box)."""
import json
import re

TYPES = [  # (scenario, regex on the ID, type); first match wins
    ("biobank", r"^S-\d", "specimen"), ("biobank", r"^CB-", "box"), ("biobank", r"^R-\d", "rack"),
    ("biobank", r"^FRZ-", "freezer"), ("biobank", r"^DS-", "shipper"), ("biobank", r"FedEx", "courier"),
    ("biobank", r"PCR bench", "bench"),
    ("warehouse", r"^LPN ", "pallet"), ("warehouse", r"^[A-D]-\d\d-\d\d$", "rack_slot"), ("warehouse", r"^DOOR", "door"),
    ("warehouse", r"^STG", "staging"), ("warehouse", r".", "area"),
    ("ops", r" primary$", "db_primary"), ("ops", r"^db-\d", "host"), ("ops", r"^(prod-|canary|staging)", "cluster"),
    ("ops", r".", "service"),
]


def etype(sc, x, log):
    if sc == "home":
        return "item" if x in {i["id"] for i in log["registry"]["items"]} else "place"
    for s, pat, t in TYPES:
        if s == sc and re.search(pat, x):
            return t
    return "other"


def truth_history(log):
    """Every (thing, place) pair the truth state ever holds, replaying the log's events (move and swap)."""
    st, pairs = {}, set()
    for line in log["lines"]:
        for ev in line.get("events") or []:
            if ev[0] == "move":
                st[ev[1]] = ev[2]
            elif ev[0] == "swap":  # same semantics as apps/real world.World.apply
                a, b = ev[1], ev[2]
                ina = [o for o, p in st.items() if p == a]
                inb = [o for o, p in st.items() if p == b]
                for o in ina:
                    st[o] = b
                for o in inb:
                    st[o] = a
        pairs |= set(st.items())
    return pairs


def learn_table(logs):
    table = set()
    for log in logs:
        for x, p in truth_history(log):
            table.add((log["scenario"], etype(log["scenario"], x, log), etype(log["scenario"], p, log)))
    return table
