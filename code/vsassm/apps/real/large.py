"""The same day at a large site: a log's lines, questions and answers unchanged, with thousands of other pallets or
specimens on the books. The others come from the system of record only: an inventory count before the shift, and
scanned moves among their own slots (boxes) during it, so they never touch a place a question is about. They are
named the way the site names its own (an LPN and its last four digits; a specimen ID and its digits), so a tail the
floor uses can now be ambiguous, as it would be.

    enlarge(log, 2000)    # warehouse: + 2,000 pallets, 1,000 slots; biobank: + 2,000 specimens, 300 boxes, 60 racks
"""
import json
import random
import re


def _insert(log, at, text, events, src):
    line = {"kind": "struct", "src": src, "time": None, "who": None, "events": events, "text": text, "spec": None}
    log["lines"].insert(at, line)
    log["text_lines"].insert(at, text)


def _day(log, r, scan, start, rate=0.5):
    """After each line of the day, with probability `rate`, one scanned move of another item: scan(clock) -> text."""
    i, clock = start, None
    while i < len(log["lines"]):
        m = re.match(r"\[?(?:\d{4}-\d\d-\d\d )?(\d\d:\d\d)", log["text_lines"][i])
        clock = m.group(1) if m else clock
        if clock and r.random() < rate:
            text, events, src = scan(clock)
            _insert(log, i + 1, text, events, src)
            i += 1
        i += 1


def warehouse(log, n, r):
    reg = log["registry"]
    taken = {e["id"] for e in reg["items"]}
    lpns = []
    while len(lpns) < n:
        x = f"LPN {r.randint(40, 49):03d}{r.randint(1000, 9999)}"
        if x not in taken:
            taken.add(x)
            lpns.append(x)
    slots = [f"{a}-{b:02d}-{c:02d}" for a in "EFGHJKLMNPRSTUVW" for b in range(1, 19) for c in range(1, 5)]
    slots = r.sample(slots, min(len(slots), n // 2))
    reg["items"] += [{"id": x, "aliases": [x[-4:]]} for x in lpns]
    reg["places"] += [{"id": s, "aliases": [f"{s[0]}{s[2:4]}-{s[5:]}", f"aisle {s[0]} bay {int(s[2:4])}"]} for s in slots]
    at = {x: r.choice(slots) for x in lpns}
    rf = lambda: f"RF-{r.randint(1, 12):02d}"
    for k, x in enumerate(lpns):
        _insert(log, k, f"05:{k * 59 // n:02d}  {rf()}  CYCLECNT  {x}  @ {at[x]}  OK  user=inventory",
                [["move", x, at[x]]], "wms")

    def scan(clock):
        x = r.choice(lpns)
        dst = r.choice([s for s in r.sample(slots, 2) if s != at[x]])
        text = f"{clock}  {rf()}  MOVE      {x}  {at[x]} -> {dst}  user=nightshift"
        at[x] = dst
        return text, [["move", x, dst]], "wms"
    _day(log, r, scan, n)


def biobank(log, n, r):
    reg = log["registry"]
    specs = r.sample(range(10000, 99999), n)  # five digits: never one of the day's own S-26-0xxxx
    specs = [f"S-26-{x}" for x in specs]
    boxes = [f"CB-{x}" for x in range(100, 400)]
    racks = [f"R-{x}" for x in range(20, 80)]
    freezers = [f"FRZ-{x}" for x in range(10, 20)]
    reg["items"] += ([{"id": s, "aliases": [s[-5:], "S-" + s[-5:]]} for s in specs]
                     + [{"id": b, "aliases": [f"box {b[3:]}"]} for b in boxes]
                     + [{"id": k, "aliases": [f"rack {k[2:]}"]} for k in racks])
    reg["places"] += ([{"id": b, "aliases": [f"box {b[3:]}"]} for b in boxes]
                      + [{"id": k, "aliases": [f"rack {k[2:]}"]} for k in racks]
                      + [{"id": f, "aliases": [f"freezer {f[4:]}"]} for f in freezers])
    at = {s: r.choice(boxes) for s in specs}
    ts = lambda clock="07:59": f"2026-10-14 {clock}:{r.randint(0, 59):02d}"
    rows = ([(k, "LOCATE   ", f"in {r.choice(freezers)}") for k in racks]
            + [(b, "STORE    ", f"{r.choice(racks)}/slot {r.randint(1, 12)}") for b in boxes]
            + [(s, "ACCESSION", f"{at[s]}/{r.choice('ABCDEFGH')}{r.randint(1, 9)}") for s in specs])
    for k, (x, op, where) in enumerate(rows):
        dst = where.replace("in ", "").split("/")[0]
        _insert(log, k, f"{ts()} | inventory | {op}| {x} | {where}", [["move", x, dst]], "lims")

    def scan(clock):
        s = r.choice(specs)
        dst = r.choice([b for b in r.sample(boxes, 2) if b != at[s]])
        text = f"{ts(clock)} | nightshift | TRANSFER | {s} | {at[s]}/A1 -> {dst}/{r.choice('ABCDEFGH')}{r.randint(1, 9)}"
        at[s] = dst
        return text, [["move", s, dst]], "lims"
    _day(log, r, scan, len(rows))


def enlarge(log, n, seed=0):
    out = json.loads(json.dumps(log))
    {"warehouse": warehouse, "biobank": biobank}[out["scenario"]](out, n, random.Random(f"{log['id']}:{seed}"))
    return out
