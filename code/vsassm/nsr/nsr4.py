"""NSR4 (development): a state-estimation engine that repairs the reader's events before the state is built.

The reader's events, replayed in order, reproduce the product's state exactly (88/88 development logs). The engine
checks each message's readings against symbolic consistency rules and repairs them, then replays:
  R1 duplicate swap      a message read as the same swap twice is one swap (two identical swaps are a no-op, which
                         nobody reports);
  R2 double assignment   one thing given two places by the same human message: the first stands; a later place goes to
                         a tracked item the message names but gave no place to (the name nearest before the place in
                         the text), else it is dropped;
  R3 type violation      a move the learned containment-type table forbids (e.g. a rack into a box) is dropped.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr2_types as T  # noqa: E402
import nsr3_questions as Q3  # noqa: E402


def mentions(text, log, kind="items"):
    """[(position, id)] of every registry name (ID, alias, 4-digit suffix) found in the text."""
    t = text.lower()
    out = []
    for e in log["registry"][kind]:
        names = [e["id"]] + [a for a in e["aliases"] if len(a) >= 3]
        if kind == "items" and e["id"].startswith("LPN "):
            names.append(e["id"][-4:])
        for nm in names:
            for m in re.finditer(re.escape(nm.lower()), t):
                out.append((m.start(), e["id"]))
    return sorted(out)


def place_pos(text, place, log):
    pos = [p for p, i in mentions(text, log, "places") if i == place]
    return pos[0] if pos else None


def repair_line(evs, text, log, line_kind, table, stats):
    evs = [list(e) for e in (evs or [])]
    # R1: identical swaps (unordered pair) collapse to one
    seen, out = set(), []
    for e in evs:
        if e[0] == "swap":
            key = tuple(sorted(e[1:3]))
            if key in seen:
                stats["R1"] += 1
                continue
            seen.add(key)
        out.append(e)
    evs = out
    # R2: one thing, two places in one human message
    if line_kind == "human":
        assigned, out = {}, []
        named = mentions(text, log, "items")
        moved = {e[1] for e in evs if e[0] == "move"}
        for e in evs:
            if e[0] == "move" and e[1] in assigned and assigned[e[1]] != e[2]:
                pp = place_pos(text, e[2], log)
                cands = [(pos, i) for pos, i in named if i not in moved and i not in assigned]
                if pp is not None:
                    before = [c for c in cands if c[0] < pp]
                    cands = before[-1:] if before else cands[:1]
                if cands:
                    new = cands[0][1]
                    out.append(["move", new, e[2]])
                    assigned[new] = e[2]
                    moved.add(new)
                    stats["R2_reassign"] += 1
                else:
                    stats["R2_drop"] += 1
                continue
            if e[0] == "move":
                assigned.setdefault(e[1], e[2])
            out.append(e)
        evs = out
    # R3: containment-type violations
    out = []
    sc = log["scenario"]
    for e in evs:
        if e[0] == "move" and table and (sc, T.etype(sc, e[1], log), T.etype(sc, e[2], log)) not in table:
            stats["R3"] += 1
            continue
        out.append(e)
    return out


def estimate(log, rec, table, stats, rules=("R1", "R2", "R3")):
    """States after every line from the repaired reader events (and the final state)."""
    st, states = {}, []
    for i, evs in enumerate(rec["events"]):
        line = log["lines"][i]
        if rules:
            evs = repair_line(evs, log["text_lines"][i], log, line["kind"], table if "R3" in rules else None, stats)
            if "R1" not in rules or "R2" not in rules:
                pass
        for ev in evs or []:
            Q3.apply(st, ev)
        states.append(dict(st))
    return states
