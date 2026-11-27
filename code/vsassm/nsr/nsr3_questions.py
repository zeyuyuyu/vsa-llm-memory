"""NSR3 (development): temporal impact questions over a log's history, and history engines.

A question names an event window [t1, t2] in the log's own clock and asks which things were in the target region at ANY
point in the window (things moved out since still count). Gold = union of the truth states over the window (replaying
the log's events). Engines: the same union over the read state's per-line snapshots (history engine), the final read
state only (NSR1's engine, ignores time), and the system-of-record history.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_catalog as C  # noqa: E402

QTEXT = {
    "freezer_down": "{t} logged a defrost between {a} and {b}. Which specimens were inside it at any point in that window?",
    "aisle_closed": "A sprinkler leaked over {t} between {a} and {b}. Which pallets were in that aisle at any point in that window?",
    "room_cleared": "The {t} was being painted between {a} and {b}. Which tracked items were in that room at any point in that window?",
    "region_down": "{t} was degraded between {a} and {b}. Which services ran there at any point in that window?",
    "host_down": "Host {t} was down between {a} and {b}. Which services had the database primary they need on it at any point in that window?",
}


def apply(st, ev):
    if ev[0] == "move":
        st[ev[1]] = ev[2]
    elif ev[0] == "swap":
        a, b = ev[1], ev[2]
        ina = [o for o, p in st.items() if p == a]
        inb = [o for o, p in st.items() if p == b]
        for o in ina:
            st[o] = b
        for o in inb:
            st[o] = a


def truth_states(log, sor=False):
    """State after each line (truth, or the system of record: system-feed lines only)."""
    st, out = {}, []
    for line in log["lines"]:
        if not sor or line["kind"] == "struct":
            for ev in line.get("events") or []:
                apply(st, ev)
        out.append(dict(st))
    return out


def read_states(rec):
    out = []
    for snap in rec["snapshots"]:
        st = {}
        for x, (v, _c) in snap.items():
            if isinstance(v, list):
                st[x] = v[0]
                for a, b in zip(v, v[1:]):
                    st.setdefault(a, b)
            else:
                st[x] = v
        out.append(st)
    return out


def windows(log, k=2, seed=0):
    """k windows per question target: [line i, line j] spanning 15-35% of the log, from the log's own clock."""
    rng = random.Random(seed)
    n = len(log["lines"])
    out = []
    for _ in range(k):
        span = rng.randint(int(0.15 * n), int(0.35 * n))
        i = rng.randint(int(0.2 * n), n - span - 1)
        out.append((i, i + span))
    return out


def union_answer(qtype, target, states, i, j, cat):
    got = set()
    for st in states[max(i - 1, 0):j + 1]:  # the state when the window opens, and after every line inside it
        got |= set(C.answer(qtype, target, st, cat))
    return sorted(got)


def questions(log, seed=0):
    cat = C.catalog(log)
    truth = truth_states(log)
    out = []
    for k, q in enumerate(C.questions(log)):
        for w, (i, j) in enumerate(windows(log, k=2, seed=sum(map(ord, log["id"])) * 101 + k * 7 + seed)):  # deterministic
            a = log["lines"][i]["time"]
            b = log["lines"][j]["time"]
            out.append({"log": log["id"], "scenario": log["scenario"], "qtype": q["qtype"], "target": q["target"],
                        "i": i, "j": j, "t1": a, "t2": b, "q": QTEXT[q["qtype"]].format(t=q["target"], a=a, b=b),
                        "a": union_answer(q["qtype"], q["target"], truth, i, j, cat)})
    return out
