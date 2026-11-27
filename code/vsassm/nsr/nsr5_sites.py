"""NSR5 (scale): multi-day sites built by running K logs of one scenario back to back (day k's events continue from day
k-1's state; items placed on day 1 and never touched again stay where they are). Gold = replay of the concatenated true
events; the pipeline = replay of the concatenated reader events (each day read by the shipped reader with its own channel
context, i.e. the reader is reset at each day boundary). Questions: NSR1's impact questions over the merged site.
Sources: the retired test, nsr1 and nsr2 (30 logs per scenario); test2 is never used.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr2 as M  # noqa: E402
import nsr3_questions as Q3  # noqa: E402

M.SPLITS.setdefault("nsr2", ("nsr/data", "nsr/out/held/reader_nsr2_verify_v4.jsonl"))
SOURCES = ["test", "nsr1", "nsr2"]


def merge(logs, recs):
    """One multi-day site from consecutive day logs (same scenario)."""
    items, places = {}, {}
    for log in logs:
        for e in log["registry"]["items"]:
            items.setdefault(e["id"], {"id": e["id"], "aliases": list(e["aliases"])})
        for e in log["registry"]["places"]:
            places.setdefault(e["id"], {"id": e["id"], "aliases": list(e["aliases"])})
    lines, text, events = [], [], []
    for d, (log, rec) in enumerate(zip(logs, recs)):
        text.append(f"=== Day {d + 1} ===")
        lines.append({"kind": "struct", "src": "day", "events": [], "time": None})
        events.append([])
        lines += log["lines"]
        text += log["text_lines"]
        events += rec["events"]
    site = {"id": "+".join(l["id"] for l in logs), "scenario": logs[0]["scenario"], "lines": lines, "text_lines": text,
            "registry": {"items": list(items.values()), "places": list(places.values())}}
    truth = {}
    for line in lines:
        for ev in line.get("events") or []:
            Q3.apply(truth, ev)
    site["truth"] = truth
    sor = {}
    for line in lines:
        if line["kind"] == "struct":
            for ev in line.get("events") or []:
                Q3.apply(sor, ev)
    site["sor"] = sor
    read = {}
    for evs in events:
        for ev in evs or []:
            Q3.apply(read, ev)
    return site, read


def sites(K):
    out = []
    for sc in ("warehouse", "biobank", "home", "ops"):
        pool = []
        for sp in SOURCES:
            logs, rd = M.load(sp)
            pool += [(logs[k], rd[k]) for k in sorted(logs) if logs[k]["scenario"] == sc]
        for s in range(0, len(pool) - K + 1, K):
            chunk = pool[s:s + K]
            out.append(merge([c[0] for c in chunk], [c[1] for c in chunk]))
    return out
