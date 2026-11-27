"""NSR1 independent scorer and trace checker (prereg/NSR1_prereg.md, amendment 1).

Re-derives every gold answer from the logs' truth state with its own code (it imports only the catalog's constant
tables, not its question or answer functions), scores each arm's answer file, and checks arm-1 traces:
  validity    every 'in' hop is in the RM final state and the cited log line's reading wrote it (a move of that thing
              to that place, or a swap involving that place); every catalog hop is a catalog fact;
  correctness every hop is true in the truth state / catalog.

python3 nsr/nsr1_score.py --split dev --reader apps/real/out/reader_dev_verify_v4.jsonl [--logs DIR] [--arms ...]
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nsr1_catalog import NEEDS, REGIONS, ROOMS  # noqa: E402  (constant tables only)


def site_facts(log):
    sc, items = log["scenario"], [e["id"] for e in log["registry"]["items"]]
    places = [e["id"] for e in log["registry"]["places"]]
    part, needs = {}, {}
    if sc == "warehouse":
        part = {p: "aisle " + p[0] for p in places if len(p) > 2 and p[0] in "ABCD" and p[1] == "-"}
    elif sc == "home":
        part = {p: ROOMS[p] for p in places if p in ROOMS}
    elif sc == "ops":
        part = {p: REGIONS[p] for p in places if p in REGIONS}
        needs = {s: NEEDS[s] + " primary" for s in items if s in NEEDS and NEEDS[s] and NEEDS[s] + " primary" in items}
    return part, needs


def gold(log, qtype, target):
    part, needs = site_facts(log)
    st = log["truth"]

    def reaches(x):
        seen = set()
        while x not in seen:
            seen.add(x)
            x = st[x] if x in st else part.get(x)
            if x is None:
                return False
            if x == target:
                return True
        return False

    if qtype == "freezer_down":
        return sorted(x for x in st if x.startswith("S-") and reaches(x))
    if qtype in ("aisle_closed", "room_cleared"):
        return sorted(x for x in st if reaches(x))
    if qtype == "region_down":
        return sorted(x for x in st if not x.endswith("primary") and reaches(x))
    if qtype == "host_down":
        return sorted(s for s, d in needs.items() if st.get(d) == target)
    raise ValueError(qtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--reader")
    ap.add_argument("--logs", default=os.path.join(HERE, "..", "apps", "real", "data"))
    ap.add_argument("--arms", nargs="*", default=["arm1", "arm2a", "arm2b", "arm4", "sanity", "negctl"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    logs = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for line in open(os.path.join(a.logs, f"{sc}_{a.split}.jsonl")):
            d = json.loads(line)
            logs[d["id"]] = d
    res = {}
    keys = None
    for arm in a.arms:
        recs = [json.loads(l) for l in open(os.path.join(HERE, "out", f"nsr1_{a.split}{a.tag}_{arm}.jsonl"))]
        k = sorted((r["log"], r["qtype"], r["target"]) for r in recs)
        assert keys is None or k == keys, f"{arm} answers a different question set"
        keys = k
        per = collections.defaultdict(list)
        f1 = [0, 0, 0]
        for r in recs:
            g = gold(logs[r["log"]], r["qtype"], r["target"])
            p = r["pred"]
            ok = p is not None and sorted(p) == g
            per[r["scenario"]].append(ok)
            per["ALL"].append(ok)
            if g:
                per["ALL non-empty"].append(ok)
            ps, gs = set(p or []), set(g)
            f1[0] += len(ps & gs)
            f1[1] += len(ps)
            f1[2] += len(gs)
        prec, rec = f1[0] / max(f1[1], 1), f1[0] / max(f1[2], 1)
        res[arm] = {s: round(sum(v) / len(v), 4) for s, v in per.items()}
        res[arm]["micro_f1"] = round(2 * prec * rec / max(prec + rec, 1e-9), 4)
    none = [not gold(logs[l], q, t) for l, q, t in keys]
    res["always_none"] = round(sum(none) / len(none), 4)
    res["n_questions"] = len(keys)
    if a.reader and "arm1" in a.arms:
        reader = {json.loads(l)["id"]: json.loads(l) for l in open(a.reader)}
        valid = correct = hops = items = 0
        bad = []
        for r in (json.loads(l) for l in open(os.path.join(HERE, "out", f"nsr1_{a.split}{a.tag}_arm1.jsonl"))):
            log, rd = logs[r["log"]], reader[r["log"]]
            part, needs = site_facts(log)
            snap = rd["snapshots"][-1]
            assert set(r["trace"]) == set(r["pred"]), "every answer item needs a trace"
            for x, tr in r["trace"].items():
                items += 1
                v_ok = c_ok = True
                for a_, rel, b, src in tr:
                    hops += 1
                    if src == "catalog":
                        fact = part.get(a_) == b if rel == "part_of" else needs.get(a_) == b
                        v_ok &= fact
                        c_ok &= fact
                        continue
                    in_state = any(a_ == k and (v[0] == b or (isinstance(v[0], list) and v[0][0] == b))
                                   or (isinstance(v[0], list) and a_ in v[0] and v[0][v[0].index(a_) + 1:][:1] == [b])
                                   for k, v in snap.items())
                    evs = rd["events"][src] if isinstance(src, int) else []
                    wrote = any((e[0] == "move" and e[1] == a_ and e[2] == b) or (e[0] == "swap" and b in e[1:])
                                for e in evs or [])
                    v_ok &= in_state and wrote
                    c_ok &= log["truth"].get(a_) == b
                valid += v_ok
                correct += c_ok
                if not v_ok and len(bad) < 5:
                    bad.append((r["log"], x, tr))
        res["trace"] = {"items": items, "hops": hops, "validity": round(valid / max(items, 1), 4),
                        "correctness": round(correct / max(items, 1), 4), "invalid_examples": bad}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
