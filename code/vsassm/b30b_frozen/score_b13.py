"""B13 scorer (prereg/B13_prereg.md). Independent of the systems: gold answers are recomputed from the gold events.

  python3 score_b13.py --check                       # every hand-written answer must equal the event-derived one
  python3 score_b13.py PRED.jsonl [PRED2.jsonl ...]  # score prediction files

A prediction line: {"id": story id, "answers": [one per question], "events": [[op, x, y], ...] (optional),
"conf": [one per question] (optional)}. An answer is a name (where), a list (what_in, chain) or free text (LLMs):
free text is split into names (what_in: on commas / "and"; chain: on ">" / commas / "and"; "nothing" = empty set).
"""
import json
import re
import sys

ACCEPT = "b13_accept.jsonl"
DETS = r"^(?:the|a|an|his|her|their|its|my|your|our)\s+"


def norm(s):
    s = str(s).lower().strip().strip(".,;:!?\"'` ")
    s = re.sub(r"'s\b|'\B", "", s)
    while re.match(DETS, s):
        s = re.sub(DETS, "", s)
    return re.sub(r"\s+", " ", s).strip()


def final_state(story):
    loc = {}
    for op, a, b in story["events"]:
        if op == "write":
            loc[norm(a)] = norm(b)
        else:
            a, b = norm(a), norm(b)
            for k, v in list(loc.items()):
                loc[k] = b if v == a else a if v == b else v
    return loc


def gold_answer(story, q):
    loc = final_state(story)
    ent = norm(re.sub(r"^(where (is|are)|what is (in|on)|who has) ", "", q["q"].lower()).rstrip("?"))
    if q["kind"] == "where":
        return loc.get(ent)
    if q["kind"] == "what_in":
        return sorted(k for k, v in loc.items() if v == ent)
    out, cur = [], ent
    while cur in loc and len(out) < 20:
        cur = loc[cur]
        out.append(cur)
    return out


def canon(q):
    """name -> canonical name, from the question's alias map (canonical names map to themselves)."""
    m = {}
    for c, alts in q["alias"].items():
        for a in [c] + alts:
            m[norm(a)] = norm(c)
    return lambda n: m.get(norm(n), norm(n))


def as_list(pred, kind):
    if pred is None:
        return []
    if isinstance(pred, list):
        return [p for p in pred if p is not None]
    t = norm(pred)
    if kind == "what_in" and re.search(r"\b(nothing|empty|none|no objects?)\b", t):
        return []
    parts = re.split(r"\s*(?:>|->|,|;|\band\b|\bthen\b|\binside\b|\bwhich is in\b)\s*", t)
    return [p for p in (norm(x) for x in parts) if p]


def correct(q, pred, places=()):
    c = canon(q)
    if q["kind"] == "where":
        if pred is None:
            return False, False
        strict = c(pred) == norm(q["a"])
        names = [norm(q["a"])] + [norm(a) for a in q["alias"].get(q["a"], [])]
        t = " " + norm(pred) + " "
        others = [p for p in places if p not in names and not any(p in n for n in names)]
        lenient = strict or (any(f" {n} " in t for n in names) and not any(f" {p} " in t for p in others))
        return strict, lenient
    got = [c(x) for x in as_list(pred, q["kind"])]
    gold = [norm(x) for x in q["a"]]
    ok = sorted(set(got)) == sorted(gold) if q["kind"] == "what_in" else got == gold
    if q["kind"] == "chain" and not ok:  # lenient (added after the baselines, disclosed): the chain may start with
        ent = norm(re.sub(r"^(where (is|are)) ", "", q["q"].lower()).rstrip("?"))  # the asked-about object itself
        return False, bool(got) and got[0] == ent and got[1:] == gold
    return ok, ok


def check():
    bad = 0
    stories = [json.loads(l) for l in open(ACCEPT)]
    for s in stories:
        for q in s["questions"]:
            g = gold_answer(s, q)
            h = norm(q["a"]) if q["kind"] == "where" else sorted(map(norm, q["a"])) if q["kind"] == "what_in" \
                else [norm(x) for x in q["a"]]
            if g != h:
                bad += 1
                print(f"MISMATCH {s['id']} {q['q']!r}: hand-written {h} vs events {g}")
    nq = sum(len(s["questions"]) for s in stories)
    by = {}
    for s in stories:
        by[s["cat"]] = by.get(s["cat"], 0) + 1
    print(f"{len(stories)} stories {by}, {nq} questions, {bad} mismatches")
    return bad


def ev_prf(gold, got):
    g = [(o, norm(a), norm(b)) for o, a, b in gold]
    p = [(o, norm(a) if a else None, norm(b) if b else None) for o, a, b in got]
    gs = list(g)
    tp = 0
    for e in p:
        if e in gs:
            gs.remove(e)
            tp += 1
    return tp, len(p), len(g)


def score(path):
    stories = {s["id"]: s for s in map(json.loads, open(ACCEPT))}
    rows = [json.loads(l) for l in open(path)]
    res = {}
    lowconf_wrong = wrong = 0
    tp = np_ = ng = 0
    for r in rows:
        s = stories[r["id"]]
        dom = s["domain"]
        places = sorted({norm(b) for _, a, b in s["events"]} | {norm(a) for o, a, _ in s["events"] if o == "swap"})
        ok_all = True
        for i, q in enumerate(s["questions"]):
            st, le = correct(q, r["answers"][i], places)
            d = res.setdefault(dom, {"n": 0, "strict": 0, "lenient": 0, "stories": 0, "stories_ok": 0, "cat": {}})
            d["n"] += 1
            d["strict"] += st
            d["lenient"] += le
            cd = d["cat"].setdefault(s["cat"], [0, 0])
            cd[0] += st
            cd[1] += 1
            ok_all &= st
            if not st:
                wrong += 1
                if r.get("conf") and r["conf"][i] is not None and r["conf"][i] < 0.5:
                    lowconf_wrong += 1
        res[dom]["stories"] += 1
        res[dom]["stories_ok"] += ok_all
        if "events" in r:
            a, b, c = ev_prf(s["events"], r["events"])
            tp, np_, ng = tp + a, np_ + b, ng + c
    print(f"=== {path}")
    for dom, d in res.items():
        cats = " ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(d["cat"].items()))
        print(f"  {dom:10s} Q-acc {d['strict'] / d['n']:.3f} ({d['strict']}/{d['n']})  lenient {d['lenient'] / d['n']:.3f}"
              f"  stories {d['stories_ok']}/{d['stories']}  by category: {cats}")
    if np_:
        print(f"  events: precision {tp / np_:.3f} recall {tp / ng:.3f} ({tp} right of {np_} parsed, {ng} gold)")
    if any(r.get("conf") for r in rows):
        print(f"  wrong answers with confidence < 0.5: {lowconf_wrong}/{wrong}")
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--accept"]:  # B14: score against another acceptance set
        ACCEPT = args[1]
        args = args[2:]
    if args == ["--check"]:
        sys.exit(1 if check() else 0)
    for p in args:
        score(p)
