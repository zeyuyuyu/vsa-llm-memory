"""T3 translator prompt and engine call for ZebraLogic puzzles. Few-shot gold DSLs: data/zebra_fewshot.jsonl (three
puzzles excluded from the test sample; each DSL verified to give the published solution, uniquely)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine.sre_engine import DSLError  # noqa: E402
from engine.zebra_engine import solve  # noqa: E402

SCHEMA = """Translate the logic puzzle into JSON with exactly this schema:
{"houses": <number of houses>,
 "attributes": {"<Attribute>": ["<value>", ...], ...},
 "clues": [ ... one object per fact ... ]}
Use the attribute names given below, in the order the characteristics are listed, and copy each value exactly as it is
written between backticks. A reference to a person is ["<Attribute>", "<value>"]. Clue objects:
 {"type": "same", "a": ref, "b": ref}            a and b are the same person
 {"type": "not_same", "a": ref, "b": ref}        a and b are different people
 {"type": "at", "a": ref, "house": k}            a is in house k
 {"type": "not_at", "a": ref, "house": k}        a is not in house k
 {"type": "left_of", "a": ref, "b": ref}         a is somewhere to the left of b
 {"type": "right_of", "a": ref, "b": ref}        a is somewhere to the right of b
 {"type": "directly_left", "a": ref, "b": ref}   a is directly left of b
 {"type": "directly_right", "a": ref, "b": ref}  a is directly right of b
 {"type": "next_to", "a": ref, "b": ref}         a and b are next to each other
 {"type": "gap", "a": ref, "b": ref, "between": k}  there are k houses between a and b
Reply with the JSON object only."""


def prompt_translate(it, shots):
    parts = [SCHEMA]
    for k, ex in enumerate(shots, 1):
        parts.append("Example %d\nAttribute names: %s\n%s\nJSON:\n%s" % (k, ", ".join(ex["attributes"]), ex["text"],
                                                                         json.dumps(ex["dsl"], separators=(",", ":"))))
    parts.append("Now translate\nAttribute names: %s\n%s\nJSON:" % (", ".join(it["attributes"]), it["text"]))
    return "\n\n".join(parts)


def load_shots(path=os.path.join(HERE, "..", "data", "zebra_fewshot.jsonl")):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def answer_from(dsl):
    if dsl is None:
        return {"abstain": True, "reason": "no translation"}
    try:
        ans, cert = solve(dsl)
        return {"pred": ans, "cert": cert}
    except (DSLError, KeyError, TypeError, ValueError) as e:
        return {"abstain": True, "reason": "engine rejected the translation: %r" % (e,)}


def _canon_clue(c):
    if not isinstance(c, dict):
        return None
    t = str(c.get("type", "")).strip().lower()
    ref = lambda r: tuple(str(x).strip().lower() for x in r) if isinstance(r, (list, tuple)) else None  # noqa: E731
    a, b = ref(c.get("a")), ref(c.get("b"))
    if t in ("same", "not_same", "next_to"):
        a, b = sorted([a or (), b or ()])
    if t == "right_of":
        t, a, b = "left_of", b, a
    if t == "directly_right":
        t, a, b = "directly_left", b, a
    if t == "gap":
        a, b = sorted([a or (), b or ()])
    return json.dumps([t, a, b, c.get("house"), c.get("between")])


def vote(dsls):
    """Majority vote for k sampled puzzle translations: houses and attributes by majority, a clue kept if a majority of
    the parsed samples contain it (after canonicalising symmetric and mirrored forms)."""
    import collections
    dsls = [d for d in dsls if isinstance(d, dict) and isinstance(d.get("clues"), list)]
    if not dsls:
        return None, "no parsable sample"
    need = len(dsls) // 2 + 1
    h = collections.Counter(json.dumps(d.get("houses")) for d in dsls).most_common(1)[0]
    at = collections.Counter(json.dumps(d.get("attributes"), sort_keys=True) for d in dsls).most_common(1)[0]
    if h[1] < need or at[1] < need:
        return None, "no majority for houses or attributes"
    counts, first = collections.Counter(), {}
    for d in dsls:
        seen = set()
        for c in d["clues"]:
            k = _canon_clue(c)
            if k and k not in seen:
                seen.add(k)
                counts[k] += 1
                first.setdefault(k, c)
    clues = [first[k] for k, n in counts.items() if n >= need]
    return {"houses": json.loads(h[0]), "attributes": json.loads(at[0]), "clues": clues}, None
