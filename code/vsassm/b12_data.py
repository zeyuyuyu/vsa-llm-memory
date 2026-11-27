"""B12 data (prereg/B12_prereg.md): T2 stories with pronouns and with two events in one sentence.

Each episode: text (preamble + numbered lines + question), units (one per clause: span, mentions, op, ref), acts,
query, ents, places. A pronoun mention carries the pronoun as its name and the referent in `ref`. Several clauses can
share a numbered line; the question is the last unit.
"""
import random
import re

from b6_tasks import gen_t2  # noqa: F401  (B6's generator is not used: B12 needs repeats)
from b9_data import FAMILIES, POOLS, TRAIN_FAMILIES, _names

JOINERS = [", then ", ", and then ", "; ", ", and after that "]
PLACE_PREP = re.compile(r"\b(in|into|to|on|inside)\s+the\s+\{p\}")
OBJ_SLOT = re.compile(r"\b([Tt])he\s+\{o\}")


def plural(name):
    return name.endswith("s") and not name.endswith("ss")


def gen_events(rng, n, p_obj=0.3, p_place=0.2):
    """T2 events with swaps at round(n/4) random positions; puts sometimes repeat the last put's object or place."""
    swaps = set(rng.sample(range(1, n), round(n / 4))) if n > 1 else set()
    acts, last_put = [], None
    for t in range(n):
        if t in swaps:
            acts.append(["swap", *rng.sample(range(16), 2)])
            continue
        o, p = rng.randrange(16), rng.randrange(16)
        if last_put and rng.random() < p_obj:
            o = last_put[1]
        if last_put and rng.random() < p_place:
            p = last_put[2]
        if last_put and o == last_put[1] and p == last_put[2]:
            p = rng.choice([x for x in range(16) if x != p])
        acts.append(["put", o, p])
        last_put = acts[-1]
    return acts


def render_clause(rng, fam, act, ents, places, prev_put, p_pro=0.6):
    """One clause (no final period) + its mentions as (start, end, role, name) relative to the clause, + referents."""
    op, a, b = act
    if op == "swap":
        tpl = rng.choice(fam["T2_SWAP"])
        names = {"a": places[a], "b": places[b]}
        roles = {"a": "sa", "b": "sb"}
    else:
        tpl = rng.choice(fam["T2_PUT"])
        names = {"o": ents[a], "p": places[b]}
        roles = {"o": "x", "p": "y"}
        if prev_put and prev_put[1] == a and OBJ_SLOT.search(tpl) and rng.random() < p_pro:
            pro = "them" if plural(ents[a]) else "it"
            tpl = OBJ_SLOT.sub(lambda m: (pro.capitalize() if m.group(1) == "T" else pro), tpl, count=1)
            names.pop("o")
            names["PRO_O"] = pro
        if prev_put and prev_put[2] == b and PLACE_PREP.search(tpl) and rng.random() < p_pro:
            tpl = PLACE_PREP.sub("there", tpl, count=1)
            names.pop("p")
            names["PRO_P"] = "there"
    text, ments, refs = "", [], {}
    pos = 0
    for m in re.finditer(r"\{(\w+)\}|\b(It|it|Them|them|there)\b", tpl):
        text += tpl[pos:m.start()]
        if m.group(1):
            field = m.group(1)
            nm = names[field]
            ments.append((len(text), len(text) + len(nm) - 1, roles[field], nm))
            text += nm
        else:
            word = m.group(2)
            role = "y" if word == "there" else "x"
            if (role == "x" and "PRO_O" in names) or (role == "y" and "PRO_P" in names):
                ments.append((len(text), len(text) + len(word) - 1, role, word.lower()))
                refs[role] = ents[a] if role == "x" else places[b]
            text += word
        pos = m.end()
    text += tpl[pos:]
    return text.rstrip("."), ments, refs


def episode(rng, n, family, names_split, p_merge=0.3):
    fam = FAMILIES[family]
    ents, places = _names("T2", names_split, rng)
    acts = gen_events(rng, n)
    plist = ", ".join(f"the {x}" for x in places[:-1]) + f" and the {places[-1]}"
    text = rng.choice(fam["T2_PRE"]).format(places=plist) + "\n\n"
    units, line, t, prev_put = [], 0, 0, None
    while t < len(acts):
        k = 2 if t + 1 < len(acts) and rng.random() < p_merge else 1
        line += 1
        prefix = f"{line}. "
        body, local_units = "", []
        for j in range(k):
            clause, ments, refs = render_clause(rng, fam, acts[t + j], ents, places, prev_put)
            if j:
                joiner = rng.choice(JOINERS)
                clause = clause[0].lower() + clause[1:] if not clause.startswith(("The ", "In ", "Into ")) else clause[0].lower() + clause[1:]
                body += joiner
            start = len(body)
            body += clause
            local_units.append((start, len(body) - 1, ments, refs, acts[t + j]))
            if acts[t + j][0] == "put":
                prev_put = acts[t + j]
        body += "."
        off = len(text) + len(prefix)
        for start, end, ments, refs, act in local_units:
            units.append({"span": (off + start, off + end),
                          "mentions": [(off + start + a, off + start + b, r, nm) for a, b, r, nm in ments],
                          "op": "swap" if act[0] == "swap" else "write", "ref": refs})
        text += prefix + body + "\n"
        t += k
    # the question
    loc = {}
    for op, a, b in acts:
        if op == "put":
            loc[a] = b
        else:
            loc = {o: (b if c == a else a if c == b else c) for o, c in loc.items()}
    q = rng.choice(sorted(loc))
    qt, _ = rng.choice(fam["T2_Q"]), None
    qtext = qt.format(o=ents[q])
    text += "\n"
    i = qtext.find(ents[q])
    units.append({"span": (len(text), len(text) + len(qtext) - 1),
                  "mentions": [(len(text) + i, len(text) + i + len(ents[q]) - 1, "q", ents[q])], "op": "question", "ref": {}})
    text += qtext
    return {"task": "T2", "n": n, "family": family, "names": names_split, "text": text, "sents": units, "acts": acts,
            "query": q, "ents": ents, "places": places, "answer": places[loc[q]]}


def train_set(n_episodes, seed):
    rng = random.Random(seed)
    return [episode(rng, rng.randint(5, 40), rng.choice(TRAIN_FAMILIES), "train") for _ in range(n_episodes)]


def test_set(seed=1401, ns=(10, 20, 40, 80, 160), per_n=40):
    rng = random.Random(seed)
    out = []
    for fam in ("B6", "C", "F"):
        for n in ns:
            for i in range(per_n):
                e = episode(rng, n, fam, "test2")
                e["idx"], e["split"] = i, fam
                out.append(e)
    return out
