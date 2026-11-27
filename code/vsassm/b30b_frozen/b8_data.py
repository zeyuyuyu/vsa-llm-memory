"""B8 data (prereg/B8_prereg.md): more training wording (families A, B, D, E), held-out family F, and T3' (a move
sequence that always keeps >= 8 outermost containers, so the answer cannot be guessed from one giant tree).

Episodes have the B7 structure (task, n, text, ends, acts, query, answer, states) plus `spans`: (start, end) characters
of each action sentence and of the question, for attention pooling over the sentence's tokens.
"""
import json
import random
import re

from b6_tasks import ITEMS, OBJ, PLACE, gen_t2, prompt_t2, prompt_t3
from b7_data import AB, C, _assemble, root, states_t2, states_t3

D = {
    "T2_PRE": ["We keep track of 16 places: {places}. Nothing is in any of them at first. Each event below happens "
               "after the one before it. Moving an object takes it out of its old place; when two places exchange "
               "contents, both sets of objects move at the same time."],
    "T2_PUT": ["The {o} is moved to the {p}.", "In the {p}, someone leaves the {o}.", "They place the {o} into the {p}."],
    "T2_SWAP": ["The contents of the {a} and the {b} are exchanged.",
                "Someone swaps what is in the {a} with what is in the {b}."],
    "T2_Q": ["Tell me where the {o} is now.", "Think about the {o}. Which place is it in after the last event?"],
    "T3_PRE": ["There are 40 containers, named by colour and kind. All of them start out on their own. In each step "
               "below one container, with whatever it holds, is moved into another and leaves its previous holder."],
    "T3_MOVE": ["The {a} is put inside the {b}.", "The {b} receives the {a}.", "They move the {a} into the {b}."],
    "T3_Q": ["Which container, sitting inside nothing, contains the {q} directly or indirectly?",
             "Trace the {q} outward. What is the last container you reach?"],
}
E = {
    "T2_PRE": ["Sixteen locations exist ({places}), and they begin empty. Follow the steps in order. A moved object "
               "leaves its earlier location; swapping two locations trades everything inside them."],
    "T2_PUT": ["The {p} now gets the {o}.", "We drop the {o} in the {p}.", "The {o} is carried over to the {p}."],
    "T2_SWAP": ["All items in the {a} switch places with all items in the {b}.",
                "The {a} and the {b} have their contents traded."],
    "T2_Q": ["After all of this, the {o} is in which place?", "The {o}: where is it at the end?"],
    "T3_PRE": ["Forty containers, each a colour plus a kind, are lying separately. Read the steps in order; whenever a "
               "container is moved, everything inside it comes along and it is removed from its former holder."],
    "T3_MOVE": ["Someone tucks the {a} into the {b}.", "The {b} now holds the {a}.", "The {a} gets placed in the {b}."],
    "T3_Q": ["The {q} is nested in some containers. Name the outermost one.",
             "Going outward from the {q}, which container is not inside any other?"],
}
# Held-out family F: test only, written before any B8 run; reversed order and new verbs.
F = {
    "T2_PRE": ["Here are 16 spots ({places}); each is empty to begin with. The record below is in time order. When "
               "something is stashed, it is taken from wherever it was; when two spots trade contents, everything in "
               "each goes to the other."],
    "T2_PUT": ["Somebody stashes the {o} in the {p}.", "The {p} is where the {o} goes next."],
    "T2_SWAP": ["Contents of the {a} and the {b} get swapped around."],
    "T2_Q": ["Where would you find the {o} once everything is done?"],
    "T3_PRE": ["This is a record about 40 containers, each known by a colour and a kind. To begin, none holds "
               "another. When a container is nested somewhere new, what is inside it goes along and it leaves its old "
               "place."],
    "T3_MOVE": ["Somebody nests the {a} inside the {b}.", "The {b} takes in the {a}."],
    "T3_Q": ["Starting from the {q}, keep going to whatever holds it. Which container do you end at?"],
}
TRAIN_FAMILIES = [AB, D, E]
TEST_FAMILIES = {"C": C, "F": F}


def gen_t3p(rng, n, min_roots=8):
    """T3': like B6's T3 generator, but never fewer than `min_roots` outermost containers."""
    while True:
        parent, moves = {}, []
        for _ in range(n):
            while True:
                a, b = rng.sample(range(len(ITEMS)), 2)
                if parent.get(a) == b:
                    continue
                x, cyc = b, False
                while x is not None:
                    if x == a:
                        cyc = True
                        break
                    x = parent.get(x)
                if cyc:
                    continue
                n_roots = sum(1 for i in range(len(ITEMS)) if i not in parent)
                if a not in parent and n_roots <= min_roots:
                    continue  # nesting a current outermost container would drop below min_roots
                break
            parent[a] = b
            moves.append([a, b])

        def depth(i):
            d = 0
            while i in parent:
                i, d = parent[i], d + 1
            return d

        def top(i):
            while i in parent:
                i = parent[i]
            return i
        # amendment 1: pick a tree uniformly among >= 3 trees that hold an item 2-6 deep, then an item in it, so that
        # "answer the largest tree's outermost container" is not a good guess
        deep = {}
        for i in range(len(ITEMS)):
            if 2 <= depth(i) <= 6:
                deep.setdefault(top(i), []).append(i)
        if len(deep) >= 3:
            tree = rng.choice(sorted(deep))
            return {"moves": moves, "query": rng.choice(deep[tree])}


def spans(text, ends):
    out = []
    for e in ends[:-1]:
        ls = text.rfind("\n", 0, e) + 1
        out.append((ls + re.match(r"\d+\. ", text[ls:]).end(), e))
    qs = ends[-2] + 1
    while text[qs] in "\n ":
        qs += 1
    out.append((qs, ends[-1]))
    return out


def episode(task, rng, n, fam):
    if task == "T2":
        p = gen_t2(rng, n)
        acts, q = p["acts"], p["query"]
        sents = [rng.choice(fam["T2_PUT"]).format(o=OBJ[a], p=PLACE[b]) if op == "put"
                 else rng.choice(fam["T2_SWAP"]).format(a=PLACE[a], b=PLACE[b]) for op, a, b in acts]
        places = ", ".join(f"the {x}" for x in PLACE[:-1]) + f" and the {PLACE[-1]}"
        text, ends = _assemble(rng.choice(fam["T2_PRE"]).format(places=places), sents,
                               rng.choice(fam["T2_Q"]).format(o=OBJ[q]))
        st = states_t2(acts)
        ans = st[-1][q]
    else:
        p = gen_t3p(rng, n)
        acts, q = p["moves"], p["query"]
        sents = [rng.choice(fam["T3_MOVE"]).format(a=ITEMS[a], b=ITEMS[b]) for a, b in acts]
        text, ends = _assemble(rng.choice(fam["T3_PRE"]), sents, rng.choice(fam["T3_Q"]).format(q=ITEMS[q]))
        st = states_t3(acts)
        ans = root(st[-1], q)
    return {"task": task, "n": n, "text": text, "ends": ends, "spans": spans(text, ends), "acts": acts, "query": q,
            "answer": ans, "states": st}


def b6_wording(task, p, n, idx):
    """An episode in B6's exact wording (its prompt) for problem p."""
    if task == "T2":
        text, acts = prompt_t2(p), p["acts"]
    else:
        text, acts = prompt_t3(p), p["moves"]
    ends = [m.end() - 1 for m in re.finditer(r"^\d+\. [^\n]*\.$", text, re.M)]
    ends.append(text.rindex("?"))
    assert len(ends) == len(acts) + 1
    st = states_t2(acts) if task == "T2" else states_t3(acts)
    ans = st[-1][p["query"]] if task == "T2" else root(st[-1], p["query"])
    return {"task": task, "n": n, "idx": idx, "text": text, "ends": ends, "spans": spans(text, ends), "acts": acts,
            "query": p["query"], "answer": ans, "states": st}


def train_set(task, n_episodes, seed):
    rng = random.Random(seed)
    lo = 5 if task == "T2" else 6  # T3' needs >= 6 moves to form 3 trees with an item 2 deep (amendment 1)
    return [episode(task, rng, rng.randint(lo, 40), rng.choice(TRAIN_FAMILIES)) for _ in range(n_episodes)]


def test_sets(task, seed, ns=(10, 20, 40, 80, 160)):
    """{wording: episodes}: B6 wording (T2: B6's own problems; T3': new T3' problems in B6's templates), C, F."""
    rng = random.Random(seed)
    out = {"b6": [], "C": [], "F": []}
    if task == "T2":
        for r in map(json.loads, open("b6_problems.jsonl")):
            if r["task"] == "T2":
                out["b6"].append(b6_wording("T2", r["problem"], r["n"], r["idx"]))
    else:
        for n in ns:
            for i in range(40):
                out["b6"].append(b6_wording("T3", gen_t3p(rng, n), n, i))
    for name in ("C", "F"):
        for n in ns:
            for i in range(100):
                e = episode(task, rng, n, TEST_FAMILIES[name])
                e["idx"] = i
                out[name].append(e)
    return out


def t3p_b6_problems(path, seed=8301):
    """The T3' B6-wording problems in B6's JSONL format, for the Qwen3 thinking reference (same seed as test_sets)."""
    eps = test_sets("T3", seed)["b6"]
    with open(path, "w") as f:
        for e in eps:
            p = {"moves": e["acts"], "query": e["query"]}
            f.write(json.dumps({"task": "T3", "n": e["n"], "idx": e["idx"], "problem": p, "gold": ITEMS[e["answer"]],
                                "prompt": prompt_t3(p)}) + "\n")
