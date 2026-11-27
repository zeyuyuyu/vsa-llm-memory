"""B7 episodes (prereg/B7_prereg.md): paraphrased T2/T3 text with per-sentence event labels.

Every episode is a dict: task, n, text, ends (character index of the "." ending each action sentence, then of the
question's "?"), acts (T2: ["put", o, p] / ["swap", a, b]; T3: [a, b]), query, answer, and per-step states (T2:
location of each object after each step, -1 = not placed; T3: container of each item after each step, -1 = none).
Training wording never uses B6's sentences. `from_b6(row)` builds the same structure from a B6 problem.
"""
import random
import re

from b6_tasks import ITEMS, OBJ, PLACE, gen_t2, gen_t3

T2_PRE = [
    "There are 16 places: {places}. Every place starts out empty. Here is what happens, in order. When an object is "
    "moved, it leaves wherever it was. When the contents of two places are exchanged, everything in each place moves "
    "to the other place at the same moment.",
    "Sixteen places are available ({places}); all of them start empty. Read the events below in order. Moving an "
    "object takes it out of its old place; exchanging two places moves all of their contents across at once.",
]
T2_PUT = ["Move the {o} to the {p}.", "The {o} goes into the {p}.", "Someone places the {o} in the {p}.",
          "Drop the {o} into the {p}."]
T2_SWAP = ["Exchange everything in the {a} with everything in the {b}.",
           "Whatever is in the {a} and whatever is in the {b} trade places.",
           "Switch the contents of the {a} and the {b}."]
T2_Q = ["Where is the {o} now?", "Which place has the {o} at the end?", "In which place does the {o} end up?"]

T3_PRE = [
    "There are 40 containers, each named by a colour and a kind (such as the green crate). At first none is inside "
    "another. The moves below happen in order; moving a container carries everything inside it, and takes it out of "
    "whatever held it before.",
    "Forty containers (a colour plus a kind, for example the white bag) start out separate. Follow these moves in "
    "order. A moved container brings its contents along and leaves its previous holder.",
]
T3_MOVE = ["Move the {a} into the {b}.", "The {a} is placed inside the {b}.", "Someone puts the {a} in the {b}.",
           "Pack the {a} into the {b}."]
T3_Q = ["Which container, not itself inside anything, ultimately holds the {q}?",
        "What is the outermost container around the {q}?"]

# Family C (B7b): held-out wording, never trained on and never used to diagnose; some templates reverse the word order.
C = {
    "T2_PRE": ["Below is a log of events involving 16 places ({places}), all empty at the start. An object that is put "
               "somewhere leaves its previous place. When two places swap contents, all of the objects in each move to "
               "the other."],
    "T2_PUT": ["Next, the {o} is put in the {p}.", "Into the {p} goes the {o}."],
    "T2_SWAP": ["The {a} and the {b} swap all of their contents.",
                "Everything in the {a} trades places with everything in the {b}."],
    "T2_Q": ["At the very end, where can the {o} be found?"],
    "T3_PRE": ["Here is a log about 40 containers, each called by a colour and a kind. Nothing is inside anything at "
               "first. When a container is moved, its contents travel with it and it leaves its old holder."],
    "T3_MOVE": ["Next, the {a} goes inside the {b}.", "Inside the {b}, someone sets the {a}."],
    "T3_Q": ["Follow the {q} outward through every container that holds it. Which container is on the outside of all "
             "of them?"],
}
AB = {"T2_PRE": T2_PRE, "T2_PUT": T2_PUT, "T2_SWAP": T2_SWAP, "T2_Q": T2_Q, "T3_PRE": T3_PRE, "T3_MOVE": T3_MOVE,
      "T3_Q": T3_Q}


def states_t2(acts):
    loc, out = [-1] * len(OBJ), []
    for op, a, b in acts:
        if op == "put":
            loc[a] = b
        else:
            loc = [b if c == a else a if c == b else c for c in loc]
        out.append(list(loc))
    return out


def states_t3(moves):
    par, out = [-1] * len(ITEMS), []
    for a, b in moves:
        par[a] = b
        out.append(list(par))
    return out


def root(par, i):
    while par[i] != -1:
        i = par[i]
    return i


def _assemble(pre, sentences, question):
    text, ends = pre + "\n\n", []
    for i, s in enumerate(sentences):
        line = f"{i + 1}. {s}"
        ends.append(len(text) + len(line) - 1)
        text += line + "\n"
    text += "\n" + question
    ends.append(len(text) - 1)
    return text, ends


def episode(task, rng, n, family="AB"):
    """A paraphrased training/test episode with n actions; family "AB" (training wording) or "C" (held out)."""
    L = AB if family == "AB" else C
    if task == "T2":
        p = gen_t2(rng, n) if n > 1 else {"acts": [["put", rng.randrange(16), rng.randrange(16)]], "query": None}
        acts, q = p["acts"], p["query"]
        sents = [rng.choice(L["T2_PUT"]).format(o=OBJ[a], p=PLACE[b]) if op == "put"
                 else rng.choice(L["T2_SWAP"]).format(a=PLACE[a], b=PLACE[b]) for op, a, b in acts]
        places = ", ".join(f"the {x}" for x in PLACE[:-1]) + f" and the {PLACE[-1]}"
        text, ends = _assemble(rng.choice(L["T2_PRE"]).format(places=places), sents,
                               rng.choice(L["T2_Q"]).format(o=OBJ[q]))
        st = states_t2(acts)
        return {"task": task, "n": n, "text": text, "ends": ends, "acts": acts, "query": q,
                "answer": st[-1][q], "states": st}
    p = gen_t3(rng, n)
    moves, q = p["moves"], p["query"]
    sents = [rng.choice(L["T3_MOVE"]).format(a=ITEMS[a], b=ITEMS[b]) for a, b in moves]
    text, ends = _assemble(rng.choice(L["T3_PRE"]), sents, rng.choice(L["T3_Q"]).format(q=ITEMS[q]))
    st = states_t3(moves)
    return {"task": task, "n": n, "text": text, "ends": ends, "acts": moves, "query": q,
            "answer": root(st[-1], q), "states": st}


def from_b6(row):
    """B6 problem -> episode structure; text = the exact B6 prompt (B6 wording, never seen in training)."""
    text, p = row["prompt"], row["problem"]
    ends = [m.end() - 1 for m in re.finditer(r"^\d+\. [^\n]*\.$", text, re.M)]
    ends.append(text.rindex("?"))
    acts = p["acts"] if row["task"] == "T2" else p["moves"]
    assert len(ends) == len(acts) + 1 == row["n"] + 1, (row["task"], row["n"], row["idx"])
    st = states_t2(acts) if row["task"] == "T2" else states_t3(acts)
    ans = st[-1][p["query"]] if row["task"] == "T2" else root(st[-1], p["query"])
    return {"task": row["task"], "n": row["n"], "idx": row["idx"], "text": text, "ends": ends, "acts": acts,
            "query": p["query"], "answer": ans, "states": st}


def train_set(task, n_episodes, seed):
    rng = random.Random(seed)
    return [episode(task, rng, rng.randint(5, 40)) for _ in range(n_episodes)]


def fresh_set(task, seed, per_n=200, ns=(10, 20, 40, 80, 160), family="AB"):
    rng = random.Random(seed)
    out = []
    for n in ns:
        for i in range(per_n):
            e = episode(task, rng, n, family)
            e["idx"] = i
            out.append(e)
    return out
