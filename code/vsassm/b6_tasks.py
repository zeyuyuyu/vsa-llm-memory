"""B6 tasks (prereg/B6_prereg.md + amendment 1): synthetic state-tracking problems with exactly computable answers.

`python3 b6_tasks.py b6_problems.jsonl` writes all 3 tasks x 5 N x 40 problems, the same for every model.
"""
import json
import random
import sys

NS = (10, 20, 40, 80, 160)
N_PROBLEMS = 40
TASKS = ("T1", "T2", "T3")

VARS = list("abcdef")
OBJ = ["keys", "phone", "wallet", "glasses", "red cup", "mug", "remote", "charger", "passport", "watch",
       "scissors", "tape", "spoon", "notebook", "headphones", "batteries"]
PLACE = ["top drawer", "middle drawer", "bottom drawer", "desk drawer", "shelf 1", "shelf 2", "shelf 3",
         "left cabinet", "right cabinet", "basket", "toolbox", "tray", "box a", "box b", "bin", "counter"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
KINDS = ["box", "bag", "crate", "basket", "case"]
ITEMS = [f"{c} {k}" for c in COLORS for k in KINDS]

ASK = 'Think it through, then give your final answer on the last line in the form "Answer: <answer>".'


def cell_seed(task, n):
    return 6000 + 1000 * TASKS.index(task) + n


# ---------------------------------------------------------------- T1 variable tracking
def gen_t1(rng, n):
    init = {v: rng.randrange(100) for v in VARS}
    stmts = []
    for _ in range(n):
        r, x = rng.random(), rng.choice(VARS)
        y = rng.choice([v for v in VARS if v != x])
        if r < 0.15:
            stmts.append(["set", x, rng.randrange(100)])
        elif r < 0.50:
            stmts.append(["add", x, y])
        elif r < 0.80:
            stmts.append(["sub", x, y])
        else:
            stmts.append(["swap", x, y])
    return {"init": init, "stmts": stmts, "query": rng.choice(VARS)}


def solve_t1(p):
    val = dict(p["init"])
    for op, x, y in p["stmts"]:
        if op == "set":
            val[x] = y
        elif op == "add":
            val[x] = (val[x] + val[y]) % 100
        elif op == "sub":
            val[x] = (val[x] - val[y]) % 100
        else:
            val[x], val[y] = val[y], val[x]
    return val[p["query"]]


def prompt_t1(p):
    text = {"set": "{x} = {y}", "add": "{x} = {x} + {y}", "sub": "{x} = {x} - {y}", "swap": "swap {x}, {y}"}
    lines = [f"{v} = {p['init'][v]}" for v in VARS] + [text[op].format(x=x, y=y) for op, x, y in p["stmts"]]
    return ("Here is a short program over six integer variables a, b, c, d, e and f. All values are kept modulo 100: "
            "after every statement each value is reduced to 0..99 (so 95 + 10 gives 5, and 3 - 8 gives 95). "
            "`swap x, y` exchanges the values of x and y.\n\n```\n" + "\n".join(lines) + "\n```\n\n"
            f"What is the final value of {p['query']}? {ASK}")


# ---------------------------------------------------------------- T2 object tracking
def gen_t2(rng, n):
    swaps = set(rng.sample(range(1, n), round(n / 4)))
    acts = []
    for t in range(n):
        if t in swaps:
            acts.append(["swap", *rng.sample(range(len(PLACE)), 2)])
        else:
            acts.append(["put", rng.randrange(len(OBJ)), rng.randrange(len(PLACE))])
    placed = sorted({o for op, o, _ in acts if op == "put"})
    return {"acts": acts, "query": rng.choice(placed)}


def solve_t2(p):
    loc = {}
    for op, a, b in p["acts"]:
        if op == "put":
            loc[a] = b
        else:
            loc = {o: (b if c == a else a if c == b else c) for o, c in loc.items()}
    return PLACE[loc[p["query"]]]


def prompt_t2(p):
    lines = [f"{i + 1}. Put the {OBJ[a]} in the {PLACE[b]}." if op == "put"
             else f"{i + 1}. Swap the contents of the {PLACE[a]} and the {PLACE[b]}."
             for i, (op, a, b) in enumerate(p["acts"])]
    places = ", ".join(f"the {x}" for x in PLACE[:-1]) + f" and the {PLACE[-1]}"
    return (f"There are 16 places: {places}. Every place starts empty. The following actions happen in order. "
            '"Put X in P" moves X from wherever it is into P. "Swap the contents of P and Q" moves everything that is '
            "in P into Q and everything that is in Q into P, at the same time.\n\n" + "\n".join(lines) + "\n\n"
            f"Which place holds the {OBJ[p['query']]} now? {ASK}")


# ---------------------------------------------------------------- T3 multi-hop containment (as moves)
def _inside(parent, x, a):
    """True iff x is a (or is inside a)."""
    while x is not None:
        if x == a:
            return True
        x = parent.get(x)
    return False


def _chain(parent, i):
    out = []
    while i in parent:
        i = parent[i]
        out.append(i)
    return out


def gen_t3(rng, n):
    while True:
        parent, moves = {}, []
        for _ in range(n):
            while True:
                a, b = rng.sample(range(len(ITEMS)), 2)
                if parent.get(a) != b and not _inside(parent, b, a):
                    break
            parent[a] = b
            moves.append([a, b])
        cands = [i for i in range(len(ITEMS)) if 2 <= len(_chain(parent, i)) <= 6]
        if cands:
            return {"moves": moves, "query": rng.choice(cands)}


def solve_t3(p):
    parent = {}
    for a, b in p["moves"]:
        parent[a] = b
    return ITEMS[_chain(parent, p["query"])[-1]]


def prompt_t3(p):
    lines = [f"{i + 1}. Put the {ITEMS[a]} into the {ITEMS[b]}." for i, (a, b) in enumerate(p["moves"])]
    q = ITEMS[p["query"]]
    return ("There are 40 containers, named by colour and kind (for example the red box or the blue bag). They start "
            'out separate. The following moves happen in order. "Put A into B" moves A, together with everything '
            "inside it, into B; A is no longer in whatever held it before.\n\n" + "\n".join(lines) + "\n\n"
            f"After all the moves, the {q} is inside a container, which may itself be inside others. What is the "
            f"outermost container that holds the {q}, the one that is not inside anything? {ASK}")


GEN = {"T1": gen_t1, "T2": gen_t2, "T3": gen_t3}
SOLVE = {"T1": solve_t1, "T2": solve_t2, "T3": solve_t3}
PROMPT = {"T1": prompt_t1, "T2": prompt_t2, "T3": prompt_t3}


def all_problems():
    for task in TASKS:
        for n in NS:
            rng = random.Random(cell_seed(task, n))
            for idx in range(N_PROBLEMS):
                p = GEN[task](rng, n)
                yield {"task": task, "n": n, "idx": idx, "problem": p, "gold": SOLVE[task](p),
                       "prompt": PROMPT[task](p)}


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        for row in all_problems():
            f.write(json.dumps(row) + "\n")
