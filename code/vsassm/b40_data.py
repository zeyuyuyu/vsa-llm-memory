"""B40 data: Dhanraj & Eliasmith's arithmetic tasks (task N) and a relational 'where is it' task (task R).

Pure Python + numpy RNG with fixed integer seeds; nothing depends on machine files (no /usr/share/dict).
Every split is regenerated bit-for-bit from (task, split) alone.

Task N (replicates DE25's Symbolic-Math set, complexity 2 = operands 1..999, x >= y, answers mod 1000 for
multiplication and LCM, the question templates and the 2-shot chat format of the published code
(llama/utilities.generate_dialog), and its CoT prompt).
Task R: object in container / container held by person / person in room; 'Where is X?' -> room; pairings drawn
fresh for every instance; depth = number of hops from X to its room (1 = a person).
"""
import math
import numpy as np

# ----------------------------------------------------------------------------------------------- task N
OPS_TRAIN = ["multiplication", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"]
OPS_HELD = ["addition", "division"]
# DE25's engine vocabulary order (llama/vsa_engine.SymbolicEngine possible_problems)
OPS_ALL = ["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod",
           "bitwise_and", "bitwise_xor", "bitwise_or"]
MOD = 1000  # 10 ** (complexity + 1), complexity = 2

SYS_PLAIN = ("You are a math solving helper. Don't use any commas in your output, "
             "and always answer problems according to the format of previous answers.")
SYS_COT = ("You are a math-solving assistant. Always explain your reasoning step by step. "
           "Regardless of the steps taken, ensure the final answer is clearly marked with 'Final Answer: x'.")


def op_answer(op, x, y):
    if op == "addition":
        return x + y
    if op == "multiplication":
        return (x * y) % MOD
    if op == "division":
        return x // y
    if op == "modulo":
        return x % y
    if op == "gcd":
        return math.gcd(x, y)
    if op == "lcm":
        return (x * y // math.gcd(x, y)) % MOD
    if op == "square_mod":
        return (x * x) % y
    if op == "bitwise_and":
        return x & y
    if op == "bitwise_xor":
        return x ^ y
    if op == "bitwise_or":
        return x | y
    raise ValueError(op)


def op_question(op, x, y, cot=False):
    q = {"addition": f"What is {x} plus {y}?",
         "multiplication": f"What is {x} times {y} mod {MOD}?",
         "division": f"What is {x} // {y}?",
         "modulo": f"What is {x} mod {y}?",
         "gcd": f"What is the GCD of {x} and {y}?",
         "lcm": f"What is the LCM of {x} and {y} mod {MOD}?",
         "square_mod": f"What is {x}^2 mod {y}?",
         "bitwise_and": f"What is {x} AND {y}?",
         "bitwise_xor": f"What is {x} XOR {y}?",
         "bitwise_or": f"What is {x} OR {y}?"}[op]
    if cot:
        q = "Solve the following problem step by step: " + q
        if op.startswith("bitwise"):
            q += " Remember to convert your final answer back to decimal"
    return q


def _pair(rng):
    x, y = int(rng.integers(1, MOD)), int(rng.integers(1, MOD))
    return max(x, y), min(x, y)


N_SPLITS = {  # split -> (seed, ops, per-op count)
    "train": (1001, OPS_TRAIN, 2000),
    "dev": (2002, OPS_TRAIN, 100),
    "test": (3003, OPS_TRAIN + OPS_HELD, 250),
}


def gen_n(split):
    seed, ops, per = N_SPLITS[split]
    rng = np.random.default_rng(seed)
    banned = set()
    if split != "train":  # no (op, x, y) of an evaluation split occurs in train (or, for test, in dev)
        for s in (["train"] if split == "dev" else ["train", "dev"]):
            banned |= {(r["op"], r["x"], r["y"]) for r in gen_n(s)}
    rows = []
    for op in ops:
        seen, k = set(), 0
        while k < per:
            x, y = _pair(rng)
            e1, e2 = _pair(rng), _pair(rng)
            if (op, x, y) in banned or (op, x, y) in seen:
                continue
            seen.add((op, x, y))
            k += 1
            rows.append({"task": "N", "split": split, "id": f"N-{split}-{len(rows)}", "op": op, "x": x, "y": y,
                         "ans": op_answer(op, x, y), "ex": [list(e1), list(e2)]})
    return rows


def n_messages(r, cot=False):
    if cot:
        return [{"role": "system", "content": SYS_COT},
                {"role": "user", "content": op_question(r["op"], r["x"], r["y"], cot=True)}]
    m = [{"role": "system", "content": SYS_PLAIN}]
    for ex, ey in r["ex"]:
        m += [{"role": "user", "content": op_question(r["op"], ex, ey)},
              {"role": "assistant", "content": str(op_answer(r["op"], ex, ey))}]
    m.append({"role": "user", "content": op_question(r["op"], r["x"], r["y"])})
    return m


# ----------------------------------------------------------------------------------------------- task R
# Pools are fixed here (checked to be single Llama-3 tokens by b40_core.check_pools).
PEOPLE = ["Mary", "John", "Sandra", "Daniel", "Emma", "Peter", "Laura", "David", "Anna", "James",
          "Sarah", "Michael", "Kate", "Paul", "Lucy", "Mark", "Julia", "Henry", "Alice", "Frank"]
CONTAINERS = ["box", "bag", "basket", "drawer", "jar", "bucket", "case", "pouch", "crate", "tin",
              "wallet", "envelope", "bottle", "cup", "bowl", "purse", "sack", "chest", "folder", "pot"]
# 'rooms' are places; each is ONE Llama-3 token both with a leading space (story) and without (answer at the start of
# the assistant turn); the first pool (kitchen, garden, ...) failed the no-space check before any run (smoke record).
ROOMS = ["office", "library", "studio", "lobby", "hall", "park", "school", "church", "hotel", "farm"]
ENTS = PEOPLE + CONTAINERS + ROOMS
EID = {e: i for i, e in enumerate(ENTS)}
KIND = {**{e: "person" for e in PEOPLE}, **{e: "container" for e in CONTAINERS}, **{e: "room" for e in ROOMS}}

SYS_R = "You answer questions about a short story. Answer with one word: the name of the place."
SYS_R_COT = ("You answer questions about a short story. Always explain your reasoning step by step. "
             "Regardless of the steps taken, ensure the final answer is clearly marked with 'Final Answer: x', "
             "where x is the name of the place.")
SIZES = {"small": (4, 6), "large": (8, 12)}  # (people, containers); facts = people + containers
N_ROOMS_INST = 4


def _fact_text(kind, a, b):
    # every fact states 'a is located at b'; the mention order in the sentence differs by template
    if kind == "PR":  # person a in room b
        return f"{a} is in the {b}.", [("a", a), ("b", b)]
    if kind == "CP":  # container a held by person b
        return f"{b} has the {a}.", [("b", b), ("a", a)]
    if kind == "CC":  # container a inside container b
        return f"The {a} is in the {b}.", [("a", a), ("b", b)]
    raise ValueError(kind)


def gen_instance(rng, size, depth):
    n_p, n_c = SIZES[size]
    assert 1 <= depth <= n_c + 1
    rooms = [ROOMS[i] for i in rng.choice(len(ROOMS), N_ROOMS_INST, replace=False)]
    people = [PEOPLE[i] for i in rng.choice(len(PEOPLE), n_p, replace=False)]
    conts = [CONTAINERS[i] for i in rng.choice(len(CONTAINERS), n_c, replace=False)]
    loc = {p: rooms[int(rng.integers(len(rooms)))] for p in people}
    facts = [("PR", p, loc[p]) for p in people]
    # target chain: q -> c_2 -> ... -> c_{depth-1} -> person -> room
    holder = people[0]
    chain = conts[:depth - 1]
    cloc = {}
    for i, c in enumerate(chain):
        cloc[c] = chain[i + 1] if i + 1 < len(chain) else holder
    placed = list(chain)
    for c in conts[depth - 1:]:
        if rng.random() < 0.5 or not placed:
            cloc[c] = people[int(rng.integers(n_p))]
        else:
            cloc[c] = placed[int(rng.integers(len(placed)))]
        placed.append(c)
    for c in conts:
        facts.append(("CP" if cloc[c] in loc else "CC", c, cloc[c]))
    order = rng.permutation(len(facts))
    facts = [facts[i] for i in order]
    q = holder if depth == 1 else chain[0]
    # gold by following the located-at map (independent of how the chain was built)
    where = {a: b for _, a, b in facts}
    x, hops = q, 0
    while KIND[x] != "room":
        x = where[x]
        hops += 1
    assert hops == depth
    return {"facts": [list(f) for f in facts], "q": q, "ans": x, "depth": depth, "size": size}


def story_text(inst):
    """Story string and mention list [(char_start, entity, fact_index or -1 for question, role)]."""
    parts, ments, pos = [], [], 0
    for fi, (kind, a, b) in enumerate(inst["facts"]):
        s, order = _fact_text(kind, a, b)
        off = 0
        for role, ent in order:
            j = s.index(ent, off)
            ments.append((pos + j, ent, fi, role))
            off = j + len(ent)
        parts.append(s)
        pos += len(s) + 1
    story = " ".join(parts)
    qtext = f"Where is {inst['q']}?" if KIND[inst["q"]] == "person" else f"Where is the {inst['q']}?"
    return story, qtext, ments


def r_user(inst, cot=False):
    story, qtext, _ = story_text(inst)
    return f"Story: {story}\nQuestion: {qtext}"


R_SPLITS = {  # split -> (seed, [(size, depth, count)])
    "train": (1101, [("small", d, 4000) for d in (1, 2, 3)]),
    "dev": (2202, [("small", d, 200) for d in (1, 2, 3)]),
    "test": (3303, [(s, d, 200) for s in ("small", "large") for d in (1, 2, 3, 4, 5)]),
    "train_e2e": (1102, [("small", 2, 12000)]),
    "train_e2e3": (1103, [("small", 3, 12000)]),  # amendment 2: fixed-depth-3 label-free sub-experiment
}


def gen_r(split):
    seed, cells = R_SPLITS[split]
    rng = np.random.default_rng(seed)
    rows = []
    for size, depth, cnt in cells:
        for _ in range(cnt):
            inst = gen_instance(rng, size, depth)
            exs = [gen_instance(rng, "small", int(rng.integers(1, 4))) for _ in range(2)]
            rows.append({"task": "R", "split": split, "id": f"R-{split}-{len(rows)}", **inst, "ex": exs})
    return rows


def r_messages(r, cot=False):
    if cot:
        return [{"role": "system", "content": SYS_R_COT},
                {"role": "user", "content": r_user(r) + "\nSolve the question step by step."}]
    m = [{"role": "system", "content": SYS_R}]
    for ex in r["ex"]:
        m += [{"role": "user", "content": r_user(ex)}, {"role": "assistant", "content": ex["ans"]}]
    m.append({"role": "user", "content": r_user(r)})
    return m


def gen(task, split):
    return gen_n(split) if task == "N" else gen_r(split)


def messages(r, cot=False):
    return n_messages(r, cot) if r["task"] == "N" else r_messages(r, cot)


def gold(r):
    return str(r["ans"])


if __name__ == "__main__":
    import hashlib, json, sys
    for t, splits in (("N", ["train", "dev", "test"]), ("R", ["train", "dev", "test", "train_e2e"])):
        for s in splits:
            rows = gen(t, s)
            h = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
            print(t, s, len(rows), h)
    r = gen_r("test")[1500]
    print(json.dumps(r_messages(r), indent=1)[:1500])
