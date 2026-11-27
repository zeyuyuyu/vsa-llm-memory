"""Family "islanders": knights (always truthful) and knaves (always lying) make
statements about each other; work out who is which.

kind = "unique": the generator hides an assignment, adds statements consistent
with it, and keeps adding until exactly one assignment of all 2^N is consistent
(checked by exhaustive enumeration with numpy).  The self-test re-verifies
uniqueness with an independent backtracking counter.
"""
import numpy as np

from _common import (rng_for, item_id, pick_names, is_impossible, read_slots,
                     join_and, num_word)

FAMILY = "islanders"
KIND = "unique"
LEVELS = {1: dict(N=4), 2: dict(N=6), 3: dict(N=9), 4: dict(N=13), 5: dict(N=18)}
ANSWER_KEYS = ("knight:<person> for every person: 1 if they are a knight, "
               "0 if they are a knave")


# ---------------------------------------------------------------- semantics
def truth(st, val):
    """Truth value of statement `st` under assignment `val` (index -> 0/1).
    Works elementwise when the entries of `val` are numpy arrays."""
    t = st["t"]
    if t == "is":
        return val[st["x"]] == st["v"]
    if t == "same":
        return val[st["x"]] == val[st["y"]]
    if t == "diff":
        return val[st["x"]] != val[st["y"]]
    if t == "if":
        return (val[st["x"]] != st["vx"]) | (val[st["y"]] == st["vy"])
    if t == "count":
        c = 0
        for g in st["g"]:
            c = c + (val[g] if st["v"] == 1 else 1 - val[g])
        k = st["k"]
        if st["op"] == "exactly":
            return c == k
        if st["op"] == "atleast":
            return c >= k
        if st["op"] == "atmost":
            return c <= k
    raise ValueError(st)


def consistent(stmts, val):
    """Every speaker's statements are true iff the speaker is a knight."""
    ok = True
    for st in stmts:
        ok = ok & ((val[st["s"]] == 1) == truth(st, val))
    return ok


def refs(st):
    if st["t"] == "count":
        return set(st["g"]) | {st["s"]}
    out = {st["s"], st["x"]}
    if "y" in st:
        out.add(st["y"])
    return out


def count_solutions(data, limit=2):
    """Independent backtracking counter (no numpy): assigns people one by one
    and checks each statement as soon as everyone it mentions is assigned."""
    N = len(data["people"])
    stmts = data["statements"]
    order = list(range(N))
    # check each statement at the position of its last-assigned person
    pos = {p: i for i, p in enumerate(order)}
    by_last = [[] for _ in range(N)]
    for st in stmts:
        by_last[max(pos[p] for p in refs(st))].append(st)
    val = [0] * N
    sols = []

    def rec(i):
        if len(sols) >= limit:
            return
        if i == N:
            sols.append(list(val))
            return
        p = order[i]
        for v in (0, 1):
            val[p] = v
            if all((val[st["s"]] == 1) == bool(truth(st, val)) for st in by_last[i]):
                rec(i + 1)
        val[p] = 0

    rec(0)
    return sols


# ---------------------------------------------------------------- rendering
def _kind(v, plural=False):
    if plural:
        return "knights" if v == 1 else "knaves"
    return "a knight" if v == 1 else "a knave"


def render_statement(st, names):
    s = st["s"]
    nm = lambda i: names[i]
    t = st["t"]
    if t == "is":
        return f"{nm(st['x'])} is {_kind(st['v'])}."
    if t == "same":
        return (f"{nm(st['x'])} and {nm(st['y'])} are the same kind "
                f"(both knights or both knaves).")
    if t == "diff":
        return f"Exactly one of {nm(st['x'])} and {nm(st['y'])} is a knight."
    if t == "if":
        return (f"If {nm(st['x'])} is {_kind(st['vx'])}, then {nm(st['y'])} is "
                f"{_kind(st['vy'])}.")
    if t == "count":
        g = [x for x in st["g"] if x != s]
        has_me = s in st["g"]
        lst = [nm(x) for x in g] + (["me"] if has_me else [])
        grp = join_and(lst)
        k, n, v = st["k"], len(st["g"]), st["v"]
        if st["op"] == "exactly" and k == n:
            lst2 = [nm(x) for x in g] + (["I"] if has_me else [])
            return f"{join_and(lst2)} are all {_kind(v, True)}."
        if st["op"] == "exactly" and k == 0:
            return f"None of {grp} is {_kind(v)}."
        word = {"exactly": "Exactly", "atleast": "At least", "atmost": "At most"}[st["op"]]
        if k == 1:
            return f"{word} one of {grp} is {_kind(v)}."
        return f"{word} {num_word(k)} of {grp} are {_kind(v, True)}."
    raise ValueError(st)


# ---------------------------------------------------------------- generation
def _random_statement(rng, s, N, level):
    others = [i for i in range(N) if i != s]
    r = rng.random()
    if r < 0.07:
        return dict(s=s, t="is", x=rng.choice(others), v=rng.randint(0, 1))
    if r < 0.22:
        x, y = rng.sample(others, 2)
        return dict(s=s, t="same", x=x, y=y)
    if r < 0.37:
        x, y = rng.sample(others, 2)
        return dict(s=s, t="diff", x=x, y=y)
    if r < 0.52:
        x, y = rng.sample(others, 2)
        return dict(s=s, t="if", x=x, vx=rng.randint(0, 1), y=y, vy=rng.randint(0, 1))
    gmax = 3 if N <= 6 else (4 if N <= 12 else 5)
    size = rng.randint(2, min(gmax, N))
    if rng.random() < 0.35:
        g = [s] + rng.sample(others, size - 1)
    else:
        g = rng.sample(others, min(size, len(others)))
    g.sort()
    op = rng.choice(["exactly", "exactly", "atleast", "atmost"])
    n = len(g)
    if op == "exactly":
        k = rng.randint(0, n) if rng.random() < 0.15 else rng.randint(1, n - 1)
    else:
        k = rng.randint(1, n - 1)
    return dict(s=s, t="count", g=g, op=op, k=k, v=rng.randint(0, 1))


def _bits(N):
    c = np.arange(1 << N, dtype=np.int64)
    return [((c >> i) & 1).astype(np.int8) for i in range(N)]


def generate(seed, level):
    rng = rng_for(FAMILY, level, seed)
    N = LEVELS[level]["N"]
    for attempt in range(200):
        hidden = [rng.randint(0, 1) for _ in range(N)]
        if not (2 <= sum(hidden) <= N - 2):
            continue
        bits = _bits(N)
        alive = np.ones(1 << N, dtype=bool)
        stmts = []
        speakers = list(range(N))
        rng.shuffle(speakers)

        def sample_true(s):
            for _ in range(200):
                st = _random_statement(rng, s, N, level)
                if bool(truth(st, hidden)) == (hidden[s] == 1):
                    return st
            return None

        for s in speakers:
            st = sample_true(s)
            if st is None:
                break
            stmts.append(st)
            alive &= ((bits[s] == 1) == truth(st, bits))
        if len(stmts) < N:
            continue
        count_by = {s: 1 for s in range(N)}
        while alive.sum() > 1 and len(stmts) < 2 * N:
            # give someone with a single statement a second one that helps
            pool = [s for s in range(N) if count_by[s] == 1] or list(range(N))
            s = rng.choice(pool)
            best = None
            for _ in range(12):
                st = sample_true(s)
                if st is None:
                    continue
                m = alive & ((bits[s] == 1) == truth(st, bits))
                left = int(m.sum())
                if left < int(alive.sum()) and (best is None or left < best[0]):
                    best = (left, st, m)
            if best is None:
                continue
            stmts.append(best[1])
            alive = best[2]
            count_by[s] += 1
        if alive.sum() != 1:
            continue
        idx = int(np.nonzero(alive)[0][0])
        sol = [(idx >> i) & 1 for i in range(N)]
        assert sol == hidden
        break
    else:
        raise RuntimeError("islanders generator failed")
    names = pick_names(rng, N)
    # list the statements grouped by speaker, speakers in a shuffled order
    order = list(range(N))
    rng.shuffle(order)
    stmts.sort(key=lambda st: order.index(st["s"]))
    data = dict(people=names, statements=stmts)
    text = _render(data, rng)
    gold = {f"knight:{names[i]}": hidden[i] for i in range(N)}
    return {
        "id": item_id(FAMILY, level, seed),
        "family": FAMILY,
        "level": level,
        "seed": seed,
        "text": text,
        "answer_keys": ANSWER_KEYS,
        "kind": KIND,
        "gold": gold,
        "optimum": None,
        "data": data,
    }


def _render(data, rng):
    names = data["people"]
    N = len(names)
    intro = rng.choice([
        "I'm a teacher and I use this kind of puzzle with my older students.",
        "Here is a logic puzzle from our puzzle club's newsletter.",
        "My niece gave me this puzzle and I'd like to check the answer.",
    ])
    fluff = [
        f"{names[0]} is the island's baker.",
        "It was raining lightly when you arrived.",
        f"{names[-1]} keeps goats near the harbour.",
        "The ferry to the mainland only runs on Thursdays.",
        f"{names[1]} and {names[2]} are neighbours.",
    ]
    rng.shuffle(fluff)
    lines = [intro, ""]
    lines.append(
        "On an island, every inhabitant is either a knight or a knave. Knights always "
        "tell the truth: every statement a knight makes is true. Knaves always lie: "
        "every statement a knave makes is false. A statement of the form \"If P, then "
        "Q\" counts as true when P is false or Q is true, and as false only when P is "
        "true and Q is false. When a speaker says \"me\" or \"I\", they mean themselves.")
    lines.append("")
    lines.append(f"You meet {num_word(N) if N <= 10 else N} inhabitants: {join_and(names)}. "
                 + fluff[0] + " They say the following.")
    seen = set()
    for st in data["statements"]:
        sp = names[st["s"]]
        verb = "also says" if sp in seen else "says"
        seen.add(sp)
        lines.append(f"- {sp} {verb}: \"{render_statement(st, names)}\"")
    lines.append("")
    lines.append(fluff[1])
    lines.append("")
    lines.append("Who is a knight and who is a knave? Exactly one assignment of knights "
                 "and knaves fits everything that was said.")
    return "\n".join(lines)


# ---------------------------------------------------------------- checking
def slot_keys(data):
    return [f"knight:{p}" for p in data["people"]]


def check(item, answer):
    data = item["data"]
    if is_impossible(answer):
        return False
    keys = slot_keys(data)
    ans = read_slots(answer, keys)
    if ans is None:
        return False
    val = [ans[k] for k in keys]
    if any(v not in (0, 1) for v in val):
        return False
    # the generator guarantees uniqueness, so consistency is equivalent to
    # matching the gold assignment; test both
    return bool(consistent(data["statements"], val)) and \
        all(ans[k] == item["gold"][k] for k in keys)


# ---------------------------------------------------------------- self-test hooks
def wrong_answers(item, rng, n=20):
    out = [{"impossible": True}]
    keys = list(item["gold"])
    for _ in range(n):
        ans = dict(item["gold"])
        k = rng.choice(keys)
        ans[k] = 1 - ans[k]
        out.append(ans)
    return out


def solve_item(item, stats=None):
    sols = count_solutions(item["data"], limit=2)
    if stats is not None:
        stats["solutions"] = len(sols)
    names = item["data"]["people"]
    return {f"knight:{names[i]}": sols[0][i] for i in range(len(names))}
