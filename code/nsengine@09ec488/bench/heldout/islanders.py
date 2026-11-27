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
LEVELS = {1: dict(N=4), 2: dict(N=6), 3: dict(N=9), 4: dict(N=13), 5: dict(N=18),
          6: dict(N=30), 7: dict(N=45), 8: dict(N=60)}
BRUTE_MAX_N = 20      # up to here uniqueness is decided by full enumeration
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
    and checks each statement as soon as everyone it mentions is assigned.
    Above BRUTE_MAX_N people it hands over to count_solutions_prop."""
    N = len(data["people"])
    if N > BRUTE_MAX_N:
        return count_solutions_prop(data, limit=limit)
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


def count_solutions_prop(data_or_n, stmts=None, limit=2, stats=None):
    """Complete search with propagation (used for the large levels).

    Each statement is a constraint "speaker is a knight  <=>  statement true" over
    the people it mentions.  After every assignment, any constraint with at most
    three people left open is checked against all completions of those people;
    a value no completion supports is ruled out.  Branching picks the open person
    who appears in the most constraints.  Returns up to `limit` solutions; finding
    fewer than `limit` is a proof that no more exist."""
    if stmts is None:
        N = len(data_or_n["people"])
        stmts = data_or_n["statements"]
    else:
        N = data_or_n
    cons = [(sorted(refs(st)), st) for st in stmts]
    var_cons = [[] for _ in range(N)]
    for ci, (vs, _) in enumerate(cons):
        for v in vs:
            var_cons[v].append(ci)
    val = [-1] * N
    sols = []

    def holds(st):
        return (val[st["s"]] == 1) == bool(truth(st, val))

    def propagate(queue, trail):
        seen = set()
        while queue:
            ci = queue.pop()
            vs, st = cons[ci]
            open_ = [v for v in vs if val[v] < 0]
            if not open_:
                if not holds(st):
                    return False
                continue
            if len(open_) > 3:
                continue
            support = [set() for _ in open_]
            m = len(open_)
            for code in range(1 << m):
                for j, v in enumerate(open_):
                    val[v] = (code >> j) & 1
                if holds(st):
                    for j in range(m):
                        support[j].add((code >> j) & 1)
            for v in open_:
                val[v] = -1
            for j, v in enumerate(open_):
                if not support[j]:
                    return False
                if len(support[j]) == 1:
                    val[v] = next(iter(support[j]))
                    trail.append(v)
                    queue.extend(var_cons[v])
        return True

    def rec():
        if stats is not None:
            stats["nodes"] = stats.get("nodes", 0) + 1
        if len(sols) >= limit:
            return
        free = [v for v in range(N) if val[v] < 0]
        if not free:
            if all(holds(st) for _, st in cons):
                sols.append(list(val))
            return
        v = max(free, key=lambda u: sum(1 for ci in var_cons[u]
                                         if any(val[w] < 0 for w in cons[ci][0])))
        for b in (0, 1):
            trail = [v]
            val[v] = b
            if propagate(list(var_cons[v]), trail):
                rec()
            for u in trail:
                val[u] = -1
            if len(sols) >= limit:
                return

    trail0 = []
    if propagate(list(range(len(cons))), trail0):
        rec()
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
    if level >= 6:
        return _random_statement_large(rng, s, N, level)
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


def _random_statement_large(rng, s, N, level):
    """Statement mix for levels 6-8: fewer direct claims, larger groups, so that
    fewer deductions are local and more case analysis is needed."""
    others = [i for i in range(N) if i != s]
    r = rng.random()
    if r < 0.02:
        return dict(s=s, t="is", x=rng.choice(others), v=rng.randint(0, 1))
    if r < 0.10:
        x, y = rng.sample(others, 2)
        return dict(s=s, t="same", x=x, y=y)
    if r < 0.18:
        x, y = rng.sample(others, 2)
        return dict(s=s, t="diff", x=x, y=y)
    if r < 0.32:
        x, y = rng.sample(others, 2)
        return dict(s=s, t="if", x=x, vx=rng.randint(0, 1), y=y, vy=rng.randint(0, 1))
    gmax = {6: 5, 7: 6, 8: 6}[level]
    size = rng.randint(3, gmax)
    if rng.random() < 0.35:
        g = [s] + rng.sample(others, size - 1)
    else:
        g = rng.sample(others, size)
    g.sort()
    op = rng.choice(["exactly", "exactly", "atleast", "atmost"])
    n = len(g)
    k = rng.randint(1, n - 1)
    return dict(s=s, t="count", g=g, op=op, k=k, v=rng.randint(0, 1))


def _bits(N):
    c = np.arange(1 << N, dtype=np.int64)
    return [((c >> i) & 1).astype(np.int8) for i in range(N)]


def _generate_large(rng, N, level):
    """Counterexample-guided: one statement per person, then while a complete
    search still finds a second consistent assignment, add a statement that is
    consistent with the hidden assignment and rules out as many of the found
    alternatives as possible."""
    for attempt in range(50):
        hidden = [rng.randint(0, 1) for _ in range(N)]
        if not (N // 4 <= sum(hidden) <= N - N // 4):
            continue

        def sample_true(s):
            for _ in range(200):
                st = _random_statement(rng, s, N, level)
                if bool(truth(st, hidden)) == (hidden[s] == 1):
                    return st
            return None

        stmts = []
        speakers = list(range(N))
        rng.shuffle(speakers)
        for s in speakers:
            st = sample_true(s)
            if st is None:
                break
            stmts.append(st)
        if len(stmts) < N:
            continue
        count_by = {s: 1 for s in range(N)}
        while len(stmts) <= 2 * N:
            sols = count_solutions_prop(N, stmts, limit=24)
            alts = [x for x in sols if x != hidden]
            if not alts:
                break
            best = None
            pool = [s for s in range(N) if count_by[s] == 1] or list(range(N))
            for _ in range(40):
                s = rng.choice(pool)
                st = sample_true(s)
                if st is None:
                    continue
                killed = sum(1 for a in alts
                             if (a[st["s"]] == 1) != bool(truth(st, a)))
                if killed and (best is None or killed > best[0]):
                    best = (killed, st)
            if best is None:
                continue
            stmts.append(best[1])
            count_by[best[1]["s"]] += 1
        sols = count_solutions_prop(N, stmts, limit=2)
        if len(sols) == 1 and sols[0] == hidden:
            return hidden, stmts
    raise RuntimeError("islanders generator failed")


def generate(seed, level):
    rng = rng_for(FAMILY, level, seed)
    N = LEVELS[level]["N"]
    for attempt in range(200 if N <= BRUTE_MAX_N else 0):
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
        if N <= BRUTE_MAX_N:
            raise RuntimeError("islanders generator failed")
        hidden, stmts = _generate_large(rng, N, level)
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
