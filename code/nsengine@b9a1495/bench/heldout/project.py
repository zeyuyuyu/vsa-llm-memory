"""Family "project": a renovation manager schedules tasks with durations,
dependencies, a limited crew, one shared cement mixer and a few delivery dates,
and wants the earliest possible finish day.

kind = "optimize": any plan that respects every rule and finishes on the optimal
day is right.  The optimum comes from the exact branch-and-bound solver below:
it enumerates left-justified schedules in non-decreasing start order (every
task starts at its release day or at the finish of another task), with
critical-path and resource-energy lower bounds and a state-dominance memo.
"""
from _common import rng_for, item_id, is_impossible, read_slots, join_and

FAMILY = "project"
KIND = "optimize"
CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# (description, typically uses the mixer)
TASK_POOL = [          # roughly in the order a renovation would go
    ("strip out the old kitchen", False), ("level the back garden", False),
    ("replace the roof tiles", False), ("replace the water pipes", False),
    ("rewire the ground floor", False), ("pour the garage floor", True),
    ("build the shed base", True), ("set the fence posts", True),
    ("install the new windows", False), ("insulate the loft", False),
    ("install the boiler", False), ("render the front wall", True),
    ("build the garden wall", True), ("plaster the living room", True),
    ("lay the driveway", True), ("fit the staircase", False),
    ("install the shower", False), ("tile the bathroom", False),
    ("fit the kitchen units", False), ("lay the patio", True),
    ("hang the internal doors", False), ("sand the floorboards", False),
    ("lay the hallway floor", False), ("paint the hallway", False),
]
LEVELS = {
    1: dict(n=5, C=(3, 4)),
    2: dict(n=8, C=(4, 5)),
    3: dict(n=11, C=(4, 6)),
    4: dict(n=15, C=(5, 6)),
    5: dict(n=20, C=(5, 7)),
}
NODE_LIMIT = 1_500_000      # generator resamples an instance whose proof needs more


class _Limit(Exception):
    pass


def answer_keys(codes):
    return (f"start:<task> for every task ({', '.join(codes)}): its start day, "
            "numbered from 1")


def slot_keys(data):
    return [f"start:{t['code']}" for t in data["tasks"]]


# ---------------------------------------------------------------- solver
def _prep(data):
    tasks = data["tasks"]
    idx = {t["code"]: i for i, t in enumerate(tasks)}
    n = len(tasks)
    dur = [t["dur"] for t in tasks]
    crew = [t["crew"] for t in tasks]
    mix = [bool(t["mixer"]) for t in tasks]
    rel = [t["release"] - 1 for t in tasks]          # 0-based earliest start
    preds = [[idx[p] for p in t["preds"]] for t in tasks]
    succs = [[] for _ in range(n)]
    for j in range(n):
        for p in preds[j]:
            succs[p].append(j)
    topo = []
    indeg = [len(preds[j]) for j in range(n)]
    stack = [j for j in range(n) if indeg[j] == 0]
    while stack:
        j = stack.pop()
        topo.append(j)
        for s in succs[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    assert len(topo) == n, "dependency cycle"
    tail = [0] * n
    for j in reversed(topo):
        tail[j] = dur[j] + max((tail[s] for s in succs[j]), default=0)
    return dict(n=n, dur=dur, crew=crew, mix=mix, rel=rel, preds=preds,
                succs=succs, topo=topo, tail=tail, C=data["crew"])


def _sgs(pr, prio):
    """Serial schedule generation: repeatedly take the eligible task with the
    highest priority and start it as early as resources allow."""
    n, dur, crew, mix, rel, preds, C = (pr["n"], pr["dur"], pr["crew"], pr["mix"],
                                         pr["rel"], pr["preds"], pr["C"])
    H = sum(dur) + max(rel) + 2
    use = [0] * H
    mx = [0] * H
    start = [None] * n
    done = 0
    while done < n:
        elig = [j for j in range(n) if start[j] is None
                and all(start[p] is not None for p in preds[j])]
        j = max(elig, key=lambda k: prio[k])
        s = max([rel[j]] + [start[p] + dur[p] for p in preds[j]])
        while True:
            if all(use[x] + crew[j] <= C and not (mix[j] and mx[x])
                   for x in range(s, s + dur[j])):
                break
            s += 1
        for x in range(s, s + dur[j]):
            use[x] += crew[j]
            if mix[j]:
                mx[x] = 1
        start[j] = s
        done += 1
    return start, max(start[j] + dur[j] for j in range(n))


def solve(data, stats=None, seed=0, node_limit=None):
    """Return (starts as 0-based list, makespan).  Exact.  Raises _Limit if a
    node_limit is given and exceeded."""
    import random
    pr = _prep(data)
    n, dur, crew, mix, rel, preds, tail, C = (pr["n"], pr["dur"], pr["crew"],
                                               pr["mix"], pr["rel"], pr["preds"],
                                               pr["tail"], pr["C"])
    topo = pr["topo"]
    rng = random.Random(seed)
    best_start, best = _sgs(pr, [tail[j] + 0.001 * crew[j] for j in range(n)])
    for _ in range(60):
        st, ms = _sgs(pr, [tail[j] * rng.uniform(0.5, 1.5) + rng.random() for j in range(n)])
        if ms < best:
            best, best_start = ms, st
    best_start = list(best_start)
    H = best + 1
    use = [0] * H
    mx = [0] * H
    start = [-1] * n
    fin = [0] * n
    pmask = [sum(1 << p for p in preds[j]) for j in range(n)]
    tot_work = [dur[j] * crew[j] for j in range(n)]
    memo = {}
    nodes = [0]
    state = {"best": best}

    def lower_bound(mask, t, cms):
        head = [0] * n
        lb = cms
        for j in topo:
            if mask >> j & 1:
                continue
            h = max(rel[j], t)
            for p in preds[j]:
                e = fin[p] if mask >> p & 1 else head[p] + dur[p]
                if e > h:
                    h = e
            head[j] = h
            if h + tail[j] > lb:
                lb = h + tail[j]
        work = 0
        mwork = 0
        for j in range(n):
            if mask >> j & 1:
                if fin[j] > t:
                    work += crew[j] * (fin[j] - t)
                    if mix[j]:
                        mwork += fin[j] - t
            else:
                work += tot_work[j]
                if mix[j]:
                    mwork += dur[j]
        e = t + -(-work // C)
        if e > lb:
            lb = e
        if t + mwork > lb:
            lb = t + mwork
        return lb

    def fits(j, s):
        e = s + dur[j]
        if e > H:
            return False
        c = crew[j]
        for x in range(s, e):
            if use[x] + c > C or (mix[j] and mx[x]):
                return False
        return True

    def dfs(mask, t, last, cms):
        nodes[0] += 1
        if node_limit is not None and nodes[0] > node_limit:
            raise _Limit()
        if mask == (1 << n) - 1:
            if cms < state["best"]:
                state["best"] = cms
                best_start[:] = start
            return
        if lower_bound(mask, t, cms) >= state["best"]:
            return
        running = tuple(sorted((fin[k], k) for k in range(n)
                               if mask >> k & 1 and fin[k] > t))
        tfin = any(mask >> k & 1 and fin[k] == t for k in range(n))
        key = (mask, t, last, tfin, running)
        old = memo.get(key)
        if old is not None and old <= cms:
            return
        memo[key] = cms
        finishes = sorted({fin[k] for k in range(n) if mask >> k & 1 and fin[k] >= t})
        cands = []
        for j in range(n):
            if mask >> j & 1 or (pmask[j] & mask) != pmask[j]:
                continue
            est = max([rel[j], t] + [fin[p] for p in preds[j]])
            times = set(f for f in finishes if f >= est)
            if rel[j] >= est:
                times.add(rel[j])
            for s in sorted(times):
                if s == t and j < last:
                    continue
                if s + tail[j] >= state["best"]:
                    break
                if fits(j, s):
                    cands.append((s, -tail[j], j))
        cands.sort()
        for s, _, j in cands:
            if s + tail[j] >= state["best"]:
                continue
            e = s + dur[j]
            for x in range(s, e):
                use[x] += crew[j]
                if mix[j]:
                    mx[x] = 1
            start[j], fin[j] = s, e
            dfs(mask | (1 << j), s, j, max(cms, e))
            for x in range(s, e):
                use[x] -= crew[j]
                if mix[j]:
                    mx[x] = 0
            start[j] = -1
        return

    dfs(0, 0, -1, 0)
    if stats is not None:
        stats["nodes"] = nodes[0]
        stats["memo"] = len(memo)
    return best_start, state["best"]


def critical_path_bound(data):
    pr = _prep(data)
    head = [0] * pr["n"]
    lb = 0
    for j in pr["topo"]:
        head[j] = max([pr["rel"][j]] + [head[p] + pr["dur"][p] for p in pr["preds"][j]])
        lb = max(lb, head[j] + pr["tail"][j])
    return lb


# ---------------------------------------------------------------- checking
def plan_finish(data, starts):
    """Validate a plan {code: start day (1-based)}; return its finish day or None."""
    tasks = data["tasks"]
    by = {t["code"]: t for t in tasks}
    for t in tasks:
        s = starts[t["code"]]
        if s < 1 or s < t["release"]:
            return None
        for p in t["preds"]:
            if s < starts[p] + by[p]["dur"]:
                return None
    last = max(starts[t["code"]] + t["dur"] - 1 for t in tasks)
    for day in range(1, last + 1):
        on = [t for t in tasks if starts[t["code"]] <= day <= starts[t["code"]] + t["dur"] - 1]
        if sum(t["crew"] for t in on) > data["crew"]:
            return None
        if sum(1 for t in on if t["mixer"]) > 1:
            return None
    return last


def check(item, answer):
    data = item["data"]
    if is_impossible(answer):
        return False
    ans = read_slots(answer, slot_keys(data))
    if ans is None:
        return False
    starts = {t["code"]: ans[f"start:{t['code']}"] for t in data["tasks"]}
    fin = plan_finish(data, starts)
    return fin is not None and fin == item["optimum"]


# ---------------------------------------------------------------- generation
def _random_instance(rng, level):
    L = LEVELS[level]
    n = L["n"]
    C = rng.randint(*L["C"])
    pick = sorted(rng.sample(range(len(TASK_POOL)), n))
    pool = [TASK_POOL[i] for i in pick]
    codes = list(CODES[:n])
    tasks = []
    for i, (desc, mixer_like) in enumerate(pool):
        dur = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
        crew = rng.choice([1, 2, 2, 3, 3, 4] if C >= 5 else [1, 2, 2, 3])
        crew = min(crew, C)
        mixer = mixer_like and rng.random() < 0.85
        tasks.append(dict(code=codes[i], desc=desc, dur=dur, crew=crew, mixer=mixer,
                          release=1, preds=[]))
    # dependencies only go from earlier-listed to later-listed tasks
    for j in range(1, n):
        k = rng.choices([0, 1, 1, 2, 2, 3], k=1)[0]
        k = min(k, j)
        lo = max(0, j - 6)
        cand = list(range(lo, j))
        ps = sorted(rng.sample(cand, min(k, len(cand))))
        tasks[j]["preds"] = [codes[p] for p in ps]
    # remove dependencies implied by others (keeps the text short and literal)
    for j in range(n):
        direct = set(tasks[j]["preds"])
        keep = []
        for p in tasks[j]["preds"]:
            implied = False
            for q in direct - {p}:
                if _reaches(tasks, codes, p, q):
                    implied = True
                    break
            if not implied:
                keep.append(p)
        tasks[j]["preds"] = keep
    nrel = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2}[level]
    roots = [j for j in range(n) if not tasks[j]["preds"]]
    early = [j for j in range(n) if tasks[j]["preds"] and j < n // 2]
    cand = roots + early if len(roots) < nrel else roots
    for j in rng.sample(cand, min(nrel, len(cand))):
        tasks[j]["release"] = rng.randint(3, 8)
    return dict(crew=C, tasks=tasks)


def _reaches(tasks, codes, a, b):
    """True if task a is a (transitive) predecessor of task b."""
    by = {t["code"]: t for t in tasks}
    stack = list(by[b]["preds"])
    seen = set()
    while stack:
        x = stack.pop()
        if x == a:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(by[x]["preds"])
    return False


def _render(data, rng):
    C = data["crew"]
    tasks = data["tasks"]
    intro = rng.choice([
        "We're renovating a house and I need to plan the work for my crew.",
        "I run a small building firm and I'm planning the jobs on a house renovation.",
        "I'm the site manager on a home renovation and want to plan the schedule.",
    ])
    fluff = [
        "The client has asked for a photo update at the end of every week.",
        "All materials have already been paid for.",
        "Two of the workers are apprentices, but here every worker counts the same.",
        "The skip outside the house is emptied on Fridays.",
        "Parking on the street is permitted all day.",
    ]
    rng.shuffle(fluff)
    lines = [intro, fluff[0], ""]
    lines.append(
        f"The crew has {C} workers. Each task below takes the stated number of whole "
        "days, runs on consecutive days once started, and cannot be paused. It needs "
        "the stated number of workers on every day it is in progress. On any day, the "
        f"tasks in progress together may need at most {C} workers. We have one cement "
        "mixer: a task marked \"needs the mixer\" uses it on every day it is in "
        "progress, so two such tasks can never be in progress on the same day. A task "
        "can start only after all the tasks it depends on have finished; it may start "
        "on the day right after the last of them finishes. Days are numbered 1, 2, 3, "
        "and so on, and every day is a working day.")
    lines.append("")
    lines.append(f"Tasks ({len(tasks)}):")
    for t in tasks:
        parts = [f"{t['dur']} day{'s' if t['dur'] > 1 else ''}",
                 f"{t['crew']} worker{'s' if t['crew'] > 1 else ''}"]
        if t["mixer"]:
            parts.append("needs the mixer")
        s = f"- {t['code']}: {t['desc']}. " + ", ".join(parts) + "."
        if t["preds"]:
            s += f" Depends on {join_and(t['preds'])}."
        if t["release"] > 1:
            s += (f" Its materials arrive on day {t['release']}, so it cannot start "
                  f"before day {t['release']}.")
        lines.append(s)
    lines.append("")
    lines.append(fluff[1])
    lines.append("")
    lines.append("I want every task finished as early as possible. Find a plan whose "
                 "last working day is as early as it can be, and give the start day of "
                 "every task.")
    return "\n".join(lines)


def generate(seed, level):
    rng = rng_for(FAMILY, level, seed)
    chosen = None
    for attempt in range(60):
        data = _random_instance(rng, level)
        st = {}
        try:
            starts, opt = solve(data, stats=st, seed=seed, node_limit=NODE_LIMIT)
        except _Limit:
            continue
        cp = critical_path_bound(data)
        # want the crew / mixer limits to matter (optimum above the dependency bound)
        if level >= 2 and opt <= cp:
            continue
        chosen = (data, starts, opt)
        break
    if chosen is None:
        raise RuntimeError("project generator failed")
    data, starts, opt = chosen
    codes = [t["code"] for t in data["tasks"]]
    gold = {f"start:{c}": starts[i] + 1 for i, c in enumerate(codes)}
    item = {
        "id": item_id(FAMILY, level, seed),
        "family": FAMILY,
        "level": level,
        "seed": seed,
        "text": _render(data, rng),
        "answer_keys": answer_keys(codes),
        "kind": KIND,
        "gold": gold,
        "optimum": opt,
        "data": data,
    }
    return item


# ---------------------------------------------------------------- self-test hooks
def wrong_answers(item, rng, n=20):
    out = [{"impossible": True}]
    keys = slot_keys(item["data"])
    for _ in range(n):
        ans = dict(item["gold"])
        k = rng.choice(keys)
        ans[k] = max(1, ans[k] + rng.choice([-3, -2, -1, 1, 2, 3]))
        out.append(ans)
    # a valid but slower plan: everything strictly one after another
    return out


def solve_item(item, stats=None):
    starts, opt = solve(item["data"], stats=stats, seed=item["seed"])
    return {f"start:{t['code']}": starts[i] + 1 for i, t in enumerate(item["data"]["tasks"])}
