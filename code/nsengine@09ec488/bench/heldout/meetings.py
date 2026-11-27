"""Family "meetings": fit a week of internal meetings into people's calendars.

kind = "check": the question asks whether all meetings can be scheduled.  About
half the items are possible and half impossible.  The generator walks an instance
towards the feasibility boundary by adding busy blocks one at a time; "possible"
items are the last feasible state on that walk, "impossible" items are the first
infeasible one (every meeting still has at least two individually free slots, so
impossibility is never visible from one meeting alone).

Gold comes from the backtracking solver below (MRV + forward checking).
"""
import copy

from _common import (rng_for, item_id, pick_names, is_impossible, read_slots,
                     join_and, balanced_bit)

FAMILY = "meetings"
KIND = "check"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MEETING_NAMES = ["Budget", "Hiring", "Roadmap", "Onboarding", "Pricing",
                 "Security", "Retro", "Vendor", "Launch", "Training", "Audit",
                 "Branding", "Logistics", "Payroll", "Research", "Support",
                 "Forecast", "Compliance", "Design", "Inventory"]
# start hours allowed for each duration: within 9-12 or 13-17
VALID_STARTS = {1: [9, 10, 11, 13, 14, 15, 16], 2: [9, 10, 13, 14, 15],
                3: [9, 13, 14]}
WALK_NODE_LIMIT = 20000     # a candidate busy block whose solve needs more is skipped


class _Limit(Exception):
    pass


LEVELS = {
    1: dict(D=2, P=3, M=3),
    2: dict(D=3, P=4, M=5),
    3: dict(D=4, P=6, M=8),
    4: dict(D=5, P=8, M=12),
    5: dict(D=5, P=10, M=16),
    6: dict(D=6, P=14, M=24),
    7: dict(D=8, P=20, M=36),
    8: dict(D=10, P=26, M=50),
}

ANSWER_KEYS = ("day:<meeting> for every meeting: its day, numbered from 1 "
               "(Monday = 1); start:<meeting> for every meeting: its starting "
               "hour on the 24-hour clock (for example 14 for 2 pm)")


# ---------------------------------------------------------------- utilities
def _slot_mask(d, s, dur):
    m = 0
    for h in range(s, s + dur):
        m |= 1 << (d * 8 + (h - 9))
    return m


def _busy_mask(intervals):
    m = 0
    for d, f, t in intervals:          # d is 1-based
        for h in range(f, t):
            if 9 <= h < 17:
                m |= 1 << ((d - 1) * 8 + (h - 9))
    return m


def _merge(intervals):
    """Merge overlapping or touching busy intervals per day."""
    out = []
    for d, f, t in sorted(intervals):
        if out and out[-1][0] == d and f <= out[-1][2]:
            out[-1][2] = max(out[-1][2], t)
        else:
            out.append([d, f, t])
    return out


def _prep(data):
    D = data["D"]
    people = data["people"]
    pidx = {p: i for i, p in enumerate(people)}
    meetings = data["meetings"]
    busy = [_busy_mask(data["busy"].get(p, [])) for p in people]
    cap = data["cap"]
    allowed = [set(data["allowed_days"].get(m["name"], range(1, D + 1)))
               for m in meetings]
    midx = {m["name"]: i for i, m in enumerate(meetings)}
    before = [(midx[a], midx[b]) for a, b in data["before"]]
    att = [[pidx[p] for p in m["att"]] for m in meetings]
    dur = [m["dur"] for m in meetings]
    # static day bounds from precedence (one pass is enough for chains <= 3)
    lo = [min(a) for a in allowed]
    hi = [max(a) for a in allowed]
    for _ in range(len(meetings)):
        changed = False
        for a, b in before:
            if lo[b] < lo[a] + 1:
                lo[b] = lo[a] + 1
                changed = True
            if hi[a] > hi[b] - 1:
                hi[a] = hi[b] - 1
                changed = True
        if not changed:
            break
    options = []
    for i in range(len(meetings)):
        opts = []
        if dur[i] <= cap:
            for d in sorted(allowed[i]):
                if d < lo[i] or d > hi[i]:
                    continue
                for s in VALID_STARTS[dur[i]]:
                    mk = _slot_mask(d - 1, s, dur[i])
                    if all(busy[p] & mk == 0 for p in att[i]):
                        opts.append((d, s, mk))
        options.append(opts)
    return dict(D=D, P=len(people), M=len(meetings), att=att, dur=dur,
                cap=cap, before=before, options=options)


def individual_option_counts(data):
    return [len(o) for o in _prep(data)["options"]]


def solve(data, stats=None, root_only=False, node_limit=None):
    """Return a schedule {meeting_index: (day, start)} or None if impossible.
    With root_only=True, only the static checks are run and True/False returned.

    Depth-first search, most-constrained meeting first, with forward checking of
    the other meetings' remaining options and a per-person hours bound."""
    pr = _prep(data)
    M, P, D = pr["M"], pr["P"], pr["D"]
    att, dur, cap, options = pr["att"], pr["dur"], pr["cap"], pr["options"]
    if any(len(o) == 0 for o in options):
        return False if root_only else None
    people = data["people"]
    busy = [_busy_mask(data["busy"].get(p, [])) for p in people]
    attset = [set(a) for a in att]
    prec = {}
    for a, b in pr["before"]:
        prec[(a, b)] = 1      # a must be on an earlier day than b
        prec[(b, a)] = -1
    nbrs = []
    for i in range(M):
        lst = []
        for j in range(M):
            if i == j:
                continue
            sh = tuple(sorted(attset[i] & attset[j]))
            sg = prec.get((i, j), 0)
            if sh or sg:
                lst.append((j, sh, sg))
        nbrs.append(lst)
    daymask = [sum(1 << (d * 8 + k) for k in range(8) if k != 3) for d in range(D)]
    occ = [0] * P
    hours = [[0] * (D + 1) for _ in range(P)]
    rem = [sum(dur[i] for i in range(M) if p in attset[i]) for p in range(P)]
    dom = [list(o) for o in options]
    assigned = [False] * M
    day = [0] * M

    def person_ok(p):
        avail = 0
        for d in range(D):
            free = 7 - ((busy[p] | occ[p]) & daymask[d]).bit_count()
            avail += min(cap - hours[p][d + 1], free)
        return rem[p] <= avail

    for p in range(P):
        if not person_ok(p):
            return False if root_only else None
    if root_only:
        return True

    def assign(i, o):
        d, s, mk = o
        for p in att[i]:
            occ[p] |= mk
            hours[p][d] += dur[i]
            rem[p] -= dur[i]
        assigned[i] = True
        day[i] = d
        changed = []
        ok = True
        for j, sh, sg in nbrs[i]:
            if assigned[j]:
                continue
            nd = []
            for o2 in dom[j]:
                d2 = o2[0]
                if sh:
                    if o2[2] & mk:
                        continue
                    if d2 == d and any(hours[p][d] + dur[j] > cap for p in sh):
                        continue
                if sg == 1 and not d < d2:
                    continue
                if sg == -1 and not d2 < d:
                    continue
                nd.append(o2)
            if len(nd) != len(dom[j]):
                changed.append((j, dom[j]))
                dom[j] = nd
                if not nd:
                    ok = False
                    break
        if ok:
            for p in att[i]:
                if not person_ok(p):
                    ok = False
                    break
        return ok, changed

    def unassign(i, o, changed):
        d, s, mk = o
        for j, old in changed:
            dom[j] = old
        for p in att[i]:
            occ[p] &= ~mk
            hours[p][d] -= dur[i]
            rem[p] += dur[i]
        assigned[i] = False

    sol = {}

    nodes = [0]

    def dfs(left):
        nodes[0] += 1
        if node_limit is not None and nodes[0] > node_limit:
            raise _Limit()
        if left == 0:
            return True
        best_i, best_n = -1, None
        for i in range(M):
            if not assigned[i]:
                n = len(dom[i])
                if best_n is None or n < best_n:
                    best_i, best_n = i, n
                    if n == 1:
                        break
        i = best_i
        for o in list(dom[i]):
            ok, changed = assign(i, o)
            if ok:
                sol[i] = (o[0], o[1])
                if dfs(left - 1):
                    return True
                del sol[i]
            unassign(i, o, changed)
        return False

    found = dfs(M)
    if stats is not None:
        stats["nodes"] = nodes[0]
    if found:
        return dict(sol)
    return None


def _sol_to_answer(data, sol):
    ans = {}
    for i, m in enumerate(data["meetings"]):
        d, s = sol[i]
        ans[f"day:{m['name']}"] = d
        ans[f"start:{m['name']}"] = s
    return ans


def slot_keys(data):
    keys = []
    for m in data["meetings"]:
        keys += [f"day:{m['name']}", f"start:{m['name']}"]
    return keys


# ---------------------------------------------------------------- checking
def valid_schedule(data, ans):
    """Independent rule-by-rule validation of a schedule {day:X, start:X}."""
    D, cap = data["D"], data["cap"]
    sched = {}
    for m in data["meetings"]:
        d = ans[f"day:{m['name']}"]
        s = ans[f"start:{m['name']}"]
        e = s + m["dur"]
        if not 1 <= d <= D:
            return False
        if s < 9 or e > 17:
            return False
        if s < 13 and e > 12:          # overlaps the 12:00-13:00 lunch break
            return False
        allowed = data["allowed_days"].get(m["name"])
        if allowed is not None and d not in allowed:
            return False
        for p in m["att"]:
            for bd, f, t in data["busy"].get(p, []):
                if bd == d and s < t and f < e:
                    return False
        sched[m["name"]] = (d, s, e)
    ms = data["meetings"]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            if set(ms[i]["att"]) & set(ms[j]["att"]):
                di, si, ei = sched[ms[i]["name"]]
                dj, sj, ej = sched[ms[j]["name"]]
                if di == dj and si < ej and sj < ei:
                    return False
    for p in data["people"]:
        for d in range(1, D + 1):
            tot = sum(m["dur"] for m in ms
                      if p in m["att"] and sched[m["name"]][0] == d)
            if tot > cap:
                return False
    for a, b in data["before"]:
        if not sched[a][0] < sched[b][0]:
            return False
    return True


def check(item, answer):
    data = item["data"]
    if item["gold"].get("impossible"):
        return is_impossible(answer)
    if is_impossible(answer):
        return False
    ans = read_slots(answer, slot_keys(data))
    if ans is None:
        return False
    return valid_schedule(data, ans)


# ---------------------------------------------------------------- generation
def _random_structure(rng, level):
    L = LEVELS[level]
    D, P, M = L["D"], L["P"], L["M"]
    people = pick_names(rng, P)
    if M <= len(MEETING_NAMES):
        names = list(MEETING_NAMES)
        rng.shuffle(names)
        names = names[:M]
    else:
        # large levels: topics repeat with a number, e.g. "Budget 2"
        names = [f"{t} {k}" for k in range(1, 4) for t in MEETING_NAMES]
        rng.shuffle(names)
        names = names[:M]
    meetings = []
    for nm in names:
        k = rng.choice([2, 2, 3, 3, 4]) if P >= 4 else rng.choice([2, 2, 3])
        k = min(k, P)
        att = rng.sample(people, k)
        att.sort(key=people.index)
        dur = rng.choices([1, 2, 3], weights=[4, 4, 2])[0]
        meetings.append(dict(name=nm, dur=dur, att=att))
    cap = rng.choice([3, 4]) if level >= 2 else 4
    allowed = {}
    if D >= 2:
        for m in meetings:
            if rng.random() < 0.2:
                k = rng.randint(1, D - 1)
                allowed[m["name"]] = sorted(rng.sample(range(1, D + 1), k))
    before = []
    if D >= 2 and M >= 3:
        npairs = max(1, M // 4)
        order = list(range(M))
        rng.shuffle(order)
        tries = 0
        while len(before) < npairs and tries < 50:
            tries += 1
            a, b = sorted(rng.sample(range(M), 2))
            ia, ib = order[a], order[b]
            pr = [names[ia], names[ib]]
            if pr in before:
                continue
            cand = before + [pr]
            if _longest_chain(cand) <= min(D, 3):
                before = cand
    days = DAY_NAMES[:D] if D <= 5 else [f"day {i}" for i in range(1, D + 1)]
    return dict(D=D, days=days, people=people, meetings=meetings,
                busy={}, allowed_days=allowed, before=before, cap=cap)


def _longest_chain(pairs):
    succ = {}
    nodes = set()
    for a, b in pairs:
        succ.setdefault(a, []).append(b)
        nodes |= {a, b}
    memo = {}

    def lp(x, depth=0):
        if depth > 50:
            return 99            # cycle guard (cannot happen: pairs follow one order)
        if x in memo:
            return memo[x]
        v = 1 + max((lp(y, depth + 1) for y in succ.get(x, [])), default=0)
        memo[x] = v
        return v
    return max((lp(x) for x in nodes), default=0)


def _add_block(data, p, d, f, t):
    new = copy.deepcopy(data)
    lst = new["busy"].get(p, []) + [[d, f, t]]
    new["busy"][p] = _merge(lst)
    return new


def _conflicts(data, sol, p, d, f, t):
    for i, m in enumerate(data["meetings"]):
        if p in m["att"]:
            md, ms = sol[i]
            if md == d and ms < t and f < ms + m["dur"]:
                return True
    return False


def _fmt_h(h):
    return f"{h}:00"


def _render(data, rng):
    D = data["D"]
    days = data["days"]
    span = f"{days[0]} to {days[-1]}" if D > 1 else days[0]
    if D > 5:
        intro = rng.choice([
            f"I'm planning our project meetings for the next {D} working days, which "
            f"I'll call day 1 to day {D}.",
            f"Could you help me schedule a batch of meetings? They have to fit into the "
            f"next {D} working days, numbered day 1 to day {D}.",
        ])
    else:
        intro = None
    intro = intro or rng.choice([
        f"I'm trying to fit next week's internal meetings into everyone's calendars, {span}.",
        f"I coordinate our team's calendar and need to place a set of meetings next week ({span}).",
        f"Could you help me schedule some meetings for next week? Only {span} are available for them.",
    ])
    fluff = [
        "Room bookings are handled separately, so rooms are not a concern here.",
        f"{data['people'][0]} joined the team last month.",
        "All of these are in-person meetings at the office.",
        "Lunch is catered on Wednesdays.",
        "Everyone is in the same time zone.",
        "We record the meeting notes in the shared drive afterwards.",
    ]
    rng.shuffle(fluff)
    lines = [intro, fluff[0], ""]
    lines.append(
        "Working hours are 9:00 to 17:00. Every meeting starts on the hour, lasts "
        "the whole number of hours given below, and must end by 17:00. No meeting may "
        "overlap the lunch break from 12:00 to 13:00 (a meeting may end at 12:00 or "
        "start at 13:00). Every listed attendee must be present for the whole meeting, "
        "and nobody can attend a meeting that overlaps one of their busy times. Two "
        "meetings that share at least one attendee cannot overlap in time; meetings with "
        "no attendee in common may run at the same time. Each meeting takes place on a "
        "single day.")
    lines.append("")
    lines.append(f"Meetings ({len(data['meetings'])} in total):")
    for m in data["meetings"]:
        h = m["dur"]
        lines.append(f"- {m['name']}: {h} hour{'s' if h > 1 else ''}, attended by "
                     f"{join_and(m['att'])}.")
    lines.append("")
    lines.append("Times when people are unavailable:")
    for p in data["people"]:
        blocks = data["busy"].get(p, [])
        if not blocks:
            lines.append(f"- {p}: no busy times.")
        else:
            parts = [f"{days[d - 1]} {_fmt_h(f)}-{_fmt_h(t)}" for d, f, t in blocks]
            lines.append(f"- {p}: busy " + "; ".join(parts) + ".")
    lines.append("")
    lines.append("Other rules:")
    rules = []
    for a, b in data["before"]:
        rules.append(f"- {a} must be held on an earlier day than {b}.")
    for m in data["meetings"]:
        al = data["allowed_days"].get(m["name"])
        if al is not None:
            ds = [days[d - 1] for d in al]
            if len(ds) == 1:
                rules.append(f"- {m['name']} can only be held on {ds[0]}.")
            else:
                rules.append(f"- {m['name']} can only be held on " +
                             " or ".join([", ".join(ds[:-1]), ds[-1]]) + ".")
    rules.append(f"- No person may attend more than {data['cap']} hours of these "
                 f"meetings on the same day.")
    lines += rules
    lines.append("")
    lines.append(fluff[1])
    lines.append("")
    lines.append(
        f"Can all {len(data['meetings'])} meetings be scheduled under these rules? If yes, "
        "give a schedule: the day and the starting hour of every meeting. If it cannot "
        "be done, say that it is impossible.")
    return "\n".join(lines)


WALKS = {6: 3, 7: 5, 8: 8}     # large levels keep the hardest of several walks


def _one_walk(rng, level, want_possible):
    """Add busy blocks until the instance just turns infeasible (see module doc).
    Every state kept on the walk was decided by the complete search."""
    D = LEVELS[level]["D"]
    for attempt in range(500):
        data = _random_structure(rng, level)
        try:
            sol = solve(data, node_limit=WALK_NODE_LIMIT)
        except _Limit:
            continue
        if sol is None:
            continue
        infeasible = None
        fails = 0
        added = 0
        while fails < 80:
            p = rng.choice(data["people"])
            d = rng.randint(1, D)
            f = rng.randint(9, 16)
            t = min(17, f + rng.choice([1, 1, 2, 2, 3]))
            new = _add_block(data, p, d, f, t)
            if new["busy"][p] == data["busy"].get(p, []):
                fails += 1
                continue
            if min(individual_option_counts(new)) < 2:
                fails += 1
                continue
            if not solve(new, root_only=True):
                # keep impossibility out of reach of a simple hours count
                fails += 1
                continue
            if not _conflicts(data, sol, p, d, f, t):
                data = new
                added += 1
                continue
            try:
                sol2 = solve(new, node_limit=WALK_NODE_LIMIT)
            except _Limit:
                fails += 1          # undecided within budget: do not add this block
                continue
            if sol2 is not None:
                data, sol = new, sol2
                added += 1
            else:
                infeasible = new
                break
        if want_possible:
            final, gold_sol = data, sol
        else:
            if infeasible is None:
                continue
            final, gold_sol = infeasible, None
        return final, gold_sol
    raise RuntimeError("meetings generator failed")


def generate(seed, level):
    rng = rng_for(FAMILY, level, seed)
    want_possible = balanced_bit(FAMILY, level, seed)
    if level not in WALKS:
        final, gold_sol = _one_walk(rng, level, want_possible)
    else:
        best = None
        for w in range(WALKS[level]):
            fin, gs = _one_walk(rng, level, want_possible)
            st = {}
            sol = solve(fin, stats=st)        # complete search; also the proof size
            assert (sol is None) == (gs is None)
            if best is None or st["nodes"] > best[0]:
                best = (st["nodes"], fin, sol)
        _, final, gold_sol = best
    text = _render(final, rng)
    if gold_sol is None:
        gold = {"impossible": True}
    else:
        gold = _sol_to_answer(final, gold_sol)
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
        "data": final,
    }


# ---------------------------------------------------------------- self-test hooks
def wrong_answers(item, rng, n=20):
    data = item["data"]
    out = []
    if item["gold"].get("impossible"):
        for _ in range(n):
            ans = {}
            for m in data["meetings"]:
                ans[f"day:{m['name']}"] = rng.randint(1, data["D"])
                ans[f"start:{m['name']}"] = rng.choice(VALID_STARTS[m["dur"]])
            out.append(ans)
        return out
    out.append({"impossible": True})
    for _ in range(n):
        ans = dict(item["gold"])
        m = rng.choice(data["meetings"])
        ans[f"day:{m['name']}"] = rng.randint(1, data["D"])
        ans[f"start:{m['name']}"] = rng.choice(VALID_STARTS[m["dur"]])
        out.append(ans)
    return out


def solve_item(item, stats=None):
    """Re-solve from data; returns a gold-format answer (used for timing)."""
    sol = solve(item["data"], stats=stats)
    return {"impossible": True} if sol is None else _sol_to_answer(item["data"], sol)
