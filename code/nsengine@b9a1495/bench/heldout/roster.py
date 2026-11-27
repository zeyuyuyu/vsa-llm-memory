"""Family "roster": a ward manager builds a nurse roster with exact shift coverage,
rest rules, a consecutive-days limit, shift-count bounds, senior cover on nights
and individual requests.

kind = "find": any roster meeting every rule is right.  The generator first plants
a random roster that satisfies the coverage and rest rules, then derives the
per-nurse bounds and requests from it, so every item is feasible; the gold roster
is whatever the complete backtracking solver below finds (it need not be the
planted one).
"""
from _common import rng_for, item_id, pick_names, read_slots, is_impossible, join_and

FAMILY = "roster"
KIND = "find"
SHIFT_NAMES = {1: "early", 2: "late", 3: "night"}
LEVELS = {
    1: dict(N=4, D=4, K=3, u=0.75),
    2: dict(N=5, D=5, K=3, u=0.62),
    3: dict(N=7, D=7, K=3, u=0.62),
    4: dict(N=10, D=10, K=4, u=0.62),
    5: dict(N=13, D=14, K=4, u=0.62),
}


def answer_keys(D):
    return (f"shift:<nurse>:<day> for every nurse and every day from 1 to {D}: "
            "0 if the nurse is off, 1 for the early shift, 2 for the late shift, "
            "3 for the night shift")


def slot_keys(data):
    return [f"shift:{n}:{d}" for n in data["nurses"] for d in range(1, data["D"] + 1)]


# ---------------------------------------------------------------- solver
class _Limit(Exception):
    pass


def solve(data, rng=None, node_limit=None, stats=None, rules=None):
    """Complete DFS (day by day, nurse by nurse).  Returns {nurse: [codes per day]}
    or None if no roster exists.  `rules` can override fields of the data (used
    when planting).

    Pruning: (1) per-shift supply within the day: the nurses still to be decided
    today who are allowed each shift must cover what that shift still needs, and a
    senior must remain if the night still lacks one; (2) at each day boundary, an
    exact per-nurse bound on how many more shifts they can work (off requests and
    the consecutive-days limit, computed greedily, which is optimal for that
    relaxation) against their minimum, and the aggregate remaining coverage
    against the nurses' remaining capacity."""
    r = dict(data)
    if rules:
        r.update(rules)
    nurses = r["nurses"]
    N, D, K = len(nurses), r["D"], r["max_consec"]
    A, B, MN = r["min_shifts"], r["max_shifts"], r["max_nights"]
    cover = r["cover"]                         # cover[d] = [early, late, night]
    senior = [n in r["senior"] for n in nurses]
    off = [[False] * D for _ in range(N)]
    for i, n in enumerate(nurses):
        for d in r["off_requests"].get(n, []):
            off[i][d - 1] = True
    nonight = [n in r["no_nights"] for n in nurses]
    order = sorted(range(N), key=lambda i: (not senior[i], i))   # seniors first
    rem_cover = [sum(sum(cover[e]) for e in range(d + 1, D)) for d in range(D)]
    rem_night = [sum(cover[e][2] for e in range(d + 1, D)) for d in range(D)]
    avg = sum(map(sum, cover)) / N

    x = [[0] * D for _ in range(N)]
    tot = [0] * N
    nights = [0] * N
    run = [0] * N          # consecutive worked days ending the day before the current one
    cnt = [0, 0, 0, 0]
    sen_night = [False]
    nodes = [0]
    day_info = [None] * D  # per day: allowed-shift masks and suffix supplies

    def maxwork(i, frm, rr):
        c = 0
        for e in range(frm, D):
            if not off[i][e] and rr < K:
                c += 1
                rr += 1
            else:
                rr = 0
        return c

    def start_day(d):
        el = [0] * N
        for i in range(N):
            if off[i][d] or run[i] >= K or tot[i] >= B:
                continue
            prev = x[i][d - 1] if d > 0 else 0
            m = 0
            if prev not in (2, 3):
                m |= 2
            if prev != 3:
                m |= 4
            if not nonight[i] and nights[i] < MN:
                m |= 8
            el[i] = m
        suf = [[0, 0, 0, 0, 0] for _ in range(N + 1)]   # [any, e, l, n, senior-n]
        for k in range(N - 1, -1, -1):
            i = order[k]
            m = el[i]
            row = list(suf[k + 1])
            if m:
                row[0] += 1
            if m & 2:
                row[1] += 1
            if m & 4:
                row[2] += 1
            if m & 8:
                row[3] += 1
                if senior[i]:
                    row[4] += 1
            suf[k] = row
        day_info[d] = (el, suf)

    def day_end_ok(d):
        need_min = 0
        can_max = 0
        can_night = 0
        for i in range(N):
            mw = maxwork(i, d + 1, run[i])
            if tot[i] + mw < A:
                return False
            need_min += max(0, A - tot[i])
            can_max += min(B - tot[i], mw)
            if not nonight[i]:
                can_night += min(MN - nights[i], mw)
        return need_min <= rem_cover[d] <= can_max and rem_night[d] <= can_night

    def rec(pos):
        nodes[0] += 1
        if node_limit is not None and nodes[0] > node_limit:
            raise _Limit()
        d, k = divmod(pos, N)
        if d == D:
            return True
        if k == 0:
            start_day(d)
        el, suf = day_info[d]
        cov = cover[d]
        ne, nl, nn = cov[0] - cnt[1], cov[1] - cnt[2], cov[2] - cnt[3]
        sk = suf[k]
        if ne > sk[1] or nl > sk[2] or nn > sk[3] or ne + nl + nn > sk[0]:
            return False
        if nn > 0 and not sen_night[0] and sk[4] == 0:
            return False
        i = order[k]
        m = el[i]
        need = (0, ne, nl, nn)
        vals = []
        for s in (1, 2, 3):
            if need[s] > 0 and m & (1 << s):
                if s == 3 and not senior[i] and not sen_night[0] and nn == 1:
                    continue
                vals.append(s)
        if ne + nl + nn <= N - k - 1 and tot[i] + maxwork(i, d + 1, 0) >= A:
            vals.append(0)
        if rng is not None:
            rng.shuffle(vals)
        else:
            target = avg * (d + 1) / D
            work = [v for v in vals if v]
            work.sort(key=lambda s: -need[s])
            vals = ([0] + work if 0 in vals else work) if tot[i] >= target else \
                   (work + ([0] if 0 in vals else []))
        for v in vals:
            x[i][d] = v
            old_sen = sen_night[0]
            if v:
                tot[i] += 1
                cnt[v] += 1
                if v == 3:
                    nights[i] += 1
                    if senior[i]:
                        sen_night[0] = True
            if k == N - 1:
                saved = (list(run), list(cnt), sen_night[0])
                for j in range(N):
                    run[j] = run[j] + 1 if x[j][d] else 0
                cnt[1] = cnt[2] = cnt[3] = 0
                sen_night[0] = False
                if day_end_ok(d) and rec(pos + 1):
                    return True
                run[:], cnt[:], sen_night[0] = saved[0], saved[1], saved[2]
            elif rec(pos + 1):
                return True
            if v:
                tot[i] -= 1
                cnt[v] -= 1
                if v == 3:
                    nights[i] -= 1
            sen_night[0] = old_sen
            x[i][d] = 0
        return False

    try:
        found = rec(0)
    finally:
        if stats is not None:
            stats["nodes"] = stats.get("nodes", 0) + nodes[0]
    if not found:
        return None
    return {nurses[i]: list(x[i]) for i in range(N)}


SOLVE_BUDGET = 1_000_000     # total nodes; the generator resamples above this


def solve_robust(data, seed=0, stats=None, budget=SOLVE_BUDGET):
    """Exact search with restarts: a deterministic run, then randomised runs with
    growing node limits, all within `budget` nodes in total; raises _Limit when
    the budget is spent (each run is complete, so a None result is a proof)."""
    import random
    st = stats if stats is not None else {}
    st.setdefault("nodes", 0)
    first = min(20000, budget)
    try:
        return solve(data, node_limit=first, stats=st)
    except _Limit:
        pass
    rng = random.Random(seed)
    lim = 20000
    while st["nodes"] < budget:
        lim = min(lim * 2, budget - st["nodes"])
        try:
            return solve(data, rng=rng, node_limit=lim, stats=st)
        except _Limit:
            pass
    raise _Limit()


# ---------------------------------------------------------------- checking
def valid_roster(data, x):
    """x[nurse] = list of codes for days 1..D.  Rule-by-rule validation."""
    nurses, D = data["nurses"], data["D"]
    for n in nurses:
        if len(x[n]) != D or any(v not in (0, 1, 2, 3) for v in x[n]):
            return False
    for d in range(D):
        for s in (1, 2, 3):
            if sum(1 for n in nurses if x[n][d] == s) != data["cover"][d][s - 1]:
                return False
        if data["cover"][d][2] >= 1:
            if not any(x[n][d] == 3 for n in data["senior"]):
                return False
    for n in nurses:
        row = x[n]
        for d in range(D - 1):
            if row[d] == 3 and row[d + 1] in (1, 2):
                return False
            if row[d] == 2 and row[d + 1] == 1:
                return False
        streak = 0
        for v in row:
            streak = streak + 1 if v else 0
            if streak > data["max_consec"]:
                return False
        t = sum(1 for v in row if v)
        if not data["min_shifts"] <= t <= data["max_shifts"]:
            return False
        if sum(1 for v in row if v == 3) > data["max_nights"]:
            return False
        for d in data["off_requests"].get(n, []):
            if row[d - 1] != 0:
                return False
        if n in data["no_nights"] and 3 in row:
            return False
    return True


def check(item, answer):
    data = item["data"]
    if is_impossible(answer):
        return False
    ans = read_slots(answer, slot_keys(data))
    if ans is None:
        return False
    x = {n: [ans[f"shift:{n}:{d}"] for d in range(1, data["D"] + 1)]
         for n in data["nurses"]}
    return valid_roster(data, x)


# ---------------------------------------------------------------- generation
def _random_cover(rng, N, D, u):
    cover = []
    for _ in range(D):
        T = max(3, min(N - 1, round(N * u + rng.choice([-1, 0, 0, 1]))))
        nn = 2 if (N >= 9 and rng.random() < 0.4) else 1
        rest = T - nn
        e = rest // 2 + (rest % 2 if rng.random() < 0.5 else 0)
        l = rest - e
        if e < 1 or l < 1:
            e, l = max(1, e), max(1, l)
        cover.append([e, l, nn])
    return cover


def _render(data, rng):
    D, nurses = data["D"], data["nurses"]
    sen = [n for n in nurses if n in data["senior"]]
    intro = rng.choice([
        f"I manage a hospital ward and need a nurse roster for the next {D} days, "
        f"numbered day 1 to day {D}.",
        f"Could you help me draw up the nursing roster for our ward? It covers {D} days, "
        f"day 1 to day {D}.",
        f"I'm putting together the ward roster for the coming period of {D} days "
        f"(day 1 to day {D}).",
    ])
    fluff = [
        "The ward has 24 beds and is usually full at this time of year.",
        f"{sen[0]} is also the ward's fire warden.",
        "Agency staff are not available, so only the nurses listed below can be used.",
        "The handover at the start of each shift is part of that shift.",
        "We print the roster and pin it up in the staff room.",
    ]
    rng.shuffle(fluff)
    lines = [intro, fluff[0], ""]
    lines.append("There are three shifts every day: early (7:00-15:00), late "
                 "(15:00-23:00) and night (23:00 until 7:00 the next morning). The night "
                 "shift of a day is the one that starts at 23:00 on that day.")
    lines.append("")
    lines.append(f"Nurses ({len(nurses)}): {join_and(nurses)}. Of these, "
                 f"{join_and(sen)} {'is a senior nurse' if len(sen) == 1 else 'are senior nurses'}.")
    lines.append("")
    lines.append("Number of nurses needed on each shift (exactly this many, no more and "
                 "no fewer):")
    for d in range(D):
        e, l, n = data["cover"][d]
        lines.append(f"- Day {d + 1}: {e} early, {l} late, {n} night.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- A nurse works at most one shift per day.")
    lines.append("- A nurse who works the night shift on one day cannot work the early or "
                 "the late shift on the next day. (Working the night shift again on the "
                 "next day is allowed.)")
    lines.append("- A nurse who works the late shift on one day cannot work the early "
                 "shift on the next day.")
    lines.append(f"- No nurse may work on more than {data['max_consec']} days in a row "
                 "(any shift counts as working that day).")
    if data["min_shifts"] == data["max_shifts"]:
        lines.append(f"- Every nurse works exactly {data['min_shifts']} shifts in total "
                     f"over the {D} days.")
    else:
        lines.append(f"- Every nurse works at least {data['min_shifts']} and at most "
                     f"{data['max_shifts']} shifts in total over the {D} days.")
    lines.append(f"- No nurse works more than {data['max_nights']} night "
                 f"shift{'s' if data['max_nights'] != 1 else ''} in total.")
    lines.append("- Every night shift must include at least one senior nurse.")
    for n in nurses:
        req = data["off_requests"].get(n)
        if req:
            ds = [str(d) for d in req]
            lines.append(f"- {n} cannot work on day{'s' if len(ds) > 1 else ''} "
                         f"{join_and(ds)}.")
    for n in data["no_nights"]:
        lines.append(f"- {n} does not work night shifts.")
    lines.append(f"- Nothing before day 1 or after day {D} matters: for example, anyone "
                 "may work the early shift on day 1.")
    lines.append("")
    lines.append(fluff[1])
    lines.append("")
    lines.append("Please find a roster that follows all of these rules: for every nurse "
                 "and every day, say whether they are off or which shift they work.")
    return "\n".join(lines)


def generate(seed, level):
    rng = rng_for(FAMILY, level, seed)
    L = LEVELS[level]
    N, D, K, u = L["N"], L["D"], L["K"], L["u"]
    for attempt in range(200):
        nurses = pick_names(rng, N)
        nsen = max(2, round(N * 0.4))
        senior = sorted(rng.sample(nurses, nsen), key=nurses.index)
        cover = _random_cover(rng, N, D, u)
        avg = sum(map(sum, cover)) / N
        import math
        if level >= 3:      # balanced workloads: every nurse within one shift of the mean
            lo, hi = math.floor(avg), math.ceil(avg)
            if lo == hi:
                lo -= 1
        else:
            lo, hi = max(0, int(avg) - 1), min(D, int(avg) + 2)
        base = dict(nurses=nurses, D=D, cover=cover, senior=senior, max_consec=K,
                    min_shifts=lo, max_shifts=hi,
                    max_nights=D, off_requests={}, no_nights=[])
        try:
            plant = solve(base, rng=rng, node_limit=20000)
        except _Limit:
            continue
        if plant is None:
            continue
        totals = {n: sum(1 for v in plant[n] if v) for n in nurses}
        nts = {n: sum(1 for v in plant[n] if v == 3) for n in nurses}
        data = dict(base)
        data["min_shifts"] = min(totals.values())
        data["max_shifts"] = max(totals.values())
        data["max_nights"] = max(1, max(nts.values()))
        offreq = {}
        nreq = {1: 1, 2: 2, 3: 4, 4: 6, 5: 9}[level]
        cand = [n for n in nurses]
        rng.shuffle(cand)
        for n in cand[:nreq]:
            offdays = [d + 1 for d in range(D) if plant[n][d] == 0]
            if offdays:
                k = min(len(offdays), rng.choice([1, 1, 2]) if D >= 5 else 1)
                offreq[n] = sorted(rng.sample(offdays, k))
        data["off_requests"] = {n: offreq[n] for n in nurses if n in offreq}
        zero_night = [n for n in nurses if nts[n] == 0 and n not in senior]
        rng.shuffle(zero_night)
        data["no_nights"] = sorted(zero_night[:{1: 1, 2: 1, 3: 1, 4: 2, 5: 2}[level]],
                                   key=nurses.index)
        assert valid_roster(data, plant)
        st = {}
        try:
            sol = solve_robust(data, seed=seed, stats=st)
        except _Limit:
            continue             # too hard for our search budget: resample
        if sol is None:          # cannot happen: the planted roster is valid
            raise RuntimeError("roster solver missed a planted solution")
        break
    else:
        raise RuntimeError("roster generator failed")
    gold = {f"shift:{n}:{d + 1}": sol[n][d] for n in nurses for d in range(D)}
    return {
        "id": item_id(FAMILY, level, seed),
        "family": FAMILY,
        "level": level,
        "seed": seed,
        "text": _render(data, rng),
        "answer_keys": answer_keys(D),
        "kind": KIND,
        "gold": gold,
        "optimum": None,
        "data": data,
    }


# ---------------------------------------------------------------- self-test hooks
def wrong_answers(item, rng, n=20):
    data = item["data"]
    out = [{"impossible": True}]
    keys = slot_keys(data)
    for _ in range(n):
        ans = dict(item["gold"])
        k = rng.choice(keys)
        ans[k] = rng.choice([v for v in (0, 1, 2, 3) if v != ans[k]])
        out.append(ans)
    return out


def solve_item(item, stats=None):
    sol = solve_robust(item["data"], seed=item["seed"], stats=stats)
    data = item["data"]
    return {f"shift:{n}:{d + 1}": sol[n][d] for n in data["nurses"] for d in range(data["D"])}
