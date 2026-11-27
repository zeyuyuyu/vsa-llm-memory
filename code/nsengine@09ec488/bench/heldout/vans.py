"""Family "vans": a dispatcher loads parcels into identical vans with a weight
limit and a number of pallet spaces; some pairs of parcels must travel apart and
some must travel together.  Fewest vans wins.

kind = "optimize": any assignment that respects every rule and uses the optimal
number of vans is right.  The optimum comes from the exact solver below: for
k = lower bound, lower bound + 1, ... it runs a complete depth-first packing
search (largest parcels first, symmetric bins merged, failed states memoised)
until some k is feasible.
"""
from _common import rng_for, item_id, is_impossible, read_slots

FAMILY = "vans"
KIND = "optimize"
GOODS = ["boxed books", "floor tiles", "paint tins", "garden furniture",
         "office chairs", "kitchen cabinets", "bathroom fittings", "rolls of carpet",
         "shelving units", "mattresses", "table lamps", "rugs", "ceramic pots",
         "garden tools", "printer paper", "bicycle parts", "curtain fabric",
         "outdoor heaters", "flat-pack desks", "flat-pack wardrobes", "dining chairs",
         "bed frames", "washing machines", "fridges", "cookers", "bookcases",
         "wall mirrors", "sofa cushions", "door handles", "patio slabs"]
LEVELS = {
    1: dict(n=6, apart=1, together=0),
    2: dict(n=9, apart=2, together=1),
    3: dict(n=13, apart=3, together=1),
    4: dict(n=18, apart=4, together=2),
    5: dict(n=24, apart=6, together=2),
    # levels 6-8: planted near-perfect loadings (see _planted_instance)
    6: dict(n=50, apart=8, together=3, per_van=(3, 6)),
    7: dict(n=80, apart=12, together=4, per_van=(3, 6)),
    8: dict(n=120, apart=18, together=6, per_van=(3, 7)),
}
PLANTED_FROM = 6
NODE_LIMIT = 2_000_000


class _Limit(Exception):
    pass


def answer_keys(n):
    return (f"van:<parcel> for every parcel (P1 to P{n}): the number of the van it "
            "goes in (vans numbered 1, 2, 3, ...)")


def slot_keys(data):
    return [f"van:{p['id']}" for p in data["parcels"]]


# ---------------------------------------------------------------- solver
def _groups(data):
    """Merge must-travel-together parcels into groups (union-find)."""
    ids = [p["id"] for p in data["parcels"]]
    par = {i: i for i in ids}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in data["together"]:
        par[f(a)] = f(b)
    groups = {}
    for i in ids:
        groups.setdefault(f(i), []).append(i)
    by = {p["id"]: p for p in data["parcels"]}
    out = []
    for members in groups.values():
        out.append(dict(members=members,
                        w=sum(by[m]["w"] for m in members),
                        s=sum(by[m]["s"] for m in members)))
    gid = {m: gi for gi, g in enumerate(out) for m in g["members"]}
    conf = [set() for _ in out]
    bad = False
    for a, b in data["apart"]:
        ga, gb = gid[a], gid[b]
        if ga == gb:
            bad = True
        conf[ga].add(gb)
        conf[gb].add(ga)
    return out, conf, bad


def lower_bound(data):
    W, S = data["W"], data["S"]
    groups, conf, _ = _groups(data)
    n = len(groups)
    lb = max(-(-sum(g["w"] for g in groups) // W), -(-sum(g["s"] for g in groups) // S), 1)
    inc = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (j in conf[i] or groups[i]["w"] + groups[j]["w"] > W
                           or groups[i]["s"] + groups[j]["s"] > S):
                inc[i].add(j)
    # greedy cliques of pairwise-incompatible groups, from several starts
    best = 1
    for start in range(n):
        cl = [start]
        for j in sorted(range(n), key=lambda x: -len(inc[x])):
            if j not in cl and all(j in inc[c] for c in cl):
                cl.append(j)
        best = max(best, len(cl))
    return max(lb, best)


def _ffd(groups, conf, W, S, order):
    bins = []
    for gi in order:
        g = groups[gi]
        for b in bins:
            if b[0] + g["w"] <= W and b[1] + g["s"] <= S and not (b[2] & conf[gi]):
                b[0] += g["w"]
                b[1] += g["s"]
                b[2].add(gi)
                break
        else:
            bins.append([g["w"], g["s"], {gi}])
    return bins


def _feasible(groups, conf, W, S, k, node_limit, nodes):
    """Complete search for a packing into k vans; returns group -> bin or None."""
    n = len(groups)
    order = sorted(range(n), key=lambda i: -(groups[i]["w"] / W + groups[i]["s"] / S
                                             + 0.05 * len(conf[i])))
    cbits = [0] * n
    for gi in range(n):
        for c in conf[gi]:
            cbits[gi] |= 1 << c
    rem_w = [0] * (n + 1)
    rem_s = [0] * (n + 1)
    rem_mask = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        g = order[i]
        rem_w[i] = rem_w[i + 1] + groups[g]["w"]
        rem_s[i] = rem_s[i + 1] + groups[g]["s"]
        rem_mask[i] = rem_mask[i + 1] | (1 << g)
    gw = [g["w"] for g in groups]
    gs = [g["s"] for g in groups]
    bw, bs, bcf = [], [], []        # per bin: weight, spaces, OR of members' conflicts
    assign = [None] * n
    failed = set()

    def rec(i):
        nodes[0] += 1
        if node_limit is not None and nodes[0] > node_limit:
            raise _Limit()
        if i == n:
            return True
        nb = len(bw)
        if rem_w[i] > k * W - sum(bw) or rem_s[i] > k * S - sum(bs):
            return False
        rm = rem_mask[i]
        kk = (i, tuple(sorted(zip(bw, bs, [c & rm for c in bcf]))))
        if kk in failed:
            return False
        gi = order[i]
        w, s, bit = gw[gi], gs[gi], 1 << gi
        tried = set()
        for b in range(nb):
            if bw[b] + w > W or bs[b] + s > S or (bcf[b] & bit):
                continue
            sig = (bw[b], bs[b], bcf[b] & rm)
            if sig in tried:
                continue
            tried.add(sig)
            old = bcf[b]
            bw[b] += w
            bs[b] += s
            bcf[b] = old | cbits[gi]
            assign[gi] = b
            if rec(i + 1):
                return True
            bw[b] -= w
            bs[b] -= s
            bcf[b] = old
        if nb < k:
            bw.append(w)
            bs.append(s)
            bcf.append(cbits[gi])
            assign[gi] = nb
            if rec(i + 1):
                return True
            bw.pop()
            bs.pop()
            bcf.pop()
        failed.add(kk)
        return False

    if rec(0):
        return list(assign)
    return None


def solve(data, stats=None, node_limit=None):
    """Return ({parcel_id: van number}, optimal number of vans)."""
    W, S = data["W"], data["S"]
    groups, conf, bad = _groups(data)
    if bad or any(g["w"] > W or g["s"] > S for g in groups):
        raise ValueError("infeasible instance")
    n = len(groups)
    lb = lower_bound(data)
    # upper bound from first-fit decreasing under a few orderings
    best_bins = None
    for keyf in (lambda i: -groups[i]["w"], lambda i: -groups[i]["s"],
                 lambda i: -(groups[i]["w"] / W + groups[i]["s"] / S)):
        bins = _ffd(groups, conf, W, S, sorted(range(n), key=keyf))
        if best_bins is None or len(bins) < len(best_bins):
            best_bins = bins
    ub = len(best_bins)
    nodes = [0]
    assign = None
    k = lb
    while k < ub:
        assign = _feasible(groups, conf, W, S, k, node_limit, nodes)
        if assign is not None:
            break
        k += 1
    if assign is None:
        k = ub
        assign = [None] * n
        for b, (_, _, members) in enumerate(best_bins):
            for gi in members:
                assign[gi] = b
    if stats is not None:
        stats["nodes"] = nodes[0]
        stats["lb"] = lb
        stats["ffd"] = ub
    out = {}
    for gi, g in enumerate(groups):
        for m in g["members"]:
            out[m] = assign[gi] + 1
    return out, k


# ---------------------------------------------------------------- checking
def valid_loading(data, vans):
    """vans: {parcel_id: van label}.  Returns number of vans used, or None."""
    loads = {}
    for p in data["parcels"]:
        v = vans[p["id"]]
        if v < 1:
            return None
        lw, ls = loads.get(v, (0, 0))
        loads[v] = (lw + p["w"], ls + p["s"])
    for lw, ls in loads.values():
        if lw > data["W"] or ls > data["S"]:
            return None
    for a, b in data["apart"]:
        if vans[a] == vans[b]:
            return None
    for a, b in data["together"]:
        if vans[a] != vans[b]:
            return None
    return len(loads)


def check(item, answer):
    data = item["data"]
    if is_impossible(answer):
        return False
    ans = read_slots(answer, slot_keys(data))
    if ans is None:
        return False
    vans = {p["id"]: ans[f"van:{p['id']}"] for p in data["parcels"]}
    used = valid_loading(data, vans)
    return used is not None and used == item["optimum"]


# ---------------------------------------------------------------- generation
def _random_instance(rng, level):
    L = LEVELS[level]
    n = L["n"]
    W = rng.choice([800, 1000, 1200, 1500])
    S = rng.choice([6, 8, 10])
    parcels = []
    goods = list(GOODS)
    rng.shuffle(goods)
    for i in range(n):
        # mostly mid-sized parcels plus some large ones
        if rng.random() < 0.25:
            w = rng.uniform(0.4, 0.7) * W
        else:
            w = rng.uniform(0.08, 0.4) * W
        w = max(10, int(round(w / 10.0)) * 10)
        s = rng.choices(range(1, S // 2 + 2), k=1)[0]
        s = min(s, S - 1)
        parcels.append(dict(id=f"P{i + 1}", w=w, s=s, desc=goods[i % len(goods)]))
    ids = [p["id"] for p in parcels]
    pairs = set()
    apart = []
    while len(apart) < L["apart"]:
        a, b = sorted(rng.sample(range(n), 2))
        if (a, b) in pairs:
            continue
        pairs.add((a, b))
        apart.append([ids[a], ids[b]])
    together = []
    by = {p["id"]: p for p in parcels}
    tries = 0
    while len(together) < L["together"] and tries < 100:
        tries += 1
        a, b = sorted(rng.sample(range(n), 2))
        if (a, b) in pairs:
            continue
        pa, pb = by[ids[a]], by[ids[b]]
        if pa["w"] + pb["w"] > 0.8 * W or pa["s"] + pb["s"] > S - 1:
            continue
        pairs.add((a, b))
        together.append([ids[a], ids[b]])
    data = dict(W=W, S=S, parcels=parcels, apart=apart, together=together)
    groups, _, bad = _groups(data)
    if bad or any(g["w"] > W or g["s"] > S for g in groups):
        return None
    return data


def greedy_vans(data):
    """Fewest vans reached by first-fit decreasing and best-fit decreasing under
    three size orderings (the 'simple heuristic' the large levels must defeat)."""
    W, S = data["W"], data["S"]
    groups, conf, _ = _groups(data)
    n = len(groups)
    best = None
    for keyf in (lambda i: -groups[i]["w"], lambda i: -groups[i]["s"],
                 lambda i: -(groups[i]["w"] / W + groups[i]["s"] / S)):
        order = sorted(range(n), key=keyf)
        k1 = len(_ffd(groups, conf, W, S, order))
        bins = []
        for gi in order:
            g = groups[gi]
            fit = [b for b in bins if b[0] + g["w"] <= W and b[1] + g["s"] <= S
                   and not (b[2] & conf[gi])]
            if fit:
                b = min(fit, key=lambda b: (W - b[0] - g["w"]) / W + (S - b[1] - g["s"]) / S)
                b[0] += g["w"]
                b[1] += g["s"]
                b[2].add(gi)
            else:
                bins.append([g["w"], g["s"], {gi}])
        k = min(k1, len(bins))
        best = k if best is None else min(best, k)
    return best


def weight_space_bound(data):
    """The provable lower bound used for the planted levels: total weight over
    the van limit, and total pallet spaces over the spaces per van, rounded up."""
    tw = sum(p["w"] for p in data["parcels"])
    ts = sum(p["s"] for p in data["parcels"])
    return max(-(-tw // data["W"]), -(-ts // data["S"]))


def _split(rng, total, m, lo):
    """Split integer `total` into m parts, each >= lo, uniformly at random."""
    free = total - m * lo
    if free < 0:
        return None
    cuts = sorted(rng.randint(0, free) for _ in range(m - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [free])]
    return [p + lo for p in parts]


def _planted_instance(rng, level):
    """Plant k nearly full vans.  Every van is filled to within 30 kg of the
    weight limit, and the total slack is kept below one van's limit, so the
    weight bound ceil(total / W) equals k: no loading can use fewer vans, and
    the planted loading uses exactly k."""
    L = LEVELS[level]
    n = L["n"]
    W = rng.choice([1000, 1200, 1500])
    S = rng.choice([8, 10, 12])
    lo, hi = L["per_van"]
    counts = []
    while sum(counts) < n:
        counts.append(rng.randint(lo, hi))
    counts[-1] -= sum(counts) - n
    if counts[-1] < 2:
        return None
    k = len(counts)
    parcels = []
    total_slack = 0
    for v, m in enumerate(counts):
        slack = rng.choice([0, 0, 10, 10, 20, 30])
        total_slack += slack
        ws = _split(rng, (W - slack) // 10, m, 4)          # in units of 10 kg, >= 40 kg
        sslack = rng.choice([0, 0, 1, 2])
        ss = _split(rng, S - sslack, m, 1)
        if ws is None or ss is None:
            return None
        for w, s in zip(ws, ss):
            parcels.append(dict(w=10 * w, s=s, van=v + 1))
    if total_slack >= W:
        return None
    rng.shuffle(parcels)
    goods = list(GOODS)
    rng.shuffle(goods)
    planted = {}
    out = []
    for i, p in enumerate(parcels):
        pid = f"P{i + 1}"
        out.append(dict(id=pid, w=p["w"], s=p["s"], desc=goods[i % len(goods)]))
        planted[pid] = p["van"]
    ids = [p["id"] for p in out]
    apart, together, used = [], [], set()
    tries = 0
    while len(apart) < L["apart"] and tries < 1000:
        tries += 1
        a, b = sorted(rng.sample(range(n), 2))
        if (a, b) in used or planted[ids[a]] == planted[ids[b]]:
            continue
        used.add((a, b))
        apart.append([ids[a], ids[b]])
    tries = 0
    while len(together) < L["together"] and tries < 1000:
        tries += 1
        a, b = sorted(rng.sample(range(n), 2))
        if (a, b) in used or planted[ids[a]] != planted[ids[b]]:
            continue
        used.add((a, b))
        together.append([ids[a], ids[b]])
    data = dict(W=W, S=S, parcels=out, apart=apart, together=together)
    if valid_loading(data, planted) != k or weight_space_bound(data) != k:
        return None
    return data, planted, k


def proof_check(item):
    """For planted items: the gold loading is valid, uses `optimum` vans, and the
    weight/space bound equals `optimum`, so the optimum is proved."""
    data = item["data"]
    vans = {p["id"]: item["gold"][f"van:{p['id']}"] for p in data["parcels"]}
    return (valid_loading(data, vans) == item["optimum"]
            and weight_space_bound(data) == item["optimum"])


def _render(data, rng):
    n = len(data["parcels"])
    intro = rng.choice([
        "I run dispatch at a furniture and homeware warehouse.",
        "I'm planning tomorrow morning's loading at our distribution centre.",
        "I look after the delivery vans at a small removals company.",
    ])
    fluff = [
        "The drivers start at 6:30 and like to be on the road by 7:15.",
        "All vans are fuelled and have passed their checks this week.",
        "The forklift will be serviced next Tuesday.",
        "Every parcel is already wrapped and labelled.",
        "Last month we used six vans on an average morning.",
    ]
    rng.shuffle(fluff)
    lines = [intro, fluff[0], ""]
    lines.append(
        f"Tomorrow {n} parcels go to the same depot, so any parcel can go in any van. "
        f"All our vans are identical and we have more of them than we could need. Each "
        f"van can carry at most {data['W']} kg in total and has {data['S']} pallet "
        f"spaces; each parcel uses the stated number of pallet spaces, and the parcels "
        f"in a van may use at most {data['S']} spaces together. A parcel cannot be "
        f"split between vans.")
    lines.append("")
    lines.append("Parcels:")
    for p in data["parcels"]:
        lines.append(f"- {p['id']} (an order of {p['desc']}): {p['w']} kg, {p['s']} "
                     f"pallet space{'s' if p['s'] > 1 else ''}.")
    lines.append("")
    lines.append("Other rules:")
    for a, b in data["apart"]:
        lines.append(f"- {a} and {b} must not go in the same van.")
    for a, b in data["together"]:
        lines.append(f"- {a} and {b} are for the same customer and must go in the "
                     "same van.")
    lines.append("")
    lines.append(fluff[1])
    lines.append("")
    lines.append("I want to send out as few vans as possible. Find a loading of all the "
                 "parcels that uses the smallest possible number of vans, and say which "
                 "van each parcel goes in.")
    return "\n".join(lines)


def generate(seed, level):
    rng = rng_for(FAMILY, level, seed)
    chosen = None
    if level >= PLANTED_FROM:
        for attempt in range(500):
            res = _planted_instance(rng, level)
            if res is None:
                continue
            data, planted, k = res
            if greedy_vans(data) <= k:      # the simple heuristic must fail
                continue
            chosen = (data, planted, k)
            break
    for attempt in range(100 if level < PLANTED_FROM else 0):
        data = _random_instance(rng, level)
        if data is None:
            continue
        st = {}
        try:
            vans, opt = solve(data, stats=st, node_limit=NODE_LIMIT)
        except _Limit:
            continue
        # skip instances where simple first-fit-decreasing is already optimal
        # (at the higher levels, so that real search is needed)
        if level >= 3 and st["ffd"] == opt and attempt < 60:
            continue
        chosen = (data, vans, opt)
        break
    if chosen is None:
        raise RuntimeError("vans generator failed")
    data, vans, opt = chosen
    gold = {f"van:{p['id']}": vans[p["id"]] for p in data["parcels"]}
    return {
        "id": item_id(FAMILY, level, seed),
        "family": FAMILY,
        "level": level,
        "seed": seed,
        "text": _render(data, rng),
        "answer_keys": answer_keys(len(data["parcels"])),
        "kind": KIND,
        "gold": gold,
        "optimum": opt,
        "data": data,
    }


# ---------------------------------------------------------------- self-test hooks
def wrong_answers(item, rng, n=20):
    out = [{"impossible": True}]
    keys = slot_keys(item["data"])
    k = item["optimum"]
    for _ in range(n):
        ans = dict(item["gold"])
        key = rng.choice(keys)
        ans[key] = rng.choice([v for v in range(1, k + 2) if v != ans[key]])
        out.append(ans)
    return out


def solve_item(item, stats=None):
    if item["level"] >= PLANTED_FROM:
        # planted levels: the gold is certified by proof_check instead of search
        if not proof_check(item):
            raise AssertionError("planted vans item fails its proof")
        return dict(item["gold"])
    vans, opt = solve(item["data"], stats=stats)
    return {f"van:{p['id']}": vans[p["id"]] for p in item["data"]["parcels"]}
