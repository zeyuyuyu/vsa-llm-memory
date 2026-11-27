"""Independent z3 cross-checks, used only by selftest.py.

Each function encodes the item's rules directly from item["data"] (not from the
family module's solver) and returns (verdict, answer):
  verdict "agree"    z3 confirms the gold verdict / optimum / uniqueness
          "disagree" z3 contradicts it
          "unknown"  z3 timed out
  answer  a z3-found answer in slot format (or None); selftest feeds it to the
          family's check() to confirm check() accepts answers other than the gold.
"""
import z3


def _solver(timeout_ms):
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    return s


def meetings(item, timeout_ms=30000):
    d = item["data"]
    s = _solver(timeout_ms)
    day, st = {}, {}
    for m in d["meetings"]:
        n, h = m["name"], m["dur"]
        day[n], st[n] = z3.Int(f"day_{n}"), z3.Int(f"st_{n}")
        s.add(day[n] >= 1, day[n] <= d["D"], st[n] >= 9, st[n] + h <= 17,
              z3.Or(st[n] + h <= 12, st[n] >= 13))
        al = d["allowed_days"].get(n)
        if al is not None:
            s.add(z3.Or([day[n] == a for a in al]))
        for p in m["att"]:
            for bd, f, t in d["busy"].get(p, []):
                s.add(z3.Or(day[n] != bd, st[n] >= t, st[n] + h <= f))
    ms = d["meetings"]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            if set(ms[i]["att"]) & set(ms[j]["att"]):
                a, b = ms[i]["name"], ms[j]["name"]
                s.add(z3.Or(day[a] != day[b], st[a] + ms[i]["dur"] <= st[b],
                            st[b] + ms[j]["dur"] <= st[a]))
    for p in d["people"]:
        mine = [m for m in ms if p in m["att"]]
        for dd in range(1, d["D"] + 1):
            if mine:
                s.add(z3.Sum([z3.If(day[m["name"]] == dd, m["dur"], 0) for m in mine])
                      <= d["cap"])
    for a, b in d["before"]:
        s.add(day[a] < day[b])
    r = s.check()
    if r == z3.unknown:
        return "unknown", None
    possible = not item["gold"].get("impossible")
    if r == z3.sat:
        mod = s.model()
        ans = {}
        for m in ms:
            ans[f"day:{m['name']}"] = mod.eval(day[m["name"]]).as_long()
            ans[f"start:{m['name']}"] = mod.eval(st[m["name"]]).as_long()
        return ("agree" if possible else "disagree"), ans
    return ("agree" if not possible else "disagree"), None


def roster(item, timeout_ms=30000):
    d = item["data"]
    s = _solver(timeout_ms)
    D, K = d["D"], d["max_consec"]
    x = {(n, t): z3.Int(f"x_{n}_{t}") for n in d["nurses"] for t in range(D)}
    for v in x.values():
        s.add(v >= 0, v <= 3)
    for t in range(D):
        for sh in (1, 2, 3):
            s.add(z3.Sum([z3.If(x[n, t] == sh, 1, 0) for n in d["nurses"]])
                  == d["cover"][t][sh - 1])
        if d["cover"][t][2] >= 1:
            s.add(z3.Or([x[n, t] == 3 for n in d["senior"]]))
    for n in d["nurses"]:
        for t in range(D - 1):
            s.add(z3.Implies(x[n, t] == 3, z3.Or(x[n, t + 1] == 0, x[n, t + 1] == 3)))
            s.add(z3.Implies(x[n, t] == 2, x[n, t + 1] != 1))
        for t in range(D - K):
            s.add(z3.Sum([z3.If(x[n, u] != 0, 1, 0) for u in range(t, t + K + 1)]) <= K)
        tot = z3.Sum([z3.If(x[n, t] != 0, 1, 0) for t in range(D)])
        s.add(tot >= d["min_shifts"], tot <= d["max_shifts"])
        s.add(z3.Sum([z3.If(x[n, t] == 3, 1, 0) for t in range(D)]) <= d["max_nights"])
        for dd in d["off_requests"].get(n, []):
            s.add(x[n, dd - 1] == 0)
        if n in d["no_nights"]:
            for t in range(D):
                s.add(x[n, t] != 3)
    r = s.check()
    if r == z3.unknown:
        return "unknown", None
    if r == z3.unsat:
        return "disagree", None
    mod = s.model()
    ans = {f"shift:{n}:{t + 1}": mod.eval(x[n, t]).as_long()
           for n in d["nurses"] for t in range(D)}
    return "agree", ans


def project(item, timeout_ms=30000):
    d = item["data"]
    opt = item["optimum"]
    tasks = d["tasks"]

    def model_for(T):
        s = _solver(timeout_ms)
        st = {t["code"]: z3.Int(f"s_{t['code']}") for t in tasks}
        by = {t["code"]: t for t in tasks}
        for t in tasks:
            c = t["code"]
            s.add(st[c] >= 1, st[c] >= t["release"], st[c] + t["dur"] - 1 <= T)
            for p in t["preds"]:
                s.add(st[c] >= st[p] + by[p]["dur"])
        for day in range(1, T + 1):
            on = lambda t: z3.And(st[t["code"]] <= day, day <= st[t["code"]] + t["dur"] - 1)
            s.add(z3.Sum([z3.If(on(t), t["crew"], 0) for t in tasks]) <= d["crew"])
            mixers = [t for t in tasks if t["mixer"]]
            if mixers:
                s.add(z3.Sum([z3.If(on(t), 1, 0) for t in mixers]) <= 1)
        return s, st

    s, st = model_for(opt)
    r1 = s.check()
    if r1 == z3.unknown:
        return "unknown", None
    if r1 == z3.unsat:
        return "disagree", None
    mod = s.model()
    ans = {f"start:{c}": mod.eval(v).as_long() for c, v in st.items()}
    if opt - 1 < 1:
        return "agree", ans
    s2, _ = model_for(opt - 1)
    r2 = s2.check()
    if r2 == z3.unknown:
        return "unknown", ans
    return ("agree" if r2 == z3.unsat else "disagree"), ans


def islanders(item, timeout_ms=30000):
    d = item["data"]
    N = len(d["people"])
    k = [z3.Bool(f"k_{i}") for i in range(N)]
    s = _solver(timeout_ms)

    def is_v(i, v):
        return k[i] if v == 1 else z3.Not(k[i])

    for st in d["statements"]:
        t = st["t"]
        if t == "is":
            e = is_v(st["x"], st["v"])
        elif t == "same":
            e = k[st["x"]] == k[st["y"]]
        elif t == "diff":
            e = k[st["x"]] != k[st["y"]]
        elif t == "if":
            e = z3.Implies(is_v(st["x"], st["vx"]), is_v(st["y"], st["vy"]))
        else:
            c = z3.Sum([z3.If(is_v(g, st["v"]), 1, 0) for g in st["g"]])
            e = {"exactly": c == st["k"], "atleast": c >= st["k"],
                 "atmost": c <= st["k"]}[st["op"]]
        s.add(k[st["s"]] == e)
    r = s.check()
    if r == z3.unknown:
        return "unknown", None
    if r == z3.unsat:
        return "disagree", None
    mod = s.model()
    names = d["people"]
    ans = {f"knight:{names[i]}": 1 if z3.is_true(mod.eval(k[i])) else 0 for i in range(N)}
    if ans != item["gold"]:
        return "disagree", ans
    s.add(z3.Or([k[i] != (item["gold"][f"knight:{names[i]}"] == 1) for i in range(N)]))
    r2 = s.check()
    if r2 == z3.unknown:
        return "unknown", ans
    return ("agree" if r2 == z3.unsat else "disagree"), ans


def vans(item, timeout_ms=30000):
    d = item["data"]
    opt = item["optimum"]
    ps = d["parcels"]

    def model_for(K):
        s = _solver(timeout_ms)
        v = {p["id"]: z3.Int(f"v_{p['id']}") for p in ps}
        for i, p in enumerate(ps):
            s.add(v[p["id"]] >= 1, v[p["id"]] <= min(K, i + 1))   # weak symmetry break
        for b in range(1, K + 1):
            s.add(z3.Sum([z3.If(v[p["id"]] == b, p["w"], 0) for p in ps]) <= d["W"])
            s.add(z3.Sum([z3.If(v[p["id"]] == b, p["s"], 0) for p in ps]) <= d["S"])
        for a, b in d["apart"]:
            s.add(v[a] != v[b])
        for a, b in d["together"]:
            s.add(v[a] == v[b])
        return s, v

    s, v = model_for(opt)
    r1 = s.check()
    if r1 == z3.unknown:
        return "unknown", None
    if r1 == z3.unsat:
        return "disagree", None
    mod = s.model()
    ans = {f"van:{pid}": mod.eval(x).as_long() for pid, x in v.items()}
    if opt - 1 < 1:
        return "agree", ans
    s2, _ = model_for(opt - 1)
    r2 = s2.check()
    if r2 == z3.unknown:
        return "unknown", ans
    return ("agree" if r2 == z3.unsat else "disagree"), ans


CHECKS = dict(meetings=meetings, roster=roster, project=project,
              islanders=islanders, vans=vans)
