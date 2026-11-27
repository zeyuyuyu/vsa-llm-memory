"""NSR2 (development prototype): a neuro-symbolic engine for impact questions over the read state.

  neural   the shipped reader writes the state; a small learned reliability model scores each state fact from features
           anyone can observe in the log (which channel wrote it, how many moves the message carried, how many tracked
           names it mentions, how late it came) -> P(fact correct)
  symbolic every thing gets a distribution over candidate places (the place read, with P = reliability; the place the
           memory held before that write, with the rest); impact questions are answered by exact probabilistic
           inference over these candidates and the site catalog: P(x is affected) for every candidate x, the answer set
           {x : P >= 0.5}, and a question confidence = prod_x max(P, 1 - P), used to abstain.

python3 nsr/nsr2.py --train dev,test --eval nsr1
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1 as N  # noqa: E402
import nsr1_catalog as C  # noqa: E402
import nsr2_types as T  # noqa: E402

SPLITS = {"dev": ("apps/real/data", "apps/real/out/reader_dev_verify_v4.jsonl"),
          "test": ("apps/real/data", "apps/real/out/reader_test_verify_v4.jsonl"),
          "nsr1": ("nsr/data", "nsr/out/held/reader_nsr1_verify_v4_all.jsonl")}
ROOT = os.path.dirname(HERE)


def load(split):
    ddir, rpath = SPLITS[split]
    logs = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for line in open(os.path.join(ROOT, ddir, f"{sc}_{split}.jsonl")):
            d = json.loads(line)
            logs[d["id"]] = d
    rd = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, rpath))}
    return logs, rd


CHANNELS = ["chat", "speech", "incident", "bench", "email", "handover", "wms", "lims", "vision", "deploy", "infra"]


def names_in(text, log):
    t = (text or "").lower()
    n = 0
    for e in log["registry"]["items"]:
        if e["id"].lower() in t or any(a.lower() in t for a in e["aliases"] if len(a) >= 4):
            n += 1
    return n


def prev_place(rec, x, i):
    if i is None or i == 0:
        return None
    v = rec["snapshots"][i - 1].get(x)
    if v is None:
        return None
    v = v[0]
    return v[0] if isinstance(v, list) else v


def facts(log, rec):
    """One row per state fact (thing, place): observable features, the alternative place, and the label (correct?)."""
    rm, src = N.rm_state(rec)
    rows = []
    n_lines = len(log["lines"])
    for x, p in rm.items():
        i = src.get(x)
        line = log["lines"][i] if i is not None else None
        human = line is not None and line["kind"] == "human"
        ch = line["src"] if line is not None else "none"
        nev = len(rec["events"][i] or []) if i is not None else 0
        f = [1.0, float(human), float(human and nev > 1), float(nev > 1), min(names_in(log["text_lines"][i], log), 4) / 4
             if i is not None else 0.0, (i / n_lines) if i is not None else 0.0, float(i is None)]
        f += [float(ch == c) for c in CHANNELS]
        alt = prev_place(rec, x, i)
        rows.append({"x": x, "p": p, "alt": alt, "feat": f, "ok": log["truth"].get(x) == p,
                     "truth_is_prev": alt is not None and log["truth"].get(x) == alt,
                     "kind": T.etype(log["scenario"], x, log)})
    return rows


def fit_reliability(train_rows):
    from sklearn.linear_model import LogisticRegression
    X = np.array([r["feat"] for r in train_rows])
    y = np.array([r["ok"] for r in train_rows])
    return LogisticRegression(C=1.0, max_iter=2000, class_weight=None).fit(X, y)


def place_dist(rows, model, places, w_prev):
    """thing -> {place: prob}. Read place: the reliability q. The rest, 1 - q, goes to the previous place with weight
    w_prev (learned: how often a wrong fact's truth is the previous place) and is spread uniformly over the other
    type-consistent places (the candidate places of that thing's kind in this log)."""
    if not rows:
        return {}
    pr = model.predict_proba(np.array([r["feat"] for r in rows]))[:, 1]
    out = {}
    for r, q in zip(rows, pr):
        q = float(q)
        allowed = set(places.get(r["kind"], []))
        if allowed and r["p"] not in allowed:
            q = 0.0  # the read place breaks the learned containment-type table (e.g. a rack in a box): no mass
        d = collections.defaultdict(float)
        d[r["p"]] += q
        rest = 1.0 - q
        others = [pl for pl in places.get(r["kind"], []) if pl != r["p"]]
        if r["alt"] and r["alt"] != r["p"] and (not allowed or r["alt"] in allowed):
            d[r["alt"]] += rest * w_prev
            rest *= (1 - w_prev)
        if others:
            for pl in others:
                d[pl] += rest / len(others)
        else:
            d[r["p"]] += rest
        out[r["x"]] = dict(d)
    return out


def p_reaches(x, target, dist, part, depth=0, memo=None):
    """P(the containment chain of x reaches target), recursing through the distributions of containers."""
    if memo is None:
        memo = {}
    if (x, target) in memo:
        return memo[(x, target)]
    if depth > 6:
        return 0.0
    if x in dist:
        tot = 0.0
        for loc, q in dist[x].items():
            tot += q * (1.0 if loc == target else p_reaches(loc, target, dist, part, depth + 1, memo))
        memo[(x, target)] = tot
        return tot
    y = part.get(x)
    res = 1.0 if y == target else (p_reaches(y, target, dist, part, depth + 1, memo) if y else 0.0)
    memo[(x, target)] = res
    return res


def candidate_places(log, table):
    """kind of thing -> places in this log whose kind may hold it (the learned containment-type table)."""
    sc = log["scenario"]
    names = [e["id"] for e in log["registry"]["places"]] + [e["id"] for e in log["registry"]["items"]]
    out = collections.defaultdict(list)
    for child in {c for s_, c, p in table if s_ == sc}:
        for pl in names:
            if (sc, child, T.etype(sc, pl, log)) in table:
                out[child].append(pl)
    return out


def answer(q, dist, cat):
    """Item marginals, MAP answer set and question confidence for one impact question."""
    part = {a: b for a, r, b in cat if r == "part_of"}
    needs = {a: b for a, r, b in cat if r == "needs"}
    cls = N.ASKED_CLASS[q["qtype"]]
    marg = {}
    if q["qtype"] == "host_down":
        for s, d in needs.items():
            if cls(s):
                marg[s] = dist.get(d, {}).get(q["target"], 0.0)
    else:
        for x in dist:
            if cls(x):
                pr = p_reaches(x, q["target"], dist, part)
                if pr > 0:
                    marg[x] = pr
    pred = sorted(x for x, p in marg.items() if p >= 0.5)
    conf = float(np.prod([max(p, 1 - p) for p in marg.values()])) if marg else 1.0
    return pred, conf, marg


def auroc(conf, ok):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(ok, conf) if 0 < sum(ok) < len(ok) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dev,test")
    ap.add_argument("--eval", default="nsr1")
    a = ap.parse_args()
    train_rows = []
    for sp in a.train.split(","):
        logs, rd = load(sp)
        for lid, log in logs.items():
            train_rows += facts(log, rd[lid])
    model = fit_reliability(train_rows)
    table = set()
    wrong = [r for r in train_rows if not r["ok"]]
    for sp in a.train.split(","):
        table |= T.learn_table(load(sp)[0].values())
    w_prev = sum(1 for r in wrong if r.get("truth_is_prev")) / max(len(wrong), 1)
    logs, rd = load(a.eval)
    ev_rows, res = [], []
    for lid, log in logs.items():
        rows = facts(log, rd[lid])
        ev_rows += rows
        dist = place_dist(rows, model, candidate_places(log, table), w_prev)
        cat = C.catalog(log)
        rm, _ = N.rm_state(rd[lid])
        for q in C.questions(log):
            det = C.answer(q["qtype"], q["target"], rm, cat)
            pred, conf, _ = answer(q, dist, cat)
            res.append({"det_ok": det == q["a"], "ns_ok": pred == q["a"], "conf": conf, "q": q})
    print(f"w_prev (learned) {w_prev:.3f}")
    pf = model.predict_proba(np.array([r["feat"] for r in ev_rows]))[:, 1]
    fok = [r["ok"] for r in ev_rows]
    print(f"train facts {len(train_rows)} ({sum(not r['ok'] for r in train_rows)} wrong); eval facts {len(ev_rows)} "
          f"({sum(not o for o in fok)} wrong); fact-level AUROC of reliability {auroc(list(pf), fok):.3f}")
    n = len(res)
    print(f"questions {n}: deterministic {sum(r['det_ok'] for r in res) / n:.3f}  NS MAP {sum(r['ns_ok'] for r in res) / n:.3f}")
    conf = [r["conf"] for r in res]
    print(f"question-level AUROC (confidence vs NS correct) {auroc(conf, [r['ns_ok'] for r in res]):.3f}")
    order = sorted(range(n), key=lambda i: -conf[i])
    for cov in (1.0, 0.95, 0.9, 0.85, 0.8):
        k = int(round(cov * n))
        sel = order[:k]
        print(f"  coverage {cov:.2f}: NS selective accuracy {sum(res[i]['ns_ok'] for i in sel) / k:.3f} "
              f"(deterministic on the same questions {sum(res[i]['det_ok'] for i in sel) / k:.3f})")


if __name__ == "__main__":
    main()
