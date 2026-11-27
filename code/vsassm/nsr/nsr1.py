"""NSR1 engine (prereg/NSR1_prereg.md): impact answers from a state + site catalog, with three composition engines.

  arm 1   RM state + R1's FHRR rule memory (rules learned from n worked traces per question type, dev truth)
  arm 2a  RM state + the same learner with a count table
  arm 2b  RM state + hand-written graph query (nsr1_catalog.answer)
  arm 4   system-of-record state + graph query
  sanity  truth state + graph query;  negctl  RM state with places permuted among things + graph query

python3 nsr/nsr1.py --split dev --reader apps/real/out/reader_dev_verify_v4.jsonl [--n 1]
Writes nsr/out/nsr1_<split>_<arm>.jsonl (one record per question: pred set, and for arm 1 the trace).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_catalog as C  # noqa: E402

ASKED_CLASS = {"freezer_down": lambda x: x.startswith("S-"), "aisle_closed": lambda x: True,
               "room_cleared": lambda x: True, "region_down": lambda x: not x.endswith("primary"),
               "host_down": lambda x: not x.endswith("primary")}


# ------------------------------------------------------------------------------------------------ state
def rm_state(reader_rec):
    """Final Readable Memory state (thing -> place) and, per thing, the log line whose reading last wrote it."""
    snap = reader_rec["snapshots"][-1]
    st = {}
    for x, (v, _c) in snap.items():
        if isinstance(v, list):  # biobank: the product answers with the whole chain; keep one-hop parents
            st[x] = v[0]
            for a, b in zip(v, v[1:]):
                st.setdefault(a, b)
        else:
            st[x] = v
    src = {}
    for i, evs in enumerate(reader_rec["events"]):
        for ev in evs or []:
            if ev[0] == "move":
                src[ev[1]] = i
            elif ev[0] == "swap":
                for x, p in st.items():
                    if p in ev[1:]:
                        src[x] = i
    return st, src


# ------------------------------------------------------------------------------------------------ rule stores
class TableRules:
    def __init__(self):
        self.votes = collections.defaultdict(collections.Counter)

    def write(self, a, b, v):
        self.votes[(a, b)][v] += 1

    def rules(self):
        return {k: c.most_common(1)[0][0] for k, c in self.votes.items()}


class VSARules(TableRules):
    """R1's rule memory: rules voted as in the table, then written once each into an FHRR vector (three codebooks) and
    read back through cleanup. `known` uses the vote dictionary, as in R1."""

    def __init__(self, D=2048, seed=0):
        super().__init__()
        self.D, self.rng = D, np.random.default_rng(seed)
        self.cb = [{}, {}, {}]

    def code(self, i, name):
        if name not in self.cb[i]:
            self.cb[i][name] = np.exp(1j * self.rng.uniform(-np.pi, np.pi, self.D))
        return self.cb[i][name]

    def rules(self):
        voted = super().rules()
        names = sorted({x for k, v in voted.items() for x in (*k, v)})
        for n in names:
            for i in range(3):
                self.code(i, n)
        M = np.zeros(self.D, complex)
        for (a, b), v in voted.items():
            M += self.code(0, a) * self.code(1, b) * self.code(2, v)
        out = {}
        vals = sorted(self.cb[2])
        V = np.stack([self.cb[2][v] for v in vals])
        for (a, b) in voted:
            u = np.conj(self.code(0, a) * self.code(1, b)) * M
            out[(a, b)] = vals[int(np.argmax(np.real(np.conj(V) @ u)))]
        return out


def worked_traces(logs, n, seed=0):
    """n worked traces per question type from dev truth: the hop relations of one answer's path and the asked
    relation, with every intermediate conclusion (left fold), as CLUTRR's proof_state gives them."""
    rng = np.random.default_rng(seed)
    pool = collections.defaultdict(list)
    for log in logs:
        cat = C.catalog(log)
        for q in C.questions(log):
            for x in q["a"]:
                path = paths(x, q["target"], log["truth"], cat)
                if path:
                    pool[q["qtype"]].append(path[0])
    asked = {"freezer_down": "in", "aisle_closed": "in", "room_cleared": "in", "region_down": "in",
             "host_down": "needs_host"}
    out = {}
    for qt, ps in sorted(pool.items()):
        idx = rng.choice(len(ps), size=min(n, len(ps)), replace=False)
        out[qt] = [([r for _, r, _ in ps[i]], asked[qt]) for i in idx]
    return out


def learn(store, traces):
    """Each trace (r1, ..., rk) -> asked gives k - 1 two-step examples: (acc, r_{i+1}) -> conclusion, where acc is the
    previous conclusion (the trace's intermediate steps carry the asked relation)."""
    for qt, ts in traces.items():
        for rels, asked in ts:
            acc = rels[0]
            for r in rels[1:]:
                store.write(acc, r, asked)
                acc = asked
    return store.rules()


# ------------------------------------------------------------------------------------------------ inference
def facts_of(state, cat):
    out = collections.defaultdict(list)
    for x, p in state.items():
        out[x].append((x, "in", p))
    for a, r, b in cat:
        out[a].append((a, r, b))
    return out


def paths(start, target, state, cat, max_hops=4):
    F = facts_of(state, cat)
    res, stack = [], [(start, [])]
    while stack:
        node, path = stack.pop()
        if node == target and path:
            res.append(path)
            continue
        if len(path) >= max_hops:
            continue
        for f in F.get(node, []):
            if f[2] not in {p[0] for p in path} | {start}:
                stack.append((f[2], path + [f]))
    return sorted(res, key=len)


def fold(rels, R):
    """Any bracketing (CYK); returns the unique derivable relation or None."""
    k = len(rels)
    T = {(i, i): {rels[i]} for i in range(k)}
    for span in range(2, k + 1):
        for i in range(k - span + 1):
            j = i + span - 1
            T[(i, j)] = {R[(x, y)] for m in range(i, j) for x in T[(i, m)] for y in T[(m + 1, j)] if (x, y) in R}
    res = T[(0, k - 1)]
    return next(iter(res)) if len(res) == 1 else None


def answer_rules(q, state, cat, R, asked, src=None):
    things = sorted(set(state) | {a for a, r, _ in cat if r == "needs"})
    pred, traces = [], {}
    for x in things:
        if not ASKED_CLASS[q["qtype"]](x):
            continue
        for path in paths(x, q["target"], state, cat):
            if fold([r for _, r, _ in path], R) == asked[q["qtype"]]:
                pred.append(x)
                traces[x] = [[a, r, b, (src or {}).get(a, "?") if r == "in" else "catalog"] for a, r, b in path]
                break
    return sorted(pred), traces


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--reader", required=True)
    ap.add_argument("--logs", default="")  # directory with <scenario>_<split>.jsonl (default apps/real/data)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    logs = C.load_logs(args.split) if not args.logs else [
        json.loads(l) for sc in ("warehouse", "biobank", "home", "ops")
        for l in open(os.path.join(args.logs, f"{sc}_{args.split}.jsonl"))]
    reader = {json.loads(l)["id"]: json.loads(l) for l in open(args.reader)}
    traces = worked_traces(C.load_logs("dev"), args.n)
    asked = {qt: ts[0][1] for qt, ts in traces.items()}
    R_vsa, R_tab = learn(VSARules(), traces), learn(TableRules(), traces)
    print("worked traces:", json.dumps(traces), "\nrules (VSA):", R_vsa, "\nrules (table):", R_tab)
    rng = np.random.default_rng(7)
    outs = collections.defaultdict(list)
    for log in logs:
        cat = C.catalog(log)
        rm, src = rm_state(reader[log["id"]])
        things = sorted(rm)
        perm = rng.permutation(len(things))
        shuffled = {things[i]: rm[things[perm[i]]] for i in range(len(things))}
        for q in C.questions(log):
            base = {"log": log["id"], "scenario": log["scenario"], "qtype": q["qtype"], "target": q["target"]}
            p1, tr = answer_rules(q, rm, cat, R_vsa, asked, src)
            outs["arm1"].append(dict(base, pred=p1, trace=tr))
            outs["arm2a"].append(dict(base, pred=answer_rules(q, rm, cat, R_tab, asked)[0]))
            outs["arm2b"].append(dict(base, pred=C.answer(q["qtype"], q["target"], rm, cat)))
            outs["arm4"].append(dict(base, pred=C.answer(q["qtype"], q["target"], log["sor"], cat)))
            outs["sanity"].append(dict(base, pred=C.answer(q["qtype"], q["target"], log["truth"], cat)))
            outs["negctl"].append(dict(base, pred=C.answer(q["qtype"], q["target"], shuffled, cat)))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for arm, recs in outs.items():
        with open(os.path.join(HERE, "out", f"nsr1_{args.split}{args.tag}_{arm}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print({arm: len(r) for arm, r in outs.items()})


if __name__ == "__main__":
    main()
