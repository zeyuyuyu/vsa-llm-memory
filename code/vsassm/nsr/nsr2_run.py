"""NSR2 run (prereg/NSR2_prereg.md): per-question records for every registered arm on one split.

Reliability model, w_prev and the containment-type table are fitted on dev + retired test + nsr1 (development data).
For each impact question: the deterministic answer (graph query on the read state), the NS engine's MAP answer,
marginals and confidence, baselines 2a-2c, ProbLog marginals (second implementation), VSA-substrate confidences
(D = 256, 1024, 2048) and the negative control (reliabilities permuted among a log's facts).

python3 nsr/nsr2_run.py --eval nsr2 [--train dev,test,nsr1] [--no_problog]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1 as N  # noqa: E402
import nsr1_catalog as C  # noqa: E402
import nsr2 as M  # noqa: E402
import nsr2_types as T  # noqa: E402

M.SPLITS["nsr2"] = ("nsr/data", "nsr/out/held/reader_nsr2_verify_v4.jsonl")


def problog_marginals(q, dist, cat, cands):
    """The same inference in ProbLog: one annotated disjunction per thing over its places, deterministic catalog."""
    from problog import get_evaluatable
    from problog.program import PrologString
    sym = {}

    def s(x):
        if x not in sym:
            sym[x] = f"e{len(sym)}"
        return sym[x]
    lines = []
    for x, d in dist.items():
        alts = [f"{p:.12f}::at({s(x)},{s(l)})" for l, p in d.items() if p > 0]
        lines.append("; ".join(alts) + ".")
    for a, r, b in cat:
        lines.append(f"{'part' if r == 'part_of' else 'needs'}({s(a)},{s(b)}).")
    lines += ["loc(X,Y) :- at(X,Y).", "loc(X,Y) :- part(X,Y).",
              "reach(X,T) :- loc(X,T).", "reach(X,T) :- loc(X,Y), reach(Y,T).",
              "aff(S,H) :- needs(S,D), at(D,H)."]
    if not any(r == "part_of" for _, r, _ in cat):
        lines.append("part(dummy_a,dummy_b).")
    if not any(r == "needs" for _, r, _ in cat):
        lines.append("needs(dummy_c,dummy_d).")
    t = s(q["target"])
    for x in cands:
        lines.append(f"query(aff({s(x)},{t}))." if q["qtype"] == "host_down" else f"query(reach({s(x)},{t})).")
    res = get_evaluatable().create_from(PrologString("\n".join(lines))).evaluate()
    inv = {v: k for k, v in sym.items()}
    return {inv[str(k.args[0])]: float(v) for k, v in res.items()}


def vsa_dist(dist, D, seed):
    """Store the soft state as one FHRR superposition and read the probabilities back (clip at 0, renormalise)."""
    rng = np.random.default_rng(seed)
    things = sorted(dist)
    places = sorted({l for d in dist.values() for l in d})
    cx = {x: np.exp(1j * rng.uniform(-np.pi, np.pi, D)) for x in things}
    cl = {l: np.exp(1j * rng.uniform(-np.pi, np.pi, D)) for l in places}
    Mv = np.zeros(D, complex)
    for x, d in dist.items():
        for l, p in d.items():
            Mv += p * cx[x] * cl[l]
    out = {}
    for x, d in dist.items():
        u = np.conj(cx[x]) * Mv
        est = {l: max(0.0, float(np.real(np.vdot(cl[l], u))) / D) for l in d}
        z = sum(est.values()) or 1.0
        out[x] = {l: v / z for l, v in est.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dev,test,nsr1")
    ap.add_argument("--eval", default="nsr2")
    ap.add_argument("--no_problog", action="store_true")
    a = ap.parse_args()
    train_rows, table = [], set()
    for sp in a.train.split(","):
        logs, rd = M.load(sp)
        table |= T.learn_table(logs.values())
        for lid, log in logs.items():
            train_rows += M.facts(log, rd[lid])
    model = M.fit_reliability(train_rows)
    wrong = [r for r in train_rows if not r["ok"]]
    w_prev = sum(r["truth_is_prev"] for r in wrong) / max(len(wrong), 1)
    mean_rel = float(np.mean([r["ok"] for r in train_rows]))
    logs, rd = M.load(a.eval)
    rng = np.random.default_rng(0)
    out_facts, out_q = [], []
    for lid, log in sorted(logs.items()):
        rows = M.facts(log, rd[lid])
        pr = model.predict_proba(np.array([r["feat"] for r in rows]))[:, 1] if rows else []
        for r, p in zip(rows, pr):
            out_facts.append({"log": lid, "x": r["x"], "p": r["p"], "rel": float(p)})
        rel = {r["x"]: float(p) for r, p in zip(rows, pr)}
        human = {r["x"]: r["feat"][1] for r in rows}
        cands = M.candidate_places(log, table)
        dist = M.place_dist(rows, model, cands, w_prev)
        perm_model = _Permuted(model, rng, len(rows))
        dist_neg = M.place_dist(rows, perm_model, cands, w_prev)
        dist_const = M.place_dist(rows, _Constant(mean_rel), cands, w_prev)
        lseed = sum(map(ord, lid))  # deterministic across runs (hash() is salted per process)
        vsa = {(D, k): vsa_dist(dist, D, seed=lseed * 7919 + D * 13 + k) for D in (256, 1024, 2048) for k in range(3)}
        cat = C.catalog(log)
        rm, _ = N.rm_state(rd[lid])
        needs = {x: d for x, r_, d in cat if r_ == "needs"}
        for q in C.questions(log):
            det = C.answer(q["qtype"], q["target"], rm, cat)
            pred, conf, marg = M.answer(q, dist, cat)
            dep = [needs.get(x, x) for x in det]
            rec = {"log": lid, "scenario": log["scenario"], "qtype": q["qtype"], "target": q["target"],
                   "pred_det": det, "pred_ns": pred, "marg": marg,
                   "conf": {"ns": conf,
                            "min_rel": min([rel.get(x, 1.0) for x in dep], default=1.0),
                            "human": -float(sum(human.get(x, 0.0) for x in dep)),
                            "size": -float(len(det)),
                            "negctl": M.answer(q, dist_neg, cat)[1],
                            "ns_const": M.answer(q, dist_const, cat)[1],
                            "log_rel": float(np.mean(pr)) if len(rows) else 1.0}}
            for (D, k), dv in vsa.items():
                rec["conf"][f"vsa{D}_s{k}"] = M.answer(q, dv, cat)[1]
            if not a.no_problog:
                rec["marg_problog"] = problog_marginals(q, dist, cat, sorted(marg))
            out_q.append(rec)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", f"nsr2_{a.eval}_records.jsonl"), "w") as f:
        for r in out_q:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(HERE, "out", f"nsr2_{a.eval}_facts.jsonl"), "w") as f:
        for r in out_facts:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out_q)} questions, {len(out_facts)} facts; w_prev {w_prev:.3f}; trained on {len(train_rows)} facts")


class _Constant:
    """Structure-only ablation: every fact gets the training-set mean reliability."""

    def __init__(self, q):
        self.q = q

    def predict_proba(self, X):
        return np.tile([1 - self.q, self.q], (len(X), 1))


class _Permuted:
    """Negative control: the model's reliabilities, permuted among the facts of one log."""

    def __init__(self, model, rng, n):
        self.model, self.perm = model, rng.permutation(n) if n else np.array([], int)

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        return p[self.perm]


if __name__ == "__main__":
    main()
