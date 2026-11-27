"""B58 compute note 5: the cut test (i.i.d. part (a)) and the cut E-LDPC re-score (16 Fig. 5b conditions), resuming
from the checkpoints of the stopped runs (evaluations already there are not repeated).

python3 b58_cut.py test W      -> results/b58_test_iid2.json
python3 b58_cut.py rescore W   -> results/b58_ldpc_rescore2.json
"""
import collections
import json
import sys

import b38_eq as E
import b58_store as P

GRID = E.GRID
gi = P.gidx
ONLY = {"e05": "P-BCH", "k8": "P-BCH", "k16": "P-LDPC", "k32": "P-LDPC"}
NAT_REF = {"lines-nat-k4": 128.0, "lines-nat-e05": 512.0, "lines-nat-k8": 4096.0, "lines-nat-k16": None,
           "clust-nat-k4": 128.0, "clust-nat-e05": 512.0, "clust-nat-k8": 1024.0, "clust-nat-k16": 2048.0}
IL_REF = {"lines-il-k4": 27.0, "lines-il-e05": 45.0, "lines-il-k8": 76.0, "lines-il-k16": 215.0,
          "clust-il-k4": 27.0, "clust-il-e05": 38.0, "clust-il-k8": 64.0, "clust-il-k16": 304.0}


def test_need():
    ch = json.load(open("results/b58_choice.json"))
    need = collections.defaultdict(set)
    for c in P.IIDC:
        for var in P.VARS:
            if c in ONLY and ONLY[c] != var:
                continue
            ns = ch["dev_nstar"][var][c]
            g = gi(ns) if ns else len(GRID) - 1
            for k in range(max(0, g - 2), min(len(GRID), g + 3)):
                b, R = P.design_at(ch, var, c, k)
                need[(var, b, R, k)].add(c)
    return need


def rescore_need():
    ch = json.load(open("results/b38_choice.json"))
    stores = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    need = collections.defaultdict(set)
    for c in list(NAT_REF) + list(IL_REF):
        m = E.parse_cond(c)["match"]
        if c in NAT_REF:
            ref = NAT_REF[c]
            gs = [len(GRID) - 1] if ref is None else list(range(gi(ref) - 4, gi(ref) + 1))
        else:
            g = gi(IL_REF[c])
            gs = [g - 1, g, g + 1]
        for k in gs:
            d = ch["designs"]["E-LDPC"][m].get(str(k))
            if d is None:
                near = min((int(x) for x in ch["designs"]["E-LDPC"][m]), key=lambda x: (abs(x - k), -x))
                d = ch["designs"]["E-LDPC"][m][str(near)]
            for sid in stores:
                need[(sid, k, json.dumps(json.loads(d)))].add(c)
    return need


if __name__ == "__main__":
    W = int(sys.argv[2])
    stores = E.load_stores("b34_out/test_stores.json.gz")
    if sys.argv[1] == "test":
        done = P.done_set(["results/b58_test_iid1.json.partial"])
        jobs = P.make_jobs(stores, test_need(), P.TEST_CAP, per_job=3, done=done)
        print(len(jobs), "jobs", flush=True)
        P.pool_run(jobs, W, "results/b58_test_iid2.json")
    else:
        have = set()
        for r in json.load(open("results/b58_ldpc_rescore1.json.partial")):
            for x in r["res"]:
                if "cond" in x:
                    have.add((r["sid"], gi(x["B"]), x["cond"]))
        need = rescore_need()
        keep = collections.defaultdict(set)
        for (sid, k, dj), cs in need.items():
            cs2 = {c for c in cs if (sid, k, c) not in have}
            if cs2:
                keep[(sid, k, dj)] = cs2
        json.dump(dict(need=[[list(k), sorted(v)] for k, v in keep.items()]),
                  open("results/b58_plan_rescore_cut.json", "w"))
        print(len(keep), "entries", sum(len(v) for v in keep.values()), "evaluations", flush=True)
        P.rescore("b34_out/test_stores.json.gz", "results/b58_plan_rescore_cut.json", "results/b58_ldpc_rescore2.json",
                  W)
