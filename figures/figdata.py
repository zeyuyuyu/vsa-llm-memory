"""Data layer of make_figures.py: every plotted number is read from a file under figdata/.

    J(rel)                 a banked artifact (JSON), e.g. J("v13_logs/v13_fixed.json")
    accs(rows, **eq)       the `acc` of the records of a list-of-runs artifact that match the filters, in file order
    rows(rows, **eq)       the matching records themselves
    V(panel, label)        the value that figdata/INDEX.json records for that label (the index of the old figure set)
    same(panel, label, v)  assert that a value read from an artifact equals the INDEX entry (cross-check, no effect on plots)

Values that have no JSON artifact are in small files, each value with its `source` string:
    figdata/oplogs.json, figdata/benchmarks/*.json, figdata/constants.json, figdata/worked_example.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "figdata")
IDX = json.load(open(os.path.join(D, "INDEX.json")))
_cache = {}


def J(rel):
    if rel not in _cache:
        _cache[rel] = json.load(open(os.path.join(D, rel)))
    return _cache[rel]


def rows(lst, **eq):
    return [r for r in lst if all(r.get(k) == v for k, v in eq.items())]


def accs(lst, **eq):
    return [r["acc"] for r in rows(lst, **eq)]


def V(panel, label):
    hit = [e for e in IDX[panel] if e["label"] == label]
    if len(hit) != 1:
        raise KeyError("%s / %r -> %d entries" % (panel, label, len(hit)))
    return hit[0]["value"]


def same(panel, label, value, tol=1e-12):
    """The artifact read by the script and the INDEX entry of the old figure set must agree."""
    import numpy as np
    a, b = np.asarray(V(panel, label), dtype=float), np.asarray(value, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=0):
        raise AssertionError("INDEX mismatch at %s / %r: index %s, artifact %s" % (panel, label, a.tolist(), b.tolist()))
    return value
