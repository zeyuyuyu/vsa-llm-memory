"""V-164 (prereg in ledger.md, committed before this file): how often does a random codebook read all 117 CLUTRR
composition rules back from ONE Fourier-HRR vector, and does noise counting predict it?

The memory is R1's (vsassm r1_oracle.py, RuleMemory): unit-modulus complex codes with independent uniform phases, a
separate codebook for the first relation, the second relation and the value; a rule (r1, r2) -> r3 is written as
first[r1] * second[r2] * value[r3] into one vector M, and read as the value code nearest to conj(first[r1] * second[r2]) * M.
The 117 rules are v164_logs/clutrr_rules117.json, extracted with r1_oracle.py's own rule learner (vsassm f4f7716,
r1_data/train.csv sha256 38963d01a20789ac...). No language model, no training, numpy only.
"""
import json, math, sys
import numpy as np

RULES = [tuple(r) for r in json.load(open("v164_logs/clutrr_rules117.json"))]
A = sorted({r[0] for r in RULES}); B = sorted({r[1] for r in RULES}); V = sorted({r[2] for r in RULES})
ia = np.array([A.index(r[0]) for r in RULES]); ib = np.array([B.index(r[1]) for r in RULES])
iv = np.array([V.index(r[2]) for r in RULES])
N_CODEBOOKS = 1000
out = {"n_rules": len(RULES), "n_first": len(A), "n_second": len(B), "n_value": len(V), "n_codebooks": N_CODEBOOKS,
       "cells": {}}
assert len(RULES) == 117 and len(set((r[0], r[1]) for r in RULES)) == 117


def predicted(D):
    """Noise counting: the target scores 1 + N(0, K/2D), each of the n_value - 1 competitors N(0, K/2D), K = 116."""
    K = len(RULES) - 1
    p = 0.5 * math.erfc(math.sqrt(D / K) / math.sqrt(2))
    return 1.0 - math.exp(-len(RULES) * (len(V) - 1) * p)


for D in (1024, 2048, 4096):
    misread = []
    for s in range(N_CODEBOOKS):
        rng = np.random.default_rng([164, D, s])
        first = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(A), D)))
        second = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(B), D)))
        value = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(V), D)))
        keys = first[ia] * second[ib]                                   # 117 x D
        M = (keys * value[iv]).sum(0)
        sims = np.real((np.conj(keys) * M[None, :]) @ np.conj(value).T) / D     # 117 x n_value
        misread.append(int((sims.argmax(1) != iv).sum()))
    misread = np.array(misread)
    cell = {"codebooks_with_a_misread": int((misread > 0).sum()), "share_with_a_misread": float((misread > 0).mean()),
            "mean_misread_rules": float(misread.mean()), "max_misread_rules": int(misread.max()),
            "predicted_share_with_a_misread": predicted(D)}
    out["cells"][str(D)] = cell
    print("D=%d: %d of %d codebooks misread at least one rule (%.3f; noise counting predicts %.3f); mean %.3f rules, "
          "max %d" % (D, cell["codebooks_with_a_misread"], N_CODEBOOKS, cell["share_with_a_misread"],
                      cell["predicted_share_with_a_misread"], cell["mean_misread_rules"], cell["max_misread_rules"]),
          flush=True)

c = out["cells"]
b1 = 0.01 <= c["2048"]["share_with_a_misread"] <= 0.05
b2 = c["1024"]["share_with_a_misread"] >= 0.85
b3 = c["4096"]["codebooks_with_a_misread"] == 0
out.update(B1=b1, B2=b2, B3=b3)
print("V-164 VERDICT: B1 (D=2048 share in [0.01, 0.05]) %s | B2 (D=1024 share >= 0.85) %s | B3 (D=4096: none) %s"
      % tuple("HOLDS" if x else "FAILS" for x in (b1, b2, b3)))
json.dump(out, open("v164_logs/v164.json", "w"), indent=1)
