#!/usr/bin/env python3
"""Recompute figdata/ruler_loads.json (Fig. 8a: pairs per vector over every vector of the 5,400 B21 RULER stores)
from the saved B21 per-sample outputs, with the shard hash of ruler_vsa.Store (sha256("shard:" + key), first 8 bytes,
little-endian, mod S; S = ceil(pairs / 16)), and compare with the banked file. Added for the deposit: the session
that banked ruler_loads.json did not save its script.

    python3 figures/tools/recompute_ruler_loads.py [B21_DIR]     # default: results/vsassm/b21_dl/b21_test
"""
import collections
import glob
import gzip
import hashlib
import json
import math
import os
import sys

FIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.dirname(FIG)
B21 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "results", "vsassm", "b21_dl", "b21_test")


def seed(s):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")


hist, n_samples, n_vec, several, bad_shards = collections.Counter(), 0, 0, 0, 0
for f in sorted(glob.glob(os.path.join(B21, "*.jsonl")) + glob.glob(os.path.join(B21, "*.jsonl.gz"))):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            pairs = r["pairs"]
            S = max(1, math.ceil(len(pairs) / 16))
            if S != r["shards"]:
                bad_shards += 1
            loads = collections.Counter(seed("shard:" + k) % S for k, _, _ in pairs)
            for s in range(S):
                hist[loads.get(s, 0)] += 1
            n_samples += 1
            n_vec += S
            several += S > 1
banked = json.load(open(os.path.join(FIG, "figdata", "ruler_loads.json")))
got = {str(k): v for k, v in sorted(hist.items())}
same_hist = {k: v for k, v in banked["hist"].items()} == got
print("samples %d (banked %d), vectors %d (banked %d), with several vectors %d (banked %d), shard counts differing %d"
      % (n_samples, banked["n_samples"], n_vec, banked["n_vectors"], several, banked["samples_with_several_vectors"], bad_shards))
print("histogram identical to the banked one: %s; max load %d; mean load %.2f"
      % (same_hist, max(hist), sum(k * v for k, v in hist.items()) / n_vec))
sys.exit(0 if same_hist and n_vec == banked["n_vectors"] and not bad_shards else 1)
