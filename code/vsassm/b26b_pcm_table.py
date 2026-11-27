"""B26b (prereg/B26_vector_form_prereg.md, part b): the RULER VSA store against a table holding the same facts, on the
same simulated phase-change-memory device (the B25 model: Joshi et al. 2020, programming noise, drift, 1/f read noise,
8-bit ADC, every cell a differential pair read as a sign).

python3 b26b_pcm_table.py B21_DIR OUT.json [--workers 4]
python3 b26b_pcm_table.py --selftest

Arms (the 50 B22/B25 stores, the same 4,776 probed keys, 16 pairs per vector):
  VSA    the bipolar store (B25's B1) with its store crossbar on the device and a digital clean-up (B25 stage S).
  T1     a table at its native cell count: one row per (key, j) holding the index of the j-th value in the store's value
         vocabulary, ceil(log2 N_values) bits, one differential pair per bit.
  Trep   the same table with every bit repeated r times, r = floor(VSA cells / native table cells), read by majority
         (ties broken at random): about the cell count of the VSA store.
The key -> row directory and each key's number of values are digital in both tables (this favours the tables: the VSA
store keeps neither). A key counts as exact if all its values come back, in order (B22/B25 metric).
"""
import collections
import json
import math
import sys

import numpy as np

import b22_robust as B
import b25_pcm as P
import ruler_vsa as V

LOAD = 16
KS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)               # device-noise multiplier (B25's descriptive extras); 1 = measured
TIMES = P.TIMES


def cells_read(bits, seed, t, k):
    """Read a +-1 array of differential-pair cells at time t: B25's store-crossbar model for a B1 row (programming
    noise, per-device drift exponent, read noise, 8-bit ADC, sign, ADC zero -> random tie). bits: any 2-D +-1 array."""
    r = np.random.default_rng([seed, 3])
    gp = np.where(bits > 0, P.G_ON, 0.0).astype(np.float32)
    gm = np.where(bits < 0, P.G_ON, 0.0).astype(np.float32)
    nu_p, g1p = P.program(gp, gp.shape, r, k)
    nu_m, g1m = P.program(gm, gm.shape, r, k)
    rr = np.random.default_rng([seed, 4, int(t)])
    Gp, _, _ = P.at_time(gp, nu_p, g1p, t, rr, k)
    Gm, _, _ = P.at_time(gm, nu_m, g1m, t, rr, k)
    q = P.adc(Gp - Gm, P.G_ON, P.ADC_BITS)
    sgn = np.sign(q)
    ties = sgn == 0
    sgn[ties] = np.random.default_rng([seed, 5, int(t)]).choice([-1.0, 1.0], size=int(ties.sum()))
    return sgn.astype(np.float32)


def table_layout(pairs, st, cd):
    """Rows of the probed keys' values (index bits, MSB first, +1 = 1), the bit width and the repetition factor."""
    table = V.table_of(pairs)
    nv = len(cd["vocab"])
    b = max(1, math.ceil(math.log2(nv))) if nv > 1 else 1
    vi = {v: i for i, v in enumerate(cd["vocab"])}
    rows, owner = [], []
    for a, k in enumerate(st["keys"]):
        for v in table[k]:
            idx = vi[v]
            rows.append([1.0 if (idx >> (b - 1 - i)) & 1 else -1.0 for i in range(b)])
            owner.append(a)
    native_cells = len(pairs) * b
    vsa_cells = st["S"] * P.D
    r = max(1, vsa_cells // native_cells)
    return np.array(rows, np.float32), owner, b, r, native_cells, vsa_cells


def decode(signs, b, r, seed, t):
    """signs: (rows, b * r), copy c of bit i at column i * r + c. Majority vote per bit, ties at random."""
    s = signs.reshape(len(signs), b, r).sum(2)
    ties = s == 0
    s[ties] = np.random.default_rng([seed, 7, int(t)]).choice([-1.0, 1.0], size=int(ties.sum()))
    bits = (s > 0).astype(np.int64)
    return (bits * (1 << np.arange(b - 1, -1, -1))).sum(1)


def run_store(job):
    L, task, idx, pairs = job
    seed = V._seed("b26b:%d:%s:%d" % (L, task, idx)) % (2 ** 62)
    cd = P.codes(pairs)
    st = P.build(pairs, LOAD, cd)
    rows, owner, b, r, ncells, vcells = table_layout(pairs, st, cd)
    want_idx = [w for w in st["want"]]
    res = {}
    dig = lambda Y, Vc=st["Vc"]: Y @ Vc.T / P.D
    for k in KS:
        for tl, t in TIMES:
            # VSA (B25 B1-S): store crossbar on the device, digital clean-up
            out, dg = P.store_array("B1", st["mem"], st["b1"], seed, t, k)
            got = P.read_keys(out["gdc"], st["rms0"], st["sh"], st["kidx"], st["Kc"], st["Pc"], dig, st["maxj"])
            res["VSA | x%g | %s" % (k, tl)] = [sum(g == w for g, w in zip(got, want_idx)), len(got)]
            res["bit_err | x%g | %s" % (k, tl)] = dg["bit_err"]
            # tables
            for name, rep in (("T1", 1), ("Trep", r)):
                cells = np.repeat(rows, rep, axis=1)          # copy c of bit i at column i * rep + c
                sg = cells_read(cells, seed + (1 if rep == 1 else 2), t, k)
                dec = decode(sg, b, rep, seed + (1 if rep == 1 else 2), t)
                ok = collections.defaultdict(list)
                for a, d in zip(owner, dec):
                    ok[a].append(int(d))
                exact = sum(ok[a] == w for a, w in enumerate(want_idx))
                res["%s | x%g | %s" % (name, k, tl)] = [exact, len(want_idx)]
    return {"store": "%d/%s/%d" % (L, task, idx), "n_pairs": len(pairs), "n_values": len(cd["vocab"]), "bits": b,
            "rep": r, "table_cells": ncells, "vsa_cells": vcells, "S": st["S"], "res": res}


def selftest():
    rng = np.random.default_rng(1)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(1500):
        k = "key%d" % rng.integers(0, 1200)
        pairs.append((k, "val%d" % i, seen[k]))
        seen[k] += 1
    global KS, TIMES
    KS, TIMES = (1.0, 8.0, 32.0), [P.TIMES[2]]
    out = run_store((0, "synthetic", 0, pairs))
    print({k: v for k, v in out.items() if k != "res"})
    for kk, v in out["res"].items():
        print("%-28s %s" % (kk, v))
    # the noiseless decode is exact
    st = P.build(pairs, LOAD, P.codes(pairs))
    rows, owner, b, r, _, _ = table_layout(pairs, st, P.codes(pairs))
    dec = decode(np.repeat(rows, 3, axis=1), b, 3, 0, 1)
    vi = {v: i for i, v in enumerate(P.codes(pairs)["vocab"])}
    assert [int(d) for d in dec] == [vi[v] for k in st["keys"] for v in V.table_of(pairs)[k]]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit()
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    stores = B.load_stores(src)
    from multiprocessing import Pool
    jobs = sorted(stores, key=lambda j: -len(j[3]))
    per = []
    with Pool(workers) as pool:
        for rr in pool.imap_unordered(run_store, jobs):
            per.append(rr)
            print("done %-28s %d/50  rep %d  VSA x8 30 d %s  Trep %s  T1 %s" % (
                rr["store"], len(per), rr["rep"], rr["res"]["VSA | x8 | 30 d"], rr["res"]["Trep | x8 | 30 d"],
                rr["res"]["T1 | x8 | 30 d"]), flush=True)
    tot = {}
    for name in per[0]["res"]:
        if name.startswith("bit_err"):
            tot[name] = float(np.mean([p["res"][name] for p in per]))
        else:
            e = sum(p["res"][name][0] for p in per)
            n = sum(p["res"][name][1] for p in per)
            tot[name] = [e, n, e / n, sum(p["res"][name][0] == p["res"][name][1] for p in per)]
    json.dump({"totals": tot, "per_store": per}, open(out, "w"), indent=1)
    for name, v in tot.items():
        print(name, v)
