"""B21 verdict (prereg/B21_ruler_vsa_prereg.md, amendments 1-2): P1-P3 + table-arm reproducibility vs B19.

python3 b21_verdict.py B21_DIR B19_RM_DIR
Samples are de-duplicated on (task, length, index, outputs), as B19's scorer does (RULER repeats some indices);
the raw (no de-dup) numbers are printed too.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

import ruler_rm as R

NV = R.NIAH + ["vt"]
LEN = [4096, 8192, 16384, 32768, 65536, 131072]


def load(d, pred_keys):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))) for l in open(f)]
    raw, ded, seen = collections.defaultdict(list), collections.defaultdict(list), set()
    for r in rows:
        cell = (r["task"], int(r["length"]))
        raw[cell].append(r)
        key = (r["task"], int(r["length"]), r["index"], json.dumps(r["outputs"]))
        if key not in seen:
            seen.add(key)
            ded[cell].append(r)
    return raw, ded


def acc(rs, key):
    fn = lambda r: (R.string_match_part if r["task"].startswith("qa") else R.string_match_all)(r[key], r["outputs"])
    return 100 * float(np.mean([fn(r) for r in rs]))


def main(b21, b19):
    raw, ded = load(b21, None)
    _, b19d = load(b19, None)
    out = {"per_cell": {}, "per_length": {}}
    for tag, cells in (("dedup", ded), ("raw", raw)):
        print(f"\n== {tag} ==")
        print("len     " + " ".join(t[:12].rjust(12) for t in NV) + "   mean_tab mean_vsa  agree   B19")
        for L in LEN:
            tab = [acc(cells[(t, L)], "pred_table") for t in NV]
            vsa = [acc(cells[(t, L)], "pred_vsa") for t in NV]
            n_ag = sum(r["pred_table"] == r["pred_vsa"] for t in NV for r in cells[(t, L)])
            n = sum(len(cells[(t, L)]) for t in NV)
            b = [acc(b19d[(t, L)], "pred") for t in NV] if tag == "dedup" else [float("nan")]
            print(f"{L // 1024:>4}k vsa " + " ".join(f"{v:12.1f}" for v in vsa) +
                  f"   {np.mean(tab):8.2f} {np.mean(vsa):8.2f} {n_ag / n:7.4f} {np.mean(b):6.2f}")
            if tag == "dedup":
                out["per_length"][L] = {"table": round(float(np.mean(tab)), 2), "vsa": round(float(np.mean(vsa)), 2),
                                        "agree": round(n_ag / n, 4), "n": n, "b19": round(float(np.mean(b)), 2)}
                for t, a, v, bb in zip(NV, tab, vsa, b):
                    out["per_cell"][f"{L}/{t}"] = {"table": round(a, 2), "vsa": round(v, 2), "b19": round(bb, 2),
                                                   "n": len(cells[(t, L)])}
    # disagreements, listed
    dis = [(t, L, r["index"], r["pred_table"], r["pred_vsa"], r["outputs"]) for (t, L), rs in ded.items() for r in rs
           if r["pred_table"] != r["pred_vsa"]]
    print(f"\ndisagreements (dedup): {len(dis)}")
    for d in dis[:40]:
        print("  ", d)
    out["disagreements"] = dis
    # P3: 13-task average at 128k, VSA for the 9 memory tasks, B19 for cwe/fwe/qa_1/qa_2
    L = 131072
    nine = [acc(ded[(t, L)], "pred_vsa") for t in NV]
    four = [acc(b19d[(t, L)], "pred") for t in ("cwe", "fwe", "qa_1", "qa_2")]
    nine_tab = [acc(ded[(t, L)], "pred_table") for t in NV]
    nine_b19 = [acc(b19d[(t, L)], "pred") for t in NV]
    p3 = float(np.mean(nine + four))
    out["p3_avg13_128k_vsa"] = round(p3, 2)
    out["avg13_128k_table_this_run"] = round(float(np.mean(nine_tab + four)), 2)
    out["avg13_128k_b19"] = round(float(np.mean(nine_b19 + four)), 2)
    # store statistics
    st = [(r["n_pairs"], r["shards"], r["max_load"]) for rs in ded.values() for r in rs]
    out["store"] = {"pairs_max": max(s[0] for s in st), "shards_max": max(s[1] for s in st),
                    "max_load_max": max(s[2] for s in st),
                    "pairs_median": float(np.median([s[0] for s in st]))}
    P1 = all(out["per_length"][L]["vsa"] >= 99.0 for L in LEN)
    P2 = all(out["per_length"][L]["agree"] >= 0.99 for L in LEN)
    P3 = p3 >= 94.0
    out["bars"] = {"P1": P1, "P2": P2, "P3": P3}
    print("\nP1 (NIAH+VT vsa >= 99.0 every length):", P1, [out["per_length"][L]["vsa"] for L in LEN])
    print("P2 (agree >= 0.99 every length):", P2, [out["per_length"][L]["agree"] for L in LEN])
    print("P3 (13-task avg at 128k >= 94.0):", P3, out["p3_avg13_128k_vsa"],
          "| table this run", out["avg13_128k_table_this_run"], "| B19", out["avg13_128k_b19"])
    print("store:", out["store"])
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "b21_scores.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
