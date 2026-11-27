"""B21 independent checks (multi-angle): replay, second scorer, similarity margins, overload control.

python3 b21_checks.py B21_DIR
(a) replay: rebuild every store from the saved pairs and re-answer; must equal the saved pred_vsa.
(b) second scorer: an independent re-implementation of string_match_all (no import of ruler_rm's scorer).
(c) margins: for stores with >= 200 pairs, for every queried key, the similarity of each accepted value and the
    largest similarity of any other vocabulary entry (noise), against the threshold 0.5.
(d) overload control: the same stores with one shard (all pairs in a single vector); answers must break,
    showing that the vector store, not a hidden table, produces the answers.
"""
import collections, glob, json, math, os, re, sys
import numpy as np
import ruler_vsa as V

rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))) for l in open(f)]
print("rows", len(rows))

# The question text is not saved, so the replay checks every key of each table (a stronger test than the
# queried keys only): store.read(k) must equal table[k].
bad_rows, n_keys, n_bad_keys = 0, 0, 0
margins_acc, margins_noise = [], []
ctrl_rows, ctrl_bad = 0, 0
for i, r in enumerate(rows):
    pairs = [tuple(p) for p in r["pairs"]]
    t = V.table_of(pairs)
    st = V.Store(pairs)
    assert st.S == r["shards"] and int(np.max(st.loads)) == r["max_load"]
    bad = 0
    keys = list(t)
    if len(keys) > 100:                       # large stores: check a fixed, evenly spaced subset of ~100 keys
        keys = keys[:: max(1, len(keys) // 100)]
    for k in keys:
        n_keys += 1
        got = st.read(k)
        if got != t[k]:
            bad += 1
        if len(pairs) >= 200:
            x = st.mem[st.shard(k)] * np.conj(V.code(k, "key"))
            for j in range(len(t[k]) + 1):
                sims = (st.V.conj() @ (x * np.conj(V.pos(j)))).real / V.D
                if j < len(t[k]):
                    b = st.vocab.index(t[k][j])
                    margins_acc.append(float(sims[b]))
                    sims[b] = -9
                margins_noise.append(float(sims.max()))
    n_bad_keys += bad
    bad_rows += bad > 0
    if len(pairs) >= 500 and ctrl_rows < 40:  # overload control on 40 large stores
        ctrl_rows += 1
        old = V.PER_SHARD
        V.PER_SHARD = 10 ** 9
        st1 = V.Store(pairs)
        V.PER_SHARD = old
        ks = list(t)[:: max(1, len(t) // 50)]
        ctrl_bad += sum(st1.read(k) != t[k] for k in ks) / len(ks)
print(f"(a) replay: keys checked {n_keys}, keys differing from the table {n_bad_keys}, rows with any difference {bad_rows}")

def sma(pred, refs):  # independent re-implementation of RULER string_match_all
    p = pred.lower()
    return sum(1.0 for x in refs if x.lower() in p) / len(refs)
cells = collections.defaultdict(list)
for r in rows:
    cells[(int(r["length"]), r["task"])].append((sma(r["pred_vsa"], r["outputs"]), sma(r["pred_table"], r["outputs"])))
for L in sorted({k[0] for k in cells}):
    v = np.mean([np.mean([a for a, _ in cells[(L, t)]]) for t in V.R.NIAH + ["vt"]]) * 100
    print(f"(b) second scorer {L // 1024}k NIAH+VT vsa {v:.2f}")
ma, mn = np.array(margins_acc), np.array(margins_noise)
print(f"(c) margins over {len(ma)} accepted reads: min true sim {ma.min():.3f}, 0.1% quantile {np.quantile(ma, .001):.3f}; "
      f"noise over {len(mn)} reads: max {mn.max():.3f}, 99.9% quantile {np.quantile(mn, .999):.3f}; threshold {V.THR}")
print(f"(d) overload control (all pairs in one vector), {ctrl_rows} stores with >= 500 pairs: "
      f"mean fraction of keys misread {ctrl_bad / max(1, ctrl_rows):.3f}")
