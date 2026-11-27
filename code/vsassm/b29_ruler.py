"""B29 RULER part (prereg/B29_same_tooling_prereg.md): the same tooling (B19/B21 extraction prompt, chunking, grounding,
fact table = plain dict, sharded Fourier HRR store, read rules), with the reader model varied.

  python3 b29_ruler.py prompts rel  ROOT OUT.jsonl [--lengths ...] [--n 100]
        unique reader prompts of the chunks that can change the dict's answer (see below), every length and task
  python3 b29_ruler.py prompts full ROOT OUT.jsonl --lengths 131072 [--tasks ...] [--n 100]
        unique reader prompts of every chunk (what B21 read)
  python3 b29_ruler.py assemble rel|full ROOT READS.jsonl[,READS2...] OUT.jsonl [--lengths ...] [--tasks ...]
        per-sample predictions of the dict (and, where every chunk was read, of the store) from the reader's outputs
  python3 b29_ruler.py score OUT.jsonl
  python3 b29_ruler.py gate8b REL_8B.jsonl B21_DIR         reproduction of B21 by the 8B replay (per chunk and per sample)
  python3 b29_ruler.py selftest ROOT

Why reading only some chunks gives the dict's exact answer (NIAH). ruler_rm.answer_niah returns, for every queried key
k, table[k.lower()]; facts_of writes table[k'.lower()] only for a pair whose key k' passes the grounding check
k'.lower() in chunk.lower(). So table[k] can only be filled from chunks that contain k (case-insensitive), in reading
order. Reading only those chunks (every other chunk's output replaced by "[]") therefore gives the same table entries
for the queried keys, hence the same answer, as reading every chunk, provided the reader's output on a chunk does not
depend on the other chunks in its batch (greedy decoding; checked by gate8b against B21's banked per-chunk output).
VT's answer scans every key in the table, so VT reads every chunk (and both the dict and the store are exact there).
The store's answer for NIAH depends on every pair (shard count, cross-talk, clean-up vocabulary), so the store arm
needs every chunk read: the 'full' prompts.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ruler_rm as R  # noqa: E402
import ruler_vsa as V  # noqa: E402

LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

_CODE, _code0 = {}, V.code


def _cached_code(s, slot):
    """ruler_vsa.code with a cache (identical outputs; the deterministic seeded code of each string is computed once).
    Position codes are kept; key and value codes are dropped when the cache grows past 50,000 entries."""
    c = _CODE.get((s, slot))
    if c is None:
        if len(_CODE) > 50000:
            for k in [k for k in _CODE if k[1] != "pos"]:
                del _CODE[k]
        c = _CODE[(s, slot)] = _code0(s, slot)
    return c


V.code = _cached_code
TASKS = R.NIAH + ["vt"]


def pid(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def queried_keys(q):
    """The keys answer_niah looks up (same regex and split as ruler_rm.answer_niah), lower case and stripped."""
    m = re.search(r"for (.+?) mentioned", q)
    keys = re.split(r",\s*and\s+|,\s*|\s+and\s+", m.group(1)) if m else []
    return [k.strip().lower() for k in keys]


def needed(task, cs, q, arm):
    """Indices of the chunks to read."""
    if arm == "full" or task == "vt":
        return list(range(len(cs)))
    keys = queried_keys(q)
    if not keys:
        return []                      # no key parsed: the answer is "" whatever is read
    if any(k == "" for k in keys):
        return list(range(len(cs)))    # an empty key matches every chunk (never seen; kept exact)
    low = [c.lower() for c in cs]
    return [i for i, c in enumerate(low) if any(k in c for k in keys)]


def samples(root, lengths, tasks, n):
    for L in lengths:
        for t in tasks:
            for i, s in enumerate(R.load(root, L, t, n)):
                yield L, t, i, s


def grounded(cs, raw):
    """ruler_vsa.grounded_pairs on given per-chunk outputs (same parse, grounding and order)."""
    seen, pairs = collections.defaultdict(list), []
    for c, txt in zip(cs, raw):
        for k, v in R.PAIR.findall(txt):
            k, v = re.sub(r"^VAR\s+", "", k.strip()), re.sub(r"^VAR\s+", "", v.strip().rstrip("."))
            if k.lower() in c.lower() and v.lower() in c.lower() and k.lower() != v.lower():
                if v not in seen[k.lower()]:
                    seen[k.lower()].append(v)
                    pairs.append((k.lower(), v, len(seen[k.lower()]) - 1))
    return pairs


def cmd_prompts(a):
    seen, n_chunks, n_need = set(), 0, 0
    with open(a.out, "w") as f:
        for L, t, i, s in samples(a.root, a.lengths, a.tasks, a.n):
            if a.only and i not in a.only.get(t, range(10 ** 9)):
                continue
            ctx, q = R.split(s)
            cs = R.chunks(ctx)
            idx = needed(t, cs, q, a.arm)
            n_chunks += len(cs)
            n_need += len(idx)
            for j in idx:
                p = R.FACT_PROMPT.format(chunk=cs[j])
                h = pid(p)
                if h not in seen:
                    seen.add(h)
                    f.write(json.dumps({"id": h, "prompt": p}) + "\n")
    print(f"{a.arm}: {n_chunks} chunks, {n_need} to read, {len(seen)} unique prompts -> {a.out}")


def load_reads(paths):
    out = {}
    for p in paths.split(","):
        for l in open(p):
            r = json.loads(l)
            out[r["id"]] = r["text"]
    return out


def cmd_assemble(a):
    reads = load_reads(a.reads)
    miss = 0
    with open(a.out, "w") as f:
        for L, t, i, s in samples(a.root, a.lengths, a.tasks, a.n):
            if a.only and i not in a.only.get(t, range(10 ** 9)):
                continue
            ctx, q = R.split(s)
            cs = R.chunks(ctx)
            idx = needed(t, cs, q, a.arm)
            raw = ["[]"] * len(cs)
            for j in idx:
                h = pid(R.FACT_PROMPT.format(chunk=cs[j]))
                if h not in reads:
                    miss += 1
                    raw[j] = None
                else:
                    raw[j] = reads[h]
            if any(raw[j] is None for j in idx):
                rec = {"task": t, "length": L, "i": i, "index": s["index"], "outputs": s["outputs"], "missing": True}
                f.write(json.dumps(rec) + "\n")
                continue
            pairs = grounded(cs, raw)
            table = V.table_of(pairs)
            full = a.arm == "full" or t == "vt"
            rec = {"task": t, "length": L, "i": i, "index": s["index"], "outputs": s["outputs"], "n_chunks": len(cs),
                   "read": idx if len(idx) < len(cs) else "all", "n_pairs": len(pairs)}
            if t.startswith("niah"):
                rec["pred_dict"] = R.answer_niah(table, q)
            else:
                rec["pred_dict"] = R.answer_vt(table, q)
            if full:
                st = V.Store(pairs)
                if t.startswith("niah"):
                    rec["pred_store"] = V.answer_niah_vsa(st, q)
                else:
                    names = list(dict.fromkeys(k for k, _, _ in pairs))
                    rec["pred_store"] = V.answer_vt_vsa(st, names, q)
                rec.update(shards=st.S, max_load=int(np.max(st.loads)))
            # per-chunk outputs of the chunks that can matter (all, for VT and the full arm, are long: keep them only
            # for NIAH relevant chunks; full-arm outputs stay in READS)
            if not full:
                rec["raw"] = {str(j): raw[j] for j in idx}
            rec["pairs_q"] = [p for p in pairs if (not t.startswith("niah")) or p[0] in set(queried_keys(q))]
            f.write(json.dumps(rec) + "\n")
    print(f"assembled -> {a.out}; prompts missing from the reads: {miss}")


def acc(pred, refs):
    return 100.0 * R.string_match_all(pred, refs)


def cmd_score(a):
    rows = [json.loads(l) for l in open(a.out)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["length"], r["task"])].append(r)
    for (L, t), rs in sorted(by.items()):
        ok = [r for r in rs if not r.get("missing")]
        d = np.mean([acc(r["pred_dict"], r["outputs"]) for r in ok]) if ok else float("nan")
        st = [acc(r["pred_store"], r["outputs"]) for r in ok if "pred_store" in r]
        ag = [r["pred_store"] == r["pred_dict"] for r in ok if "pred_store" in r]
        print(f"{L:>7} {t:16s} n {len(ok):3d}/{len(rs):3d}  dict {d:6.2f}" +
              (f"  store {np.mean(st):6.2f}  store=dict {sum(ag)}/{len(ag)}" if st else ""))


def b21_rows(b21_dir):
    """B21's banked samples keyed by (length, task, position within the task), in file order."""
    out, pos = {}, collections.Counter()
    for fp in sorted(glob.glob(os.path.join(b21_dir, "*.jsonl")), key=lambda p: (int(os.path.basename(p).split("_")[0]),
                                                                                  int(p.rsplit("_", 1)[1].split(".")[0]))):
        for l in open(fp):
            r = json.loads(l)
            k = (int(r["length"]), r["task"])
            out[k + (pos[k],)] = r
            pos[k] += 1
    return out


def cmd_gate8b(a):
    """Reproduction of B21 by the 8B replay: (1) per chunk, the replayed output equals B21's banked raw output on every
    read chunk; (2) per sample, the dict's prediction equals B21's pred_table."""
    b21 = b21_rows(a.b21)
    rows = [json.loads(l) for l in open(a.rel)]
    reads = load_reads(a.reads) if a.reads else None
    per_len = collections.defaultdict(lambda: [0, 0, 0, 0])  # chunks equal, chunks, samples equal, samples
    diffs = []
    for r in rows:
        b = b21[(r["length"], r["task"], r["i"])]
        assert b["outputs"] == r["outputs"] and b["index"] == r["index"], (r["length"], r["task"], r["i"])
        c = per_len[r["length"]]
        if isinstance(r.get("raw"), dict):
            for j, txt in r["raw"].items():
                c[0] += txt == b["raw"][int(j)]
                c[1] += 1
                if txt != b["raw"][int(j)]:
                    diffs.append(("chunk", r["length"], r["task"], r["i"], int(j), b["raw"][int(j)], txt))
        elif reads is not None and r["task"] == "vt":
            pass  # VT: compared through the sample prediction and the pairs below
        same = r["pred_dict"] == b["pred_table"]
        c[2] += same
        c[3] += 1
        if not same:
            diffs.append(("sample", r["length"], r["task"], r["i"], b["pred_table"], r["pred_dict"],
                          acc(b["pred_table"], b["outputs"]), acc(r["pred_dict"], r["outputs"])))
        if r["task"] == "vt" and "pred_store" in r:
            c.append(r["pred_store"] == b["pred_vsa"])
    for L in sorted(per_len):
        c = per_len[L]
        print(f"{L:>7}: chunks equal {c[0]}/{c[1]}  samples equal {c[2]}/{c[3]}")
    for d in diffs[:40]:
        print("DIFF", json.dumps(d)[:400])
    return per_len, diffs


def cmd_selftest(a):
    """No model: on a few 128k samples per task, B21's banked per-chunk outputs fed through 'rel' (only the needed
    chunks, the rest '[]') must give B21's pred_table, and fed through 'full' its pred_table and pred_vsa."""
    b21 = b21_rows(a.b21)
    bad = n = 0
    for L, t, i, s in samples(a.root, [131072, 4096], TASKS, 6):
        b = b21[(L, t, i)]
        ctx, q = R.split(s)
        cs = R.chunks(ctx)
        assert len(cs) == len(b["raw"]), (L, t, i, len(cs), len(b["raw"]))
        for arm in ("rel", "full"):
            idx = set(needed(t, cs, q, arm))
            raw = [b["raw"][j] if j in idx else "[]" for j in range(len(cs))]
            pairs = grounded(cs, raw)
            table = V.table_of(pairs)
            pd = R.answer_niah(table, q) if t.startswith("niah") else R.answer_vt(table, q)
            n += 1
            bad += pd != b["pred_table"]
            if arm == "full" or t == "vt":
                st = V.Store(pairs)
                ps = V.answer_niah_vsa(st, q) if t.startswith("niah") else V.answer_vt_vsa(
                    st, list(dict.fromkeys(k for k, _, _ in pairs)), q)
                n += 1
                bad += ps != b["pred_vsa"]
                if arm == "full":
                    n += 1
                    bad += [list(p) for p in pairs] != [list(p) for p in b["pairs"]]
    print(f"selftest: {bad} mismatches in {n} checks")
    assert bad == 0


def cmd_g0(a):
    """Gate G0 (no model): B21's own banked per-chunk outputs, with every chunk outside the 'rel' set replaced by "[]",
    give B21's pred_table on every sample (and on VT, where every chunk is read, its pred_vsa)."""
    b21 = b21_rows(a.b21)
    per = collections.defaultdict(lambda: [0, 0])
    bad = []
    for L, t, i, s in samples(a.root, a.lengths, a.tasks, a.n):
        b = b21[(L, t, i)]
        assert b["outputs"] == s["outputs"]
        ctx, q = R.split(s)
        cs = R.chunks(ctx)
        assert len(cs) == len(b["raw"])
        idx = set(needed(t, cs, q, "rel"))
        raw = [b["raw"][j] if j in idx else "[]" for j in range(len(cs))]
        pairs = grounded(cs, raw)
        table = V.table_of(pairs)
        if t.startswith("niah"):
            okk = R.answer_niah(table, q) == b["pred_table"]
        else:
            st = V.Store(pairs)
            okk = R.answer_vt(table, q) == b["pred_table"] and V.answer_vt_vsa(
                st, list(dict.fromkeys(k for k, _, _ in pairs)), q) == b["pred_vsa"]
        per[L][0] += okk
        per[L][1] += 1
        if not okk:
            bad.append((L, t, i))
    for L in sorted(per):
        print(f"G0 {L:>7}: {per[L][0]}/{per[L][1]} samples reproduce B21")
    print("G0 mismatches:", bad[:20])
    return per, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "assemble", "score", "gate8b", "selftest", "g0"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", default="")      # JSON {task: [positions]} restricting the samples
    ap.add_argument("--reads", default="")
    a = ap.parse_args()
    a.lengths = [int(x) for x in a.lengths.split(",")]
    a.tasks = a.tasks.split(",")
    a.only = {t: set(v) for t, v in json.loads(a.only).items()} if a.only else None
    if a.cmd == "prompts":
        a.arm, a.root, a.out = a.args
        cmd_prompts(a)
    elif a.cmd == "assemble":
        a.arm, a.root, a.reads, a.out = a.args
        cmd_assemble(a)
    elif a.cmd == "score":
        a.out = a.args[0]
        cmd_score(a)
    elif a.cmd == "gate8b":
        a.rel, a.b21 = a.args
        cmd_gate8b(a)
    elif a.cmd == "g0":
        a.root, a.b21 = a.args
        cmd_g0(a)
    elif a.cmd == "selftest":
        a.root, a.b21 = a.args
        cmd_selftest(a)


if __name__ == "__main__":
    main()
