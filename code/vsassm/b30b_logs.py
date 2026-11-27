"""B30b part (a) (prereg/B30b_referee_baselines_prereg.md): Qwen3-32B over the whole operational log, re-run with
Qwen3's recommended thinking-mode sampling and a 32,768-token reasoning budget, on the paper's three held-out one-day
sets (location 414 / impact 230 / window 460), with the paper's prompts and scorers.

Read-only over nsr/ and apps/real/: the scorers are verbatim copies in b30b_frozen/ (md5 in the prereg); the logs, the
banked problem files and the banked runs are only read.

  python3 -B b30b_logs.py gate                 G-L0: the banked Qwen3-32B runs and the systems, rescored here, give the
                                               published accuracies (and the published per-question answers)
  python3 -B b30b_logs.py prompts OUT.jsonl    the test prompts (ids "<set>|<n>|<idx>"), copied from the banked files
  python3 -B b30b_logs.py split P.jsonl DIR    the prompts as the original runner's input files (G-E)

Library: per_question(set, runs, forced) -> {qkey: (cluster, scenario, ok, extra)}; loaders for every arm.
"""
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FZ = os.path.join(HERE, "b30b_frozen")
for p in (FZ, os.path.join(FZ, "apps", "real"), os.path.join(FZ, "nsr")):
    sys.path.insert(0, p)
import build_real  # noqa: E402  parse_forced (location forced answers)
import nsr1_llm as L  # noqa: E402  prompts' parser (resolve)
import nsr1_score  # noqa: E402  gold()
import nsr3_score  # noqa: E402  states(), affected()
import run_real  # noqa: E402  ok(), resolve_answer(), gid()
from product import Registry  # noqa: E402

REAL = os.path.join(HERE, "apps", "real")
NSR = os.path.join(HERE, "nsr")
SCEN = ["warehouse", "biobank", "home", "ops"]
SETS = ["where", "impact", "window"]
PROBLEMS = {"where": [os.path.join(REAL, f"llm_problems_test2_assets_{sc}.jsonl") for sc in SCEN],
            "impact": [os.path.join(NSR, "out", "held", "nsr1_llm_problems_nsr1_arm3.jsonl")],
            "window": [os.path.join(NSR, "out", "held", "nsr3_llm_problems_nsr2.jsonl")]}
TASK = {"where": "REAL", "impact": "NSR1", "window": "NSR3"}
BANKED = {  # the published Qwen3-32B runs (b6 record format) and their budget-forced answers
    "where": (os.path.join(REAL, "out", "test2a", "qwen3-32b_REAL_n*.jsonl"), os.path.join(REAL, "out", "test2a", "forced_qwen3-32b.jsonl")),
    "impact": (os.path.join(NSR, "out", "held", "llm_arm3v", "qwen3-32b_NSR1_n*_vllm.jsonl"),
               os.path.join(NSR, "out", "held", "llm_arm3v", "forced512_qwen3-32b.jsonl")),
    "window": (os.path.join(NSR, "out", "held", "nsr3_llm", "qwen3-32b_NSR3_n*_vllm.jsonl"),
               os.path.join(NSR, "out", "held", "nsr3_llm", "forced512_qwen3-32b.jsonl"))}
PUBLISHED = {"where": (0.543, 0.531), "impact": (0.2783, 0.2652), "window": (0.2022, 0.1935)}  # (forced, strict)
SYSTEM_PUB = {"where": 0.959, "impact": 0.9304, "window": 0.8739}
AMBIG = "Which of Grandpa's things"  # the 80 home impact questions read as "Grandpa's own things" (NSR1 verdict)


def jl(p):
    return [json.loads(l) for l in open(p)]


def ungz(root=os.path.join(HERE, "b30b_out")):
    """The committed large outputs are gzipped; restore any *.jsonl.gz whose .jsonl is missing (no-op otherwise)."""
    import gzip
    import shutil
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".jsonl.gz") and not os.path.exists(os.path.join(d, f[:-3])):
                with gzip.open(os.path.join(d, f), "rb") as a, open(os.path.join(d, f[:-3]), "wb") as b:
                    shutil.copyfileobj(a, b)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


_LOGS = {}


def logs(s):
    if s not in _LOGS:
        if s == "where":
            ls = [d for sc in SCEN for d in jl(os.path.join(REAL, "data", f"{sc}_test2.jsonl"))]
        else:
            split = "nsr1" if s == "impact" else "nsr2"
            ls = [d for sc in SCEN for d in jl(os.path.join(NSR, "data", f"{sc}_{split}.jsonl"))]
        _LOGS[s] = {d["id"]: d for d in ls}
    return _LOGS[s]


def problems(s):
    return [r for p in PROBLEMS[s] for r in jl(p)]


def questions(s):
    """Every question of a set, in a fixed order: (qkey, log id, scenario, n, idx, extra)."""
    out = []
    pr = {(r["n"], r["idx"]): r for r in problems(s)}
    if s == "where":
        for lid, l in logs(s).items():
            n = run_real.gid(l)
            for qi, q in enumerate(l["questions"]):
                assert (n, qi) in pr
                out.append(((lid, qi), lid, l["scenario"], n, qi, {"q": q["q"]}))
        return out
    for (n, idx), r in sorted(pr.items()):
        lid = r["id"].split("|")[0]
        out.append((r["id"], lid, logs(s)[lid]["scenario"], n, idx, {"q": r["prompt"].rsplit("Question: ", 1)[1].split("\n")[0]}))
    return out


_G3 = {}


def window_gold(qid):
    lid, qt, tg, i, j = qid.split("|")
    if lid not in _G3:
        _G3[lid] = nsr3_score.states(logs("window")[lid])
    g = set()
    for st in _G3[lid][max(int(i) - 1, 0):int(j) + 1]:
        g |= nsr3_score.affected(logs("window")[lid], st, qt, tg)
    return g


def score_one(s, qkey, lid, txt_strict, txt_forced, hit):
    """(strict ok, forced ok) with the paper's parser and scorer for that set."""
    log = logs(s)[lid]
    if s == "where":
        reg = Registry(log)
        q = log["questions"][qkey[1]]
        strict = (not hit) and txt_strict is not None and run_real.ok(q, run_real.resolve_answer(q, txt_strict, reg), reg)
        forced = strict or (hit and txt_forced is not None and run_real.ok(q, build_real.parse_forced(q, txt_forced, reg), reg))
        return strict, forced
    if s == "impact":
        _, qt, tg = qkey.split("|")
        g = nsr1_score.gold(log, qt, tg)
        ok = lambda t: (lambda p: p is not None and sorted(p) == g)(L.resolve(t, log))
    else:
        g = window_gold(qkey)
        ok = lambda t: (lambda p: p is not None and set(p) == g)(L.resolve(t, log))
    strict = (not hit) and ok(txt_strict)
    forced = ok(txt_forced if hit else txt_strict)
    return strict, forced


def answers(s, runs, forced):
    """Per question: the parsed answer the registered score uses (completed run: the answer text; run at the limit: its
    forced answer, as the published 'forced' score), and both scores. For the second scorer and the hand-reading."""
    out = {}
    for qk, lid, sc, n, idx, ex in questions(s):
        r = runs.get((n, idx))
        log = logs(s)[lid]
        hit = bool(r and r["hit_limit"])
        txt = None if r is None else (forced.get((n, idx)) if hit else r["answer_text"])
        if s == "where":
            reg = Registry(log)
            q = log["questions"][idx]
            pred = None if txt is None else (build_real.parse_forced(q, txt, reg) if hit else run_real.resolve_answer(q, txt, reg))
        else:
            pred = L.resolve(txt, log)
        st, fo = (False, False) if r is None else score_one(s, qk, lid, None if hit else r["answer_text"], forced.get((n, idx)), hit)
        out[qk] = {"id": "|".join(map(str, qk)) if s == "where" else qk, "log": lid, "scenario": sc, "q": ex["q"],
                   "pred": pred, "ok": fo, "ok_strict": st, "hit_limit": hit, "missing": r is None}
    return out


def per_question(s, runs, forced):
    """runs: {(n, idx): {"hit_limit", "answer_text"}}, forced: {(n, idx): text}. Missing runs count as wrong."""
    out = {}
    for qk, lid, sc, n, idx, ex in questions(s):
        r = runs.get((n, idx))
        if r is None:
            out[qk] = (lid, sc, False, False, ex)
            continue
        st, fo = score_one(s, qk, lid, None if r["hit_limit"] else r["answer_text"], forced.get((n, idx)), r["hit_limit"])
        out[qk] = (lid, sc, st, fo, ex)
    return out


def banked(s):
    runs = {(r["n"], r["idx"]): r for fp in sorted(glob.glob(BANKED[s][0])) for r in jl(fp)}
    forced = {(r["n"], r["idx"]): r["forced_answer"] for r in jl(BANKED[s][1])}
    return runs, forced


def b6_dir(s, d, name="qwen3-32b", forced_name=None):
    runs = {(r["n"], r["idx"]): r for fp in sorted(glob.glob(os.path.join(d, f"{name}_{TASK[s]}_n*.jsonl"))) for r in jl(fp)}
    fp = os.path.join(d, forced_name or f"forced512_{name}.jsonl")
    forced = {(r["n"], r["idx"]): r["forced_answer"] for r in jl(fp)} if os.path.exists(fp) else {}
    return runs, forced


def gen_file(s, path):
    """b30b_gen.py think32k records ("<set>|<n>|<idx>") -> (runs, forced) in the b6 shape."""
    runs, forced = {}, {}
    for r in jl(path):
        st, n, idx = r["id"].split("|")
        if st != s:
            continue
        runs[(int(n), int(idx))] = {"hit_limit": r["hit_limit"], "answer_text": r["answer_text"], "rec": r}
        if r["hit_limit"]:
            forced[(int(n), int(idx))] = r["forced_answer"]
    return runs, forced


def system(s):
    """The paper's system on each set, per question: (cluster, scenario, ok)."""
    out = {}
    if s == "where":
        rd = {r["id"]: r for r in jl(os.path.join(REAL, "out", "reader_test2_verify_v4.jsonl"))}
        for lid, l in logs(s).items():
            reg = Registry(l)
            for qi, q in enumerate(l["questions"]):
                out[(lid, qi)] = (lid, l["scenario"], run_real.ok(q, rd[lid]["answers"][qi]["pred"], reg))
        return out
    if s == "impact":
        for r in jl(os.path.join(NSR, "out", "nsr1_nsr1_arm1.jsonl")):
            g = nsr1_score.gold(logs(s)[r["log"]], r["qtype"], r["target"])
            out[f"{r['log']}|{r['qtype']}|{r['target']}"] = (r["log"], r["scenario"], r["pred"] is not None and sorted(r["pred"]) == g)
        return out
    for r in jl(os.path.join(NSR, "out", "nsr3_nsr2_history.jsonl")):
        lid = r["id"].split("|")[0]
        out[r["id"]] = (lid, logs(s)[lid]["scenario"], r["pred"] is not None and set(r["pred"]) == window_gold(r["id"]))
    return out


def cmd_gate():
    allok = True
    for s in SETS:
        pq = per_question(s, *banked(s))
        fo = sum(x[3] for x in pq.values()) / len(pq)
        st = sum(x[2] for x in pq.values()) / len(pq)
        pf, ps = PUBLISHED[s]
        ok = abs(fo - pf) <= 5e-4 and abs(st - ps) <= 5e-4  # published to 3-4 decimals
        line = f"G-L0 {s}: banked Qwen3-32B rescored forced {fo:.4f} strict {st:.4f} (n={len(pq)}); published {pf} / {ps}"
        if s == "impact":
            ex = [x for x in pq.values() if AMBIG not in x[4]["q"]]
            line += f"; excl. the {len(pq) - len(ex)} ambiguous home questions {sum(x[3] for x in ex) / len(ex):.3f} (n={len(ex)}, published 0.273)"
            ok &= len(ex) == 150 and abs(sum(x[3] for x in ex) / len(ex) - 0.273) <= 5e-4
            pub = {f"{r['log']}|{r['qtype']}|{r['target']}": r["pred"] for r in jl(os.path.join(NSR, "out", "nsr1_nsr1_arm3_qwen3-32b_forced.jsonl"))}
            runs, forced = banked(s)
            same = sum(L.resolve(forced.get((n, i)) if runs[(n, i)]["hit_limit"] else runs[(n, i)]["answer_text"],
                                 logs(s)[lid]) == pub[qk] for qk, lid, sc, n, i, e in questions(s))
            line += f"; per-question answers equal to the published file {same}/{len(pq)}"
            ok &= same == len(pq)
        if s == "window":
            pub = {r["id"]: r["pred"] for r in jl(os.path.join(NSR, "out", "nsr3_nsr2_llm_qwen3-32b_forced.jsonl"))}
            runs, forced = banked(s)
            same = sum(L.resolve(forced.get((n, i)) if runs[(n, i)]["hit_limit"] else runs[(n, i)]["answer_text"],
                                 logs(s)[lid]) == pub[qk] for qk, lid, sc, n, i, e in questions(s))
            line += f"; per-question answers equal to the published file {same}/{len(pq)}"
            ok &= same == len(pq)
        sy = system(s)
        sa = sum(x[2] for x in sy.values()) / len(sy)
        ok &= abs(sa - SYSTEM_PUB[s]) <= 5e-4 and set(sy) == set(pq)
        line += f"; system {sa:.4f} (published {SYSTEM_PUB[s]})"
        print(line + f" -> {'PASS' if ok else 'FAIL'}")
        allok &= ok
    for s in SETS:
        for p in PROBLEMS[s]:
            print(f"  problems {os.path.relpath(p, HERE)} md5 {md5(p)}")
    return allok


def cmd_prompts(out):
    n = 0
    with open(out, "w") as f:
        for s in SETS:
            for r in problems(s):
                f.write(json.dumps({"id": f"{s}|{r['n']}|{r['idx']}", "prompt": r["prompt"]}) + "\n")
                n += 1
    print(out, n, md5(out))


def cmd_split(prompts, d):
    """G-E input files: the banked problem files in the original runner's format, one per set."""
    os.makedirs(d, exist_ok=True)
    for s in SETS:
        with open(os.path.join(d, f"problems_{s}.jsonl"), "w") as f:
            for r in problems(s):
                f.write(json.dumps({"task": TASK[s], "n": r["n"], "idx": r["idx"], "id": r.get("id", ""), "prompt": r["prompt"]}) + "\n")
    print(d)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "gate":
        raise SystemExit(0 if cmd_gate() else 3)
    if a[1] == "prompts":
        cmd_prompts(a[2])
    if a[1] == "split":
        cmd_split(a[2], a[3])
