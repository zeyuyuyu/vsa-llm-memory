"""B29 log part (prereg/B29_same_tooling_prereg.md): the operational-log pipeline with the reader model varied.

The pipeline is used exactly as each banked result used it, imported read-only (run with python3 -I -B):
  test2 (location questions)       apps/real at commit cd9de8d, copied verbatim to b29_frozen/cd9de8d/
  nsr1 (impact), nsr2 (window), dev  nsr/vendor_707351b/ (the code NSR1-NSR3 used, via nsr/nsr1_reader.py)
Only the reader object is swapped (b14.Reader is replaced by a factory returning the chosen reader); the prompts
(prompt C with its worked examples, the forced-choice prompt and the yes/no check), the parser, the registry, the
connectors, the memory and the question engines are unchanged. Every reader call is logged (prompt, text, tokens).

  read     --split S --reader hf --model DIR [--adapter DIR] --tag T [--logs id,id,...]     (GPU)
  read     --split S --reader zheng --model gpt-6.1-sol --effort low --tag T [--logs ...]    (API, api_zheng)
  read     --split S --reader replay --calls CALLS.jsonl --memory dict --tag T               (CPU: same reader
           output, a plain dict in place of the 256 x 256 matrix memory)
  answer   --split S --tag T          per-question answers of the registered engine(s) for that split
  score    --split S --tag T          accuracy (the registered scorers' gold)
Outputs: b29_out/logs/reader_<split>_<tag>.jsonl, calls_<split>_<tag>.jsonl, ans_<split>_<tag>.jsonl.
"""
import argparse
import collections
import json
import os
import sys
import time
import types

import site  # python3 -I drops the user site-packages (numpy lives there on some hosts): append it, last
if os.path.isdir(site.getusersitepackages()) and site.getusersitepackages() not in sys.path:
    sys.path.append(site.getusersitepackages())

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "b29_out", "logs")
PIPE = {"test2": ("b29_frozen/cd9de8d", "apps/real/data"),
        "dev_cd": ("b29_frozen/cd9de8d", "apps/real/data"),
        "nsr1": ("nsr/vendor_707351b", "nsr/data"),
        "nsr2": ("nsr/vendor_707351b", "nsr/data"),
        "dev": ("nsr/vendor_707351b", "nsr/data")}
DATASPLIT = {"dev_cd": "dev"}
SETS = {"test2": "where", "dev_cd": "where", "nsr1": "impact", "nsr2": "window", "dev": "impact+window"}


def setup(split):
    vend, data = PIPE[split]
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, vend))
    import run_real  # noqa: E402  (the pinned pipeline; its own path inserts put ROOT and its directory first)
    assert os.path.dirname(os.path.abspath(run_real.__file__)) == os.path.join(ROOT, vend), run_real.__file__
    run_real.DATA = os.path.join(ROOT, data)
    run_real.OUT = OUTD
    return run_real


class Logged:
    """Wraps a reader (reader(prompts) -> [(text, n)]) and appends every call to a JSONL file."""

    def __init__(self, base, path):
        self.base, self.path = base, path
        self.n = 0

    def __call__(self, prompts, bs=64, budget=0):
        outs = self.base(prompts) if not isinstance(self.base, HFBase) else self.base(prompts, bs=bs, budget=budget)
        with open(self.path, "a") as f:
            for p, (t, n) in zip(prompts, outs):
                f.write(json.dumps({"p": p, "t": t, "n": n}) + "\n")
        self.n += len(prompts)
        return outs


class HFBase:
    def __init__(self, r):
        self.r = r

    def __call__(self, prompts, bs=64, budget=0):
        return self.r(prompts, bs=bs, budget=budget)


class Replay:
    """A reader that returns the logged output of an earlier run for the same prompt (raises on an unseen prompt)."""

    def __init__(self, path):
        self.c = {}
        for l in open(path):
            r = json.loads(l)
            self.c.setdefault(r["p"], (r["t"], r["n"]))
        self.miss = 0

    def __call__(self, prompts, bs=64, budget=0):
        out = []
        for p in prompts:
            if p not in self.c:
                self.miss += 1
                raise KeyError("prompt not in the logged calls: the replay diverged")
            out.append(self.c[p])
        return out


def dict_memory(b14):
    """A plain dict with b14.Memory's interface: the same name lists and codebook order (so the reader's prompts are
    unchanged), the same canonicalisation and the same write / swap semantics, but the state is a Python dict."""

    class DictMemory(b14.Memory):
        def __init__(self, domain):
            super().__init__(domain)
            self.d = {}

        def apply(self, ev):
            op, a, b = ev
            if op == "swap" and self.domain == "locations":
                a, b = b14.canon(a, self.places), b14.canon(b, self.places)
                if not a or not b or a == b:
                    return None
                for n in (a, b):
                    if n not in self.places:
                        self.places.append(n)
                self.V.get(a), self.V.get(b)
                for o, p in list(self.d.items()):
                    if p == a:
                        self.d[o] = b
                    elif p == b:
                        self.d[o] = a
                return ("swap", a, b)
            if op != "move":
                return None
            a = b14.canon(a, self.objs)
            b = b14.canon(b, self.objs if self.domain == "containers" else self.places)
            if not a or not b or a == b:
                return None
            if a not in self.objs:
                self.objs.append(a)
            lst = self.objs if self.domain == "containers" else self.places
            if b not in lst:
                lst.append(b)
            self.K.get(a), self.V.get(b)
            self.d[a] = b
            return ("write", a, b)

        def read(self, name):
            if name not in self.d:
                return None, 0.0
            return self.d[name], 1.0

    return DictMemory


class ClaudeReader:
    """Claude Opus 5.5 through the user's endpoint (~/nsengine/runs/claude_api.py, which reads the key file and logs
    spend to its own ledger at list price). One message per prompt, effort as given, a prompt -> answer cache on disk.
    A call that errors returns ("", 0) and is counted, as APIReader does."""

    def __init__(self, model, effort, cache, workers=16, max_tokens=2000):
        sys.path.insert(0, os.path.expanduser("~/nsengine/runs"))
        import claude_api  # noqa: E402
        import concurrent.futures as cf
        import threading
        self.api, self.cf, self.lock = claude_api, cf, threading.Lock()
        self.model, self.effort, self.workers, self.max_tokens, self.cache_path = model, effort, workers, max_tokens, cache
        self.cache = {}
        if os.path.exists(cache):
            for l in open(cache):
                try:
                    r = json.loads(l)
                    self.cache[r["p"]] = (r["t"], r["n"])
                except Exception:  # noqa: BLE001
                    continue
        self.m = {"calls": 0, "cached": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
        self.errors = []

    def _one(self, p):
        with self.lock:
            if p in self.cache:
                self.m["cached"] += 1
                return self.cache[p]
        for attempt in range(3):
            r = self.api.chat(self.model, [{"role": "user", "content": p}], max_tokens=self.max_tokens,
                              effort=self.effort, tag="b29")
            if "error" not in r:
                break
            time.sleep(5 * (attempt + 1))
        with self.lock:
            if "error" in r:
                self.m["failed"] += 1
                self.errors.append(r["error"])
                return "", 0
            u = r["usage"]
            out = (r["text"].strip(), u["output_tokens"])
            self.m["calls"] += 1
            self.m["prompt_tokens"] += u["input_tokens"] + u["cache_read"] + u["cache_write"]
            self.m["completion_tokens"] += u["output_tokens"]
            self.m["usd"] += r["usd"]
            self.cache[p] = out
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"p": p, "t": out[0], "n": out[1], "model": r.get("model")}) + "\n")
            return out

    def __call__(self, prompts, bs=None, budget=0):
        with self.cf.ThreadPoolExecutor(max(1, min(self.workers, len(prompts)))) as ex:
            return list(ex.map(self._one, prompts))

    def meter(self):
        return dict(self.m, model=self.model, effort=self.effort)


def make_reader(a, b14):
    if a.reader == "hf":
        return HFBase(b14.Reader(a.model, a.adapter or None))
    if a.reader == "zheng":
        sys.path.insert(0, os.path.join(ROOT, "apps", "real"))
        from api_reader import zheng_reader  # noqa: E402
        cache = os.path.join(OUTD, f"zheng_cache_{a.model}_{a.effort}.jsonl")
        return zheng_reader(a.model, a.effort, cache=cache, workers=a.workers, max_tokens=a.max_tokens)
    if a.reader == "claude":
        return ClaudeReader(a.model, a.effort, os.path.join(OUTD, f"claude_cache_{a.model}_{a.effort}.jsonl"),
                            workers=a.workers)
    if a.reader == "replay":
        return Replay(a.calls)
    raise ValueError(a.reader)


def cmd_read(a):
    os.makedirs(OUTD, exist_ok=True)
    rr = setup(a.split)
    import b14  # noqa: E402  (ROOT/b14.py, unchanged since both pinned commits)
    base = make_reader(a, b14)
    calls = os.path.join(OUTD, f"calls_{a.split}_{a.tag}.jsonl")
    if os.path.exists(calls) and a.reader != "replay":
        os.rename(calls, calls + f".old{int(time.time())}")
    logged = Logged(base, calls if a.reader != "replay" else calls + ".replay")
    b14.Reader = lambda *x, **k: logged
    if a.memory == "dict":
        b14.Memory = dict_memory(b14)
    ids = set(a.logs.split(",")) if a.logs else None
    load0 = rr.load
    rr.load = lambda split, suffix="": [l for l in load0(split, suffix) if ids is None or l["id"] in ids]
    ns = types.SimpleNamespace(split=DATASPLIT.get(a.split, a.split), suffix="", model=a.model, adapter=a.adapter,
                               verify=True, tag="_" + a.split + "_" + a.tag, out="", only="", large=0, list_limit=None)
    t0 = time.time()
    rr.run_reader(ns)
    src = os.path.join(OUTD, f"reader_{ns.split}_verify{ns.tag}.jsonl")
    dst = os.path.join(OUTD, f"reader_{a.split}_{a.tag}.jsonl")
    os.replace(src, dst)
    meta = {"split": a.split, "tag": a.tag, "reader": a.reader, "model": a.model, "adapter": a.adapter,
            "memory": a.memory, "logs": a.logs, "calls": logged.n, "wall_s": round(time.time() - t0, 1),
            "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.reader == "hf":
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
    if hasattr(base, "meter"):
        meta["meter"] = base.meter()
        meta["errors"] = base.errors[:20]
    if isinstance(base, Replay):
        meta["replay_misses"] = base.miss
    with open(dst + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


# ------------------------------------------------------------------------------------------------ engines
def logs_of(split):
    vend, data = PIPE[split]
    out = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        p = os.path.join(ROOT, data, f"{sc}_{DATASPLIT.get(split, split)}.jsonl")
        for line in open(p):
            d = json.loads(line)
            out[d["id"]] = d
    return out


def where_answers(split, reader_path):
    logs = logs_of(split)
    out = []
    for r in map(json.loads, open(reader_path)):
        l = logs[r["id"]]
        for qi, q in enumerate(l["questions"]):
            out.append({"id": f"{l['id']}|{qi}", "log": l["id"], "scenario": l["scenario"], "kind": q["kind"],
                        "pred": r["answers"][qi]["pred"]})
    return out


def impact_answers(reader_path, split="nsr1"):
    """NSR1's engines on a reader output (nsr/nsr1.py functions, unchanged): arm1 = FHRR rule memory, arm2a = count
    table, arm2b = hand-written graph query, all over the read state; one worked trace per question type (n = 1)."""
    sys.path.insert(0, os.path.join(ROOT, "nsr"))
    import nsr1  # noqa: E402
    import nsr1_catalog as C  # noqa: E402
    logs = logs_of(split)
    reader = {json.loads(l)["id"]: json.loads(l) for l in open(reader_path)}
    traces = nsr1.worked_traces(C.load_logs("dev"), 1)
    asked = {qt: ts[0][1] for qt, ts in traces.items()}
    R_vsa, R_tab = nsr1.learn(nsr1.VSARules(), traces), nsr1.learn(nsr1.TableRules(), traces)
    out = []
    for lid in sorted(reader):
        log = logs[lid]
        cat = C.catalog(log)
        rm, src = nsr1.rm_state(reader[lid])
        for q in C.questions(log):
            p1, tr = nsr1.answer_rules(q, rm, cat, R_vsa, asked, src)
            out.append({"id": f"{lid}|{q['qtype']}|{q['target']}", "log": lid, "scenario": log["scenario"],
                        "qtype": q["qtype"], "target": q["target"], "pred": p1,
                        "pred_arm2a": nsr1.answer_rules(q, rm, cat, R_tab, asked)[0],
                        "pred_arm2b": C.answer(q["qtype"], q["target"], rm, cat), "trace": tr})
    return out


def window_answers(reader_path, split="nsr2"):
    """NSR3's history engine on a reader output (nsr/nsr3_engines.py logic, nsr3_questions functions unchanged)."""
    sys.path.insert(0, os.path.join(ROOT, "nsr"))
    import nsr1_catalog as C  # noqa: E402
    import nsr3_questions as Q3  # noqa: E402
    logs = logs_of(split)
    rd = {json.loads(l)["id"]: json.loads(l) for l in open(reader_path)}
    out = []
    for lid in sorted(rd):
        log = logs[lid]
        cat = C.catalog(log)
        hist = Q3.read_states(rd[lid])
        for q in Q3.questions(log):
            key = f"{q['log']}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"
            out.append({"id": key, "log": lid, "scenario": log["scenario"],
                        "pred": Q3.union_answer(q["qtype"], q["target"], hist, q["i"], q["j"], cat)})
    return out


def cmd_answer(a):
    rp = os.path.join(OUTD, f"reader_{a.split}_{a.tag}.jsonl")
    kind = SETS[a.split]
    if kind == "where":
        out = where_answers(a.split, rp)
    elif kind == "impact":
        out = impact_answers(rp)
    elif kind == "window":
        out = window_answers(rp)
    else:  # dev: both engines, written to two files
        out = impact_answers(rp, "dev") if a.engine == "impact" else window_answers(rp, "dev")
    path = os.path.join(OUTD, f"ans_{a.split}_{a.tag}" + (f"_{a.engine}" if kind == "impact+window" else "") + ".jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} answers -> {path}")


# ------------------------------------------------------------------------------------------------ gold (registered)
def gold_where(split):
    logs = logs_of(split)
    return {f"{l['id']}|{qi}": (q["kind"], q["a"]) for l in logs.values() for qi, q in enumerate(l["questions"])}


def ok_where(kind, pred, a):
    """apps/real/run_real.py ok() (identical at cd9de8d and HEAD)."""
    if kind == "where":
        return pred is not None and pred == a
    if kind == "what_in":
        return isinstance(pred, list) and sorted(pred) == sorted(a)
    return isinstance(pred, list) and list(pred) == list(a)


def gold_impact(split="nsr1"):
    sys.path.insert(0, os.path.join(ROOT, "nsr"))
    import nsr1_score  # noqa: E402  (the registered independent scorer)
    logs = logs_of(split)
    import nsr1_catalog as C  # noqa: E402
    return {f"{lid}|{q['qtype']}|{q['target']}": nsr1_score.gold(log, q["qtype"], q["target"])
            for lid, log in logs.items() for q in C.questions(log)}


def gold_window(split="nsr2"):
    sys.path.insert(0, os.path.join(ROOT, "nsr"))
    import nsr3_score  # noqa: E402  (the registered independent scorer: own replay and window union)
    import nsr3_questions as Q3  # noqa: E402
    logs = logs_of(split)
    out = {}
    for lid, log in logs.items():
        S = nsr3_score.states(log)
        for q in Q3.questions(log):
            g = set()
            for st in S[max(q["i"] - 1, 0):q["j"] + 1]:
                g |= nsr3_score.affected(log, st, q["qtype"], q["target"])
            out[f"{lid}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"] = sorted(g)
    return out


def correct(split, recs, field="pred", engine="impact"):
    kind = SETS[split]
    if kind == "where":
        g = gold_where(split)
        return {r["id"]: ok_where(g[r["id"]][0], r[field], g[r["id"]][1]) for r in recs}
    if kind == "impact+window":
        kind = engine
    g = gold_impact(split) if kind == "impact" else gold_window(split)
    return {r["id"]: (r[field] is not None and sorted(r[field]) == sorted(g[r["id"]])) for r in recs}


def cmd_score(a):
    dev = SETS[a.split] == "impact+window"
    recs = [json.loads(l) for l in open(os.path.join(OUTD, f"ans_{a.split}_{a.tag}" + (f"_{a.engine}" if dev else "") + ".jsonl"))]
    imp = SETS[a.split] == "impact" or (dev and a.engine == "impact")
    for field in ["pred"] + (["pred_arm2a", "pred_arm2b"] if imp else []):
        c = correct(a.split, recs, field, a.engine)
        per = collections.defaultdict(list)
        for r in recs:
            per[r["scenario"]].append(c[r["id"]])
            per["ALL"].append(c[r["id"]])
        print(field, {k: f"{sum(v) / len(v):.3f} ({sum(v)}/{len(v)})" for k, v in sorted(per.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "answer", "score"])
    ap.add_argument("--split", required=True, choices=sorted(PIPE))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reader", default="hf", choices=["hf", "zheng", "claude", "replay"])
    ap.add_argument("--model", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--calls", default="")
    ap.add_argument("--memory", default="matrix", choices=["matrix", "dict"])
    ap.add_argument("--logs", default="")
    ap.add_argument("--engine", default="impact")
    a = ap.parse_args()
    {"read": cmd_read, "answer": cmd_answer, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
