"""B30b second angle (prereg/B30b_referee_baselines_prereg.md): recompute every accuracy of results/b30b_scores.json
from the saved per-question / per-sample outputs with separate code, and report any difference.

  Logs:     gold from a different derivation than the registered scorers (as B29's b29_check): impact =
            nsr1_catalog.answer (hand-written graph query) on the simulation's TRUE state; window =
            nsr3_questions.union_answer over nsr3_questions.truth_states; location = the stored answer, own comparison.
            Also an independent, lenient answer parse straight from the raw model text (last 'Answer:' line or last
            line, IDs / aliases matched case-insensitively), reported as a robustness number, not required to agree.
  BABILong: the memory's answers re-derived by a plain-dict state tracker written here (no vectors), the official
            compare_answers on every arm, and the per-length means recomputed from the scored files.

python3 -B b30b_check.py
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "b30b_frozen", "nsr"))
import nsr1_catalog as C  # noqa: E402
import nsr3_questions as Q3  # noqa: E402

LO = os.path.join(HERE, "b30b_out", "logs")
BA = os.path.join(HERE, "b30b_out", "bab")


def jl(p):
    return [json.loads(l) for l in open(p)]


def logs_of(ddir, split):
    out = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for d in jl(os.path.join(HERE, ddir, f"{sc}_{split}.jsonl")):
            out[d["id"]] = d
    return out


def gold2():
    g = {"impact": {}, "window": {}, "where": {}}
    for lid, log in logs_of("nsr/data", "nsr1").items():
        cat = C.catalog(log)
        for q in C.questions(log):
            g["impact"][f"{lid}|{q['qtype']}|{q['target']}"] = sorted(C.answer(q["qtype"], q["target"], log["truth"], cat))
    for lid, log in logs_of("nsr/data", "nsr2").items():
        cat = C.catalog(log)
        tr = Q3.truth_states(log)
        for q in Q3.questions(log):
            g["window"][f"{lid}|{q['qtype']}|{q['target']}|{q['i']}|{q['j']}"] = sorted(
                Q3.union_answer(q["qtype"], q["target"], tr, q["i"], q["j"], cat))
    for lid, log in logs_of("apps/real/data", "test2").items():
        for qi, q in enumerate(log["questions"]):
            g["where"][f"{lid}|{qi}"] = (q["kind"], q["a"])
    return g


def right(kind, pred, gold):
    if kind == "where":
        k, a = gold
        if pred is None:
            return False
        if k == "where":
            return pred == a
        if k == "what_in":
            return isinstance(pred, list) and collections.Counter(pred) == collections.Counter(a)
        return isinstance(pred, list) and pred == a
    return pred is not None and sorted(set(pred)) == gold and len(set(pred)) == len(pred)


def lenient(text, log):
    """Own parse: the last 'Answer:' line (else the last line), split on commas / 'and' / '>', every part matched
    against the registry's IDs and aliases (case-insensitive, markdown and parentheses removed)."""
    if not text:
        return None
    lines = [x.strip() for x in text.strip().split("\n") if x.strip()]
    if not lines:
        return None
    ans = [x for x in lines if re.match(r"^[\*\s]*(final\s+)?answer[\*\s]*:", x, re.I)]
    t = re.sub(r"\*\*|`", "", (ans or lines)[-1])
    t = re.sub(r"^\s*(final\s+)?answer\s*:\s*", "", t, flags=re.I).strip().strip(".")
    if re.fullmatch(r"(none|nothing|no \w+)", t, re.I):
        return []
    names = {}
    for e in log["registry"]["items"] + log["registry"]["places"]:
        for n in [e["id"]] + e["aliases"]:
            names.setdefault(n.lower(), e["id"])
    out = []
    for p in re.split(r",|;|\band\b|>|→", t):
        p = re.sub(r"\(.*?\)", "", p).strip().strip(".").lower()
        if p in names:
            out.append(names[p])
    return out


def check_logs(J, g):
    bad = 0
    print("## Logs: accuracy with an independently derived gold (registered parse), and an independent parse")
    raw = {}
    for tag in ("s32_r1", "s32_r2"):
        p = os.path.join(LO, f"{tag}.jsonl")
        if os.path.exists(p):
            raw[tag] = {r["id"]: r for r in jl(p)}
    logs = {"where": logs_of("apps/real/data", "test2"), "impact": logs_of("nsr/data", "nsr1"), "window": logs_of("nsr/data", "nsr2")}
    for s in ("where", "impact", "window"):
        for tag, name in (("s32_r1", "S32 R1"), ("s32_r2", "S32 R2")):
            p = os.path.join(LO, f"ans_{s}_{tag}.jsonl")
            if not os.path.exists(p):
                continue
            recs = jl(p)
            ok = [right(s, r["pred"], g[s][r["id"]]) for r in recs]
            acc = sum(ok) / len(ok)
            v = J["a"][s]["arms"].get(name, {}).get("forced")
            same = v is not None and abs(v - acc) < 5e-5
            bad += not same
            # independent parse from the raw text, matched against the problem id
            n_ok = 0
            if tag in raw and s != "where":
                sys.path.insert(0, HERE)
                import b30b_logs
                prob = {(r["n"], r["idx"]): r["id"] for f in b30b_logs.PROBLEMS[s] for r in jl(f)}
                for (n, i), qid in prob.items():
                    r = raw[tag].get(f"{s}|{n}|{i}")
                    txt = None if r is None else (r["forced_answer"] if r["hit_limit"] else r["answer_text"])
                    n_ok += right(s, lenient(txt, logs[s][qid.split("|")[0]]), g[s][qid])
                ind = f"; independent parse {n_ok / len(prob):.3f}"
            else:
                ind = ""
            print(f"  {s:6s} {name}: {acc:.4f} ({sum(ok)}/{len(ok)}); verdict {v} {'agrees' if same else 'DIFFERS'}{ind}")
    return bad


class Dict:
    """Plain-dict tracker of the B18 event semantics (independent of b18.World's vector memory)."""

    def __init__(self):
        self.loc, self.hold, self.oloc, self.hist, self.gives, self.rels = {}, {}, {}, {}, [], []
        self.either, self.ents = {}, []

    def place(self, x):
        if x in self.hold:
            return self.loc.get(self.hold[x])
        if x in self.oloc:
            return self.oloc[x]
        return self.loc.get(x)

    def apply(self, ev):
        op, a = ev[0], ev[1:]
        for e in [a[0]] + ([a[1]] if op in ("take", "drop", "give") else []) + ([a[2]] if op == "give" else []):
            if op != "rel" and e not in self.ents:
                self.ents.append(e)
        if op == "go":
            self.loc[a[0]] = a[1]
            self.either.pop(a[0], None)
        elif op == "take":
            self.hold[a[1]] = a[0]
            self.oloc.pop(a[1], None)
        elif op == "drop":
            self.hold.pop(a[1], None)
            self.oloc[a[1]] = self.loc.get(a[0])
        elif op == "give":
            self.hold[a[1]] = a[2]
            self.oloc.pop(a[1], None)
            self.gives.append(tuple(a))
        elif op == "not_at":
            if self.loc.get(a[0]) in (a[1], None):
                self.loc[a[0]] = None
            self.either.pop(a[0], None)
        elif op == "either":
            self.loc[a[0]] = None
            self.either[a[0]] = {a[1], a[2]}
        elif op == "rel":
            d = a[1].replace(" of", "").strip()
            self.rels.append((a[0], d, a[2]))
            o = {"north": "south", "south": "north", "east": "west", "west": "east"}.get(d)
            if o:
                self.rels.append((a[2], o, a[0]))
            return
        for e in self.ents:
            p = self.place(e)
            h = self.hist.setdefault(e, [])
            if p and (not h or h[-1] != p):
                h.append(p)

    def answer(self, q):
        q = q.strip().rstrip("?").lower()
        m = re.match(r"where was (?:the )?(.+?) before (?:the )?(.+)$", q)
        if m:
            h = self.hist.get(m.group(1), [])
            idx = max((i for i, p in enumerate(h) if p == m.group(2)), default=None)
            return h[idx - 1] if idx else None
        m = re.match(r"what is (north|south|east|west) of (?:the )?(.+)$", q)
        if m:
            return next((x for x, d, y in reversed(self.rels) if d == m.group(1) and y == m.group(2)), None)
        m = re.match(r"what is (?:the )?(.+?) (north|south|east|west) of$", q)
        if m:
            return next((y for x, d, y in reversed(self.rels) if d == m.group(2) and x == m.group(1)), None)
        m = re.match(r"where is (?:the )?(.+)$", q)
        if m:
            return self.place(m.group(1))
        for pat, f in [(r"who gave (?:the )?(.+) to (\w+)$", lambda g, m: g[0] if g[1] == m.group(1) and g[2] == m.group(2) else None),
                       (r"who did (\w+) give (?:the )?(.+) to$", lambda g, m: g[2] if g[0] == m.group(1) and g[1] == m.group(2) else None),
                       (r"what did (\w+) give to (\w+)$", lambda g, m: g[1] if g[0] == m.group(1) and g[2] == m.group(2) else None),
                       (r"who received (?:the )?(.+)$", lambda g, m: g[2] if g[1] == m.group(1) else None),
                       (r"who gave (?:the )?(.+)$", lambda g, m: g[0] if g[1] == m.group(1) else None)]:
            m = re.match(pat, q)
            if m:
                return next((r for r in (f(x, m) for x in reversed(self.gives)) if r), None)
        return None


def check_bab(J):
    sys.path.insert(0, os.path.join(HERE, "b30b_frozen"))
    import babilong_metrics_official as OFF  # noqa: E402
    bad = 0
    print("\n## BABILong: memory answers from a plain-dict tracker; official scorer; means from the scored files")
    for L in ("128k", "512k", "1M"):
        p = os.path.join(BA, f"assembled_{L}.jsonl")
        if not os.path.exists(p):
            continue
        rows = jl(p)
        agree = okd = 0
        for r in rows:
            d = Dict()
            for e in r["events"]:
                d.apply(e[1:])
            pd = d.answer(r["question"])
            agree += pd == r["pred"]
            okd += bool(OFF.compare_answers(r["target"], str(pd), r["question"], OFF.TASK_LABELS[r["task"]])) if pd else 0
        print(f"  {L}: dict tracker gives the memory's answer on {agree}/{len(rows)}; dict accuracy (official scorer) "
              f"{100 * okd / len(rows):.1f} vs memory {J['b'][L]['memory']}")
        for tag in ("q8T", "q17T", "rdT", "q8N", "q17N", "rdN"):
            fp = os.path.join(BA, f"scored_{L}_{tag}.jsonl")
            if not os.path.exists(fp):
                continue
            sc = jl(fp)
            a = 100 * sum(x["ok"] for x in sc) / len(sc)
            ao = 100 * sum(bool(OFF.compare_answers(x["target"], x["answer"], "", OFF.TASK_LABELS[x["task"]])) for x in sc) / len(sc)
            v = J["b"][L]["arms"][tag]["acc"]
            same = abs(v - a) < 0.006
            bad += not same
            print(f"    {tag}: {a:.2f} (verdict {v} {'agrees' if same else 'DIFFERS'}); official scorer without the "
                  f"question filter {ao:.2f}")
    return bad


def main():
    sys.path.insert(0, HERE)
    import b30b_logs
    b30b_logs.ungz()
    J = json.load(open(os.path.join(HERE, "results", "b30b_scores.json")))
    bad = check_logs(J, gold2()) if J.get("a") else 0
    bad += check_bab(J) if J.get("b") else 0
    print(f"\n{bad} disagreements with the verdict")


if __name__ == "__main__":
    main()
