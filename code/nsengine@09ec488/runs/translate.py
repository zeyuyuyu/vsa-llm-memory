"""Translator prompt, parser and voting (arms A2 and A3). The few-shot examples come from the 'fewshot' split
(data/sre_fewshot.jsonl), never from dev or held-out items.

Voting (A2, k samples): field-level majority. A rack or service is kept if it appears in a majority of the parsed
samples; each of its fields (zone, hosts / replicas, quorum, needs) takes the value that a majority of those samples
give; the query takes the majority canonical value. Any field without a majority -> abstain. The engine then checks
the result (DSLError -> abstain).
"""
import collections
import json
import re
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine.sre_engine import DSLError, solve  # noqa: E402

SCHEMA = """Translate a production runbook and a question about it into JSON with exactly this schema:
{"racks": {"<rack>": {"zone": "<zone>", "hosts": ["<host>", ...]}, ...},
 "services": {"<service>": {"replicas": ["<host>", ...], "quorum": <number>, "needs": ["<service>", ...]}, ...},
 "query": one of
   {"type": "blast_radius", "failed_hosts": ["<host>", ...], "failed_racks": ["<rack>", ...]}
   {"type": "min_failure_sets", "service": "<service>"}
   {"type": "maintenance_waves", "hosts": ["<host>", ...], "max_waves": <number>}}
Copy what the text says; do not compute. "quorum" is the number of healthy replicas the text says the service needs
(a primary with an automatic standby is two replicas with quorum 1; a single instance is one replica with quorum 1;
"needs every one" is quorum equal to the number of replicas). If the text instead says how many replicas the service
can lose ("can lose up to two of them"), write "can_lose": <that number> in place of "quorum". "needs" lists the services
it depends on (empty if none). Copy every name exactly. Ignore sentences that do not affect availability. Reply with the
JSON object only."""


def dsl_of(it):
    """Gold DSL in the form the text states it: "can_lose" where the runbook says "can lose up to t of them"."""
    services = {}
    for s, v in it["spec"]["services"].items():
        v = dict(v)
        sent = next((p for p in it["text"].split("\n\n") if p.startswith(s + " ") or p.startswith("The primary of " + s + " ")), "")
        if re.search(r"can lose up to \w+ of them", sent):
            v["can_lose"] = len(v["replicas"]) - v.pop("quorum")
        services[s] = v
    return {"racks": it["spec"]["racks"], "services": services, "query": it["query"]}


def to_quorum(v):
    """A service entry with "can_lose" rewritten as "quorum" (for voting)."""
    if isinstance(v, dict) and v.get("quorum") is None and v.get("can_lose") is not None:
        try:
            v = {**v, "quorum": len(v.get("replicas") or []) - int(v["can_lose"])}
            v.pop("can_lose")
        except (TypeError, ValueError):
            pass
    return v


def prompt_translate(it, shots):
    parts = [SCHEMA]
    for k, ex in enumerate(shots, 1):
        parts.append("Example %d\n<runbook>\n%s\n</runbook>\nQuestion: %s\nJSON:\n%s" %
                     (k, ex["text"], ex["question"], json.dumps(dsl_of(ex), separators=(",", ":"))))
    parts.append("Now translate\n<runbook>\n%s\n</runbook>\nQuestion: %s\nJSON:" % (it["text"], it["question"]))
    return "\n\n".join(parts)


def load_shots(path=os.path.join(HERE, "..", "data", "sre_fewshot.jsonl")):
    items = [json.loads(l) for l in open(path)]
    want, shots = ["Q1", "Q2", "Q3"], []
    for q in want:                                   # one example per question type, the first of each
        shots.append(next(it for it in items if it["qtype"] == q))
    return shots


def parse_json(text):
    """The outermost JSON object in a reply (code fences and a leading thinking block tolerated)."""
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    i = text.find("{")
    while i >= 0:
        depth = 0
        for j in range(i, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    break
        i = text.find("{", i + 1)
    return None


def _canon(v):
    if isinstance(v, list):
        return json.dumps(sorted(str(x).strip().lower() for x in v))
    if isinstance(v, dict):
        return json.dumps({k: (_canon(x) if isinstance(x, list) else x) for k, x in sorted(v.items())}, sort_keys=True)
    return json.dumps(v)


def vote(dsls):
    """Field-level majority over parsed samples; (dsl, None) or (None, reason)."""
    dsls = [d for d in dsls if isinstance(d, dict)]
    if not dsls:
        return None, "no parsable sample"
    need = len(dsls) // 2 + 1

    def maj(values):
        c = collections.Counter(_canon(v) for v in values)
        val, n = c.most_common(1)[0]
        if n < need:
            return None
        return next(v for v in values if _canon(v) == val)

    out = {"racks": {}, "services": {}}
    for part, fields in (("racks", ("zone", "hosts")), ("services", ("replicas", "quorum", "needs"))):
        names = collections.Counter(str(k).strip().lower() for d in dsls for k in (d.get(part) or {}))
        for name, n in names.items():
            if n < need:
                continue
            entries = [to_quorum(v) if part == "services" else v for d in dsls for k, v in (d.get(part) or {}).items()
                       if str(k).strip().lower() == name and isinstance(v, dict)]
            rec = {}
            for f in fields:
                vals = [e.get(f, [] if f in ("hosts", "replicas", "needs") else None) for e in entries]
                m = maj(vals)
                if m is None and len(entries) >= need:
                    return None, "no majority for %s.%s.%s" % (part, name, f)
                rec[f] = m
            out[part][name] = rec
    q = maj([d.get("query") for d in dsls])
    if q is None:
        return None, "no majority for the query"
    out["query"] = q
    return out, None


def answer_from(dsl):
    """Engine answer for a translated DSL, or an abstention with the reason."""
    if dsl is None:
        return {"abstain": True, "reason": "no translation"}
    try:
        ans, cert = solve(dsl)
        return {"pred": ans, "cert": cert}
    except (DSLError, KeyError, TypeError, ValueError, AssertionError) as e:
        return {"abstain": True, "reason": "engine rejected the translation: %r" % (e,)}
