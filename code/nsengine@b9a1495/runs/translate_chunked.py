"""Chunked translation for large runbooks (NSR6b): the runbook is split into paragraphs (one per service, the hardware
section by rack sentence), paragraphs are translated in batches, and the partial DSLs are merged; the question is
translated on its own. Every call sees a short text, so translation cost grows linearly with the runbook and no single
output has to hold the whole topology. The engine then does all the global reasoning.

Chunk output schema (per batch): {"racks": {...}, "services": {...}} with the same fields as translate.SCHEMA.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import translate as TR  # noqa: E402

CHUNK_SCHEMA = """Translate these runbook paragraphs into JSON with exactly this schema (only what these paragraphs state):
{"racks": {"<rack>": {"zone": "<zone>", "hosts": ["<host>", ...]}, ...},
 "services": {"<service>": {"replicas": ["<host>", ...], "quorum": <number>, "needs": ["<service>", ...]}, ...}}
Copy what the text says; do not compute. "quorum" is the number of healthy replicas the text says the service needs
(a primary with an automatic standby is two replicas with quorum 1; a single instance is one replica with quorum 1;
"needs every one" is quorum equal to the number of replicas). If the text instead says how many replicas the service
can lose ("can lose up to two of them"), write "can_lose": <that number> in place of "quorum". "needs" lists the services
it depends on (empty if none). Copy every name exactly. Ignore sentences that do not affect availability. Use {} for a
part with no entries. Reply with the JSON object only."""

QUERY_SCHEMA = """Translate the question into JSON, one of:
 {"type": "blast_radius", "failed_hosts": ["<host>", ...], "failed_racks": ["<rack>", ...]}
 {"type": "min_failure_sets", "service": "<service>"}
 {"type": "maintenance_waves", "hosts": ["<host>", ...], "max_waves": <number>}
Copy every name exactly. Reply with the JSON object only."""


def paragraphs(text):
    """Rack sentences of the hardware section and service paragraphs; the notes section is dropped."""
    hw = text.split("## Hardware\n", 1)[1].split("\n\n## Services", 1)[0]
    svc = text.split("## Services\n", 1)[1].split("\n\n## Notes", 1)[0]
    racks = [s.strip() for s in re.split(r"(?<=\.)\s+(?=[A-Z]|node)", hw) if s.strip()]
    return racks + [p.strip() for p in svc.split("\n\n") if p.strip()]


def chunk_prompts(it, size=20, shot=None):
    paras = paragraphs(it["text"])
    out = []
    for i in range(0, len(paras), size):
        body = "\n\n".join(paras[i:i + size])
        ex = ""
        if shot:
            ex = "Example\n<paragraphs>\n%s\n</paragraphs>\nJSON:\n%s\n\n" % (shot[0], json.dumps(shot[1], separators=(",", ":")))
        out.append(CHUNK_SCHEMA + "\n\n" + ex + "Now translate\n<paragraphs>\n" + body + "\n</paragraphs>\nJSON:")
    return out


def query_prompt(it):
    return QUERY_SCHEMA + "\n\nQuestion: " + it["question"] + "\nJSON:"


def merge(parts, query):
    dsl = {"racks": {}, "services": {}, "query": query}
    for p in parts:
        if not isinstance(p, dict):
            return None, "a chunk did not parse"
        for k in ("racks", "services"):
            for name, v in (p.get(k) or {}).items():
                if name in dsl[k] and dsl[k][name] != v:
                    return None, "conflicting translations of %s" % name
                dsl[k][name] = v
    return dsl, None


def shot_from(items_path=os.path.join(HERE, "..", "data", "sre_fewshot.jsonl")):
    """One worked chunk example built from the first few-shot item (a rack sentence and three service paragraphs)."""
    it = json.loads(open(items_path).readline())
    paras = paragraphs(it["text"])
    gold = TR.dsl_of(it)
    rack_p = paras[0]
    svc_ps = [p for p in paras if any(p.startswith(s + " ") or p.startswith("The primary of " + s + " ") for s in gold["services"])][:3]
    racks = {r: v for r, v in gold["racks"].items() if r in rack_p.split()}
    svcs = {s: v for s, v in gold["services"].items() if any(p.startswith(s + " ") or p.startswith("The primary of " + s + " ") for p in svc_ps)}
    return "\n\n".join([rack_p] + svc_ps), {"racks": racks, "services": svcs}


def vote_chunk(samples):
    """Field-level majority over k sampled translations of one chunk (translate.vote with a placeholder query)."""
    dsls = [dict(d, query={"type": "chunk"}) for d in samples if isinstance(d, dict)]
    dsl, why = TR.vote(dsls)
    if dsl is None:
        return None, why
    dsl.pop("query", None)
    return dsl, None
