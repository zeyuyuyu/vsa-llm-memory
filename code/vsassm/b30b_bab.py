"""B30b part (b) (prereg/B30b_referee_baselines_prereg.md): BABILong notepad arm, the analogue of RULER's
facts-in-prompt arm (B24 arm F). The B18 reader (Qwen3-1.7B + the distilled LoRA student_s0, compact prompt, greedy,
three sentences of context, B18's grounding rule) re-reads the official test samples and its grounded events are kept;
the memory (b18.World) answers from them as in B18, and the same events, written as text lines (a notepad), are given
to a language model that answers instead of the memory.

Commands (CPU unless noted; the GPU reads go through b30b_gen.py --mode reader / think32k / nothink):
  size      LENGTHS                    sentences, sentences with a cue word, distinct reader prompts (no model)
  prompts   LENGTHS OUT.jsonl          the distinct reader prompts of the cue sentences of qa1-qa5 (lengths comma-separated)
  assemble  LENGTH READS[,READS] OUT   replay every sample: grounded events applied to b18.World in sentence order,
                                       the memory's answer, and the events it accepted (per-sample JSONL)
  gate      LENGTH ASSEMBLED BANKED_DIR  the memory's answers against the banked B18 per-sample results
  notepad   ASSEMBLED OUT.jsonl        the notepad prompts (one per sample)

Exactness of the cue-word shortcut: b18.grounded() drops every event whose type has no cue word in the sentence
(B18 amendment 2), so a sentence with no cue word of any type contributes no event whatever the reader writes. Only
cue sentences are read; the prompt of a sentence depends only on the text (the sentence and the three before it),
never on earlier outputs, so identical prompts are read once.
"""
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "b30b_frozen"))
import b18  # noqa: E402  (verbatim copy, b30b_frozen/b18.py)

COMPACT = "Context: {context}\nSentence: {sentence}\nEvents:"  # b18.main --compact (the student's prompt)
CTX = 3
TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]
NSAMP = {"128k": 100, "512k": 25, "1M": 25}
ALLCUES = [c for cs in b18.CUES.values() for c in cs]

NOTE = "The text has been replaced by the events read from it, one per line, in reading order."
TAIL = ("\n\nAnswer the question about the people and objects in the text above. Question: {q}\nGive only the answer: "
        "a single word (a place, a name, an object, a number word, yes / no / maybe), or a comma-separated list for "
        "'what is ... carrying' (or 'nothing').")  # b18_llm.prep, verbatim


def cue(sentence):
    words = " " + re.sub(r"[^a-z ]", " ", sentence.lower()) + " "  # b18.grounded's word string
    return any(f" {c} " in words for c in ALLCUES)


def key(p):
    return hashlib.sha1(p.encode()).hexdigest()


def sample_prompts(sents):
    for i, s in enumerate(sents):
        if cue(s):
            yield i, COMPACT.format(context=" ".join(sents[max(0, i - CTX):i]) or "(none)", sentence=s)


def samples(length, task):
    return b18.load("test100", length, task, NSAMP[length])


def render(ev):
    """One event as a line of plain English (names as the reader wrote them, lower-cased by b18.parse)."""
    op, a = ev[0], ev[1:]
    s = {"go": lambda: f"{a[0]} went to the {a[1]}.",
         "take": lambda: f"{a[0]} took the {a[1]}.",
         "drop": lambda: f"{a[0]} dropped the {a[1]}.",
         "give": lambda: f"{a[0]} gave the {a[1]} to {a[2]}.",
         "not_at": lambda: f"{a[0]} is no longer in the {a[1]}.",
         "either": lambda: f"{a[0]} is either in the {a[1]} or the {a[2]}.",
         "rel": lambda: f"the {a[0]} is {a[1]} of the {a[2]}."}[op]()
    return s[0].upper() + s[1:]


def cmd_size(lengths):
    for L in lengths.split(","):
        tot = cues = 0
        keys = set()
        for t in TASKS:
            for s in samples(L, t):
                tot += len(s["sents"])
                for i, p in sample_prompts(s["sents"]):
                    cues += 1
                    keys.add(key(p))
        print(f"{L}: {tot} sentences, {cues} with a cue word ({cues / tot:.3f}), {len(keys)} distinct prompts", flush=True)


def cmd_prompts(lengths, out):
    """Distinct reader prompts of the cue sentences of qa1-qa5 at one or more lengths (comma-separated)."""
    seen = set()
    with open(out, "w") as f:
        for L, t in [(L, t) for L in lengths.split(",") for t in TASKS]:
            for s in samples(L, t):
                for i, p in sample_prompts(s["sents"]):
                    k = key(p)
                    if k not in seen:
                        seen.add(k)
                        f.write(json.dumps({"id": k, "prompt": p}) + "\n")
    print(out, len(seen))


def cmd_assemble(L, reads, out):
    txt = {}
    for fp in reads.split(","):
        for l in open(fp):
            r = json.loads(l)
            txt[r["id"]] = r["text"]
    miss = 0
    with open(out, "w") as f:
        for t in TASKS:
            for j, s in enumerate(samples(L, t)):
                w = b18.World()
                evs, dropped, nread = [], 0, 0
                for i, p in sample_prompts(s["sents"]):
                    k = key(p)
                    if k not in txt:
                        miss += 1
                        continue
                    nread += 1
                    for ev in b18.parse(txt[k]):
                        if b18.GROUND and not b18.grounded(ev, s["sents"][i]):
                            dropped += 1
                            continue
                        if w.apply(ev):
                            evs.append([i] + ev)
                pred = w.answer(s["question"])
                f.write(json.dumps({"length": L, "task": t, "i": j, "question": s["question"], "target": s["target"],
                                    "pred": pred, "ok_official": b18.correct_official(pred, s["target"], s["question"], t),
                                    "n_sent": len(s["sents"]), "n_read": nread, "n_dropped": dropped,
                                    "events": evs}) + "\n")
    print(out, "missing reads:", miss)
    if miss:
        raise SystemExit(3)


def labels(path):
    for t, v in json.load(open(path)).items():
        b18._LABELS[t] = set(v)


def cmd_gate(L, assembled, banked_dir):
    rows = [json.loads(l) for l in open(assembled)]
    same = ok_new = ok_old = n = 0
    diff = []
    for t in TASKS:
        fs = glob.glob(f"{banked_dir}/{L}_{t}.jsonl")
        old = [json.loads(l) for l in open(fs[0])]
        new = [r for r in rows if r["task"] == t]
        assert len(old) == len(new) == NSAMP[L], (t, len(old), len(new))
        for o, r in zip(old, new):
            assert o["question"] == r["question"] and str(o["target"]) == str(r["target"])
            n += 1
            same += o["pred"] == r["pred"]
            ok_new += bool(r["ok_official"])
            ok_old += bool(o["ok_official"])
            if o["pred"] != r["pred"]:
                diff.append((t, r["i"], o["pred"], r["pred"], o["ok_official"], r["ok_official"]))
    d = 100 * (ok_new - ok_old) / n
    ok = same >= 0.99 * n and abs(d) <= 1.0
    print(f"G-B {L}: same memory answer {same}/{n}; QA1-5 re-run {100 * ok_new / n:.1f} vs banked {100 * ok_old / n:.1f} "
          f"(diff {d:+.1f}) -> {'PASS' if ok else 'FAIL'}")
    for x in diff:
        print("  differs:", x)
    return ok


def cmd_notepad(assembled, out):
    with open(out, "w") as f:
        for l in open(assembled):
            r = json.loads(l)
            lines = [render(e[1:]) for e in r["events"]] or ["(no events)"]
            p = NOTE + "\n" + "\n".join(lines) + TAIL.format(q=r["question"])
            f.write(json.dumps({"id": f"{r['length']}|{r['task']}|{r['i']}", "prompt": p, "n_lines": len(r["events"])}) + "\n")
    print(out)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "size":
        cmd_size(a[2])
    elif a[1] == "prompts":
        cmd_prompts(a[2], a[3])
    elif a[1] == "assemble":
        cmd_assemble(a[2], a[3], a[4])
    elif a[1] == "gate":
        if len(a) > 5:
            labels(a[5])
        raise SystemExit(0 if cmd_gate(a[2], a[3], a[4]) else 4)
    elif a[1] == "notepad":
        cmd_notepad(a[2], a[3])
