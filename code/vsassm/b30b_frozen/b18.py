"""B18 (prereg/B18_prereg.md): BABILong state-tracking tasks with the readable memory.

python3 b18.py --model <Qwen3> [--adapter student] --split train|test --lengths 0k,4k --tasks qa1,qa2 --n 100 --out DIR
"""
import argparse
import glob
import json
import os
import re
import time

import pandas as pd
import torch

from b9 import DH, Codebook

GROUND = True  # drop events not supported by the current sentence (development finding, B18 amendment 1)
TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10", "qa11", "qa12", "qa13"]
BENCH = os.environ.get("BENCH_ROOT", "/path/to/work/bench")  # PACE: /path/to/pace-scratch/bench
ROOT = {"test": f"{BENCH}/RMT-team__babilong-1k-samples",
        "test_long": f"{BENCH}/RMT-team__babilong",
        "train": f"{BENCH}/RMT-team__babilong-train-5k-samples"}

PROMPT_F = """Task: keep a record of where people and things are, updated one sentence at a time.

Events (as JSON lists):
["go", person, place]               the person is now in that place (went, moved, travelled, journeyed, is in)
["take", person, object]            the person now carries the object (got, took, picked up, grabbed)
["drop", person, object]            the person no longer carries it; it stays where the person is (left, dropped, put down, discarded)
["give", giver, object, receiver]   the receiver now carries the object (gave, handed, passed)
["not_at", person, place]           the person is no longer in / is not in that place
["either", person, place, place]    the person is either in one place or the other
["rel", place, direction, place]    the first place is north / south / east / west of the second

Examples
Context: Mary went to the kitchen.
Sentence: Afterwards she and John journeyed to the garden.
Output: {{"events": [["go", "Mary", "garden"], ["go", "John", "garden"]]}}

Context: John moved to the office.
Sentence: John grabbed the apple there and then handed it to Sandra.
Output: {{"events": [["take", "John", "apple"], ["give", "John", "apple", "Sandra"]]}}

Context: (none)
Sentence: Bill is either in the park or the school.
Output: {{"events": [["either", "Bill", "park", "school"]]}}

Context: (none)
Sentence: The garden is south of the office.
Output: {{"events": [["rel", "garden", "south", "office"]]}}

Context: The rain had not stopped.
Sentence: She looked out of the window and thought of her father's old house in the country.
Output: {{"events": []}}

Now
Context: {context}
Sentence: {sentence}
Rules: only the events stated in the sentence; replace pronouns with names; "they" means the people of the last
event; sentences that are not about these people moving or carrying things have no events; names without "the".
Output:"""

EVENT = re.compile(r'\[\s*"(go|take|drop|give|not_at|either|rel)"((?:\s*,\s*"[^"]*")+)\s*\]', re.I)


def parse(txt):
    out = []
    first = re.split(r"\n\s*\n|```", txt.strip())
    block = next((b for b in first if EVENT.search(b) or '"events"' in b), "")
    for m in EVENT.finditer(block):
        args = re.findall(r'"([^"]*)"', m.group(2))
        out.append([m.group(1).lower()] + [a.strip().lower().removeprefix("the ").strip() for a in args])
    return out


CUES = {  # general English cue words per event type (not taken from any test data)
    "go": ["went", "go", "goes", "gone", "going", "moved", "move", "moves", "travelled", "traveled", "travels",
           "journeyed", "journeys", "walked", "ran", "came", "come", "returned", "headed", "is in", "are in",
           "was in", "were in", "back to", "entered", "arrived"],
    "take": ["got", "get", "gets", "took", "take", "takes", "picked", "grabbed", "grabs", "collected", "fetched",
             "lifted"],
    "drop": ["left", "leaves", "dropped", "drops", "put down", "puts down", "discarded", "discards", "set down",
             "placed", "put", "threw"],
    "give": ["gave", "give", "gives", "handed", "hands", "passed", "passes", "offered"],
    "not_at": ["no longer", "not in", "isn t in", "is not", "not"],
    "either": ["either"],
    "rel": ["north", "south", "east", "west"],
}


def grounded(ev, sentence):
    """An event must be supported by the current sentence (the reader sometimes restates events from the context):
    the place(s) of go / not_at / either, or the object of take / drop / give (or a pronoun standing for it), must occur
    in the sentence."""
    words = " " + re.sub(r"[^a-z ]", " ", sentence.lower()) + " "
    has = lambda x: f" {x} " in words
    op, a = ev[0], ev[1:]
    if not any(f" {c} " in words for c in CUES.get(op, ())):  # the event type needs a cue word (amendment 2)
        return False
    if op in ("go", "not_at"):
        return len(a) == 2 and has(a[1])
    if op == "either":
        return len(a) == 3 and has(a[1]) and has(a[2])
    if op == "rel":
        return len(a) == 3 and has(a[0]) and has(a[2]) and has(a[1])
    if op in ("take", "drop"):
        return len(a) == 2 and (has(a[1]) or has("it") or has("them"))
    if op == "give":
        return len(a) == 3 and (has(a[1]) or has("it") or has("them"))
    return False


def sentences(text):
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+|\n{2,}", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]


class World:
    """Holder / place of every entity in a readable vector memory (delta-rule writes, codes per name, cleanup reads),
    plus plain logs for history and gives."""

    def __init__(self):
        self.K, self.V = Codebook("key"), Codebook("val")
        self.S = torch.zeros(DH, DH)
        self.persons, self.objects, self.carry_order = set(), set(), {}
        self.history, self.gives, self.either = {}, [], {}
        self.rels = []  # qa4: (place, direction, place)

    def _write(self, x, y):
        k, v = self.K.get(x), self.V.get(y)
        self.S = self.S + torch.outer(v - self.S @ k, k)

    def parent(self, x):
        k = self.K.get(x, create=False)
        if k is None:
            return None
        r = self.S @ k
        if r.norm() < 0.5:
            return None
        val, _ = self.V.cleanup(r)
        return None if val == "__unknown__" else val

    def place(self, x):
        cur, seen = x, 0
        while seen < 10:
            p = self.parent(cur)
            if p is None:
                return None if cur == x else (cur if cur not in self.persons and cur not in self.objects else None)
            if p not in self.persons and p not in self.objects:
                return p
            cur, seen = p, seen + 1
        return None

    def apply(self, ev):
        op, a = ev[0], ev[1:]
        if op == "go" and len(a) == 2:
            self.persons.add(a[0])
            self._write(a[0], a[1])
            self.either.pop(a[0], None)
        elif op == "take" and len(a) == 2:
            self.persons.add(a[0]); self.objects.add(a[1])
            self._write(a[1], a[0])
            self.carry_order[a[1]] = len(self.carry_order)
        elif op == "drop" and len(a) == 2:
            self.objects.add(a[1])
            here = self.place(a[0])
            self._write(a[1], here or "__unknown__")
        elif op == "give" and len(a) == 3:
            self.persons.update([a[0], a[2]]); self.objects.add(a[1])
            self._write(a[1], a[2])
            self.carry_order[a[1]] = len(self.carry_order)
            self.gives.append((a[0], a[1], a[2]))
        elif op == "not_at" and len(a) == 2:
            self.persons.add(a[0])
            if self.place(a[0]) == a[1] or self.place(a[0]) is None:
                self._write(a[0], "__unknown__")
            self.either.pop(a[0], None)
        elif op == "rel" and len(a) == 3:
            d = a[1].replace(" of", "").strip()
            opp = {"north": "south", "south": "north", "east": "west", "west": "east"}.get(d)
            self.rels.append((a[0], d, a[2]))
            if opp:  # "A is south of B" also means "B is north of A"
                self.rels.append((a[2], opp, a[0]))
            return True
        elif op == "either" and len(a) == 3:
            self.persons.add(a[0])
            self._write(a[0], "__unknown__")
            self.either[a[0]] = {a[1], a[2]}
        else:
            return False
        for e in self.persons | self.objects:  # history of each entity's place (for "where was X before Y")
            p = self.place(e)
            h = self.history.setdefault(e, [])
            if p and (not h or h[-1] != p):
                h.append(p)
        return True

    def answer(self, q):
        q = q.strip().rstrip("?").lower()
        num = ["none", "one", "two", "three", "four", "five", "six"]
        m = re.match(r"where was (?:the )?(.+?) before (?:the )?(.+)$", q)
        if m:
            h = self.history.get(m.group(1), [])
            idx = max((i for i, p in enumerate(h) if p == m.group(2)), default=None)
            return h[idx - 1] if idx else None
        m = re.match(r"what is (north|south|east|west) of (?:the )?(.+)$", q)
        if m:
            return next((a for a, d, b in reversed(self.rels) if d == m.group(1) and b == m.group(2)), None)
        m = re.match(r"what is (?:the )?(.+?) (north|south|east|west) of$", q)
        if m:
            return next((b for a, d, b in reversed(self.rels) if d == m.group(2) and a == m.group(1)), None)
        m = re.match(r"where is (?:the )?(.+)$", q)
        if m:
            return self.place(m.group(1))
        m = re.match(r"is (\w+) in the (.+)$", q)
        if m:
            p, c = m.group(1), m.group(2)
            if p in self.either:
                return "maybe" if c in self.either[p] else "no"
            return "yes" if self.place(p) == c else "no"
        m = re.match(r"how many objects is (\w+) carrying$", q)
        if m:
            n = sum(1 for o in self.objects if self.parent(o) == m.group(1))
            return num[n] if n < len(num) else str(n)
        m = re.match(r"what is (\w+) carrying$", q)
        if m:
            held = sorted((o for o in self.objects if self.parent(o) == m.group(1)), key=lambda o: self.carry_order.get(o, 0))
            return ",".join(held) if held else "nothing"
        for pat, pick in [(r"who gave (?:the )?(.+) to (\w+)$", lambda g, m: g[0] if g[1] == m.group(1) and g[2] == m.group(2) else None),
                          (r"who did (\w+) give (?:the )?(.+) to$", lambda g, m: g[2] if g[0] == m.group(1) and g[1] == m.group(2) else None),
                          (r"what did (\w+) give to (\w+)$", lambda g, m: g[1] if g[0] == m.group(1) and g[2] == m.group(2) else None),
                          (r"who received (?:the )?(.+)$", lambda g, m: g[2] if g[1] == m.group(1) else None),
                          (r"who gave (?:the )?(.+)$", lambda g, m: g[0] if g[1] == m.group(1) else None)]:
            m = re.match(pat, q)
            if m:
                return next((r for r in (pick(g, m) for g in reversed(self.gives)) if r), None)
        return None


_LABELS = {}


def labels(task):
    """Answer labels of a task, as in the official BABILong metric: every answer seen in the training split."""
    if task in ("qa11", "qa12", "qa13"):  # no training data; their official labels are the room names, as qa1
        task = "qa1"
    if task not in _LABELS:
        out = set()
        for f in glob.glob(f"{ROOT['train']}/data/{task}/*.json"):
            for r in json.load(open(f)):
                out.update(x.strip().lower() for x in str(r["target"]).split(","))
        _LABELS[task] = out
    return _LABELS[task]


def correct_official(pred, target, question, task):
    """Official BABILong rule (babilong/metrics.py compare_answers): labels mentioned in the question are ignored; a
    single target must be the only label in the output; a list target needs all its parts and no other label."""
    if pred is None:
        return False
    out = str(pred).lower().split(".")[0]
    q = question.lower()
    labs = {l for l in labels(task) if not re.search(r"\b" + re.escape(l) + r"\b", q)}
    found = {l for l in labs if re.search(r"\b" + re.escape(l) + r"\b", out)}
    tgt = {x.strip() for x in str(target).lower().split(",")}
    return tgt <= found and not (found - tgt) if len(tgt) > 1 else found == tgt


def correct(pred, target):
    t = str(target).strip().lower()
    if pred is None:
        return False
    p = str(pred).strip().lower()
    if "," in t or "," in p:
        return set(x.strip() for x in p.split(",")) == set(x.strip() for x in t.split(","))
    return p == t or {p, t} <= {"none", "nothing"}


def run(reader, samples, record=None, ctx_n=3, template=None):
    """samples: [{"sents": [...], "question": ...}] -> (worlds, stats); sentence i of every sample in one batch."""
    worlds = [World() for _ in samples]
    stats = {"calls": 0, "tokens": 0, "events": 0}
    for i in range(max(len(s["sents"]) for s in samples)):
        live = [j for j, s in enumerate(samples) if i < len(s["sents"])]
        prompts = [(template or PROMPT_F).format(context=" ".join(samples[j]["sents"][max(0, i - ctx_n):i]) or "(none)",
                                                 sentence=samples[j]["sents"][i]) for j in live]
        for j, p, (txt, n) in zip(live, prompts, reader(prompts)):
            stats["calls"] += 1
            stats["tokens"] += n
            if record is not None:
                record.append({"prompt": p, "output": txt})
            for ev in parse(txt):
                if GROUND and not grounded(ev, samples[j]["sents"][i]):
                    stats["dropped"] = stats.get("dropped", 0) + 1
                    continue
                stats["events"] += worlds[j].apply(ev)
    return worlds, stats


def run_blocked(reader, sample, block=2048, ctx_n=3, template=None, log=None):
    """One sample, sentences read in blocks: a prompt depends only on the text (the sentence and the ctx_n before it),
    never on earlier outputs, so a block is one batch and its events are applied in sentence order. B18 amendment 8:
    run() batches sentence i across samples, which leaves 5 prompts per batch at 10M (n = 5) and timed out."""
    w = World()
    stats = {"calls": 0, "tokens": 0, "events": 0}
    sents, t0 = sample["sents"], time.time()
    for b in range(0, len(sents), block):
        idx = range(b, min(b + block, len(sents)))
        prompts = [(template or PROMPT_F).format(context=" ".join(sents[max(0, i - ctx_n):i]) or "(none)",
                                                 sentence=sents[i]) for i in idx]
        for i, (txt, n) in zip(idx, reader(prompts)):
            stats["calls"] += 1
            stats["tokens"] += n
            for ev in parse(txt):
                if GROUND and not grounded(ev, sents[i]):
                    stats["dropped"] = stats.get("dropped", 0) + 1
                    continue
                stats["events"] += w.apply(ev)
        if log and (b // block) % 25 == 0:
            log(f"  {idx.stop}/{len(sents)} sentences  events {stats['events']}  {time.time() - t0:.0f}s")
    return w, stats


def load(split, length, task, n, offset=0):
    if split == "test100":  # RMT-team/babilong: data/<task>/<length>.json, 100 samples, all tasks, 0k ... 1M
        rows = json.load(open(f"{ROOT['test_long']}/data/{task}/{length}.json"))[offset:offset + n]
        return [{"sents": sentences(r["input"]), "question": r["question"], "target": r["target"]} for r in rows]
    if split == "train":  # RMT-team/babilong-train-5k-samples: data/<task>/<length>.json (qa1-qa10 only)
        rows = json.load(open(f"{ROOT['train']}/data/{task}/{length}.json"))[offset:offset + n]
        return [{"sents": sentences(r["input"]), "question": r["question"], "target": r["target"]} for r in rows]
    f = glob.glob(f"{ROOT[split]}/{length}/{task}-*.parquet") or glob.glob(f"{ROOT[split]}/{task}/{length}*.parquet") \
        or glob.glob(f"{ROOT[split]}/{task}/{length}/*.parquet")
    d = pd.read_parquet(f[0])
    return [{"sents": sentences(r["input"]), "question": r["question"], "target": r["target"]} for _, r in d.head(n).iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--lengths", default="0k")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=0)  # characters per batch (0 = fixed batches)
    ap.add_argument("--offset", type=int, default=0)  # skip the samples used for distillation (dev on train split)
    ap.add_argument("--compact", action="store_true")  # the fine-tuned student's prompt (b18_distill.COMPACT)
    ap.add_argument("--block", type=int, default=0)  # amendment 8: read one sample at a time in sentence blocks
    args = ap.parse_args()
    import b14
    reader = b14.Reader(args.model, args.adapter or None)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for length in args.lengths.split(","):
        for task in args.tasks.split(","):
            samples = load(args.split, length, task, args.n, args.offset)
            t0 = time.time()
            tpl = "Context: {context}\nSentence: {sentence}\nEvents:" if args.compact else None
            if args.block:  # per-sample checkpoint: a finished sample is never read again
                part = f"{args.out}/{length}_{task}.partial.jsonl"
                done = [json.loads(l) for l in open(part)] if os.path.exists(part) else []
                stats = {"calls": 0, "tokens": 0, "events": 0}
                for k, s in enumerate(samples):
                    if k < len(done):
                        continue
                    print(f"{length} {task} sample {args.offset + k}: {len(s['sents'])} sentences", flush=True)
                    w, st = run_blocked(lambda ps: reader(ps, bs=args.bs, budget=args.budget), s, block=args.block,
                                        template=tpl, log=lambda m: print(m, flush=True))
                    pred = w.answer(s["question"])
                    r = {"question": s["question"], "target": s["target"], "pred": pred,
                         "ok": correct(pred, s["target"]),
                         "ok_official": correct_official(pred, s["target"], s["question"], task), **st}
                    with open(part, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    done.append(r)
                    print(f"  -> pred {pred!r} target {s['target']!r} ok {r['ok_official']}", flush=True)
                rows = done
                for r in rows:
                    for key in stats:
                        stats[key] += r.get(key, 0)
                acc = sum(r["ok_official"] for r in rows) / len(rows)
                summary[f"{length}/{task}"] = {"acc": round(acc, 3), "n": len(rows), **stats,
                                               "secs": round(time.time() - t0)}
                with open(f"{args.out}/{length}_{task}.jsonl", "w") as f:
                    for r in rows:
                        f.write(json.dumps({k: r[k] for k in ("question", "target", "pred", "ok", "ok_official")}) + "\n")
                print(f"{length} {task}: {acc:.3f} ({len(rows)})  calls {stats['calls']}  {time.time() - t0:.0f}s",
                      flush=True)
                json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
                continue
            worlds, stats = run(lambda ps: reader(ps, bs=args.bs, budget=args.budget), samples, template=tpl)
            rows = []
            for s, w in zip(samples, worlds):
                pred = w.answer(s["question"])
                rows.append({"question": s["question"], "target": s["target"], "pred": pred,
                             "ok": correct(pred, s["target"]),
                             "ok_official": correct_official(pred, s["target"], s["question"], task)})
            acc = sum(r["ok_official"] for r in rows) / len(rows)
            summary[f"{length}/{task}"] = {"acc": round(acc, 3), "n": len(rows), **stats,
                                           "secs": round(time.time() - t0)}
            with open(f"{args.out}/{length}_{task}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"{length} {task}: {acc:.3f} ({len(rows)})  calls {stats['calls']}  {time.time() - t0:.0f}s", flush=True)
            json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
