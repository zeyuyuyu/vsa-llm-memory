"""B14 (prereg/B14_prereg.md): a prompted Qwen3 reads one sentence at a time and writes events into the readable memory.

python3 b14.py accept --model <snapshot> --prompt A --out b14_preds/E1.7_A.jsonl            # acceptance set
python3 b14.py long   --model <snapshot> --prompt A --out b14_long/E1.7_A.jsonl              # long model-written stories
"""
import argparse
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from b9 import DH, Codebook
from readable_memory import WHAT_IN, split_sentences
from score_b13 import norm

PROMPTS = {
    ("A", "locations"): """You keep track of where things are. Read the new sentence (the earlier sentences are only context) and list the changes it states.

Earlier sentences:
{context}

Known objects: {objs}
Known places: {places}

New sentence: {sentence}

List, in order, every event in the new sentence:
- ["move", object, place] when an object ends up in or on a place (put, moved, left, hung, stashed, stayed, was found in, and so on), and also when the sentence only says where an object is;
- ["swap", place, place] when everything in one place and everything in another place trade places.
Do not list plans, wishes, negated or hypothetical actions, or the place an object was taken from. Replace pronouns with the names they refer to. Use short names without "the", "his", "her"; reuse a known name when the sentence refers to the same thing.
Return only JSON: {{"events": [...]}}""",
    ("A", "containers"): """You keep track of what is inside what. Read the new sentence (the earlier sentences are only context) and list the changes it states.

Earlier sentences:
{context}

Known items: {objs}

New sentence: {sentence}

List, in order, every event in the new sentence: ["move", item, container] when an item is put into a container (a bag, box, pocket, drawer, car and so on), and also when the sentence only says which container an item is in. Moving a container moves everything inside it; list only the container's own move.
Do not list plans, wishes, negated or hypothetical actions, or the container an item was taken out of. Replace pronouns with the names they refer to. Use short names without "the", "his", "her"; reuse a known name when the sentence refers to the same thing.
Return only JSON: {{"events": [...]}}""",
    ("B", "locations"): """Task: update a record of object locations from a story, one sentence at a time.

Example
Context: Ann put the keys in the drawer.
Sentence: Later she took them out, answered the phone, and left them on the shelf next to the radio.
Output: {{"events": [["move", "keys", "shelf"]]}}

Context: {context}
Objects seen so far: {objs}. Places seen so far: {places}.
Sentence: {sentence}
Output the events of this sentence only, as JSON {{"events": [...]}}, each ["move", object, place] or ["swap", place, place]. A move means the object is now in or on that place; statements of where something is count too. A swap means the contents of two places are exchanged. Skip anything planned, negated or not done, and the place something came from. Write names instead of pronouns, without "the" or "his".""",
    ("B", "containers"): """Task: update a record of which container holds each item, one sentence at a time.

Example
Context: Ann put the ring in the box.
Sentence: Then she put the box in her bag, zipped it and left.
Output: {{"events": [["move", "box", "bag"]]}}

Context: {context}
Items seen so far: {objs}.
Sentence: {sentence}
Output the events of this sentence only, as JSON {{"events": [["move", item, container], ...]}}. A move means the item is now directly inside that container; statements of where something is count too. Skip anything planned, negated or not done, and the container something was taken out of. Write names instead of pronouns, without "the" or "his".""",
    ("C", "locations"): """Task: keep a record of where objects are, updated one sentence at a time.

Examples
Context: (none)
Sentence: Maria put the stapler in the cupboard and then answered the phone.
Output: {{"events": [["move", "stapler", "cupboard"]]}}

Context: Ben put the keys in the drawer.
Sentence: Later he took them out of the drawer and left them on the shelf by the window.
Output: {{"events": [["move", "keys", "shelf"]]}}

Context: Sam made coffee.
Sentence: The tape was still on the windowsill, but he moved the scissors and the glue to the desk.
Output: {{"events": [["move", "tape", "windowsill"], ["move", "scissors", "desk"], ["move", "glue", "desk"]]}}

Context: Kim put the plates in the left cupboard and the cups in the right cupboard.
Sentence: Then she swapped what the two cupboards held and dropped her keys in the bowl on the counter.
Output: {{"events": [["swap", "left cupboard", "right cupboard"], ["move", "keys", "bowl"]]}}

Context: Tom was tidying up.
Sentence: He thought about putting the lamp in the attic, but walked over to the window and opened it.
Output: {{"events": []}}

Now
Context: {context}
Objects seen so far: {objs}. Places seen so far: {places}.
Sentence: {sentence}
Rules: a move means the object is now in or on that place, including statements of where it is; give the most specific place (the bowl, not the counter it stands on); a swap only when the contents of two places are exchanged; nothing for plans, negated actions, or people walking somewhere; replace pronouns with names; reuse a known name for the same thing; names without "the", "his", "her".
Output:""",
    ("C", "containers"): """Task: keep a record of which container directly holds each item, updated one sentence at a time.

Examples
Context: (none)
Sentence: Ann put the ring in a small box and the box in her bag, then called her sister.
Output: {{"events": [["move", "ring", "small box"], ["move", "small box", "bag"]]}}

Context: Ann put the ring in a small box and the box in her bag.
Sentence: At the hotel she took the box out of the bag and locked it in the safe.
Output: {{"events": [["move", "small box", "safe"]]}}

Context: Joe was packing.
Sentence: The coins were in a jar, which he had wanted to put in the suitcase but did not.
Output: {{"events": [["move", "coins", "jar"]]}}

Now
Context: {context}
Items seen so far: {objs}.
Sentence: {sentence}
Rules: a move means the item is now directly inside that container, including statements of where it is; moving a container moves its contents, so list only the container's own move; nothing for plans or negated actions, and not the container something was taken out of; replace pronouns with names; reuse a known name for the same thing; names without "the", "his", "her".
Output:""",
}


PROMPTS[("D", "locations")] = PROMPTS[("C", "locations")].replace("""
Now
Context:""", """
Context: Whatever had been in the red box and whatever had been in the blue box traded places.
Sentence: Before that, the red box held the cards and the blue box held the dice.
Output: {{"events": [["earlier", "cards", "red box"], ["earlier", "dice", "blue box"]]}}

Context: Ben came home in the rain.
Sentence: He grabbed the keys from the hook and gave the umbrella to Ana.
Output: {{"events": [["move", "keys", "Ben"], ["move", "umbrella", "Ana"]]}}

Now
Context:""").replace(
    "a swap only when the contents of two places are exchanged;",
    "a swap only when the contents of two places are exchanged; \"earlier\" for a fact that was already true before "
    "the events told so far; a person who takes or holds an object is its place;")
PROMPTS[("E", "locations")] = PROMPTS[("C", "locations")].replace("""
Now
Context:""", """
Context: Whatever had been in the red box and whatever had been in the blue box traded places.
Sentence: Before that, the red box held the cards and the blue box held the dice.
Output: {{"events": [["earlier", "cards", "red box"], ["earlier", "dice", "blue box"]]}}

Now
Context:""").replace(
    "a swap only when the contents of two places are exchanged;",
    "a swap only when the contents of two places are exchanged; \"earlier\" for a fact that was already true before "
    "the events told so far;")
PROMPTS[("E", "containers")] = PROMPTS[("C", "containers")]
PROMPTS[("D", "containers")] = PROMPTS[("C", "containers")].replace("""
Now
Context:""", """
Context: Tom packed for the trip.
Sentence: He put his wallet in the jacket and handed the jacket to Sue.
Output: {{"events": [["move", "wallet", "jacket"], ["move", "jacket", "Sue"]]}}

Now
Context:""").replace(
    "a move means the item is now directly inside that container,",
    "a move means the item is now directly inside that container or held by that person,")


class Reader:
    def __init__(self, path, adapter=None):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                          attn_implementation="sdpa").eval()
        if adapter:  # B16: a distilled LoRA student
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload().eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        # greedy; installed on the model because transformers overrides a passed config's defaults (see b6_qwen.py)
        self.model.generation_config = GenerationConfig(max_new_tokens=200, do_sample=False, eos_token_id=eos,
                                                        pad_token_id=self.tok.pad_token_id)
        self.eos = set(eos)

    @torch.no_grad()
    def __call__(self, prompts, bs=64, budget=0):
        if budget:  # B18: batches by length so a few very long inputs cannot exhaust memory; output order kept
            order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
            res, k = [None] * len(prompts), 0
            while k < len(order):
                n = 1
                while k + n < len(order) and n < bs and len(prompts[order[k + n]]) * (n + 1) <= budget:
                    n += 1
                for i, o in zip(order[k:k + n], self(prompts=[prompts[j] for j in order[k:k + n]], bs=n)):
                    res[i] = o
                k += n
            return res
        outs, self.spans = [], []  # B17: self.spans[i] = [(start, end, prob)] per generated token of output i
        for k in range(0, len(prompts), bs):
            texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=getattr(self, "think", False))
                     for p in prompts[k:k + bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
            if not B17:
                gen = self.model.generate(**enc)[:, enc.input_ids.shape[1]:].tolist()
                for ids in gen:
                    n = next((i + 1 for i, t in enumerate(ids) if t in self.eos), len(ids))
                    outs.append((self.tok.decode(ids[:n], skip_special_tokens=True), n))
                continue
            res = self.model.generate(**enc, output_scores=True, return_dict_in_generate=True)
            gen = res.sequences[:, enc.input_ids.shape[1]:]
            probs = torch.stack([sc.float().softmax(-1) for sc in res.scores], 1)  # (batch, steps, vocab)
            chosen = probs.gather(2, gen[:, :probs.shape[1], None])[..., 0].tolist()
            for ids, ps in zip(gen.tolist(), chosen):
                n = next((i + 1 for i, t in enumerate(ids) if t in self.eos), len(ids))
                spans, prev = [], ""
                for t in range(n):  # character span of every generated token, with its probability
                    cur = self.tok.decode(ids[:t + 1], skip_special_tokens=True)
                    spans.append((len(prev), len(cur), ps[t]))
                    prev = cur
                self.spans.append(spans)
                outs.append((prev, n))
        return outs


TRIPLE = re.compile(r'\[\s*"(move|swap)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]', re.I)
TRIPLE17 = re.compile(r'\[\s*"(move|swap|earlier)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]', re.I)
B17 = False  # B17: "earlier" facts, event log + replay, reading confidence


def parse17(txt, spans=None):
    """B17: like parse(), with "earlier" facts and, when token spans are given, each event's confidence = the product
    of the probabilities of the tokens inside its triple."""
    first = None
    for part in re.finditer(r"(?:(?!\n\s*\n|```).)+", txt, re.S):  # first answer block, as in parse()
        if TRIPLE17.search(part.group(0)) or '"events"' in part.group(0):
            first = (part.start(), part.end())
            break
    if first is None:
        return None if "events" not in txt else []
    out = []
    for m in TRIPLE17.finditer(txt, first[0], first[1]):
        op, a, b = m.group(1).lower(), m.group(2), m.group(3)
        if a.strip().lower() in NOWHERE or b.strip().lower() in NOWHERE:
            continue
        c = 1.0
        for s0, s1, p in spans or []:
            if s1 > m.start() and s0 < m.end():
                c *= p
        out.append([op, a, b, round(c, 4)])
    return out
NOWHERE = {"", "unknown", "none", "n/a", "nowhere", "somewhere", "null", "elsewhere"}


def parse(txt):
    """Every ["move"|"swap", a, b] triple in the first answer block (the model sometimes drops the outer brackets of
    the list, then repeats its answer in a code block). Moves to "unknown" / "somewhere" are not events."""
    blocks = [b for b in re.split(r"\n\s*\n|```", txt) if '"events"' in b or TRIPLE.search(b)]
    if not blocks:
        return None if "events" not in txt else []
    return [[op.lower(), a, b] for op, a, b in TRIPLE.findall(blocks[0]) if b.strip().lower() not in NOWHERE
            and a.strip().lower() not in NOWHERE]


MERGE = "suffix"  # B14 rule; B15 dev: "none" (no merging) is compared
GUARD = False  # post-B15 option: in a sentence with a swap, drop moves into either swapped place (the swap covers them)


def swap_guard(evs):
    if not GUARD:
        return evs
    sw = {norm(x) for e in evs if e[0].lower() == "swap" for x in e[1:]}
    return [e for e in evs if e[0].lower() == "swap" or norm(e[2]) not in sw]


def canon(name, known):
    """Normalise a name; merge it with a known name when one is a whole-word suffix of the other (and unique)."""
    n = norm(name)
    if not n or n in known or MERGE == "none":
        return n or None
    c = [k for k in known if k.endswith(" " + n) or n.endswith(" " + k)]
    return c[0] if len(c) == 1 else n


class Memory:
    def __init__(self, domain):
        self.domain = domain
        self.K, self.V = Codebook("key"), Codebook("val")
        self.S = torch.zeros(DH, DH)
        self.objs, self.places = [], []  # known names; containers use objs for every item

    def apply(self, ev):
        op, a, b = ev
        if op == "swap" and self.domain == "locations":
            a, b = canon(a, self.places), canon(b, self.places)
            if not a or not b or a == b:
                return None
            for n in (a, b):
                if n not in self.places:
                    self.places.append(n)
            wv = F.normalize(self.V.get(a) - self.V.get(b), dim=0)
            self.S = self.S - 2 * torch.outer(wv, wv @ self.S)
            return ("swap", a, b)
        if op != "move":
            return None
        a = canon(a, self.objs)
        b = canon(b, self.objs if self.domain == "containers" else self.places)
        if not a or not b or a == b:
            return None
        if a not in self.objs:
            self.objs.append(a)
        lst = self.objs if self.domain == "containers" else self.places
        if b not in lst:
            lst.append(b)
        k, v = self.K.get(a), self.V.get(b)
        self.S = self.S + torch.outer(v - self.S @ k, k)
        return ("write", a, b)

    def read(self, name):
        k = self.K.get(name, create=False)
        if k is None:
            return None, 0.0
        r = self.S @ k
        if r.norm() < 0.5:
            return None, 0.0
        val, margin = self.V.cleanup(r)
        return val, round(margin, 3)

    def where(self, name):
        if self.domain == "locations":
            return self.read(name)
        hops, cur, margins = [], name, []
        for _ in range(40):
            val, m = self.read(cur)
            if val is None or val in hops:
                break
            hops.append(val)
            margins.append(m)
            cur = val
        return hops, (min(margins) if margins else 0.0)

    def contents(self, place):
        items = [(o, self.read(o)) for o in self.K.names]
        inside = [(o, m) for o, (v, m) in items if v == place]
        return [o for o, _ in inside], min([m for _, m in inside], default=1.0)

    def ask(self, q):
        qn = " " + norm(q.rstrip("?")) + " "
        what_in = bool(WHAT_IN.search(q)) and self.domain == "locations"
        pools = [self.places, self.objs] if what_in else [self.objs, self.places]
        for pool in pools:
            hit = sorted((n for n in pool if f" {n} " in qn), key=len, reverse=True)
            if hit:
                return (self.contents(hit[0]) if what_in else self.where(hit[0])) + (hit[0],)
        return (None, 0.0, None)


class Memory17(Memory):
    """B17: keeps the event log; an "earlier" fact is inserted before the most recent swap involving its place (else
    before the most recent event involving its object, else appended); the state is rebuilt by replaying the log (codes
    are kept, so replay is exact). Each object carries the confidence of the events that put it where it is."""

    def __init__(self, domain):
        super().__init__(domain)
        self.log, self.conf = [], {}

    def apply(self, ev):
        op, a, b = ev[:3]
        c = ev[3] if len(ev) > 3 else 1.0
        if op == "swap" and self.domain == "locations":
            a, b = canon(a, self.places), canon(b, self.places)
            if not a or not b or a == b:
                return None
            for n in (a, b):
                if n not in self.places:
                    self.places.append(n)
            e = ("swap", a, b, c)
        elif op in ("move", "earlier"):
            a = canon(a, self.objs)
            b = canon(b, self.objs if self.domain == "containers" else self.places)
            if not a or not b or a == b:
                return None
            if a not in self.objs:
                self.objs.append(a)
            lst = self.objs if self.domain == "containers" else self.places
            if b not in lst:
                lst.append(b)
            e = ("write", a, b, c)
        else:
            return None
        if op == "earlier":
            pos = next((i for i in range(len(self.log) - 1, -1, -1) if self.log[i][0] == "swap" and b in self.log[i][1:3]),
                       None)
            if pos is None:
                pos = next((i for i in range(len(self.log) - 1, -1, -1) if self.log[i][1] == a), len(self.log))
            self.log.insert(pos, e)
        else:
            self.log.append(e)
        self.replay()
        return e[:3] if op != "earlier" else ("earlier", a, b)

    def replay(self):
        self.S = torch.zeros(DH, DH)
        loc, self.conf = {}, {}
        for op, a, b, c in self.log:
            if op == "swap":
                wv = F.normalize(self.V.get(a) - self.V.get(b), dim=0)
                self.S = self.S - 2 * torch.outer(wv, wv @ self.S)
                for o, p in list(loc.items()):
                    if p in (a, b):
                        loc[o] = b if p == a else a
                        self.conf[o] = self.conf.get(o, 1.0) * c
            else:
                k, v = self.K.get(a), self.V.get(b)
                self.S = self.S + torch.outer(v - self.S @ k, k)
                loc[a], self.conf[a] = b, c

    def read(self, name):
        val, margin = super().read(name)
        return val, (round(min(margin, self.conf.get(name, 1.0)), 4) if val is not None else margin)


def run_stories(reader, stories, prompt, record=None):
    """stories: [{"domain", "sentences"}]. Reads sentence i of every story in one batch. Returns per story the memory,
    its events and reader statistics."""
    mems = [(Memory17 if B17 else Memory)(s["domain"]) for s in stories]
    logs = [{"events": [], "tokens": 0, "calls": 0, "parse_fail": 0} for _ in stories]
    for i in range(max(len(s["sentences"]) for s in stories)):
        live = [j for j, s in enumerate(stories) if i < len(s["sentences"])]
        prompts = []
        for j in live:
            s, m = stories[j], mems[j]
            ctx = " ".join(s["sentences"][max(0, i - 3):i]) or "(none)"
            prompts.append(PROMPTS[(prompt, s["domain"])].format(
                context=ctx, sentence=s["sentences"][i], objs=", ".join(m.objs) or "(none)",
                places=", ".join(m.places) or "(none)"))
        outs = reader(prompts)
        spans = getattr(reader, "spans", None) if B17 else None
        for q, (j, p, (txt, n)) in enumerate(zip(live, prompts, outs)):
            if record is not None:  # B16: teacher examples (prompt, output)
                record.append({"story": j, "domain": stories[j]["domain"], "prompt": p, "output": txt})
            evs = parse17(txt, spans[q] if spans else None) if B17 else parse(txt)
            logs[j]["tokens"] += n
            logs[j]["calls"] += 1
            if evs is None:
                logs[j]["parse_fail"] += 1
                continue
            if B17 and any(e[0] == "earlier" for e in evs):  # a restated swap next to earlier facts is context
                last = next((l for l in reversed(mems[j].log) if l[0] == "swap"), None)
                evs = [e for e in evs if not (e[0] == "swap" and last and
                                              {canon(e[1], mems[j].places), canon(e[2], mems[j].places)} == set(last[1:3]))]
            for e in swap_guard(evs):
                got = mems[j].apply([x if isinstance(x, float) else str(x) for x in e])
                if got:
                    logs[j]["events"].append(list(got))
    return mems, logs


def accept(args):
    from score_b13 import ACCEPT
    path = args.accept or ACCEPT
    stories = [json.loads(l) for l in open(path)]
    reader = Reader(args.model, args.adapter)
    t0 = time.time()
    mems, logs = run_stories(reader, [{"domain": s["domain"], "sentences": split_sentences(s["text"])} for s in stories],
                             args.prompt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for s, m, lg in zip(stories, mems, logs):
            answers, conf = [], []
            for q in s["questions"]:
                a, c, _ = m.ask(q["q"])
                answers.append(a)
                conf.append(c)
            f.write(json.dumps({"id": s["id"], "answers": answers, "conf": conf, "events": lg["events"],
                                "gen_tokens": lg["tokens"], "calls": lg["calls"], "parse_fail": lg["parse_fail"]}) + "\n")
    print(f"{args.out}: {len(stories)} stories, {sum(l['calls'] for l in logs)} reader calls, "
          f"{sum(l['tokens'] for l in logs) / max(1, sum(l['calls'] for l in logs)):.1f} tokens per call, "
          f"{sum(l['parse_fail'] for l in logs)} parse failures, {time.time() - t0:.0f}s", flush=True)


def long_test(args):
    eps = [json.loads(l) for l in open(args.data)]
    reader = Reader(args.model, args.adapter)
    t0 = time.time()
    mems, logs = run_stories(reader, [{"domain": "locations", "sentences": e["sentences"]} for e in eps], args.prompt)
    acc = {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for e, m, lg in zip(eps, mems, logs):
            q = e["text"][e["sents"][-1]["span"][0]:]
            qname = [mm for mm in e["sents"][-1]["mentions"] if mm[2] == "q"][0][3]
            gold = norm(e["final"][qname])
            ans, conf, ent = m.ask(q)
            ok = ans == gold
            a = acc.setdefault(e["n"], [0, 0])
            a[0] += ok
            a[1] += 1
            f.write(json.dumps({"n": e["n"], "idx": e["idx"], "question": q, "pred": ans, "gold": gold, "ok": ok,
                                "conf": conf, "entity": ent, "gen_tokens": lg["tokens"], "calls": lg["calls"],
                                "parse_fail": lg["parse_fail"]}) + "\n")
    for n, (k, m) in sorted(acc.items()):
        print(f"{args.out}: N={n} accuracy {k / m:.3f} ({k}/{m})", flush=True)
    print(f"reader calls {sum(l['calls'] for l in logs)}, tokens per call "
          f"{sum(l['tokens'] for l in logs) / max(1, sum(l['calls'] for l in logs)):.1f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["accept", "long"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="A", choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--merge", default="suffix", choices=["suffix", "none"])
    ap.add_argument("--guard", action="store_true")  # post-B15 swap guard
    ap.add_argument("--adapter", default="")  # B16: LoRA student
    ap.add_argument("--b17", action="store_true")  # B17: earlier facts, replay, confidence
    ap.add_argument("--accept", default="b14_accept.jsonl")
    ap.add_argument("--data", default="b13_data/long_test.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    MERGE = a.merge
    GUARD = a.guard
    B17 = a.b17
    accept(a) if a.cmd == "accept" else long_test(a)
