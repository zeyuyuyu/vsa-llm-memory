"""Readable Memory SDK (v1): a small language model reads free text; a fixed-size readable memory tracks the state.

    rm = ReadableMemory(base_model, writer_ckpt, domain="locations")   # or domain="containers"
    rm.observe("Anna put the keys in the drawer. Later the drawer and the basket swapped contents.")
    rm.ask("Where are the keys now?")          # -> {"answer": "basket", "confidence": 1.0, ...}
    rm.ask("What is in the drawer?")           # reverse read of the memory
    rm.beliefs()                               # every tracked entity, its value and confidence
    rm.correct("keys", "drawer")               # overwrite one belief

Domains: "locations" = objects in places, moves and whole-place swaps; "containers" = containers moved into other
containers (questions about the outermost container are answered hop by hop). The writer is the B9b model
(open vocabulary); the memory is B7b's algebra on a 256 x 256 state with codes allocated per name.
"""
import re

import torch
import torch.nn.functional as F

from b9 import DH, ROLES, Codebook, Writer, decode_words, mention_vecs

PREAMBLE = {
    "locations": ("There are several places: {places}. Every place starts empty. The following actions happen in order. "
                  "Putting X in P moves X from wherever it is into P. Swapping the contents of P and Q moves everything "
                  "that is in P into Q and everything that is in Q into P, at the same time."),
    "containers": ("There are several containers, named by colour and kind. They start out separate. The following moves "
                   "happen in order. Putting A into B moves A, together with everything inside it, into B; A is no "
                   "longer in whatever held it before."),
}
CLAUSE_SPLIT = re.compile(r", and after that |, and then |, then |; ")  # B12: two events in one sentence
OBJ_PRO, PLACE_PRO = {"it", "them"}, {"there"}  # B12: resolved to the previous move's object / place
WHAT_IN = re.compile(r"\bwhat(?:'s| is| are|\s+things|\s+objects)?\b.*\b(?:in|on|inside|at|under|contained in)\b",
                     re.I)  # B13: on / at / under too


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class ReadableMemory:
    def __init__(self, base_model, writer_ckpt, domain="locations", device="cuda", decoder="b12"):
        assert domain in PREAMBLE and decoder in ("b12", "b13")
        self.domain, self.device, self.decoder = domain, device, decoder
        self.task = "T2" if domain == "locations" else "T3"
        if decoder == "b13":  # B13: a role per mention, events paired per sentence, pronouns = last mentioned
            from b13 import Writer13
            self.w = Writer13(base_model).to(device).eval()
        else:
            self.w = Writer(base_model).to(device).eval()
        self.w.load_state_dict(torch.load(writer_ckpt, weights_only=True)["params"], strict=False)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.sentences, self.events, self.corrections = [], [], []  # corrections: (position, entity, value)
        self._rebuild()

    def _rebuild(self):
        """Replay every parsed event (and correction) into a fresh memory; the algebra is exact and cheap."""
        self.K, self.V = Codebook("key"), Codebook("val")
        self.S = torch.zeros(DH, DH)
        for i in range(len(self.events) + 1):  # a correction takes effect where it was made in the story
            for pos, name, value in self.corrections:
                if pos == i:
                    self._apply("write", name, value)
            if i < len(self.events):
                ev = self.events[i]
                self._apply(ev["op"], ev.get("x"), ev.get("y"))

    def _apply(self, op, a, b):
        if a is None or b is None or a == b:
            return
        if op == "swap":
            wv = F.normalize(self.V.get(a) - self.V.get(b), dim=0)
            self.S = self.S - 2 * torch.outer(wv, wv @ self.S)
        else:
            k, v = self.K.get(a), self.V.get(b)
            self.S = self.S + torch.outer(v - self.S @ k, k)

    # ------------------------------------------------------------------ reading text
    @torch.no_grad()
    def _parse(self, sentences, question=None):
        """Run the writer over the story (+ an optional question) in its training layout; returns per-sentence picks."""
        places = []
        for _ in range(2 if self.domain == "locations" else 1):  # locations: a second pass lists the places found
            plist = ", ".join(f"the {p}" for p in places) if places else "a few"
            pre = PREAMBLE[self.domain].format(places=plist)
            text, spans = pre + "\n\n", []
            for i, s in enumerate(sentences):
                prefix = f"{i + 1}. "
                start = len(text) + len(prefix)
                body = s.rstrip(".")
                pos = 0
                for m in CLAUSE_SPLIT.finditer(body):  # one span per clause (B12)
                    spans.append((start + pos, start + m.start() - 1))
                    pos = m.end()
                spans.append((start + pos, start + len(body) - 1))
                text += prefix + s + "\n"
            if question:
                text += "\n"
                spans.append((len(text), len(text) + len(question) - 1))
                text += question
            ep = {"text": text, "sents": [{"span": sp, "mentions": [], "op": "write"} for sp in spans]}
            h, offs = self.w.encode([ep])
            picks = []
            for sp in spans:
                t0, t1 = Writer.tok_range(offs[0], *sp)
                H = h[0, t0:t1]
                tags = self.w.bio(H).argmax(-1).tolist()
                ments = decode_words(text, offs[0], t0, t1, tags)
                pick = {"mentions": [m[2] for m in ments]}
                if ments:
                    rec = {"H": H, "t0": t0}
                    mv, sv = mention_vecs(rec, [(a, b) for a, b, _ in ments])
                    probs = self.w.role(mv).softmax(0)  # over mentions, per role
                    for ri, role in enumerate(ROLES):
                        j = int(probs[:, ri].argmax())
                        pick[role] = (ments[j][2], float(probs[j, ri]))
                    pick["swap_p"] = float(self.w.op(sv).softmax(-1)[1]) if self.domain == "locations" else 0.0
                picks.append(pick)
            places = sorted({p[r][0] for p in (picks[:-1] if question else picks) for r in ("y", "sa", "sb")
                             if r in p and p[r][0].lower() not in PLACE_PRO})
        return picks

    def _layout13(self, sentences, places, question=None):
        pl = ", ".join(f"the {p}" for p in places) if places else "a few"
        text, spans, kinds = PREAMBLE[self.domain].format(places=pl) + "\n\n", [], []
        for i, s in enumerate(sentences):
            prefix = f"{i + 1}. "
            spans.append((len(text) + len(prefix), len(text) + len(prefix) + len(s) - 1))
            kinds.append("story")
            text += prefix + s + "\n"
        if question:
            text += "\n"
            spans.append((len(text), len(text) + len(question) - 1))
            kinds.append("question")
            text += question
        return text, spans, kinds

    def _decode13(self, question=None):
        from b13 import PLACE_ROLES, decode_story
        places = []
        for _ in range(2 if self.domain == "locations" else 1):  # second pass lists the places found
            dec = decode_story(self.w, *self._layout13(self.sentences, places, question))
            places = sorted({m["ref"] for d in dec for m in d.get("mentions", [])
                             if isinstance(m, dict) and m["role"] in PLACE_ROLES and m["ref"]})
        return dec

    def observe(self, text):
        """Add sentences to the story; returns the events parsed from them, each with a confidence."""
        n_before = len(self.events)
        self.sentences += split_sentences(text)
        if self.decoder == "b13":
            self.events = []
            for s, d in zip(self.sentences, self._decode13()):
                evs = d.get("events") or [{"op": "none", "confidence": 0.0}]
                self.events += [{"sentence": s, **ev, "mentions": d.get("mentions", [])} for ev in evs]
            self._rebuild()
            return self.events[n_before:]
        clauses = [c for s in self.sentences for c in CLAUSE_SPLIT.split(s.rstrip("."))]
        picks = self._parse(self.sentences)
        self.events, last_put = [], None
        for s, p in zip(clauses, picks):
            if not p.get("mentions"):
                self.events.append({"sentence": s, "op": "none", "confidence": 0.0})
            elif p.get("swap_p", 0) > 0.5:
                a = p["sa"][0]
                b = next((m for m in p["mentions"] if m != a), None)
                self.events.append({"sentence": s, "op": "swap", "x": a, "y": b,
                                    "confidence": round(p["swap_p"] * p["sa"][1], 3)})
            else:
                x, y = p["x"][0], p["y"][0]
                if x.lower() in OBJ_PRO:  # "it" / "them": the object of the previous move
                    x = last_put["x"] if last_put else None
                if y.lower() in PLACE_PRO:  # "there": the place of the previous move
                    y = last_put["y"] if last_put else None
                ev = {"sentence": s, "op": "write", "x": x, "y": y,
                      "confidence": round(p["x"][1] * p["y"][1] * (1 - p.get("swap_p", 0)), 3)}
                if x != p["x"][0] or y != p["y"][0]:
                    ev["resolved"] = True
                self.events.append(ev)
                last_put = ev
        self._rebuild()
        return self.events[n_before:]

    # ------------------------------------------------------------------ reading memory
    def _read(self, name):
        k = self.K.get(name, create=False)
        if k is None:
            return None, 0.0, 0.0
        r = self.S @ k
        if r.norm() < 0.5:
            return None, 0.0, float(r.norm())
        val, margin = self.V.cleanup(r)
        return val, round(margin, 3), float(r.norm())

    def ask(self, question):
        if self.decoder == "b13":
            q = self._decode13(question)[-1]
            if not q.get("q"):
                return {"answer": None, "confidence": 0.0, "note": "no entity found in the question"}
            name = q["q"]
        else:
            picks = self._parse(self.sentences, question=question)
            q = picks[-1]
            if not q.get("mentions"):
                return {"answer": None, "confidence": 0.0, "note": "no entity found in the question"}
            name = q["q"][0]
        if WHAT_IN.search(question) and self.domain == "locations":
            return self.contents(name)
        if self.domain == "locations":
            val, margin, _ = self._read(name)
            return {"kind": "where", "entity": name, "answer": val, "confidence": margin}
        hops, cur, margins = [], name, []
        for _ in range(40):
            val, margin, nrm = self._read(cur)
            if val is None:
                break
            hops.append(val)
            margins.append(margin)
            cur = val
        return {"kind": "outermost", "entity": name, "answer": hops[-1] if hops else None, "hops": hops,
                "confidence": min(margins) if margins else 0.0}

    def contents(self, place):
        """Which tracked objects does the memory currently place in `place`? (one read per tracked object)"""
        if self.V.get(place, create=False) is None or not len(self.K.names):
            return {"kind": "contents", "entity": place, "answer": [], "items": []}
        inside = []
        for name in self.K.names:
            val, margin, _ = self._read(name)
            if val == place:
                inside.append({"entity": name, "confidence": margin})
        return {"kind": "contents", "entity": place, "answer": [i["entity"] for i in inside], "items": inside}

    def beliefs(self):
        out = []
        for name in self.K.names:
            val, margin, _ = self._read(name)
            row = {"entity": name, "value": val, "confidence": margin}
            if self.domain == "containers":
                hops, cur = [], name
                for _ in range(40):
                    nxt, _, _ = self._read(cur)
                    if nxt is None:
                        break
                    hops.append(nxt)
                    cur = nxt
                row["outermost"] = hops[-1] if hops else None
            out.append(row)
        return out

    def correct(self, entity, value):
        """Overwrite one belief at the current point of the story; later events still update it as usual."""
        self.corrections.append((len(self.events), entity, value))
        self._rebuild()
        return self._read(entity)


class LLMReadableMemory:
    """B15/B16 reader: Qwen3-8B (B15) or the distilled Qwen3-1.7B student (B16: reader_model = the 1.7B snapshot,
    adapter = b16_models/student_s0; results/b16_verdict.txt). The B15 configuration: a prompted Qwen3-8B reads one sentence at a
    time (prompt C, no name merging, no thinking) and writes the events into the readable memory. Same interface as
    ReadableMemory: observe / ask / beliefs / contents / correct / reset. Pass `reader` to share one model between
    several memories."""

    def __init__(self, reader_model=None, domain="locations", reader=None, prompt="C", guard=False, adapter=None,
                 items=None, places=None):
        """items, places: the site's asset list (optional). When given, a pronoun left in an event is resolved to the
        most recently moved item, and only changes to listed items in listed places are stored -- the configuration
        of apps/run.py --registry (the application scenarios)."""
        import b14
        b14.MERGE, b14.GUARD = "none", guard  # guard: post-B15 swap guard, off = the tested B15 config (dev: -1 story on B13 set, 0 on B14 and B15)
        self.b14, self.domain, self.prompt = b14, domain, prompt
        from score_b13 import norm
        self.norm = norm
        self.items = {norm(x) for x in items} if items else None
        self.places_ok = {norm(x) for x in places} if places else None
        self.reader = reader or b14.Reader(reader_model, adapter)  # adapter: the B16 1.7B student
        self.reset()

    def reset(self):
        self.mem = self.b14.Memory(self.domain)
        self.sentences, self.events = [], []
        self.last = None

    def _screen(self, evs):
        if self.items is None:
            return evs
        out = []
        for ev in evs:
            op, a, b = ev[0].lower(), self.norm(ev[1]), self.norm(ev[2])
            if op == "move" and a in {"it", "them", "they", "its"}:
                if not self.last:
                    continue
                a = self.last
            if op == "move" and (a not in self.items or b not in self.places_ok):
                continue
            if op == "swap" and (a not in self.places_ok or b not in self.places_ok):
                continue
            out.append([op, a, b])
            if op == "move":
                self.last = a
        return out

    def observe(self, text):
        new = []
        for s in split_sentences(text):
            ctx = " ".join(self.sentences[-3:]) or "(none)"
            prompt = self.b14.PROMPTS[(self.prompt, self.domain)].format(
                context=ctx, sentence=s, objs=", ".join(self.mem.objs) or "(none)",
                places=", ".join(self.mem.places) or "(none)")
            (txt, _), = self.reader([prompt])
            self.sentences.append(s)
            evs = [self.mem.apply([str(x) for x in e]) for e in self._screen(self.b14.swap_guard(self.b14.parse(txt) or []))]
            evs = [{"sentence": s, "op": e[0], "x": e[1], "y": e[2], "confidence": 1.0} for e in evs if e]
            self.events += evs or [{"sentence": s, "op": "none", "confidence": 0.0}]
            new += evs
        return new

    def ask(self, question):
        a, c, ent = self.mem.ask(question)
        if ent is None:
            return {"answer": None, "confidence": 0.0, "note": "no known entity in the question"}
        if WHAT_IN.search(question) and self.domain == "locations":
            return {"kind": "contents", "entity": ent, "answer": a,
                    "items": [{"entity": o, "confidence": self.mem.read(o)[1]} for o in a]}
        if self.domain == "containers":
            return {"kind": "outermost", "entity": ent, "answer": a[-1] if a else None, "hops": a, "confidence": c}
        return {"kind": "where", "entity": ent, "answer": a, "confidence": c}

    def contents(self, place):
        a, _ = self.mem.contents(place)
        return {"kind": "contents", "entity": place, "answer": a}

    def beliefs(self):
        out = []
        for name in self.mem.K.names:
            val, margin = self.mem.read(name)
            row = {"entity": name, "value": val, "confidence": margin}
            if self.domain == "containers":
                hops, _ = self.mem.where(name)
                row["outermost"] = hops[-1] if hops else None
            out.append(row)
        return out

    def correct(self, entity, value):
        got = self.mem.apply(["move", entity, value])
        if got:
            self.events.append({"sentence": "(correction)", "op": "write", "x": got[1], "y": got[2], "confidence": 1.0})
        return self.mem.read(got[1]) if got else (None, 0.0)


class SiteMemory:
    """A site's readable memory: system feeds through connectors, human messages through the reader, every name
    resolved against the site's asset list (apps/real/product.py; the configuration of the application scenarios).

        mem = SiteMemory(registry={"items": [{"id": "LPN 0435871", "aliases": ["5871", "the Dyson pallet"]}, ...],
                                   "places": [{"id": "QA HOLD", "aliases": ["QA"]}, ...]},
                         reader_model=QWEN3_1_7B, adapter="b16_models/student_s0", connector="warehouse")
        mem.observe(lines)                 # system records and messages, in time order
        mem.ask("Where's 5871?")           # {"answer": "QA HOLD", "confidence": 1.0}
        mem.correct("LPN 0435871", "OVERFLOW")
    """

    def __init__(self, registry, reader_model=None, adapter=None, domain="locations", connector=None, reader=None):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps", "real"))
        import b14
        from product import SiteMemory as _Site, connect
        b14.MERGE, b14.GUARD = "none", False
        self.b14, self.connect, self.connector = b14, connect, connector
        self.site = _Site({"registry": registry, "domain": domain}, b14)
        self.reader = reader or b14.Reader(reader_model, adapter)
        self.lines, self.threads = [], {}

    def observe(self, lines, channel="chat"):
        """System records go through the connector; anything else is a human message of `channel`: the reader sees
        the last three messages of that channel, writes what changed, and a one-word check keeps only changes the
        message says actually happened (the configuration of apps/real/run_real.py --verify)."""
        from run_real import verify_prompt
        if isinstance(lines, str):
            lines = [lines]
        for line in lines:
            evs = self.connect(self.connector, line) if self.connector else None
            if evs is None:
                thread = self.threads.setdefault(channel, [])
                prompt = self.b14.PROMPTS[("C", self.site.log["domain"])].format(
                    context=" ".join(thread[-3:]) or "(none)", sentence=line,
                    objs=", ".join(self.site.mem.objs) or "(none)", places=", ".join(self.site.mem.places) or "(none)")
                (txt, _), = self.reader([prompt])
                evs = self.b14.parse(txt) or []
                if evs:
                    ans = self.reader([verify_prompt(line, ev, self.site.log["domain"]) for ev in evs])
                    evs = [ev for ev, (a, _) in zip(evs, ans) if a.strip().lower().startswith("yes")]
                self.site.apply(evs, from_reader=True, channel=channel)
                thread.append(line)
            else:
                self.site.apply(evs, from_reader=False)
            self.lines.append(line)

    def ask(self, question):
        kind = "what_in" if WHAT_IN.search(question) and self.site.log["domain"] == "locations" else (
            "chain" if self.site.log["domain"] == "containers" else "where")
        a, c, ent = self.site.ask(question, kind)
        return {"entity": ent, "answer": a, "confidence": round(c, 3)}

    def beliefs(self):
        return self.site.board()

    def correct(self, item, place):
        self.site.apply([["move", item, place]], from_reader=False)
        return self.ask(f"Where is {item}?")
