"""Readable Memory as a site would deploy it, for the realistic scenarios.

  - System feeds (WMS scanner transactions, LIMS audit trail, robot vision, deploy bot / Patroni) are parsed by small
    connectors (one regular expression per record type), as an integration would read them from the system's API.
  - Human lines (chat, handover notes, email, bench notes, speech transcripts, incident channel) are read by the
    shipped reader: Qwen3-1.7B with the distilled adapter, one line at a time, with the three lines before it.
  - Every name the reader writes is resolved against the site's registry: exact ID, a known alias ("QA", "dock 2",
    "the Dyson pallet", "#5"), or a number that identifies exactly one ID ("5871" -> "LPN 0435871"). Names that
    resolve to nothing are dropped.
  - An item must be named in the message itself. A pronoun, or a name the reader took from the thread rather than
    the message ("Stuck that one in B-11-03"), goes to the item last mentioned in the same channel.
  - Two moves that trade the contents of two places ("pulled everything from A-10-02 into A-06-04 and moved what was
    in A-06-04 to A-10-02") are one swap.
  - Questions are resolved the same way, then answered from the 256 x 256 memory (b14.Memory).
"""
import re

from score_b13 import norm

PRON = {"it", "them", "they", "its", "that", "that one", "this one", "this"}


class Registry:
    def __init__(self, log):
        self.items = [e["id"] for e in log["registry"]["items"]]
        self.places = [e["id"] for e in log["registry"]["places"]]
        self.names = {}  # normalised name -> id, for ids and aliases
        self.wid = {}  # id -> its names as message words (for mentions)
        self.said = {}  # message words -> the name as written (for asking about it in the message's own terms)
        for e in log["registry"]["items"] + log["registry"]["places"]:
            for n in [e["id"]] + e["aliases"]:
                self.names.setdefault(self.key(n), set()).add(e["id"])
                w = self.words(self.key(n)).strip()  # "the Bosch pallet" is said as "Bosch pallet"
                if len(w) >= 2:
                    self.wid.setdefault(e["id"], []).append(w)
                    self.said.setdefault(w, n)

    @staticmethod
    def key(s):
        s = norm(s).replace("_", " ")
        s = re.sub(r"^(?:grandpa|anna|leo|his|her)\s+", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def resolve(self, name, pool=None):
        """Registry ID for a name, or None. pool restricts the candidates (items or places)."""
        pool = set(pool) if pool else set(self.items) | set(self.places)
        k = self.key(name)
        hits = self.names.get(k, set()) & pool
        if len(hits) == 1:
            return next(iter(hits))
        for nk, ids in self.names.items():  # "the dyson pallet" / "dyson pallet" / "dyson" / "orders-db"
            if (k and (nk.endswith(" " + k) or k.endswith(" " + nk) or nk == "the " + k or nk.startswith(k + " "))) \
                    and len(ids & pool) == 1:
                return next(iter(ids & pool))
        digits = re.findall(r"\d{3,}", name)
        if digits:
            c = {i for i in pool if any(re.sub(r"\D", "", i).endswith(d.lstrip("0") or d) for d in digits)}
            if len(c) == 1:
                return next(iter(c))
        # a known name inside a longer one: "chat-gateway v2.15.1", "the keys to the corolla on the hall table"
        best = None
        for nk, ids in self.names.items():
            if len(nk) >= 3 and f" {nk} " in f" {k} " and len(ids & pool) == 1 and (best is None or len(nk) > len(best[0])):
                best = (nk, next(iter(ids & pool)))
        return best[1] if best else None

    @staticmethod
    def words(s):
        """Lowercase words with the punctuation removed, padded with spaces, for finding a name inside a message."""
        return " " + re.sub(r"\s+", " ", re.sub(r"[^\w\s-]|_", " ", s.lower())).strip() + " "

    def find_in(self, question, pool=None, message=False):
        """The registry entry a question (or, with message=True, a chat message) mentions: the longest name or
        number in it."""
        pool = set(pool) if pool else set(self.items) | set(self.places)
        q = self.words(question) if message else " " + self.key(question.rstrip("?")) + " "
        best = None
        for nk, ids in self.names.items():
            nw = self.words(nk).strip() if message else nk
            if len(ids & pool) == 1 and len(nw) >= 2 and f" {nw} " in q and (best is None or len(nk) > len(best[0])):
                best = (nk, next(iter(ids & pool)))
        if best:
            return best[1]
        for d in re.findall(r"\d{3,}", question):
            c = [i for i in pool if re.sub(r"\D", "", i).endswith(d)]
            if len(c) == 1:
                return c[0]
        return None

    def said_as(self, text, i):
        """The name a message uses for registry entry i (its longest name found in the message), else the ID."""
        t = self.words(text)
        hits = [w for w in self.wid.get(i, []) if f" {w} " in t]
        return self.said[max(hits, key=len)] if hits else i

    def mentions(self, text, i):
        """Whether a message names registry entry i: one of its names, or a number that ends its ID."""
        t = self.words(text)
        if any(f" {w} " in t for w in self.wid[i]):
            return True
        tail = re.sub(r"\D", "", i)
        return any(len(d) >= 3 and tail.endswith(d.lstrip("0") or d) for d in re.findall(r"\d+", text))


# ------------------------------------------------------------------------------------------------ connectors
def connect(scenario, text):
    """Events from one system-feed line, or None if the line is not a system record."""
    if scenario == "warehouse":
        m = re.search(r"RECEIVE\s+(LPN \d+)\s+@ (.+?)\s+user=", text)
        if m:
            return [["move", m.group(1), m.group(2)]]
        m = re.search(r"(?:PUTAWAY|MOVE)\s+(LPN \d+)\s+(.+?) -> (.+?)\s+user=", text)
        if m:
            return [["move", m.group(1), m.group(3)]]
        m = re.search(r"CYCLECNT\s+(LPN \d+)\s+@ (.+?)\s+OK", text)
        if m:
            return [["move", m.group(1), m.group(2)]]
        if re.match(r"\d\d:\d\d\s+(RF|PRN)-", text):
            return []
    elif scenario == "biobank":
        if re.match(r"\d{4}-\d\d-\d\d ", text):
            f = [x.strip() for x in text.split("|")]
            op = f[2]
            if op == "LOCATE":
                return [["move", f[3], f[4].replace("in ", "")]]
            if op in ("STORE", "ACCESSION"):
                return [["move", f[3], f[4].split("/")[0]]]
            if op == "TRANSFER":
                return [["move", f[3], f[4].split("->")[1].strip().split("/")[0]]]
            if op == "CHECKOUT":
                return [["move", f[3], "PCR bench"]]
            if op == "SHIP":
                return [["move", f[3], re.search(r"in (\S+)", f[4]).group(1)]]
            return []
    elif scenario == "home":
        m = re.match(r"\d\d:\d\d:\d\d\s+VISION\s+(\w+) @ (\w+)\s+conf=", text)
        if m:
            return [["move", m.group(1).replace("_", " "), m.group(2).replace("_", " ")]]
        if re.match(r"\d\d:\d\d:\d\d\s+", text):
            return []
    elif scenario == "ops":
        m = re.search(r"deploy-bot\s+(\S+) v\S+ live in (\S+)", text)
        if m:
            return [["move", m.group(1), m.group(2)]]
        m = re.search(r"patroni\s+(\S+): (?:leader is|failover, leader \S+ ->) (\S+)", text)
        if m:
            return [["move", m.group(1) + " primary", m.group(2)]]
        if re.match(r"\[\d\d:\d\d\] (pagerduty|k8s|grafana)", text):
            return []
    return None


# ------------------------------------------------------------------------------------------------ the memory
class SiteMemory:
    """One site's readable memory: connectors + reader + registry + b14.Memory."""

    def __init__(self, log, b14):
        self.log, self.b14 = log, b14
        self.reg = Registry(log)
        self.mem = b14.Memory(log["domain"])
        self.last = None
        self.last_by_channel = {}  # channel -> the item last mentioned there (the thread's "it" / "that one")
        self.dropped = 0

    def apply(self, evs, from_reader, channel=None, text=None):
        """from_reader: events the reader wrote for the human message `text`. An item the message does not name
        (a pronoun, or a name taken from the thread) is the item last mentioned in the same channel: the chat's
        "that one" is the chat's last pallet, not the last scan."""
        out = []
        evs = self.pair_swaps(evs) if from_reader else evs
        named = self.reg.find_in(text, self.reg.items, message=True) if from_reader and text else None
        for ev in evs:
            op, a, b = ev[0].lower(), str(ev[1]), str(ev[2])
            if op == "move":
                ka = self.reg.key(a)
                last = self.last_by_channel.get(channel, self.last) if from_reader else self.last
                ra = last if ka in PRON else self.reg.resolve(a, self.reg.items)
                rb = self.reg.resolve(b, self.reg.places)
                if from_reader and text is not None:
                    ra, rb = self.ground(text, ka in PRON, ra, rb, last)
                if not ra or not rb or ra == rb:
                    self.dropped += from_reader
                    continue
                ev2 = ["move", ra, rb]
                self.last = ra
                if from_reader:
                    self.last_by_channel[channel] = ra
            elif op == "swap":
                ra, rb = self.reg.resolve(a, self.reg.places), self.reg.resolve(b, self.reg.places)
                if not ra or not rb or ra == rb:
                    self.dropped += from_reader
                    continue
                ev2 = ["swap", ra, rb]
            else:
                continue
            got = self.mem.apply([ev2[0], norm(ev2[1]), norm(ev2[2])])
            if got:
                out.append(ev2)
        if from_reader and named and not any(e[0] == "move" for e in out):
            self.last_by_channel[channel] = named  # a plan or a search still makes it the thread's "that one"
        return out

    def ground(self, text, pron, ra, rb, last):
        """Hold a move to the message it was read from. The place is the one the message names, if it names exactly
        one; the item is the one it names other than that place, or, if it names none, the thread's antecedent."""
        said = lambda i: self.reg.mentions(text, i)
        if not rb or not said(rb):
            ps = [p for p in self.reg.places if said(p)]
            rb = ps[0] if len(ps) == 1 else rb
        if pron or (ra and not said(ra)):
            its = [i for i in self.reg.items if i != rb and said(i)]
            ra = its[0] if len(its) == 1 else None if its else last
        return ra, rb

    def orient(self, ev):
        """A move written backwards, [move FRZ-07 "Rack 4"], is [move "Rack 4" FRZ-07]."""
        a, b = str(ev[1]), str(ev[2])
        if ev[0].lower() == "move" and self.reg.resolve(a, self.reg.items) is None \
                and self.reg.resolve(a, self.reg.places) and self.reg.resolve(b, self.reg.items):
            return [ev[0], b, a]
        return ev

    def places_named(self, text, ev):
        """The places a message names other than the item moved: where a single move could have gone."""
        it = self.reg.resolve(str(ev[1]), self.reg.items)
        return [p for p in self.reg.places if p != it and self.reg.mentions(text, p)]

    def pair_swaps(self, evs):
        """[move P Q] + [move Q P] where P and Q are places (their contents trade): one swap."""
        pl = lambda x: self.reg.resolve(str(x), self.reg.places) if self.reg.resolve(str(x), self.reg.items) is None else None
        moves = [(e, pl(e[1]), pl(e[2])) for e in evs if e[0].lower() == "move"]
        for i, (e, p, q) in enumerate(moves):
            for f, p2, q2 in moves[i + 1:]:
                if p and q and p != q and (p2, q2) == (q, p):
                    return [x for x in evs if x is not e and x is not f] + [["swap", p, q]]
        return evs

    def ask(self, question, kind):
        ent = self.reg.find_in(question)
        if ent is None:
            return None, 0.0, None
        if kind == "what_in":
            items, c = self.mem.contents(norm(ent))
            return [self.canon(x) for x in items], float(c), ent
        if kind == "chain":
            hops, c = self.mem.where(norm(ent))
            return [self.canon(x) for x in hops], float(c), ent
        v, c = self.mem.read(norm(ent))
        return (self.canon(v) if v else None), float(c), ent

    def canon(self, n):
        for i in self.reg.items + self.reg.places:
            if norm(i) == n:
                return i
        return n

    def board(self):
        out = {}
        for i in self.reg.items:
            n = norm(i)
            if n not in self.mem.K.names:
                continue
            if self.log["domain"] == "containers":
                hops, c = self.mem.where(n)
                out[i] = [[self.canon(x) for x in hops], round(float(c), 3)]
            else:
                v, c = self.mem.read(n)
                out[i] = [self.canon(v) if v else None, round(float(c), 3)]
        return out
