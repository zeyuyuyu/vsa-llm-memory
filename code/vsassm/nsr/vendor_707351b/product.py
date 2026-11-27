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
    """A site's asset list. Lookups go through indexes built here; they give the answers of scanning every name in
    registry order (apps/real/tools/check_registry_index.py), at a cost that does not grow with the registry."""

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
        self._all = set(self.items) | set(self.places)
        self._pools = {id(self.items): set(self.items), id(self.places): set(self.places)}
        self._rank = {nk: r for r, nk in enumerate(self.names)}  # scan order of the names
        self._pre, self._suf, self._msg = {}, {}, {}  # proper word prefix / word suffix of a name, name as message words
        for nk in self.names:
            for m in re.finditer(" ", nk):
                self._pre.setdefault(nk[:m.start()], []).append(nk)
                self._suf.setdefault(nk[m.end():], []).append(nk)
            self._msg.setdefault(self.words(nk).strip(), []).append(nk)
        self._wmap = {}  # message words -> the IDs they name
        for i, ws in self.wid.items():
            for w in ws:
                self._wmap.setdefault(w, set()).add(i)
        self._tail = {}  # every ending of an ID's digits -> the IDs
        for i in self.items + self.places:
            t = re.sub(r"\D", "", i)
            for j in range(len(t)):
                self._tail.setdefault(t[j:], set()).add(i)
        self._span = max([len(x.split()) for x in list(self.names) + list(self._msg) + list(self._wmap)] + [1])
        self._last = (None, set())  # the last text mentioned() was asked about
        self.by_norm = {}
        for i in self.items + self.places:
            self.by_norm.setdefault(norm(i), i)

    @staticmethod
    def key(s):
        s = norm(s).replace("_", " ")
        s = re.sub(r"^(?:grandpa|anna|leo|his|her)\s+", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def _pool(self, pool):
        return (self._pools.get(id(pool)) or set(pool)) if pool else self._all

    def _spans(self, text):
        """Every run of up to _span consecutive words of a single-spaced text."""
        t = text.split(" ")
        t = [w for w in t if w] if text.strip() else []
        return {" ".join(t[a:b]) for a in range(len(t)) for b in range(a + 1, min(len(t), a + self._span) + 1)}

    def _first(self, cands, pool, longest=False):
        """The name scanning would stop at (first in registry order; with longest, the longest, then the first) among
        names that identify exactly one entry of the pool."""
        ok = [nk for nk in cands if len(self.names[nk] & pool) == 1]
        if not ok:
            return None
        nk = min(ok, key=lambda n: (-len(n) if longest else 0, self._rank[n]))
        return next(iter(self.names[nk] & pool))

    def resolve(self, name, pool=None):
        """Registry ID for a name, or None. pool restricts the candidates (items or places)."""
        pool = self._pool(pool)
        k = self.key(name)
        hits = self.names.get(k, set()) & pool
        if len(hits) == 1:
            return next(iter(hits))
        if k:  # "the dyson pallet" / "dyson pallet" / "dyson" / "orders-db"
            c = set(self._suf.get(k, [])) | set(self._pre.get(k, []))
            c |= {k[m.end():] for m in re.finditer(" ", k)} & self.names.keys()
            if "the " + k in self.names:
                c.add("the " + k)
            got = self._first(c, pool)
            if got:
                return got
        digits = re.findall(r"\d{3,}", name)
        if digits:
            c = set().union(*[self._tail.get(d.lstrip("0") or d, set()) for d in digits]) & pool
            if len(c) == 1:
                return next(iter(c))
        # a known name inside a longer one: "chat-gateway v2.15.1", "the keys to the corolla on the hall table"
        return self._first([nk for nk in self._spans(k) if len(nk) >= 3 and nk in self.names], pool, longest=True)

    @staticmethod
    def words(s):
        """Lowercase words with the punctuation removed, padded with spaces, for finding a name inside a message."""
        return " " + re.sub(r"\s+", " ", re.sub(r"[^\w\s-]|_", " ", s.lower())).strip() + " "

    def find_in(self, question, pool=None, message=False):
        """The registry entry a question (or, with message=True, a chat message) mentions: the longest name or
        number in it."""
        pool = self._pool(pool)
        q = self.words(question) if message else " " + self.key(question.rstrip("?")) + " "
        spans = self._spans(q)
        if message:
            c = [nk for w in spans if len(w) >= 2 for nk in self._msg.get(w, [])]
        else:
            c = [nk for nk in spans if len(nk) >= 2 and nk in self.names]
        got = self._first(c, pool, longest=True)
        if got:
            return got
        for d in re.findall(r"\d{3,}", question):
            c = self._tail.get(d, set()) & pool
            if len(c) == 1:
                return next(iter(c))
        return None

    def said_as(self, text, i):
        """The name a message uses for registry entry i (its longest name found in the message), else the ID."""
        t = self.words(text)
        hits = [w for w in self.wid.get(i, []) if f" {w} " in t]
        return self.said[max(hits, key=len)] if hits else i

    def mentioned(self, text):
        """Every registry entry a message names: one of its names, or a number of three or more digits that ends its
        ID."""
        if self._last[0] != text:
            ids = set().union(*[self._wmap.get(w, set()) for w in self._spans(self.words(text))])
            for d in re.findall(r"\d+", text):
                if len(d) >= 3:
                    ids |= self._tail.get(d.lstrip("0") or d, set())
            self._last = (text, ids)
        return self._last[1]

    def mentions(self, text, i):
        """Whether a message names registry entry i."""
        return i in self.mentioned(text)


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
class ShardedMemory:
    """b14.Memory for sites with more items or places than one state holds. One 256 x 256 state is exact up to 256
    keys and 256 values (its codes are orthogonal) and wrong beyond, so items go to shards in the order they are first
    written, `keys` to a shard; a shard about to allocate value code number `values + 1` is first rebuilt from its own
    answers, keeping only the places its items are in now. A swap touches only the shards that have a code for one of
    the two places. Below both limits this is one b14.Memory, and it gives the same answers at any size."""

    class _Names:
        def __init__(self):
            self.names = []

    def __init__(self, domain, b14, keys=None, values=None):
        self.domain, self.b14 = domain, b14
        self.cap_k, self.cap_v = keys or b14.DH // 2, values or b14.DH
        assert self.cap_k < self.cap_v <= b14.DH  # a rebuilt shard keeps at most cap_k values, so one more always fits
        self.shards, self.home = [b14.Memory(domain)], {}  # home: key -> shard index
        self.K = self._Names()
        self.objs, self.places = [], []  # every name in first-write order, as one b14.Memory lists them
        self._in = {"objs": set(), "places": set()}
        self.rebuilds = 0

    def _note(self, which, n):
        if n not in self._in[which]:
            self._in[which].add(n)
            getattr(self, which).append(n)

    def _room(self, i, need):
        """Make room in shard i for `need` new value codes, rebuilding it if it would pass the limit."""
        m = self.shards[i]
        if len(m.V.names) + need <= self.cap_v:
            return
        new = self.b14.Memory(self.domain)
        for o in m.K.names:
            v, _ = m.read(o)
            if v is not None:
                new.apply(["move", o, v])
        self.shards[i] = new
        self.rebuilds += 1

    def apply(self, ev):
        op, a, b = ev
        canon = self.b14.canon if self.b14.MERGE != "none" else lambda n, known: norm(n) or None
        if op == "swap" and self.domain == "locations":
            a, b = canon(a, self.places), canon(b, self.places)
            if not a or not b or a == b:
                return None
            for n in (a, b):
                self._note("places", n)
            for i in range(len(self.shards)):
                have = [n in self.shards[i].V.names for n in (a, b)]
                if not any(have):
                    continue  # none of this shard's items is or was in either place
                self._room(i, 2 - sum(have))
                if any(n in self.shards[i].V.names for n in (a, b)):
                    self.shards[i].apply(["swap", a, b])
            return ("swap", a, b)
        if op != "move":
            return None
        a = canon(a, self.objs)
        b = canon(b, self.objs if self.domain == "containers" else self.places)
        if not a or not b or a == b:
            return None
        if a not in self.home:
            if len(self.shards[-1].K.names) >= self.cap_k:
                self.shards.append(self.b14.Memory(self.domain))
            self.home[a] = len(self.shards) - 1
            self.K.names.append(a)
        self._note("objs", a)
        self._note("objs" if self.domain == "containers" else "places", b)
        i = self.home[a]
        self._room(i, b not in self.shards[i].V.names)
        self.shards[i].apply(["move", a, b])
        return ("write", a, b)

    def read(self, name):
        i = self.home.get(name)
        return (None, 0.0) if i is None else self.shards[i].read(name)

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
        got = [m.contents(place) for m in self.shards if place in m.V.names]  # shards hold keys in first-write order
        return [o for its, _ in got for o in its], min([c for its, c in got if its], default=1.0)


class SiteMemory:
    """One site's readable memory: connectors + reader + registry + b14.Memory (or, for large sites, ShardedMemory:
    memory="sharded")."""

    def __init__(self, log, b14, memory="single"):
        self.log, self.b14 = log, b14
        self.reg = Registry(log)
        self.mem = b14.Memory(log["domain"]) if memory == "single" else ShardedMemory(log["domain"], b14)
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
        return self.reg.by_norm.get(n, n)

    def board(self):
        out, known = {}, set(self.mem.K.names)
        for i in self.reg.items:
            n = norm(i)
            if n not in known:
                continue
            if self.log["domain"] == "containers":
                hops, c = self.mem.where(n)
                out[i] = [[self.canon(x) for x in hops], round(float(c), 3)]
            else:
                v, c = self.mem.read(n)
                out[i] = [self.canon(v) if v else None, round(float(c), 3)]
        return out
