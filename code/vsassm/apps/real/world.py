"""Realistic application scenarios: what really happened, what the system of record saw, and what people said.

Each scenario simulates one site for one day (or one incident). Every state change happens in exactly one channel:
  - a system feed (WMS scanner transactions, LIMS audit trail, robot perception, deploy bot / orchestrator events),
    written in that system's own record format, or
  - a human channel only (shift chat, handover note, email, spoken remark, incident thread), because in practice many
    moves are never scanned or logged.
So the system of record goes stale, and the true state can only be recovered by reading both. The generator keeps
three states: the truth, the system-of-record view (system feeds only), and nothing else.

Human lines are left as specs here ({"kind": "human", "spec": ...}); apps/real/gen.py has an LLM write them in the
channel's own voice and a second pass checks that each one states exactly the intended changes. render_template()
writes them from templates instead (for local dry runs, and as the fallback when the LLM keeps failing the check).

Items and places carry aliases the way people actually refer to them ("5871", "the Dyson pallet", "rack 3", "#5").
"""
import random

# ------------------------------------------------------------------------------------------------ helpers


def clock(start_min):
    t = [start_min]

    def tick(lo=1, hi=7):
        t[0] += random_state[0].randint(lo, hi)
        return f"{(t[0] // 60) % 24:02d}:{t[0] % 60:02d}"
    return tick


random_state = [random.Random(0)]


class World:
    domain = "locations"
    n_lines = 240

    def __init__(self, seed):
        self.r = random.Random(seed)
        random_state[0] = self.r
        self.seed = seed

    def pick(self, xs):
        return self.r.choice(xs)

    # state -----------------------------------------------------------------------------------------
    def apply(self, state, ev):
        op, a, b = ev
        if op == "move":
            state[a] = b
        elif op == "swap":
            ina = [o for o, p in state.items() if p == a]
            inb = [o for o, p in state.items() if p == b]
            for o in ina:
                state[o] = b
            for o in inb:
                state[o] = a

    def holding(self, state, place):
        return sorted(o for o, p in state.items() if p == place)

    def chain(self, state, x):
        out = []
        while x in state and len(out) < 20:
            x = state[x]
            out.append(x)
        return out

    def add(self, kind, src, events, text=None, spec=None, who=None):
        line = {"kind": kind, "src": src, "time": self.tick(), "who": who, "events": events, "text": text,
                "spec": spec}
        self.lines.append(line)
        for ev in events:
            self.apply(self.truth, ev)
            if kind == "struct":
                self.apply(self.sor, ev)
        return line

    def finish(self, idx, questions):
        return {"id": f"{self.name}-{self.seed}-{idx}", "scenario": self.name, "domain": self.domain,
                "title": self.title, "lines": self.lines, "questions": questions,
                "registry": {"items": [{"id": i, "aliases": self.alias.get(i, [])} for i in self.items],
                             "places": [{"id": p, "aliases": self.alias.get(p, [])} for p in self.places]},
                "truth": dict(self.truth), "sor": dict(self.sor)}


# ------------------------------------------------------------------------------------------------ warehouse
GOODS = [("Bosch", "cordless drills"), ("Dyson", "V15 vacuums"), ("Nestle", "bottled water"),
         ("Kirkland", "paper towels"), ("Lego", "Star Wars sets"), ("Philips", "air fryers"),
         ("Patagonia", "jackets"), ("Logitech", "keyboards"), ("Hasbro", "board games"), ("Samsung", "monitors"),
         ("Yeti", "coolers"), ("Nike", "running shoes"), ("KitchenAid", "stand mixers"), ("Coleman", "tents"),
         ("Brita", "filter jugs"), ("Garmin", "watches"), ("DeWalt", "mitre saws"), ("Sony", "headphones"),
         ("Hydro Flask", "bottles"), ("Weber", "grills"), ("Bose", "speakers"), ("Ninja", "blenders"),
         ("Osprey", "backpacks"), ("Anker", "power banks"), ("Vitamix", "blenders"), ("Igloo", "ice chests")]


class Warehouse(World):
    name = "warehouse"
    title = "3PL warehouse, one day shift"
    n_lines = 260
    PEOPLE = ["Ramirez", "Okafor", "Lindqvist", "Patel", "Nguyen", "Kowalski", "Haddad", "Brennan"]

    def build(self, idx):
        r = self.r
        self.tick = clock(6 * 60)
        goods = r.sample(GOODS, 16)
        self.items = []
        self.alias = {}
        for cust, prod in goods:
            lpn = f"LPN {r.randint(40, 49):03d}{r.randint(1000, 9999)}"
            self.items.append(lpn)
            self.alias[lpn] = [lpn[-4:], f"the {cust} pallet", f"the {prod} pallet"]
        racks = sorted({f"{a}-{r.randint(1, 18):02d}-{r.randint(1, 4):02d}" for a in "ABCD" for _ in range(3)})[:10]
        fixed = ["DOOR 1", "DOOR 2", "DOOR 3", "STG-01", "STG-02", "QA HOLD", "OVERFLOW", "RETURNS"]
        self.places = fixed + racks
        al = {"DOOR 1": ["dock 1", "door 1"], "DOOR 2": ["dock 2", "door 2"], "DOOR 3": ["dock 3", "door 3"],
              "STG-01": ["staging 1", "stage 1"], "STG-02": ["staging 2", "stage 2"],
              "QA HOLD": ["QA", "quality hold", "the QA cage"], "OVERFLOW": ["overflow", "the overflow area"],
              "RETURNS": ["returns", "the returns cage"]}
        for rk in racks:
            a, b, c = rk.split("-")
            al[rk] = [f"{a}{b}-{c}", f"aisle {a} bay {int(b)}"]
        self.alias.update(al)
        self.lines, self.truth, self.sor = [], {}, {}
        rack = lambda: self.pick(racks)
        # morning: pallets arrive at the doors, are received in the WMS and put away
        for lpn in self.items:
            door = self.pick(["DOOR 1", "DOOR 2", "DOOR 3"])
            self.add("struct", "wms", [["move", lpn, door]],
                     text=f"{self.tick()}  RF-{r.randint(1, 12):02d}  RECEIVE   {lpn}  @ {door}  user={self.pick(self.PEOPLE).lower()}")
        last_human = None
        while len(self.lines) < self.n_lines:
            u = r.random()
            lpn = self.pick(self.items)
            cur = self.truth[lpn]
            who = self.pick(self.PEOPLE)
            if u < 0.30:  # scanned move: the WMS sees it
                dst = rack() if cur.startswith(("DOOR", "STG")) else self.pick([rack(), "STG-01", "STG-02", "DOOR 1", "DOOR 2", "DOOR 3"])
                if dst == cur:
                    continue
                kind = "PUTAWAY" if cur.startswith(("DOOR", "STG")) else "MOVE"
                self.add("struct", "wms", [["move", lpn, dst]],
                         text=f"{self.tick()}  RF-{r.randint(1, 12):02d}  {kind:<9} {lpn}  {cur} -> {dst}  user={who.lower()}")
            elif u < 0.48:  # unscanned move, reported in the floor chat
                dst = self.pick(["QA HOLD", "OVERFLOW", "RETURNS", rack(), "STG-02"])
                if dst == cur:
                    continue
                last_human = lpn
                self.add("human", "chat", [["move", lpn, dst]], who=who,
                         spec={"mode": "report", "why": self.pick(["crushed corner", "customer hold", "short-shipped",
                                                                      "making room", "wrong slot", "label damaged", ""])})
            elif u < 0.52:  # follow-up about the pallet just mentioned, often by pronoun
                if not last_human or last_human == lpn:
                    continue
                cur = self.truth[last_human]
                dst = self.pick([d for d in ["QA HOLD", "OVERFLOW", rack(), "STG-01"] if d != cur])
                self.add("human", "chat", [["move", last_human, dst]], who=who, spec={"mode": "follow_up"})
                last_human = None
            elif u < 0.55:  # re-slotting: two slots trade their loads
                full = [p for p in racks if self.holding(self.truth, p)]
                if len(full) < 2:
                    continue
                a, b = r.sample(full, 2)
                last_human = None  # "that one" would no longer be unambiguous
                self.add("human", "chat", [["swap", a, b]], who=who, spec={"mode": "swap"})
            elif u < 0.60:  # a plan that has not happened, or a request that was turned down
                other = self.pick([p for p in self.places if p != cur])
                last_human = last_human if lpn == last_human else None
                self.add("human", "chat", [], who=who, spec={"mode": "plan", "item": lpn, "place": other})
            elif u < 0.65:  # looked for it somewhere and it was not there
                other = self.pick([p for p in self.places if p != cur])
                last_human = last_human if lpn == last_human else None
                self.add("human", "chat", [], who=who, spec={"mode": "not_found", "item": lpn, "place": other})
            elif u < 0.72:  # cycle count confirms the WMS location (a state statement, not a change)
                if self.sor.get(lpn) != cur:
                    continue
                self.add("struct", "wms", [],
                         text=f"{self.tick()}  RF-{r.randint(1, 12):02d}  CYCLECNT  {lpn}  @ {cur}  OK  user={who.lower()}")
            elif u < 0.82:  # picking and printing: no location change
                self.add("struct", "wms", [], text=self.pick([
                    f"{self.tick()}  RF-{r.randint(1, 12):02d}  PICK      ORD-{r.randint(80000, 99999)}  line {r.randint(1, 9)}  qty {r.randint(1, 48)}  from {lpn}",
                    f"{self.tick()}  PRN-0{r.randint(1, 4)}  LABEL     ORD-{r.randint(80000, 99999)}  {r.randint(1, 30)} labels",
                    f"{self.tick()}  RF-{r.randint(1, 12):02d}  LOGIN     user={who.lower()}"]))
            else:  # floor chatter
                self.add("human", "chat", [], who=who, spec={"mode": "noise"})
            if len(self.lines) == self.n_lines - 1:
                break
        # end of shift: a handover note that restates where a few things are, plus one last unscanned move
        lpn = self.pick(self.items)
        dst = self.pick([d for d in ["OVERFLOW", "QA HOLD", "STG-01"] if d != self.truth[lpn]])
        restate = self.r.sample([i for i in self.items if i != lpn], 2)
        self.add("human", "handover", [["move", lpn, dst]] + [["move", i, self.truth[i]] for i in restate],
                 who=self.pick(self.PEOPLE), spec={"mode": "handover"})
        return self.finish(idx, self.questions())

    def questions(self):
        r = self.r
        stale = [i for i in self.items if self.sor.get(i) != self.truth[i]]
        fresh = [i for i in self.items if i not in stale]
        pick = r.sample(stale, min(6, len(stale)))
        pick += r.sample(fresh, min(10 - len(pick), len(fresh)))
        qs = []
        for i in pick:
            ref = self.pick([i, i[-4:], self.alias[i][1]])
            qs.append({"q": self.pick([f"Where is {ref} now?", f"Where's {ref}?", f"Which location has {ref}?"]),
                       "kind": "where", "entity": i, "a": self.truth[i], "sor": self.sor.get(i)})
        for p in ["QA HOLD", "OVERFLOW"]:
            qs.append({"q": f"What's in {p} right now?", "kind": "what_in", "entity": p,
                       "a": self.holding(self.truth, p), "sor": self.holding(self.sor, p)})
        return qs


# ------------------------------------------------------------------------------------------------ biobank
class Biobank(World):
    name = "biobank"
    title = "Biobank, specimen chain of custody"
    domain = "containers"
    n_lines = 200
    PEOPLE = ["mchen", "jalvarez", "fsaleh", "kwong", "dobrien"]
    NAMES = {"mchen": "Mei Chen", "jalvarez": "Jorge Alvarez", "fsaleh": "Farah Saleh", "kwong": "Kevin Wong",
             "dobrien": "Dana O'Brien"}

    def build(self, idx):
        r = self.r
        self.tick = clock(8 * 60)
        specs = sorted({f"S-26-{r.randint(1000, 9999):05d}" for _ in range(30)})[:14]
        boxes = sorted({f"CB-{r.randint(1, 40):03d}" for _ in range(20)})[:6]
        racks = sorted({f"R-{r.randint(1, 12):02d}" for _ in range(10)})[:4]
        freezers = ["FRZ-02", "FRZ-05", "FRZ-07"]
        self.items = specs + boxes + racks + ["DS-14"]
        self.places = boxes + racks + freezers + ["DS-14", "FedEx courier", "PCR bench"]
        self.alias = {}
        for s in specs:
            self.alias[s] = [s[-5:].lstrip("0") or s[-5:], "S-" + s[-5:]]
        for b in boxes:
            self.alias[b] = [f"box {int(b[3:])}", f"CB-{int(b[3:])}"]
        for k in racks:
            self.alias[k] = [f"rack {int(k[2:])}"]
        for f in freezers:
            self.alias[f] = [f"freezer {int(f[4:])}", f"#{int(f[4:])}", f"the -80 #{int(f[4:])}"]
        self.alias["DS-14"] = ["the dry-ice shipper", "shipper 14"]
        self.alias["FedEx courier"] = ["FedEx", "the courier"]
        self.alias["PCR bench"] = ["the PCR bench", "the bench"]
        self.lines, self.truth, self.sor = [], {}, {}
        ts = lambda: f"2026-10-{r.randint(14, 14):02d} {self.tick()}:{r.randint(0, 59):02d}"
        for k in racks:
            f = self.pick(freezers)
            self.add("struct", "lims", [["move", k, f]], text=f"{ts()} | {self.pick(self.PEOPLE)} | LOCATE   | {k} | in {f}")
        for b in boxes:
            k = self.pick(racks)
            self.add("struct", "lims", [["move", b, k]], text=f"{ts()} | {self.pick(self.PEOPLE)} | STORE    | {b} | {k}/slot {r.randint(1, 12)}")
        for s in specs:
            b = self.pick(boxes)
            self.add("struct", "lims", [["move", s, b]],
                     text=f"{ts()} | {self.pick(self.PEOPLE)} | ACCESSION| {s} | {b}/{r.choice('ABCDEFGH')}{r.randint(1, 9)}")
        shipped = False
        last_spec = None
        while len(self.lines) < self.n_lines:
            u = r.random()
            who = self.pick(self.PEOPLE)
            s = self.pick(specs)
            if u < 0.22:  # LIMS-recorded transfer between boxes
                b = self.pick([x for x in boxes if x != self.truth[s]])
                self.add("struct", "lims", [["move", s, b]],
                         text=f"{ts()} | {who} | TRANSFER | {s} | {self.truth[s]}/{r.choice('ABCDEFGH')}{r.randint(1, 9)} -> {b}/{r.choice('ABCDEFGH')}{r.randint(1, 9)}")
            elif u < 0.32:  # checked out to the bench in the LIMS
                if self.truth[s] == "PCR bench":
                    continue
                self.add("struct", "lims", [["move", s, "PCR bench"]], text=f"{ts()} | {who} | CHECKOUT | {s} | to PCR bench")
            elif u < 0.44:  # put back after bench work, written only in the bench notebook / chat
                bench = [x for x in specs if self.truth[x] == "PCR bench"]
                if not bench:
                    continue
                s = self.pick(bench)
                b = self.pick(boxes)
                last_spec = s
                self.add("human", "bench", [["move", s, b]], who=self.NAMES[who], spec={"mode": "report"})
            elif u < 0.47 and last_spec:  # follow-up on the specimen just mentioned
                b = self.pick([x for x in boxes if x != self.truth[last_spec]])
                self.add("human", "bench", [["move", last_spec, b]], who=self.NAMES[who], spec={"mode": "follow_up"})
                last_spec = None
            elif u < 0.53:  # racks moved for freezer maintenance: email only, the LIMS still has the old freezer
                k = self.pick(racks)
                f = self.pick([x for x in freezers if x != self.truth[k]])
                self.add("human", "email", [["move", k, f]], who=self.NAMES[who],
                         spec={"mode": "report", "why": f"{self.truth[k]} is being defrosted"})
            elif u < 0.58:  # a box re-shelved during the audit, recorded in the LIMS
                b = self.pick(boxes)
                if self.truth[b] in ("DS-14",):
                    continue
                k = self.pick([x for x in racks if x != self.truth[b]])
                self.add("struct", "lims", [["move", b, k]], text=f"{ts()} | {who} | STORE    | {b} | {k}/slot {r.randint(1, 12)}")
            elif u < 0.61 and not shipped:  # a box packed for shipment (LIMS) and handed to the courier (chat)
                b = self.pick(boxes)
                self.add("struct", "lims", [["move", b, "DS-14"]], text=f"{ts()} | {who} | SHIP     | {b} | in DS-14 (dry ice) to Broad, Boston")
                self.add("human", "chat", [["move", "DS-14", "FedEx courier"]], who=self.NAMES[who], spec={"mode": "report"})
                shipped = True
            elif u < 0.66:  # looked for a specimen and it was not there
                other = self.pick([x for x in boxes if x != self.truth[s]])
                self.add("human", "chat", [], who=self.NAMES[who], spec={"mode": "not_found", "item": s, "place": other})
            elif u < 0.78:  # LIMS lines that move nothing
                self.add("struct", "lims", [], text=self.pick([
                    f"{ts()} | {who} | QC       | {s} | RIN 8.{r.randint(1, 9)} PASS",
                    f"{ts()} | {who} | LOGIN    | -",
                    f"{ts()} | {who} | PRINT    | {self.pick(boxes)} | box map",
                    f"{ts()} | system | ALARM    | {self.pick(freezers)} | temp -7{r.randint(0, 9)} C (cleared)"]))
            else:  # lab chatter
                self.add("human", "chat", [], who=self.NAMES[who], spec={"mode": "noise"})
        qs = []
        stale = [s for s in specs if self.chain(self.sor, s) != self.chain(self.truth, s)]
        fresh = [s for s in specs if s not in stale]
        pick = r.sample(stale, min(6, len(stale))) + []
        pick += r.sample(fresh, min(8 - len(pick), len(fresh)))
        for s in pick:
            ref = self.pick([s, self.alias[s][0]])
            qs.append({"q": self.pick([f"Where is {ref}?", f"Where is specimen {ref} stored right now?"]), "kind": "chain",
                       "entity": s, "a": self.chain(self.truth, s), "sor": self.chain(self.sor, s)})
        for b in r.sample(boxes, 2):
            qs.append({"q": f"Where is {b}?", "kind": "chain", "entity": b, "a": self.chain(self.truth, b),
                       "sor": self.chain(self.sor, b)})
        return self.finish(idx, qs)


# ------------------------------------------------------------------------------------------------ home robot
class HomeRobot(World):
    name = "home"
    title = "Home robot for an older adult"
    n_lines = 220
    ITEMS = {"reading glasses": ["Grandpa's glasses", "his glasses"], "hearing aid case": ["the hearing aid box"],
             "pill organizer": ["the pill box", "his pills"], "car keys": ["the keys to the Corolla"],
             "house keys": ["the front door keys"], "wallet": ["Grandpa's wallet"], "TV remote": ["the remote"],
             "cane": ["the walking stick"], "phone charger": ["the charger"], "library book": ["the book"],
             "tablet": ["the iPad"], "blood pressure cuff": ["the BP cuff"], "slippers": ["his slippers"]}
    PLACES = {"kitchen table": [], "kitchen counter": [], "nightstand": ["the bedside table"], "sofa": ["the couch"],
              "hall table": ["the table by the door"], "key bowl": ["the bowl by the door"],
              "coat pocket": ["his coat"], "bathroom shelf": [], "garden bench": [], "car": ["the Corolla"],
              "desk": [], "bookshelf": [], "recliner": ["his chair"], "medicine cabinet": []}
    PEOPLE = ["Grandpa", "Anna", "Leo"]
    VISIBLE = {"kitchen table", "kitchen counter", "nightstand", "sofa", "hall table", "key bowl", "garden bench",
               "desk", "bookshelf", "recliner"}

    def build(self, idx):
        r = self.r
        self.tick = clock(7 * 60)
        self.items = r.sample(list(self.ITEMS), 11)
        self.places = list(self.PLACES)
        self.alias = {k: v for k, v in {**self.ITEMS, **self.PLACES}.items()}
        self.lines, self.truth, self.sor = [], {}, {}
        tag = lambda s: s.replace(" ", "_")
        for it in self.items:
            p = self.pick([x for x in self.places if x in self.VISIBLE])
            self.add("struct", "vision", [["move", it, p]],
                     text=f"{self.tick()}:{r.randint(10, 59)}  VISION  {tag(it)} @ {tag(p)}  conf={r.uniform(0.82, 0.98):.2f}")
        last = None
        while len(self.lines) < self.n_lines:
            u = r.random()
            it = self.pick(self.items)
            cur = self.truth[it]
            who = self.pick(self.PEOPLE)
            if u < 0.25:  # the robot sees an item somewhere it can see (a move it did not watch happen)
                p = self.pick([x for x in sorted(self.VISIBLE) if x != cur])
                self.add("struct", "vision", [["move", it, p]],
                         text=f"{self.tick()}:{r.randint(10, 59)}  VISION  {tag(it)} @ {tag(p)}  conf={r.uniform(0.80, 0.98):.2f}")
            elif u < 0.33:  # re-sighting where it already is
                if self.sor.get(it) != cur:
                    continue
                self.add("struct", "vision", [], text=f"{self.tick()}:{r.randint(10, 59)}  VISION  {tag(it)} @ {tag(cur)}  conf={r.uniform(0.80, 0.98):.2f}")
            elif u < 0.53:  # someone says where they put something (often somewhere the camera cannot see)
                p = self.pick([x for x in self.places if x != cur])
                last = it
                self.add("human", "speech", [["move", it, p]], who=who, spec={"mode": "report"})
            elif u < 0.57 and last:
                p = self.pick([x for x in self.places if x != self.truth[last]])
                self.add("human", "speech", [["move", last, p]], who=who, spec={"mode": "follow_up"})
                last = None
            elif u < 0.62:
                other = self.pick([x for x in self.places if x != cur])
                last = last if it == last else None
                self.add("human", "speech", [], who=who, spec={"mode": "not_found", "item": it, "place": other})
            elif u < 0.66:
                other = self.pick([x for x in self.places if x != cur])
                last = last if it == last else None
                self.add("human", "speech", [], who=who, spec={"mode": "plan", "item": it, "place": other})
            elif u < 0.80:
                self.add("struct", "robot", [], text=self.pick([
                    f"{self.tick()}:{r.randint(10, 59)}  NAV     docked, battery {r.randint(40, 99)}%",
                    f"{self.tick()}:{r.randint(10, 59)}  NAV     patrol living_room -> kitchen",
                    f"{self.tick()}:{r.randint(10, 59)}  REMIND  medication 'Metformin 500mg' acknowledged",
                    f"{self.tick()}:{r.randint(10, 59)}  VISION  person=Grandpa @ {tag(self.pick(sorted(self.VISIBLE)))}"]))
            else:
                self.add("human", "speech", [], who=who, spec={"mode": "noise"})
        qs = []
        stale = [i for i in self.items if self.sor.get(i) != self.truth[i]]
        pick = r.sample(stale, min(6, len(stale)))
        pick += r.sample([i for i in self.items if i not in pick], min(9 - len(pick), len(self.items) - len(pick)))
        for i in pick:
            ref = self.pick(["the " + i] + self.alias.get(i, [])[:1])
            plural = ref.endswith(("glasses", "keys", "slippers", "pills"))
            qs.append({"q": self.pick([f"Where {'are' if plural else 'is'} {ref}?", f"Have you seen {ref}?",
                                       f"Where did Grandpa leave {ref}?" if ref.startswith("the ") else f"Where {'are' if plural else 'is'} {ref} now?"]),
                       "kind": "where", "entity": i, "a": self.truth[i], "sor": self.sor.get(i)})
        p = "key bowl"
        qs.append({"q": "What's in the key bowl?", "kind": "what_in", "entity": p, "a": self.holding(self.truth, p),
                   "sor": self.holding(self.sor, p)})
        return self.finish(idx, qs)


# ------------------------------------------------------------------------------------------------ cloud operations
class CloudOps(World):
    name = "ops"
    title = "Cloud platform, a day with an incident"
    n_lines = 240
    SERVICES = ["payments-api", "auth-service", "search-indexer", "checkout-web", "notification-worker",
                "fraud-scorer", "image-resizer", "billing-cron", "recs-model", "chat-gateway"]
    CLUSTERS = ["prod-use1-a", "prod-use1-b", "prod-euw1-a", "canary", "staging"]
    DBS = ["orders-db", "users-db", "ledger-db", "sessions-redis"]
    HOSTS = ["db-07", "db-11", "db-14", "db-19", "db-22"]
    PEOPLE = ["priya", "marcus", "chen.l", "sofia", "tunde", "elena"]

    def build(self, idx):
        r = self.r
        self.tick = clock(0)
        svcs = r.sample(self.SERVICES, 8)
        dbs = r.sample(self.DBS, 3)
        self.items = svcs + [f"{d} primary" for d in dbs]
        self.places = self.CLUSTERS + self.HOSTS
        self.alias = {s: [s.split("-")[0]] for s in svcs}
        for d in dbs:
            self.alias[f"{d} primary"] = [f"{d} leader", f"the {d.split('-')[0]} primary"]
        for c in self.CLUSTERS:
            self.alias[c] = [c.replace("prod-", ""), c.split("-")[-1] if c.startswith("prod") else c]
        self.alias["prod-use1-a"] = ["use1-a", "use1a"]
        self.alias["prod-use1-b"] = ["use1-b", "use1b"]
        self.alias["prod-euw1-a"] = ["euw1-a", "eu-west"]
        self.lines, self.truth, self.sor = [], {}, {}
        for s in svcs:
            c = self.pick(self.CLUSTERS[:3])
            self.add("struct", "deploy", [["move", s, c]],
                     text=f"[{self.tick()}] deploy-bot  {s} v{r.randint(1, 4)}.{r.randint(0, 30)}.{r.randint(0, 9)} live in {c} (100%)")
        for d in dbs:
            h = self.pick(self.HOSTS)
            self.add("struct", "infra", [["move", f"{d} primary", h]], text=f"[{self.tick()}] patroni     {d}: leader is {h}")
        last = None
        while len(self.lines) < self.n_lines:
            u = r.random()
            who = self.pick(self.PEOPLE)
            x = self.pick(self.items)
            cur = self.truth[x]
            is_db = x.endswith("primary")
            dsts = self.HOSTS if is_db else self.CLUSTERS
            if u < 0.22:  # recorded by the deploy bot or the orchestrator
                d = self.pick([p for p in dsts if p != cur])
                text = (f"[{self.tick()}] patroni     {x.split()[0]}: failover, leader {cur} -> {d}" if is_db else
                        f"[{self.tick()}] deploy-bot  {x} v{r.randint(1, 4)}.{r.randint(0, 30)}.{r.randint(0, 9)} live in {d} (100%)")
                self.add("struct", "infra" if is_db else "deploy", [["move", x, d]], text=text)
            elif u < 0.42:  # done by hand during the incident, said only in the incident channel
                d = self.pick([p for p in dsts if p != cur])
                last = x
                self.add("human", "incident", [["move", x, d]], who=who,
                         spec={"mode": "report", "why": self.pick(["noisy neighbour", "latency spike", "disk pressure",
                                                                     "rollback", "capacity", ""])})
            elif u < 0.46 and last:
                d = self.pick([p for p in (self.HOSTS if last.endswith("primary") else self.CLUSTERS) if p != self.truth[last]])
                self.add("human", "incident", [["move", last, d]], who=who, spec={"mode": "follow_up"})
                last = None
            elif u < 0.52:
                other = self.pick([p for p in dsts if p != cur])
                last = last if x == last else None
                self.add("human", "incident", [], who=who, spec={"mode": "plan", "item": x, "place": other})
            elif u < 0.70:
                self.add("struct", "alerts", [], text=self.pick([
                    f"[{self.tick()}] pagerduty   TRIGGERED HighLatency {self.pick(svcs)} p99={r.uniform(0.8, 3.0):.1f}s",
                    f"[{self.tick()}] pagerduty   RESOLVED  HighLatency {self.pick(svcs)}",
                    f"[{self.tick()}] k8s         hpa {self.pick(svcs)} scaled {r.randint(2, 6)} -> {r.randint(6, 12)} replicas",
                    f"[{self.tick()}] k8s         node ip-10-0-{r.randint(1, 9)}-{r.randint(10, 250)} NotReady (kubelet)",
                    f"[{self.tick()}] grafana     disk {self.pick(self.HOSTS)} at {r.randint(70, 96)}%"]))
            else:
                self.add("human", "incident", [], who=who, spec={"mode": "noise"})
        qs = []
        stale = [i for i in self.items if self.sor.get(i) != self.truth[i]]
        pick = r.sample(stale, min(6, len(stale)))
        pick += r.sample([i for i in self.items if i not in pick], min(9 - len(pick), len(self.items) - len(pick)))
        for i in pick:
            if i.endswith("primary"):
                q = self.pick([f"Which host is the {i} on?", f"Where is the {i} right now?"])
            else:
                q = self.pick([f"Where is {i} running?", f"Which cluster is {i} in right now?"])
            qs.append({"q": q, "kind": "where", "entity": i, "a": self.truth[i], "sor": self.sor.get(i)})
        c = self.pick(self.CLUSTERS[:3])
        qs.append({"q": f"What is running in {c} right now?", "kind": "what_in", "entity": c,
                   "a": self.holding(self.truth, c), "sor": self.holding(self.sor, c)})
        return self.finish(idx, qs)


WORLDS = {"warehouse": Warehouse, "biobank": Biobank, "home": HomeRobot, "ops": CloudOps}
