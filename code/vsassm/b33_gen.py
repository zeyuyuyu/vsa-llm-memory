"""B33: fresh bAbI task-2 stories ("two supporting facts": where is the object a person carried?).

Why a re-implementation. The released bAbI generator (facebookarchive/bAbI-tasks, Lua, task WhereIsObject) is, in its
own README's words, "a rewrite of the original code that was used to generate the publicly available dataset ... not
possible to produce exactly the same dataset". For task 2 it differs from the public data in ways that matter here:
its templates have no "picked up", "went back to", "left", "discarded" or "there" and add "let go of"; it never asks
about a dropped object (its question filter needs the container's location, which a location does not have), while
70.5% of the public questions are about dropped objects; and it asks after >= 3 single statements, while the public
stories advance in pairs of statements. So this module keeps the released generator's world logic (a hidden true
world: get valid only if actor and object are co-located and the object is free; drop only by the holder; move to
another location; five questions per story, the story ending at the fifth; the answer read from the world) and takes
the rest from the public data's training split (900 train + 100 validation questions of Muennighoff/babi task 2 =
the official bAbI v1.2 en-valid 1k set; the test split is never read), measured before registration by
b33_gencheck.py:

  * surface: exactly the public templates, uniform within each action (public counts are uniform within 2-3 sd)
      move: "X moved/went/journeyed/travelled/went back to the L."   get: "X got/took/picked up/grabbed the O there."
      drop: "X dropped/left/put down/discarded the O" + " there." with probability P_THERE_DROP, else "."
  * questions (deterministic in the public data, 1000 of 1000 question slots and 1169 of 1169 non-question slots
    explained): the story advances in steps of two statements; after each step a question is asked iff some object
    is ANSWERABLE, i.e. its last get/drop is among the last WINDOW + 1 statements, its place is stated by the text
    (dropped: the dropper's last stated place; held: the holder's last stated place) and, if held, the holder moved
    after the get; the object is chosen uniformly among the answerable ones ("Where is the O?")
  * statements: a person uniform (after a get, the same person again with probability R_GET, after a drop with
    probability R_DROP: in the public data the next statement is by the same person 39% of the time after a get and
    34% after a drop, against 25% after a move); if the person holds objects, a drop of a uniformly chosen held
    object with probability P_DROP; otherwise a get with probability P_GET (P_GET_HOLDER if the person already holds
    something) if some free object can be where the person is (chosen among them with weight W_UNSEEN for an object
    no statement has mentioned yet and 1 otherwise), else a move to a uniformly chosen other location
  * hidden places (INIT; registered: "random"): "random" = every person and object placed uniformly at the start (the
    public data's first pickups of never-mentioned objects by people whose place is stated are spread uniformly over
    the six locations); alternatives tried in the fit: "lazy" (a person or object whose place no event has fixed is
    consistent with any place), "objlazy" (persons random, an unmentioned object can be where its first getter is),
    "basic" (the released world file's object places); "one" was explored before the fit and not used

The gold answer is the world's location of the object; it is asserted equal to the text-only reading (the place the
text determines) for every question, and b33_track gives a second, independent reading.
"""
import argparse, json, random

ACTORS = ["Mary", "Sandra", "John", "Daniel"]
LOCS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
OBJS = ["football", "milk", "apple"]
MOVE_T = ["{a} moved to the {l}.", "{a} went to the {l}.", "{a} journeyed to the {l}.", "{a} travelled to the {l}.",
          "{a} went back to the {l}."]
GET_T = ["{a} got the {o} there.", "{a} took the {o} there.", "{a} picked up the {o} there.",
         "{a} grabbed the {o} there."]
DROP_T = ["{a} dropped the {o}", "{a} left the {o}", "{a} put down the {o}", "{a} discarded the {o}"]

# fitted on the first 100 official training stories (b33_gencheck.py --fit --seed 3, 2026-11-18 ~07:05 UTC;
# results/b33_gencheck_fit.json), frozen at registration
PARAMS = {"INIT": "random", "P_GET": 0.62, "P_GET_HOLDER": 0.71, "P_DROP": 0.55, "R_GET": 0.22, "R_DROP": 0.13,
          "W_UNSEEN": 1.05, "P_THERE_DROP": 0.228, "WINDOW": 4, "STEP": 2, "N_Q": 5}


class World:
    def __init__(self, rng, init, q_cluster=0.0):
        self.q_cluster = q_cluster
        if init == "random":
            self.aloc = {a: rng.choice(LOCS) for a in ACTORS}
            self.oloc = {o: rng.choice(LOCS) for o in OBJS}
        elif init == "basic":                       # the released generator's world_basic.txt objects; actors random
            self.aloc = {a: rng.choice(LOCS) for a in ACTORS}
            self.oloc = {"apple": "kitchen", "football": "garden", "milk": "kitchen"}
        elif init == "cluster":                     # persons random; each object at the previous object's place
            self.aloc = {a: rng.choice(LOCS) for a in ACTORS}   # with probability Q_CLUSTER, else uniform
            self.oloc = {}
            for o in OBJS:
                prev = list(self.oloc.values())
                self.oloc[o] = rng.choice(prev) if prev and rng.random() < self.q_cluster else rng.choice(LOCS)
        elif init == "objlazy":                     # persons placed at random; an object is placed by its first get
            self.aloc = {a: rng.choice(LOCS) for a in ACTORS}
            self.oloc = {o: None for o in OBJS}
        elif init == "one":                         # all objects start together at one random place
            self.aloc = {a: rng.choice(LOCS) for a in ACTORS}
            l0 = rng.choice(LOCS)
            self.oloc = {o: l0 for o in OBJS}
        else:
            self.aloc = {a: None for a in ACTORS}   # hidden true place (None = not yet fixed)
            self.oloc = {o: None for o in OBJS}     # hidden true place of a free object
        self.held = {}                              # object -> holder
        self.k_aloc = {}                            # what the text says: actor -> last stated place
        self.k_rest = {}                            # object -> place where it was dropped (None = not stated)
        self.last_ev = {}                           # object -> (statement index, kind, actor)
        self.moves = {a: [] for a in ACTORS}        # actor -> indices of their moves
        self.t = 0
        self.prev = None                            # (kind, actor) of the last statement

    def can_get(self, a, o):
        if o in self.held:
            return False
        la, lo = self.aloc[a], self.oloc[o]
        return la is None or lo is None or la == lo

    def perform(self, kind, a, x):
        if kind == "move":
            self.aloc[a] = x
            self.k_aloc[a] = x
            self.moves[a].append(self.t)
        elif kind == "get":
            la, lo = self.aloc[a], self.oloc[x]
            if la is None and lo is not None:
                self.aloc[a] = lo                   # the actor must have been where the object lay
            self.oloc[x] = None
            self.held[x] = a
            self.k_rest.pop(x, None)
            self.last_ev[x] = (self.t, kind, a)
        else:
            self.held.pop(x)
            self.oloc[x] = self.aloc[a]             # None if the holder's place was never fixed
            self.k_rest[x] = self.k_aloc.get(a)
            self.last_ev[x] = (self.t, kind, a)
        self.prev = (kind, a)
        self.t += 1

    def true_loc(self, o):
        return self.aloc[self.held[o]] if o in self.held else self.oloc[o]

    def text_loc(self, o):
        if o in self.held:
            return self.k_aloc.get(self.held[o])
        return self.k_rest.get(o)

    def answerable(self, window):
        now = self.t - 1
        out = []
        for o, (j, kind, a) in self.last_ev.items():
            if now - j > window or self.text_loc(o) is None:
                continue
            if kind == "get" and not any(m > j for m in self.moves[a]):
                continue
            out.append(o)
        return sorted(out, key=OBJS.index)


def statement(w, rng, P):
    a = rng.choice(ACTORS)
    if w.prev is not None and w.prev[0] != "move" and rng.random() < P["R_" + w.prev[0].upper()]:
        a = w.prev[1]                               # after a get or drop the same person often acts again
    mine = sorted([o for o, h in w.held.items() if h == a], key=OBJS.index)
    if mine and rng.random() < P["P_DROP"]:
        o = rng.choice(mine)
        w.perform("drop", a, o)
        return rng.choice(DROP_T).format(a=a, o=o) + (" there." if rng.random() < P["P_THERE_DROP"] else ".")
    free = [o for o in OBJS if w.can_get(a, o)]
    if free and rng.random() < (P["P_GET_HOLDER"] if mine else P["P_GET"]):
        o = rng.choices(free, weights=[1.0 if o in w.last_ev else P["W_UNSEEN"] for o in free])[0]
        w.perform("get", a, o)
        return rng.choice(GET_T).format(a=a, o=o)
    dest = rng.choice([l for l in LOCS if l != w.aloc[a]])
    w.perform("move", a, dest)
    return rng.choice(MOVE_T).format(a=a, l=dest)


def story(rng, P=PARAMS):
    """One story as Muennighoff/babi-style rows: passage = every statement before the question, one per line."""
    w = World(rng, P["INIT"], P.get("Q_CLUSTER", 0.0))
    stmts, rows = [], []
    while len(rows) < P["N_Q"]:
        for _ in range(P["STEP"]):
            stmts.append(statement(w, rng, P))
        cand = w.answerable(P["WINDOW"])
        if cand:
            o = rng.choice(cand)
            ans = w.text_loc(o)
            assert ans == w.true_loc(o), "text answer differs from the world"
            rows.append({"passage": "\n".join(stmts) + "\n", "question": "Where is the %s?" % o, "answer": ans})
    return rows


def generate(n_stories, seed, P=PARAMS):
    rng = random.Random(seed)
    out = []
    for si in range(n_stories):
        for qi, r in enumerate(story(rng, P)):
            out.append(dict(r, task=2, story=si, q_in_story=qi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_stories", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = generate(a.n_stories, a.seed)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote %d questions (%d stories, seed %d) -> %s" % (len(rows), a.n_stories, a.seed, a.out))


if __name__ == "__main__":
    main()
