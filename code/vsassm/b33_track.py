"""B33: a text-only world tracker for bAbI task 2, written independently of the memory route's parse (v37b) and of
the generator's world (b33_gen.World). It reads a story statement by statement and keeps what the text says: each
person's last stated location, each object's holder (from a get) or resting place (from a drop: the dropper's
location at that time, if stated). answer(obj) = the holder's location if held, else the resting place.

Used for: (1) checking that its semantics reproduce every official training answer (1000/1000 before registration);
(2) the second scorer of the held-out labels (every generated answer must equal the tracker's); (3) the
distribution statistics of b33_gencheck.py.
"""
import re

MOVE = re.compile(r"^(\w+) (moved|went|journeyed|travelled|went back) to the (\w+)\.$")
GET = re.compile(r"^(\w+) (got|took|picked up|grabbed) the (\w+)( there)?\.$")
DROP = re.compile(r"^(\w+) (dropped|left|put down|discarded) the (\w+)( there)?\.$")
QUESTION = re.compile(r"^Where is the (\w+)\?$")


def parse_stmt(s):
    """(kind, actor, object_or_location, verb, there) for one statement; raises on anything else."""
    for pat, k in ((MOVE, "move"), (GET, "get"), (DROP, "drop")):
        m = pat.match(s)
        if m:
            return (k, m.group(1), m.group(3), m.group(2), bool(m.group(4)) if k != "move" else False)
    raise ValueError("unknown statement: %r" % s)


def stories(rows):
    """Group consecutive Muennighoff-style rows (passage = statements so far, question, answer) into stories:
    {"stmts": [...], "qs": [(n_statements_before, object, answer)]}."""
    out, cur = [], None
    for r in rows:
        st = [x.strip() for x in r["passage"].strip().split("\n")]
        if cur is not None and len(st) >= len(cur["stmts"]) and st[:len(cur["stmts"])] == cur["stmts"]:
            cur["stmts"] = st
        else:
            cur = {"stmts": st, "qs": []}
            out.append(cur)
        cur["qs"].append((len(st), QUESTION.match(r["question"].strip()).group(1), r["answer"]))
    return out


class Tracker:
    def __init__(self):
        self.loc, self.held, self.rest = {}, {}, {}

    def step(self, s):
        k, a, x, v, th = parse_stmt(s)
        info = {"kind": k, "actor": a, "arg": x, "verb": v, "there": th, "actor_loc_known": a in self.loc}
        if k == "move":
            self.loc[a] = x
        elif k == "get":
            info["obj_unseen"] = x not in self.held and x not in self.rest
            info["n_held_by_actor"] = sum(1 for h in self.held.values() if h == a)
            self.held[x] = a
            self.rest.pop(x, None)
        else:
            info["holds"] = self.held.get(x) == a
            self.held.pop(x, None)
            self.rest[x] = self.loc.get(a)
        return info

    def answer(self, o):
        if o in self.held:
            return self.loc.get(self.held[o]), "held"
        if o in self.rest:
            return self.rest[o], "dropped"
        return None, "unknown"


def answer_question(passage, question):
    """Answer one Muennighoff-style row from its text alone."""
    T = Tracker()
    for s in passage.strip().split("\n"):
        T.step(s.strip())
    return T.answer(QUESTION.match(question.strip()).group(1))[0]
