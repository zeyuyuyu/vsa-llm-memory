"""B13 (prereg/B13_prereg.md section 2.2): stories written freely by Qwen3-8B from event lists, labelled for the writer.

python3 b13_distill.py --model <Qwen3-8B snapshot> --domain locations --mode train --n 4000 --seed 0 --out b13_data/loc_0.jsonl
python3 b13_distill.py ... --mode long --ns 40,160 --per_n 40 --out b13_data/long_test.jsonl   # P3 test (fresh names)
(3-6 events per training story and 5 per long-story chunk: Qwen3-8B dropped or reordered events in longer lists)

Each story: an event list (moves, swaps of contents; containers: nesting), Qwen3-8B writes it with annotations (the exact
words used for every event's object and place), and a story is kept only if
  (a) every annotated phrase is the name (optionally with a determiner) or a pronoun, and occurs in its sentence,
  (b) every event appears once, in order,
  (c) pronouns agree with the recency rule the decoder uses ("it"/"them" = the last object written, "there" = the
      last place written), and an annotated pronoun word occurs only as often as it is annotated,
  (d) a separate greedy extraction pass by the same model reproduces every sentence's events (or none).
Names of the story's objects and places that occur in a sentence but are not event arguments get role "none".
Kept stories are converted to the writer's training layout (preamble, numbered sentences, one question).
"""
import argparse
import json
import os
import random
import re
import time

from b13_rules import OBJ_PRO, PLACE_PRO, note, pair_events, resolve
from b9_data import POOLS

SCENES = ["a kitchen", "an office", "a garage", "a camping trip", "a classroom", "a hotel room", "a workshop",
          "a garden", "a dorm room", "a small shop", "a restaurant kitchen", "an art studio", "a laboratory",
          "a hospital ward", "a warehouse", "a family home", "a boat", "a library", "a music room", "a bakery",
          "a farm", "a gym", "a beach day", "a move to a new flat", "a birthday party", "a school trip",
          "a doctor's office", "a car journey", "a picnic", "a hardware store"]
PREAMBLES = {
    "locations": ["There are several places: {places}. Every place starts empty. The following actions happen in order. "
                  "Putting X in P moves X from wherever it is into P. Swapping the contents of P and Q moves everything "
                  "that is in P into Q and everything that is in Q into P, at the same time.",
                  "The story below mentions these places: {places}. Read it in order and keep track of where every "
                  "object is.",
                  "Keep track of where things are. Places: {places}."],
    "containers": ["There are several containers, named by colour and kind. They start out separate. The following moves "
                   "happen in order. Putting A into B moves A, together with everything inside it, into B; A is no "
                   "longer in whatever held it before.",
                   "Keep track of what is inside what. Putting one thing into another moves it with everything inside "
                   "it."],
}
DETS = r"(?:the|a|an|his|her|their|its|my|your|our)\s+"


# ---------------------------------------------------------------------------------------------- event lists
def make_events(domain, rng, n_ev, n_obj, n_place, p_swap=0.12):
    """Locations: ("write", oi, pi) / ("swap", pi, pj). Containers: ("write", ai, bi) on one item list, no cycles."""
    evs = []
    if domain == "locations":
        loc = {}
        while len(evs) < n_ev:
            filled = sorted(set(loc.values()))
            if filled and rng.random() < p_swap and n_place > 1:
                a = rng.choice(filled)
                b = rng.choice([p for p in range(n_place) if p != a])
                evs.append(("swap", a, b))
                loc = {o: (b if p == a else a if p == b else p) for o, p in loc.items()}
                continue
            o = rng.randrange(n_obj)
            p = rng.choice([p for p in range(n_place) if p != loc.get(o)])
            evs.append(("write", o, p))
            loc[o] = p
        return evs
    parent = {}

    def inside(a, b):  # is a (transitively) inside b?
        while a in parent:
            a = parent[a]
            if a == b:
                return True
        return False
    tries = 0
    while len(evs) < n_ev and tries < 1000:
        tries += 1
        a, b = sorted(rng.sample(range(n_obj), 2))  # items are listed smallest first: a smaller one goes into a larger
        if parent.get(a) == b or inside(b, a):
            continue
        evs.append(("write", a, b))
        parent[a] = b
    return evs


def describe(domain, ev, objs, places):
    op, a, b = ev
    if domain == "containers":
        return f"{objs[a]} is put into {objs[b]}"
    if op == "swap":
        return (f"the contents of {places[a]} and {places[b]} are swapped (everything in {places[a]} goes to "
                f"{places[b]} and everything in {places[b]} goes to {places[a]})")
    return f"{objs[a]} is put in (or on) {places[b]}"


def write_prompt(domain, evs, objs, places, rng, scene, context="", first_id=1, only=None):
    lines = "\n".join(f"{first_id + i}. {describe(domain, e, objs, places)}" for i, e in enumerate(evs))
    k = rng.randint(1, 3)
    cont = (f"\nThis continues an earlier part of the story, which ended:\n\"{context}\"\nContinue it naturally; do not "
            f"repeat it.\n") if context else ""
    no, npl = only or (objs, places)  # long stories: only the names this chunk uses
    names = ", ".join(no) + ("" if domain == "containers" else "; places: " + ", ".join(npl))
    extra = ("\n- Putting one item into another moves everything inside it too; you do not need to say so."
             if domain == "containers" else "")
    return f"""Write a short, natural story in plain English, in the past tense, set in {scene}. It must describe these events, in this order:
{lines}
{cont}
Rules:
- Every event appears exactly once and in this order. You may describe two or three events in one sentence.
- Vary the wording: use different verbs (put, moved, tossed, hung, stashed, dropped, slid, packed, left, carried, placed...), and sometimes the passive voice or the place first.
- You may refer to an object with "it" or "them", or to a place with "there", but only when it is the object or place mentioned most recently.
- Add {k} sentence(s) or clause(s) about unrelated things people do (talking, eating, phone calls, chores) that do not move any listed object. You may say where an object was taken from; then add "from": "<the exact words for that place>" to that event.
- Never move a listed object except in the listed events, and never say an event happened that is not listed.
- You may mention a listed object without moving it (someone looks for it, opens it), but never move a listed object anywhere except in the listed events.
- Use these names exactly as written (you may put "the", "a", "his", "her" before them): {names}.{extra}

Return only JSON, no other text, in this form:
{{"sentences": [{{"text": "...", "events": [{{"id": 1, "object": "...", "place": "..."}}]}}]}}
"object" and "place" must be the exact words used in that sentence for the event's object and place: the name, or "it", "them" or "there". For a swap event use {{"id": 2, "place_a": "...", "place_b": "..."}}. A sentence with no listed event has "events": []."""


def extract_prompt(domain, sentences, objs, places):
    story = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    if domain == "containers":
        what = (f"Items: {', '.join(objs)}.\nFor each numbered sentence, list every event in which one of these items is "
                f"put into another item or moved anywhere else, as [\"move\", item, where_it_went] (where_it_went may be "
                f"any item or place, listed or not).")
    else:
        what = (f"Objects: {', '.join(objs)}.\nPlaces: {', '.join(places)}.\nFor each numbered sentence, list every event "
                f"in which one of these objects is put or moved anywhere (to a listed place or not), as [\"move\", object, "
                f"place], and every exchange of the contents of two places, as [\"swap\", place, place].")
    return f"""Story:
{story}

{what}
Use the names from the lists, not pronouns. Do not count where something was taken from, or plans and things that did not happen.
Return only JSON, one key per sentence number, for example {{"1": [["move", "x", "y"]], "2": []}}."""


# ---------------------------------------------------------------------------------------------- checks + labels
def parse_json(txt):
    txt = txt.strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def phrase_ok(phrase, name, pronouns):
    p = phrase.strip().lower()
    p = re.sub("^" + DETS, "", p)
    return p == name.lower() or p in pronouns


def find_word(text, word, start=0):
    m = re.compile(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", re.I).search(text, start)
    return (m.start(), m.end()) if m else None


def check_story(domain, js, evs, objs, places):
    """Returns ((sentences, per-sentence labelled mentions, per-sentence gold events), "ok") or (None, reason).
    Roles: x / y (a move: object, place), sa / sb (a swap of contents), nx / ny (a story name that is not an argument
    of an event in that sentence: an object, or a place such as where something was taken from)."""
    if not js or not isinstance(js.get("sentences"), list) or not js["sentences"]:
        return None, "json"
    sents, labels, gold, nxt = [], [], [], 0
    last = {"obj": None, "place": None}  # the pronoun rule: the most recently MENTIONED object / place
    onames = set(objs)
    names = sorted({*objs, *places} if domain == "locations" else set(objs), key=len, reverse=True)
    for s in js["sentences"]:
        if not isinstance(s, dict) or not isinstance(s.get("text"), str) or not isinstance(s.get("events", []), list):
            return None, "json"
        text = s["text"].strip()
        if not text or "\n" in text:
            return None, "text"
        ments, ev_here, pron_used = [], [], {}
        cur = {"o": 0, "p": 0}

        def find_free(w, start):  # the first occurrence at or after start that no labelled mention overlaps
            for m in re.finditer(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", text, re.I):
                if m.start() >= start and not any(m.start() < b and m.end() > a for a, b, _, _ in ments):
                    return m.start(), m.end()
            return None

        def place(w, role, ref, kind):
            hit = find_free(w, cur[kind])
            if hit:
                ments.append((hit[0], hit[1], role, ref))
                cur[kind] = hit[1]
                return True
            # the model annotated a name where the sentence uses a pronoun ("... and then moved it to the shelf")
            for pw in (OBJ_PRO if role == "x" else PLACE_PRO if role == "y" and domain == "locations" else ()):
                hit = find_free(pw, cur[kind])
                if hit:
                    ments.append((hit[0], hit[1], role, ref))
                    cur[kind] = hit[1]
                    pron_used[pw] = pron_used.get(pw, 0) + 1
                    return True
            # one place phrase shared by two moves ("the keys and the wallet in the drawer")
            if role == "y" and any(m[2] == "y" and m[3] == ref for m in ments):
                return True
            return False

        for a in s.get("events", []):
            try:
                aid = int(a.get("id")) if isinstance(a, dict) else -1
            except (TypeError, ValueError):
                aid = -1
            if aid != nxt + 1 or nxt >= len(evs):
                return None, "order"
            op, i, j = evs[nxt]
            if op == "swap":
                pa, pb = str(a.get("place_a", "")), str(a.get("place_b", ""))
                if phrase_ok(pa, places[i], ()) and phrase_ok(pb, places[j], ()):
                    refs = (places[i], places[j])
                elif phrase_ok(pa, places[j], ()) and phrase_ok(pb, places[i], ()):
                    refs = (places[j], places[i])
                else:
                    return None, "phrase"
                for role, ph, ref in (("sa", pa, refs[0]), ("sb", pb, refs[1])):
                    if not place(re.sub("^" + DETS, "", ph.strip(), flags=re.I), role, ref, "p"):
                        return None, "missing"
                ev_here.append(("swap", places[i], places[j]))
            else:
                xo = objs[i]
                yo = objs[j] if domain == "containers" else places[j]
                po, pp = str(a.get("object", "")), str(a.get("place", ""))
                if not phrase_ok(po, xo, OBJ_PRO) or not phrase_ok(pp, yo, PLACE_PRO if domain == "locations" else ()):
                    return None, "phrase"
                for role, ph, ref, kind in (("x", po, xo, "o"), ("y", pp, yo, "p")):
                    w = re.sub("^" + DETS, "", ph.strip(), flags=re.I)
                    if w.lower() in OBJ_PRO + PLACE_PRO:
                        pron_used[w.lower()] = pron_used.get(w.lower(), 0) + 1
                    if not place(w, role, ref, kind):
                        return None, "missing"
                src = a.get("from")
                if src:  # where it was taken from: a non-argument place
                    w = re.sub("^" + DETS, "", str(src).strip(), flags=re.I)
                    hit = find_word(text, w, 0)
                    if not hit or w.lower() not in {n.lower() for n in names}:
                        return None, "from"
                    if not any(m[0] < hit[1] and m[1] > hit[0] for m in ments):
                        ments.append((hit[0], hit[1], "ny", next(n for n in names if n.lower() == w.lower())))
                ev_here.append(("write", xo, yo))
            nxt += 1
        for pw, c in pron_used.items():  # an annotated pronoun word may not also occur unannotated
            if len(re.findall(r"(?<![\w-])" + pw + r"(?![\w-])", text, re.I)) != c:
                return None, "pronoun_extra"
        taken = [(a, b) for a, b, _, _ in ments]
        for nm in names:  # other occurrences of story names
            for m in re.finditer(r"(?<![\w-])" + re.escape(nm) + r"(?![\w-])", text, re.I):
                if not any(m.start() < b and m.end() > a for a, b in taken):
                    role = "nx" if (domain == "containers" or nm in onames) else "ny"
                    ments.append((m.start(), m.end(), role, nm))
                    taken.append((m.start(), m.end()))
        ments.sort()
        for a0, a1, role, ref in ments:  # pronouns must agree with the decoder's rule
            if resolve(text[a0:a1], role, last) != (ref if text[a0:a1].lower() in OBJ_PRO + PLACE_PRO else text[a0:a1]) \
                    and text[a0:a1].lower() in OBJ_PRO + PLACE_PRO:
                return None, "pronoun"
            note(ref, role, last)
        # the labels must decode back to this sentence's events under the decoder's pairing rule
        got = [(op, a, b) if op == "write" else ("swap", *sorted([a, b]))
               for _, op, a, b, _ in pair_events([(a0, ref, role, 1.0) for a0, _, role, ref in ments])]
        want = [(op, a, b) if op == "write" else ("swap", *sorted([a, b])) for op, a, b in ev_here]
        if got != want:
            return None, "decode"
        sents.append(text)
        labels.append(ments)
        gold.append(ev_here)
    if nxt != len(evs):
        return None, "count"
    return (sents, labels, gold), "ok"


def check_extraction(js, gold, objs, places):
    """The extraction must reproduce each sentence's events. Ignored: moves of untracked things, and "moves" that leave
    an object where it already is (state statements such as "the spoon was still in the bread box")."""
    if not isinstance(js, dict):
        return False
    state = {}
    low = {n.lower(): n for n in [*objs, *places]}
    tracked = {n.lower() for n in objs}
    for i, g in enumerate(gold):
        after = dict(state)
        for o, x, y in g:
            if o == "write":
                after[x] = y
            else:
                after = {k: (y if v == x else x if v == y else v) for k, v in after.items()}
        got = js.get(str(i + 1), [])
        if not isinstance(got, list):
            return False
        norm = []
        for e in got:
            if not isinstance(e, list) or len(e) != 3:
                return False
            op = "swap" if str(e[0]).lower() == "swap" else "write"
            if op == "write" and re.sub("^" + DETS, "", str(e[1]).lower().strip()) not in tracked:
                continue  # moves of untracked things ("left the pliers on the table") do not matter
            a = low.get(re.sub("^" + DETS, "", str(e[1]).lower().strip()))
            b = low.get(re.sub("^" + DETS, "", str(e[2]).lower().strip()))
            if op == "write" and (a, b) not in [(x, y) for o, x, y in g if o == "write"] and a is not None \
                    and b is not None and (state.get(a) == b or after.get(a) == b):
                continue  # a no-op "move": the object is already there
            norm.append((op, a, b) if op == "write" else ("swap", *sorted([str(a), str(b)])))
        want = [(op, a, b) if op == "write" else ("swap", *sorted([a, b])) for op, a, b in g]
        if sorted(map(str, norm)) != sorted(map(str, want)):
            return False
        state = after
    return True


def final_state(domain, evs, objs, places):
    loc = {}
    for op, a, b in evs:
        if op == "swap":
            loc = {o: (places[b] if p == places[a] else places[a] if p == places[b] else p) for o, p in loc.items()}
        else:
            loc[objs[a]] = objs[b] if domain == "containers" else places[b]
    return loc


def to_episode(domain, rng, sents, labels, objs, places, final, where_only=False):
    """Writer training layout: preamble, numbered sentences (one span each), a question."""
    pl = list(places) if domain == "locations" else []
    rng.shuffle(pl)
    pre = rng.choice(PREAMBLES[domain])
    pre = pre.format(places=", ".join(f"the {p}" for p in pl) if pl and rng.random() < 0.7 else "a few")
    text, out = pre + "\n\n", []
    for i, (s, ms) in enumerate(zip(sents, labels)):
        prefix = f"{i + 1}. "
        start = len(text) + len(prefix)
        out.append({"span": (start, start + len(s) - 1), "op": "write",
                    "mentions": [(start + a, start + b - 1, role, nm) for a, b, role, nm in ms]})
        text += prefix + s + "\n"
    text += "\n"
    placed = sorted(final)
    if domain == "locations" and not where_only and rng.random() < 0.3:
        p = rng.choice(sorted(set(final.values())))
        q = rng.choice([f"What is in the {p}?", f"What is in the {p} now?", f"What is on the {p}?"])
        qname = p
    else:
        o = rng.choice(placed)
        q = rng.choice([f"Where is the {o} now?", f"Where is the {o}?", f"Where did the {o} end up?"])
        qname = o
    a = len(text) + q.index(qname)
    out.append({"span": (len(text), len(text) + len(q) - 1), "op": "question",
                "mentions": [(a, a + len(qname) - 1, "q", qname)]})
    text += q
    return {"text": text, "sents": out, "domain": domain, "final": final, "question": q}


# ---------------------------------------------------------------------------------------------- generation
class LM:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                          attn_implementation="sdpa").eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]

    def gen(self, prompts, sample, max_new=1200):
        from transformers import GenerationConfig
        gc = GenerationConfig(max_new_tokens=max_new, do_sample=sample, temperature=0.8 if sample else None,
                              top_p=0.95 if sample else None, top_k=None, eos_token_id=self.eos,
                              pad_token_id=self.tok.pad_token_id)
        self.model.generation_config = gc  # see b6_qwen.py: a passed config's defaults get overridden
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc)
        return [self.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


def sample_names(domain, rng, mode, n_obj, n_place):
    if domain == "locations":
        po, pp = ("obj_test2", "place_test2") if mode == "long" else ("obj_train", "place_train")
        return rng.sample(POOLS[po], n_obj), rng.sample(POOLS[pp], n_place)
    sfx = "test2" if mode == "long" else "train2"
    items = [f"{c} {k}" for c in POOLS[f"colour_{sfx}"] for k in POOLS[f"kind_{sfx}"]]
    return rng.sample(items, n_obj), []


def free_names_prompt(domain, scene, n_obj, n_place):
    if domain == "containers":
        return (f"List {n_obj} different everyday things found in {scene}, ordered from smallest to largest, where each "
                f"one could fit inside the larger ones (for example a ring, a small box, a bag, a suitcase). Each name "
                f"1-3 words, lowercase, no name inside another. Return only JSON: {{\"items\": [...]}}")
    return (f"List {n_obj} different small everyday objects and {n_place} different places to put them (furniture, "
            f"containers, rooms, spots) found in {scene}. Each name 1-3 words, lowercase, no name inside another. Return "
            f"only JSON: {{\"objects\": [...], \"places\": [...]}}")


def names_ok(objs, places):
    allp = [n.lower() for n in [*objs, *places]]
    if len(set(allp)) != len(allp) or any(not re.fullmatch(r"[a-z][a-z' -]{1,30}", n) for n in allp):
        return False
    bad = set(OBJ_PRO + PLACE_PRO)
    return not any(a != b and re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", b) for a in allp for b in allp) \
        and not bad & set(allp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--domain", choices=["locations", "containers"], required=True)
    ap.add_argument("--mode", choices=["train", "long"], default="train")
    ap.add_argument("--n", type=int, default=1000)  # stories to keep (train)
    ap.add_argument("--ns", default="40,160")  # long: story lengths
    ap.add_argument("--per_n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--max_fails", type=int, default=6)  # long: attempts per chunk before a story is dropped
    ap.add_argument("--idx", default="")  # long: "lo-hi" story indices to write (shards)
    ap.add_argument("--k", type=int, default=1)  # long: candidates per chunk per round
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(10_000 + args.seed)
    lm = LM(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stats = {}
    t0 = time.time()
    if args.mode == "train":
        kept = sum(1 for _ in open(args.out)) if os.path.exists(args.out) else 0
        f = open(args.out, "a")
        rej = open(args.out + ".rejects", "a")  # rejected stories with the reason, for inspecting the checks
        while kept < args.n:
            jobs = []
            for _ in range(args.bs):
                scene = rng.choice(SCENES)
                n_obj = rng.randint(2, 5)
                n_place = rng.randint(2, 5) if args.domain == "locations" else 0
                if args.domain == "containers":
                    n_obj = rng.randint(3, 6)
                jobs.append({"scene": scene, "n_obj": n_obj, "n_place": n_place,
                             "n_ev": rng.randint(3, 6), "free": rng.random() < 0.5})  # longer lists: events dropped
            free = [j for j in jobs if j["free"]]
            if free:
                outs = lm.gen([free_names_prompt(args.domain, j["scene"], j["n_obj"], j["n_place"]) for j in free],
                              sample=True, max_new=200)
                for j, o in zip(free, outs):
                    js = parse_json(o) or {}
                    objs = [str(x).lower().strip() for x in js.get("items" if args.domain == "containers" else "objects", [])]
                    places = [str(x).lower().strip() for x in js.get("places", [])]
                    if len(objs) >= j["n_obj"] and (args.domain == "containers" or len(places) >= j["n_place"]):
                        j["objs"], j["places"] = objs[:j["n_obj"]], places[:j["n_place"]]
            for j in jobs:
                if "objs" not in j:
                    j["objs"], j["places"] = sample_names(args.domain, rng, "train", j["n_obj"], j["n_place"])
                    j["free"] = False
                if not names_ok(j["objs"], j["places"]):
                    j["skip"] = "names"
                    continue
                j["evs"] = make_events(args.domain, rng, j["n_ev"], j["n_obj"], j["n_place"])
            live = [j for j in jobs if "skip" not in j]
            outs = lm.gen([write_prompt(args.domain, j["evs"], j["objs"], j["places"], rng, j["scene"]) for j in live],
                          sample=True)
            cand = []
            for j, o in zip(live, outs):
                res, why = check_story(args.domain, parse_json(o), j["evs"], j["objs"], j["places"])
                stats[why] = stats.get(why, 0) + 1
                if res:
                    j["res"], j["raw"] = res, o
                    cand.append(j)
                else:
                    rej.write(json.dumps({"why": why, "events": [describe(args.domain, e, j["objs"], j["places"])
                                                                for e in j["evs"]], "raw": o}) + "\n")
            if cand:
                outs = lm.gen([extract_prompt(args.domain, j["res"][0], j["objs"], j["places"]) for j in cand],
                              sample=False, max_new=600)
                for j, o in zip(cand, outs):
                    if not check_extraction(parse_json(o), j["res"][2], j["objs"], j["places"]):
                        stats["extract"] = stats.get("extract", 0) + 1
                        rej.write(json.dumps({"why": "extract", "sentences": j["res"][0], "gold": j["res"][2],
                                              "extracted": o}) + "\n")
                        continue
                    stats["kept"] = stats.get("kept", 0) + 1
                    sents, labels, gold = j["res"]
                    final = final_state(args.domain, j["evs"], j["objs"], j["places"])
                    ep = to_episode(args.domain, rng, sents, labels, j["objs"], j["places"], final)
                    ep.update({"scene": j["scene"], "free_names": j["free"], "events": [list(e) for e in j["evs"]],
                               "raw": j["raw"],
                               "objs": j["objs"], "places": j["places"], "sentences": sents})
                    f.write(json.dumps(ep) + "\n")
                    kept += 1
            f.flush()
            rej.flush()
            print(f"kept {kept}/{args.n}  {stats}  t={time.time() - t0:.0f}s", flush=True)
        return
    # long test: stories written chunk by chunk (5 events per chunk, all stories batched), fresh test-pool names (T2)
    f = open(args.out, "a")
    done = {(r["n"], r["idx"]) for r in map(json.loads, open(args.out))} if os.path.getsize(args.out) else set()
    stories = []  # a story's pronouns may not reach back into the previous chunk: check_story starts each chunk fresh
    lo, hi = (int(x) for x in args.idx.split("-")) if args.idx else (0, args.per_n - 1)
    for n in [int(x) for x in args.ns.split(",")]:
        for idx in range(lo, hi + 1):
            if (n, idx) in done:
                continue
            srng = random.Random(f"long-{n}-{idx}")
            objs, places = sample_names("locations", srng, "long", 16, 16)
            evs = make_events("locations", srng, n, 16, 16)
            stories.append({"n": n, "idx": idx, "rng": srng, "objs": objs, "places": places, "evs": evs,
                            "scene": srng.choice(SCENES), "chunks": [evs[i:i + 5] for i in range(0, n, 5)],
                            "next": 0, "fails": 0, "s": [], "l": []})
    while True:
        live = [st for st in stories if st["next"] < len(st["chunks"]) and st["fails"] < args.max_fails]
        if not live:
            break
        jobs = []  # args.k candidates per story per round; the writer sees only the names its chunk uses
        for st in live:
            ch = st["chunks"][st["next"]]
            co = sorted({st["objs"][e[1]] for e in ch if e[0] == "write"})
            cp = sorted({st["places"][e[2]] for e in ch if e[0] == "write"} | {st["places"][i] for e in ch if e[0] == "swap"
                                                                             for i in e[1:]})
            for _ in range(args.k):
                jobs.append((st, write_prompt("locations", ch, st["objs"], st["places"], st["rng"], st["scene"],
                                              " ".join(st["s"][-2:]), only=(co, cp))))
        outs = []
        for k in range(0, len(jobs), args.bs):
            outs += lm.gen([p for _, p in jobs[k:k + args.bs]], sample=True)
        passed = {}
        for (st, _), o in zip(jobs, outs):  # event ids restart at 1 in every chunk
            res, why = check_story("locations", parse_json(o), st["chunks"][st["next"]], st["objs"], st["places"])
            stats[why] = stats.get(why, 0) + 1
            if res and len(passed.setdefault(id(st), [])) < 2:
                passed[id(st)].append((st, res))
        cand = [c for v in passed.values() for c in v]
        ex = []
        for k in range(0, len(cand), args.bs):
            ex += lm.gen([extract_prompt("locations", res[0], st["objs"], st["places"]) for st, res in cand[k:k + args.bs]],
                         sample=False, max_new=600)
        advanced = set()
        for (st, res), o in zip(cand, ex):
            if id(st) in advanced:
                continue
            if check_extraction(parse_json(o), res[2], st["objs"], st["places"]):
                st["s"] += res[0]
                st["l"] += res[1]
                st["next"] += 1
                st["fails"] = 0
                advanced.add(id(st))
            else:
                stats["extract"] = stats.get("extract", 0) + 1
        for st in live:
            if id(st) not in advanced:
                st["fails"] += args.k
        print(f"long: {sum(st['next'] == len(st['chunks']) for st in stories)}/{len(stories)} stories complete, "
              f"chunks {sum(st['next'] for st in stories)}/{sum(len(st['chunks']) for st in stories)}  {stats}  "
              f"t={time.time() - t0:.0f}s", flush=True)
    for st in stories:
        if st["next"] < len(st["chunks"]):
            print(f"n={st['n']} idx={st['idx']}: a chunk failed {args.max_fails} times; story dropped", flush=True)
            continue
        final = final_state("locations", st["evs"], st["objs"], st["places"])
        ep = to_episode("locations", st["rng"], st["s"], st["l"], st["objs"], st["places"], final, where_only=True)
        ep.update({"n": st["n"], "idx": st["idx"], "events": [list(e) for e in st["evs"]], "objs": st["objs"],
                   "places": st["places"], "sentences": st["s"]})
        f.write(json.dumps(ep) + "\n")
    f.close()


if __name__ == "__main__":
    main()
