"""B48 structure layer (prereg/B48_one_form_and_alias_naming_prereg.md).

Part A, arena C1: arena C (v13.Arena, b31_struct.build_C) with ONE surface form per symbol. Every symbol is written in
decimal, as a role ("[R63]", "[F63]") and as a filler ("{63}"), and in the question ("Q [R63] {50} ?"). In arena C the
roles are hexadecimal ("[R3F]") and the fillers decimal ("{63}"). Nothing else changes: the same random streams, the
same chains, labels, candidates, pair orders and question position; only the rendering of the role digits differs
(h2(a) -> str(a)). Mentions keep b31_struct's layout (level-0 pairs, level-1 pairs, then query and candidate), so the
B31 / B42 arms run unchanged once b31_struct.build is replaced by build_patched (see patch()).

Part B, alias stories: two-level stories over K = 16 entities whose level-1 subject is written by a name that is not
the entity's listed name.
  W  Wikidata aliases (b32_data/entities.json, PopQA subjects): the level-1 subject is a Wikidata alias of the entity
     that shares no normalised word with its listed name; identity is not in the surface string (world knowledge only).
  P  pronouns, bAbI-style (task 11 "basic coreference"): "<Name> went to the <place>. Then <he|she> visited <C>."; the
     pronoun refers to the subject of the sentence just before it; identity comes from the context only.
Entities (W) and first names (P) are split by a hash into train / dev / test pools, so dev and test stories contain only
entities never seen in training.

Mention tuple (B): (kind, ent, cs, ce, mtype) with ent = roster slot 0..15 and mtype in
  roster  the entity in the list at the start                    can   a level-0 subject / object or level-1 object
  alias   W level-1 subject (Wikidata alias)                     pron  P level-1 subject (pronoun)
  ante    P antecedent ("<Name> went to the <place>.")           q / cand  the question's two names
Structure: lv[0] = 16 and lv[1] = N1 = 12 (subject, object) mention-index pairs; q, cand; chain = [q, x1, x2] slots.
"""
import hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b27_common as BC                                                     # noqa: E402
import b31_struct as S31                                                    # noqa: E402

# ====================================================================== part A: arena C1 (one surface form)

PAT1 = re.compile(r"\[([RFGH])(\d+)\]\{(\d+)\}")
QPAT1 = re.compile(r"Q \[R(\d+)\] \{(\d+)\} \?")


def _c1_struct(text, label, q, cand, V):
    men, lv = [], [[] for _ in range(V.HOPS)]
    for m in PAT1.finditer(text):
        L = V.LVL.index(m.group(1))
        if L >= V.HOPS:
            continue
        a, b = int(m.group(2)), int(m.group(3))
        men.append(((L, a, "r"), "R", a, m.start(2), m.end(2)))
        men.append(((L, a, "f"), "F", b, m.start(3), m.end(3)))
        lv[L].append((len(men) - 2, len(men) - 1))
    qm = QPAT1.search(text)
    assert int(qm.group(1)) == q and int(qm.group(2)) == cand
    men.append((("q",), "R", q, qm.start(1), qm.end(1)))
    men.append((("cand",), "F", cand, qm.start(2), qm.end(2)))
    assert all(len(l) == V.N_CTX for l in lv)
    return {"text": text, "label": int(label), "mentions": men, "lv": lv, "q": len(men) - 2, "cand": len(men) - 1}


def build_C1(split):
    """Line by line b31_struct.build_C (= v13.Arena.__init__), with the role digits written in decimal."""
    M = BC.load_arena_modules()
    V = M["v13"]
    n, seed = {"train": (BC.N_SPLIT["C"][0], 42), "dev": (BC.N_SPLIT["C"][1], BC.DEV_SEED_C),
               "test": (BC.N_SPLIT["C"][2], 4242)}[split]
    rng = random.Random(seed)
    out = []
    for i in range(n):
        chain = [rng.randrange(V.N_SYM) for _ in range(V.HOPS + 1)]
        levels, toks = [], []
        for L in range(V.HOPS):
            pairs, seen = [(chain[L], chain[L + 1])], {chain[L]}
            while len(pairs) < V.N_CTX:
                a = rng.randrange(V.N_SYM)
                if a in seen:
                    continue
                seen.add(a); pairs.append((a, rng.randrange(V.N_SYM)))
            rng.shuffle(pairs); levels.append(pairs)
        for L, pairs in enumerate(levels):
            for a, b in pairs:
                toks.append("[%s%d]{%d}" % (V.LVL[L], a, b))                  # C: "[%s%s]{%s}" % (LVL[L], h2(a), b)
        label = 1 if rng.random() < 0.5 else 0
        cand = chain[-1] if label == 1 else rng.choice([s for s in range(V.N_SYM) if s != chain[-1]])
        q = "Q [R%d] {%d} ?" % (chain[0], cand)                                # C: "Q [R%s] {%d} ?" % (h2(chain[0]), cand)
        assert not V.Q_AT_END
        text = " ".join(toks[:len(toks) // 2]) + " " + q + " " + " ".join(toks[len(toks) // 2:])
        s = _c1_struct(text, label, chain[0], cand, V)
        s["chain"] = chain
        out.append(s)
    return out


_ORIG_BUILD = S31.build


def build_patched(arena, split, view2=False):
    """b31_struct.build with arena C replaced by C1 (no second view is defined for C1)."""
    if arena == "C":
        assert not view2, "C1 has no reshuffled second view"
        return build_C1(split)
    return _ORIG_BUILD(arena, split, view2)


def patch():
    """Make every B31 / B42 module that calls b31_struct.build read arena C1 instead of arena C."""
    S31.build = build_patched


# ====================================================================== part B: alias stories

K = 16
N1 = 12                                   # level-1 subjects per story (a subset of the K entities that contains x1)
SPLITS_B = {"train": (800, 4801), "dev": (500, 4802), "test": (500, 4803)}   # stories, generator seed
PLACES = ["kitchen", "garden", "office", "hallway", "bathroom", "bedroom", "cellar", "park", "school", "cinema"]


def norm(s):
    """b32_data.norm: lower case, every run of non-alphanumerics -> one space."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def split_of(key, salt):
    h = int(hashlib.sha256((salt + "|" + key).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 30 else ("dev" if h < 50 else "test")


def _ok_string(s):
    return 2 <= len(s) <= 60 and ";" not in s and "\n" not in s and "|" not in s and re.search(r"[A-Za-z]", s) \
        and norm(s) != ""


def w_pool(split):
    """W entities of a split (b32_data's entity split): listed name = PopQA label; candidate aliases = Wikidata aliases
    sharing no normalised word with the label (b32's ambiguity filter already applied). Entities with at least one
    such alias are kept."""
    d = json.load(open(os.path.join(HERE, "b32_data", "entities.json")))["entities"]
    out = []
    for e in d:
        if e["split"] != split or not _ok_string(e["name"]):
            continue
        nw = set(norm(e["name"]).split())
        al = [a for a in e["aliases"] if _ok_string(a) and not (set(norm(a).split()) & nw)]
        if al:
            out.append({"id": e["uri"].rsplit("/", 1)[-1], "name": e["name"], "aliases": al})
    return out


FEMALE = """Mary Sandra Julie Emma Olivia Sophia Isabella Mia Charlotte Amelia Harper Evelyn Abigail Emily Elizabeth Sofia
Avery Ella Scarlett Grace Chloe Victoria Riley Aria Lily Aubrey Zoey Hannah Layla Brooklyn Alexa Natalie Leah Zoe Audrey
Savannah Claire Anna Samantha Allison Gabriella Sarah Lucy Caroline Madison Alice Hazel Violet Stella Nora Paisley
Bella Ellie Naomi Sadie Aaliyah Ariana Eva Ruby Maya Alyssa Kaylee Faith Lauren Rachel Megan Jessica Ashley Amanda
Nicole Stephanie Heather Rebecca Laura Michelle Kimberly Melissa Amy Angela Helen Deborah Karen Nancy Betty Margaret
Dorothy Lisa Linda Susan Donna Carol Ruth Sharon Diane Joyce Martha Gloria Teresa Doris Judith Rose Janet Irene Clara""".split()
MALE = """John Daniel James Robert Michael William David Richard Joseph Thomas Charles Christopher Matthew Anthony Mark
Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Edward Ronald Timothy Jason Jeffrey Ryan Jacob Gary Nicholas
Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Frank Alexander Raymond Patrick Jack Dennis
Jerry Tyler Aaron Henry Adam Douglas Nathan Peter Zachary Kyle Walter Harold Jeremy Ethan Carl Keith Roger Gerald
Christian Terry Sean Arthur Austin Noah Lawrence Jesse Joe Bryan Billy Jordan Albert Dylan Bruce Willie Gabriel Alan
Juan Logan Wayne Ralph Roy Eugene Randy Vincent Russell Louis Philip Bobby Johnny Bradley Oliver Lucas Liam Mason Owen""".split()


def p_pool(split):
    """P first names of a split (hash split of the name): (name, pronoun)."""
    out = [{"id": "f:" + n, "name": n, "pron": "she"} for n in FEMALE if split_of(n, "b48p") == split]
    out += [{"id": "m:" + n, "name": n, "pron": "he"} for n in MALE if split_of(n, "b48p") == split]
    return out


def _story(rng, kind, ents):
    """One story over the K entities `ents` (roster order = slot order). Returns the struct."""
    text, men = "", []

    def put(s, ent=None, mtype=None, kind_=None):
        nonlocal text
        cs = len(text)
        text += s
        if ent is not None:
            men.append([kind_, ent, cs, len(text), mtype])
            return len(men) - 1
        return None

    put("Entities: " if kind == "W" else "Characters: ")
    for j in range(K):
        put(ents[j]["name"], j, "roster", "E")
        put("; " if j < K - 1 else ".\n")
    # level 0: every entity once as subject, object = another entity (uniform); level 1: N1 of the K entities once as
    # subject (always including x1, the object of the queried entity), object = another entity. With N1 < K the level-1
    # aliases are not a permutation of the list, so they cannot be matched to it by elimination alone.
    qe = rng.randrange(K)
    f0 = {a: rng.choice([b for b in range(K) if b != a]) for a in range(K)}
    x1 = f0[qe]
    s1 = [x1] + rng.sample([b for b in range(K) if b != x1], N1 - 1)
    f1 = {a: rng.choice([b for b in range(K) if b != a]) for a in s1}
    o0 = list(range(K)); rng.shuffle(o0)
    o1 = list(s1); rng.shuffle(o1)
    lv = [[], []]
    for a in o0:
        r = put(ents[a]["name"], a, "can", "R")
        put(" is linked to " if kind == "W" else " called ")
        f = put(ents[f0[a]]["name"], f0[a], "can", "F")
        put(".\n")
        lv[0].append((r, f))
    alias_used = {}
    for a in o1:
        if kind == "W":
            al = rng.choice(ents[a]["aliases"])
            alias_used[a] = al
            r = put(al, a, "alias", "R")
            put(" is paired with ")
        else:
            put(ents[a]["name"], a, "ante", "A")
            put(" went to the %s. Then " % rng.choice(PLACES))
            r = put(ents[a]["pron"], a, "pron", "R")
            put(" visited ")
        f = put(ents[f1[a]]["name"], f1[a], "can", "F")
        put(".\n")
        lv[1].append((r, f))
    x2 = f1[x1]
    label = 1 if rng.random() < 0.5 else 0
    ce = x2 if label == 1 else rng.choice([e for e in range(K) if e != x2])
    if kind == "W":
        put("Question: is the entity that ")
        q = put(ents[qe]["name"], qe, "q", "R")
        put(" is linked to paired with ")
        c = put(ents[ce]["name"], ce, "cand", "F")
        put("?")
    else:
        put("Question: did the person whom ")
        q = put(ents[qe]["name"], qe, "q", "R")
        put(" called visit ")
        c = put(ents[ce]["name"], ce, "cand", "F")
        put("?")
    st = {"text": text, "label": label, "mentions": [tuple(m) for m in men], "lv": lv, "q": q, "cand": c,
          "chain": [qe, x1, x2], "ents": [{k: e[k] for k in ("id", "name")} for e in ents], "kind": kind}
    if kind == "P":
        st["prons"] = [e["pron"] for e in ents]
    return st


def build_B(kind, split):
    """Stories of dataset kind (W | P) and split; deterministic (generator seed per split and kind)."""
    n, seed = SPLITS_B[split]
    rng = random.Random(seed + (0 if kind == "W" else 100))
    pool = w_pool(split) if kind == "W" else p_pool(split)
    out = []
    for i in range(n):
        if kind == "W":
            ents = rng.sample(pool, K)
        else:
            fem = [e for e in pool if e["pron"] == "she"]; mal = [e for e in pool if e["pron"] == "he"]
            ents = rng.sample(fem, K // 2) + rng.sample(mal, K // 2)
            rng.shuffle(ents)
        out.append(_story(rng, kind, ents))
    return out


def dict_answer_B(s):
    """Gold answer by a dictionary walk over the gold identities."""
    ent = [m[1] for m in s["mentions"]]
    k = ent[s["q"]]
    for L in range(2):
        k = {ent[r]: ent[f] for r, f in s["lv"][L]}[k]
    return int(k == ent[s["cand"]]), k


def check_B(kind, split):
    st = build_B(kind, split)
    for s in st:
        for kd, e, cs, ce, mt in s["mentions"]:
            sub = s["text"][cs:ce]
            if mt == "alias":
                assert norm(sub) != norm(s["ents"][e]["name"])
            elif mt == "pron":
                assert sub in ("he", "she")
            else:
                assert sub == s["ents"][e]["name"], (sub, s["ents"][e])
        assert dict_answer_B(s)[0] == s["label"]
    return st


if __name__ == "__main__":
    for kind in "WP":
        pools = {sp: len(w_pool(sp) if kind == "W" else p_pool(sp)) for sp in ("train", "dev", "test")}
        print(kind, "pool sizes", pools)
        for sp in ("train", "dev", "test"):
            st = check_B(kind, sp)
            print(kind, sp, len(st), "stories; mean chars %.0f; positives %d" % (
                sum(len(s["text"]) for s in st) / len(st), sum(s["label"] for s in st)))
        print(st[0]["text"])
    c1 = build_C1("train")
    print(c1[0]["text"][:300])
