"""B9 data (prereg/B9_prereg.md): open-vocabulary episodes with disjoint train/test name pools.

Each sentence records its entity mentions as character spans with a role: x (object / moved item), y (place / target),
sa, sb (first / second swap place), q (queried entity). Names are strings; nothing downstream uses a fixed class list.
"""
import random
import re

from b7_data import AB, C
from b8_data import D, E, F, gen_t3p, root, states_t2, states_t3
from b6_tasks import gen_t2

POOLS = {
    "obj_train": ["keys", "phone", "wallet", "glasses", "red cup", "mug", "remote", "charger", "passport", "watch",
                  "scissors", "tape", "spoon", "notebook", "headphones", "batteries", "stapler", "ruler", "pencil",
                  "eraser", "calculator", "flashlight", "umbrella", "scarf", "hat", "gloves", "sunglasses", "bracelet",
                  "ring", "necklace", "comb", "toothbrush", "razor", "lipstick", "perfume", "candle", "matches",
                  "lighter", "screwdriver", "hammer", "wrench", "pliers", "tape measure", "glue", "envelope", "stamp",
                  "ticket", "coin", "key card", "badge", "laptop", "tablet", "camera", "tripod", "cable", "adapter",
                  "mouse", "keyboard", "speaker", "microphone", "book", "magazine", "map", "compass", "whistle",
                  "thermos", "water bottle", "lunchbox", "apple", "banana", "orange", "sandwich", "cookie", "tea bag",
                  "sugar", "salt", "pepper", "honey jar", "jam jar", "cutting board"],
    "obj_test": ["harmonica", "kazoo", "yo-yo", "puzzle cube", "snow globe", "music box", "pocket knife", "fountain pen",
                 "monocle", "hourglass", "pinecone", "seashell", "marble", "dice", "domino", "chess knight", "teapot",
                 "ladle", "whisk", "corkscrew", "sponge", "thimble", "button", "spool", "yarn ball", "crayon",
                 "paintbrush", "postcard", "bookmark", "locket", "brooch", "cufflink", "tie clip", "hairpin", "bandana",
                 "mitten", "sock", "slipper", "earmuffs", "pocket watch"],
    "place_train": ["top drawer", "middle drawer", "bottom drawer", "desk drawer", "shelf 1", "shelf 2", "shelf 3",
                    "left cabinet", "right cabinet", "basket", "toolbox", "tray", "box a", "box b", "bin", "counter",
                    "pantry", "closet", "wardrobe", "nightstand", "bookcase", "cupboard", "fridge", "freezer", "oven",
                    "microwave", "sink", "bathtub", "laundry basket", "garage", "attic", "basement", "car trunk",
                    "glovebox", "backpack", "suitcase", "handbag", "briefcase", "gym bag", "mailbox", "safe",
                    "jewelry box", "shoebox", "crate", "barrel", "bucket", "flower pot", "vase", "fruit bowl",
                    "cookie jar", "bread box", "sewing kit", "first aid kit", "tackle box", "filing cabinet", "locker",
                    "cubby", "hamper", "chest", "trunk"],
    "place_test": ["pigeonhole", "dumbwaiter", "larder", "armoire", "credenza", "hutch", "ottoman", "footlocker",
                   "satchel", "knapsack", "duffel bag", "tote bag", "coffer", "strongbox", "caddy", "canister", "tureen",
                   "humidor", "terrarium", "birdhouse", "doghouse", "greenhouse", "shed", "loft", "cellar", "alcove",
                   "niche", "cubbyhole", "workbench drawer", "potting bench"],
    "colour_train": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"],
    "kind_train": ["box", "bag", "crate", "basket", "case", "tin"],
    "colour_test": ["teal", "maroon", "beige", "ivory", "olive", "navy", "coral", "lilac"],
    "kind_test": ["hamper", "pouch", "chest", "jar", "tube", "sack"],
    # B9b: larger T3' training lexicon (many multi-token words) and a fresh test pool used for nothing else
    "colour_train2": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "crimson", "scarlet",
                      "burgundy", "magenta", "fuchsia", "violet", "indigo", "azure", "cyan", "turquoise", "emerald",
                      "jade", "chartreuse", "amber", "saffron", "mustard", "ochre", "bronze", "copper", "silver",
                      "golden", "charcoal", "slate", "ash", "cream", "khaki", "tan", "chestnut", "mahogany", "rust",
                      "pink", "peach"],
    "kind_train2": ["box", "bag", "crate", "basket", "case", "tin", "bin", "carton", "trunk", "pail", "bucket",
                    "barrel", "drum", "vat", "urn", "bottle", "flask", "canteen", "cooler", "locker", "caddy", "holder",
                    "pot", "kettle", "bowl", "cup", "mug", "envelope", "folder", "wallet"],
    "colour_test2": ["lavender", "periwinkle", "vermilion", "sepia", "umber", "cerulean", "mauve", "taupe"],
    "kind_test2": ["cask", "crock", "tote", "pod", "keg", "vial"],
    "obj_test2": ["accordion", "tambourine", "ocarina", "metronome", "abacus", "sextant", "telescope", "kaleidoscope",
                  "slingshot", "boomerang", "frisbee", "pinwheel", "origami crane", "sundial", "barometer", "quill",
                  "inkwell", "wax seal", "magnifying glass", "stopwatch", "castanets", "maracas", "figurine",
                  "nutcracker", "harpsichord key", "pocket mirror", "lorgnette", "snuffbox", "lantern", "spyglass"],
    "place_test2": ["gazebo", "pergola", "conservatory", "scullery", "vestibule", "foyer", "mezzanine", "belfry",
                    "boathouse", "carport", "porch swing", "window seat", "plant stand", "coat rack", "umbrella stand",
                    "magazine rack", "wine rack", "spice rack", "hat stand", "display cabinet"],
}
# B6's own sentence templates, with names as parameters
B6W = {
    "T2_PRE": ["There are 16 places: {places}. Every place starts empty. The following actions happen in order. "
               '"Put X in P" moves X from wherever it is into P. "Swap the contents of P and Q" moves everything that is '
               "in P into Q and everything that is in Q into P, at the same time."],
    "T2_PUT": ["Put the {o} in the {p}."], "T2_SWAP": ["Swap the contents of the {a} and the {b}."],
    "T2_Q": ["Which place holds the {o} now?"],
    "T3_PRE": ["There are 40 containers, named by colour and kind (for example the red box or the blue bag). They start "
               'out separate. The following moves happen in order. "Put A into B" moves A, together with everything '
               "inside it, into B; A is no longer in whatever held it before."],
    "T3_MOVE": ["Put the {a} into the {b}."],
    "T3_Q": ["After all the moves, the {q} is inside a container, which may itself be inside others. What is the "
             "outermost container that holds the {q}, the one that is not inside anything?"],
}
FAMILIES = {"AB": AB, "D": D, "E": E, "C": C, "F": F, "B6": B6W}
TRAIN_FAMILIES = ("AB", "D", "E")
ROLE_OF = {"o": "x", "p": "y", "a": "sa", "b": "sb", "q": "q"}


def fill(template, names, offset):
    """Format a template; return (text, mentions) with mentions = [(start, end_inclusive, role, name)]."""
    out, mentions, pos = "", [], 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out += template[pos:m.start()]
        field = m.group(1)
        name = names[field]
        mentions.append((offset + len(out), offset + len(out) + len(name) - 1, ROLE_OF.get(field, field), name))
        out += name
        pos = m.end()
    return out + template[pos:], mentions


def _names(task, split, rng):
    """split: train (B9 pools), train2 (B9b: T3' larger lexicon, T2 = train), test (B9), test2 (B9b fresh pool)."""
    if task == "T2":
        s = {"train": "train", "train2": "train", "test": "test", "test2": "test2"}[split]
        return rng.sample(POOLS[f"obj_{s}"], 16), rng.sample(POOLS[f"place_{s}"], 16)
    items = [f"{c} {k}" for c in POOLS[f"colour_{split}"] for k in POOLS[f"kind_{split}"]]
    return rng.sample(items, 40), None


def episode(task, rng, n, family, names_split):
    fam = FAMILIES[family]
    ents, places = _names(task, names_split, rng)
    if task == "T2":
        p = gen_t2(rng, n)
        acts = p["acts"]
        if family == "B6" or "{places}" in fam["T2_PRE"][0]:
            plist = ", ".join(f"the {x}" for x in places[:-1]) + f" and the {places[-1]}"
            pre = rng.choice(fam["T2_PRE"]).format(places=plist)
        else:
            pre = rng.choice(fam["T2_PRE"])
    else:
        p = gen_t3p(rng, n)
        acts = p["moves"]
        pre = rng.choice(fam["T3_PRE"])
    text, sents = pre + "\n\n", []
    for i, a in enumerate(acts):
        if task == "T2" and a[0] == "put":
            tpl, nm = rng.choice(fam["T2_PUT"]), {"o": ents[a[1]], "p": places[a[2]]}
        elif task == "T2":
            tpl, nm = rng.choice(fam["T2_SWAP"]), {"a": places[a[1]], "b": places[a[2]]}
        else:
            tpl, nm = rng.choice(fam["T3_MOVE"]), {"a": ents[a[0]], "b": ents[a[1]]}
        prefix = f"{i + 1}. "
        s, ments = fill(tpl, nm, len(text) + len(prefix))
        if task == "T3":  # in T3 the mover is x and the target is y
            ments = [(a0, a1, {"sa": "x", "sb": "y"}[r], nmv) for a0, a1, r, nmv in ments]
        sents.append({"span": (len(text) + len(prefix), len(text) + len(prefix) + len(s) - 1), "mentions": ments,
                      "op": "swap" if task == "T2" and a[0] == "swap" else "write"})
        text += prefix + s + "\n"
    text += "\n"
    qname = ents[p["query"]]
    s, ments = fill(rng.choice(fam[f"{task}_Q"]), {"o": qname, "q": qname}, len(text))
    ments = [(a0, a1, "q", nmv) for a0, a1, _, nmv in ments]
    sents.append({"span": (len(text), len(text) + len(s) - 1), "mentions": ments, "op": "question"})
    text += s
    st = states_t2(acts) if task == "T2" else states_t3(acts)
    ans_idx = st[-1][p["query"]] if task == "T2" else root(st[-1], p["query"])
    answer = places[ans_idx] if task == "T2" else ents[ans_idx]
    return {"task": task, "n": n, "family": family, "names": names_split, "text": text, "sents": sents,
            "acts": acts, "query": p["query"], "ents": ents, "places": places, "answer": answer, "states": st}


def train_set(task, n_episodes, seed, names="train"):
    rng = random.Random(seed)
    lo = 5 if task == "T2" else 6
    return [episode(task, rng, rng.randint(lo, 40), rng.choice(TRAIN_FAMILIES), names) for _ in range(n_episodes)]


def test_sets(task, seed, ns=(10, 20, 40, 80, 160), fresh=False):
    """{name: episodes}: test-pool names in B6 / C / F wording, and train-pool names in C (wording novelty only)."""
    rng = random.Random(seed)
    spec = {"B6-new": ("B6", "test", 40), "C-new": ("C", "test", 100), "F-new": ("F", "test", 100),
            "C-old": ("C", "train", 100)}
    if fresh:  # B9b: the fresh pool, drawn after B9's sets so those stay identical
        spec.update({"B6-new2": ("B6", "test2", 40), "C-new2": ("C", "test2", 100), "F-new2": ("F", "test2", 100)})
    out = {}
    for name, (fam, pool, per_n) in spec.items():
        out[name] = []
        for n in ns:
            for i in range(per_n):
                e = episode(task, rng, n, fam, pool)
                e["idx"] = i
                out[name].append(e)
    return out
