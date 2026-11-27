"""Shared helpers for the held-out problem families.

Nothing here knows about any particular family: seeded RNGs, lenient reading of
proposed answers, name pools and small text utilities.
"""
import random
import re

# Single-token given names that are not also common English words.
NAMES = [
    "Ana", "Ben", "Chloe", "Dev", "Elif", "Farid", "Greta", "Hiro", "Ines",
    "Jonas", "Kemi", "Leo", "Maya", "Nikhil", "Olga", "Pablo", "Rosa", "Sami",
    "Tomas", "Uma", "Viktor", "Wen", "Yara", "Zane", "Amir", "Bea", "Carlos",
    "Dana", "Emil", "Fiona", "Gus", "Hana", "Ivan", "Jade", "Kofi", "Lena",
    "Marco", "Nora", "Omar", "Priya", "Rafael", "Sofia", "Theo", "Vera",
]


def rng_for(family, level, seed):
    """Deterministic RNG for one (family, level, seed) item."""
    return random.Random(f"heldout|{family}|{int(level)}|{int(seed)}")


def balanced_bit(family, level, seed):
    """A yes/no target that is balanced over any run of consecutive seeds: seeds
    2k and 2k+1 always get opposite values, and which one gets True is decided
    by a hash, so the seed's parity alone says nothing."""
    import hashlib
    h = hashlib.sha256(f"balance|{family}|{int(level)}|{int(seed) // 2}".encode()).digest()
    return bool((h[0] & 1) ^ (int(seed) % 2))


def item_id(family, level, seed):
    return f"{family}-L{int(level)}-s{int(seed)}"


# Extra names, only drawn on when more people are needed than NAMES holds (so the
# items of the smaller levels are unchanged).
NAMES_EXTRA = [
    "Abel", "Adele", "Aiko", "Akira", "Alma", "Anton", "Arjun", "Astrid", "Axel",
    "Bianca", "Boris", "Bruno", "Celine", "Cyrus", "Dalia", "Dario", "Delia",
    "Diego", "Dmitri", "Edgar", "Elias", "Enzo", "Esme", "Ezra", "Felix", "Freya",
    "Gemma", "Gideon", "Hamza", "Helga", "Hugo", "Idris", "Ilse", "Imran",
    "Ingrid", "Isak", "Jasper", "Jorge", "Karim", "Keiko", "Lars", "Leila",
    "Lucia", "Luka", "Magnus", "Malik", "Mateo", "Nadia", "Nils", "Oskar",
    "Paolo", "Petra", "Rhea", "Rocco", "Selma", "Silas", "Talia", "Tariq",
    "Ulla", "Yusuf", "Zara", "Zeno",
]


def pick_names(rng, n):
    names = list(NAMES) if n <= len(NAMES) else list(NAMES) + list(NAMES_EXTRA)
    assert n <= len(names) and len(set(names)) == len(names)
    rng.shuffle(names)
    return names[:n]


def is_impossible(answer):
    """True if the answer claims the problem has no solution."""
    if not isinstance(answer, dict):
        return False
    for k, v in answer.items():
        if str(k).strip().lower() == "impossible":
            if v is True or v == 1:
                return True
            if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"):
                return True
    return False


def as_int(v):
    """Read an integer slot value leniently (int, integral float, numeric string)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?\d+\.0*", s):
            return int(float(s))
    raise ValueError(f"not an integer: {v!r}")


def norm_key(k):
    return re.sub(r"\s*:\s*", ":", str(k).strip()).lower()


def read_slots(answer, keys):
    """Return {key: int} for the expected slot keys, or None if any is missing
    or not an integer.  Key matching ignores case and spaces around ':'.
    Extra keys in the answer are ignored."""
    if not isinstance(answer, dict) or is_impossible(answer):
        return None
    normd = {norm_key(k): v for k, v in answer.items()}
    out = {}
    for k in keys:
        nk = norm_key(k)
        if nk not in normd:
            return None
        try:
            out[k] = as_int(normd[nk])
        except (ValueError, TypeError):
            return None
    return out


def join_and(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten"]


def num_word(k):
    return NUM_WORDS[k] if 0 <= k < len(NUM_WORDS) else str(k)
