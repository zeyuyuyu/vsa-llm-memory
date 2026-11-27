"""B41 entity names, typo model and mention banks (prereg/B41_superposed_uncertain_multihop_prereg.md).

Everything here is generated from fixed lists in this file and numpy seeds: nothing depends on machine files.
E = 4096 entities in 1,024 clusters of four confusable names (a base name and three variants that differ from it by
one letter of the surname). A mention is the entity's name passed through a character-level typo model with rate rho
(per character: substitution, deletion, insertion or transposition, each with probability rho / 4).

Banks (each a list of (entity, rho, variant index, text)):
  train  32 variants per entity, rho in TRAIN_RHOS (four each)          -> naming-head training
  dev    3 variants per entity per rho in DEV_RHOS                       -> layer, temperature, rho levels, arm choices
  test   8 variants per entity per registered test rho (made after the dev record is committed)
"""
import hashlib
import numpy as np

E_CLUSTERS = 1024
PER_CLUSTER = 4
E = E_CLUSTERS * PER_CLUSTER
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr", "tr", "st", "sh",
          "ch", "th", "gl", "pl"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ei", "ou", "ia"]
CODAS = ["", "", "", "n", "r", "l", "s", "m", "k"]
V1 = "aeiou"
C1 = "bdfgklmnprstvz"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
TRAIN_RHOS = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3]
DEV_RHOS = [0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3]
CARRIER = "Note: {} attended the meeting."
RELATIONS = ["mentor", "employer", "neighbour", "landlord", "auditor", "tutor", "rival", "sponsor", "doctor",
             "lawyer", "banker", "coach", "editor", "agent", "partner", "deputy", "supplier", "client", "heir",
             "guardian", "tenant", "pupil", "patron", "courier", "broker", "witness", "trustee", "mediator",
             "inspector", "translator", "registrar", "envoy"]
M_REL = len(RELATIONS)


def _syl(rng):
    return ONSETS[rng.integers(len(ONSETS))] + VOWELS[rng.integers(len(VOWELS))] + CODAS[rng.integers(len(CODAS))]


def _word(rng, lo, hi):
    n = int(rng.integers(lo, hi + 1))
    return "".join(_syl(rng) for _ in range(n)).capitalize()


def _variant(last, rng, taken):
    """One-letter substitution in the surname (vowel->vowel, consonant->consonant), never the first letter."""
    for _ in range(200):
        i = int(rng.integers(1, len(last)))
        ch = last[i]
        pool = V1 if ch in V1 else (C1 if ch in C1 else None)
        if pool is None:
            continue
        new = pool[rng.integers(len(pool))]
        if new == ch:
            continue
        cand = last[:i] + new + last[i + 1:]
        if cand not in taken:
            return cand
    raise RuntimeError("no variant")


def entity_names(seed=41):
    """4,096 unique 'First Last' names; entity e belongs to cluster e // 4."""
    rng = np.random.default_rng([seed, 7])
    names, seen = [], set()
    while len(names) < E:
        first, last = _word(rng, 2, 2), _word(rng, 2, 3)
        base = first + " " + last
        if base in seen:
            continue
        lasts = {last}
        group = [base]
        ok = True
        for _ in range(PER_CLUSTER - 1):
            try:
                v = _variant(last, rng, lasts)
            except RuntimeError:
                ok = False
                break
            lasts.add(v)
            group.append(first + " " + v)
        if not ok or any(g in seen for g in group):
            continue
        for g in group:
            seen.add(g)
        names.extend(group)
    return names


def cluster_of(e):
    return e // PER_CLUSTER


def corrupt(name, rho, rng):
    if rho <= 0:
        return name
    out = list(name)
    i = 0
    res = []
    while i < len(out):
        ch = out[i]
        if rng.random() < rho:
            op = int(rng.integers(4))
            if op == 0:                                      # substitution
                res.append(LETTERS[rng.integers(26)])
            elif op == 1:                                    # deletion
                pass
            elif op == 2:                                    # insertion after
                res.append(ch)
                res.append(LETTERS[rng.integers(26)])
            else:                                            # transposition with the next character
                if i + 1 < len(out):
                    res.append(out[i + 1])
                    res.append(ch)
                    i += 1
                else:
                    res.append(ch)
        else:
            res.append(ch)
        i += 1
    s = "".join(res).strip()
    return s if s else name[0]


def bank(split, rhos=None, names=None):
    """Deterministic bank rows (entity, rho, j, text)."""
    names = names or entity_names()
    rows = []
    sid = {"train": 0, "dev": 1, "test": 2}[split]
    if split == "train":
        spec = [(r, j) for r in TRAIN_RHOS for j in range(4)]
    elif split == "dev":
        spec = [(r, j) for r in DEV_RHOS for j in range(3)]
    else:
        assert rhos, "test rhos are registered after the dev record"
        spec = [(r, j) for r in rhos for j in range(8)]
    for e in range(E):
        for (r, j) in spec:
            rng = np.random.default_rng([41, sid, e, int(round(r * 1000)), j])
            rows.append((e, r, j, corrupt(names[e], r, rng)))
    return rows


def digest(rows):
    h = hashlib.md5()
    for r in rows:
        h.update(("%d|%.3f|%d|%s\n" % r).encode())
    return h.hexdigest()


if __name__ == "__main__":
    nm = entity_names()
    print(len(nm), len(set(nm)), nm[:8])
    d = bank("dev", names=nm)
    print(len(d), digest(d), d[:5])
    for r in (0.03, 0.1, 0.2, 0.3):
        rng = np.random.default_rng(0)
        print(r, [corrupt(nm[0], r, rng) for _ in range(4)])
