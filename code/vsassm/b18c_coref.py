"""B18c: bAbI's pronouns refer to the subject(s) of the previous sentence (qa11: he/she; qa13: they). Resolve them
sequentially from the story text, so a pronoun sentence after another pronoun sentence inherits its referents."""
import re

NAMES = {"john", "mary", "sandra", "daniel"}
LEAD = {"then", "afterwards", "following", "that", "after"}


def subjects(sentence, prev):
    words = re.findall(r"[a-z]+", sentence.lower())
    out = []
    for w in words:
        if w in NAMES:
            out.append(w)
        elif w == "and" or (not out and w in LEAD):
            continue
        elif w in {"he", "she", "they"} and not out:
            return list(prev)
        else:
            break
    return out


def resolve(sentences):
    """[sentence] -> [subjects of each sentence], pronouns resolved to the previous sentence's subjects."""
    res, prev = [], []
    for s in sentences:
        cur = subjects(s, prev)
        res.append(cur)
        prev = cur
    return res
