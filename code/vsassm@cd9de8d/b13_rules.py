"""B13 decoding rules (prereg/B13_prereg.md + amendment 1), shared by the decoder (b13.py) and the training-data checks
(b13_distill.py) so that every training sentence is checked with exactly the rules used at inference. No torch here.
"""
ROLES = ("x", "y", "sa", "sb", "q", "nx", "ny")
OBJ_PRO, PLACE_PRO = ("it", "them"), ("there",)
PLACE_ROLES = ("y", "sa", "sb", "ny")


def resolve(word, role, last):
    """The referent of one mention: pronouns by the most recently mentioned object / place; names stand for themselves."""
    lw = word.lower()
    if lw in PLACE_PRO or (lw in OBJ_PRO and role in PLACE_ROLES):
        return last["place"]
    if lw in OBJ_PRO:
        return last["obj"]
    return word


def note(ref, role, last):
    if ref is not None:
        last["place" if role in PLACE_ROLES else "obj"] = ref


def pair_events(res):
    """res: per mention (position, referent, role, prob), text order -> events [(position, op, a, b, confidence)].
    Swap mentions pair up in order; each x goes to the only y, or to the y with the same index when the counts match,
    or else to the nearest y after it (before it if none follows)."""
    evs = []
    sw = [r for r in res if r[2] in ("sa", "sb") and r[1]]
    for a, b in zip(sw[0::2], sw[1::2]):
        evs.append((a[0], "swap", a[1], b[1], a[3] * b[3]))
    xs = [r for r in res if r[2] == "x" and r[1]]
    ys = [r for r in res if r[2] == "y" and r[1]]
    if ys:
        for i, x in enumerate(xs):
            if len(ys) == 1:
                y = ys[0]
            elif len(xs) == len(ys):
                y = ys[i]
            else:
                y = next((y for y in ys if y[0] > x[0]), None) or max((y for y in ys if y[0] < x[0]), key=lambda r: r[0])
            evs.append((x[0], "write", x[1], y[1], x[3] * y[3]))
    return sorted(evs, key=lambda e: e[0])
