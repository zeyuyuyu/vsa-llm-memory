"""V-97 store: string-hash identity + exact HRR two-hop composition (prereg 851404a).

No training here and no parameters. Identity comes from hashing the normalised surface form into a fixed
random codebook, which gives cross-talk 0 by construction -- V-95 measured the tolerance at 0.0989 and V-94
measured that no continuous map of transformer states gets below 0.3904, so hashing is not a shortcut, it is
the only route to the required geometry.

The two-hop answer is computed by unbind-and-cleanup over the HRR superposition. A symbolic walk over the
same parsed triples is computed alongside purely as an agreement check: with exact codes the algebra and the
walk must coincide, and reporting their agreement is how we detect a broken store rather than assuming one.
"""
import hashlib
import re
import string as _string

import numpy as np
import torch
import torch.nn.functional as F

D_VSA = 2048
N_ROWS = 200003          # prime, so the hash spreads


def _codebook(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    # generated in blocks to keep peak memory modest
    rows = []
    for i in range(0, N_ROWS, 20000):
        n = min(20000, N_ROWS - i)
        rows.append(F.normalize(torch.randn(n, D_VSA, generator=g), dim=-1))
    return torch.cat(rows).to(device)


_CB = {}


def codebook(device, seed=0):
    key = (str(device), seed)
    if key not in _CB:
        _CB[key] = _codebook(device, seed)
    return _CB[key]


def norm_str(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in _string.punctuation)
    return " ".join(s.split())


def row_index(s):
    h = hashlib.blake2b(norm_str(s).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % N_ROWS


def vec(s, device, seed=0):
    return codebook(device, seed)[row_index(s)]


def bind(a, b):
    return torch.fft.irfft(torch.fft.rfft(a.float()) * torch.fft.rfft(b.float()), n=D_VSA).to(a.dtype)


def involution(x):
    return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=[-1])], dim=-1)


def unbind(ab, a):
    """HRR's standard unbind: circular correlation, i.e. convolution with the involution.

    NOT exact frequency-domain deconvolution. Dividing by |A|^2 inverts a single binding perfectly but
    amplifies the cross-terms of a superposition without bound, which is why the store disagreed with the
    symbolic walk on 90% of instances in the first smoke."""
    return bind(involution(a), ab)


def parse_triples(text, max_triples=40):
    """Parses lines of the form `subject | relation | object`. Malformed lines are dropped and counted, so
    extraction failures are visible rather than silently reducing recall."""
    out, bad = [], 0
    for line in text.split("\n"):
        line = line.strip().strip("-").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            out.append(tuple(parts))
        else:
            bad += 1
        if len(out) >= max_triples:
            break
    return out, bad


def triple_recall(pred, gold):
    """Recall of gold (subject, object) pairs among the predicted triples, matched after normalisation.
    Relations are ignored because the datasets' relation strings are not canonical and matching them would
    understate recall for reasons unrelated to the question."""
    if not gold:
        return float("nan")
    P = {(norm_str(s), norm_str(o)) for s, _, o in pred}
    G = {(norm_str(t[0]), norm_str(t[2])) for t in gold if len(t) >= 3}
    if not G:
        return float("nan")
    return len(P & G) / len(G)


def answer_two_hop(triples, head, device, seed=0, topk=1):
    """Two hops through the HRR store, plus the symbolic walk for agreement.

    store  = sum over triples of bind(hash(subject), hash(object))
    hop    = cleanup(unbind(store, hash(current))) over the objects actually present
    Returns (hrr_answer, symbolic_answer, agreed).
    """
    if not triples:
        return "", "", True
    objs = [o for _, _, o in triples]
    obj_vecs = torch.stack([vec(o, device, seed) for o in objs])
    store = torch.stack([bind(vec(s, device, seed), vec(o, device, seed)) for s, _, o in triples]).sum(0)

    # An item that is genuinely in a superposition of n bindings comes back at cosine on the order of
    # 1/sqrt(n); an item that is absent comes back at noise. Without a floor the store returns an arbitrary
    # nearest object for a query it never stored, which is exactly the hallucination the algebra is supposed
    # to preclude. Half of 1/sqrt(n) sits between the two regimes.
    thresh = 0.5 / max(1.0, float(len(triples))) ** 0.5

    def hop(cur_str):
        hat = F.normalize(unbind(store, vec(cur_str, device, seed)), dim=-1)
        sims = F.cosine_similarity(obj_vecs, hat.unsqueeze(0))
        best = int(sims.argmax().item())
        return objs[best] if float(sims[best]) >= thresh else ""      # abstain rather than hallucinate

    h1 = hop(head)
    h2 = hop(h1) if h1 else ""

    # symbolic walk over the same triples
    by_subj = {}
    for s, _, o in triples:
        by_subj.setdefault(norm_str(s), []).append(o)
    s1 = by_subj.get(norm_str(head), [""])[0]
    s2 = by_subj.get(norm_str(s1), [""])[0] if s1 else ""

    # Agreement is only meaningful where the walk is unambiguous. When a subject carries several objects the
    # symbolic walk takes the first and the store's cleanup takes the nearest, so a mismatch there reflects
    # tie-breaking rather than the algebra, and counting it would hide real store faults behind noise.
    # Agreement is meaningful only where the walk is BOTH present and single-valued. Counting absent heads
    # was the flaw in the first two smokes: with recall at 0.35 the head is usually missing, the symbolic
    # walk returns empty, and any store answer disagrees -- which looked like a broken store.
    unamb = (len(by_subj.get(norm_str(head), [])) == 1 and len(by_subj.get(norm_str(s1), [])) == 1)
    agreed = None if not unamb else (norm_str(h2) == norm_str(s2))
    return h2, s2, agreed
