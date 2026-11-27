"""V-85 shared data layer (prereg 85461e8): 2WikiMultihopQA compositional, the same slice the paper uses.

Kept separate from the model code so every arm -- frozen prompt, LoRA-only, LoRA+VSA -- reads exactly the
same instances through exactly the same normalisation, and so a fault here is caught once rather than three
times.
"""
import ast, os, re, string

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

FEWSHOT = [
    ("Context:\nBlade Runner: Blade Runner is a 1982 film directed by Ridley Scott.\n"
     "Ridley Scott: Ridley Scott was born in South Shields, England.",
     "Where was the director of the film Blade Runner born?", "South Shields, England"),
    ("Context:\nThe Matrix: The Matrix is a 1999 film directed by Lana Wachowski.\n"
     "Lana Wachowski: Lana Wachowski is an American film director.",
     "What is the nationality of the director of the film The Matrix?", "American"),
]


def norm(s):
    """The paper's normalisation, unchanged."""
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def em(pred, gold):
    return int(norm(pred) == norm(gold))


def _parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return x


def load_rows(n, split="validation", cache_dir=None, skip=0):
    """Compositional two-hop instances with usable context. Mirrors v65.py's filter so the slice matches
    the paper's. `skip` passes over the first `skip` FILTER-MATCHING rows before collecting, so skip=300
    yields matching rows 301..300+n of the same deterministic stream (V-122's registered fresh-row draw);
    the default 0 leaves every earlier caller byte-identical."""
    from datasets import load_dataset
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split, streaming=True,
                      cache_dir=cache_dir)
    rows, matched = [], 0
    for r in ds:
        if r.get("type") != "compositional":
            continue
        ev, sf, cx = _parse(r.get("evidences")), _parse(r.get("supporting_facts")), _parse(r.get("context"))
        if not ev or len(ev) < 2 or not sf or not cx:
            continue
        if "content" not in cx:
            for k in ("sentences", "sent", "text"):
                if k in cx:
                    cx["content"] = cx[k]
                    break
        if "content" not in cx:
            continue
        matched += 1
        if matched <= skip:
            continue
        rows.append({"question": r["question"], "answer": r["answer"], "evidences": ev,
                     "supporting_facts": sf, "context": cx})
        if len(rows) >= n:
            break
    return rows


def build_context(row, tok, budget):
    """Supporting paragraphs first, then the rest, under a token budget. NO gold triples are verbalised --
    every V-85 arm must work from the passages alone, which is the point of the experiment."""
    sup = set(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
    titles, sents = row["context"]["title"], row["context"]["content"]
    first = [f"{t}: {ss[i]}" for (t, i) in sup
             for tt, ss in zip(titles, sents) if tt == t and i < len(ss)]
    rest = [f"{t}: {' '.join(ss[:2])}" for t, ss in zip(titles, sents)
            if t not in {x[0] for x in sup}]
    out, used, truncated = [], 0, False
    for block in first + rest:
        k = len(tok(block)["input_ids"])
        if used + k > budget:
            truncated = True
            if block in first:
                continue
            break
        out.append(block)
        used += k
    return "\n".join(out), truncated


def prompt_for(row, tok, budget=1200, with_fewshot=True):
    ctx, tr = build_context(row, tok, budget)
    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % f for f in FEWSHOT) if with_fewshot else ""
    body = "Context:\n%s\nQuestion: %s\nAnswer:" % (ctx, row["question"])
    return ((shots + "\n\n" + body) if with_fewshot else body), tr
