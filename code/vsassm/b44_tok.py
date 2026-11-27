"""B44 tokenisation cache (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md).

For each encoder and split: token ids of the arena text (no truncation, no padding) and the token span of every symbol
mention, found by character offsets exactly as b31_struct.tokenize / v13.token_span (every token that overlaps the
mention's characters). A char -> token map replaces the quadratic scan; the first 300 instances of every split are
checked against b31_struct.tokenize.

Also the scratchpad target of every instance (b44_lora --task scratch): the bridge symbol as written at level 0
(decimal), the bridge as written at level 1 (the marker "F" + two hex digits), the answer symbol (decimal), then yes/no.

  python b44_tok.py --enc qwen8 --items std:train std:dev std:test ho:train ho:dev ho:test
"""
import argparse, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b44_data as BD                                                       # noqa: E402
import b31_struct as S                                                      # noqa: E402

PROMPT_SUFFIX = "\nAnswer:"


def scratch_target(s):
    c = s["chain"]
    return " %d F%s %d %s" % (c[1], BD.h2(c[1]), c[2], "yes" if s["label"] == 1 else "no")


def fast_spans(text, offs, mentions):
    owner = [-1] * len(text)
    for i, (x, y) in enumerate(offs):
        if y <= x:
            continue
        for ch in range(x, y):
            if owner[ch] == -1:
                owner[ch] = i
    sp = []
    for key, kind, sym, cs, ce in mentions:
        toks = sorted(set(owner[ch] for ch in range(cs, ce)))
        assert toks and toks[0] >= 0 and toks == list(range(toks[0], toks[-1] + 1)), (text[cs:ce], toks)
        sp.append((toks[0], toks[-1] + 1))
    return sp


def build(tok, st, check=300):
    out = {"ids": [], "spans": [], "label": [], "sym": [], "tgt": [], "chain": [], "text_md5": BD.md5_texts(st)}
    texts = [s["text"] for s in st]
    for k in range(0, len(st), 2000):
        enc = tok(texts[k:k + 2000], return_offsets_mapping=True, add_special_tokens=False)
        for j, s in enumerate(st[k:k + 2000]):
            ids, offs = enc["input_ids"][j], enc["offset_mapping"][j]
            sp = fast_spans(s["text"], offs, s["mentions"])
            if k + j < check:
                ids2, _, sp2 = S.tokenize(tok, s)
                assert ids2 == ids and sp2 == sp, "span mismatch at %d" % (k + j)
            out["ids"].append(torch.tensor(ids, dtype=torch.int32))
            out["spans"].append(torch.tensor(sp, dtype=torch.int16))
            out["label"].append(s["label"]); out["sym"].append([m[2] for m in s["mentions"]])
            out["tgt"].append(scratch_target(s)); out["chain"].append(s["chain"])
    out["label"] = torch.tensor(out["label"]); out["sym"] = torch.tensor(out["sym"], dtype=torch.int16)
    return out


def path(root, enc, item):
    return os.path.join(root, "%s_%s.pt" % (enc, item.replace(":", "_")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="qwen8", choices=["qwen8", "phi2"])
    ap.add_argument("--items", nargs="+", required=True, help="variant:split, or k<k>:split for the Fig. 2c sweep")
    ap.add_argument("--root", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    import b31_prep as BP
    tok = BP.tokenizer(a.enc)
    os.makedirs(a.root, exist_ok=True)
    for item in a.items:
        p = path(a.root, a.enc, item)
        if os.path.exists(p):
            print("exists", p); continue
        t0 = time.time()
        v, sp = item.split(":")
        st = BD.ksplit(int(v[1:]), sp) if v.startswith("k") else BD.split(v, sp)
        obj = build(tok, st)
        obj.update(enc=a.enc, item=item, n=len(st))
        torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)
        L = [len(x) for x in obj["ids"]]
        print("tok", a.enc, item, len(st), "max_len", max(L), "mean_len %.1f" % (sum(L) / len(L)),
              "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
