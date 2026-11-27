"""B48 part B, arm "LM-written canonical names" (prereg/B48_one_form_and_alias_naming_prereg.md).

Qwen3-8B (the same snapshot whose frozen states the naming head reads; chat template, thinking off, greedy) reads the
story with its 32 binding statements numbered [1]..[28] (the question is left out) and writes, for every statement, its
two participants by the name listed at the top. The written strings are later hashed (b48_alias.py): a mention gets
the code of norm(written string). The prompt is fixed here, before any output is read.

  python b48_lmcanon.py --kind W --split dev --out out/lm/W_dev.json [--start i --stop j] [--bs 16]
"""
import argparse, json, os, re, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b48_struct as S48                                                    # noqa: E402

QWEN8 = os.environ.get("B31_QWEN8", "/path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
MAX_NEW = 1600


def numbered_text(s):
    """Story text without the question, every binding statement prefixed by [n] (level 0: 1-16, level 1: 17-28, in
    text order). P: the number goes before "Then", i.e. it marks the sentence whose subject is the pronoun."""
    text = s["text"][:s["text"].index("Question:")].rstrip("\n")
    ins = []
    for L in range(2):
        for j, (r, f) in enumerate(s["lv"][L]):
            cs = s["mentions"][r][2]
            if s["mentions"][r][4] == "pron":
                cs -= len("Then ")
                assert s["text"][cs:cs + 5] == "Then "
            ins.append((cs, "[%d] " % (16 * L + j + 1)))
    for cs, t in sorted(ins, reverse=True):
        text = text[:cs] + t + text[cs:]
    return text


def prompt(s):
    what = "entities" if s["kind"] == "W" else "characters"
    return ("Below is a list of %s followed by numbered statements. Some statements refer to an entry of the list by "
            "another name or by a pronoun.\n\n%s\n\nFor every numbered statement [1] to [28], write the two participants "
            "of that statement, each by its name exactly as written in the list at the top (if the statement uses "
            "another name or a pronoun, write the listed name it refers to). Answer with 28 lines and nothing else, in "
            "the form:\n1. <first participant> | <second participant>" % (what, numbered_text(s)))


LINE = re.compile(r"^\s*\[?(\d+)\]?\s*[.:)]?\s*(.*?)\s*\|\s*(.*?)\s*$")


def parse(out):
    """{n: (first, second)}; the first line of a number counts; lines without '|' are ignored."""
    res = {}
    for ln in out.splitlines():
        m = LINE.match(ln)
        if m and 1 <= int(m.group(1)) <= 28 and int(m.group(1)) not in res:
            res[int(m.group(1))] = (m.group(2).strip().strip("<>").strip(), m.group(3).strip().strip("<>").strip())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["W", "P"])
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    st = S48.build_B(a.kind, a.split)
    idx = list(range(len(st)))[a.start:a.stop]
    tok = AutoTokenizer.from_pretrained(QWEN8)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN8, torch_dtype=torch.bfloat16).to(a.device).eval()
    texts = {i: tok.apply_chat_template([{"role": "user", "content": prompt(st[i])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) for i in idx}
    part = a.out + ".part"
    done = json.load(open(part)) if os.path.exists(part) else {}
    order = sorted([i for i in idx if str(i) not in done], key=lambda i: len(texts[i]))
    t0 = time.time()
    for k in range(0, len(order), a.bs):
        b = order[k:k + a.bs]
        enc = tok([texts[i] for i in b], return_tensors="pt", padding=True).to(a.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                               top_k=None, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(b):
            o = tok.decode(g[j, enc["input_ids"].size(1):], skip_special_tokens=True)
            done[str(i)] = {"out": o, "parsed": parse(o), "n_new": int((g[j, enc["input_ids"].size(1):] != tok.pad_token_id).sum())}
        json.dump(done, open(part, "w"))
        print("  %d/%d stories, %.0fs" % (len(done), len(idx), time.time() - t0), flush=True)
    res = {"args": vars(a), "model": QWEN8, "max_new_tokens": MAX_NEW, "host": os.uname()[1],
           "gpu": torch.cuda.get_device_name(a.device), "secs": round(time.time() - t0, 1),
           "stories": {i: done[str(i)] for i in idx}}
    json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    if os.path.exists(part):
        os.remove(part)


if __name__ == "__main__":
    main()
