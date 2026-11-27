"""B44 chain-of-thought prompting of Qwen3-8B on arena C (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md).

No training. The model is the same Qwen/Qwen3-8B snapshot the encoder arms read (b968826d), run with its chat template
(HF generate; vLLM is not usable on the box). Four arms:
  zs-nothink   zero-shot, enable_thinking=False, greedy decoding, max 2,048 new tokens
  zs-think     zero-shot, enable_thinking=True, the model card's sampling for thinking mode (temperature 0.6, top-p
               0.95, top-k 20), seed 0 per batch, max 8,192 new tokens
  fs-nothink   three worked examples (fixed training instances, see FEWSHOT) as earlier chat turns, then the instance
  fs-think     the same with thinking
The prompt lists every pair, one per line, grouped by table in the order of the text (level 0 = table R, level 1 =
table F), explains the notation (keys in hexadecimal, values in decimal, one shared set of 64 symbols) and the
two-step question in words, and asks for a final line "Answer: yes" or "Answer: no". A reply with no parsable final
answer (including a thinking trace cut at the token limit) is scored wrong; its rate is reported.

  python b44_cot.py --arm zs-think --variant std --split test --out out/cot/zs-think_std_test.json
"""
import argparse, json, os, re, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b44_data as BD                                                       # noqa: E402
import b31_prep as BP                                                       # noqa: E402

ARMS = {"zs-nothink": (0, False), "zs-think": (0, True), "fs-nothink": (3, False), "fs-think": (3, True)}
MAXNEW = {False: 2048, True: 8192}
ANS = re.compile(r"answer\s*[:：]?\s*\**\s*(yes|no)\b", re.I)


def describe(s):
    lv = s["levels"]
    lines = ["You are given two lookup tables, R and F, over the same 64 symbols (numbered 0 to 63). Each entry is "
             "written [Txx]{n}: in table T, the key symbol xx (two hexadecimal digits) maps to the value symbol n (a "
             "decimal number). The same symbol is written in hexadecimal when it is a key and in decimal when it is a "
             "value; for example the value {37} is the same symbol as the key 25, because hexadecimal 25 = decimal 37.",
             "", "Table R (%d entries):" % len(lv[0])]
    lines += ["[R%s]{%d}" % (BD.h2(a), b) for a, b in lv[0]]
    lines += ["", "Table F (%d entries):" % len(lv[1])]
    lines += ["[F%s]{%d}" % (BD.h2(a), b) for a, b in lv[1]]
    q, cand = s["chain"][0], s["mentions"][s["cand"]][2]
    lines += ["", "Question: start from the key %s in table R and read its value v. Then find v as a key in table F "
                  "(write v as two hexadecimal digits) and read its value w. Is w equal to {%d}?" % (BD.h2(q), cand),
              "Think step by step, then end your reply with a final line \"Answer: yes\" or \"Answer: no\"."]
    return "\n".join(lines)


def worked(s):
    c = s["chain"]; cand = s["mentions"][s["cand"]][2]
    eq = c[2] == cand
    return ("In table R, [R%s]{%d}, so v = %d. Decimal %d is hexadecimal %s. In table F, [F%s]{%d}, so w = %d. "
            "w = %d and the candidate is {%d}: %s.\nAnswer: %s") % (
        BD.h2(c[0]), c[1], c[1], c[1], BD.h2(c[1]), BD.h2(c[1]), c[2], c[2], c[2], cand,
        "equal" if eq else "not equal", "yes" if eq else "no")


def fewshot():
    """Fixed examples from the std training stream (seed 44042): the first positive, the first negative and the second
    positive, in that order. Training instances, never dev or test."""
    tr = BD.gen(50, BD.SEEDS["std"]["train"], "std", "train")
    pos = [s for s in tr if s["label"] == 1]; neg = [s for s in tr if s["label"] == 0]
    return [pos[0], neg[0], pos[1]]


def messages(s, shots):
    msgs = []
    for e in shots:
        msgs += [{"role": "user", "content": describe(e)}, {"role": "assistant", "content": worked(e)}]
    msgs.append({"role": "user", "content": describe(s)})
    return msgs


def parse_main(reply):
    """Final answer after the thinking block: the last 'Answer: yes/no'."""
    if "<think>" in reply and "</think>" not in reply:
        return None, "truncated_thinking"
    final = reply.split("</think>")[-1]
    m = ANS.findall(final)
    if not m:
        return None, "no_answer_line"
    return int(m[-1].lower() == "yes"), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--variant", default="std", choices=["std", "ho"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=None, help="first n instances (dev selection uses 100)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--maxnew", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BP.QWEN8)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BP.QWEN8, torch_dtype=torch.bfloat16).to(a.device).eval()
    nshot, think = ARMS[a.arm]
    shots = fewshot()[:nshot]
    st = BD.split(a.variant, a.split)
    if a.n:
        st = st[:a.n]
    prompts = [tok.apply_chat_template(messages(s, shots), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=think) for s in st]
    maxnew = a.maxnew or MAXNEW[think]
    part = a.out + ".part"
    done = json.load(open(part)) if os.path.exists(part) else {"rows": []}
    t0 = time.time()
    for k in range(len(done["rows"]), len(st), a.bs):
        P = prompts[k:k + a.bs]
        enc = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
        torch.manual_seed(k)
        kw = dict(max_new_tokens=maxnew, pad_token_id=tok.pad_token_id)
        if think:
            kw.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        L = enc["input_ids"].size(1)
        for j, o in enumerate(out):
            gen = o[L:]
            ntok = int((gen != tok.pad_token_id).sum())
            reply = tok.decode(gen, skip_special_tokens=False).replace(tok.pad_token, "")
            for sp in ("<|im_end|>", "<|endoftext|>"):
                reply = reply.replace(sp, "")
            pred, why = parse_main(reply)
            s = st[k + j]
            done["rows"].append({"i": k + j, "label": s["label"], "pred": pred, "parse": why, "new_tokens": ntok,
                                 "hit_limit": ntok >= maxnew, "reply": reply})
        json.dump(done, open(part, "w"))
        acc = sum(r["pred"] == r["label"] for r in done["rows"]) / len(done["rows"])
        print("  %s %s %s %d/%d acc %.4f (%.0fs)" % (a.arm, a.variant, a.split, len(done["rows"]), len(st), acc,
                                                    time.time() - t0), flush=True)
    rows = done["rows"]
    res = {"args": vars(a), "n": len(rows), "acc": sum(r["pred"] == r["label"] for r in rows) / len(rows),
           "unparsed": sum(r["pred"] is None for r in rows) / len(rows),
           "hit_limit": sum(r["hit_limit"] for r in rows) / len(rows),
           "mean_new_tokens": sum(r["new_tokens"] for r in rows) / len(rows),
           "prompt_example": prompts[0], "text_md5": BD.md5_texts(st), "rows": rows,
           "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    os.remove(part)
    print("COT %s %s %s acc %.4f unparsed %.3f hit_limit %.3f mean_new %.0f" % (
        a.arm, a.variant, a.split, res["acc"], res["unparsed"], res["hit_limit"], res["mean_new_tokens"]), flush=True)


if __name__ == "__main__":
    main()
