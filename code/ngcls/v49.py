"""V-49: the baseline this paper never ran -- prompt the frozen model and read its answer (prereg a89a12b).

Every comparison so far is against a linear probe on frozen states. None is against inference. If prompting is
already near ceiling on bAbI task 1 then the handover's 1.000 buys nothing there, and the paper says so.

The next-token distribution is restricted to the six location words. That is generous to the baseline and it is
the fair comparison, since every arm in this paper also chooses among the same six.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
from v5 import load_babi                                              # noqa: E402
from v30 import STMT, build                                           # noqa: E402

SHORT_MAX, LONG_MIN = 6, 10


def loc_token_ids(tok, vocab):
    """One leading-space token id per location word; skip any that is not single-token."""
    ids, kept = [], []
    for i, w in enumerate(vocab):
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.append(t[0]); kept.append(i)
    return ids, kept


@torch.no_grad()
def score(model, tok, texts, ids, device, bs=8):
    """arg-max over the restricted answer set at the final position."""
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        lg = model(**enc).logits
        last = enc["attention_mask"].sum(-1) - 1
        sel = lg[torch.arange(lg.size(0), device=device), last][:, ids]
        out += sel.argmax(-1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v49_logs/v49.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                                 torch_dtype=torch.float16).to(a.device).eval()

    print("===== V-49  prompting the frozen model, the baseline we never ran =====", flush=True)
    out = {"shots": a.shots, "cells": {}}

    for task in (1, 2):
        d = load_babi(task, a.n_train, 200, a.n_test)
        vocab = d["vocab"]
        ids, kept = loc_token_ids(tok, vocab)
        print("  task %d: %d locations, %d single-token and usable as answers"
              % (task, len(vocab), len(ids)), flush=True)
        if len(ids) < len(vocab):
            print("    NOTE: %d location words are multi-token and are dropped from the answer set; "
                  "instances whose gold answer is one of them are excluded and counted."
                  % (len(vocab) - len(ids)), flush=True)
        keep = set(kept)
        remap = {g: j for j, g in enumerate(kept)}

        pool = [(t, int(y)) for t, y in d["test"] if int(y) in keep]
        dropped = len(d["test"]) - len(pool)
        shots_pool = [(t, int(y)) for t, y in d["train"] if int(y) in keep]

        def buckets(rows):
            b = {"all": rows}
            if task == 1:
                def nst(t):
                    return len(STMT.findall(t))
                b["short"] = [r for r in rows if nst(r[0]) <= SHORT_MAX]
                b["long"] = [r for r in rows if nst(r[0]) >= LONG_MIN]
            return b

        for mode in ("zeroshot", "fewshot"):
            for bname, rows in buckets(pool).items():
                if len(rows) < 50:
                    continue
                accs = []
                for draw in a.draws:
                    rng = np.random.RandomState(1000 + draw)
                    if mode == "fewshot":
                        pick = rng.choice(len(shots_pool), size=a.shots, replace=False)
                        prefix = "".join("%s Answer: %s\n\n" % (shots_pool[k][0].strip().replace("\n", " "),
                                                                vocab[shots_pool[k][1]])
                                         for k in pick)
                    else:
                        prefix = ""
                    texts = [prefix + r[0].strip().replace("\n", " ") + " Answer:" for r in rows]
                    gold = [remap[r[1]] for r in rows]
                    pred = score(model, tok, texts, ids, a.device)
                    accs.append(float(np.mean([p == g for p, g in zip(pred, gold)])))
                key = "task%d/%s/%s" % (task, mode, bname)
                out["cells"][key] = {"acc": accs, "n": len(rows), "dropped_multitoken": dropped}
                print("    %-24s n=%4d  %s mean=%.4f" % (key, len(rows),
                                                         [round(x, 4) for x in accs],
                                                         float(np.mean(accs))), flush=True)
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-49 SCORING (bars registered at a89a12b) =====", flush=True)

    def get(k):
        return out["cells"].get(k, {}).get("acc")

    z1, f1 = get("task1/zeroshot/all"), get("task1/fewshot/all")
    best = max([x for x in (z1, f1) if x], key=lambda v: float(np.mean(v)))
    bn = "fewshot" if best is f1 else "zeroshot"
    print("  task 1 all-lengths: zeroshot %s | fewshot %s -> better is %s"
          % ([round(x, 4) for x in (z1 or [])], [round(x, 4) for x in (f1 or [])], bn), flush=True)
    if all(x >= 0.95 for x in best):
        v = ("THE ALGEBRA ADDS NOTHING on task 1 (prompting is at ceiling; the handover's 1.000 is not a "
             "capability gain and the Discussion must lead with that)")
    elif all(x <= 0.80 for x in best):
        v = ("THE ALGEBRA ADDS SOMETHING MEASURABLE on task 1 (prompting stays well below the handover, and "
             "the paper may say so, scoped to this task)")
    else:
        v = "GRADED on task 1 (report the number, claim nothing either way)"

    zs, zl = get("task1/%s/short" % bn), get("task1/%s/long" % bn)
    if zs and zl:
        drop = [s - l for s, l in zip(zs, zl)]
        print("  task 1 length, %s: short %s long %s -> drop %s (probe lost .183-.195)"
              % (bn, [round(x, 4) for x in zs], [round(x, 4) for x in zl], [round(x, 4) for x in drop]),
              flush=True)
        out["prompting_length_drop"] = drop
        if max(drop) < 0.183:
            v += (" | LENGTH CONTRAST IS ABOUT PROBES: prompting degrades less than the probe did, so the "
                  "contrast does not extend to the model's own inference and the manuscript must say so")
        else:
            v += " | prompting degrades at least as much as the probe with length"
    t2 = get("task2/%s/all" % bn) or get("task2/fewshot/all") or get("task2/zeroshot/all")
    if t2:
        print("  task 2: prompting %s mean=%.4f  against handover 1.0000 and floor .2452"
              % ([round(x, 4) for x in t2], float(np.mean(t2))), flush=True)
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-49 VERDICT: %s" % v, flush=True)
    print("V49-DONE", flush=True)


if __name__ == "__main__":
    main()
