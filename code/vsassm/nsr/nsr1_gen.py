"""NSR1 held-out log generator (prereg/NSR1_prereg.md): apps/real/gen.py's main loop, unchanged except for the split
(seeds 400+, 10 logs per scenario) and the output directory (nsr/data/). Everything else (world simulation, writer and
checker prompts, acceptance, template fallback) is imported from a read-only copy of apps/real at commit 707351b
(nsr/vendor_707351b/), so the other session's edits to apps/real cannot change this split.

python3 nsr/nsr1_gen.py --split nsr1 --model <Qwen3-32B>      # PACE
python3 nsr/nsr1_gen.py --split nsr1smoke --dry                # local template-only smoke test
"""
import argparse
import json
import os
import sys

NSR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(NSR)
VENDOR = os.path.join(NSR, "vendor_707351b")
DATA = os.path.join(NSR, "data")
sys.path.insert(0, ROOT)    # score_b13 (via product.py)
sys.path.insert(0, VENDOR)  # world, product, gen at 707351b
from gen import accept, checker_prompt, clean, context, fmt, render_template, targeted_prompt, writer_prompt  # noqa: E402,F401
from world import WORLDS  # noqa: E402

HERE = VENDOR  # gen.main refers to HERE; only the data paths are redirected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="nsr1")
    ap.add_argument("--model")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--quant", default="none", choices=["none", "8bit"])
    ap.add_argument("--scen", default="")  # comma-separated scenarios for a shard (default: all four)
    args = ap.parse_args()
    counts = {"nsr1": 10, "nsr2": 10, "nsr3": 10, "nsr1smoke": 1}
    base = {"nsr1": 400, "nsr2": 500, "nsr3": 600, "nsr1smoke": 900}[args.split]  # held-out: NSR1 seeds 400+, NSR2 500+ (dev 100+, test 200+, test2 300+)
    scen = args.scen.split(",") if args.scen else list(WORLDS)
    logs = [WORLDS[sc](base + k).build(k) for sc in scen for k in range(counts[args.split])]
    ckpt = os.path.join(DATA, f"ckpt_{args.split}_{'-'.join(scen)}.json")
    done_upto = -1
    if os.path.exists(ckpt) and not args.dry:  # resume: the texts written so far, line by line
        saved = json.load(open(ckpt))
        done_upto = saved["upto"]
        byid = {l["id"]: l for l in logs}
        for lid, texts in saved["texts"].items():
            for i, rec in texts.items():
                byid[lid]["lines"][int(i)].update(rec)
        print(f"resumed after line {done_upto}", flush=True)
    if not args.dry:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.padding_side = "left"
        q = {}
        if args.quant == "8bit":  # for a single 48 GB card: 32B in int8 (bf16 compute)
            from transformers import BitsAndBytesConfig
            q = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", **q).eval()
        torch.manual_seed(0)

        def gen(prompts, sample):
            outs = []
            for k in range(0, len(prompts), args.bs):
                texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) for p in prompts[k:k + args.bs]]
                enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
                kw = dict(max_new_tokens=220, pad_token_id=tok.pad_token_id)
                kw.update(dict(do_sample=True, temperature=0.9, top_p=0.95) if sample else dict(do_sample=False))
                with torch.no_grad():
                    g = model.generate(**enc, **kw)[:, enc.input_ids.shape[1]:]
                outs += [tok.decode(x, skip_special_tokens=True).strip() for x in g]
            return outs
    stats = {l["id"]: {"human": 0, "fallback": 0, "tries": 0} for l in logs}
    from product import Registry
    regs = {l["id"]: Registry(l) for l in logs}
    for i in range(max(len(l["lines"]) for l in logs)):
        todo = [l for l in logs if i < len(l["lines"]) and l["lines"][i]["kind"] == "human"]
        for l in todo:
            stats[l["id"]]["human"] += 1
        if i <= done_upto:
            stats_fb = [l for l in todo if l["lines"][i].get("fallback")]
            for l in stats_fb:
                stats[l["id"]]["fallback"] += 1
            continue
        if args.dry:
            for l in todo:
                l["lines"][i]["text"] = render_template(l, i)
            continue
        pending = todo
        for t in range(args.tries):
            if not pending:
                break
            ctxs = [context(l, i) for l in pending]
            texts = gen([writer_prompt(l, i, c) for l, c in zip(pending, ctxs)], sample=True)
            texts = [clean(t_, l["lines"][i]["who"]) for t_, l in zip(texts, pending)]
            checks = gen([checker_prompt(l, i, tx, c) for l, tx, c in zip(pending, texts, ctxs)], sample=False)
            tq = [k for k, l in enumerate(pending) if l["lines"][i]["spec"]["mode"] in ("plan", "not_found")]
            ta = gen([targeted_prompt(pending[k], i, texts[k], ctxs[k]) for k in tq], sample=False) if tq else []
            yesno = dict(zip(tq, ta))
            nxt = []
            for k, (l, tx, ck) in enumerate(zip(pending, texts, checks)):
                stats[l["id"]]["tries"] += 1
                if accept(l, i, tx, ck, yesno.get(k), regs[l["id"]]):
                    l["lines"][i]["text"] = tx
                    l["lines"][i]["check"] = ck + (f" | targeted: {yesno[k]}" if k in yesno else "")
                else:
                    l["lines"][i].setdefault("rejected", []).append([tx, ck, yesno.get(k)])
                    nxt.append(l)
            pending = nxt
        for l in pending:
            l["lines"][i]["text"] = render_template(l, i)
            l["lines"][i]["fallback"] = True
            stats[l["id"]]["fallback"] += 1
        if i % 20 == 0:
            print(f"line {i}: {sum(s['fallback'] for s in stats.values())} fallbacks so far", flush=True)
            keep = ("text", "check", "fallback", "rejected")
            json.dump({"upto": i, "texts": {l["id"]: {str(k): {f: x[f] for f in keep if f in x}
                                                      for k, x in enumerate(l["lines"][:i + 1]) if x["kind"] == "human"}
                                            for l in logs}}, open(ckpt + ".tmp", "w"))
            os.replace(ckpt + ".tmp", ckpt)
    os.makedirs(DATA, exist_ok=True)
    by = {}
    for l in logs:
        l["gen"] = stats[l["id"]]
        l["text_lines"] = [fmt(l, x) for x in l["lines"]]
        by.setdefault(l["scenario"], []).append(l)
    for sc, ls in by.items():
        path = os.path.join(DATA, f"{sc}_{args.split}{'_dry' if args.dry else ''}.jsonl")
        with open(path, "w") as f:
            for l in ls:
                f.write(json.dumps(l) + "\n")
        h = sum(s["human"] for l in ls for s in [stats[l["id"]]])
        fb = sum(stats[l["id"]]["fallback"] for l in ls)
        print(f"{path}: {len(ls)} logs, {h} human lines, {fb} template fallbacks", flush=True)



if __name__ == "__main__":
    main()
