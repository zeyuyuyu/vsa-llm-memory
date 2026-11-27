"""Write the human lines of the realistic scenarios with an LLM, and check each one.

  python3 apps/real/gen.py --split dev --dry                 # templates only (local check of the pipeline)
  python3 apps/real/gen.py --split test --model <Qwen3-32B>  # on PACE: writer + checker

For every human line, the writer gets the channel, the speaker, the state changes the line must convey (or, for
plans, failed searches and chatter, that it must convey none), how people refer to the things involved, and the two
lines before it. The checker, a separate greedy pass of the same model with a different prompt, lists the location
changes it reads in the written line, using registry IDs. A line is kept only when the checker's list equals the
intended one; otherwise it is rewritten (up to --tries times) and then falls back to a template. The number of
template fallbacks is recorded per log.

Output: apps/real/data/<scenario>_<split>.jsonl, one log per line, every line with its final "text".
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repo root, for score_b13 (via product.py)
from world import WORLDS  # noqa: E402

CHANNEL = {
    ("warehouse", "chat"): "the #floor channel in Microsoft Teams at a 3PL warehouse. Messages are short and informal, written on a phone by forklift drivers and leads",
    ("warehouse", "handover"): "the end-of-shift handover note the shift lead posts in the #floor channel. A few plain sentences",
    ("biobank", "bench"): "the lab's electronic bench notebook. Short entries written by the person at the bench",
    ("biobank", "email"): "a short email to the biobank staff mailing list",
    ("biobank", "chat"): "the biobank team's Slack channel. Short, informal messages",
    ("home", "speech"): "speech picked up by a home robot's microphone and transcribed automatically. A family member talking in casual spoken English, with fillers like 'uh' or 'okay', exactly as an automatic transcript would show it",
    ("ops", "incident"): "the #inc-2291 incident channel in Slack. Terse messages from site reliability engineers during an incident",
}
VERB = {"locations": "ends up in or at", "containers": "ends up directly inside"}


def refs(log, name):
    al = {e["id"]: e["aliases"] for e in log["registry"]["items"] + log["registry"]["places"]}
    opts = [name] + al.get(name, [])
    return " or ".join(f'"{o}"' for o in opts)


def describe(log, ev):
    op, a, b = ev
    if op == "swap":
        return f"everything that was in {refs(log, a)} and everything that was in {refs(log, b)} traded places"
    return f"{refs(log, a)} {VERB[log['domain']]} {refs(log, b)}"


def writer_prompt(log, i, ctx):
    line = log["lines"][i]
    sp = line["spec"]
    mode = sp["mode"]
    chan = CHANNEL[(log["scenario"], line["src"])]
    head = (f"Write one message for {chan}. The writer is {line['who']}.\n\n"
            f"Messages just before it:\n{ctx or '(none)'}\n\n")
    if mode in ("report", "follow_up", "swap", "handover"):
        evs = "\n".join(f"- {describe(log, ev)}" for ev in line["events"])
        extra = {"report": "Say it as something the writer just did or saw" + (f" (the reason: {sp['why']})" if sp.get("why") else "") + ".",
                 "follow_up": "The item was the subject of the previous message: refer to it with a pronoun or a short phrase such as 'that one' instead of its name.",
                 "swap": "Say that the contents of the two places were exchanged.",
                 "handover": "Write it as a handover: where things stand at the end of the shift. Mention each fact once."}[mode]
        task = (f"The message must convey exactly these facts and no other change of location:\n{evs}\n{extra} "
                "Put it in your own words, the way people at this site talk (pulled, dropped, parked, stuck it in, "
                "moved, put, left, it's in ...); never use the phrases 'is now in', 'ends up' or 'directly inside'. "
                "Refer to things the way people at this site do (any of the quoted names). Do not mention where "
                "anything was before.")
    elif mode == "plan":
        task = (f"The writer suggests or plans moving {refs(log, sp['item'])} to {refs(log, sp['place'])}, but it has NOT been "
                "done yet (or someone says no). Make it clear that it has not happened. Do not say where it is now.")
    elif mode == "not_found":
        task = (f"The writer looked for {refs(log, sp['item'])} at {refs(log, sp['place'])} and it was not there. Do not say "
                "where it actually is.")
    else:
        task = ("The message is ordinary chatter for this setting (breaks, equipment, people, the weather, a problem with a "
                "system). It must not say where any item is or that anything was moved.")
    size = "a few short sentences" if mode == "handover" else "one or two short sentences"
    return head + task + f"\n\nKeep it to {size}. Reply with the message text only, without the writer's name, quotation marks or a timestamp."


def checker_prompt(log, i, text, ctx):
    items = "; ".join(f'{e["id"]} (also called {", ".join(e["aliases"])})' if e["aliases"] else e["id"]
                      for e in log["registry"]["items"])
    places = "; ".join(f'{e["id"]} (also called {", ".join(e["aliases"])})' if e["aliases"] else e["id"]
                       for e in log["registry"]["places"])
    rel = "is now in or at" if log["domain"] == "locations" else "is now directly inside"
    return (f"Earlier messages:\n{ctx or '(none)'}\n\nNew message: {text}\n\n"
            f"Items: {items}\nPlaces: {places}\n\n"
            f"List what the new message says about where items are now: [\"move\", ITEM, PLACE] when it says an item {rel} a "
            "place (moved there, left there, or simply stated to be there now), and [\"swap\", PLACE, PLACE] when the "
            "contents of two places were exchanged. Use the IDs exactly as listed. Leave out plans, things that did not "
            "happen, searches that failed, and where something used to be. Resolve pronouns from the earlier messages.\n"
            'Reply with JSON only: {"events": [...]}')


TRIPLE = re.compile(r'\[\s*"(move|swap)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', re.I)


def parse(txt, reg=None):
    """The checker's events; with a registry, names are resolved to IDs the way the product resolves them, so that
    a checker writing "QA" for "QA HOLD" is not counted as a disagreement."""
    out = []
    for op, a, b in TRIPLE.findall(txt):
        op, a, b = op.lower(), a.strip(), b.strip()
        if reg is not None:
            if op == "move":
                a, b = reg.resolve(a, reg.items) or a, reg.resolve(b, reg.places) or b
            else:
                a, b = reg.resolve(a, reg.places) or a, reg.resolve(b, reg.places) or b
        out.append((op,) + ((min(a, b), max(a, b)) if op == "swap" else (a, b)))
    return sorted(out)


def targeted_prompt(log, i, text, ctx):
    """A plain yes/no question for lines that must not move anything: a plan, or a failed search."""
    sp = log["lines"][i]["spec"]
    item, place = sp["item"], sp["place"]
    if sp["mode"] == "plan":
        q = f"Does the new message say that {item} has ALREADY been moved to {place} (not just planned, suggested or requested)?"
    else:
        q = f"Does the new message say where {item} is right now?"
    return (f"Earlier messages:\n{ctx or '(none)'}\n\nNew message: {text}\n\n{q} Here {item} may be called "
            f"{refs(log, item)}. Answer with one word: yes or no.")


def names_other_item(log, text, reg):
    """True when the text names an item by ID, alias or the last digits of its ID (a follow-up must use a pronoun)."""
    t = " " + reg.key(text) + " "
    for it in reg.items:
        for n in [it] + next(e["aliases"] for e in log["registry"]["items"] if e["id"] == it):
            k = reg.key(n)
            if len(k) >= 3 and f" {k} " in t:
                return True
    return False


def accept(log, i, text, check, yesno, reg):
    """Is the written line faithful? report/swap/handover: the checker's events equal the intended ones.
    follow_up: one move to the intended place and no item named (the checker may resolve the pronoun to the wrong
    item; the reader is scored on the intended one). plan/not_found: the targeted question is answered no and the
    checker sees no other change. noise: no change at all."""
    line = log["lines"][i]
    mode = line["spec"]["mode"]
    got = parse(check, reg)
    want = sorted((op,) + ((min(a, b), max(a, b)) if op == "swap" else (a, b)) for op, a, b in line["events"])
    if not text:
        return False
    if mode in ("report", "swap", "handover"):
        return got == want
    if mode == "follow_up":
        return (len(got) == 1 and got[0][0] == "move" and got[0][2] == want[0][2]
                and not names_other_item(log, text, reg))
    if mode in ("plan", "not_found"):
        item = line["spec"]["item"]
        rest = [e for e in got if not (e[0] == "move" and e[1] == item)]
        return yesno is not None and yesno.strip().lower().startswith("no") and not rest
    return got == []


def clean(t, who):
    """The writer's text without a leading speaker name, quotes or a second paragraph."""
    t = t.strip().split("\n\n")[0].strip()
    for _ in range(2):
        t = re.sub(rf"^(?:@?{re.escape(str(who))}|[A-Z][a-z]+)\s*:\s*", "", t).strip()
        t = t.strip().strip('"').strip("\u201c\u201d").strip()
    return t


def render_template(log, i, ctx=None):
    line = log["lines"][i]
    sp, ev = line["spec"], line["events"]
    al = {e["id"]: e["aliases"] for e in log["registry"]["items"] + log["registry"]["places"]}
    nm = lambda x: (al.get(x) or [x])[0]
    mode = sp["mode"]
    if mode in ("report", "follow_up"):
        a, b = ev[0][1], ev[0][2]
        return (f"moved {'it' if mode == 'follow_up' else nm(a)} to {nm(b)}" + (f" ({sp['why']})" if sp.get("why") else ""))
    if mode == "swap":
        return f"swapped what was in {nm(ev[0][1])} and {nm(ev[0][2])}"
    if mode == "handover":
        return "Handover: " + "; ".join(f"{nm(a)} is in {nm(b)}" for _, a, b in ev) + "."
    if mode == "plan":
        return f"thinking of moving {nm(sp['item'])} to {nm(sp['place'])} later, not done yet"
    if mode == "not_found":
        return f"checked {nm(sp['place'])} for {nm(sp['item'])}, not there"
    return "taking my break now"


def fmt(log, line):
    if line["kind"] == "struct":
        return line["text"]
    if log["scenario"] == "ops":
        return f"[{line['time']}] @{line['who']}: {line['text']}"
    if log["scenario"] == "home":
        return f"{line['time']}  SPEECH  {line['who']}: \"{line['text']}\""
    if line["src"] == "email":
        return f"[{line['time']}] email from {line['who']}: {line['text']}"
    return f"[{line['time']}] {line['who']}: {line['text']}"


def context(log, i, k=4):
    return "\n".join(fmt(log, l) for l in log["lines"][max(0, i - k):i] if l.get("text"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--quant", default="none", choices=["none", "8bit"])
    ap.add_argument("--scen", default="")  # comma-separated scenarios for a shard (default: all four)
    args = ap.parse_args()
    counts = {"dev": 2, "test": 10, "test2": 10}
    base = {"dev": 100, "test": 200, "test2": 300}[args.split]  # test2: fresh logs after the follow-up fix
    scen = args.scen.split(",") if args.scen else list(WORLDS)
    logs = [WORLDS[sc](base + k).build(k) for sc in scen for k in range(counts[args.split])]
    ckpt = os.path.join(HERE, "data", f"ckpt_{args.split}_{'-'.join(scen)}.json")
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
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    by = {}
    for l in logs:
        l["gen"] = stats[l["id"]]
        l["text_lines"] = [fmt(l, x) for x in l["lines"]]
        by.setdefault(l["scenario"], []).append(l)
    for sc, ls in by.items():
        path = os.path.join(HERE, "data", f"{sc}_{args.split}{'_dry' if args.dry else ''}.jsonl")
        with open(path, "w") as f:
            for l in ls:
                f.write(json.dumps(l) + "\n")
        h = sum(s["human"] for l in ls for s in [stats[l["id"]]])
        fb = sum(stats[l["id"]]["fallback"] for l in ls)
        print(f"{path}: {len(ls)} logs, {h} human lines, {fb} template fallbacks", flush=True)


if __name__ == "__main__":
    main()
