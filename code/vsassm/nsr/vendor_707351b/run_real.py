"""Run and score the realistic application scenarios.

  python3 apps/real/run_real.py reader --model <Qwen3-1.7B> --adapter b16_models/student_s0 --split dev
  python3 apps/real/run_real.py prep_llm --split test      # -> apps/real/llm_problems_<split>.jsonl (b6_qwen.py --task REAL)
  python3 apps/real/run_real.py score --split test [--llm qwen3-8b ...]

The reader configuration is the shipped one (distilled Qwen3-1.7B student, prompt C, one line at a time); system
feeds go through connectors and every name through the site registry (product.py). Nothing is tuned on test.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from product import Registry, SiteMemory, connect  # noqa: E402
from score_b13 import norm  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
SCEN = ["warehouse", "biobank", "home", "ops"]
UNKNOWN = {"warehouse": "Where is LPN 0999999?", "biobank": "Where is S-26-99999?", "home": "Where is the stapler?",
           "ops": "Where is video-transcoder running?"}


def gid(log):
    """A problem number that does not depend on which scenario files exist yet: warehouse 0-99, biobank 100-199,
    home 200-299, ops 300-399 (the log's index within its scenario)."""
    return SCEN.index(log["scenario"]) * 100 + int(log["id"].rsplit("-", 1)[1])


def load(split, suffix=""):
    logs = []
    for sc in SCEN:
        p = os.path.join(DATA, f"{sc}_{split}{suffix}.jsonl")
        if os.path.exists(p):
            logs += [json.loads(l) for l in open(p)]
    return logs


def message(log, line):
    """What the reader is shown for a human line: speaker and message, as in the channel, without the timestamp."""
    t = log["text_lines"][log["lines"].index(line)]
    return re.sub(r"^\[\d\d:\d\d\]\s*|^\d\d:\d\d\s+SPEECH\s+", "", t)


def run_reader(args):
    import b14
    b14.MERGE, b14.GUARD = "none", False
    logs = load(args.split, args.suffix)
    reader = b14.Reader(args.model, args.adapter)
    sites = [SiteMemory(l, b14) for l in logs]
    rec = [{"events": [], "snapshots": [], "tokens": 0, "reader_lines": 0, "raw": {}, "vetoed": 0, "chosen": 0} for _ in logs]
    t0 = time.time()
    for i in range(max(len(l["lines"]) for l in logs)):
        live = [j for j, l in enumerate(logs) if i < len(l["lines"])]
        human, jobs = [], []
        for j in live:
            l, line = logs[j], logs[j]["lines"][i]
            evs = connect(l["scenario"], l["text_lines"][i]) if line["kind"] == "struct" else None
            if evs is not None:
                rec[j]["events"].append(sites[j].apply(evs, from_reader=False))
                continue
            # the reader sees the thread it is reading: the last three messages of the same channel
            same = [message(l, x) for x in l["lines"][:i] if x["kind"] == "human" and x["src"] == line["src"]][-3:]
            human.append(j)
            jobs.append((sites[j], message(l, line), " ".join(same) or "(none)", line["src"]))
        if jobs:
            for j, r in zip(human, read_messages(reader, b14, jobs, args.verify)):
                rec[j]["tokens"] += r["tokens"]
                rec[j]["reader_lines"] += 1
                rec[j]["raw"][i] = r["raw"]
                rec[j]["vetoed"] += r["vetoed"]
                rec[j]["chosen"] += r["chosen"]
                rec[j]["events"].append(r["events"])
        for j in live:
            rec[j]["snapshots"].append(sites[j].board())
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"reader_{args.split}{args.suffix}{'_verify' if args.verify else ''}{args.tag}.jsonl")
    with open(path, "w") as f:
        for j, l in enumerate(logs):
            s, ans = sites[j], []
            for q in l["questions"]:
                t1 = time.time()
                a, c, ent = s.ask(q["q"], q["kind"])
                ans.append({"pred": a, "conf": round(c, 3), "entity": ent, "ms": round(1000 * (time.time() - t1), 3)})
            demo = correction_demo(l, s)
            f.write(json.dumps({"id": l["id"], "scenario": l["scenario"], "answers": ans, "events": rec[j]["events"],
                                "snapshots": rec[j]["snapshots"], "gen_tokens": rec[j]["tokens"],
                                "reader_lines": rec[j]["reader_lines"], "dropped": s.dropped, "vetoed": rec[j]["vetoed"],
                                "chosen": rec[j]["chosen"],
                                "raw": rec[j]["raw"], "demo": demo}) + "\n")
    print(f"{path}: {len(logs)} logs in {time.time() - t0:.0f}s, "
          f"{sum(r['tokens'] for r in rec) / max(1, sum(r['reader_lines'] for r in rec)):.1f} tokens per human line", flush=True)


def read_messages(reader, b14, jobs, verify=True):
    """Read human messages in one batch and write what they say into their sites: the step shared by the evaluation
    (run_reader) and the SDK (readable_memory.SiteMemory.observe). jobs: (site, message, thread context, channel).

    The reader writes the changes; with verify, a single move in a message that names several places ("Rack 4 is in
    the -80 #5, FRZ-07 is being defrosted") first gets a forced choice among those places, in the message's own words
    and in registry order, and every change then takes a one-word check that the message says it actually happened.
    Returns, per job: the events written, the reader's raw text, generated tokens, and vetoed / chosen counts."""
    prompts = [b14.PROMPTS[("C", site.log["domain"])].format(
        context=ctx, sentence=msg, objs=", ".join(site.mem.objs) or "(none)",
        places=", ".join(site.mem.places) or "(none)") for site, msg, ctx, channel in jobs]
    outs = reader(prompts)
    res = [{"raw": txt, "tokens": n, "vetoed": 0, "chosen": 0} for txt, n in outs]
    parsed = [b14.parse(txt) or [] for txt, n in outs]
    if verify:
        ch, say = [], {}  # say: (message, event) -> the chosen place as the message names it
        for j, (site, msg, ctx, channel) in enumerate(jobs):
            parsed[j] = [site.orient(ev) for ev in parsed[j]]
            single = sum(ev[0].lower() == "move" for ev in parsed[j]) == 1
            for k, ev in enumerate(parsed[j]):
                cands = sorted(site.places_named(msg, ev)) if single and ev[0].lower() == "move" else []
                if len(cands) >= 2:
                    ch.append((j, k, cands, [site.reg.said_as(msg, p) for p in cands]))
        if ch:
            co = reader([choice_prompt(jobs[j][1], parsed[j][k][1], said) for j, k, cands, said in ch])
            for (j, k, cands, said), (txt, n) in zip(ch, co):
                res[j]["tokens"] += n
                m = re.match(r"\s*\(?([A-Z])\b", txt.strip().upper())
                pick = ord(m.group(1)) - 65 if m else -1
                if 0 <= pick < len(cands):
                    ev = parsed[j][k]
                    parsed[j][k] = [ev[0], ev[1], cands[pick]]
                    say[(j, k)] = said[pick]
                    res[j]["chosen"] += 1
        vq = [(j, k, ev) for j in range(len(jobs)) for k, ev in enumerate(parsed[j])]
        if vq:
            vo = reader([verify_prompt(jobs[j][1], [ev[0], ev[1], say.get((j, k), ev[2])], jobs[j][0].log["domain"])
                         for j, k, ev in vq])
            keep = {}
            for (j, k, ev), (txt, n) in zip(vq, vo):
                res[j]["tokens"] += n
                if txt.strip().lower().startswith("yes"):
                    keep.setdefault(j, []).append(ev)
                else:
                    res[j]["vetoed"] += 1
            parsed = [keep.get(j, []) for j in range(len(jobs))]
    for j, (site, msg, ctx, channel) in enumerate(jobs):
        res[j]["events"] = site.apply(parsed[j], from_reader=True, channel=channel, text=msg)
    return res


def verify_prompt(msg, ev, domain):
    if ev[0].lower() == "swap":
        q = f"were the contents of {ev[1]} and {ev[2]} actually exchanged"
    else:
        q = f"is {ev[1]} now actually {'in or at' if domain == 'locations' else 'inside'} {ev[2]}"
    return (f"Message: {msg}\n\nAccording to this message, {q}? Answer yes if the message says it was put, left or "
            "moved there, or that it is there now. Answer no if it is only planned, suggested or asked about, if "
            "someone looked there and did not find it, or if it is negated. Answer with one word: yes or no.")


def choice_prompt(msg, item, places):
    opts = "  ".join(f"{chr(65 + n)}) {p}" for n, p in enumerate(places))
    return (f"Message: {msg}\n\nAccording to this message, where is {item} now? Choose one: {opts}\n"
            "Answer with the letter only.")


def correction_demo(l, s):
    before = s.board()
    q = l["questions"][0]
    target = q["entity"]
    if l["domain"] == "containers":
        boxes = [i for i in s.reg.items if i.startswith("CB-") and i in before]
        target = boxes[0] if boxes else target
        pool = [p for p in s.reg.places if p.startswith("R-")]
    else:
        pool = s.reg.places
    cur = before.get(target, [None])[0]
    cur1 = cur[0] if isinstance(cur, list) and cur else cur
    new = next((p for p in pool if p != cur1), None)
    if new is None:
        return None
    s.apply([["move", target, new]], from_reader=False)
    after = s.board()
    a, c, _ = s.ask(UNKNOWN[l["scenario"]], "where")
    return {"target": target, "from": cur, "to": new, "changed": sorted(k for k in after if after[k][0] != before.get(k, [None])[0]),
            "n": len(after), "unknown_q": UNKNOWN[l["scenario"]], "unknown_answer": a, "unknown_conf": c}


# ------------------------------------------------------------------------------------------------ LLM reference
ASK = {"where": "Answer with the location only.",
       "what_in": "Answer with the items only, separated by commas, or \"nothing\".",
       "chain": "Answer with the containers that hold it, from the one directly holding it to the outermost, separated by \" > \"."}


def prep_llm(args):
    logs = [l for l in load(args.split, args.suffix) if not args.scen or l["scenario"] in args.scen.split(",")]
    path = os.path.join(HERE, f"llm_problems_{args.split}{args.suffix}{args.tag}{'_' + args.scen.replace(',', '-') if args.scen else ''}.jsonl")
    with open(path, "w") as f:
        for l in logs:
            n = gid(l)
            text = "\n".join(l["text_lines"])
            intro = {"warehouse": "Below is today's record from a warehouse: WMS scanner transactions and the floor team's chat, in time order.",
                     "biobank": "Below is today's record from a biobank: the LIMS audit trail, bench notebook entries, emails and chat, in time order.",
                     "home": "Below is today's record from a home robot: what its camera saw and what it heard people say, in time order.",
                     "ops": "Below is today's record from a cloud platform: deploy bot and database events, alerts and the incident channel, in time order."}[l["scenario"]]
            assets = assets_block(l)
            for qi, q in enumerate(l["questions"]):
                prompt = (f"{intro} Answer the question about the state at the end of the record.\n\n{assets}\n\n{text}\n\n"
                          f"Question: {q['q']}\n{ASK[q['kind']]}")
                f.write(json.dumps({"task": "REAL", "n": n, "idx": qi, "id": l["id"], "prompt": prompt}) + "\n")
    print(path)


def assets_block(l):
    """The site's asset list, as the product is given it: every ID with the other names people use for it."""
    row = lambda e: e["id"] + (": " + ", ".join(e["aliases"]) if e["aliases"] else "")
    return ("Site asset list (each ID, then other names people use for it; a number can stand for the ID that ends in "
            "it). Answer with these IDs.\nItems:\n" + "\n".join(row(e) for e in l["registry"]["items"]) +
            "\nPlaces:\n" + "\n".join(row(e) for e in l["registry"]["places"]))


# ------------------------------------------------------------------------------------------------ scoring
def ok(q, pred, reg):
    if q["kind"] == "where":
        return pred is not None and pred == q["a"]
    if q["kind"] == "what_in":
        return isinstance(pred, list) and sorted(pred) == sorted(q["a"])
    return isinstance(pred, list) and list(pred) == list(q["a"])


def resolve_answer(q, txt, reg):
    """An LLM's free-text answer, mapped to registry IDs the same way the product maps names. Lenient: the last line
    (for a chain, the last line with an arrow), without markdown, an "Answer:" lead-in or slot positions ("CB-007/A5")."""
    lines = [x.strip() for x in txt.strip().split("\n") if x.strip()]
    if not lines:
        return None
    arrows = [x for x in lines if ">" in x or "→" in x]
    t = arrows[-1] if q["kind"] == "chain" and arrows else lines[-1]
    t = re.sub(r"\*\*|`", "", t)
    t = re.sub(r"^(?:final answer|answer)\s*:\s*", "", t, flags=re.I).strip().strip(".").strip()
    name = lambda p, pool: reg.resolve(p, pool) or reg.find_in(p, pool, message=True)
    if q["kind"] == "where":  # the last line that names a place ("db-07 (final state)" before a closing remark)
        for x in [t] + [re.sub(r"\*\*|`", "", y) for y in reversed(lines[:-1])]:
            if name(x, reg.places):
                return name(x, reg.places)
        return None
    if q["kind"] == "what_in":
        if re.search(r"\b(nothing|none|empty)\b", t, re.I):
            return []
        parts = [p for p in re.split(r",|\band\b|;", t) if p.strip()]
        out = [name(p, reg.items) for p in parts]
        return None if None in out else out
    parts = [re.sub(r"/\S*|\(.*?\)", "", p).strip() for p in re.split(r">|→", t) if p.strip()]
    out = [name(p, reg.places) for p in parts if p]
    if out and out[0] == reg.find_in(q["q"]):  # "CB-016 > R-04 > FRZ-05" for "Where is CB-016?": the chain after it
        out = out[1:]
    return None if None in out or not out else out


def score(args):
    logs = load(args.split, args.suffix)
    rd = {r["id"]: r for r in map(json.loads, open(args.reader_file or os.path.join(OUT, f"reader_{args.split}{args.suffix}.jsonl")))}
    llm = {}
    for m in args.llm or []:
        g = {}
        for p in glob.glob(os.path.join(args.llm_dir or OUT, f"{m}_REAL*.jsonl")):
            for r in map(json.loads, open(p)):
                g[(r["n"], r["idx"])] = r
        llm[m] = g
    res = {}
    for l in logs:
        n = gid(l)
        reg = Registry(l)
        r = rd.get(l["id"])
        if r is None:
            continue
        for qi, q in enumerate(l["questions"]):
            k = res.setdefault(l["scenario"], {"rm": [], "sor": [], **{m: [] for m in llm}})
            k["rm"].append(ok(q, r["answers"][qi]["pred"], reg))
            k["sor"].append(ok(q, q["sor"], reg))
            for m, g in llm.items():
                x = g.get((n, qi))
                if x is not None:
                    k[m].append(not x["hit_limit"] and ok(q, resolve_answer(q, x["answer_text"], reg), reg))
    for sc, k in res.items():
        print(f"{sc:10s} " + "  ".join(f"{name} {sum(v) / len(v):.3f} ({sum(v)}/{len(v)})" for name, v in k.items() if v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reader", "prep_llm", "score"])
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--suffix", default="")  # "_dry" for the template-only files
    ap.add_argument("--llm", nargs="*")
    ap.add_argument("--verify", action="store_true")  # second pass of the reader: did it actually happen?
    ap.add_argument("--reader_file")
    ap.add_argument("--llm_dir")  # score: where the LLM outputs are (default apps/real/out)
    ap.add_argument("--tag", default="")  # reader: appended to the output name ("_v2")
    ap.add_argument("--scen", default="")  # prep_llm: only these scenarios  # score: which reader output file (default reader_<split><suffix>.jsonl)
    args = ap.parse_args()
    {"reader": run_reader, "prep_llm": prep_llm, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
