"""Build the showcase page for the realistic scenarios: apps/real/page/readable_memory_in_practice.html.

  python3 apps/real/build_real.py --split test --llm qwen3-1.7b qwen3-8b qwen3-32b

Every number on the page is computed here from the run files; the template only renders it.
"""
import argparse
import glob
import json
import os
import re

from run_real import HERE, OUT, SCEN, gid, load, ok, resolve_answer
from product import Registry

LLM_NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B"}
INFO = {
    "warehouse": {"title": "Warehouse floor", "sor": "WMS alone",
                  "who": "3PLs and warehouse operators; yard and dock management",
                  "ask": "Where is each pallet now, when many moves are never scanned and only mentioned in the floor chat?",
                  "src": {"wms": "WMS scan", "chat": "floor chat", "handover": "handover note"}},
    "biobank": {"title": "Biobank custody", "sor": "LIMS alone",
                "who": "Biobanks, clinical and pharma labs, chain-of-custody audits",
                "ask": "Which box, rack and freezer holds each specimen, when racks are moved for a defrost and the move is only in an email?",
                "src": {"lims": "LIMS audit", "bench": "bench notebook", "email": "email", "chat": "lab Slack"}},
    "home": {"title": "Home robot", "sor": "Camera alone",
             "who": "Home and elder-care robots, smart-home assistants",
             "ask": "Where are Grandpa's glasses, keys and pills, when the camera cannot see into pockets, drawers or the car?",
             "src": {"vision": "camera", "robot": "robot", "speech": "heard"}},
    "ops": {"title": "Cloud incident", "sor": "Bots alone",
            "who": "SRE and platform teams; incident and on-call copilots",
            "ask": "Where is every service and database primary running, after an incident where engineers moved things by hand?",
            "src": {"deploy": "deploy bot", "infra": "Patroni", "alerts": "alert", "incident": "incident channel"}},
}


def parse_forced(q, fa, reg):
    """A budget-forced answer is a sentence ("The final location of LPN 0499237 is **DOOR 3**."), sometimes followed
    by an explanation or cut at 64 tokens: the first bold span that names an answer (for a chain, one with an arrow
    first), else the first line, else the whole text."""
    spans = re.findall(r"\*\*(.+?)\*\*", fa)
    if q["kind"] == "chain":
        spans = [x for x in spans if ">" in x] + spans
    for t in spans + [fa.strip().split("\n")[0]]:
        a = resolve_answer(q, t, reg)
        if a is not None:
            return a
    return resolve_answer(q, fa, reg)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--llm", nargs="*", default=["qwen3-1.7b", "qwen3-8b", "qwen3-32b"])
    ap.add_argument("--reader_file")
    ap.add_argument("--retired")  # reader output on the retired first test set, same configuration (for the notes)
    ap.add_argument("--llm_dir")
    ap.add_argument("--template", default=os.path.join(HERE, "page_template.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "page", "readable_memory_in_practice.html"))
    args = ap.parse_args()
    logs = load(args.split, args.suffix)
    rd = {r["id"]: r for r in map(json.loads, open(args.reader_file or os.path.join(OUT, f"reader_{args.split}{args.suffix}.jsonl")))}
    llms, forced = {}, {}
    for m in args.llm:
        g = {}
        for p in glob.glob(os.path.join(args.llm_dir or OUT, f"{m}_REAL*.jsonl")):
            for r in map(json.loads, open(p)):
                g[(r["n"], r["idx"])] = r
        if g:
            llms[m] = g
            fp = os.path.join(args.llm_dir or OUT, f"forced_{m}.jsonl")  # budget-forced answers (force_answer.py)
            forced[m] = {(r["n"], r["idx"]): r["forced_answer"] for r in map(json.loads, open(fp))} if os.path.exists(fp) else {}
    acc = {sc: {"rm": [], "sor": [], "ms": [], "logs": 0, "lines": 0, "human_lines": 0, "tok": 0, "hidden": 0, "fb": 0, "gh": 0,
                "changes": 0, "llm": {m: {"ok": [], "strict": [], "think": [], "prompt": [], "limit": 0} for m in llms}} for sc in SCEN}
    show = {}
    for l in logs:
        n = gid(l)
        reg = Registry(l)
        r = rd[l["id"]]
        a = acc[l["scenario"]]
        a["logs"] += 1
        a["fb"] += l["gen"]["fallback"]  # human lines whose writer text failed the check and fell back to a template
        a["gh"] += l["gen"]["human"]
        a["lines"] += len(l["lines"])
        a["human_lines"] += r["reader_lines"]
        a["tok"] += r["gen_tokens"]
        a["changes"] += sum(len(x["events"]) for x in l["lines"])
        a["hidden"] += sum(len(x["events"]) for x in l["lines"] if x["kind"] == "human")
        rows = []
        for qi, q in enumerate(l["questions"]):
            ans = r["answers"][qi]
            okr, oks = ok(q, ans["pred"], reg), ok(q, q["sor"], reg)
            a["rm"].append(okr)
            a["sor"].append(oks)
            a["ms"].append(ans["ms"])
            lrow = {}
            for m, g in llms.items():
                x = g.get((n, qi))
                if x is None:
                    continue
                strict = (not x["hit_limit"]) and ok(q, resolve_answer(q, x["answer_text"], reg), reg)
                fa = forced[m].get((n, qi)) if x["hit_limit"] else None
                okl = strict or (fa is not None and ok(q, parse_forced(q, fa, reg), reg))
                s = a["llm"][m]
                s["ok"].append(okl)
                s["strict"].append(strict)
                s["think"].append(x["think_tokens"])
                s["prompt"].append(x.get("prompt_tokens", 0))
                s["limit"] += bool(x["hit_limit"])
                lrow[m] = {"ans": (f"{fa.strip()} (reasoning hit the 8,192-token limit; answer forced)" if fa is not None else
                                   "(no answer: reasoning hit the 8,192-token limit)") if x["hit_limit"] else x["answer_text"].strip(),
                           "ok": okl, "think": x["think_tokens"]}
            rows.append({"q": q["q"], "kind": q["kind"], "truth": q["a"], "sor": q["sor"], "sor_ok": oks,
                         "rm": ans["pred"], "rm_ok": okr, "conf": ans["conf"], "llm": lrow})
        if l["id"].rsplit("-", 1)[1] == "0":  # the first log of each scenario is the one shown
            show[l["scenario"]] = {"id": l["id"], "domain": l["domain"],
                                   "lines": [{"t": t, "src": x["src"], "kind": x["kind"], "truth": x["events"]}
                                             for t, x in zip(l["text_lines"], l["lines"])],
                                   "events": r["events"], "snapshots": r["snapshots"], "questions": rows,
                                   "demo": r["demo"]}
    summary = {}
    for sc, a in acc.items():
        if not a["logs"]:
            continue
        summary[sc] = {"n": len(a["rm"]), "logs": a["logs"], "rm_acc": mean(a["rm"]), "sor_acc": mean(a["sor"]),
                       "ms": mean(a["ms"]), "lines_per_log": a["lines"] / a["logs"],
                       "tok_per_human_line": a["tok"] / max(1, a["human_lines"]),
                       "hidden_share": a["hidden"] / max(1, a["changes"]),
                       "fallback_share": a["fb"] / max(1, a["gh"]),
                       "llm": {m: {"acc": mean(s["ok"]), "acc_strict": mean(s["strict"]), "n": len(s["ok"]), "think": mean(s["think"]),
                                   "prompt": mean(s["prompt"]), "limit": s["limit"]} for m, s in a["llm"].items() if s["ok"]}}
    allv = lambda key: [x for a in acc.values() for x in a[key]]
    total = {"n": len(allv("rm")), "logs": sum(a["logs"] for a in acc.values()), "rm_acc": mean(allv("rm")),
             "sor_acc": mean(allv("sor")),
             "llm": {m: {"acc": mean([x for a in acc.values() for x in a["llm"][m]["ok"]]),
                         "acc_strict": mean([x for a in acc.values() for x in a["llm"][m]["strict"]]),
                         "think": mean([x for a in acc.values() for x in a["llm"][m]["think"]])} for m in llms}}
    retired = {}
    if args.retired:
        rr = {r["id"]: r for r in map(json.loads, open(args.retired))}
        for l in load("test"):
            reg = Registry(l)
            retired.setdefault(l["scenario"], []).extend(ok(q, rr[l["id"]]["answers"][qi]["pred"], reg)
                                                         for qi, q in enumerate(l["questions"]))
        retired = {sc: mean(v) for sc, v in retired.items()}
    data = {"info": INFO, "retired": retired, "tabs": [s for s in SCEN if s in show], "show": show, "summary": summary, "total": total,
            "llms": [{"key": m, "name": LLM_NAMES.get(m, m)} for m in llms]}
    html = open(args.template).read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(html)
    print(args.out, f"{len(html) / 1e6:.2f} MB")
    for sc, s in summary.items():
        print(f"  {sc:10s} RM {s['rm_acc']:.3f}  SoR {s['sor_acc']:.3f}  " +
              "  ".join(f"{m} {v['acc']:.3f}" for m, v in s["llm"].items()) + f"  (n={s['n']}, hidden {s['hidden_share']:.2f})")
    print("  total", round(total["rm_acc"], 3), "SoR", round(total["sor_acc"], 3), {m: round(v["acc"], 3) for m, v in total["llm"].items()})


if __name__ == "__main__":
    main()
