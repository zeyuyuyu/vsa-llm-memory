"""NSR1 LLM arms (prereg/NSR1_prereg.md): prompts and answer parsing.

  arm 3   the whole log + site asset list + site catalog (apps/real protocol: same intro and asset block)
  arm 3b  the Readable Memory state written as sentences + asset list + catalog, no log (diagnostic)

python3 nsr/nsr1_llm.py prep --split nsr1 --logs nsr/data [--reader <reader.jsonl> for 3b]
python3 nsr/nsr1_llm.py parse --split nsr1 --logs nsr/data --problems <problems.jsonl> \
    --runs 'nsr/out/llm/qwen3-32b_NSR1_n*.jsonl' [--forced nsr/out/llm/forced_qwen3-32b.jsonl] --arm arm3_qwen3-32b
Runs come from b6_qwen.py (--task NSR1; records n, idx, answer_text, hit_limit) and budget forcing from
nsr/vendor_707351b/force_answer.py (records n, idx, forced_answer). parse writes two answer files: <arm>_strict (a run
that hit the reasoning limit gives no answer) and <arm>_forced (its forced answer is used; primary per the prereg).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_catalog as C  # noqa: E402
from nsr1 import rm_state  # noqa: E402

INTRO = {"warehouse": "Below is today's record from a warehouse: WMS scanner transactions and the floor team's chat, in time order.",
         "biobank": "Below is today's record from a biobank: the LIMS audit trail, bench notebook entries, emails and chat, in time order.",
         "home": "Below is today's record from a home robot: what its camera saw and what it heard people say, in time order.",
         "ops": "Below is today's record from a cloud platform: deploy bot and database events, alerts and the incident channel, in time order."}
ASK = ("List every ID from the asset list that answers the question. End with one line of the form "
       "'Answer: ID1, ID2, ...', or 'Answer: none' if nothing qualifies.")


def assets_block(log):  # same text as apps/real run_real.assets_block at 707351b
    row = lambda e: e["id"] + (": " + ", ".join(e["aliases"]) if e["aliases"] else "")
    return ("Site asset list (each ID, then other names people use for it; a number can stand for the ID that ends in "
            "it). Answer with these IDs.\nItems:\n" + "\n".join(row(e) for e in log["registry"]["items"]) +
            "\nPlaces:\n" + "\n".join(row(e) for e in log["registry"]["places"]))


def catalog_block(log):
    lines = []
    for a, r, b in C.catalog(log):
        lines.append(f"{a} is part of {b}." if r == "part_of" else f"{a} needs the {b}.")
    return "Site catalog (fixed facts that do not change during the day):\n" + ("\n".join(lines) if lines else "(none)")


def state_block(state):
    return "Current state (where each thing is at the end of the day):\n" + "\n".join(
        f"{x} is in {p}." for x, p in sorted(state.items()))


def prompts(log, reader=None):
    out = []
    for i, q in enumerate(C.questions(log)):
        pid = f"{log['id']}|{q['qtype']}|{q['target']}"
        body = (state_block(rm_state(reader)[0]) if reader is not None else "\n".join(log["text_lines"]))
        lead = (INTRO[log["scenario"]] + " Answer the question about the state at the end of the record."
                if reader is None else "Below is the current state of a site and its fixed facts. Answer the question.")
        p = f"{lead}\n\n{assets_block(log)}\n\n{catalog_block(log)}\n\n{body}\n\nQuestion: {q['q']}\n{ASK}"
        n = ["warehouse", "biobank", "home", "ops"].index(log["scenario"]) * 100 + int(log["id"].rsplit("-", 1)[1])
        out.append({"task": "NSR1", "n": n, "idx": i, "id": pid, "log": log["id"], "qtype": q["qtype"],
                    "target": q["target"], "prompt": p})
    return out


NORMALIZE = False  # sensitivity analysis (NSR1 verdict): "reading_glasses" -> "reading glasses", case-insensitive


REPAIR = False  # frontier amendment 2: restore the break before an 'Answer:' glued to the gateway's previous segment


def resolve(text, log):
    """Last 'Answer:' line -> registry IDs (IDs, aliases, digit suffixes), lenient on markdown; None if unparseable."""
    lines = [x.strip() for x in (text or "").strip().split("\n") if x.strip()]
    ans = [x for x in lines if re.match(r"^\**\s*answer\s*\**\s*:", x, re.I)]
    if not ans and REPAIR and text:
        # stream reassembly joined output segments without a break ("...that list.Answer: S-26-..."); only answers with
        # no Answer line at all are touched
        lines = [x.strip() for x in re.sub(r"(?<=\S)(\**\s*answer\s*\**\s*:)", r"\n\1", text, flags=re.I).split("\n") if x.strip()]
        ans = [x for x in lines if re.match(r"^\**\s*answer\s*\**\s*:", x, re.I)]
    if not ans:
        return None
    t = re.sub(r"\*\*|`", "", ans[-1]).split(":", 1)[1].strip().strip(".")
    if re.fullmatch(r"(none|nothing|no (items|services|specimens|pallets))\.?", t, re.I):
        return []
    names = {}
    for e in log["registry"]["items"]:
        names[e["id"].lower()] = e["id"]
        for a in e["aliases"]:
            names.setdefault(a.lower(), e["id"])
    out = set()
    for part in re.split(r",|;|\band\b", t):
        p = part.strip().strip(".").lower()
        if NORMALIZE:
            p = re.sub(r"\s+", " ", p.replace("_", " ")).strip()
        p = re.sub(r"\s*\(.*?\)\s*", " ", p).strip()
        if p in names:
            out.add(names[p])
        else:
            hit = [i for k, i in names.items() if k and (p.endswith(k) or k.endswith(p)) and len(p) >= 4]
            if len(set(hit)) == 1:
                out.add(hit[0])
    return sorted(out)


def load(split, logs_dir):
    if not logs_dir:
        return C.load_logs(split)
    paths = [os.path.join(logs_dir, f"{sc}_{split}.jsonl") for sc in ("warehouse", "biobank", "home", "ops")]
    return [json.loads(l) for p in paths if os.path.exists(p) for l in open(p)]  # scenarios not generated yet: skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "parse"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--logs", default="")
    ap.add_argument("--reader")
    ap.add_argument("--problems")
    ap.add_argument("--runs")
    ap.add_argument("--forced")
    ap.add_argument("--arm", default="arm3")
    ap.add_argument("--normalize", action="store_true")  # sensitivity: underscores -> spaces before matching
    ap.add_argument("--repair", action="store_true")  # frontier amendment 2
    a = ap.parse_args()
    global NORMALIZE, REPAIR
    NORMALIZE, REPAIR = a.normalize, a.repair
    logs = load(a.split, a.logs)
    if a.cmd == "prep":
        rd = {json.loads(l)["id"]: json.loads(l) for l in open(a.reader)} if a.reader else None
        arm = "arm3b" if rd else "arm3"
        path = os.path.join(HERE, "out", f"nsr1_llm_problems_{a.split}_{arm}.jsonl")
        with open(path, "w") as f:
            for log in logs:
                for p in prompts(log, rd[log["id"]] if rd else None):
                    f.write(json.dumps(p) + "\n")
        print(path)
    else:
        import glob
        byid = {l["id"]: l for l in logs}
        pid_of = {(r["n"], r["idx"]): r["id"] for r in map(json.loads, open(a.problems))}
        runs = {}
        for fp in sorted(glob.glob(a.runs)):
            for r in map(json.loads, open(fp)):
                runs[(r["n"], r["idx"])] = r
        forced = {(r["n"], r["idx"]): r["forced_answer"] for r in map(json.loads, open(a.forced))} if a.forced else {}
        texts = {"strict": {}, "forced": {}}
        hit = 0
        for key, r in runs.items():
            pid = pid_of[key]
            texts["strict"][pid] = None if r["hit_limit"] else r["answer_text"]
            texts["forced"][pid] = forced.get(key) if r["hit_limit"] else r["answer_text"]
            hit += r["hit_limit"]
        for mode, tx in texts.items():
            path = os.path.join(HERE, "out", f"nsr1_{a.split}_{a.arm}_{mode}.jsonl")
            with open(path, "w") as f:
                for log in logs:
                    for q in C.questions(log):
                        pid = f"{log['id']}|{q['qtype']}|{q['target']}"
                        f.write(json.dumps({"log": log["id"], "scenario": log["scenario"], "qtype": q["qtype"],
                                            "target": q["target"], "pred": resolve(tx.get(pid), byid[log["id"]])}) + "\n")
            print(path)
        print(f"{len(runs)} runs, {hit} hit the reasoning limit, {len(forced)} forced answers")


if __name__ == "__main__":
    main()
