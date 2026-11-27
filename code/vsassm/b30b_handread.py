"""B30b hand-reading helper (prereg/B30b_referee_baselines_prereg.md, Verification): prints the cases to be read by hand.

  logs: up to 20 questions per set on which S32 R1 and the published Qwen3-32B run disagree (question, gold, both
        answers, reasoning lengths, the end of the S32 reasoning);
  BABILong: every 128k sample on which the memory and q8T disagree (up to 30) and up to 15 for q17T (question, target,
        memory answer, notepad answer, the notepad lines that mention the question's entity).

python3 -B b30b_handread.py logs|bab [N]
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def jl(p):
    return [json.loads(l) for l in open(p)]


def logs(nmax):
    import b30b_logs as B
    raw = {r["id"]: r for r in jl(os.path.join(HERE, "b30b_out", "logs", "s32_r1.jsonl"))}
    for s in B.SETS:
        new = B.answers(s, *B.gen_file(s, os.path.join(HERE, "b30b_out", "logs", "s32_r1.jsonl")))
        old = B.answers(s, *B.banked(s))
        pruns, _ = B.banked(s)
        disc = [k for k in new if new[k]["ok"] != old[k]["ok"]]
        random.Random(0).shuffle(disc)
        print(f"\n######## {s}: {len(disc)} discordant (S32 right {sum(new[k]['ok'] for k in disc)}, published right "
              f"{sum(old[k]['ok'] for k in disc)}); reading {min(nmax, len(disc))}")
        qs = {qk: (n, i) for qk, lid, sc, n, i, ex in B.questions(s)}
        for k in disc[:nmax]:
            n, i = qs[k]
            r = raw[f"{s}|{n}|{i}"]
            po = pruns[(n, i)]
            print(f"\n--- {k} | Q: {new[k]['q']}")
            print(f"  S32 ok={new[k]['ok']} pred={new[k]['pred']} think={r['think_tokens']} phase={r['phase']}")
            print(f"  pub ok={old[k]['ok']} pred={old[k]['pred']} think={po['think_tokens']} hit={po['hit_limit']}")
            print("  S32 answer:", (r["answer_text"] or r["forced_answer"] or "")[-300:].replace("\n", " | "))
            print("  S32 reasoning end:", r["thinking"][-600:].replace("\n", " | "))


def bab(nmax):
    BA = os.path.join(HERE, "b30b_out", "bab")
    rows = {(r["task"], r["i"]): r for r in jl(os.path.join(BA, "assembled_128k.jsonl"))}
    for tag, cap in (("q8T", nmax), ("q17T", max(1, nmax // 2))):
        p = os.path.join(BA, f"scored_128k_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        sc = [x for x in jl(p) if x["ok"] != x["memory_ok"]]
        print(f"\n######## 128k {tag}: {len(sc)} samples where memory and notepad disagree "
              f"(memory right {sum(x['memory_ok'] for x in sc)}); reading {min(cap, len(sc))}")
        for x in sc[:cap]:
            r = rows[(x["task"], x["i"])]
            words = set(r["question"].lower().rstrip("?").split()) - {"where", "is", "the", "was", "before", "what",
                                                                    "who", "did", "give", "to", "of", "gave", "received"}
            lines = [e for e in r["events"] if words & set(map(str, e[2:]))]
            print(f"\n--- {x['id']} | Q: {r['question']} | target {r['target']} | memory {r['pred']} ({x['memory_ok']}) | "
                  f"notepad {x['answer'][-120:]!r} ({x['ok']}) | {x['n_lines']} lines, forced {x['hit_limit']}")
            for e in lines[-14:]:
                print("    ", e)


def babclass(L="128k"):
    """Classify every notepad error at length L (all arms): the answer names several labels with the target among them;
    a single wrong label; no label. qa3: does the wrong answer equal the place before the FIRST time the thing was in
    the asked place (instead of the most recent, which bAbI's gold uses)? qa5: does it equal an earlier give's answer?"""
    import re
    from b30b_check import Dict
    sys.path.insert(0, os.path.join(HERE, "b30b_frozen"))
    import babilong_metrics_official as OFF
    BA = os.path.join(HERE, "b30b_out", "bab")
    rows = {(r["task"], r["i"]): r for r in jl(os.path.join(BA, f"assembled_{L}.jsonl"))}
    for tag in ("q8T", "q17T", "rdT", "q8N", "q17N", "rdN"):
        p = os.path.join(BA, f"scored_{L}_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        cnt = {}
        for x in jl(p):
            if x["ok"]:
                continue
            r = rows[(x["task"], x["i"])]
            out = x["answer"].lower().split(".")[0]
            labs = [l.lower() for l in OFF.TASK_LABELS[x["task"]] if not re.search(r"\b" + re.escape(l.lower()) + r"\b", r["question"].lower())]
            found = {l for l in labs if re.search(r"\b" + re.escape(l) + r"\b", out)}
            tgt = str(r["target"]).lower()
            kind = "no label" if not found else ("several, target among them" if tgt in found and len(found) > 1 else "single wrong label" if len(found) == 1 else "several, target not among them")
            extra = ""
            if x["task"] == "qa3" and len(found) == 1:
                d = Dict()
                for e in r["events"]:
                    d.apply(e[1:])
                m = re.match(r"where was (?:the )?(.+?) before (?:the )?(.+)$", r["question"].strip().rstrip("?").lower())
                h = d.hist.get(m.group(1), [])
                first = next((i for i, q in enumerate(h) if q == m.group(2)), None)
                if first:
                    extra = " / equals the place before the FIRST visit" if h[first - 1] in found else " / other"
            if x["task"] == "qa5" and len(found) == 1:
                d = Dict()
                for e in r["events"]:
                    d.apply(e[1:])
                vals = {v for g in d.gives for v in g}
                extra = " / names a participant of some give" if found <= vals else " / other"
            k = f"{x['task']} {kind}{extra}"
            cnt[k] = cnt.get(k, 0) + 1
        print(f"\n{L} {tag}: errors by kind")
        for k, v in sorted(cnt.items()):
            print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if sys.argv[1] == "logs":
        logs(n)
    elif sys.argv[1] == "babclass":
        for L in ("128k", "512k", "1M"):
            if os.path.exists(os.path.join(HERE, "b30b_out", "bab", f"assembled_{L}.jsonl")):
                babclass(L)
    else:
        bab(30 if len(sys.argv) <= 2 else n)
