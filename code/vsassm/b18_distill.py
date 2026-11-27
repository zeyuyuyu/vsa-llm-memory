"""B18 amendment 2: training examples for the BABILong student (training split + bAbI-original qa11-qa13 only).

python3 b18_distill.py --model <Qwen3-8B> --out b18_data/train.jsonl
Each example is (compact prompt, target events). bAbI-fact sentences (every word in the 0k training vocabulary) get the
teacher's events (prompt F, greedy) after the grounding + cue checks; other sentences get no events (a sample of them).
"""
import argparse
import glob
import json
import os
import random
import re

import b18

COMPACT = "Context: {context}\nSentence: {sentence}\nEvents:"  # the student's prompt (fine-tuned, so no instructions)


def vocab_from_0k(babi_root=None):
    """Words of the 0k training stories (+ the bAbI-original qa11-qa13 training stories, B18b: without them the
    pronoun / "afterwards" sentences of those tasks were labelled as book text)."""
    v = set()
    for f in glob.glob(f"{b18.ROOT['train']}/data/qa*/0k.json"):
        for r in json.load(open(f)):
            v.update(re.findall(r"[a-z]+", r["input"].lower()))
    for f in glob.glob(f"{babi_root}/tasks_1-20_v1-2/en-10k/qa1[123]_*_train.txt") if babi_root else []:
        v.update(w for w in re.findall(r"[a-z]+", open(f).read().lower()))
    return v


def is_fact(sentence, vocab):
    ws = re.findall(r"[a-z]+", sentence.lower())
    return bool(ws) and all(w in vocab for w in ws)


def babi_original(task_num, n, root):
    """bAbI v1.2 en-10k training stories for qa11-qa13 -> [{"sents": [...]}] (stories cut at each question)."""
    f = glob.glob(f"{root}/tasks_1-20_v1-2/en-10k/qa{task_num}_*_train.txt")[0]
    out, cur = [], []
    for line in open(f):
        num, text = line.strip().split(" ", 1)
        if num == "1":
            cur = []
        if "\t" in text:
            out.append({"sents": list(cur)})
        else:
            cur.append(text)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--babi_root", default=os.environ.get("BENCH_ROOT", ".") + "/babi")
    ap.add_argument("--neg_per_fact", type=float, default=3.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)  # facts before this index are labelled elsewhere
    ap.add_argument("--tasks", default="qa1,qa2,qa3,qa5,qa6,qa7,qa8,qa9,qa10")
    ap.add_argument("--babi", type=int, default=1)  # add bAbI-original qa11-qa13 stories
    ap.add_argument("--vocab_babi", type=int, default=0)  # B18b: include bAbI-original qa11-13 words in the vocabulary
    ap.add_argument("--negs_only", action="store_true")  # write only the book-sentence examples (no model)
    args = ap.parse_args()
    rng = random.Random(18)
    vocab = vocab_from_0k(args.babi_root if args.vocab_babi else None)
    stories = []
    for task in args.tasks.split(","):
        for length, n in (("0k", 200), ("4k", 100), ("16k", 30)):
            try:
                stories += [{"sents": s["sents"]} for s in b18.load("train", length, task, n)]
            except FileNotFoundError:
                print("missing", task, length)
    for t in ([11, 12, 13] if args.babi else []):
        stories += babi_original(t, 300, args.babi_root)
    print(len(stories), "stories", flush=True)
    if args.negs_only:
        with open(args.out, "w") as f:
            for c, s, _ in negs:
                f.write(json.dumps({"prompt": COMPACT.format(context=c, sentence=s), "output": '{"events": []}'}) + "\n")
        print("negatives", len(negs))
        return
    import b14
    reader = b14.Reader(args.model)  # loaded after the lists so a bad shard fails fast
    # teacher only on fact sentences; context = the 3 previous sentences, as at test time
    jobs = []
    for s in stories:
        for i, sent in enumerate(s["sents"]):
            ctx = " ".join(s["sents"][max(0, i - 3):i]) or "(none)"
            if is_fact(sent, vocab):
                jobs.append((ctx, sent, True))
            elif rng.random() < 0.05:
                jobs.append((ctx, sent, False))
    facts = [j for j in jobs if j[2]]
    negs = [j for j in jobs if not j[2]]
    rng.shuffle(negs)
    negs = negs[:int(args.neg_per_fact * len(facts))]
    print(len(facts), "fact sentences,", len(negs), "book sentences", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    facts.sort(key=lambda j: len(j[0]) + len(j[1]))  # similar lengths per batch: less padding, steady memory
    if args.nshards:  # PACE array: shard i labels facts[start + i*size : ...] only, no book sentences
        rest = facts[args.start:]
        size = (len(rest) + args.nshards - 1) // args.nshards
        facts, negs = rest[args.shard * size:(args.shard + 1) * size], []
        print(f"shard {args.shard}/{args.nshards}: {len(facts)} facts", flush=True)
    done = sum(1 for _ in open(args.out)) if os.path.exists(args.out) else 0  # resume: facts are written first
    kept = dropped = 0
    with open(args.out, "a") as f:
        for c0 in range(min(done, len(facts)), len(facts), 2048):
            chunk = facts[c0:c0 + 2048]
            outs = reader([b18.PROMPT_F.format(context=c, sentence=s) for c, s, _ in chunk], bs=64)
            for (c, s, _), (txt, _) in zip(chunk, outs):
                evs = [e for e in b18.parse(txt) if b18.grounded(e, s)]
                dropped += len(b18.parse(txt)) - len(evs)
                kept += len(evs)
                f.write(json.dumps({"prompt": COMPACT.format(context=c, sentence=s),
                                    "output": json.dumps({"events": evs})}) + "\n")
            f.flush()
            print(f"facts {c0 + len(chunk)}/{len(facts)}", flush=True)
        if done <= len(facts):
            for c, s, _ in negs:
                f.write(json.dumps({"prompt": COMPACT.format(context=c, sentence=s), "output": '{"events": []}'}) + "\n")
    print(f"wrote {len(facts) + len(negs)} examples; teacher events kept {kept}, dropped by checks {dropped}")


if __name__ == "__main__":
    main()
