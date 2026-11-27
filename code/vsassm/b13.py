"""B13 writer (prereg/B13_prereg.md + amendment 1): a role for every mention, events decoded per sentence.

Roles: x / y (a move: object, place), sa / sb (a swap of contents), q (the question's entity), nx / ny (an object or
place named in the sentence but not part of its events). Events are decoded per sentence with no clause-split rule:
swap mentions pair up in text order; each x goes to the only y, or to the y with the same index when the counts match,
or else to the nearest y after it (before it if none follows). Pronouns: "it" / "them" = the object mentioned most
recently, "there" = the place mentioned most recently (a resolved pronoun counts as a mention of its referent).

python3 b13.py train   --model <Qwen3-1.7B> --domain locations --data b13_data/loc_0.jsonl,b13_data/loc_1.jsonl --seed 0
python3 b13.py regress --model <Qwen3-1.7B> --domain locations --writer b13_models/locations.pt --out b13_reg_s0
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from b9 import Writer, decode_words, mention_vecs, run_episode, sentence_batch

from b13_rules import OBJ_PRO, PLACE_PRO, PLACE_ROLES, ROLES, note, pair_events, resolve  # noqa: F401

TASK = {"locations": "T2", "containers": "T3"}


class Writer13(Writer):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.role = nn.Linear(self.role.in_features, len(ROLES))  # one score per role, softmax over roles per mention


def by_sentence(ep):
    """Merge clause spans that share a numbered line into one sentence span (B12 stories)."""
    out = []
    for s in ep["sents"]:
        line = ep["text"].rfind("\n", 0, s["span"][0])
        if out and out[-1]["line"] == line and "question" not in (s["op"], out[-1]["op"]):
            o = out[-1]
            o["span"] = (o["span"][0], s["span"][1])
            o["mentions"] = o["mentions"] + [tuple(m) for m in s["mentions"]]
        else:
            out.append({"span": tuple(s["span"]), "mentions": [tuple(m) for m in s["mentions"]], "op": s["op"],
                        "line": line})
    return {**ep, "sents": out}


# ---------------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def decode_story(w, text, spans, kinds):
    """spans: (start, end) per sentence; kinds: "story" / "question". Returns per-span lists of events (dicts) and,
    for a question span, {"q": name}; plus every mention with its role, for inspection."""
    h, offs = w.encode([{"text": text}])
    last = {"obj": None, "place": None}
    out = []
    for (a, b), kind in zip(spans, kinds):
        t0, t1 = Writer.tok_range(offs[0], a, b)
        H = h[0, t0:t1]
        ments = decode_words(text, offs[0], t0, t1, w.bio(H).argmax(-1).tolist())
        if not ments:
            out.append({"events": [], "mentions": []})
            continue
        mv, _ = mention_vecs({"H": H, "t0": t0}, [(x, y) for x, y, _ in ments])
        probs = w.role(mv).float().softmax(-1)
        if kind == "question":
            j = int(probs[:, ROLES.index("q")].argmax())
            out.append({"q": ments[j][2], "q_prob": float(probs[j, ROLES.index("q")]), "mentions": [m[2] for m in ments]})
            continue
        probs[:, ROLES.index("q")] = 0
        res, shown = [], []
        for (ta, tb, name), p in zip(ments, probs):
            role = ROLES[int(p.argmax())]
            ref = resolve(name, role, last)
            res.append((ta, ref, role, float(p.max())))
            shown.append({"text": name, "role": role, "ref": ref, "p": round(float(p.max()), 3)})
            note(ref, role, last)
        evs = [{"op": op, "x": x, "y": y, "confidence": round(c, 3)} for _, op, x, y, c in pair_events(res)]
        out.append({"events": evs, "mentions": shown})
    return out


# ---------------------------------------------------------------------------------------------- training
def training_sources(domain, seed, data):
    distilled = [json.loads(l) for p in data.split(",") for l in open(p)]
    t = TASK[domain]
    b11 = [json.loads(l) for p in (f"b11c_data/{t}_train0.jsonl", f"b11c_data/{t}_train1.jsonl") for l in open(p)]
    if domain == "locations":
        from b12_data import train_set as b12_train
        other = [by_sentence(e) for e in b12_train(4000, seed=1501 + seed)]
    else:
        from b9_data import train_set
        other = train_set("T3", 20000, seed=93, names="train2")
    print(f"{domain}: {len(distilled)} distilled, {len(other)} B12/template, {len(b11)} B11c episodes", flush=True)
    return [(distilled, 4), (other, 2), (b11, 2)]


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sources = training_sources(args.domain, args.seed, args.data)
    w = Writer13(args.model).cuda()
    heads = [p for n, p in w.named_parameters() if p.requires_grad and not n.startswith("lm.")]
    lora = [p for n, p in w.named_parameters() if p.requires_grad and n.startswith("lm.")]
    opt = torch.optim.AdamW([{"params": heads, "lr": 1e-3}, {"params": lora, "lr": 2e-4}], weight_decay=0.0)
    w.train()
    t_start = time.time()
    for step in range(args.steps):
        batch = [e for src, k in sources for e in rng.sample(src, k)]
        recs = sentence_batch(w, batch, True)
        l_bio = F.cross_entropy(torch.cat([w.bio(r["H"]) for r in recs]),
                                torch.tensor([x for r in recs for x in r["bio"]], device="cuda"))
        logits, tgt = [], []
        for r in recs:
            ms = r["s"]["mentions"]
            if not ms:
                continue
            mv, _ = mention_vecs(r, [Writer.tok_range(r["offs"], m[0], m[1]) for m in ms])
            logits.append(w.role(mv))
            tgt += [ROLES.index(m[2]) for m in ms]
        l_role = F.cross_entropy(torch.cat(logits), torch.tensor(tgt, device="cuda"))
        loss = l_bio + l_role
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads + lora, 1.0)
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.item():.4f} bio {l_bio.item():.4f} role {l_role.item():.4f} "
                  f"t={time.time() - t_start:.0f}s", flush=True)
    keep = {n: p.detach().cpu() for n, p in w.named_parameters() if p.requires_grad}
    os.makedirs(args.out, exist_ok=True)
    torch.save({"params": keep, "roles": ROLES}, f"{args.out}/{args.domain}_s{args.seed}.pt")


def load(model, ckpt):
    w = Writer13(model).cuda().eval()
    w.load_state_dict(torch.load(ckpt, weights_only=True)["params"], strict=False)
    return w


# ---------------------------------------------------------------------------------------------- regression tests
def regress_sets(domain, long_path):
    from b9_data import test_sets
    t = TASK[domain]
    sets = {f"tmpl-{k}": v for k, v in test_sets(t, 9302 if t == "T2" else 9301, fresh=True).items()}
    sets["b11b"] = [json.loads(l) for l in open(f"b11b_data/{t}.jsonl")]
    if domain == "locations":
        from b12_data import test_set as b12_test
        sets["b12c"] = [by_sentence(e) for e in b12_test(seed=1403)]
        if long_path and os.path.exists(long_path):
            sets["long"] = [json.loads(l) for l in open(long_path)]
    return sets


def gold_answer(task, e):
    if "final" in e:  # model-written long stories: the queried name is the question's labelled mention
        q = [m for m in e["sents"][-1]["mentions"] if m[2] == "q"][0][3]
        return e["final"][q]
    from score_b9 import gold
    return gold(task, e)


@torch.no_grad()
def regress(args):
    w = load(args.model, args.writer)
    task = TASK[args.domain]
    os.makedirs(args.out, exist_ok=True)
    summary = json.load(open(f"{args.out}/summary.json")) if args.only and os.path.exists(f"{args.out}/summary.json") else {}
    for k in [k for k in summary if k.split("/")[0] in args.only.split(",")]:  # replace, never merge, a rerun set
        del summary[k]
    summary = {k: {int(n): v for n, v in row.items()} for k, row in summary.items()}
    for name, eps in regress_sets(args.domain, args.long).items():
        if args.only and name not in args.only.split(","):
            continue
        if args.limit:  # smoke test only
            eps = eps[::max(1, len(eps) // args.limit)][:args.limit]
        acc = defaultdict(lambda: [0, 0])
        with open(f"{args.out}/{name}.jsonl", "w") as f:
            for e in eps:
                spans = [tuple(s["span"]) for s in e["sents"]]
                kinds = ["question" if s["op"] == "question" else "story" for s in e["sents"]]
                dec = decode_story(w, e["text"], spans, kinds)
                evs = [(ev["op"], ev["x"], ev["y"]) for d in dec[:-1] for ev in d["events"]]
                q = dec[-1].get("q")
                ans, margin, hops, _, _ = run_episode(task, evs, q)
                g = gold_answer(task, e)
                ok = ans == g
                n = e.get("n", 0)
                split = e.get("split", name)
                acc[(split, n)][0] += ok
                acc[(split, n)][1] += 1
                f.write(json.dumps({"split": split, "n": n, "idx": e.get("idx"), "pred": ans, "gold": g, "ok": ok,
                                    "query": q, "margin": margin, "hops": hops}) + "\n")
        for (split, n), (k, m) in sorted(acc.items()):
            summary.setdefault(f"{name}/{split}", {})[n] = round(k / m, 3)
        for key, row in summary.items():
            if key.startswith(name + "/"):
                print(f"{key:24s} " + " ".join(f"N={n}:{v:.3f}" for n, v in sorted(row.items())), flush=True)
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "regress"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--domain", choices=["locations", "containers"], required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="b13_models")
    ap.add_argument("--writer", default="")
    ap.add_argument("--long", default="b13_data/long_test.jsonl")
    ap.add_argument("--limit", type=int, default=0)  # regress: episodes per set (smoke test)
    ap.add_argument("--only", default="")  # regress: comma-separated set names (e.g. long), merged into summary.json
    a = ap.parse_args()
    train(a) if a.cmd == "train" else regress(a)
