"""B9 (prereg/B9_prereg.md): open-vocabulary writer + readable memory.

The writer (Qwen3-1.7B + LoRA) tags entity mentions (B/I/O per token) and, per role, picks one of the sentence's
mentions; names are strings. The memory allocates a code to each new name at its first use (random, orthogonalised
against the codes already allocated in that episode), then runs B7b's algebra on a 256 x 256 state.

python3 b9.py train --model <snapshot> --task T2 --out b9_models
python3 b9.py eval  --model <snapshot> --task T2 --models b9_models --out b9_out
"""
import argparse
import bisect
import hashlib
import json
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from b9_data import test_sets, train_set

LAYERS = (14, 21)
ROLES = ("x", "y", "sa", "sb", "q")
DH = 256


def norm_name(s):
    s = s.strip().strip(".,?!;:\"'").strip()
    s = re.sub(r"(['\u2019]s|['\u2019])$", "", s).strip()  # B11b: drop a possessive ("scullery's" -> "scullery")
    return s[4:] if s.lower().startswith("the ") else s


class Writer(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "right"
        from peft import LoraConfig, get_peft_model
        lm = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(lm, cfg)
        d = 2048 * len(LAYERS)
        self.ln = nn.LayerNorm(d)
        self.bio = nn.Linear(d, 3)  # 0 outside, 1 begin, 2 inside
        self.role = nn.Linear(2 * d, len(ROLES))  # [mention vector, sentence vector] -> one score per role
        self.op = nn.Linear(d, 2)  # write, swap

    def encode(self, eps):
        """Token states per episode, plus each token's character offsets."""
        enc = self.tok([e["text"] for e in eps], return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        lens = enc["attention_mask"].sum(1).tolist()
        dev = self.op.weight.device
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.lm(**enc.to(dev), output_hidden_states=True).hidden_states
        h = self.ln(torch.cat([hs[l] for l in LAYERS], -1).float())
        return h, [offs[b, :lens[b]].tolist() for b in range(len(eps))]

    @staticmethod
    def tok_range(offs, a, b):
        starts = [o[0] for o in offs]
        t0 = bisect.bisect_left(starts, a - 1)
        while t0 < len(offs) and offs[t0][1] <= a:
            t0 += 1
        t1 = bisect.bisect_right(starts, b)
        return t0, max(t1, t0 + 1)


def bio_labels(offs, t0, t1, mentions):
    lab = [0] * (t1 - t0)
    for m0, m1, _, _ in mentions:
        first = True
        for t in range(t0, t1):
            s, e = offs[t]
            if e > m0 and s <= m1 and e > s:
                lab[t - t0] = 1 if first else 2
                first = False
    return lab


def decode_words(text, offs, t0, t1, tags):
    """B9b word-level decoding: a word is in a mention if any of its tokens is tagged; consecutive words join."""
    import re
    a, b = offs[t0][0], offs[t1 - 1][1]
    res, cur = [], None
    for m in re.finditer(r"\S+", text[a:b]):
        ws, we = a + m.start(), a + m.end()
        wt = [tags[t - t0] for t in range(t0, t1) if offs[t][1] > ws and offs[t][0] < we and offs[t][1] > offs[t][0]]
        hit = [x for x in wt if x > 0]
        pronoun = m.group(0).strip(".,;:?!").lower() in ("it", "them", "there")
        if hit:
            # B12c: pronouns ("it", "them", "there") always stand alone; other adjacent tagged words merge (B9b)
            if cur and (pronoun or cur[2]):
                res.append(cur[:2])
                cur = None
            cur = [cur[0], we, pronoun] if cur else [ws, we, pronoun]
        elif cur:
            res.append(cur[:2])
            cur = None
    if cur:
        res.append(cur[:2])
    out = []
    for ws, we in res:
        name = norm_name(text[ws:we])
        ta = next(t for t in range(t0, t1) if offs[t][1] > ws)
        tb = max(t for t in range(t0, t1) if offs[t][0] < we)
        if name:
            out.append((ta, tb + 1, name))
    return out


def decode_mentions(text, offs, t0, t1, tags):
    out, cur = [], None
    for t in range(t0, t1):
        tg = tags[t - t0]
        if tg == 1 or (tg == 2 and cur is None):
            if cur:
                out.append(cur)
            cur = [t, t]
        elif tg == 2:
            cur[1] = t
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    res = []
    for a, b in out:
        name = norm_name(text[offs[a][0]:offs[b][1]])
        if name:
            res.append((a, b + 1, name))
    return res


def sentence_batch(w, eps, train):
    """Run the LM once; return per-sentence records with token states, labels and (train) gold mention ranges."""
    h, offs = w.encode(eps)
    recs = []
    for b, e in enumerate(eps):
        for si, s in enumerate(e["sents"]):
            t0, t1 = Writer.tok_range(offs[b], *s["span"])
            recs.append({"b": b, "e": e, "si": si, "s": s, "t0": t0, "t1": t1, "offs": offs[b],
                         "H": h[b, t0:t1], "bio": bio_labels(offs[b], t0, t1, s["mentions"])})
    return recs


def mention_vecs(rec, spans):
    H, t0 = rec["H"], rec["t0"]
    sv = H.mean(0)
    mv = torch.stack([H[a - t0:b - t0].mean(0) for a, b in spans])
    return torch.cat([mv, sv.expand_as(mv)], -1), sv


def train(args):
    torch.manual_seed(args.seed)
    if args.b12:  # B12: stories with pronouns and two-event sentences (objects in places only)
        from b12_data import train_set as b12_train
        eps = b12_train(4000, seed=1501 + args.seed)
    else:
        eps = train_set(args.task, 20000, seed=91 if args.task == "T2" else 93, names=args.names)
    extra = []
    for path in args.mix.split(",") if args.mix else []:  # B11c: rewritten episodes, half of every batch
        extra += [json.loads(l) for l in open(path)]
    if extra:
        print(f"mixing {len(extra)} rewritten episodes 50/50", flush=True)
    w = Writer(args.model).cuda()
    heads = [p for n, p in w.named_parameters() if p.requires_grad and not n.startswith("lm.")]
    lora = [p for n, p in w.named_parameters() if p.requires_grad and n.startswith("lm.")]
    opt = torch.optim.AdamW([{"params": heads, "lr": 1e-3}, {"params": lora, "lr": 2e-4}], weight_decay=0.0)
    rng = random.Random(args.seed)
    w.train()
    t_start = time.time()
    for step in range(args.steps):
        batch = rng.sample(eps, 4) + rng.sample(extra, 4) if extra else rng.sample(eps, 8)
        recs = sentence_batch(w, batch, True)
        l_bio = F.cross_entropy(torch.cat([w.bio(r["H"]) for r in recs]),
                                torch.tensor([x for r in recs for x in r["bio"]], device="cuda"))
        l_role, l_op, n_role = 0.0, 0.0, 0
        for r in recs:
            spans = [Writer.tok_range(r["offs"], m0, m1) for m0, m1, _, _ in r["s"]["mentions"]]
            mv, sv = mention_vecs(r, spans)
            scores = w.role(mv)  # (mentions, roles)
            for ri, role in enumerate(ROLES):
                tgt = [i for i, m in enumerate(r["s"]["mentions"]) if m[2] == role]
                if tgt:
                    lp = scores[:, ri].log_softmax(0)
                    l_role = l_role - torch.logsumexp(lp[tgt], 0)  # any mention with that role (q can repeat)
                    n_role += 1
            if args.task == "T2" and r["s"]["op"] != "question":
                l_op = l_op + F.cross_entropy(w.op(sv)[None], torch.tensor([int(r["s"]["op"] == "swap")], device="cuda"))
        loss = l_bio + l_role / max(n_role, 1) + (l_op / len(recs) if args.task == "T2" else 0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads + lora, 1.0)
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.item():.4f} bio {l_bio.item():.4f} t={time.time() - t_start:.0f}s", flush=True)
    keep = {n: p.detach().cpu() for n, p in w.named_parameters() if p.requires_grad}
    os.makedirs(args.out, exist_ok=True)
    torch.save({"params": keep}, f"{args.out}/{args.task}{args.tag}{'_s%d' % args.seed if args.seed else ''}.pt")


class Codebook:
    """Codes allocated on first use: a seeded random vector orthogonalised against earlier codes (one episode)."""

    def __init__(self, space):
        self.space, self.names, self.codes = space, [], torch.zeros(0, DH)

    def get(self, name, create=True):
        if name in self.names:
            return self.codes[self.names.index(name)]
        if not create:
            return None
        seed = int(hashlib.sha1(f"{self.space}:{name}".encode()).hexdigest()[:8], 16)
        v = torch.randn(DH, generator=torch.Generator().manual_seed(seed))
        if len(self.names):
            v = v - self.codes.T @ (self.codes @ v)
        v = F.normalize(v, dim=0)
        self.names.append(name)
        self.codes = torch.cat([self.codes, v[None]])
        return v

    def cleanup(self, r):
        if not len(self.names):
            return None, 0.0
        sims = F.normalize(r, dim=0) @ self.codes.T
        if len(self.names) == 1:
            return self.names[0], float(sims[0])
        top = sims.topk(2)
        return self.names[int(top.indices[0])], float(top.values[0] - top.values[1])


def run_episode(task, events, query):
    """events: per action (op, name1, name2). Returns (memory answer, margin, hops, memory, dict answer)."""
    K, V = Codebook("key"), Codebook("val")
    S = torch.zeros(DH, DH)
    d = {}
    for op, a, b in events:
        if op == "swap":
            if a is None or b is None or a == b:
                continue
            wv = F.normalize(V.get(a) - V.get(b), dim=0)
            S = S - 2 * torch.outer(wv, wv @ S)
            d = {k: (b if v == a else a if v == b else v) for k, v in d.items()}
        else:
            if a is None or b is None:
                continue
            k, v = K.get(a), V.get(b)
            S = S + torch.outer(v - S @ k, k)
            d[a] = b
    if task == "T2":
        k = K.get(query, create=False) if query else None
        ans, margin = V.cleanup(S @ k) if k is not None else (None, 0.0)
        return ans, margin, None, (S, K, V), d.get(query)
    hops, cur, margins = [], query, []
    for _ in range(40):
        k = K.get(cur, create=False) if cur else None
        if k is None:
            break
        r = S @ k
        if r.norm() < 0.5:
            break
        cur, m = V.cleanup(r)
        hops.append(cur)
        margins.append(m)
    dc, seen = query, 0
    while dc in d and seen < 40:
        dc, seen = d[dc], seen + 1
    return (hops[-1] if hops else None), (min(margins) if margins else 0.0), hops, (S, K, V), (dc if seen else None)


@torch.no_grad()
def evaluate(args):
    w = Writer(args.model).cuda().eval()
    w.load_state_dict(torch.load(f"{args.models}/{args.task}{args.tag}{'_s%d' % args.seed if args.seed else ''}.pt",
                                 weights_only=True)["params"], strict=False)
    os.makedirs(args.out, exist_ok=True)
    if args.episodes:  # B11: episodes from a file (model-rewritten sentences)
        sets = {"para": [json.loads(l) for l in open(args.episodes)]}
    else:
        sets = test_sets(args.task, 9302 if args.task == "T2" else 9301, fresh=args.word_level)
    for name, eps in sets.items():
        rows, cnt = [], {"tag_tp": 0, "tag_fp": 0, "tag_fn": 0, "role_ok": 0, "role_n": 0}
        for s0 in range(0, len(eps), 4):
            chunk = eps[s0:s0 + 4]
            recs = sentence_batch(w, chunk, False)
            per_ep = {}
            for r in recs:
                tags = w.bio(r["H"]).argmax(-1).tolist()
                dec = decode_words if args.word_level else decode_mentions
                ments = dec(r["e"]["text"], r["offs"], r["t0"], r["t1"], tags)
                gold = {m[3] for m in r["s"]["mentions"]}
                pred_names = {m[2] for m in ments}
                cnt["tag_tp"] += len(gold & pred_names)
                cnt["tag_fp"] += len(pred_names - gold)
                cnt["tag_fn"] += len(gold - pred_names)
                pick = {}
                if ments:
                    mv, sv = mention_vecs(r, [(a, b) for a, b, _ in ments])
                    sc = w.role(mv)
                    for ri, role in enumerate(ROLES):
                        order = sc[:, ri].argsort(descending=True).tolist()
                        pick[role] = [ments[i][2] for i in order]
                    op = "swap" if args.task == "T2" and r["s"]["op"] != "question" and w.op(sv).argmax() == 1 else "write"
                else:
                    op = "write"
                first = lambda role: pick[role][0] if role in pick else None
                if r["s"]["op"] == "question":
                    ev = ("question", first("q"), None)
                elif op == "swap":
                    a = first("sa")
                    b = next((x for x in pick.get("sb", []) if x != a), None)
                    ev = ("swap", a, b)
                else:
                    ev = ("write", first("x"), first("y"))
                for role in ("x", "y", "sa", "sb", "q"):  # role accuracy against the generator's labels
                    g = [m[3] for m in r["s"]["mentions"] if m[2] == role]
                    if g:
                        cnt["role_n"] += 1
                        cnt["role_ok"] += int(first(role) in g)
                per_ep.setdefault(r["b"], []).append(ev)
            for b, e in enumerate(chunk):
                evs = per_ep[b]
                q = evs[-1][1]
                ans, margin, hops, (S, K, V), dans = run_episode(args.task, evs[:-1], q)
                row = {"split": name, "task": args.task, "n": e["n"], "idx": e["idx"], "mem": ans, "dict": dans,
                       "margin": margin, "hops": hops, "query_read": q}
                if args.task == "T2":  # readability: every placed object's belief, read with its own key
                    final = e["states"][-1]
                    ok = tot = 0
                    for oi, pl in enumerate(final):
                        if pl < 0:
                            continue
                        tot += 1
                        k = K.get(e["ents"][oi], create=False)
                        if k is not None:
                            ok += int(V.cleanup(S @ k)[0] == e["places"][pl])
                    row["read_ok"], row["read_total"] = ok, tot
                rows.append(row)
        with open(f"{args.out}/{args.task}_{name}.jsonl", "w") as f:
            for r, e in zip(rows, eps):
                f.write(json.dumps({**r, "acts": e["acts"], "query": e["query"], "ents": e["ents"],
                                    "places": e["places"]}) + "\n")
        p = cnt["tag_tp"] / max(1, cnt["tag_tp"] + cnt["tag_fp"])
        rc = cnt["tag_tp"] / max(1, cnt["tag_tp"] + cnt["tag_fn"])
        stats = {"tag_f1": round(2 * p * rc / max(1e-9, p + rc), 4), "role_acc": round(cnt["role_ok"] / max(1, cnt["role_n"]), 4)}
        json.dump(stats, open(f"{args.out}/{args.task}_{name}_writer.json", "w"))
        print(args.task, name, stats, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="b9_models")
    ap.add_argument("--models", default="b9_models")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--names", default="train")  # B9b: train2 (larger T3' lexicon)
    ap.add_argument("--word_level", action="store_true")
    ap.add_argument("--seed", type=int, default=0)  # replication seed (init + batch order; data fixed)
    ap.add_argument("--episodes", default="")  # B11: evaluate on episodes stored in a JSONL file
    ap.add_argument("--mix", default="")  # B11c: comma-separated JSONL files of rewritten training episodes
    ap.add_argument("--tag", default="")  # B11c: suffix for the saved / loaded writer
    ap.add_argument("--b12", action="store_true")  # B12: train on pronoun / two-event stories  # B9b: word-level decoding + the fresh test pool
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
