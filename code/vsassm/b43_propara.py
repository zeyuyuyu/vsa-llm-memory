"""B43 part A (prereg/B43_natural_text_and_chunked_baseline_prereg.md): ProPara (Mishra et al. NAACL 2018), natural
crowd-written process paragraphs; the state of each participant before and after every sentence is discrete: "-" (does
not exist), "?" (exists, location unknown) or a location phrase from the paragraph.

GPU (Qwen3-8B, HF generate):
  python3 b43_propara.py gen  ARMS SPLIT OUTDIR --model DIR         ARMS: comma-separated of direct_think,
                                                                    direct_nothink, extract -> OUTDIR/raw_<arm>.jsonl
CPU:
  python3 b43_propara.py assemble SPLIT OUTDIR                      grids of every arm -> OUTDIR/<arm>.tsv (official
                                                                    action files) + OUTDIR/<arm>_grids.json
  python3 b43_propara.py score SPLIT OUTDIR                         official metric, cell accuracy, extraction metrics,
                                                                    paired bootstrap over paragraphs

Arms (same model, same worked example from the training split):
  direct_think    the whole paragraph and the participant list; the model writes the full state grid as JSON; thinking
                  on (chain of thought), Qwen3's recommended thinking sampling (T 0.6, top-p 0.95, top-k 20), seed
                  fixed per paragraph, reasoning budget 8,192 tokens then budget-forced (B30b's CLOSE phrase), answer
                  up to 2,048 tokens.
  direct_nothink  the same prompt, thinking off, greedy, up to 2,048 tokens.
  extract         the paper's route: the model reads ONE sentence at a time (previous sentences as context, the
                  participant list) and writes state-update events (create / destroy / move); thinking off, greedy,
                  up to 400 tokens (the paper's readers). The events are written into
                    store_vsa   a Fourier HRR key-value memory (delta-rule overwrite, clean-up read), and
                    store_dict  a Python dict,
                  by the same read rules (below); the grid is read back from the memory after every sentence.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PROPARA_DIR", "/path/to/HOME/bench_dl/propara_dl")
LB = os.environ.get("PROPARA_LB", "/path/to/HOME/bench_dl/propara_lb/propara")


def load(split):
    return [json.loads(l) for l in open(f"{DATA}/grids.v1.{split}.json")]


def gold_actions(r):
    """(participant, step) -> (action, before, after), as the official answers.tsv (checked identical: 1,674 rows)."""
    out = {}
    for P, st in zip(r["participants"], r["states"]):
        for t in range(1, len(st)):
            out[(P, t)] = (action(st[t - 1], st[t]), st[t - 1], st[t])
    return out


def action(b, a):
    if b == "-" and a != "-":
        return "CREATE"
    if b != "-" and a == "-":
        return "DESTROY"
    if b != a:
        return "MOVE"
    return "NONE"


# ------------------------------------------------------------------ worked example (training split, fixed rule)
def example_para():
    """First training paragraph (file order) with at most 6 sentences and 4 participants that has a CREATE, a DESTROY
    and a MOVE with a named destination."""
    for r in load("train"):
        if len(r["sentence_texts"]) > 6 or len(r["participants"]) > 4:
            continue
        acts = gold_actions(r).values()
        kinds = {a for a, _, _ in acts}
        if {"CREATE", "DESTROY", "MOVE"} <= kinds and any(a == "MOVE" and af not in "?-" for a, _, af in acts):
            return r
    raise RuntimeError("no example")


def gold_events(r):
    """Per sentence, the events the extraction prompt asks for, from the gold grid (used only for the worked example)."""
    ev = [[] for _ in r["sentence_texts"]]
    for P, st in zip(r["participants"], r["states"]):
        name = P.split(";")[0].strip()
        for t in range(1, len(st)):
            a = action(st[t - 1], st[t])
            if a == "CREATE":
                ev[t - 1].append(["create", name, st[t]])
            elif a == "DESTROY":
                ev[t - 1].append(["destroy", name])
            elif a == "MOVE":
                ev[t - 1].append(["move", name, st[t - 1], st[t]])
    return ev


def para_text(r):
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(r["sentence_texts"]))


def parts_text(r):
    return ", ".join(f'"{p}"' for p in r["participants"])


STATE_RULES = """States:
- "-" if the participant does not exist at that time (it has not been created yet, or it has been used up, destroyed or converted into something else);
- "?" if it exists but its location is not stated and cannot be inferred from the paragraph;
- otherwise its location: a short phrase copied from the paragraph (for example "soil", "leaf", "blood stream")."""

DIRECT = """Track the participants of a process described in a paragraph. For each participant give its state before the first sentence (state 0) and after each sentence (states 1 to N, where N is the number of sentences), so each participant has N + 1 states.

{rules}
A participant that is created in a sentence is "-" in the states before that sentence. A participant that is used up, destroyed or converted in a sentence is "-" from that sentence on.

Example
Paragraph:
{ex_para}
Participants: {ex_parts}
Answer: {ex_answer}

Now
Paragraph:
{para}
Participants: {parts}
Give the answer as one JSON object that maps each participant, written exactly as in the list, to its list of {n1} states."""

EXTRACT = """Task: keep a record of the participants of a process, updated one sentence at a time.

Events (as JSON lists):
["create", participant, location]       the participant comes into existence in this sentence (is made, formed, produced); location = where, or "?"
["destroy", participant]                the participant stops existing in this sentence (is used up, consumed, converted into something else, broken down, dies)
["move", participant, from, to]         the participant is now in a different place (moves, flows, is carried, enters, is absorbed, is released); from = where it was, to = where it is now ("?" if not stated)
Locations are short phrases copied from the paragraph.

Example
Participants: {ex_parts}
{ex_lines}

Now
Participants: {parts}
Previous sentences: {context}
Sentence: {sentence}
Rules: only the events stated or clearly implied by this sentence; use the participant names exactly as listed; participants not affected by this sentence have no event.
Output:"""


def direct_prompt(r, ex):
    ans = json.dumps({p: st for p, st in zip(ex["participants"], ex["states"])})
    return DIRECT.format(rules=STATE_RULES, ex_para=para_text(ex), ex_parts=parts_text(ex), ex_answer=ans,
                         para=para_text(r), parts=parts_text(r), n1=len(r["sentence_texts"]) + 1)


def extract_prompts(r, ex):
    ev = gold_events(ex)
    lines = []
    for i, s in enumerate(ex["sentence_texts"]):
        ctx = " ".join(ex["sentence_texts"][:i]) or "(none)"
        lines.append(f"Previous sentences: {ctx}\nSentence: {s}\nOutput: {json.dumps({'events': ev[i]})}")
    exs = "\n\n".join(lines)
    out = []
    for i, s in enumerate(r["sentence_texts"]):
        ctx = " ".join(r["sentence_texts"][:i]) or "(none)"
        out.append(EXTRACT.format(ex_parts=parts_text(ex), ex_lines=exs, parts=parts_text(r), context=ctx, sentence=s))
    return out


# ------------------------------------------------------------------ generation (GPU)
CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
THINK_BUDGET, ANSWER_MAX, EXTRACT_MAX = 8192, 2048, 400


def gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    think_end = tok.convert_tokens_to_ids("</think>")
    os.makedirs(args.outdir, exist_ok=True)
    for arm in args.arms.split(","):
        args.arm, args.out = arm, f"{args.outdir}/raw_{arm}.jsonl"
        gen_arm(args, torch, tok, model, eos, think_end)


def gen_arm(args, torch, tok, model, eos, think_end):
    ex = example_para()
    rows = load(args.split)
    jobs = []  # (id, prompt)
    for r in rows:
        if args.arm == "extract":
            jobs += [(f"{r['para_id']}:{i + 1}", p) for i, p in enumerate(extract_prompts(r, ex))]
        else:
            jobs.append((r["para_id"], direct_prompt(r, ex)))
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out)}
    jobs = [j for j in jobs if j[0] not in done]
    think = args.arm == "direct_think"
    print(f"{args.arm} {args.split}: {len(jobs)} prompts, gpu {torch.cuda.get_device_name(0)}", flush=True)

    def texts(ps, prefix=None):
        t = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=think) for p in ps]
        return [a + b for a, b in zip(t, prefix)] if prefix else t

    def run(tx, max_new, sample, seed):
        enc = tok(tx, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if sample else dict(do_sample=False)
        if sample:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                                 **kw)[:, enc.input_ids.shape[1]:].tolist()
        res = []
        for j, ids in enumerate(out):
            m = next((i + 1 for i, x in enumerate(ids) if x in eos), len(ids))
            res.append((ids[:m], int(enc.attention_mask[j].sum())))
        return res

    bs = min(args.bs, 16) if think else args.bs  # a thinking batch holds up to 10k tokens per sequence
    with open(args.out, "a") as f:
        for k in range(0, len(jobs), bs):
            batch = jobs[k:k + bs]
            tx = texts([p for _, p in batch])
            if not think:
                for (i, p), (ids, plen) in zip(batch, run(tx, EXTRACT_MAX if args.arm == "extract" else ANSWER_MAX,
                                                          False, 0)):
                    f.write(json.dumps({"id": i, "text": tok.decode(ids, skip_special_tokens=True), "gen_tokens": len(ids),
                                        "prompt_tokens": plen}) + "\n")
                f.flush()
                continue
            res = run(tx, THINK_BUDGET + ANSWER_MAX, True, 43 + k)
            for (i, p), t0, (ids, plen) in zip(batch, tx, res):
                rec = {"id": i, "gen_tokens": len(ids), "forced": False}
                if think_end in ids and ids.index(think_end) < THINK_BUDGET:
                    te = ids.index(think_end) + 1
                    rec.update(think_tokens=te, thinking=tok.decode(ids[:te], skip_special_tokens=True),
                               text=tok.decode(ids[te:], skip_special_tokens=True), closed=True)
                else:  # reasoning did not close within the budget: keep its first 8,192 tokens, close, answer
                    keep = ids[:THINK_BUDGET]
                    t2 = t0 + tok.decode(keep, skip_special_tokens=False) + CLOSE
                    (ids2, _), = run([t2], ANSWER_MAX, True, 43 + k + 1)
                    rec.update(think_tokens=len(keep), thinking=tok.decode(keep, skip_special_tokens=True),
                               text=tok.decode(ids2, skip_special_tokens=True), closed=False, forced=True,
                               gen_tokens=len(keep) + len(ids2))
                rec["prompt_tokens"] = len(tok(t0, add_special_tokens=False).input_ids)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  {min(k + bs, len(jobs))}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------ parsing and read rules (CPU)
def norm_name(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def match_participant(name, parts):
    """The listed participant a written name refers to: exact (normalised) match of the whole entry or of one of its
    ';'-separated alternatives, else the same after dropping a leading article / trailing plural s; None if none."""
    n = norm_name(name)
    n2 = re.sub(r"^(the|a|an) ", "", n)
    for P in parts:
        alts = [norm_name(P)] + [norm_name(a) for a in P.split(";")]
        if n in alts or n2 in alts:
            return P
    for P in parts:
        alts = [re.sub(r"s$", "", norm_name(a)) for a in P.split(";")]
        if re.sub(r"s$", "", n2) in alts:
            return P
    # dev record A: misspelled entries in the gold participant list ("Enyzmes") -> closest entry, if unique and close
    import difflib
    scored = sorted(((max(difflib.SequenceMatcher(None, n2, norm_name(a)).ratio() for a in P.split(";")), P)
                     for P in parts), reverse=True)
    if scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
        return scored[0][1]
    return None


EV = re.compile(r'\[\s*"(create|destroy|move)"((?:\s*,\s*"[^"]*")*)\s*\]', re.I)


def parse_events(txt):
    out = []
    for m in EV.finditer(txt):
        args = re.findall(r'"([^"]*)"', m.group(2))
        out.append([m.group(1).lower()] + [a.strip() for a in args])
    return out


def loc_or_unknown(x):
    x = (x or "").strip()
    return "?" if x.lower() in ("", "?", "unknown", "none", "null", "n/a", "-") else x


def events_by_step(r, raw):
    """raw: {step: model text} -> {step: [(participant, op, from, to)]} for matched participants, plus counts."""
    parts = r["participants"]
    per, unmatched, total = collections.defaultdict(list), 0, 0
    for t in range(1, len(r["sentence_texts"]) + 1):
        for ev in parse_events(raw.get(t, "")):
            total += 1
            op, a = ev[0], ev[1:]
            P = match_participant(a[0], parts) if a else None
            if P is None:
                unmatched += 1
                continue
            if op == "create":
                per[t].append((P, "create", "-", loc_or_unknown(a[1] if len(a) > 1 else "?")))
            elif op == "destroy":
                per[t].append((P, "destroy", None, "-"))
            else:
                fr = loc_or_unknown(a[1] if len(a) > 1 else "?")
                to = loc_or_unknown(a[2] if len(a) > 2 else (a[1] if len(a) == 2 else "?"))
                if len(a) == 2:
                    fr = "?"
                per[t].append((P, "move", fr, to))
    return per, unmatched, total


def initial_state(P, per, n):
    """Read rule for state 0: created later -> "-"; first event a move with a stated origin -> that origin; else "?"."""
    for t in range(1, n + 1):
        for (Q, op, fr, to) in per.get(t, []):
            if Q != P:
                continue
            if op == "create":
                return "-"
            if op == "move" and fr not in ("?", "-"):
                return fr
            return "?"
    return "?"


def seed(s):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")


class VSAStore:
    """Fourier HRR key-value memory: one vector, keys and values from seeded hashes (separate codebooks), write =
    delta rule (unbind and remove the current value, add the new one), read = unbind + clean-up against the values
    written so far (plus "-" and "?")."""
    D = 2048

    def __init__(self):
        self.m = np.zeros(self.D, np.complex128)
        self.vals = {}

    def code(self, s, slot):
        rng = np.random.default_rng(seed(f"{slot}:{s}"))
        return np.exp(1j * rng.uniform(-np.pi, np.pi, self.D))

    def read(self, k):
        if not self.vals:
            return None
        y = self.m * np.conj(self.code(k, "key"))
        names = list(self.vals)
        sims = np.array([(np.conj(self.vals[v]) @ y).real / self.D for v in names])
        b = int(np.argmax(sims))
        return names[b] if sims[b] > 0.5 else None

    def write(self, k, v):
        for s in (v, "-", "?"):
            self.vals.setdefault(s, self.code(s, "value"))
        old = self.read(k)
        kc = self.code(k, "key")
        if old is not None:
            self.m -= kc * self.vals[old]
        self.m += kc * self.vals[v]


class DictStore:
    def __init__(self):
        self.d = {}

    def read(self, k):
        return self.d.get(k)

    def write(self, k, v):
        self.d[k] = v


def replay(r, per, store):
    """Write state 0 of every participant, then the events of sentence t in order; read every participant after
    every sentence. The same rules for both memories."""
    n = len(r["sentence_texts"])
    grid = {P: [] for P in r["participants"]}
    for P in r["participants"]:
        store.write(P, initial_state(P, per, n))
        grid[P].append(store.read(P))
    for t in range(1, n + 1):
        for (P, op, fr, to) in per.get(t, []):
            store.write(P, to)  # create -> location (or "?"), destroy -> "-", move -> destination (or "?")
        for P in r["participants"]:
            grid[P].append(store.read(P))
    return grid


def parse_grid(r, txt):
    """The direct arm's JSON grid -> {participant: [N + 1 states]}; missing / malformed rows become all "?" (and are
    counted)."""
    n1 = len(r["sentence_texts"]) + 1
    obj, bad = None, 0
    for m in re.finditer(r"\{.*\}", txt, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
    if obj is None:  # try the last {...} block greedily from each opening brace
        for i in [k for k, c in enumerate(txt) if c == "{"]:
            try:
                obj = json.loads(txt[i:txt.rindex("}") + 1])
                break
            except Exception:
                continue
    grid = {}
    for P in r["participants"]:
        v = None
        if isinstance(obj, dict):
            v = obj.get(P)
            if v is None:
                for k2, v2 in obj.items():
                    if match_participant(k2, [P]) == P:
                        v = v2
                        break
        if not isinstance(v, list) or len(v) != n1:
            bad += 1
            v = (list(v) + ["?"] * n1)[:n1] if isinstance(v, list) else ["?"] * n1
        grid[P] = [loc_or_unknown(str(x)) if str(x).strip() != "-" else "-" for x in v]
    return grid, bad


def action_rows(r, grid):
    rows = []
    for t in range(1, len(r["sentence_texts"]) + 1):
        for P in r["participants"]:
            b, a = grid[P][t - 1], grid[P][t]
            b = b if b is not None else "?"
            a = a if a is not None else "?"
            rows.append([r["para_id"], str(t), P, action(b, a), b, a])
    return rows


def load_raw(path):
    return {x["id"]: x for x in map(json.loads, open(path))} if os.path.exists(path) else {}


def assemble(args):
    rows = load(args.split)
    os.makedirs(args.outdir, exist_ok=True)
    stats = {}
    for arm in ("direct_think", "direct_nothink"):
        raw = load_raw(f"{args.outdir}/raw_{arm}.jsonl")
        if not raw:
            continue
        grids, bad, missing = {}, 0, 0
        for r in rows:
            x = raw.get(r["para_id"])
            missing += x is None
            g, b = parse_grid(r, x["text"] if x else "")
            grids[r["para_id"]], bad = g, bad + b
        stats[arm] = {"participants_malformed": bad, "paragraphs_missing": missing}
        write_arm(args.outdir, arm, rows, grids)
    raw = load_raw(f"{args.outdir}/raw_extract.jsonl")
    if raw:
        ev_all, unmatched, total, missing = {}, 0, 0, 0
        for r in rows:
            txt = {}
            for t in range(1, len(r["sentence_texts"]) + 1):
                x = raw.get(f"{r['para_id']}:{t}")
                missing += x is None
                txt[t] = x["text"] if x else ""
            per, u, tot = events_by_step(r, txt)
            ev_all[r["para_id"]] = {str(t): v for t, v in per.items()}
            unmatched, total = unmatched + u, total + tot
        for arm, cls in (("store_vsa", VSAStore), ("store_dict", DictStore)):
            grids = {r["para_id"]: replay(r, {int(t): v for t, v in ev_all[r["para_id"]].items()}, cls())
                     for r in rows}
            write_arm(args.outdir, arm, rows, grids)
        json.dump(ev_all, open(f"{args.outdir}/events.json", "w"))
        stats["extract"] = {"events": total, "unmatched_participant": unmatched, "sentences_missing": missing}
    json.dump(stats, open(f"{args.outdir}/assemble_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def write_arm(outdir, arm, rows, grids):
    json.dump(grids, open(f"{outdir}/{arm}_grids.json", "w"))
    with open(f"{outdir}/{arm}.tsv", "w") as f:
        for r in rows:
            for row in action_rows(r, grids[r["para_id"]]):
                f.write("\t".join(row) + "\n")


# ------------------------------------------------------------------ scoring (CPU)
def official_scores(split, pred_tsv):
    """Per-process official question scores, computed by the leaderboard evaluator's own modules."""
    sys.path.insert(0, f"{LB}/evaluator")
    from process import ActionFile
    from scoring import QuestionScores
    ans = ActionFile.from_file(f"{LB}/data/{split}/answers.tsv").summarize()
    pred = ActionFile.from_file(pred_tsv).summarize()
    return {str(pid): QuestionScores.from_summaries(a, pred[pid]) for pid, a in ans.items()}


def pr_matrix(scores, pids):
    """(n, 8): per process the precision and recall of inputs, outputs, conversions, moves."""
    return np.array([[s.inputs.precision, s.outputs.precision, s.conversions.precision, s.moves.precision,
                      s.inputs.recall, s.outputs.recall, s.conversions.recall, s.moves.recall]
                     for s in (scores[p] for p in pids)])


def f1_of(m):
    """The evaluator's aggregation (unrounded): per question type P and R averaged over processes; overall = mean of
    the four; F1 of overall P and R. m: (n, 8) or (B, n, 8) -> points."""
    mu = m.mean(-2)
    P, R = mu[..., :4].mean(-1), mu[..., 4:].mean(-1)
    return 100 * np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0), 100 * P, 100 * R


def loc_norm(x):
    sys.path.insert(0, f"{LB}/evaluator")
    from text.terms import extract_termsets_with_normalization
    return frozenset().union(*extract_termsets_with_normalization(x.lower()))


def cell_correct(g, p):
    if g in ("-", "?") or p in ("-", "?", None):
        return g == p
    return loc_norm(g) == loc_norm(p)


ARMS = ["direct_think", "direct_nothink", "store_vsa", "store_dict"]


def implied_actions(r, grid):
    return {(P, t): action(grid[P][t - 1] or "?", grid[P][t] or "?") for P in r["participants"]
            for t in range(1, len(r["sentence_texts"]) + 1)}


def extracted_actions(r, ev):
    """Per (participant, sentence) the action the extractor's own events imply (create > destroy > move), and the
    written destination."""
    out = {}
    for t in range(1, len(r["sentence_texts"]) + 1):
        for (P, op, fr, to) in ev.get(str(t), []):
            a = {"create": "CREATE", "destroy": "DESTROY", "move": "MOVE"}[op]
            prev = out.get((P, t))
            rank = {"CREATE": 0, "DESTROY": 1, "MOVE": 2}
            if prev is None or rank[a] < rank[prev[0]]:
                out[(P, t)] = (a, to)
    return out


def action_prf(rows, pred_fn):
    """Precision / recall / F1 of non-NONE actions (type must match) and destination accuracy on matched CREATE/MOVE."""
    tp = npred = ngold = dst_ok = dst_n = 0
    for r in rows:
        gold = gold_actions(r)
        pred = pred_fn(r)
        for key, (ga, gb, gaft) in gold.items():
            pa, pdst = pred.get(key, ("NONE", None))
            ngold += ga != "NONE"
            npred += pa != "NONE"
            if ga != "NONE" and pa == ga:
                tp += 1
                if ga in ("CREATE", "MOVE"):
                    dst_n += 1
                    dst_ok += cell_correct(gaft, pdst)
    P, R = tp / max(npred, 1), tp / max(ngold, 1)
    return {"precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-12), "gold": ngold, "pred": npred, "tp": tp,
            "dest_acc": dst_ok / max(dst_n, 1), "dest_n": dst_n}


def score(args):
    import subprocess
    rows = load(args.split)
    pids = [r["para_id"] for r in rows]
    gold = {r["para_id"]: dict(zip(r["participants"], r["states"])) for r in rows}
    res, mats, cells = {}, {}, {}
    for arm in ARMS:
        tsv = f"{args.outdir}/{arm}.tsv"
        if not os.path.exists(tsv):
            continue
        sc = official_scores(args.split, tsv)
        mats[arm] = pr_matrix(sc, pids)
        f1, P, R = f1_of(mats[arm])
        cli = subprocess.run([sys.executable, "evaluator.py", "--predictions", os.path.abspath(tsv), "--answers",
                              f"{LB}/data/{args.split}/answers.tsv", "--output", os.path.abspath(f"{args.outdir}/{arm}_cli.json")],
                             cwd=f"{LB}/evaluator", env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True)
        cj = json.load(open(f"{args.outdir}/{arm}_cli.json"))
        grids = json.load(open(f"{args.outdir}/{arm}_grids.json"))
        per = []
        for r in rows:
            g = grids[r["para_id"]]
            c = [cell_correct(gs, ps) for P in r["participants"] for gs, ps in zip(gold[r["para_id"]][P], g[P])]
            per.append((sum(c), len(c)))
        cells[arm] = np.array(per, float)
        res[arm] = {"official_f1": float(f1), "official_p": float(P), "official_r": float(R),
                    "cli": cj, "cli_rc": cli.returncode,
                    "state_acc": 100 * cells[arm][:, 0].sum() / cells[arm][:, 1].sum(),
                    "cells": int(cells[arm][:, 1].sum()),
                    "actions_implied": action_prf(rows, lambda r, g=grids: {k: (a, g[r["para_id"]][k[0]][k[1]])
                                                                         for k, a in implied_actions(r, g[r["para_id"]]).items()})}
    if os.path.exists(f"{args.outdir}/events.json"):
        ev = json.load(open(f"{args.outdir}/events.json"))
        res["extraction"] = action_prf(rows, lambda r: extracted_actions(r, ev[r["para_id"]]))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pids), (10000, len(pids)))
    comps = {}
    for a, b in [("store_vsa", "direct_think"), ("store_vsa", "direct_nothink"), ("direct_think", "direct_nothink"),
                 ("store_dict", "direct_think"), ("store_vsa", "store_dict")]:
        if a in mats and b in mats:
            d = f1_of(mats[a])[0] - f1_of(mats[b])[0]
            bd = f1_of(mats[a][idx])[0] - f1_of(mats[b][idx])[0]
            ca, cb = cells[a][idx], cells[b][idx]
            dc = 100 * (ca[..., 0].sum(1) / ca[..., 1].sum(1) - cb[..., 0].sum(1) / cb[..., 1].sum(1))
            comps[f"{a} - {b}"] = {"f1_d": float(d), "f1_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                                   "state_d": float(res[a]["state_acc"] - res[b]["state_acc"]),
                                   "state_ci": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))]}
    if "store_vsa" in res and "store_dict" in res:
        gv = json.load(open(f"{args.outdir}/store_vsa_grids.json"))
        gd = json.load(open(f"{args.outdir}/store_dict_grids.json"))
        same = sum(gv[p][P][t] == gd[p][P][t] for p in gv for P in gv[p] for t in range(len(gv[p][P])))
        tot = sum(len(gv[p][P]) for p in gv for P in gv[p])
        res["vsa_eq_dict"] = [same, tot]
    res["comparisons"] = comps
    json.dump(res, open(f"{args.outdir}/scores.json", "w"), indent=1)
    for arm in ARMS:
        if arm in res:
            x = res[arm]
            print(f"{arm:15s} official F1 {x['official_f1']:5.1f} (P {x['official_p']:5.1f} R {x['official_r']:5.1f}; "
                  f"CLI F1 {100 * x['cli']['f1']:5.1f})  state acc {x['state_acc']:5.1f} ({x['cells']} cells)  "
                  f"implied actions P {x['actions_implied']['precision']:.3f} R {x['actions_implied']['recall']:.3f}")
    if "extraction" in res:
        print("extraction", json.dumps(res["extraction"]))
    if "vsa_eq_dict" in res:
        print("store_vsa == store_dict cells:", res["vsa_eq_dict"])
    for k, v in comps.items():
        print(f"{k:32s} F1 {v['f1_d']:+6.1f} [{v['f1_ci'][0]:+6.1f}, {v['f1_ci'][1]:+6.1f}]   state "
              f"{v['state_d']:+6.1f} [{v['state_ci'][0]:+6.1f}, {v['state_ci'][1]:+6.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "assemble", "score", "example"])
    ap.add_argument("a1", nargs="?")
    ap.add_argument("a2", nargs="?")
    ap.add_argument("a3", nargs="?")
    ap.add_argument("--model")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "gen":
        a.arms, a.split, a.outdir = a.a1, a.a2, a.a3
        gen(a)
    elif a.cmd in ("assemble", "score"):
        a.split, a.outdir = a.a1, a.a2
        (assemble if a.cmd == "assemble" else score)(a)
    else:
        ex = example_para()
        print(ex["para_id"], ex["sentence_texts"], ex["participants"], ex["states"], sep="\n")
        print(direct_prompt(load("dev")[0], ex))
        print("=" * 80)
        print(extract_prompts(load("dev")[0], ex)[1])
