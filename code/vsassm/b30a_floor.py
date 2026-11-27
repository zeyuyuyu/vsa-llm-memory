"""B30a (prereg/B30a_babilong_regex_floor_prereg.md): BABILong QA1-QA5 with no language model.

A keyword/template parser of bAbI sentences, a Python dict in place of the 256 x 256 matrix memory, and B18's own
answer rules (b18.World.apply / b18.World.answer, imported, not copied). Arms:
  F1   (primary)   vocabulary filter (every word of a sentence in the 0k training-split vocabulary, the filter that
                   built the B18 student's training labels, B18 amendment 2) + templates -> dict
  F1m  (check)     the same events written into B18's 256 x 256 matrix memory (b18.World unchanged)
  F2   (sensitivity) templates only, no vocabulary filter -> dict
Developed on the BABILong training split only (RMT-team/babilong-train-5k-samples, 0k and 4k).

python3 b30a_floor.py dev  --train DIR                       # training split only
python3 b30a_floor.py test --train DIR --test DIR --out DIR  # official test set (after the prereg commit)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b18  # noqa: E402  sentences(), World, CUES, grounded(), correct(), correct_official()
import babilong_metrics_official as OFF  # noqa: E402  verbatim babilong/metrics.py (second scorer)

TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]
B18_SAMPLES = {"0k": 100, "4k": 100, "16k": 100, "64k": 100, "128k": 100, "512k": 25, "1M": 25, "10M": 5}


def alt(words):
    """Regex alternation of B18's cue words (b18.CUES, written from general English in B18 amendment 2)."""
    return "(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + ")"


GO, TAKE, DROP, GIVE = (alt(b18.CUES[k]) for k in ("go", "take", "drop", "give"))
DIR = r"(north|south|east|west)"
TEMPLATES = [  # (event type, pattern over the lower-cased sentence without final punctuation)
    ("rel", re.compile(rf"^the (\w+) is {DIR} of the (\w+)$")),
    ("give", re.compile(rf"^(\w+) {GIVE} the (\w+) to (\w+)$")),
    ("take", re.compile(rf"^(\w+) {TAKE}(?: up)? the (\w+)(?: there)?$")),
    ("drop", re.compile(rf"^(\w+) {DROP} the (\w+)(?: there)?$")),
    ("go", re.compile(rf"^(\w+) {GO}(?: back)?(?: to)? the (\w+)$")),
]
PRONOUNS = {"he", "she", "it", "they", "we", "i", "you", "him", "her", "them", "there", "this", "that", "the", "a",
            "an", "one", "who", "what", "which"}


def words_of(s):
    return re.sub(r"[^a-z ]", " ", s.lower()).split()


def vocabulary(train_dir):
    """Every word of the 0k training stories and questions of qa1-qa5 (the bAbI-fact filter of B18 amendment 2)."""
    v = set()
    for t in TASKS:
        for r in json.load(open(f"{train_dir}/{t}/0k.json")):
            v.update(words_of(r["input"]))
            v.update(words_of(r["question"]))
    return v


def labels(train_dir, task):
    """Official-metric labels: every answer seen in the training split (as b18.labels)."""
    out = set()
    for f in glob.glob(f"{train_dir}/{task}/*.json"):
        for r in json.load(open(f)):
            out.update(x.strip().lower() for x in str(r["target"]).split(","))
    return out


def parse(sentence, vocab=None):
    """Events of one sentence, in B18's event format; [] for a sentence that is not a bAbI fact.

    BABILong inserts a fact into the book text without a sentence break before it ("...but I The office is east of
    the garden."; found on the training split, dev 4k), so the splitter can return book words + fact. The fact is
    then a suffix of the sentence that starts at a capitalised word: every such suffix is tried, longest first.
    A fact can also be followed by book text with no space after its full stop ("John went to the bathroom.'s face";
    "...the bedroom..0 8 1/2"; dev 4k), so the sentence is first cut at every full stop."""
    out = []
    for piece in sentence.split("."):
        toks = re.sub(r"[^A-Za-z ]", " ", piece).split()
        for j, tok in enumerate(toks):
            if not tok[:1].isupper():
                continue
            ev = parse_words([t.lower() for t in toks[j:]], vocab)
            if ev:
                out.extend(ev)
                break
    return out


def parse_words(w, vocab):
    if not w or (vocab is not None and any(x not in vocab for x in w)):
        return []
    s = " ".join(w)
    for op, pat in TEMPLATES:
        m = pat.match(s)
        if not m:
            continue
        a = list(m.groups())
        if op in ("give", "take", "drop", "go") and a[0] in PRONOUNS:
            return []
        if op == "give" and a[2] in PRONOUNS:
            return []
        if op == "rel":
            return [["rel", a[0], a[1], a[2]]]
        return [[op] + a]
    return []


class DictWorld(b18.World):
    """b18.World with its matrix memory replaced by a Python dict; apply() and answer() are B18's, unchanged."""

    def __init__(self):
        super().__init__()
        self.D = {}

    def _write(self, x, y):
        self.D[x] = y

    def parent(self, x):
        v = self.D.get(x)
        return None if v in (None, "__unknown__") else v


def run_sample(sample, arm, vocab):
    world = b18.World() if arm == "F1m" else DictWorld()
    st = {"sents": 0, "events": 0, "applied": 0}
    for s in b18.sentences(sample["input"]):
        st["sents"] += 1
        for ev in parse(s, vocab if arm in ("F1", "F1m") else None):
            st["events"] += 1
            if b18.grounded(ev, s):  # the reader pipeline's grounding and cue checks, applied for parity
                st["applied"] += int(bool(world.apply(ev)))
    pred = world.answer(sample["question"])
    st["path"] = path_of(sample["question"])
    st["hops"] = hops_of(world, sample["question"]) if arm != "F1m" else None
    keys = set(world.K.names) if arm == "F1m" else set(world.D)
    vals = set(world.V.names) if arm == "F1m" else set(world.D.values())
    st.update({"n_keys": len(keys), "n_values": len(vals),
               "n_names": len(keys | vals - {"__unknown__"}), "n_rels": len(world.rels), "n_gives": len(world.gives)})
    return pred, st


def path_of(question):
    """The branch of b18.World.answer that answers this question (its dispatch depends on the question text alone,
    tested in the same order as there)."""
    q = question.strip().rstrip("?").lower()
    if re.match(r"where was (?:the )?(.+?) before (?:the )?(.+)$", q):
        return "history log"
    if re.match(r"what is (north|south|east|west) of (?:the )?(.+)$", q) or \
            re.match(r"what is (?:the )?(.+?) (north|south|east|west) of$", q):
        return "relation log"
    if re.match(r"where is (?:the )?(.+)$", q) or re.match(r"is (\w+) in the (.+)$", q) or \
            re.match(r"how many objects is (\w+) carrying$", q) or re.match(r"what is (\w+) carrying$", q):
        return "matrix memory"
    for pat in [r"who gave (?:the )?(.+) to (\w+)$", r"who did (\w+) give (?:the )?(.+) to$",
                r"what did (\w+) give to (\w+)$", r"who received (?:the )?(.+)$", r"who gave (?:the )?(.+)$"]:
        if re.match(pat, q):
            return "give log"
    return "rules alone"


def hops_of(world, question):
    """Matrix hops that b18.World.place() makes for a 'where is X' question, counted on the dict world."""
    m = re.match(r"where is (?:the )?(.+)$", question.strip().rstrip("?").lower())
    if not m:
        return None
    cur, n = m.group(1), 0
    while n < 10:
        p = world.parent(cur)
        if p is None:
            return n
        n += 1
        if p not in world.persons and p not in world.objects:
            return n
        cur = p
    return n


def score(pred, target, question, task, labs):
    p = "" if pred is None else str(pred)
    return {"ok_official": bool(b18_official(pred, target, question, labs)),
            "ok_metrics_py": bool(OFF.compare_answers(str(target), p, question, OFF.TASK_LABELS[task])),
            "ok_exact": bool(b18.correct(pred, target))}


def b18_official(pred, target, question, labs):
    """b18.correct_official with the label set passed in (b18 reads its labels from a fixed path)."""
    if pred is None:
        return False
    out = str(pred).lower().split(".")[0]
    q = question.lower()
    L = {l for l in labs if not re.search(r"\b" + re.escape(l) + r"\b", q)}
    found = {l for l in L if re.search(r"\b" + re.escape(l) + r"\b", out)}
    tgt = {x.strip() for x in str(target).lower().split(",")}
    return tgt <= found and not (found - tgt) if len(tgt) > 1 else found == tgt


def code_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def dev(args):
    vocab = vocabulary(args.train)
    print(f"vocabulary {len(vocab)} words; code {code_hash()}")
    for length in args.lengths.split(","):
        for t in TASKS:
            f = f"{args.train}/{t}/{length}.json"
            if not os.path.exists(f):
                continue
            rows = json.load(open(f))[args.offset:args.offset + args.n]
            labs = labels(args.train, t)
            for arm in args.arms.split(","):
                ok, miss = 0, []
                for r in rows:
                    pred, st = run_sample(r, arm, vocab)
                    sc = score(pred, r["target"], r["question"], t, labs)
                    ok += sc["ok_official"]
                    if not sc["ok_official"] and len(miss) < 3:
                        miss.append((r["question"], r["target"], pred))
                print(f"train {length} {t} {arm}: {ok}/{len(rows)}  misses {miss}", flush=True)


def test(args):
    vocab = vocabulary(args.train)
    os.makedirs(args.out, exist_ok=True)
    rows_out, summary = [], {}
    for length in args.lengths.split(","):
        for t in TASKS:
            f = f"{args.test}/{t}/{length}.json"
            if length == "10M":
                f = f"{args.test}/{t}/10M_first5.json"
            if not os.path.exists(f):
                print(f"missing {f}", flush=True)
                continue
            data = json.load(open(f))
            n = len(data) if args.all else min(len(data), B18_SAMPLES[length])
            labs = labels(args.train, t)
            for arm in args.arms.split(","):
                t0, ok = time.time(), 0
                for i, r in enumerate(data[:n]):
                    pred, st = run_sample(r, arm, vocab)
                    sc = score(pred, r["target"], r["question"], t, labs)
                    ok += sc["ok_official"]
                    rows_out.append({"length": length, "task": t, "i": i, "arm": arm, "question": r["question"],
                                     "target": r["target"], "pred": pred, **sc, **st,
                                     "in_b18": i < B18_SAMPLES[length]})
                summary[f"{arm}/{length}/{t}"] = {"acc": round(ok / n, 4), "n": n, "secs": round(time.time() - t0, 1)}
                print(f"{arm} {length} {t}: {ok}/{n}  {time.time() - t0:.0f}s", flush=True)
            with open(f"{args.out}/b30a_rows.jsonl", "w") as fo:
                for r in rows_out:
                    fo.write(json.dumps(r) + "\n")
            json.dump({"code": code_hash(), "summary": summary}, open(f"{args.out}/b30a_summary.json", "w"), indent=1)


def gate(args):
    """G0: the local test files are B18's inputs. Reads no floor output.
    (1) questions and targets equal B18's banked per-question outputs (b23_ref/b18_s0, 0k-128k, 100 per task; b18_10M);
    (2) the inputs equal those sent to the frontier models (b18_api_subset.jsonl.gz: first 25 / 5 per task);
    (3) at 10M, b18.sentences() gives exactly B18's reader-call count per sample (b18_10M partial files)."""
    import gzip
    here = os.path.dirname(os.path.abspath(__file__))
    bad, n = [], 0
    cache = {}

    class Data:  # loads one file at a time (the 1M files are 400 MB each)
        def __getitem__(self, key):
            if key not in cache:
                cache.clear()
                length, t = key
                cache[key] = json.load(open(f"{args.test}/{t}/{'10M_first5' if length == '10M' else length}.json"))
            return cache[key]

    data = Data()
    for length in ["0k", "4k", "16k", "64k", "128k"]:
        for t in TASKS:
            banked = [json.loads(l) for l in open(f"{here}/b23_ref/b18_s0/{length}_{t}.jsonl")]
            for i, b in enumerate(banked):
                r = data[(length, t)][i]
                n += 1
                if (b["question"], str(b["target"])) != (r["question"], str(r["target"])):
                    bad.append(("b18_s0", length, t, i))
    for t in TASKS:
        banked = [json.loads(l) for l in open(f"{here}/b18_10M/10M_{t}.partial.jsonl")]
        for i, b in enumerate(banked):
            r = data[("10M", t)][i]
            n += 2
            if (b["question"], str(b["target"])) != (r["question"], str(r["target"])):
                bad.append(("b18_10M q", t, i))
            ns = len(b18.sentences(r["input"]))
            if ns != b["calls"]:
                bad.append(("b18_10M calls", t, i, ns, b["calls"]))
    with gzip.open(f"{here}/b18_api_subset.jsonl.gz", "rt") as fh:
        for line in fh:
            s = json.loads(line)
            if "length" not in s:
                continue
            r = data[(s["length"], s["task"])][s["i"]]
            n += 1
            if (s["input"], s["question"], str(s["target"])) != (r["input"], r["question"], str(r["target"])):
                bad.append(("subset", s["length"], s["task"], s["i"]))
    print(f"G0: {n} checks, {len(bad)} mismatches {bad[:10]}")
    print("G0 PASS" if not bad else "G0 FAIL")


def ours_rows(here, length, t):
    """B18's per-question correctness on the B18 samples (official rule), in sample order."""
    if length in ("0k", "4k", "16k", "64k", "128k"):
        return [json.loads(l)["ok_official"] for l in open(f"{here}/b23_ref/b18_s0/{length}_{t}.jsonl")]
    if length == "10M":
        return [json.loads(l)["ok_official"] for l in open(f"{here}/b18_10M/10M_{t}.partial.jsonl")]
    # 512k / 1M: per-question outputs banked locally for the first 5 (b18_api/ours_subset.json); B18's verdict gives
    # 25/25 per task at both lengths, so the other 20 are correct.
    sub = json.load(open(f"{here}/b18_api/ours_subset.json"))
    first = [sub[f"{t}-{length}-{i}"]["ok"] for i in range(5)]
    assert all(first)
    return first + [True] * 20


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return (100 * (c - h), 100 * (c + h))


def analyze(args):
    """Verdict generator (written before any test output): reads b30a_rows.jsonl, writes the verdict text."""
    from scipy.stats import binomtest
    here = os.path.dirname(os.path.abspath(__file__))
    rows = [json.loads(l) for l in open(f"{args.out}/b30a_rows.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["length"], r["task"]), []).append(r)
    L = [l for l in B18_SAMPLES if any(k[1] == l for k in by)]
    out = []
    P = out.append
    P("# B30a verdict (prereg/B30a_babilong_regex_floor_prereg.md): BABILong QA1-5 with no language model")
    P(f"code {json.load(open(f'{args.out}/b30a_summary.json'))['code']}; rows {args.out}/b30a_rows.jsonl")
    P("")
    P("QA1-5 mean accuracy (%), official rule, B18's samples (n per length 500 / ... / 125 at 512k, 1M / 25 at 10M)")
    P("length  n    ours   F1 floor [95% Wilson]   F1m (matrix)  F2 (no vocab)  F1 - ours  McNemar P (ours/F1 alone)"
      "  metrics.py agrees")
    verdict_e1, verdict_e2 = True, []
    for length in L:
        ours = sum((ours_rows(here, length, t) for t in TASKS), [])
        f1 = sum(([r for r in by.get(("F1", length, t), []) if r["in_b18"]] for t in TASKS), [])
        f1m = sum(([r for r in by.get(("F1m", length, t), []) if r["in_b18"]] for t in TASKS), [])
        f2 = sum(([r for r in by.get(("F2", length, t), []) if r["in_b18"]] for t in TASKS), [])
        n = len(f1)
        assert n == len(ours), (length, n, len(ours))
        a_o = 100 * sum(ours) / n
        a1 = 100 * sum(r["ok_official"] for r in f1) / n
        lo, hi = wilson(sum(r["ok_official"] for r in f1), n)
        a1m = 100 * sum(r["ok_official"] for r in f1m) / max(1, len(f1m))
        a2 = 100 * sum(r["ok_official"] for r in f2) / max(1, len(f2))
        b = sum(1 for o, r in zip(ours, f1) if o and not r["ok_official"])
        c = sum(1 for o, r in zip(ours, f1) if r["ok_official"] and not o)
        pm = binomtest(b, b + c).pvalue if b + c else 1.0
        agree = sum(r["ok_official"] == r["ok_metrics_py"] for r in f1)
        same_m = sum(r1["pred"] == r2["pred"] for r1, r2 in zip(f1, f1m))
        P(f"{length:6s} {n:4d}  {a_o:5.1f}   {a1:5.1f} [{lo:4.1f}-{hi:5.1f}]       {a1m:5.1f} ({same_m}/{len(f1m)} same)"
          f"  {a2:5.1f}        {a1 - a_o:+5.1f}      {pm:.3g} ({b}/{c})   {agree}/{n}")
        verdict_e1 &= a1 >= a_o - 1.0
        if a1 <= a_o - 5.0:
            verdict_e2.append(length)
    P("")
    P(f"E1 (F1 >= ours - 1.0 at every length): {'HOLDS' if verdict_e1 else 'FAILS'}")
    P(f"E2 (F1 below ours by >= 5 at some length): {'HOLDS at ' + ', '.join(verdict_e2) if verdict_e2 else 'does not hold'}")
    P("")
    P("Per task, F1 (%):  " + "  ".join(TASKS))
    for length in L:
        cells = []
        for t in TASKS:
            rr = [r for r in by.get(("F1", length, t), []) if r["in_b18"]]
            cells.append(f"{100 * sum(r['ok_official'] for r in rr) / len(rr):5.1f}")
        P(f"  {length:6s} " + "  ".join(cells))
    secondary = [(length, t) for length in ("512k", "1M") for t in TASKS
                 if len(by.get(("F1", length, t), [])) > B18_SAMPLES[length]]
    if secondary:
        P("")
        P("Secondary: all 100 samples per task at 512k and 1M (B18 ran only the first 25), F1 / F2 (%)")
        for length in ("512k", "1M"):
            f1 = sum((by.get(("F1", length, t), []) for t in TASKS), [])
            f2 = sum((by.get(("F2", length, t), []) for t in TASKS), [])
            if len(f1) > 125:
                P(f"  {length:6s} n={len(f1)}  F1 {100 * sum(r['ok_official'] for r in f1) / len(f1):.1f}"
                  f"  F2 {100 * sum(r['ok_official'] for r in f2) / max(1, len(f2)):.1f}")
    P("")
    P("Code paths of b18.World.answer (determined by the question text), share of the B18 questions, and accuracy")
    P("length  path            share   ours    F1     'where is' hops (F1 parse): 1 / 2")
    for length in L:
        f1 = sum(([r for r in by.get(("F1", length, t), []) if r["in_b18"]] for t in TASKS), [])
        ours = sum((ours_rows(here, length, t) for t in TASKS), [])
        for path in ("matrix memory", "history log", "relation log", "give log", "rules alone"):
            idx = [i for i, r in enumerate(f1) if r["path"] == path]
            if not idx:
                P(f"{length:6s}  {path:14s}  {0:5.1f}%")
                continue
            h1 = sum(1 for i in idx if f1[i]["hops"] == 1)
            h2 = sum(1 for i in idx if f1[i]["hops"] == 2)
            hop = f"{h1} / {h2}" if path == "matrix memory" else ""
            P(f"{length:6s}  {path:14s}  {100 * len(idx) / len(f1):5.1f}%  {100 * sum(ours[i] for i in idx) / len(idx):5.1f}"
              f"  {100 * sum(f1[i]['ok_official'] for i in idx) / len(idx):5.1f}   {hop}")
    P("")
    P("Distinct names written per story (dict keys, values, both), max over the B18 samples; cap of B18's memory 256")
    for arm in ("F1", "F2"):
        for length in L:
            rr = sum(([r for r in by.get((arm, length, t), []) if r["in_b18"]] for t in TASKS), [])
            if not rr:
                continue
            P(f"  {arm} {length:6s} keys max {max(r['n_keys'] for r in rr):4d}  values max {max(r['n_values'] for r in rr):4d}"
              f"  names max {max(r['n_names'] for r in rr):4d} (mean {sum(r['n_names'] for r in rr) / len(rr):.1f})"
              f"  over 256: {sum(r['n_keys'] > 256 or r['n_values'] > 256 for r in rr)}")
    P("")
    P("The reader's own codebooks (not logged): 10M bound from b18_10M (keys written <= events applied)")
    for t in TASKS:
        b = [json.loads(l) for l in open(f"{here}/b18_10M/10M_{t}.partial.jsonl")]
        f1 = [r for r in by.get(("F1", "10M", t), [])]
        for i, x in enumerate(b):
            g = f1[i] if i < len(f1) else None
            extra = ""
            if g is not None:
                bound = g["n_keys"] + max(0, x["events"] - g["applied"])
                extra = (f"  story facts applied by F1 {g['applied']}, story keys {g['n_keys']}, values {g['n_values']};"
                         f" bound if the reader applied every story fact: keys <= {bound}")
            P(f"  {t} sample {i}: reader events applied {x['events']}" + (" (> 256)" if x["events"] > 256 else "") + extra)
    txt = "\n".join(out) + "\n"
    print(txt)
    if args.verdict:
        open(args.verdict, "w").write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "test", "gate", "analyze"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--lengths", default="0k,4k,16k,64k,128k,512k,1M,10M")
    ap.add_argument("--arms", default="F1,F1m,F2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--all", action="store_true")  # every sample in the file, not only B18's
    ap.add_argument("--verdict", default="")
    args = ap.parse_args()
    {"dev": dev, "test": test, "gate": gate, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
