"""B33 common code (prereg/B33_fresh_task2_prereg.md): the three evaluation sets and the loader of the verbatim
prompting script.

Evaluation sets (each a list of (text, label) in Muennighoff/babi's load_babi format: passage.strip() + "\\n" +
question.strip(), label = index into the training answer vocabulary):
  orig500  the 500 task-2 test questions the paper used (load_babi(2, 2000, 200, 500)["test"]) -- sanity gates only
  fresh    the held-out set: b33_gen stories, registered seed, generated after registration (PRIMARY)
  hfrest   the other 500 task-2 test questions of the same distribution (load_babi(2, 2000, 200, 1000)["test"][500:]);
           never scored by any route the paper reports, but featurized and scored by the V-5 / V-7 pooled-state
           probes in August (n_test = 1000), so it is a SECONDARY check, not the confirmatory set
"""
import hashlib, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b28_common as C28                                              # noqa: E402

B33_NG = os.path.join(HERE, "b33_ngcls")
SETS = ["orig500", "fresh", "hfrest"]
FRESH_SEED = 20261118          # registered; never used before the registration commit
FRESH_STORIES = 200            # 200 stories x 5 questions = 1,000 questions (the size of an official test split)


def load_v49():
    """The verbatim V-49 prompting script (copy in b33_ngcls/, md5 in MD5SUMS), imported after the B28 copies so that
    its `import v13` / `from v5 import load_babi` / `from v30 import ...` resolve to the verbatim copies."""
    want = dict(l.split()[::-1] for l in open(os.path.join(B33_NG, "MD5SUMS")) if l.strip())
    for f, h in want.items():
        if hashlib.md5(open(os.path.join(B33_NG, f), "rb").read()).hexdigest() != h:
            raise SystemExit("b33_ngcls copy changed: %s" % f)
    M = C28.load_ngcls()
    keep = list(sys.path)
    spec = importlib.util.spec_from_file_location("v49", os.path.join(B33_NG, "v49.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["v49"] = m
    spec.loader.exec_module(m)
    sys.path[:] = keep
    M["v49"] = m
    return M


def load_sets(v5, fresh_file, names=SETS):
    """Training split (as published), vocabulary, and the requested evaluation sets."""
    d = v5.load_babi(2, 2000, 200, 500)
    vocab = d["vocab"]
    idx = {a: i for i, a in enumerate(vocab)}
    out = {}
    if "orig500" in names:
        out["orig500"] = d["test"]
    if "hfrest" in names:
        full = v5.load_babi(2, 2000, 200, 1000)
        assert full["vocab"] == vocab and full["test"][:500] == d["test"]
        out["hfrest"] = full["test"][500:]
        assert len(out["hfrest"]) == 500
    if "fresh" in names:
        rows = [json.loads(l) for l in open(fresh_file)]
        out["fresh"] = [(r["passage"].strip() + "\n" + r["question"].strip(), idx[r["answer"]]) for r in rows]
    return d, vocab, out


def story_ids(split):
    """Story index of each (text, label) row: a row continues the previous story when its statements extend the
    previous row's statements (Muennighoff rows carry every statement since the story began)."""
    ids, prev, k = [], None, -1
    for text, _ in split:
        st = text.strip().split("\n")[:-1]                   # drop the question line
        if prev is None or len(st) < len(prev) or st[:len(prev)] != prev:
            k += 1
        ids.append(k)
        prev = st
    return ids
