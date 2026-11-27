"""V-3 (prereg 57406f1): rebuild the compositional benchmark into a valid instrument, then re-measure.

Why this exists: over 8,000 generated samples, rule_priority and state_update have P(label=1)=1/3 by
construction, so the majority class scores 0.66-0.67 and the frozen-LLM probe was measured within 0.05pp
and 0.11pp of exactly that. variable_tracking sat BELOW its own majority baseline in every arm. Only
binding_test carried signal. Three of four instruments could not register an effect in either direction.

F1  rule_priority / state_update: the queried item is drawn so P(label=1)=0.5 exactly.
F2  every reported accuracy is accompanied by the per-task majority-class baseline.
F3  variable_tracking is diagnosed before use: a TF-IDF bag-of-words logistic baseline on the raw text
    tests whether a surface cue anti-correlated with the label explains the below-majority result.
F4  train / val / test are three genuinely disjoint draws, deduplicated by exact text.

ADMISSION GATE (per task, evaluated before any augmentation is compared on it):
    majority_baseline + 3pp <= frozen-probe accuracy <= 0.95, in 3/3 seeds.
"""
import argparse, hashlib, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TASKS = ["variable_tracking", "rule_priority", "state_update", "binding_test"]
HID = 2560                     # Phi-2 hidden size


# ------------------------------------------------------------------ generators

def gen_variable_tracking(rng):
    names = rng.sample(["x", "y", "z", "a", "b", "c", "p", "q"], 5)
    vals = [rng.randint(1, 50) for _ in range(5)]
    order = list(zip(names, vals))
    rng.shuffle(order)
    i, j = rng.sample(range(5), 2)
    while vals[i] == vals[j]:                       # remove ties: they made label=0 by default
        vals[j] = rng.randint(1, 50)
        order = [(n, vals[names.index(n)]) for n, _ in order]
    text = ". ".join("%s = %d" % (n, v) for n, v in order) + \
           ". Is %s greater than %s?" % (names[i], names[j])
    return {"text": text, "label": int(vals[i] > vals[j]), "task": "variable_tracking"}


def gen_rule_priority(rng):
    e = rng.choice(["Alice", "Bob", "Carol", "David", "Eve", "Frank"])
    props = rng.sample(["happy", "sad", "tired", "excited", "calm", "angry", "nervous"], 4)
    rules = ["%s is %s." % (e, props[0]),
             "But wait, %s is actually %s." % (e, props[1]),
             "No, %s is %s." % (e, props[2])]
    if rng.random() < 0.5:
        rules.append("Finally, %s is %s." % (e, props[3])); final = props[3]; others = props[:3]
    else:
        final = props[2]; others = props[:2]
    # F1: label is 1 with probability exactly 1/2
    q = final if rng.random() < 0.5 else rng.choice(others)
    return {"text": " ".join(rules) + " Is %s %s?" % (e, q),
            "label": int(q == final), "task": "rule_priority"}


def gen_state_update(rng):
    obj = rng.choice(["box", "ball", "cup", "book", "key", "phone", "wallet"])
    locs = rng.sample(["table", "shelf", "floor", "drawer", "desk", "chair", "bed", "counter"], 5)
    n_moves = rng.randint(3, 5)
    moves = ["The %s starts on the %s." % (obj, locs[0])]
    moves += ["Then it is moved to the %s." % locs[i] for i in range(1, n_moves)]
    final = locs[n_moves - 1]
    others = [l for l in locs[:n_moves] if l != final]
    q = final if rng.random() < 0.5 else rng.choice(others)          # F1
    return {"text": " ".join(moves) + " Is the %s on the %s?" % (obj, q),
            "label": int(q == final), "task": "state_update"}


def gen_binding_test(rng):
    ppl = rng.sample(["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"], 4)
    jobs = rng.sample(["doctor", "teacher", "engineer", "lawyer", "artist", "chef", "pilot"], 4)
    cities = rng.sample(["Paris", "London", "Tokyo", "Berlin", "Rome", "Madrid", "Sydney"], 4)
    b = list(zip(ppl, jobs, cities)); rng.shuffle(b)
    stmts = ["%s is a %s living in %s" % (p, j, c) for p, j, c in b]
    qi = rng.randrange(4)
    if rng.random() < 0.5:
        correct, wrong = b[qi][1], b[(qi + 1) % 4][1]
        ans = correct if rng.random() < 0.5 else wrong
        q = "Is %s a %s?" % (b[qi][0], ans)
    else:
        correct, wrong = b[qi][2], b[(qi + 1) % 4][2]
        ans = correct if rng.random() < 0.5 else wrong
        q = "Does %s live in %s?" % (b[qi][0], ans)
    return {"text": ". ".join(stmts) + ". " + q, "label": int(ans == correct), "task": "binding_test"}


GEN = {"variable_tracking": gen_variable_tracking, "rule_priority": gen_rule_priority,
       "state_update": gen_state_update, "binding_test": gen_binding_test}


def make_splits(task, n_train, n_val, n_test, seed=0):
    """F4: three disjoint splits, by HASH BUCKET rather than by cross-split dedup.

    Dedup was the wrong mechanism and the label check caught it: for rule_priority the number of DISTINCT
    texts carrying label 0 exceeds those carrying label 1 by 2-3x (label 1 admits one query, label 0
    admits several), so as a small text space exhausted, dedup pushed the later splits toward label 0 --
    test came out at P(1)=0.38. Bucketing by md5(text) is independent of the label, gives disjointness by
    construction, and leaves the generative label distribution untouched. Repeated draws of the same text
    within a split are kept: they are honest i.i.d. samples, and filtering them would reintroduce exactly
    the bias being removed.
    """
    rng = random.Random(10_000 + seed)
    want = {"train": n_train, "val": n_val, "test": n_test}
    out = {"train": [], "val": [], "test": []}
    guard = 0
    while any(len(out[k]) < want[k] for k in want) and guard < 40_000_000:
        guard += 1
        s = GEN[task](rng)
        b = int(hashlib.md5(s["text"].encode()).hexdigest(), 16) % 10
        split = "train" if b < 7 else ("val" if b == 7 else "test")
        if len(out[split]) < want[split]:
            out[split].append(s)
    return out


def majority_baseline(rows):
    p = sum(r["label"] for r in rows) / len(rows)
    return max(p, 1.0 - p), p


# ------------------------------------------------------------------ F3: surface-cue probe

def bow_baseline(splits):
    """TF-IDF + logistic regression on the raw text. If this beats the majority baseline the task has a
    surface cue; if it is far below, a cue anti-correlated with the label is implicated."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr = v.fit_transform([r["text"] for r in splits["train"]])
    ytr = [r["label"] for r in splits["train"]]
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    Xte = v.transform([r["text"] for r in splits["test"]])
    yte = np.array([r["label"] for r in splits["test"]])
    return float((clf.predict(Xte) == yte).mean())


# ------------------------------------------------------------------ torch plumbing

class DS(Dataset):
    def __init__(self, rows, tok, max_len=256):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        e = self.tok(r["text"], truncation=True, max_length=self.max_len,
                     padding="max_length", return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(r["label"])}


class Head(nn.Module):
    """Frozen-LLM heads. 'probe' is the floor; 'mlp' is PARAMETER-MATCHED to the VSA module, which V-2
    failed to do (34.2M against 5.1K cannot separate structure from capacity)."""
    def __init__(self, kind, n_class=2, mlp_hidden=4707):
        super().__init__()
        self.kind = kind
        if kind == "probe":
            self.net = nn.Linear(HID, n_class)
        else:
            self.net = nn.Sequential(nn.Linear(HID, mlp_hidden), nn.GELU(),
                                     nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
                                     nn.Linear(mlp_hidden, n_class))

    def forward(self, pooled):
        return self.net(pooled)


def build_llm(device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    return tok, llm


@torch.no_grad()
def pool(llm, ids, am):
    h = llm(input_ids=ids, attention_mask=am).last_hidden_state
    m = am.unsqueeze(-1).to(h.dtype)
    return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float()


@torch.no_grad()
def featurize(rows, tok, llm, device, bs=32, max_len=256):
    """Pool the frozen LLM once per split. The head is then trained on cached features, which is exact
    (the LLM is frozen, so its output cannot change between epochs) and about 8x cheaper than recomputing
    the forward pass every epoch."""
    X, y = [], []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        e = tok([r["text"] for r in chunk], truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt")
        X.append(pool(llm, e["input_ids"].to(device), e["attention_mask"].to(device)).cpu())
        y += [r["label"] for r in chunk]
    return torch.cat(X), torch.tensor(y)


def train_head(kind, feats, device, seed, epochs=60, bs=64, lr=1e-3, wd=1e-2):
    """feats: {'train': (X, y), 'val': ..., 'test': ...}. Epochs are cheap on cached features, so the
    head is given enough of them to converge; the reported epoch is still selected on val."""
    torch.manual_seed(seed)
    head = Head(kind).to(device)
    ntr = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    Xtr, ytr = feats["train"][0].to(device), feats["train"][1].to(device)

    def ev(split):
        X, y = feats[split][0].to(device), feats[split][1].to(device)
        head.eval()
        with torch.no_grad():
            return float((head(X).argmax(-1) == y).float().mean().item())

    n = Xtr.size(0)
    best_v, best_sd, best_ep = -1.0, None, 0
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g).to(device)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        v = ev("val")
        if v > best_v:
            best_v, best_ep = v, ep + 1
            best_sd = {kk: t.detach().clone() for kk, t in head.state_dict().items()}
    head.load_state_dict(best_sd)
    return {"kind": kind, "seed": seed, "trainable": ntr, "val": best_v,
            "val_epoch": best_ep, "test": ev("test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="calib", choices=["calib", "labels"])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="v3_logs/v3_calib.json")
    a = ap.parse_args()

    if a.mode == "labels":
        print("=== F1/F2 check: label balance and majority baseline after the fix ===")
        for t in TASKS:
            sp = make_splits(t, 4000, 1000, 2000, seed=0)
            for k in ["train", "val", "test"]:
                mb, p1 = majority_baseline(sp[k])
                print("  %-19s %-5s n=%-5d P(1)=%.4f  majority=%.4f" % (t, k, len(sp[k]), p1, mb))
        return

    tok, llm = build_llm(a.device)
    rows = []
    for t in TASKS:
        sp = make_splits(t, a.n_train, a.n_val, a.n_test, seed=0)
        mb, p1 = majority_baseline(sp["test"])
        bow = bow_baseline(sp)
        feats = {k: featurize(sp[k], tok, llm, a.device) for k in ["train", "val", "test"]}
        probes = [train_head("probe", feats, a.device, s) for s in range(a.seeds)]
        acc = [r["test"] for r in probes]
        admitted = all(mb + 0.03 <= x <= 0.95 for x in acc)
        print("  %-19s majority=%.4f  bow=%.4f  probe=%s  mean=%.4f  ADMITTED=%s"
              % (t, mb, bow, [round(x, 4) for x in acc], float(np.mean(acc)), admitted), flush=True)
        rows.append({"task": t, "P1_test": p1, "majority": mb, "bow": bow,
                     "probe_test": acc, "probe_mean": float(np.mean(acc)), "admitted": bool(admitted),
                     "trainable": probes[0]["trainable"]})
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    adm = [r["task"] for r in rows if r["admitted"]]
    print("\n  ADMITTED TASKS (%d/4): %s" % (len(adm), adm or "NONE"), flush=True)
    if len(adm) < 2:
        print("  >>> fewer than two admitted: this benchmark cannot support a claim about VSA "
              "augmentation, per the prereg.", flush=True)
    print("V3-CALIB-DONE", flush=True)


if __name__ == "__main__":
    main()
