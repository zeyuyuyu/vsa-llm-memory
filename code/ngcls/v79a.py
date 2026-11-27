"""V-79a: diagnostic decomposition of V-79's store arm (prereg: the correction entry following 93c249b).
No LLM, CPU only. Same corpus slice, same seeds, same store code as v79.py -- only the bookkeeping differs:
per-instance hits are retained so exact match can be split by hop count, and relation-collision counts are
recorded. Registered prediction: at +150, the 2-hop cell exceeds the 4-hop cell by >= .20.
"""
import json, os, re, string
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
VSA_DIM = 4096
POOLS = [0, 50, 150]
SEEDS = [42, 123, 456]
N = 300


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def triples_of(decomp):
    out = []
    for hop in decomp:
        q = hop.get("question", "")
        if ">>" not in q:
            return None
        subj, rel = q.split(">>", 1)
        subj, rel = subj.strip(), rel.strip()
        for m in set(re.findall(r"#(\d+)", subj)):
            k = int(m) - 1
            if not (0 <= k < len(out)):
                return None
            subj = subj.replace("#" + m, str(out[k][2]))
        ans = str(hop.get("answer", "")).strip()
        if not subj or not rel or not ans:
            return None
        out.append((subj, rel, ans))
    return out or None


def hrr_answer(gold, distractors, rng):
    ev = list(gold) + list(distractors)
    strings = sorted({s for tr in ev for s in tr})
    idx = {s: i for i, s in enumerate(strings)}
    V = torch.tensor(rng.standard_normal((len(strings), VSA_DIM)) / np.sqrt(VSA_DIM), dtype=torch.float32)
    Vn = V / (V.norm(dim=1, keepdim=True) + 1e-8)

    def bind(a, b):
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=VSA_DIM)

    def unbind(m, k):
        return torch.fft.irfft(torch.fft.rfft(m) * torch.conj(torch.fft.rfft(k)), n=VSA_DIM)

    mem = torch.zeros(VSA_DIM)
    for s, r, o in ev:
        mem = mem + bind(bind(V[idx[s]], V[idx[r]]), V[idx[o]])
    ent = sorted({tr[0] for tr in ev} | {tr[2] for tr in ev})
    ent_ids = torch.tensor([idx[e] for e in ent])
    cur = gold[0][0]
    for s, r, o in gold:
        hat = unbind(mem, bind(V[idx[cur]], V[idx[r]]))
        hat = hat / (hat.norm() + 1e-8)
        sims = Vn[ent_ids] @ hat
        sims = sims.masked_fill(~torch.tensor([e != cur for e in ent]), -1e9)
        cur = ent[int(sims.argmax())]
    return cur


def main():
    from datasets import load_dataset
    print("===== V-79a diagnostic: what drives the store's drop at +150? =====", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation", streaming=True)
    rows = []
    for r in ds:
        if not r.get("answerable", True):
            continue
        tr = triples_of(r.get("question_decomposition") or [])
        if tr is None:
            continue
        rows.append({"answer": r["answer"], "triples": tr})
        if len(rows) >= N:
            break
    hops = np.array([len(r["triples"]) for r in rows])
    print("instances %d | hop dist %s" % (len(rows), {int(h): int((hops == h).sum()) for h in sorted(set(hops))}),
          flush=True)

    all_tr = [(i, t) for i, r in enumerate(rows) for t in r["triples"]]
    out = {"n": len(rows), "hop_dist": {str(int(h)): int((hops == h).sum()) for h in sorted(set(hops))},
           "by_hop": {}, "collisions": {}}

    for k in POOLS:
        per_seed_hits, coll = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            hits, c = [], []
            for i, r in enumerate(rows):
                if k == 0:
                    dis = []
                else:
                    cand = [t for (j, t) in all_tr if j != i]
                    pick = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
                    dis = [cand[p] for p in pick]
                gold_rels = {norm(t[1]) for t in r["triples"]}
                c.append(sum(1 for t in dis if norm(t[1]) in gold_rels))
                pred = hrr_answer(r["triples"], dis, rng)
                hits.append(int(norm(pred) == norm(r["answer"])))
            per_seed_hits.append(np.array(hits))
            coll.append(float(np.mean(c)))
        H = np.stack(per_seed_hits)  # seeds x instances
        overall = float(H.mean())
        cells = {}
        for h in sorted(set(hops)):
            m = hops == h
            cells[str(int(h))] = {"n": int(m.sum()), "em": float(H[:, m].mean())}
        out["by_hop"]["p%d" % k] = {"overall": overall, "cells": cells}
        out["collisions"]["p%d" % k] = {"mean_distractors_sharing_a_gold_relation": float(np.mean(coll))}
        print("  pool+%-3d overall=%.4f | %s | mean distractors sharing a gold relation=%.1f"
              % (k, overall, " ".join("%sh n=%d em=%.4f" % (c, v["n"], v["em"]) for c, v in cells.items()),
                 float(np.mean(coll))), flush=True)
        json.dump(out, open("v79_logs/v79a.json", "w"), indent=1)

    c2 = out["by_hop"]["p150"]["cells"].get("2")
    c4 = out["by_hop"]["p150"]["cells"].get("4")
    print("", flush=True)
    print("===== V-79a SCORING =====", flush=True)
    if not c2 or not c4:
        out["verdict"] = "UNSCORABLE: a required hop cell is absent"
    else:
        gap = c2["em"] - c4["em"]
        p1 = gap >= 0.20
        out["verdict"] = ("P1 2-hop exceeds 4-hop by >=.20 at +150: %s (gap %.4f; 2h=%.4f n=%d, 4h=%.4f n=%d) "
                          "-> %s" % (p1, gap, c2["em"], c2["n"], c4["em"], c4["n"],
                          "CHAIN DEPTH is the dominant factor and the manuscript may say so" if p1 else
                          "DEPTH IS NOT THE STORY: the manuscript reports the drop WITHOUT a mechanism"))
    json.dump(out, open("v79_logs/v79a.json", "w"), indent=1)
    print("  >>> V-79a VERDICT: %s" % out["verdict"], flush=True)
    print("V79A-DONE", flush=True)


if __name__ == "__main__":
    main()
