"""B31 second implementation of the exact-symbol arms (prereg/B31_encoder_equalised_baselines_prereg.md), written
separately from b31_arms.py: numpy float64, regex parse of the raw text (not b31_struct's mention table), the HRR
chain written out per instance with numpy FFTs.

For every arena and seed: the dictionary arm, and exact codebook rows under comm and perm with the closed-form
threshold fitted on the training split. Codebooks are the published ones (drawn with the same torch generators, then
cast to float64). Writes per-instance decisions for the test split, so that the verdict can compare them with the main
runs, plus 20 fresh codebooks per arena and encoding (seeds 1000..1019).

  python b31_indep.py --out /path/to/work/b31/out/indep.json [--splits train test]
"""
import argparse, json, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402

D = 2048
RX = re.compile(r"\[([RXY])([0-9A-F]{2})\]")


def parse(arena, text):
    """Independent regex parse of the raw text into the gold chain (hand-written, separate from b31_struct)."""
    if arena == "A":
        a, rest = text[2:].split(" B ", 1); b, rest = rest.split(" C ", 1); c, q = rest.split(" Q ", 1)
        ma, mb = RX.findall(a), RX.findall(b)
        A = [(int(r[1], 16), int(x[1], 16)) for r, x in zip(ma[0::2], ma[1::2])]
        B = [(int(x[1], 16), int(y[1], 16)) for x, y in zip(mb[0::2], mb[1::2])]
        C = [(int(y, 16), int(f)) for y, f in re.findall(r"\[Y([0-9A-F]{2})\] \{(\d+)\}", c)]
        mq = re.match(r"\[R([0-9A-F]{2})\] \{(\d+)\}", q)
        return {"hops": [A, B, C], "q": int(mq.group(1), 16), "cand": int(mq.group(2))}
    if arena == "B":
        ps = [(int(r, 16), int(f)) for r, f in re.findall(r"\[R([0-9A-F]{2})\]\{(\d+)\}", text)]
        return {"hops": [ps[:-1]], "q": ps[-1][0], "cand": ps[-1][1]}
    ctx = [[], []]
    for L, a_, b_ in re.findall(r"\[([RF])([0-9A-F]{2})\]\{(\d+)\}", text):
        ctx["RF".index(L)].append((int(a_, 16), int(b_)))
    mq = re.search(r"Q \[R([0-9A-F]{2})\] \{(\d+)\} \?", text)
    return {"hops": ctx, "q": int(mq.group(1), 16), "cand": int(mq.group(2))}


def dict_pred(arena, p):
    k = p["q"]
    for hop in p["hops"]:
        k = dict(hop)[k]
    return int(k == p["cand"])


def books(arena, seed):
    def u(n, s, eps):
        g = torch.Generator(); g.manual_seed(s)
        v = torch.randn(n, D, generator=g)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8) if eps else torch.nn.functional.normalize(v, dim=1)
        return v.double().numpy()
    if arena == "A":
        torch.manual_seed(seed)
        r = torch.randn(64, D); r = (r / r.norm(dim=-1, keepdim=True)).double().numpy()
        return {"key": [r, u(256, seed + 11, 1), u(256, seed + 23, 1)], "val": [u(256, seed + 11, 1),
                u(256, seed + 23, 1), u(256, seed + 37, 1)]}
    if arena == "B":
        return {"key": [u(64, seed + 11, 0)], "val": [u(256, seed + 23, 0)]}
    s_ = u(64, seed + 31, 0)
    return {"key": [s_, s_], "val": [s_, s_]}


def cconv(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D)


def ccorr(ab, a):
    return np.fft.irfft(np.fft.rfft(ab) * np.conj(np.fft.rfft(a)), n=D)


def unitv(x):
    return x / np.linalg.norm(x)


def vsa_score(arena, p, bk, P):
    """Exact rows; value side permuted by P (None = comm). Clean-up after every hop but the last in A; after every
    hop in C (score = cosine of the last winner with the candidate); none in B."""
    Vf = (lambda v: v[P]) if P is not None else (lambda v: v)
    key = bk["key"][0][p["q"]]
    nh = len(p["hops"])
    for h, hop in enumerate(p["hops"]):
        Kb, Vb = bk["key"][h], bk["val"][h]
        mem = sum(cconv(Vf(Vb[v]), Kb[k]) for k, v in hop)
        hat = unitv(ccorr(mem, key))
        if arena == "B" or (arena == "A" and h == nh - 1):
            return float(np.dot(hat, unitv(Vf(Vb[p["cand"]]))))
        cands = [unitv(Vf(Vb[v])) for _, v in hop]
        w = int(np.argmax([np.dot(c, hat) for c in cands]))
        nxt = hop[w][1]
        key = (bk["key"][h + 1][nxt] if arena == "A" else Kb[nxt])
    return float(np.dot(unitv(key), unitv(bk["key"][0][p["cand"]])))


def thr_fit(st, yt):
    c = np.unique(st)
    accs = [(((st >= t) == (yt == 1)).mean()) for t in c]
    return float(c[int(np.argmax(accs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", type=int, default=20)
    a = ap.parse_args()
    res = {}
    for arena in S.ARENAS:
        P_ = {s: torch.randperm(D, generator=torch.Generator().manual_seed(s + 7001)).numpy() for s in
              [42, 123, 456] + list(range(1000, 1000 + a.fresh))}
        data = {}
        for sp in ("train", "test"):
            st = S.build(arena, sp)
            data[sp] = ([parse(arena, s["text"]) for s in st], np.array([s["label"] for s in st]))
        pt, yt = data["train"]; pe, ye = data["test"]
        r = {"dict_test_pred": [dict_pred(arena, p) for p in pe]}
        r["dict_test_acc"] = float((np.array(r["dict_test_pred"]) == ye).mean())
        r["dict_train_acc"] = float((np.array([dict_pred(arena, p) for p in pt]) == yt).mean())
        for seed in [42, 123, 456] + list(range(1000, 1000 + a.fresh)):
            bk = books(arena, seed)
            for enc in ("comm", "perm"):
                P = P_[seed] if enc == "perm" else None
                strn = np.array([vsa_score(arena, p, bk, P) for p in pt])
                ste = np.array([vsa_score(arena, p, bk, P) for p in pe])
                thr = thr_fit(strn, yt)
                pred = (ste >= thr).astype(int)
                k = "%s/%d" % (enc, seed)
                r[k] = {"acc": float((pred == ye).mean()), "thr": thr}
                if seed in (42, 123, 456):
                    r[k]["pred"] = pred.tolist()
                print(arena, k, "%.4f" % r[k]["acc"], flush=True)
        res[arena] = r
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
