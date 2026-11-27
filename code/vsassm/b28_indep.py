"""B28 second implementation of the exact-symbol (parser) arms, in numpy float64, written separately from
b28_common / b28_babi / b28_arenaC (prereg/B28_noncommutative_binding_prereg.md, "Checks from other angles").

1. Same codebooks as the main runs (regenerated from the same torch generators): per-question predictions under
   every encoding, compared with the main run's per-question predictions (agreement rate per cell).
2. Fresh codebooks (20 per setting and encoding, numpy RNG): the exact-symbol accuracy under each encoding, so
   that no conclusion rests on the three published codebooks.

bAbI structures (gold symbol ids) are read from the main run's per-question files; arena C instances are
regenerated here from a re-implementation of v13.Arena's random draws (no tokenizer), and checked against the
main run's gold dump.

  python b28_indep.py --runs <dir with b28_task*.json and b28_C_test_parser_*.json> --out results/b28_indep.json
"""
import argparse, json, os, random

import numpy as np

D = 2048
ENC = ["comm", "perm", "roles", "rtag", "shift"]
OFFS = {"perm": 7001, "roles": 7002, "rtag_k": 7003, "rtag_v": 7004}
INVD = {"north": "south", "south": "north", "east": "west", "west": "east"}
DIRS = ["north", "south", "east", "west"]


def nbind(a, b):
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1), n=D, axis=-1)


def nunbind(ab, a):
    return np.fft.irfft(np.fft.rfft(ab, axis=-1) * np.conj(np.fft.rfft(a, axis=-1)), n=D, axis=-1)


def rownorm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosv(M, v):
    return (M @ v) / (np.linalg.norm(M, axis=1) * np.linalg.norm(v) + 1e-30)


def torch_rows(n, seed):
    import torch
    g = torch.Generator(); g.manual_seed(seed)
    x = torch.randn(n, D, generator=g).double().numpy()
    return rownorm(x)


def torch_perm(seed):
    import torch
    return torch.randperm(D, generator=torch.Generator().manual_seed(seed)).numpy()


def codes(name, sym, seed=None, rng=None):
    """(K, V) code tables. seed given: the main runs' private generators; rng given: fresh draws."""
    n = sym.shape[0]
    if name == "comm":
        return sym, sym
    if name == "perm":
        P = torch_perm(seed + OFFS["perm"]) if rng is None else rng.permutation(D)
        return sym, sym[:, P]
    if name == "roles":
        V = torch_rows(n, seed + OFFS["roles"]) if rng is None else rownorm(rng.standard_normal((n, D)))
        return sym, V
    if name == "rtag":
        if rng is None:
            rk, rv = torch_rows(1, seed + OFFS["rtag_k"])[0], torch_rows(1, seed + OFFS["rtag_v"])[0]
        else:
            rk, rv = rownorm(rng.standard_normal(D)), rownorm(rng.standard_normal(D))
        return rownorm(nbind(np.broadcast_to(rk, sym.shape), sym)), rownorm(nbind(np.broadcast_to(rv, sym.shape), sym))
    if name == "shift":
        return sym, np.roll(sym, 1, axis=1)
    raise ValueError(name)


# ------------------------------------------------------------------ the three bAbI forwards (gold ids)

def f4(g, K, V, sym, n_r):
    dv = {d: sym[n_r + i] for i, d in enumerate(DIRS)}
    mem = np.zeros(D); ment = []
    for a, d, b in g["stmts"]:
        mem += nbind(V[a], nbind(dv[d], K[b]))
        mem += nbind(V[b], nbind(dv[INVD[d]], K[a]))
        ment += [a, b]
    got = nunbind(mem, nbind(dv[g["qdir"]], K[g["qroom"]]))
    return ment[int(np.argmax(cosv(V[ment], got)))]


def f15(g, K, V, n_an):
    mem = g["mem"]; cls = g["cls"]
    m1 = sum(nbind(V[c], K[n_an + e]) for e, c in mem)
    h = nunbind(m1, K[n_an + g["q"]])
    cc = [c for _, c in mem]
    c1 = cc[int(np.argmax(cosv(V[cc], h)))]
    m2 = sum(nbind(V[p], K[c]) for c, p in cls)
    got = nunbind(m2, K[c1])
    pp = [p for _, p in cls]
    return pp[int(np.argmax(cosv(V[pp], got)))]


def f2(g, K, V, P, L, O):
    m1 = sum(nbind(V[P + h], K[O + o]) for o, h in g["obj_ev"])
    hh = nunbind(m1, K[O + g["q_obj"]])
    hs = [P + h for _, h in g["obj_ev"]]
    h1 = hs[int(np.argmax(cosv(V[hs], hh)))]
    m2 = sum(nbind(V[L + l], K[P + p]) for p, l in g["moves"])
    got = nunbind(m2, K[h1])
    ls = [l for _, l in g["moves"]]
    return ls[int(np.argmax(cosv(V[[L + l for l in ls]], got)))]


# ------------------------------------------------------------------ arena C, regenerated

def arena_c(n, seed, N_SYM=64, HOPS=2, N_CTX=32):
    rng = random.Random(seed); rows = []
    for _ in range(n):
        chain = [rng.randrange(N_SYM) for _ in range(HOPS + 1)]
        levels = []
        for L in range(HOPS):
            pairs, seen = [(chain[L], chain[L + 1])], {chain[L]}
            while len(pairs) < N_CTX:
                a = rng.randrange(N_SYM)
                if a in seen:
                    continue
                seen.add(a); pairs.append((a, rng.randrange(N_SYM)))
            rng.shuffle(pairs); levels.append(pairs)
        label = 1 if rng.random() < 0.5 else 0
        cand = chain[-1] if label == 1 else rng.choice([s for s in range(N_SYM) if s != chain[-1]])
        rows.append({"q": chain[0], "s1": chain[1], "s2": chain[2], "cand": cand, "label": label, "ctx": levels})
    return rows


def fC(g, K, V):
    key = K[g["q"]]; hops = []
    for pairs in g["ctx"]:
        mem = sum(nbind(V[b], K[a]) for a, b in pairs)
        hat = nunbind(mem, key)
        fl = [b for _, b in pairs]
        w = fl[int(np.argmax(cosv(V[fl], hat)))]
        hops.append(w); key = K[w]
    return float(np.clip(key @ K[g["cand"]], -1, 1)), hops


def c_acc(tr, te, K, V):
    st = np.array([fC(g, K, V)[0] for g in tr]); yt = np.array([g["label"] for g in tr])
    out = [fC(g, K, V) for g in te]
    se = np.array([o[0] for o in out]); ye = np.array([g["label"] for g in te])
    c = np.unique(st)
    thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))])
    pred = (se >= thr).astype(int)
    return pred, (pred == ye).astype(int), [o[1] for o in out], thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", type=int, default=20)
    a = ap.parse_args()
    R = {"same_codebook": {}, "fresh_codebooks": {}, "arenaC_regen_check": None}
    seeds = [42, 123, 456]

    for task in (4, 15, 2):
        p = os.path.join(a.runs, "b28_task%d.json" % task)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        gold, TOT = d["gold"], d["codebook"]
        if task == 4:
            n_r = TOT - 4
            fwd = lambda g, K, V, sym: f4(g, K, V, sym, n_r)
        elif task == 15:
            n_an = len(d["vocab"])
            fwd = lambda g, K, V, sym: f15(g, K, V, n_an)
        else:
            n_p = len(d["persons"]); n_l = len(d["vocab"])
            fwd = lambda g, K, V, sym: f2(g, K, V, 0, n_p, n_p + n_l)
        for enc in ENC:
            for s in seeds:
                key = "parser/%s/%d" % (enc, s)
                if key not in d["runs"]:
                    continue
                sym = torch_rows(TOT, s + 11)
                K, V = codes(enc, sym, seed=s)
                pred = [fwd(g, K, V, sym) for g in gold]
                main_pred = [o["pred"] for o in d["runs"][key]["per_q"]]
                corr = [int(p_ == g["label"]) for p_, g in zip(pred, gold)]
                R["same_codebook"]["task%d/%s/%d" % (task, enc, s)] = {
                    "acc_indep": float(np.mean(corr)), "acc_main": d["runs"][key]["acc_scored"],
                    "agree": float(np.mean([int(x == y) for x, y in zip(pred, main_pred)])),
                    "disagree_idx": [i for i, (x, y) in enumerate(zip(pred, main_pred)) if x != y][:50]}
                print("task %d %-5s seed %d indep %.4f main %.4f agree %.4f" % (
                    task, enc, s, np.mean(corr), d["runs"][key]["acc_scored"],
                    R["same_codebook"]["task%d/%s/%d" % (task, enc, s)]["agree"]), flush=True)
        if task in (4, 15):
            for enc in ENC:
                accs = []
                for j in range(a.fresh):
                    rng = np.random.default_rng(10_000 * task + 100 * ENC.index(enc) + j)
                    sym = rownorm(rng.standard_normal((TOT, D)))
                    K, V = codes(enc, sym, rng=rng)
                    accs.append(float(np.mean([int(fwd(g, K, V, sym) == g["label"]) for g in gold])))
                R["fresh_codebooks"]["task%d/%s" % (task, enc)] = accs
                print("task %d %-5s fresh x%d: mean %.4f min %.4f max %.4f" % (
                    task, enc, a.fresh, np.mean(accs), min(accs), max(accs)), flush=True)

    # arena C
    tr, te = arena_c(800, 42), arena_c(500, 4242)
    pc = os.path.join(a.runs, "b28_C_test_parser_comm.json")
    if os.path.exists(pc):
        dg = json.load(open(pc))["gold"]
        same = all(g["q"] == h["q"] and g["s1"] == h["s1"] and g["s2"] == h["s2"] and g["cand"] == h["cand"]
                   and g["label"] == h["label"] and [list(map(tuple, l)) for l in h["ctx"]] == g["ctx"]
                   for g, h in zip(te, dg))
        R["arenaC_regen_check"] = bool(same and len(dg) == len(te))
        print("arena C regenerated instances identical to the main run's gold dump: %s" % R["arenaC_regen_check"])
    for enc in ENC:
        pe = os.path.join(a.runs, "b28_C_test_parser_%s.json" % enc)
        main_runs = {r["seed"]: r for r in json.load(open(pe))["runs"]} if os.path.exists(pe) else {}
        for s in seeds:
            sym = torch_rows(64, s + 31)
            K, V = codes(enc, sym, seed=s)
            pred, corr, hops, thr = c_acc(tr, te, K, V)
            rec = {"acc_indep": float(corr.mean()), "thr": thr,
                   "retrieval_indep": float(np.mean([int(h == [g["s1"], g["s2"]]) for h, g in zip(hops, te)]))}
            if s in main_runs:
                mp = main_runs[s]["epochs"][-1]["pred"]
                rec["acc_main"] = main_runs[s]["acc_best"]
                rec["agree"] = float(np.mean([int(x == y) for x, y in zip(pred.tolist(), mp)]))
            R["same_codebook"]["C/%s/%d" % (enc, s)] = rec
            print("C %-5s seed %d indep %.4f main %s agree %s" % (enc, s, rec["acc_indep"], rec.get("acc_main"),
                                                                 rec.get("agree")), flush=True)
        accs = []
        for j in range(a.fresh):
            rng = np.random.default_rng(900_000 + 100 * ENC.index(enc) + j)
            sym = rownorm(rng.standard_normal((64, D)))
            K, V = codes(enc, sym, rng=rng)
            accs.append(float(c_acc(tr, te, K, V)[1].mean()))
        R["fresh_codebooks"]["C/%s" % enc] = accs
        print("C %-5s fresh x%d: mean %.4f min %.4f max %.4f" % (enc, a.fresh, np.mean(accs), min(accs),
                                                                max(accs)), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
    print("B28-INDEP-DONE -> %s" % a.out)


if __name__ == "__main__":
    main()
