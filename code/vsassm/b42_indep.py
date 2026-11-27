"""B42 second scorer (prereg/B42_label_free_discrete_prereg.md), written separately from b42_arms.py: numpy float64, its
own regex parse of the raw test text (mention order, hop structure, the gold answer by a dictionary walk), the HRR chain
written out per instance with numpy FFTs.

For every discrete final run (vq, dkvb, gumbel, st, dkvbor) and seed: the per-mention codes are rebuilt from the saved
assignments and codebook / values / table of the dev-selected epoch (b42_arms side file), the chain is recomputed,
the saved training threshold is applied, and the decisions are compared with the main run's per instance. For every
final run (all families, all readings): the accuracy is recomputed from the saved scores against the dictionary-walk
labels.

  python b42_indep.py --root out --out out/indep42.json [--arenas C] [--glob 'phi2_%s_vq.json']
  (shards are merged into out/indep42.json by the caller)
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402  (raw text only)

D = 2048
TOK = re.compile(r"\[([RXYFGH])([0-9A-F]{2})\]|\{(\d+)\}")


def cconv(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D)


def ccorr(ab, a):
    return np.fft.irfft(np.fft.rfft(ab) * np.conj(np.fft.rfft(a)), n=D)


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def parse(arena, text):
    """Mentions in the arms' order with kind and symbol, the hops as (key position, value position) lists, q, cand."""
    if arena == "C":
        qm = re.search(r"Q \[R([0-9A-F]{2})\] \{(\d+)\} \?", text)
        body = text[:qm.start()] + text[qm.end():]
        men, hops = [], [[], []]
        for m in re.finditer(r"\[([RF])([0-9A-F]{2})\]\{(\d+)\}", body):
            L = "RF".index(m.group(1))
            men += [("R", int(m.group(2), 16)), ("F", int(m.group(3)))]
            hops[L].append((len(men) - 2, len(men) - 1))
        men += [("R", int(qm.group(1), 16)), ("F", int(qm.group(2)))]
        return {"men": men, "hops": hops, "q": len(men) - 2, "cand": len(men) - 1}
    men = []
    for m in TOK.finditer(text):
        if m.group(3) is not None:
            men.append(("F", int(m.group(3))))
        else:
            men.append((m.group(1) if arena == "A" else "R", int(m.group(2), 16)))
    if arena == "B":
        n = len(men) - 2
        return {"men": men, "hops": [[(i, i + 1) for i in range(0, n, 2)]], "q": n, "cand": n + 1}
    hops = [[(i, i + 1) for i in range(0, 32, 2)], [(i, i + 1) for i in range(32, 64, 2)],
            [(i, i + 1) for i in range(64, 96, 2)]]
    return {"men": men, "hops": hops, "q": 96, "cand": 97}


def gold_label(p):
    sym = [s for _, s in p["men"]]
    k = sym[p["q"]]
    for hop in p["hops"]:
        k = {sym[a]: sym[b] for a, b in hop}[k]
    return int(k == sym[p["cand"]])


def chain_score(arena, p, K, P):
    V = K[:, P]
    key = K[p["q"]]
    nh = len(p["hops"])
    for h, hop in enumerate(p["hops"]):
        mem = sum(cconv(V[v], K[k]) for k, v in hop)
        hat = unit(ccorr(mem, key))
        if arena == "B" or (arena == "A" and h == nh - 1):
            return float(np.clip(np.dot(hat, unit(V[p["cand"]])), -1, 1))
        cs = [np.dot(unit(V[v]), hat) for _, v in hop]
        key = K[hop[int(np.argmax(cs))][1]]
    return float(np.clip(np.dot(unit(key), unit(K[p["cand"]])), -1, 1))


GROUPS = {"A": "RXYF", "B": "RF", "C": "R"}


def codes_for(fam, arena, side, n_inst, men_kinds):
    asg = side["assign_test"].numpy()
    if fam in ("gumbel", "st"):
        T = side["table"].double().numpy()
        return lambda i: T[asg[i]]
    if fam == "vq":
        T = np.concatenate([c.double().numpy() for c in side["codebook"]])
        return lambda i: T[asg[i]]
    Vv = side["values"].double().numpy()                                      # (H, tot, D)
    ko = side["key_off"]
    gid = np.array([0 if arena == "C" else GROUPS[arena].index(k) for k in men_kinds])

    def f(i):
        idx = asg[i]                                                          # (M, H)
        out = np.stack([Vv[h, np.array(ko)[gid] + idx[:, h]] for h in range(Vv.shape[0])]).mean(0)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--arenas", default="ABC")
    ap.add_argument("--glob", default="*_%s_*.json", help="final files to score (%%s = arena); for parallel shards")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for arena in a.arenas:
        texts = [s["text"] for s in S.build(arena, "test")]
        ps = [parse(arena, t) for t in texts]
        y = np.array([gold_label(p) for p in ps])
        for path in sorted(glob.glob(os.path.join(a.root, "final", a.glob % arena))):
            d = json.load(open(path))
            fam = d["args"]["fam"]
            name = os.path.basename(path)[:-5]
            side_p = path.replace(".json", ".side.pt")
            side = torch.load(side_p, weights_only=False) if os.path.exists(side_p) else {}
            readings = sorted(set(k.rsplit(":", 1)[0] for k in d["runs"][0]["epochs"][0] if k.endswith(":test")))
            for rd in readings:
                accs, accm, agr, dif = [], [], [], []
                for run in d["runs"]:
                    dv = [e["%s:dev" % rd]["acc"] for e in run["epochs"]]
                    j = max(range(len(dv)), key=lambda t: (dv[t], -t))
                    e = run["epochs"][j]["%s:test" % rd]
                    main_pred = (np.array(e["scores"]) >= e["thr"]).astype(int)
                    accm.append(float((main_pred == y).mean()))
                    if rd == "perm/named" and fam in ("vq", "dkvb", "gumbel", "st", "dkvbor") and run["seed"] in side:
                        assert run["dev_selected_epoch"] == j + 1
                        P = torch.randperm(D, generator=torch.Generator().manual_seed(run["seed"] + 7001)).numpy()
                        cf = codes_for(fam, arena, side[run["seed"]], len(ps), [k for k, _ in ps[0]["men"]])
                        # scores rounded to 6 decimals, the precision the main run saves: a snapped code compared with
                        # itself gives exactly 1 in float64 but 1 - 6e-8 in float32, and thresholds often sit there
                        sc = np.round(np.array([chain_score(arena, p, cf(i), P) for i, p in enumerate(ps)]), 6)
                        pred = (sc >= e["thr"]).astype(int)
                        accs.append(float((pred == y).mean()))
                        agr.append(float((pred == main_pred).mean()))
                        dif.append(float(np.abs(sc - np.array(e["scores"])).max()))
                r = {"acc_main_vs_dictionary_labels": float(np.mean(accm))}
                if accs:
                    r["acc_indep_chain"] = float(np.mean(accs)); r["agreement"] = float(min(agr))
                    r["max_abs_score_diff"] = float(max(dif))
                res["%s %s" % (name, rd)] = r
                print(name, rd, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
