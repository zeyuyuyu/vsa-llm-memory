"""B48 second scorer (prereg/B48_one_form_and_alias_naming_prereg.md), written apart from b42_arms.py / b48_alias.py:
numpy float64, its own regex parse of the raw text, gold answers by a dictionary walk over the parsed identities, the
HRR chain written out per instance with numpy FFTs.

  A  arena C1: for every discrete final (vq, dkvb, gumbel, st) and seed, the per-mention codes are rebuilt from the
     saved assignments and codebook / values / table (b42_arms side file), the chain is recomputed and the saved
     threshold applied: per-instance agreement with the main run; every final's accuracy from its saved scores against
     the dictionary-walk labels. Also the string-hash contrast: fixed random codes per written form, no learning, in
     arena C (two forms: hexadecimal role, decimal filler) and C1 (one form), threshold 0.5.
  B  alias stories: the statements are parsed from the text; listed names from the first line; the identity of a W
     alias from the Wikidata alias table (b32_data/entities.json; aliases are unambiguous after B32's filter), of a P
     pronoun from the name that opens its line. Each arm's keys are rebuilt (oracle: parsed identities; hash: the
     written strings; lm: the saved LM output, parsed again here; head: the saved predicted slots of the dev-selected
     epoch) and the chain recomputed; per-story agreement with b48_alias.py score, accuracy recomputed.

  python b48_indep.py A --root out --out out/indepA.json
  python b48_indep.py B --root out --out out/indepB.json
"""
import argparse, glob, hashlib, json, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

D = 2048
SEEDS = [42, 123, 456]


def cconv(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D)


def ccorr(ab, a):
    return np.fft.irfft(np.fft.rfft(ab) * np.conj(np.fft.rfft(a)), n=D)


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def P_of(seed):
    return torch.randperm(D, generator=torch.Generator().manual_seed(seed + 7001)).numpy()


def run_chain(hops, q, cand, K, P):
    """hops: [[(key pos, value pos)] level 0, [...] level 1]; K: (M, D) codes. Arena C: clean-up after each hop,
    score = cos(code of the hop-2 winner, code of the candidate). Returns score, winners."""
    V = K[:, P]
    key = K[q]
    wins = []
    for hop in hops:
        mem = sum(cconv(V[v], K[k]) for k, v in hop)
        hat = unit(ccorr(mem, key))
        cs = [np.dot(unit(V[v]), hat) for _, v in hop]
        w = hop[int(np.argmax(cs))][1]
        wins.append(w)
        key = K[w]
    return float(np.clip(np.dot(unit(key), unit(K[cand])), -1, 1)), wins


# ====================================================================== part A

def parse_c(text, one_form):
    """Mentions in the arms' order (level-0 pairs, level-1 pairs in text order, then query, candidate)."""
    rnum = r"(\d+)" if one_form else r"([0-9A-F]{2})"
    conv = (lambda s: int(s)) if one_form else (lambda s: int(s, 16))
    qm = re.search(r"Q \[R%s\] \{(\d+)\} \?" % rnum, text)
    body = text[:qm.start()] + text[qm.end():]
    men, hops = [], [[], []]
    for m in re.finditer(r"\[([RF])%s\]\{(\d+)\}" % rnum, body):
        L = "RF".index(m.group(1))
        men += [("R", conv(m.group(2)), m.group(2)), ("F", int(m.group(3)), m.group(3))]
        hops[L].append((len(men) - 2, len(men) - 1))
    men += [("R", conv(qm.group(1)), qm.group(1)), ("F", int(qm.group(2)), qm.group(2))]
    # level-0 pairs come first in the arms' order: re-sort the hops so that level 0 precedes level 1
    order = [i for L in (0, 1) for p in hops[L] for i in p] + [len(men) - 2, len(men) - 1]
    men2 = [men[i] for i in order]
    pos = {o: j for j, o in enumerate(order)}
    hops2 = [[(pos[k], pos[v]) for k, v in hops[L]] for L in (0, 1)]
    return {"men": men2, "hops": hops2, "q": len(men2) - 2, "cand": len(men2) - 1}


def gold(p):
    sym = [s for _, s, _ in p["men"]]
    k = sym[p["q"]]
    for hop in p["hops"]:
        k = {sym[a]: sym[b] for a, b in hop}[k]
    return int(k == sym[p["cand"]])


def codes_for(fam, side):
    asg = side["assign_test"].numpy()
    if fam in ("gumbel", "st"):
        T = side["table"].double().numpy()
        return lambda i: T[asg[i]]
    if fam == "vq":
        T = np.concatenate([c.double().numpy() for c in side["codebook"]])
        return lambda i: T[asg[i]]
    Vv = side["values"].double().numpy()
    ko = np.array(side["key_off"])

    def f(i):
        idx = asg[i]
        out = np.stack([Vv[h, ko[0] + idx[:, h]] for h in range(Vv.shape[0])]).mean(0)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return f


def string_hash_codes(p, seed):
    """One fixed random unit vector per written form (kind letter is not part of the key: '63' as role and as filler
    is one form in C1; in C the role is written '3F' and the filler '63', two forms)."""
    out = []
    for kind, sym, written in p["men"]:
        h = int(hashlib.md5(("%d|%s" % (seed, written)).encode()).hexdigest()[:15], 16)
        out.append(unit(np.random.default_rng(h).standard_normal(D)))
    return np.stack(out)


def part_a(a):
    import b31_struct as S31
    import b48_struct as S48
    res = {}
    texts = {"C1": [s["text"] for s in S48.build_C1("test")], "C": [s["text"] for s in S31.build("C", "test")]}
    ps = {k: [parse_c(t, k == "C1") for t in v] for k, v in texts.items()}
    ys = {k: np.array([gold(p) for p in v]) for k, v in ps.items()}
    # the structure parsed here must equal b48_struct's (symbol per mention, same order)
    st1 = S48.build_C1("test")
    res["C1_parse_equals_struct"] = int(sum([m[2] for m in s["mentions"]] == [x[1] for x in p["men"]]
                                            for s, p in zip(st1, ps["C1"])))
    res["C1_labels_equal"] = int(sum(int(s["label"]) == y for s, y in zip(st1, ys["C1"])))
    for k in ("C", "C1"):
        accs = []
        for seed in SEEDS:
            P = P_of(seed)
            sc = np.array([run_chain(p["hops"], p["q"], p["cand"], string_hash_codes(p, seed), P)[0] for p in ps[k]])
            accs.append(float(((sc >= 0.5).astype(int) == ys[k]).mean()))
        res["string_hash_%s" % k] = accs
        print("string hash", k, accs, flush=True)
    y = ys["C1"]
    for path in sorted(glob.glob(os.path.join(a.root, "final", "*_C1_*.json"))):
        d = json.load(open(path))
        name = os.path.basename(path)[:-5]
        if "runs" not in d:
            continue
        if "test_scores" in d["runs"][0]:                                    # probe
            accm = [float((((np.array(r["test_scores"]) > 0).astype(int)) == y).mean()) for r in d["runs"]]
            res[name] = {"acc_main_vs_dictionary_labels": accm}
            print(name, json.dumps(res[name]), flush=True)
            continue
        fam = d["args"].get("fam", d["args"].get("arm"))
        side_p = path.replace(".json", ".side.pt")
        side = torch.load(side_p, weights_only=False) if os.path.exists(side_p) else {}
        readings = sorted(set(k.rsplit(":", 1)[0] for k in d["runs"][0]["epochs"][0] if k.endswith(":test")))
        for rd in readings:
            accm, accs, agr, dif = [], [], [], []
            for run in d["runs"]:
                dv = [e["%s:dev" % rd]["acc"] for e in run["epochs"]]
                j = max(range(len(dv)), key=lambda t: (dv[t], -t))
                e = run["epochs"][j]["%s:test" % rd]
                main_pred = (np.array(e["scores"]) >= e["thr"]).astype(int)
                accm.append(float((main_pred == y).mean()))
                if rd == "perm/named" and fam in ("vq", "dkvb", "gumbel", "st") and run["seed"] in side:
                    assert run["dev_selected_epoch"] == j + 1
                    P = P_of(run["seed"])
                    cf = codes_for(fam, side[run["seed"]])
                    sc = np.round(np.array([run_chain(p["hops"], p["q"], p["cand"], cf(i), P)[0]
                                            for i, p in enumerate(ps["C1"])]), 6)
                    pred = (sc >= e["thr"]).astype(int)
                    accs.append(float((pred == y).mean())); agr.append(float((pred == main_pred).mean()))
                    dif.append(float(np.abs(sc - np.array(e["scores"])).max()))
            r = {"acc_main_vs_dictionary_labels": accm}
            if accs:
                r.update({"acc_indep_chain": accs, "agreement_min": min(agr), "max_abs_score_diff": max(dif)})
            res["%s %s" % (name, rd)] = r
            print(name, rd, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


# ====================================================================== part B

def nrm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def alias_table():
    d = json.load(open(os.path.join(HERE, "b32_data", "entities.json")))["entities"]
    t = {}
    for e in d:
        for x in [e["name"]] + e["aliases"]:
            t.setdefault(nrm(x), set()).add(e["uri"].rsplit("/", 1)[-1])
    return t


def parse_story(text, kind, at):
    """Listed names, statements (subject string, object string, subject identity slot, object slot) per level, the
    question's two names, gold label."""
    lines = text.split("\n")
    head = lines[0]
    assert head.startswith("Entities: " if kind == "W" else "Characters: ") and head.endswith(".")
    names = head.split(": ", 1)[1][:-1].split("; ")
    slot = {n: j for j, n in enumerate(names)}
    lv = [[], []]
    for ln in lines[1:-1]:
        if kind == "W":
            m = re.fullmatch(r"(.+) is linked to (.+)\.", ln)
            if m and m.group(1) in slot and m.group(2) in slot:
                lv[0].append((m.group(1), m.group(2), slot[m.group(1)], slot[m.group(2)])); continue
            m = re.fullmatch(r"(.+) is paired with (.+)\.", ln)
            assert m and m.group(2) in slot, ln
            ids = at.get(nrm(m.group(1)), set())
            hit = [j for j, n in enumerate(names) if at.get(nrm(n), set()) & ids]
            assert len(hit) == 1, (ln, hit)
            lv[1].append((m.group(1), m.group(2), hit[0], slot[m.group(2)]))
        else:
            m = re.fullmatch(r"(\w+) called (\w+)\.", ln)
            if m:
                lv[0].append((m.group(1), m.group(2), slot[m.group(1)], slot[m.group(2)])); continue
            m = re.fullmatch(r"(\w+) went to the \w+\. Then (he|she) visited (\w+)\.", ln)
            assert m, ln
            lv[1].append((m.group(2), m.group(3), slot[m.group(1)], slot[m.group(3)]))
    qm = re.fullmatch(r"Question: is the entity that (.+) is linked to paired with (.+)\?", lines[-1]) if kind == "W" \
        else re.fullmatch(r"Question: did the person whom (\w+) called visit (\w+)\?", lines[-1])
    q, c = qm.group(1), qm.group(2)
    d0 = {a: b for _, _, a, b in lv[0]}; d1 = {a: b for _, _, a, b in lv[1]}
    y = int(d1[d0[slot[q]]] == slot[c])
    return {"names": names, "lv": lv, "q": q, "cand": c, "y": y, "qs": slot[q], "cs": slot[c]}


def key_code(key, seed, cache):
    if key not in cache:
        h = int(hashlib.md5(("%d|%s" % (seed, key)).encode()).hexdigest()[:15], 16)
        cache[key] = unit(np.random.default_rng(h).standard_normal(D))
    return cache[key]


def lm_parse(out):
    res = {}
    for ln in out.splitlines():
        m = re.match(r"^\s*\[?(\d+)\]?\s*[.:)]?\s*(.*?)\s*\|\s*(.*?)\s*$", ln)
        if m and 1 <= int(m.group(1)) <= 28 and int(m.group(1)) not in res:
            res[int(m.group(1))] = tuple(x.strip().strip("<>").strip() for x in (m.group(2), m.group(3)))
    return res


def part_b(a):
    import b48_struct as S48
    at = alias_table()
    res = {}
    for kind in "WP":
        st = S48.build_B(kind, a.split)
        ps = [parse_story(s["text"], kind, at) for s in st]
        y = np.array([p["y"] for p in ps])
        ents = [s["ents"] for s in st]
        main_p = os.path.join(a.root, "b", "scores_%s_%s.json" % (kind, a.split))
        if not os.path.exists(main_p):
            print("missing", main_p); continue
        main = json.load(open(main_p))
        res["%s_labels_equal_struct" % kind] = int(sum(p["y"] == s["label"] for p, s in zip(ps, st)))
        lm_p = os.path.join(a.root, "lm", "%s_%s.json" % (kind, a.split))
        lm = json.load(open(lm_p))["stories"] if os.path.exists(lm_p) else None
        head_p = os.path.join(a.root, "b", "final", "%s_head.json" % kind)
        head = json.load(open(head_p)) if os.path.exists(head_p) else None
        for arm in main["arms"]:
            for si, seed in enumerate(SEEDS):
                cache = {}
                P = P_of(seed)
                if arm == "head":
                    run = [r for r in head["runs"] if r["seed"] == seed][0]
                    pred = run["epochs"][run["dev_selected_epoch"] - 1][a.split]["pred"]
                sc = []
                for i, p in enumerate(ps):
                    # mention order of the chain: [l0 subj, l0 obj]*16, [l1 subj, l1 obj]*12, q, cand (own order)
                    keys = []
                    for L in (0, 1):
                        for j, (sa, sb, ia, ib) in enumerate(p["lv"][L]):
                            if arm == "oracle":
                                keys += ["ent:" + ents[i][ia]["id"], "ent:" + ents[i][ib]["id"]]
                            elif arm == "hash":
                                keys += ["str:" + nrm(sa), "str:" + nrm(sb)]
                            elif arm == "lm":
                                w = lm_parse(lm[str(i)]["out"]).get(16 * L + j + 1)
                                for pos in (0, 1):
                                    keys.append("str:" + nrm(w[pos]) if w and nrm(w[pos]) else "miss:%d:%d:%d:%d" % (i, L, j, pos))
                    if arm == "oracle":
                        keys += ["ent:" + ents[i][p["qs"]]["id"], "ent:" + ents[i][p["cs"]]["id"]]
                    elif arm in ("hash", "lm"):
                        keys += ["str:" + nrm(p["q"]), "str:" + nrm(p["cand"])]
                    else:
                        # head: predicted slot per mention in b48_struct's mention order -> rebuild that order here:
                        # roster (16), then per level-0 line (subj, obj), per level-1 line ((ante,) subj, obj), q, cand
                        pr = pred[i]
                        k0 = 16
                        for L in (0, 1):
                            for j in range(len(p["lv"][L])):
                                if kind == "P" and L == 1:
                                    k0 += 1                                   # the antecedent mention
                                keys += ["ent:" + ents[i][pr[k0]]["id"], "ent:" + ents[i][pr[k0 + 1]]["id"]]
                                k0 += 2
                        keys += ["ent:" + ents[i][pr[k0]]["id"], "ent:" + ents[i][pr[k0 + 1]]["id"]]
                    K = np.stack([key_code(k, seed, cache) for k in keys])
                    n0, n1 = len(p["lv"][0]), len(p["lv"][1])
                    hops = [[(2 * j, 2 * j + 1) for j in range(n0)], [(2 * n0 + 2 * j, 2 * n0 + 2 * j + 1) for j in range(n1)]]
                    s, _ = run_chain(hops, len(keys) - 2, len(keys) - 1, K, P)
                    sc.append(round(s, 6))
                pred_i = (np.array(sc) >= 0.5).astype(int)
                mrow = main["arms"][arm][si]
                pred_m = (np.array(mrow["scores"]) >= 0.5).astype(int)
                r = {"acc_indep": float((pred_i == y).mean()), "acc_main_recomputed": float((pred_m == y).mean()),
                     "acc_main_logged": mrow["acc"], "agreement": float((pred_i == pred_m).mean()),
                     "max_abs_score_diff": float(np.abs(np.array(sc) - np.array(mrow["scores"])).max())}
                res["%s %s seed %d" % (kind, arm, seed)] = r
                print(kind, arm, seed, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["A", "B"])
    ap.add_argument("--root", default="out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    part_a(a) if a.part == "A" else part_b(a)


if __name__ == "__main__":
    main()
