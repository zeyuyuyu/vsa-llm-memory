"""B41 G0 follow-up: for VSA configs whose second-implementation answers agree on < 99% of (query, h), list every
disagreement with the score gap between the two answers (in units of the score standard deviation over the
codebook) and whether either answer is correct. A disagreement at a gap of ~0 is a tie broken differently
(torch.topk on float32 vs numpy stable argsort on float64), not a different computation.

  python b41_g0_ties.py --indep_dir /path/to/work/b41/indep --runs /path/to/work/b41/runs \
      --logits /path/to/work/b41/head/test_logits.npy --out /path/to/work/b41/results/b41_g0_ties.json
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_indep as I                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indep_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for f in sorted(glob.glob(os.path.join(a.indep_dir, "*_indep.json"))):
        d = json.load(open(f))
        flagged = [k for k, v in d.items() if v["answer_agree"] is not None and v["answer_agree"] < 0.99]
        if not flagged:
            continue
        run = os.path.join(a.runs, os.path.basename(f).replace("_indep.json", ".npz"))
        C = I.Ctx(run, a.logits)
        Z = np.load(run)
        seed = C.meta["seed"]
        for name in flagged:
            _, mode, Ds, kbs = name.split("_")
            D, kb = int(Ds[1:]), int(kbs[2:])
            c = 0 if mode == "END" else int(mode[3:])
            PHI, P = I.codebook(seed, D)
            cache = {}

            def mem(key):
                if key not in cache:
                    fl = C.slot.get(key, [])
                    if not fl:
                        cache[key] = None
                    else:
                        m = np.zeros(D)
                        for ff in fl:
                            cs = np.zeros(D); co = np.zeros(D)
                            for e, w in C.TK("s", ff, kb).items():
                                cs += w * PHI[e]
                            for e, w in C.TK("o", ff, kb).items():
                                co += w * PHI[e]
                            m += cs * co[P]
                        cache[key] = m
                return cache[key]
            dis = []
            for q in range(150):
                tr = {}
                r = I.vsa(C, q, D, kb, c, PHI, P, mem, trace=tr)
                for hi, h in enumerate(I.HOUT):
                    ai, ae = r[h][0], int(Z["%s|answer" % name][q, hi])
                    if ai != ae:
                        sc = tr.get(h)
                        gap = None
                        if sc is not None and ae >= 0 and ai >= 0:
                            gap = float(abs(sc[ai] - sc[ae]) / (sc.std() + 1e-12))
                        truth = int(Z["_answers"][q, hi])
                        # earliest hop <= h whose top-2 (or c-th/(c+1)-th) scores are exactly tied (a tie broken differently there
                        # sends the two implementations down different trajectories)
                        tie_hop = None
                        for hh in range(1, h + 1):
                            s_ = tr.get(hh)
                            if s_ is None:
                                continue
                            o_ = np.sort(s_)[::-1]
                            cc = max(c, 1)                       # top-1 decides the answer; top-c the re-bundle
                            if abs(o_[0] - o_[1]) <= 1e-9 * max(1.0, abs(o_[0])) or \
                                    abs(o_[cc - 1] - o_[cc]) <= 1e-9 * max(1.0, abs(o_[cc - 1])):
                                tie_hop = hh
                                break
                        dis.append({"q": q, "h": h, "indep": ai, "engine": ae, "gap_sd": gap, "tie_hop": tie_hop,
                                    "indep_correct": ai == truth, "engine_correct": ae == truth})
            gaps = [x["gap_sd"] for x in dis if x["gap_sd"] is not None]
            rep["%s|%s" % (os.path.basename(run), name)] = {
                "n_disagree": len(dis), "max_gap_sd": max(gaps) if gaps else None,
                "engine_correct": sum(x["engine_correct"] for x in dis),
                "indep_correct": sum(x["indep_correct"] for x in dis), "items": dis}
            rep["%s|%s" % (os.path.basename(run), name)]["with_exact_tie_at_or_before_h"] = \
                sum(x["tie_hop"] is not None for x in dis)
            print(os.path.basename(run), name, len(dis), "with exact top-2 tie at/before h:",
                  sum(x["tie_hop"] is not None for x in dis), "max gap (sd)", max(gaps) if gaps else None,
                  "engine/indep correct among them", sum(x["engine_correct"] for x in dis),
                  sum(x["indep_correct"] for x in dis), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
