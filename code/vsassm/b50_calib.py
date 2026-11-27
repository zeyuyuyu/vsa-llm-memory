"""B50 dev calibration (prereg/B50_ood_gates_cot_budget_prereg.md, 'Matching rule'). Dev split only.

Inputs (live dev runs on the box): preds/dev/vsa_s{s}.jsonl (B40's VSA seed s, unchanged) and
preds/dev/discopen_s{s}.jsonl (B40's DISC seed s with the gate forced open, every score logged).
Rule: k_s = number of the 800 dev items on which VSA seed s's gate opens; for each gate g and DISC seed s,
tau_{g,s} = the k_s-th largest dev score of g; the gate opens iff score >= tau. Sensitivity thresholds (secondary) at
dev open counts round(0.99 * 800) and round(0.95 * 800).
usage: python b50_calib.py <b50_root>   -> <b50_root>/out/thresholds.json (+ a printed dev summary)
"""
import json, sys
import numpy as np

SEEDS = (0, 1, 2)
GATES = ("msp", "energy", "maha", "mahalw", "maxlogit")


def load(p):
    return [json.loads(l) for l in open(p)]


def kth_largest(v, k):
    s = np.sort(np.asarray(v, dtype=np.float64))[::-1]
    return float(s[k - 1])


def main():
    root = sys.argv[1]
    out = {"rule": "tau = k_s-th largest dev score; open iff score >= tau; k_s = VSA seed-s dev open count",
           "seeds": {}}
    for s in SEEDS:
        vsa = load(f"{root}/preds/dev/vsa_s{s}.jsonl")
        dis = load(f"{root}/preds/dev/discopen_s{s}.jsonl")
        assert len(vsa) == len(dis) == 800, (len(vsa), len(dis))
        assert [r["id"] for r in vsa] == [r["id"] for r in dis]
        k = sum(r["mod"]["gate"] for r in vsa)
        e = {"vsa_dev_open": k, "vsa_dev_open_rate": k / 800, "vsa_dev_acc": float(np.mean([r["correct"] for r in vsa])),
             "disc_open_dev_acc": float(np.mean([r["correct"] for r in dis])),
             "disc_p08_dev_open": int(sum(r["mod"]["op_p"] > 0.8 for r in dis)), "tau": {}, "dev_open_at_tau": {},
             "tau_sens": {}, "dev_quantiles": {}}
        for g in GATES:
            v = np.array([r["mod"][g] for r in dis], dtype=np.float64)
            t = kth_largest(v, k)
            e["tau"][g] = t
            e["dev_open_at_tau"][g] = int((v >= t).sum())
            e["tau_sens"][g] = {str(rate): kth_largest(v, int(round(rate * 800))) for rate in (0.99, 0.95)}
            e["dev_quantiles"][g] = {q: float(np.quantile(v, float(q))) for q in ("0", "0.001", "0.01", "0.05", "0.5", "1")}
        out["seeds"][str(s)] = e
        print(f"seed {s}: VSA dev open {k}/800 (acc {e['vsa_dev_acc']:.4f}); DISC forced-open dev acc "
              f"{e['disc_open_dev_acc']:.4f}; DISC p08 would open {e['disc_p08_dev_open']}/800")
        for g in GATES:
            q = e["dev_quantiles"][g]
            print(f"   {g:8s} tau {e['tau'][g]:+.6g} (dev open {e['dev_open_at_tau'][g]}/800)  dev min {q['0']:+.6g} "
                  f"q01 {q['0.01']:+.6g} median {q['0.5']:+.6g} max {q['1']:+.6g}; sens 0.99 {e['tau_sens'][g]['0.99']:+.6g} "
                  f"0.95 {e['tau_sens'][g]['0.95']:+.6g}")
    json.dump(out, open(f"{root}/out/thresholds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
