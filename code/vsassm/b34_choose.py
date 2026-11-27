"""B34 dev choice (rule fixed in the prereg before the grid was read): for each split arm, the directory share s with the
highest mean (over k = 4, 8, 16, 32 at 30 days) of the pooled share of dev keys read exactly, among shares that read
every dev key at k = 1 at 30 days; ties within 0.1 points go to the smaller share. MPHF: the lambda with the fewest
overhead bits per pair (displacements + empty rows) averaged over the dev stores.

python3 b34_choose.py GRID.json [GRID2.json ...] b34_out/dev_mphf_lambda.json OUT.json
(the T-vhd and T-vhdd grids ran after the MPHF lambda was chosen, with that lambda.)
"""
import json
import sys

ARMS = ("T-iis", "T-iix", "T-iiia", "T-iiib", "T-vhd", "T-vhdd")
SHARES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

tot = {}
for f in sys.argv[1:-2]:
    tot.update(json.load(open(f))["totals"])
lam = json.load(open(sys.argv[-2]))
choice = {}
for arm in ARMS:
    rows = []
    for s in SHARES:
        tag = "%s@%.1f" % (arm, s)
        k1 = tot["%s|30 d|x1" % tag]
        m = sum(tot["%s|30 d|x%d" % (tag, k)][2] for k in (4, 8, 16, 32)) / 4 * 100
        per_k = [round(tot["%s|30 d|x%d" % (tag, k)][2] * 100, 2) for k in (4, 8, 16, 32)]
        rows.append((s, m, k1[0] == k1[1], per_k))
        print("%-7s share %.1f  mean %.2f  x4/x8/x16/x32 %s  k=1 exact %s" % (arm, s, m, per_k, k1[0] == k1[1]))
    ok = [r for r in rows if r[2]]
    top = max(r[1] for r in ok)
    pick = min(r[0] for r in ok if r[1] >= top - 0.1)
    choice[arm] = {"share": pick, "dev_mean": round(top, 3)}
    print("  -> %s share %.1f" % (arm, pick))
best_lam = min(lam, key=lambda k: (lam[k], int(k)))
choice["T-v"] = {"lam": int(best_lam), "dev_overhead_bits_per_pair": lam}
print("T-v lambda", best_lam, lam)
json.dump(choice, open(sys.argv[-1], "w"), indent=1)
