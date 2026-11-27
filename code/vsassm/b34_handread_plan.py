"""B34 hand-reading plan (prereg 'Hand reading'): for each arm, the lowest k >= 4 where it reads < 99.5% at 30 days,
plus V-S x16, V-Snt x32, T-v x16 and T-vhdd x32; the first two stores (store order) with a failure in that cell.
Prints one b34_check.py handread command per case.

python3 b34_handread_plan.py results/b34_inmem.json > b34_out/handread_cmds.txt
"""
import json
import sys

d = json.load(open(sys.argv[1]))
tot = d["totals"]
per = d["per_job"]
arms = ("V-S", "V-Snt", "T-i", "T-iih", "T-iis", "T-iix", "T-v", "T-iiia", "T-iiib", "T-vhd", "T-vhdd")
cases = []
for a in arms:
    for k in (4, 8, 16, 32):
        v = tot["%s|30 d|x%d" % (a, k)]
        if v[2] < 0.995:
            cases.append((a, k))
            break
for extra in (("V-S", 16), ("V-Snt", 32), ("T-v", 16), ("T-vhdd", 32)):
    if extra not in cases:
        cases.append(extra)
for a, k in cases:
    fails = sorted(r["store"] for r in per if r["k"] == k and r["res"]["%s|30 d" % a][0] < r["res"]["%s|30 d" % a][1])
    for sid in fails[:2]:
        print("%s|%d|%s" % (a, k, sid))
