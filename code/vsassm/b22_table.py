"""LaTeX table of every B22 condition (registered, amendment 1 added levels, amendment 2 bipolar), from the JSONs."""
import json, os, sys
R = []
for f, tag in (("results/b22_robust.json", ""), ("results/b22_robust_ext.json", "$^{a}$"), ("results/b22_robust_ext_trades.json", "$^{a}$"), ("results/b22_robust_bip.json", "$^{b}$")):
    if os.path.exists(f):
        R += [dict(r, tag=tag) for r in json.load(open(f))]
def name(c):
    if c == "float":
        return "float32", "--"
    base, _, trade = c.partition(" | ")
    k, v = base.split(" ", 1)
    if base.startswith("Q b=1 bipolar"):
        b = "1 bit, bipolar store"
    elif k == "Q":
        b = "%s bit%s" % (v.split("=")[1], "" if v.split("=")[1] == "1" else "s")
    elif k == "N":
        b = "read noise %s $\\times$ signal" % v.split("=")[1]
    else:
        b = "%d\\%% of components failed" % round(100 * float(v.split("=")[1]))
    t = {"": "16 per vector", "8 per vector": "8 per vector", "4 per vector": "4 per vector", "d=16384": "16 per vector, $d = 16{,}384$"}[trade]
    return b, t
print(r"\begin{tabular}{llrrr}")
print(r"\toprule")
print(r"Condition & Load & Facts exact & Share & Stores fully exact \\")
print(r"\midrule")
for r in R:
    b, t = name(r["cond"])
    print("%s%s & %s & %d / %d & %.4f & %d / %d \\\\" % (b, r["tag"], t if t != "--" else "16 per vector", r["keys_exact"], r["keys"], r["frac"], r["stores_all_exact"], r["stores"]))
print(r"\bottomrule")
print(r"\end{tabular}")
