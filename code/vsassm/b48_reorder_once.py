# B48 compute note 7: one-time reorder of the remaining part-A tuning queue (run under the queue lock on the box).
import re
Q = "q/tuneA"
L = [l.rstrip("\n") for l in open(Q) if l.strip()]
def key(l):
    enc = "qwen8" if "qwen8" in l else "phi2"
    m = re.search(r"--fam (\w+)", l)
    fam = m.group(1) if m else None
    ref = ("--arm" in l) or ("probe" in l) or ("/g0/" in l) or ("/g1/" in l) or ("bindtune" in l) or ("nametune" in l)
    fo = {"vq": 0, "dkvb": 1, "gumbel": 2, "st": 3, "cw": 4}
    if ref:
        return (0, 0 if enc == "qwen8" else 1, 0)
    return (1, fo.get(fam, 9), 0 if enc == "phi2" else 1)
L2 = sorted(L, key=key)   # stable: grid order kept inside each (family, encoder)
fixed = ["b48_run.py b31_arms.py --enc_model phi2 --arena C --arm exact --mode final --enc perm --readings comm/named --seeds 42 123 456 --feats feats --out out/final/X_C1_exact.json",
         "b48_run.py b31_arms.py --enc_model phi2 --arena C --arm name --mode final --epochs 4 --readings perm/index comm/named comm/index --seeds 42 123 456 --feats feats --out out/final/phi2_C1_name.json",
         "b48_run.py b31_arms.py --enc_model qwen8 --arena C --arm name --mode final --epochs 4 --readings perm/index comm/named comm/index --seeds 42 123 456 --feats feats --out out/final/qwen8_C1_name.json"]
assert len(set(L2)) == len(L2) == len(L)
open(Q, "w").write("".join(x + "\n" for x in fixed + L2))
print(len(L), "->", len(fixed + L2)); print("\n".join(x[:110] for x in (fixed + L2)[:12]))
