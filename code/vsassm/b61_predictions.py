"""B61: predictions R1-R8 (prereg/B61_hd_protected_store_prereg.md) against results/b61_verdict.json and the dev choice."""
import json
import sys

import b61_hd as H

J = json.load(open(sys.argv[1]))
ch = json.load(open(sys.argv[2]))
K = H.IIDC
LK = [c for c in K if c != "e05"]
st = lambda v: H.NG if v is None else H.gidx(v)
ns = lambda a, c: J["nstar"]["%s|%s" % (a, c)]
model = dict(k1=32, k2=32, k4=45, e05=54, k8=91, k16=304, k32=1024)
out = []
r1 = {c: st(ns("H-SBC", c)) - H.gidx(model[c]) for c in K}
out.append(("R1 H-SBC within 2 steps of the model at every k", all(abs(v) <= 2 for v in r1.values()), r1))
g = {c: J["G"]["H-SBC"][c]["G"] for c in K}
out.append(("R2 H-SBC G >= +2 at >= 5 of 7 k and G >= +1 at every k",
            sum(v >= 2 for v in g.values()) >= 5 and all(v >= 1 for v in g.values()), g))
d = {c: J["H1"]["H-SBC"][c]["delta"] for c in K}
out.append(("R3 H-SBC TABLE WIN, -4 <= Delta <= -2 at every k", J["H1"]["H-SBC"]["label"] == "TABLE WIN"
            and all(-4 <= v <= -2 for v in d.values()), (J["H1"]["H-SBC"]["label"], d)))
r4 = {c: st(ns("H-SIC", c)) - st(ns("H-SBC", c)) for c in K}
out.append(("R4 H-SIC within 1 step of H-SBC at every k", all(abs(v) <= 1 for v in r4.values()), r4))
r5 = {c: J["G"]["H-HAD"][c]["G"] for c in K}
out.append(("R5 H-HAD within 1 step of U at every k", all(abs(v) <= 1 for v in r5.values()), r5))
r6 = {c: [a for a in ("H-SBC", "H-SIC", "H-HAD") if st(ns(a, c)) == st(ns("HD-best", c))] for c in K}
out.append(("R6 HD-best = H-SBC or H-SIC at every k", all(("H-SBC" in v or "H-SIC" in v) for v in r6.values()), r6))
r7 = {c: ch["designs"]["H-SBC"][c].get(str(H.gidx(ch["dev_nstar"]["H-SBC"][c]))) for c in K}
out.append(("R7 dev lam at dev N* 0.5 or 0.7 at every k", all(v in (0.5, 0.7) for v in r7.values()), r7))
dm = {c: J["damage"][c].get("H-SBC_eval") for c in H.DMG4}
ok8 = all(v and v["posind"] and v["delta"] >= 2 for v in dm.values()) and sum(v["G"] >= 2 for v in dm.values()) >= 2
out.append(("R8 damage: H-SBC position independent and fewer cells than the best non-HD table in 4 of 4, fewer than U in"
            " >= 2 of 4", ok8, dm))
for name, ok, det in out:
    print("%-6s %s  %s" % ("HELD" if ok else "FAILED", name, det))
