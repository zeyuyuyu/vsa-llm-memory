"""Frozen scorer for V-163 (prereg in ledger.md). Committed with v163.py before the run; reads only artifacts."""
import json, sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "v163_logs/v163.json"
d = json.load(open(RUN))
v95 = json.load(open("v95_logs/v95_scored.json"))
v13 = json.load(open("v13_logs/v13_fixed.json"))
BAR, FLOOR = 0.615, 0.515
out = {}

# ---- G0: the fixed rule reproduces V-95 per seed at each of its eleven c
cur = {round(r["target_c"], 2): r for r in d["curve"]}
worst0, n0 = 0.0, 0
for r in v95["curve"]:
    mine = cur[round(r["target_c"], 2)]["fixed"]
    for a, b in zip(mine, r["acc_per_seed"]):
        worst0 = max(worst0, abs(a - b)); n0 += 1
g0 = n0 == 33 and worst0 <= 1e-4
print("G0 fixed rule reproduces V-95 (33 values): worst |diff| %.6f -> %s" % (worst0, "PASS" if g0 else "FAIL"))

# ---- G1: the fitted rule at c = 0 reproduces the banked parser arm of the Table configuration
bank = {r["seed"]: r["acc"] for r in v13 if r["arm"] == "parser"}
worst1 = max(abs(t["fitted"] - bank[t["seed"]]) for t in d["table_config"])
g1 = len(d["table_config"]) == 3 and worst1 <= 1e-4
print("G1 fitted rule reproduces the banked parser arm %s: worst |diff| %.6f -> %s"
      % ([round(bank[s], 4) for s in (42, 123, 456)], worst1, "PASS" if g1 else "FAIL"))
out["G0"], out["G1"] = g0, g1
if not (g0 and g1):
    print("GATE FAILS: nothing is read from this run.")
    json.dump(out, open(RUN.replace(".json", "_scored.json"), "w"), indent=1)
    sys.exit(0)

# ---- reported whatever the outcome
print("\n  c     crosstalk  fixed   fitted  auroc")
for r in d["curve"]:
    print("  %.2f  %.4f     %.4f  %.4f  %.4f" % (r["target_c"], r["crosstalk_mean"], r["fixed_mean"],
                                                r["fitted_mean"], r["auroc_mean"]))
passing = [r for r in d["curve"] if r["fitted_mean"] >= BAR]
tol = max(r["crosstalk_mean"] for r in passing)
tol_c = max(r["target_c"] for r in passing)
below = [r for r in d["curve"] if r["target_c"] > tol_c]
nxt = min(below, key=lambda r: r["target_c"]) if below else None
mono = all(d["curve"][i + 1]["fitted_mean"] <= d["curve"][i]["fitted_mean"] + 0.03 for i in range(len(d["curve"]) - 1))
print("\ntolerance under the fitted rule: largest tested cross-talk clearing %.3f is %.4f (target c = %.2f); "
      "next tested point %s; curve monotone within 0.03: %s"
      % (BAR, tol, tol_c, "c = %.2f at %.4f" % (nxt["target_c"], nxt["fitted_mean"]) if nxt else "none", mono))
out.update(tolerance=tol, tolerance_target_c=tol_c, monotone=mono)

# ---- B1
b1 = tol < 0.39
print("B1 tolerance < 0.39: %s" % ("HOLDS" if b1 else "FAILS"))
# ---- B2
hi = [r for r in d["curve"] if r["target_c"] >= 0.7 - 1e-9]
b2a = all(r["fitted_mean"] <= FLOOR + 0.05 for r in hi)
b2b = all(r["auroc_mean"] <= 0.60 for r in hi)
print("B2 at c >= 0.7: fitted <= %.3f in all %d points: %s (max %.4f); auroc <= 0.60: %s (max %.4f) -> %s"
      % (FLOOR + 0.05, len(hi), b2a, max(r["fitted_mean"] for r in hi), b2b, max(r["auroc_mean"] for r in hi),
         "HOLDS" if (b2a and b2b) else "FAILS"))
# ---- B3
ea = sum(t["err_alias"] for t in d["table_config"]); ec = sum(t["err_clean"] for t in d["table_config"])
na = sum(t["n_alias"] for t in d["table_config"]); nc = sum(t["n_clean"] for t in d["table_config"])
share = ea / max(1, ea + ec); rate_clean = ec / max(1, nc)
b3 = share >= 0.90 and rate_clean <= 0.01
print("B3 first-hop errors: %d on alias instances (%d instances, rate %.3f), %d on the others (%d instances, rate %.4f); "
      "share on alias instances %.3f -> %s"
      % (ea, na, ea / max(1, na), ec, nc, rate_clean, share, "HOLDS" if b3 else "FAILS"))
print("   per seed: %s" % [(t["seed"], t["n_alias"], t["err_alias"], t["n_clean"], t["err_clean"],
                            t["err_onto_aliased_role"]) for t in d["table_config"]])
out.update(B1=b1, B2=b2a and b2b, B2_fitted=b2a, B2_auroc=b2b, B3=b3, alias_share=share, clean_rate=rate_clean,
           alias_instances=na, clean_instances=nc)
print("\nV-163 VERDICT: G0 %s | G1 %s | tolerance %.4f | B1 %s | B2 %s | B3 %s"
      % ("PASS", "PASS", tol, "HOLDS" if b1 else "FAILS", "HOLDS" if (b2a and b2b) else "FAILS",
         "HOLDS" if b3 else "FAILS"))
json.dump(out, open(RUN.replace(".json", "_scored.json"), "w"), indent=1)
