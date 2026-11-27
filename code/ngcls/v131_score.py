"""V-131 scorer. Committed with the prereg BEFORE any part is run, and run verbatim on the dumps."""
import json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "v131_logs"
RNG = np.random.default_rng(131)
BOOT = 20000

BANKED_A = {"prompt": [0.2533333333333333] * 3, "prompt_ev": [0.95] * 3,
            "hrr_gold": [0.97, 0.9766666666666667, 0.9766666666666667]}
HANDOVER = 1.0
P1_BAR, P3_BAR, P0_TOL = 0.10, 0.20, 0.01
STORE_MUSIQUE = 0.9481132075471698          # v79 hrr_gold_p0, covered slice
STORE_MUSIQUE_P150 = 0.6556                 # v79 hrr_gold_p150


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci_paired(a, b):
    """Paired bootstrap over rows of (a - b). Returns mean and the 95% interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = RNG.integers(0, n, size=(BOOT, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


print("===== V-131: the three softest points, measured (prereg %s) =====" % os.environ.get("V131_PREREG", "?"))

A = load("v65_rerun.json"); AR = load("v65_rows.json")
B = load("v131b.json"); C = load("v131c.json")
ANS = None
for p in ("/path/to/work/ngcls/v125_answerable.json", os.path.join(D, "v125_answerable.json")):
    if os.path.exists(p):
        ANS = json.load(open(p))["answerable_indices"]; break

# ---- P0 validity gate ------------------------------------------------------------------------------
print("\n>>> P0 validity gate (Part A reproduces V-65 within +-%.2f)" % P0_TOL)
p0 = A is not None and AR is not None
if not p0:
    print("      Part A artifacts absent -> gate cannot pass")
else:
    for arm, ref in BANKED_A.items():
        got = A["arms"][arm]["full"]
        ok = all(abs(g - r) <= P0_TOL for g, r in zip(got, ref))
        p0 = p0 and ok
        print("      %-10s %s vs banked %s : %s"
              % (arm, [round(g, 4) for g in got], [round(r, 4) for r in ref], ok))
    n_rows = AR["n"]
    complete = all(len(v) == n_rows for arm in AR["arms"].values() for v in arm.values())
    print("      per-row dumps complete (%d rows, every arm x seed): %s" % (n_rows, complete))
    p0 = p0 and complete
if not p0:
    print("\n      GATE FAILS. Per the prereg NOTHING IS READ from any part.")
    print("V131-SCORE-DONE"); sys.exit(0)

# ---- P1 PRIMARY: the tuned opponent ----------------------------------------------------------------
print("\n>>> P1 (PRIMARY, bar: best-of-grid LoRA trails the handover's %.3f by >= %.2f in 3/3 seeds)"
      % (HANDOVER, P1_BAR))
P1 = None
if B is None:
    print("      Part B artifact absent")
else:
    runs = [r for r in B["runs"] if r.get("accs")]
    print("      %d/%d configurations completed" % (len(runs), len(B["grid"])))
    for r in sorted(runs, key=lambda x: -sum(x["accs"]) / len(x["accs"])):
        m = sum(r["accs"]) / len(r["accs"])
        print("        %-16s mean %.4f  seeds %s" % (r["tag"], m, [round(x, 4) for x in r["accs"]]))
    if runs:
        best = max(runs, key=lambda x: sum(x["accs"]) / len(x["accs"]))
        gaps = [HANDOVER - x for x in best["accs"]]
        P1 = all(g >= P1_BAR for g in gaps)
        print("      BEST (selected on test, disclosed handicap): %s mean %.4f seeds %s"
              % (best["tag"], sum(best["accs"]) / len(best["accs"]), [round(x, 4) for x in best["accs"]]))
        print("      handover - best, per seed: %s" % [round(g, 4) for g in gaps])
        print("      manuscript's untuned point: %s (mean %.4f)"
              % ([round(x, 4) for x in B["banked_lora_per_seed"]],
                 sum(B["banked_lora_per_seed"]) / 3))
        print("      P1: %s" % P1)

# ---- P2: is the residual-halving claim resolvable? -------------------------------------------------
print("\n>>> P2 (bar: paired bootstrap CI for hrr_gold - prompt_ev on the 300 rows excludes zero)")
store_rows = np.mean([AR["arms"]["hrr_gold"][s] for s in AR["arms"]["hrr_gold"]], axis=0)
ev_rows = np.mean([AR["arms"]["prompt_ev"][s] for s in AR["arms"]["prompt_ev"]], axis=0)
sp_rows = np.mean([AR["arms"]["prompt"][s] for s in AR["arms"]["prompt"]], axis=0)
m, lo, hi = ci_paired(store_rows, ev_rows)
P2 = lo > 0
print("      store %.4f  evidence-in-prompt %.4f  single pass %.4f"
      % (store_rows.mean(), ev_rows.mean(), sp_rows.mean()))
print("      store - evidence_in_prompt = %+.4f  95%% CI [%+.4f, %+.4f]  (discordant rows: %d)"
      % (m, lo, hi, int(np.sum(store_rows != ev_rows))))
ms, los, his = ci_paired(store_rows, sp_rows)
print("      store - single_pass        = %+.4f  95%% CI [%+.4f, %+.4f]  (the headline gap, for contrast)"
      % (ms, los, his))
print("      P2: %s" % P2)

# ---- P3: the untruncated MuSiQue baseline ----------------------------------------------------------
print("\n>>> P3 (bar: untruncated single pass trails the store's %.4f by >= %.2f)"
      % (STORE_MUSIQUE, P3_BAR))
P3 = None
if C is None or not C.get("row_set_gate"):
    print("      Part C artifact absent or its row-set gate failed")
else:
    sp = float(np.mean(C["arms"]["single_pass"]["covered"]))
    tr = float(np.mean(C["diag"]["single_pass_trunc"]))
    tok = float(np.mean(C["diag"]["single_pass_tokens"]))
    P3 = (STORE_MUSIQUE - sp) >= P3_BAR
    print("      single_pass (32k reader) %.4f  trunc %.3f  mean prompt tokens %.0f"
          % (sp, tr, tok))
    print("      manuscript's phi-2 single pass %.4f at trunc %.3f" % (C["banked"]["prompt"],
                                                                       C["banked"]["prompt_trunc"]))
    print("      store - single_pass = %+.4f : P3 %s" % (STORE_MUSIQUE - sp, P3))
    if "prompt_sup" in C["arms"]:
        print("      prompt_sup (gold paragraphs only) %.4f (manuscript %.4f)"
              % (float(np.mean(C["arms"]["prompt_sup"]["covered"])), C["banked"]["prompt_sup"]))
    k = [a for a in C["arms"] if a.startswith("prompt_ev_p")]
    if k:
        a0 = k[0]
        print("      %s %.4f  trunc %.3f  gold_kept %.3f  (manuscript %.4f at gold_kept %.2f-%.2f; store %.4f)"
              % (a0, float(np.mean(C["arms"][a0]["covered"])), float(np.mean(C["diag"]["%s_trunc" % a0])),
                 float(np.mean(C["diag"]["%s_goldkept" % a0])), C["banked"]["prompt_ev_p150"],
                 C["banked"]["p150_goldkept"][0], C["banked"]["p150_goldkept"][1], STORE_MUSIQUE_P150))

# ---- P4 descriptive: the answerable subset, finally computable -------------------------------------
print("\n>>> P4 (descriptive, no bar): the V-125 answerable subset, which phi-2 dumps could not reach before")
if ANS is None:
    print("      v125_answerable.json not found")
else:
    idx = np.array(sorted(ANS))
    print("      %d answerable rows of %d" % (len(idx), AR["n"]))
    for nm, v in (("single pass", sp_rows), ("evidence in prompt", ev_rows), ("store", store_rows)):
        print("        %-20s full %.4f   answerable %.4f" % (nm, v.mean(), v[idx].mean()))
    er_full = (store_rows.mean() - sp_rows.mean()) / max(1e-9, 1 - sp_rows.mean())
    er_ans = (store_rows[idx].mean() - sp_rows[idx].mean()) / max(1e-9, 1 - sp_rows[idx].mean())
    print("      error removal (store over single pass): full %.1f%%  answerable %.1f%%"
          % (100 * er_full, 100 * er_ans))
    ma, la, ha = ci_paired(store_rows[idx], ev_rows[idx])
    print("      store - evidence_in_prompt on the answerable rows: %+.4f  95%% CI [%+.4f, %+.4f]"
          % (ma, la, ha))

# ---- registered readings ---------------------------------------------------------------------------
print("\n--- readings, as registered ---")
if P1 is True:
    print("  P1 holds: the two-hop comparison survives a tuned opponent. The manuscript restates the margin")
    print("  at the TUNED number in the same commit, and the 'a larger or longer one may close part of the")
    print("  gap' concession is replaced by this measurement.")
elif P1 is False:
    print("  REPORTED FIRST -- P1 FAILS: the headline bAbI comparison was a tuning artifact. 'About twenty")
    print("  points at two hops' is struck from the abstract, the introduction, Figure 1's caption and the")
    print("  Discussion and replaced by the measured tuned gap; if the tuned fine-tune reaches the ceiling,")
    print("  the claim that structure beats an equally supervised fine-tune at two hops is WITHDRAWN on")
    print("  natural text and survives only on the synthetic arenas.")
if P2:
    print("  P2 holds: 'halves the residual error' keeps its wording and gains an interval.")
else:
    print("  P2 fails: the store and the in-prompt evidence arm are NOT distinguishable on 300 rows. The")
    print("  sentence is restated as not resolvable at this sample size -- a stronger form of the")
    print("  counter-finding the paper already carries, and no longer resting only on the ICLR companion.")
if P3 is True:
    print("  P3 holds: the MuSiQue separation is a property of the method, not of phi-2's window.")
elif P3 is False:
    print("  P3 fails: the 78-point MuSiQue gain was substantially a context-window artifact. The benchmark")
    print("  is restated at the untruncated number and the abstract's MuSiQue figure changes.")
print("  scoping: Parts A and C hand every arm the same gold evidence (use, not acquisition); Part B moves")
print("  only the comparator's hyperparameters. Zero API spend.")
print("V131-SCORE-DONE")
