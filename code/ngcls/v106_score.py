"""V-106 scorer. Committed with the prereg, BEFORE any dump exists, and run verbatim on the dump.

Gates P1a/P1b/P1c and predictions P2/P3/P4/P5 exactly as registered. Nothing here chooses a statistic after
seeing the data; that is the whole point of committing it first. V-100's P3 fired on a .008 margin and was not
banked, so P4 here carries a .02 margin and no bar.
"""
import json, sys

F = sys.argv[1] if len(sys.argv) > 1 else "v106.json"
d = json.load(open(F))
at = lambda curve, K: next((c for c in curve if c["n_facts"] == K), None)

store, prompt, lora = d.get("store", []), d.get("prompt", []), d.get("lora", [])
gate = d.get("lora_gate", {})
print("===== V-106 =====")
print("  %-7s %-10s %-10s %-10s %-12s %-12s" % ("facts", "prompt", "lora", "store", "prompt_lat", "store_lat"))
for K in d.get("scales", []):
    p, l, s = at(prompt, K), at(lora, K), at(store, K)
    print("  %-7d %-10s %-10s %-10s %-12s %-12s" % (
        K,
        "%.4f" % p["acc"] if p else "-",
        "%.4f" % l["acc"] if l else "-",
        "%.4f" % s["acc"] if s else "-",
        "%.3fs" % p["latency_s"] if p else "-",
        "%.4fs" % s["latency_s"] if s else "-"))

# ---- gates -------------------------------------------------------------------------------------------------
s512 = at(store, 512)
p1a = s512 is not None and abs(s512["acc_norm"] - 0.9194) <= 0.05
print("\n  >>> P1a store reproduces V-102 (K=512, S=8, d=2048, norm metric) %.4f vs .9194 +-.05 : %s"
      % (s512["acc_norm"] if s512 else float("nan"), p1a))

p1b = bool(gate) and abs(gate.get("em", -1) - 0.7333) <= 0.05
print("  >>> P1b lora reproduces V-99 at its own budget %.4f vs .7333 +-.05 : %s"
      % (gate.get("em", float("nan")), p1b))

p8 = at(prompt, 8)
p1c = p8 is not None and abs(p8["acc"] - 0.9417) <= 0.05
print("  >>> P1c prompt reconstruction reproduces V-100 at K=8 %.4f vs .9417 +-.05 : %s"
      % (p8["acc"] if p8 else float("nan"), p1c))

if not (p1a and p1b):
    print("\n  >>> GATE FAILURE (P1a and/or P1b): per the prereg no reading is taken from the sweep.")
    print("      P1a failing means the store wrapper changed the path it wraps; P1b failing means the "
          "fine-tune did not reproduce. Investigate before reading P2-P5.")
    print("V106-SCORE-DONE")
    sys.exit(0)
if not p1c:
    print("      NOTE: P1c failed -- V-100's prompt curve stands unreproducible and the prompt arm is a fresh "
          "measurement. P2/P3 do not turn on the prompt arm, so the sweep still reads, with this noted.")

# ---- predictions -------------------------------------------------------------------------------------------
sK, lK = at(store, 1024), at(lora, 1024)
p2 = sK is not None and lK is not None and (sK["acc"] - lK["acc"]) >= 0.05
print("\n  >>> P2 at K=1024 store - lora = %.4f - %.4f = %+.4f >= .05 : %s"
      % (sK["acc"] if sK else float("nan"), lK["acc"] if lK else float("nan"),
         (sK["acc"] - lK["acc"]) if (sK and lK) else float("nan"), p2))

l8 = at(lora, 8)
p3 = l8 is not None and lK is not None and lK["acc"] <= l8["acc"] - 0.15
print("  >>> P3 lora pays for scale: acc(1024) %.4f <= acc(8) %.4f - .15 = %.4f : %s"
      % (lK["acc"] if lK else float("nan"), l8["acc"] if l8 else float("nan"),
         (l8["acc"] - 0.15) if l8 else float("nan"), p3))

cross = [K for K in d.get("scales", [])
         if at(store, K) and at(lora, K) and at(store, K)["acc"] - at(lora, K)["acc"] >= 0.02]
if cross:
    K = cross[0]
    s, l, p = at(store, K), at(lora, K), at(prompt, K)
    print("  >>> P4 crossover (no bar): smallest K where store beats lora by >= .02 is K=%d "
          "(store %.4f vs lora %.4f); there the LLM route spends %.0f tokens / %.3fs against the store's "
          "%.4fs -- latency ratio %.0fx"
          % (K, s["acc"], l["acc"], p["tokens"] if p else float("nan"),
             p["latency_s"] if p else float("nan"), s["latency_s"],
             (p["latency_s"] / max(1e-9, s["latency_s"])) if p else float("nan")))
else:
    print("  >>> P4 crossover (no bar): NONE up to K=%d -- the store never beats lora by >= .02, so the "
          "regime where it pays is beyond this sweep." % max(d.get("scales", [0])))

wins = [(K, at(lora, K)["acc"] - at(prompt, K)["acc"]) for K in d.get("scales", [])
        if at(lora, K) and at(prompt, K)]
p5 = all(w > 0 for _, w in wins) if wins else False
print("  >>> P5 (descriptive) lora beats prompt at every scale : %s | margins %s"
      % (p5, [(K, round(w, 4)) for K, w in wins]))

print("\n  registered readings: P2 and P3 -> first measured regime where the store beats a task-tuned LLM, "
      "mechanism = context routes pay for evidence scale while a sharded store's capacity is per shard; "
      "NEXT PAPER, nothing enters 9/16. P2 fail w/ lora holding -> direction dies as an accuracy argument, "
      "only the cost claim survives, reported first. P2 fail w/ store degrading -> contradicts V-102 at equal "
      "facts-per-shard, treat as harness fault. P3 fail -> the direction's premise is false and it closes. "
      "P2 hit w/ P3 fail -> the win is NOT attributed to capacity.")
print("  scoping: all arms receive GOLD triples; this measures USE of evidence, not acquisition. Extraction "
      "(recall .6589 / .4678) untouched and still binding. A P2 hit licenses only 'given the same evidence at "
      "scale, the store's readout beats a task-tuned model's reading of it'.")
print("V106-SCORE-DONE")
