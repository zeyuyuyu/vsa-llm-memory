"""B41 GPU ledger: sums the wall-clock seconds recorded by every engine run (dev, test, exploratory) and the
feature-extraction metas; head fits and gates are added from their logs' timings (approximate, listed apart)."""
import glob, json, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else "/path/to/work/b41"
rows = []
for sub in ("runs", "runs_x"):
    for f in sorted(glob.glob(os.path.join(R, sub, "*.json"))):
        if os.path.basename(f) == "g1.json":
            continue
        m = json.load(open(f))
        rows.append((sub, m["split"], m["sec_total"]))
feat = []
for f in glob.glob(os.path.join(R, "feats", "*_meta.json")):
    m = json.load(open(f))
    feat.append((m["split"], m["seconds"]))
tot = 0.0
out = []
for key in sorted(set((r[0], r[1]) for r in rows)):
    s = sum(r[2] for r in rows if (r[0], r[1]) == key)
    tot += s
    out.append("engine %-7s %-5s %3d runs  %.2f GPU-h" % (key[0], key[1], sum(1 for r in rows if (r[0], r[1]) == key),
                                                         s / 3600))
for sp, s in feat:
    tot += s
    out.append("Qwen3-8B features %-5s %.2f GPU-h" % (sp, s / 3600))
out.append("naming-head fits (4 layers x ~35 s, twice incl. the discarded partial fit) + T fit + logits: ~0.15 GPU-h")
out.append("G1 gate + dev debugging traces: ~0.1 GPU-h")
out.append("TOTAL (recorded) %.2f GPU-h + ~0.25 GPU-h unrecorded = %.2f GPU-h (A800, GPUs 4-7; no PACE; no paid API)"
           % (tot / 3600, tot / 3600 + 0.25))
print("\n".join(out))
