"""B65 dev summary: every dev number (layer scan, reader grids, write-back curves) and the mechanical choices, as
markdown for prereg/B65_dev_record.md. Reads $B65_ROOT/out only (dev split; seeds 1, 2 report their own dev curves).
usage: python b65_devsum.py <root>
"""
import glob, json, os, sys

root = sys.argv[1]
c = json.load(open(f"{root}/out/choices.json"))
L = []
p = L.append
p("## Layer scan (operand positions; VSA encoder, lr 1e-3, 100 epochs; dev read exact, published read)")
p("")
sc = c.get("scan_ops", {})
ks = sorted(sc, key=int)
p("| layer | " + " | ".join(str(k) for k in ks) + " |")
p("|---|" + "---|" * len(ks))
p("| read exact | " + " | ".join(f"{sc[k]:.4f}" for k in ks) + " |")
p(f"\nChoice: OPS layer {c['layer']['ops']}; LAST layer {c['layer']['last']} (B40's).\n")
p("## Readers (seed 0; dev read exact under each read rule)")
p("")
p("| position | encoder | lr | " + " | ".join(["epochs", "published", "best match / disc"]) + " |")
p("|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{root}/out/rd/*_s0/reader.json")):
    d = json.load(open(f))
    a = d["args"]
    for h in d["hist"]:
        if a["arm"] == "vsa":
            p(f"| {a['pos']} | vsa | {a['lr']:g} | {h['epoch']} | {h['pub']:.4f} | {h['bm']:.4f} |")
        else:
            p(f"| {a['pos']} | disc | {a['lr']:g} | {h['epoch']} | - | {h['disc']:.4f} |")
p("")
p("Reader choices: " + "; ".join(f"{k} lr {v['lr']} e{v['epochs']} ({v['dev_read']:.4f})" for k, v in c.get("reader", {}).items()))
p("")
p("## Write-back (seed 0; dev first-token argmax at steps 0, 250, 500, 750, 1000; best over 250-1000)")
p("")
p("| position | arm | lr | curve | best (step) |")
p("|---|---|---|---|---|")
rows = []
for f in glob.glob(f"{root}/out/dev/*/train.json"):
    d = json.load(open(f))
    a = d["args"]
    rows.append((a["pos"], a["arm"], a["lr"], d))
for pos, arm, lr, d in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
    cur = " ".join(f"{h['dev']:.4f}" for h in d["hist"])
    p(f"| {pos} | {arm} | {lr:g} | {cur} | {d['best_dev']:.4f} ({d['best_step']}) |")
p("")
p("Write-back choices (rule: best dev; ties -> lr nearest 1e-3, then smaller; strict edge optimum -> extend):")
for k, v in c.get("wb", {}).items():
    p(f"- {k}: lr {v['lr']} ({v['status']}); grid {c['wb_grid'].get(k)}; dev {v['dev']}")
fin = sorted(glob.glob(f"{root}/out/fin/*/train.json"))
if fin:
    p("")
    p("## Seeds 1 and 2 (chosen settings; own best dev step)")
    p("")
    p("| run | curve | best (step) |")
    p("|---|---|---|")
    for f in fin:
        d = json.load(open(f))
        cur = " ".join("%.4f" % h["dev"] for h in d["hist"])
        p(f"| {os.path.basename(os.path.dirname(f))} | {cur} | {d['best_dev']:.4f} ({d['best_step']}) |")
    rd = sorted(glob.glob(f"{root}/out/rd/*_s[12]/reader.json"))
    for f in rd:
        d = json.load(open(f))
        p(f"- reader {os.path.basename(os.path.dirname(f))}: " + ", ".join(
            f"e{h['epoch']} " + (f"pub {h['pub']:.4f} bm {h['bm']:.4f}" if "pub" in h else f"disc {h['disc']:.4f}") for h in d["hist"]))
print("\n".join(L))
