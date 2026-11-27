"""B68: a readable summary of out/select.json (dev only) for the dev-choices record. Prints markdown."""
import hashlib, json, sys
p = sys.argv[1]
cells = sys.argv[2].split(",") if len(sys.argv) > 2 else None
sel = json.load(open(p))
print("select.json md5 %s; complete cells %d, incomplete %s" % (hashlib.md5(open(p, "rb").read()).hexdigest(),
      len(sel["cells_complete"]), json.dumps(sel["cells_incomplete"])))
print()
print("| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |")
print("|---|---|---|---|---|---|---|")
for k in sel["cells_complete"]:
    if cells and k not in cells:
        continue
    b = sel["bind"][k]
    row = ["%g (%.3f / %.3f)" % (b["pick"]["cfg"]["lr"], b["rows"][0]["dev"], b["rows"][1]["dev"])]
    for f in ("vq", "dkvb", "gumbel", "st", "cw"):
        v = sel[f][k]
        c = ", ".join("%s %s" % (kk, ("%g" % vv) if isinstance(vv, float) else vv) for kk, vv in sorted(v["cfg"].items()))
        row.append("%.3f ep %d (%s)" % (v["dev"], v["epoch"], c))
    print("| %s | %s |" % (k, " | ".join(row)))
