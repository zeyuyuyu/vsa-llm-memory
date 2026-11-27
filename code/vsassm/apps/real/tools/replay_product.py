import importlib.util, json, sys, inspect
sys.path.insert(0, "apps/real"); sys.path.insert(0, ".")
import b14
from run_real import load, ok, message
def mod(path):
    s = importlib.util.spec_from_file_location("p" + str(abs(hash(path))), path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
split, rfile = sys.argv[1], sys.argv[2]
rd = {r["id"]: r for r in map(json.loads, open(rfile))}
for name, path in [("v1", sys.argv[3]), ("v2", "apps/real/product.py")]:
    P = mod(path); res = {}
    for l in load(split):
        r = rd[l["id"]]; s = P.SiteMemory(l, b14)
        new = "text" in inspect.signature(s.apply).parameters
        for i, x in enumerate(l["lines"]):
            evs = P.connect(l["scenario"], l["text_lines"][i]) if x["kind"] == "struct" else None
            if evs is not None:
                s.apply(evs, from_reader=False); continue
            if x["kind"] != "human": continue
            ev = b14.parse(r["raw"].get(str(i), "")) or []
            kw = {"text": message(l, x)} if new else {}
            s.apply(ev, from_reader=True, channel=x["src"], **kw)
        for q in l["questions"]:
            a, c, e = s.ask(q["q"], q["kind"])
            res.setdefault(l["scenario"], []).append(ok(q, a, s.reg))
    print(name, "  ".join(f"{k} {sum(v)/len(v):.3f}" for k, v in res.items()))
