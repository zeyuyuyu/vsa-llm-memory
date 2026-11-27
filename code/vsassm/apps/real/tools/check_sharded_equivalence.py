"""ShardedMemory must give one b14.Memory's answers. Every log of a split goes through the SDK three times with the same
deterministic fake reader as check_sdk_equivalence.py: the shipped single memory, ShardedMemory with its default limits,
and ShardedMemory with tiny limits (3 items a shard, a rebuild at 5 value codes), which shards and rebuilds all day.
The reader's prompts, the events written, every answer and the final board must be identical.

    python3 apps/real/tools/check_sharded_equivalence.py dev test test2
"""
import hashlib, json, re, sys
sys.path.insert(0, "apps/real"); sys.path.insert(0, ".")
import run_real as rr
import product
from readable_memory import SiteMemory
h = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)
for split in sys.argv[1:] or ["test2"]:
    logs = rr.load(split)
    truth = {rr.message(l, x): x["events"] for l in logs for x in l["lines"] if x["kind"] == "human"}

    def fake(prompts, seen):
        out = []
        for p in prompts:
            seen.append(p)
            if p.startswith("Message:") and "Choose one:" in p: out.append(("AB"[h(p) % 2], 1))
            elif p.startswith("Message:"): out.append(("yes" if h(p) % 5 else "no", 1))
            else:
                sent = re.findall(r"\nSentence: ([^\n]*)", p)[-1]; evs = truth.get(sent, [])
                if h(p) % 7 == 0 and evs: evs = [[e[0], e[2], e[1]] for e in evs]
                out.append((json.dumps({"events": evs}), 3))
        return out

    same, n_q, shards, rebuilds = {"default": 0, "tiny": 0}, 0, 0, 0
    for l in logs:
        runs = {}
        for name in ("single", "default", "tiny"):
            seen = []
            mem = SiteMemory(registry=l["registry"], domain=l["domain"], connector=l["scenario"],
                             reader=lambda p, seen=seen: fake(p, seen))
            if name == "default":
                mem.site.mem = product.ShardedMemory(l["domain"], mem.b14)
            if name == "tiny":
                mem.site.mem = product.ShardedMemory(l["domain"], mem.b14, keys=3, values=5)
            for t, x in zip(l["text_lines"], l["lines"]):
                mem.observe(rr.message(l, x) if x["kind"] == "human" else t, channel=x["src"])
            runs[name] = (seen, [mem.site.ask(q["q"], q["kind"]) for q in l["questions"]], mem.site.board())
            if name == "tiny":
                shards += len(mem.site.mem.shards); rebuilds += mem.site.mem.rebuilds
        n_q += len(l["questions"])
        for name in ("default", "tiny"):
            same[name] += runs[name] == runs["single"]
    print(f"{split}: {len(logs)} logs, {n_q} questions; identical prompts, answers and board: "
          f"default limits {same['default']}/{len(logs)}, tiny limits {same['tiny']}/{len(logs)} "
          f"({shards} shards, {rebuilds} rebuilds in all)")
