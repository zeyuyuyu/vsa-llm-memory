"""V-131 Part A: add a PURELY ADDITIVE --dump_rows flag to v65.py.

v65 already computes `hits` per row and throws the vector away after taking its mean. Every downstream
question the 2026-09-11 audit could not answer -- a paired interval for hrr_gold - prompt_ev, and the
answerable-subset recompute V-125 recorded as "phi2_v65: per-seed means only, ABSENT" -- needs that vector
and nothing else. So the patch stores it. No control flow, no arithmetic and no default behaviour changes,
which is why P0 (the re-run reproducing the banked per-seed means to +-.01) is a meaningful gate on the patch
as well as on the environment.
"""
import re

P = "/path/to/work/ngcls/v65.py"
s = open(P).read()

if "--dump_rows" in s:
    print("already patched"); raise SystemExit(0)

# 1. the flag
old = '    ap.add_argument("--out", default="v65_logs/v65.json")\n    a = ap.parse_args()\n'
new = ('    ap.add_argument("--out", default="v65_logs/v65.json")\n'
       '    ap.add_argument("--dump_rows", default="", help="V-131: also write per-row hits here")\n'
       '    a = ap.parse_args()\n')
assert s.count(old) == 1, "flag anchor %d" % s.count(old)
s = s.replace(old, new)

# 2. the collector, declared beside the output dict
old = '    out = {"n": len(rows), "coverage": len(covered) / len(rows), "arms": {}, "trunc_rate": None}\n'
new = (old +
       '    row_dump = {"n": len(rows), "covered": covered, "seeds": list(a.seeds), "arms": {}}\n')
assert s.count(old) == 1, "outdict anchor"
s = s.replace(old, new)

# 3. store the vector where it is already materialised
old = ('            hits = np.array(hits)\n'
       '            per_seed_full.append(float(hits.mean()))\n')
new = ('            hits = np.array(hits)\n'
       '            if a.dump_rows:\n'
       '                row_dump["arms"].setdefault(arm, {})[str(seed)] = [int(h) for h in hits]\n'
       '            per_seed_full.append(float(hits.mean()))\n')
assert s.count(old) == 1, "hits anchor"
s = s.replace(old, new)

# 4. write it out beside the verdict
old = '    print("  >>> V-65 VERDICT: %s" % out["verdict"], flush=True)\n'
new = (old +
       '    if a.dump_rows:\n'
       '        json.dump(row_dump, open(a.dump_rows, "w"))\n'
       '        print("  per-row hits -> %s" % a.dump_rows, flush=True)\n')
assert s.count(old) == 1, "verdict anchor"
s = s.replace(old, new)

open(P, "w").write(s)
print("patched v65.py: --dump_rows added (4 insertions, 0 deletions)")
