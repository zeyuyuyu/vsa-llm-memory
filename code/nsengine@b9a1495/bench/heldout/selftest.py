"""Self-test for the held-out families.

    python3 selftest.py                 # all families, levels 1-5, seeds 0-19, with z3
    python3 selftest.py --no-z3 --families vans --seeds 0:5

For every (family, level, seed) it:
  * generates the item twice (must be identical) and round-trips it through JSON;
  * checks the item's fields, and that every slot's entity name appears in the text;
  * checks that the gold answer passes check();
  * checks that an empty answer fails and at least one random wrong answer fails;
  * for "unique" items, recounts solutions with an independent backtracking search;
  * re-runs the family's exact solver and times it (the "solve" column), and checks
    its answer passes check();
  * unless --no-z3: cross-checks the verdict / optimum / uniqueness with z3 and
    feeds z3's (usually different) answer to check(), which must accept it.
Exits non-zero on any failure.  It writes no item files.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import FAMILY_NAMES, LEVELS, load  # noqa: E402

KINDS = {"unique", "find", "optimize", "check"}


def _entity(key):
    parts = key.split(":")
    return parts[1] if len(parts) > 1 else None


def run_one(args):
    fam, level, seed, use_z3, z3_ms = args
    mod = load(fam)
    res = dict(fam=fam, level=level, seed=seed, fails=[], notes={})
    try:
        t0 = time.time()
        item = mod.generate(seed, level)
        res["gen"] = time.time() - t0
        item2 = mod.generate(seed, level)
        if json.dumps(item, sort_keys=True) != json.dumps(item2, sort_keys=True):
            res["fails"].append("non-deterministic generate()")
        item = json.loads(json.dumps(item))
        for k in ("id", "family", "level", "text", "answer_keys", "kind", "gold", "optimum"):
            if k not in item:
                res["fails"].append(f"missing field {k}")
        if item["kind"] not in KINDS:
            res["fails"].append(f"bad kind {item['kind']}")
        if item["level"] != level or item["family"] != fam:
            res["fails"].append("family/level mismatch")
        if "\n" in item["answer_keys"]:
            res["fails"].append("answer_keys not one line")
        if item["kind"] == "optimize" and not isinstance(item["optimum"], int):
            res["fails"].append("optimize item without integer optimum")
        keys = mod.slot_keys(item["data"])
        for key in keys:
            ent = _entity(key)
            if ent and ent not in item["text"]:
                res["fails"].append(f"slot entity {ent!r} not in text")
                break
        gold_imp = bool(item["gold"].get("impossible"))
        if gold_imp and item["kind"] != "check":
            res["fails"].append("impossible gold outside a check family")
        if not gold_imp and set(item["gold"]) != set(keys):
            res["fails"].append("gold keys differ from slot keys")
        res["notes"]["impossible"] = gold_imp
        res["notes"]["optimum"] = item["optimum"]
        res["notes"]["chars"] = len(item["text"])
        # gold must pass
        if not mod.check(item, item["gold"]):
            res["fails"].append("gold fails check()")
        if mod.check(item, {}):
            res["fails"].append("empty answer passes check()")
        rng = random.Random(f"selftest|{fam}|{level}|{seed}")
        wrong = mod.wrong_answers(item, rng, n=20)
        nfail = sum(1 for w in wrong if not mod.check(item, w))
        res["notes"]["wrong_rejected"] = f"{nfail}/{len(wrong)}"
        if nfail == 0:
            res["fails"].append("no random wrong answer was rejected")
        if gold_imp and nfail != len(wrong):
            res["fails"].append("a schedule passed on an impossible item")
        # uniqueness
        if item["kind"] == "unique":
            sols = mod.count_solutions(item["data"], limit=2)
            if len(sols) != 1:
                res["fails"].append(f"unique item has {len(sols)}+ solutions")
        # solver timing (exact solver from the module)
        st = {}
        t0 = time.time()
        ans = mod.solve_item(item, st)
        res["solve"] = time.time() - t0
        if not mod.check(item, ans):
            res["fails"].append("re-solved answer fails check()")
        # z3 cross-check
        if use_z3:
            import z3check
            t0 = time.time()
            verdict, zans = z3check.CHECKS[fam](item, z3_ms)
            res["z3"] = verdict
            res["z3_time"] = time.time() - t0
            if verdict == "disagree":
                res["fails"].append("z3 disagrees with gold")
            if zans is not None and not mod.check(item, zans):
                res["fails"].append("check() rejects a valid z3 answer")
            if zans is not None:
                res["z3_alt"] = zans != item["gold"]
    except Exception:
        res["fails"].append("exception: " + traceback.format_exc(limit=3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0:20")
    ap.add_argument("--families", default=",".join(FAMILY_NAMES))
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--no-z3", action="store_true")
    ap.add_argument("--z3-timeout", type=int, default=60000, help="ms per z3 call")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(":"))
    fams = [f for f in a.families.split(",") if f]
    levels = [int(x) for x in a.levels.split(",")]
    jobs = [(f, L, s, not a.no_z3, a.z3_timeout)
            for f in fams for L in levels for s in range(lo, hi)]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    by = {}
    for r in results:
        by.setdefault((r["fam"], r["level"]), []).append(r)
    total_fail = 0
    hdr = (f"{'family':10s} {'L':>2s} {'n':>3s} {'ok':>3s}  {'gen mean/max s':>15s}  "
           f"{'solve mean/max s':>17s}  {'z3 agree/unk':>12s}  {'z3 alt':>6s}  details")
    print(hdr)
    print("-" * len(hdr))
    for f in fams:
        for L in levels:
            rs = by.get((f, L), [])
            ok = sum(1 for r in rs if not r["fails"])
            total_fail += len(rs) - ok
            g = [r.get("gen", 0) for r in rs]
            s = [r.get("solve", 0) for r in rs]
            z_agree = sum(1 for r in rs if r.get("z3") == "agree")
            z_unk = sum(1 for r in rs if r.get("z3") == "unknown")
            z_alt = sum(1 for r in rs if r.get("z3_alt"))
            det = ""
            kind_notes = [r["notes"] for r in rs if r["notes"]]
            if f == "meetings":
                imp = sum(1 for n in kind_notes if n.get("impossible"))
                det = f"impossible {imp}/{len(rs)}"
            elif kind_notes and kind_notes[0].get("optimum") is not None:
                opts = [n["optimum"] for n in kind_notes]
                det = f"optimum {min(opts)}-{max(opts)}"
            chars = [n.get("chars", 0) for n in kind_notes]
            if chars:
                det += f"; text {min(chars)}-{max(chars)} chars"
            ztxt = f"{z_agree}/{z_unk}" if not a.no_z3 else "-"
            print(f"{f:10s} {L:>2d} {len(rs):>3d} {ok:>3d}  "
                  f"{sum(g) / max(1, len(g)):7.2f}/{max(g, default=0):7.2f}  "
                  f"{sum(s) / max(1, len(s)):8.3f}/{max(s, default=0):8.3f}  "
                  f"{ztxt:>12s}  {z_alt if not a.no_z3 else '-':>6}  {det}")
            for r in rs:
                for msg in r["fails"]:
                    print(f"    FAIL {f} L{L} seed {r['seed']}: {msg.strip()}")
    print(f"\n{len(results)} items, {total_fail} with failures, "
          f"{time.time() - t0:.1f}s wall")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
