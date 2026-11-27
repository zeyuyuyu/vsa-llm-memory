"""B73, equal-cell part (prereg/B73_cells_prereg.md): the paper's device memories, with every design frozen at the
choice made on development stores in B38 / B61 / B63, on 50 FRESH key-value stores from an independent generator.

Memories (all code imported unchanged: b38_eq, b61_hd, b63_online through b69_vocab.run_job):
  U       dense bipolar store with a LAST code (B38's V-last); no design parameter.
  H-SBC   sparse block code store (B61); block load lam = B61's dev choice at (condition, budget).
  T-rep   minimal-perfect-hash table, repetition-coded (B38); no design parameter.
  E-BCH   the same table in BCH codewords (B38); design = B38's closed-form rule per store ('auto').
  E-LDPC  the same table in IRA LDPC codewords (B38); rate R = B38's dev choice at (condition, budget).
  O-BCH   cuckoo table of key fingerprints, BCH-protected (B63); f = B63's dev choice at (condition, budget).
  O-LDPC  the same, LDPC-protected (B63); (f, R) = B63's dev choice at (condition, budget).
  static = per condition the smallest N* of T-rep, E-BCH, E-LDPC; insert = the smallest of O-BCH, O-LDPC.
Conditions: k1, k8, k16 (independent cell errors at the device model's 30-day rate x 1, 8, 16; B38's IID).
N*: the smallest grid budget (cells per stored pair) whose pooled share of probed keys read exactly is >= 99.5% there
and at every larger evaluated budget (B38 / B58's rule), pooled over the 50 stores.

python3 b73_cells.py frozen OUT.json                 # the frozen design table (no store read)
python3 b73_cells.py selftest                        # synthetic stores, seed tag 'b73cells-selftest' (not the test's)
python3 b73_cells.py stores SPLIT OUT.json.gz        # SPLIT = test (the registered fresh stores)
python3 b73_cells.py plan FROZEN.json OUT_PLAN.json
python3 b73_cells.py run STORES.json.gz PLAN.json OUT.json [--workers N]
python3 b73_cells.py extend FROZEN.json PLAN.json[,..] TEST.json[,..] OUT_PLAN.json
"""
import collections
import gzip
import hashlib
import json
import random
import re
import sys
import time

import b38_eq as E
import b69_vocab as B69

GRID = E.GRID
NG = len(GRID)
TARGET = E.TARGET
CONDS = ("k1", "k8", "k16")
MEMS = ("U", "H-SBC", "T-rep", "E-BCH", "E-LDPC", "O-BCH", "O-LDPC")
STATIC = ("T-rep", "E-BCH", "E-LDPC")
INSERT = ("O-BCH", "O-LDPC")
NPROBE = 150                                   # b25_pcm.NKEYS, the paper's probe rule (every key of a small store)
WINDOW = 4                                     # test budgets: the paper's dev N* +- 4 grid steps (B38 / B63's rule)
WORDS_FILE = "/usr/share/dict/american-english"
WORDS_MD5 = "16de2454dee65e9ceed77f9c1cd8a15e"
FAMILIES = ("wp6k", "code1k5", "hex1k7", "mv", "tiny")
REPS = 10
# the paper's development N* (window centres; B38 dev for U, T-rep, E-BCH, E-LDPC; B61 dev for H-SBC; B63 dev for O-*)
SOURCES = {"U": ("results/b38_choice.json", "V-last"), "T-rep": ("results/b38_choice.json", "T-rep"),
           "E-BCH": ("results/b38_choice.json", "E-BCH"), "E-LDPC": ("results/b38_choice.json", "E-LDPC"),
           "H-SBC": ("results/b61_choice.json", "H-SBC"), "O-BCH": ("results/b63_choice.json", "O-BCH"),
           "O-LDPC": ("results/b63_choice.json", "O-LDPC")}
# the paper's (workload) test N*, pooled over the 50 RULER stores, uncharged (results/b69_partA.json, reproduced there
# from the B38 / B59 / B61 / B63 rows)
PAPER = {"U": {"k1": 76, "k8": 128, "k16": 431}, "H-SBC": {"k1": 32, "k8": 91, "k16": 304},
         "T-rep": {"k1": 19, "k8": 256, "k16": 1024}, "E-BCH": {"k1": 19, "k8": 108, "k16": 431},
         "E-LDPC": {"k1": 23, "k8": 54, "k16": 215}, "O-BCH": {"k1": 32, "k8": 181, "k16": 724},
         "O-LDPC": {"k1": 32, "k8": 91, "k16": 304}, "static": {"k1": 19, "k8": 54, "k16": 215},
         "insert": {"k1": 32, "k8": 91, "k16": 304}}


def gidx(B):
    return min(range(NG), key=lambda i: abs(GRID[i] - B))


def fmt(gi):
    return ">4096" if gi is None or gi >= NG else "%.0f" % GRID[gi]


# =============================================================================================== fresh stores
_W = None


def words():
    """Lower-case alphabetic words of 4-8 letters from the system dictionary (not wonderwords, which RULER and B34's
    generators use), sorted and unique."""
    global _W
    if _W is None:
        raw = open(WORDS_FILE, "rb").read()
        if hashlib.md5(raw).hexdigest() != WORDS_MD5:
            raise SystemExit("word list MD5 differs from the registered one")
        _W = sorted({w for w in raw.decode("utf-8").split("\n") if re.fullmatch(r"[a-z]{4,8}", w)})
    return _W


def _distinct(rng, n, draw, taken=()):
    out, seen = [], set(taken)
    while len(out) < n:
        x = draw()
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_store(fam, split, rep, tag="b73cells"):
    """One fresh store: (sid, pairs), pairs = [(key, value, j)], j = 0 .. count - 1 per key in store order.
    Families (10 stores each) mirror the sizes of the paper's five RULER families with other key / value formats:
      wp6k    5,975-5,995 keys 'word/word', 2-5 of them with two values; values 'word word' (two dictionary words)
      code1k5 1,460-1,495 keys 'word/word', one value each; values 'ABCD-123' (4 capitals, hyphen, 3 digits)
      hex1k7  1,695-1,725 keys of 24 hex digits, one value each; values 24 hex digits
      mv      18 keys 'word/word': 15 with one value, 2 with two, 1 with four (23 pairs); values 'word word'
      tiny    keys 'xy_7' (2 letters, underscore, digit); rep 0: 9 keys, 9 values; reps 1-9: 10 keys, 7 with one value
              and 3 with two (13 pairs) over a vocabulary of 9 six-digit numbers shared across keys (each used >= once)
    A key's values are distinct. Seeded by random.Random('{tag}:{split}:{fam}:{rep}')."""
    W = words()
    rng = random.Random("%s:%s:%s:%d" % (tag, split, fam, rep))
    wkey = lambda: "%s/%s" % (rng.choice(W), rng.choice(W))  # noqa: E731
    wval = lambda: "%s %s" % (rng.choice(W), rng.choice(W))  # noqa: E731
    hexs = lambda: "%024x" % rng.getrandbits(96)  # noqa: E731
    if fam == "wp6k":
        nk = rng.randint(5975, 5995)
        keys = _distinct(rng, nk, wkey)
        counts = [1] * nk
        for i in rng.sample(range(nk), rng.randint(2, 5)):
            counts[i] = 2
        vdraw = wval
    elif fam == "code1k5":
        nk = rng.randint(1460, 1495)
        keys = _distinct(rng, nk, wkey)
        counts = [1] * nk
        vdraw = lambda: "%s-%03d" % ("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)),  # noqa: E731
                                     rng.randrange(1000))
    elif fam == "hex1k7":
        nk = rng.randint(1695, 1725)
        keys = _distinct(rng, nk, hexs)
        counts = [1] * nk
        vdraw = hexs
    elif fam == "mv":
        keys = _distinct(rng, 18, wkey)
        counts = [1] * 15 + [2] * 2 + [4]
        rng.shuffle(counts)
        vdraw = wval
    elif fam == "tiny":
        kd = lambda: "%s%s_%d" % (rng.choice("abcdefghijklmnopqrstuvwxyz"), rng.choice("abcdefghijklmnopqrstuvwxyz"),  # noqa: E731
                                  rng.randrange(10))
        vocab = _distinct(rng, 9, lambda: "%06d" % rng.randrange(10 ** 6))
        if rep == 0:
            keys = _distinct(rng, 9, kd)
            counts = [1] * 9
            vals = list(vocab)
            rng.shuffle(vals)
        else:
            keys = _distinct(rng, 10, kd)
            counts = [1] * 7 + [2] * 3
            rng.shuffle(counts)
            while True:                                          # every vocabulary value used; a key's values distinct
                vals = list(vocab) + [rng.choice(vocab) for _ in range(sum(counts) - 9)]
                rng.shuffle(vals)
                o, ok = 0, True
                for c in counts:
                    ok &= len(set(vals[o:o + c])) == c
                    o += c
                if ok:
                    break
        pairs, o = [], 0
        for k, c in zip(keys, counts):
            for j in range(c):
                pairs.append((k, vals[o], j))
                o += 1
        return ("b73/%s/%s/%d" % (fam, split, rep), pairs)
    else:
        raise SystemExit(fam)
    pairs = []
    for k, c in zip(keys, counts):
        vs = _distinct(rng, c, vdraw)
        for j, v in enumerate(vs):
            pairs.append((k, v, j))
    return ("b73/%s/%s/%d" % (fam, split, rep), pairs)


def make_stores(split, tag="b73cells", reps=REPS, fams=FAMILIES):
    return [make_store(f, split, r, tag) for f in fams for r in range(reps)]


def write_stores(split, out_path, tag="b73cells"):
    st = make_stores(split, tag)
    json.dump([[s, p] for s, p in st], gzip.open(out_path, "wt"))
    print(split, len(st), "stores,", sum(len(p) for _, p in st), "pairs ->", out_path, flush=True)


def family(sid):
    return sid.split("/")[1]


# =============================================================================================== frozen designs
def _design_from(mem, ch, src, cond, gi):
    """The paper's dev-chosen design of `mem` at (cond, gi); nearest dev budget where gi was not evaluated on dev
    (ties: the larger budget; b61_hd.design_at / b63_online.design_at's rule)."""
    if mem in ("U", "T-rep"):
        return None
    if mem == "E-BCH":
        return "auto"
    des = ch["designs"][src][cond]
    if str(gi) in des:
        d = des[str(gi)]
        via = gi
    else:
        via = min((int(g) for g in des), key=lambda g: (abs(g - gi), -g))
        d = des[str(via)]
    if mem == "E-LDPC":
        d = json.loads(d)
    elif mem == "O-BCH":
        d = int(d[0])
    elif mem == "O-LDPC":
        d = [int(d[0]), float(d[1])]
    else:
        d = float(d)
    return d, via


def frozen(out_path):
    """Per (memory, condition): the paper's dev N*, the window of budgets (dev N* +- 4), and the design at every grid
    budget (resolved now, before any fresh store exists)."""
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
    out = dict(rule="design at (memory, condition, budget) = the paper's dev choice there; nearest dev budget if not "
                    "evaluated on dev (ties: larger); U, T-rep: none; E-BCH: B38's per-store closed form 'auto'",
               sources={p: md5(p) for p in sorted({s for s, _ in SOURCES.values()})},
               code={p: md5(p) for p in ("b38_eq.py", "b61_hd.py", "b63_online.py", "b69_vocab.py", "b34_inmem.py",
                                         "b25_pcm.py", "ruler_vsa.py", "b58_verdict.py")},
               words={"file": WORDS_FILE, "md5": WORDS_MD5}, paper_test_nstar=PAPER, mems={})
    for mem in MEMS:
        path, src = SOURCES[mem]
        ch = json.load(open(path))
        out["mems"][mem] = {}
        for c in CONDS:
            ns = ch["dev_nstar"][src][c]
            g0 = gidx(ns)
            win = list(range(max(0, g0 - WINDOW), min(NG, g0 + WINDOW + 1)))
            des = {}
            for gi in range(NG):
                r = _design_from(mem, ch, src, c, gi)
                des[str(gi)] = None if r is None else (r if mem == "E-BCH" else {"design": r[0], "dev_budget_gi": r[1]})
            out["mems"][mem][c] = dict(dev_nstar=ns, dev_nstar_gi=g0, window=win, designs=des)
    json.dump(out, open(out_path, "w"), indent=1, sort_keys=True)
    print("frozen designs ->", out_path)
    return out


def design_of(fz, mem, c, gi):
    d = fz["mems"][mem][c]["designs"][str(gi)]
    if d is None or d == "auto":
        return d
    return d["design"]


# =============================================================================================== plans and jobs
def plan(fz, out_path, extra=None):
    """need[(mem, design_json)][gi] = conditions. extra: {(mem, c): [gi, ...]} (extension) replaces the windows."""
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for mem in MEMS:
        for c in CONDS:
            gis = fz["mems"][mem][c]["window"] if extra is None else extra.get((mem, c), [])
            for gi in gis:
                need[(mem, json.dumps(design_of(fz, mem, c, gi)))][gi].add(c)
    json.dump([[list(k), {str(g): sorted(cs) for g, cs in v.items()}] for k, v in need.items()],
              open(out_path, "w"))
    n = sum(len(cs) for v in need.values() for cs in v.values())
    print(len(need), "(memory, design) entries;", n, "evaluations per store ->", out_path)
    return need


def load_plan(path):
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for k, v in json.load(open(path)):
        for g, cs in v.items():
            need[tuple(k)][int(g)] |= set(cs)
    return need


UNIT = {"U": 3.0, "H-SBC": 2.0, "T-rep": 0.3, "E-BCH": 1.0, "E-LDPC": 2.0, "O-BCH": 1.0, "O-LDPC": 2.0}


def make_jobs(stores, need, cap=12):
    jobs = []
    for sid, pairs in stores:
        n = len(pairs)
        for (mem, dj), bud in sorted(need.items()):
            ev = [(GRID[gi], sorted(cs)) for gi, cs in sorted(bud.items())]
            cp = 3 if (mem == "U" and n > 3000) else cap
            chunk = []

            def flush(ch):
                jobs.append(dict(sid=sid, pairs=pairs, extra=None, mem=mem, design=json.loads(dj), evals=ch,
                                 nprobe=NPROBE, cost=UNIT[mem] * sum(b * len(c) for b, c in ch) * (1 + n / 1500)
                                 + 5 * (1 + n / 1500)))
            for e in ev:
                if chunk and sum(len(c) for _, c in chunk) + len(e[1]) > cp:
                    flush(chunk)
                    chunk = []
                chunk.append(e)
            if chunk:
                flush(chunk)
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def run_one(job):
    """b69_vocab.run_job (unchanged) with no vocabulary patch and the paper's probe rule."""
    r = B69.run_job(job)
    r["n_pairs_store"] = len(job["pairs"])
    return r


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done, last, t0 = [], time.time(), time.time()
    with Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(run_one, jobs, chunksize=1):
            done.append(r)
            print("done %-22s %-6s %6.0fs %d/%d (%.0f s elapsed)" % (r["sid"], r["mem"], r["sec"], len(done),
                                                                     len(jobs), time.time() - t0), flush=True)
            if time.time() - last > 120:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def jl(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def counts(fz, paths):
    """C[(mem, cond)][gi][sid] = (failed, n) at the frozen design of (mem, cond, gi); infeasible rows are skipped (the
    store is not evaluated there, so that budget does not count: B38 / B69's convention)."""
    C = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in paths:
        for r in jl(p):
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                gi = gidx(x["B"])
                if json.dumps(r["design"]) != json.dumps(design_of(fz, r["mem"], x["cond"], gi)):
                    continue
                C[(r["mem"], x["cond"])][gi][r["sid"]] = (len(x["failed"]), x["n"])
    return C


def nstar(curve, sids):
    return B69.nstar_counts(curve, sids)


def extend(fz, plan_paths, test_paths, out_path):
    """B38's extension rule, one round: if the top evaluated budget fails the bar, the next 4 grid steps up; else if N*
    is the lowest evaluated budget (> grid bottom), the 4 steps below."""
    sids = sorted({r["sid"] for p in test_paths for r in jl(p)})
    C = counts(fz, test_paths)
    planned = set()
    for p in plan_paths:
        for (mem, dj), v in load_plan(p).items():
            for g, cs in v.items():
                for c in cs:
                    planned.add((mem, c, g))
    extra = {}
    for mem in MEMS:
        for c in CONDS:
            ns, sh = nstar(C.get((mem, c), {}), sids)
            if not sh:
                continue
            top, low = max(sh), min(sh)
            add = []
            if sh[top] < TARGET and top < NG - 1:
                add = list(range(top + 1, min(NG, top + 5)))
            elif ns == low and low > 0:
                add = list(range(max(0, low - 4), low))
            add = [g for g in add if (mem, c, g) not in planned]
            if add:
                extra[(mem, c)] = add
    print("extension:", sum(len(v) for v in extra.values()), "(memory, condition, budget) additions")
    plan(fz, out_path, extra)
    return extra


# =============================================================================================== self-test
def selftest():
    """Synthetic stores only (seed tag 'b73cells-selftest', so none of them is a test store): generator invariants,
    one evaluation per memory on a small store of each family at k1 and k8 (cells within budget), timing."""
    ok = True
    W = words()
    print("words", len(W), W[:3], W[-3:])
    st = make_stores("selftest", "b73cells-selftest", reps=2)
    for sid, pairs in st:
        ks = collections.Counter(k for k, _, _ in pairs)
        per = collections.defaultdict(list)
        for k, v, j in pairs:
            per[k].append((j, v))
        good = all(sorted(j for j, _ in vs) == list(range(len(vs))) and len({v for _, v in vs}) == len(vs)
                   for vs in per.values())
        ok &= good
        print("  %-28s pairs %5d keys %5d values %5d multi %s %s" % (
            sid, len(pairs), len(ks), len({v for _, v, _ in pairs}), dict(collections.Counter(ks.values())),
            "ok" if good else "BAD"))
    # determinism of the generator
    a, b = make_store("wp6k", "selftest", 0, "b73cells-selftest"), make_store("wp6k", "selftest", 0, "b73cells-selftest")
    ok &= a == b
    fz = json.load(open(sys.argv[sys.argv.index("--fz") + 1] if "--fz" in sys.argv else "results/b73_cells_frozen.json"))
    small = [s for s in st if s[0].endswith("/0")]
    for sid, pairs in small:
        for mem in MEMS:
            for c in ("k1", "k8"):
                gi = fz["mems"][mem][c]["dev_nstar_gi"]
                d = design_of(fz, mem, c, gi)
                if len(pairs) > 3000 and mem not in ("T-rep",) and c == "k8":
                    continue
                t0 = time.time()
                r = run_one(dict(sid=sid, pairs=pairs, extra=None, mem=mem, design=d, evals=[(GRID[gi], [c])],
                                 nprobe=NPROBE))
                x = r["rows"][0]
                over = (not x["infeasible"]) and x["cells"] > x["budget"]
                ok &= not over
                print("  %-26s %-7s %-4s B %6.1f design %-16s exact %s/%s cells %s budget %s %.1fs" % (
                    sid, mem, c, GRID[gi], json.dumps(d), x.get("exact"), x.get("n"), x.get("cells"),
                    x.get("budget"), time.time() - t0), flush=True)
    # N* rule
    cur = {10: {"a": (5, 100)}, 11: {"a": (0, 100)}, 12: {"a": (1, 100)}, 13: {"a": (0, 100)}}
    ok &= nstar(cur, ["a"])[0] == 13
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = lambda k, d: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)  # noqa: E731
    if cmd == "frozen":
        frozen(sys.argv[2])
    elif cmd == "selftest":
        selftest()
    elif cmd == "stores":
        if sys.argv[2] != "test":
            raise SystemExit("only the registered split 'test'")
        write_stores("test", sys.argv[3])
    elif cmd == "plan":
        plan(json.load(open(sys.argv[2])), sys.argv[3])
    elif cmd == "run":
        stores = E.load_stores(sys.argv[2])
        jobs = make_jobs(stores, load_plan(sys.argv[3]))
        print("jobs", len(jobs), "evaluations", sum(sum(len(c) for _, c in j["evals"]) for j in jobs), flush=True)
        pool_run(jobs, int(arg("--workers", 8)), sys.argv[4])
    elif cmd == "extend":
        extend(json.load(open(sys.argv[2])), sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5])
    else:
        raise SystemExit(cmd)
