"""B65 queue chaining (procedure only; the registered dev rules in b65_plan.py are unchanged). Keeps the lanes of the
workers that already hold a GPU busy across the dependent dev phases of one position (edge extensions, seeds 1-2 readers,
seeds 1-2 write-backs, then, once the dev choices are committed, the test), so that no new GPU claim is needed.
Lines of this script are put into a live worker queue:
  hold <queue> <gen>                         block until <queue>.gen<gen> exists (at most 3 h), then exit 0
  step <pos> <queue> <stage> <gen> <nhold>   stage W: wait until every seed-0 run of the position's current grid has
        its train.json; b65_plan.py W <pos> (flock); extensions -> append them, nhold holders and a new W step; none
        -> b65_plan.py F <pos>; append the seed-1/2 reader lines, holders and an R step.
        stage R: wait for those readers; append the seed-1/2 write-back lines, holders and a T step.
        stage T: wait for those write-backs AND for the file TEST_OK (created after the dev choices are committed);
        b65_plan.py T; append this position's test lines (and, for last, the determinism re-runs).
        Every step releases generation <gen> after appending, so the blocked holders' lanes pop the new lines.
"""
import fcntl, json, os, subprocess, sys, time

B = os.environ.get("B65_ROOT", "/path/to/work/b65")
os.chdir(B)
PY = sys.executable
ARMS = ("vsapub", "vsabm", "disc")
WB_GRID = ("3e-4", "1e-3", "3e-3", "1e-2")
MAXWAIT = 3 * 3600


def log(*a):
    with open("logs/chain.log", "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + " ".join(map(str, a)) + "\n")


def append(queue, lines):
    with open(queue + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(queue, "a") as f:
            for l in lines:
                f.write(l + "\n")
    log("appended", len(lines), "to", queue)


def plan(*args):
    with open("out/choices.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        r = subprocess.run([PY, "b65_plan.py", *args], capture_output=True, text=True)
    log("plan", *args, "rc", r.returncode, r.stdout.strip().replace("\n", " | "), r.stderr.strip()[-300:])
    if r.returncode:
        raise SystemExit(f"plan {args} failed")


def wait_for(pred, what):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > MAXWAIT:
            log("TIMEOUT waiting for", what)
            raise SystemExit(1)
        time.sleep(15)
    log("ready:", what)


def lines(path):
    return [l for l in open(path).read().splitlines() if l.strip()]


def outdir(line):
    t = line.split()
    return t[t.index("--out") + 1]


def grid_done(pos):
    c = json.load(open("out/choices.json"))
    for arm in ARMS:
        for lr in c.get("wb_grid", {}).get(f"{pos}_{arm}", WB_GRID):
            if not os.path.exists(f"out/dev/{pos}_{arm}_lr{lr}_s0/train.json"):
                return False
    return True


def main():
    cmd = sys.argv[1]
    if cmd == "hold":
        q, g = sys.argv[2], sys.argv[3]
        wait_for(lambda: os.path.exists(f"{q}.gen{g}"), f"hold {q} gen{g}")
        return
    pos, q, stage, g, nh = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    holders = [f"b65_chain.py hold {q} {g + 1}"] * nh
    nxt = lambda st: f"b65_chain.py step {pos} {q} {st} {g + 1} {nh}"
    if stage == "W":
        wait_for(lambda: grid_done(pos), f"grid {pos}")
        plan("W", pos)
        ext = lines(f"q/W_ext_{pos}")
        if ext:
            append(q, ext + holders + [nxt("W")])
        else:
            plan("F", pos)
            append(q, lines(f"q/F_rd_{pos}") + holders + [nxt("R")])
    elif stage == "R":
        rd = lines(f"q/F_rd_{pos}")
        wait_for(lambda: all(os.path.exists(f"{outdir(l)}/reader.json") for l in rd), f"seed readers {pos}")
        append(q, lines(f"q/F_wb_{pos}") + holders + [nxt("T")])
    elif stage == "T":
        wb = lines(f"q/F_wb_{pos}")
        wait_for(lambda: all(os.path.exists(f"{outdir(l)}/train.json") for l in wb), f"seed write-backs {pos}")
        wait_for(lambda: os.path.exists("TEST_OK"), "TEST_OK (dev choices committed)")
        with open("out/choices.lock", "a") as lk:  # one plan T for both positions
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists("q/T_test"):
                r = subprocess.run([PY, "b65_plan.py", "T"], capture_output=True, text=True)
                log("plan T rc", r.returncode, r.stdout.strip().replace("\n", " | "))
        mine = [l for l in lines("q/T_test") if f"--pos {pos} " in l]
        if pos == "last":
            mine += lines("q/T_rerun")
        append(q, mine)
    open(f"{q}.gen{g}", "w").close()
    log("released", q, "gen", g)


if __name__ == "__main__":
    main()
