"""B68 compute summary (no accuracy is read): GPU occupancy per worker from logs/gpu_ledger.txt (start to end of each
claim; a worker still running is counted to now), lane-time of every finished command by kind (dev / final / name),
failures (rc != 0), first and last start of the test commands."""
import glob, os, re
from datetime import datetime
os.chdir("/data0/path/to/HOME/b68")
now = datetime.utcnow()
st, tot, rows = {}, 0.0, []
for l in open("logs/gpu_ledger.txt"):
    p = l.split()
    if len(p) < 4 or p[2] not in ("start", "end"):
        continue
    t = datetime.strptime(p[3], "%Y-%m-%dT%H:%M:%SZ")
    if p[2] == "start":
        st[(p[0], p[1])] = t
    elif (p[0], p[1]) in st:
        h = (t - st.pop((p[0], p[1]))).total_seconds() / 3600; rows.append((p[0], p[1], h, "")); tot += h
for (n, g), t in st.items():
    if n == "dev":
        continue
    h = (now - t).total_seconds() / 3600; rows.append((n, g, h, "(still claimed)")); tot += h
for r in rows:
    print("%-6s %-5s %.2f GPU-h %s" % r)
print("total %.2f A800 GPU-hours" % tot)
kinds = {"dev": [0, 0.0], "final": [0, 0.0], "name": [0, 0.0]}
fails, opened, starts = [], {}, {"final": [], "name": []}
for p in glob.glob("logs/w*.cmds"):
    for l in open(p):
        m = re.match(r"\[(\S+)\] (\S+) (lane\d+) (rc=(\d+) )?(.*)", l.strip())
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
        c = m.group(6)
        k = "name" if "--arm name" in c else ("final" if "--mode final" in c else "dev")
        key = (p, m.group(3), c)
        if m.group(4) is None:
            opened[key] = t
            if k != "dev":
                starts[k].append(t)
        elif key in opened:
            kinds[k][0] += 1; kinds[k][1] += (t - opened.pop(key)).total_seconds() / 3600
            if m.group(5) != "0":
                fails.append(l.strip())
for k, (n, h) in kinds.items():
    print("%-5s %4d finished commands, %.2f lane-hours" % (k, n, h))
print("running now", len(opened))
for k, v in starts.items():
    if v:
        print("%s commands started %s .. %s" % (k, min(v).strftime("%H:%M:%S"), max(v).strftime("%H:%M:%S")))
print("failures (rc != 0):", len(fails))
for f in fails[:10]:
    print("  ", f[:200])
