"""B68 progress (no accuracy is read): per cell, finished dev runs out of 64 and finished per-seed finals out of 18;
queue lengths; mean wall seconds of finished dev runs (from the worker command logs)."""
import glob, os, re, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b68_queues import ORDER, tune_files, key
os.chdir("/data0/path/to/HOME/b68")
tot_t = tot_f = 0
for enc, ar in ORDER:
    k = key(enc, ar)
    n = sum(os.path.exists(o) for _, _, o in tune_files(enc, ar))
    f = len(glob.glob("out/final_seed/%s_*_s*.json" % k))
    nm = os.path.exists("out/name/%s_name.json" % k)
    tot_t += n; tot_f += f
    print("%-10s dev %2d/64  finals %2d/18  name %s" % (k, n, f, "done" if nm else "-"))
for q in ("q/tune", "q/final"):
    print(q, sum(1 for _ in open(q)) if os.path.exists(q) else 0)
st, durs = {}, []
for p in glob.glob("logs/w*.cmds"):
    for l in open(p):
        m = re.match(r"\[(\S+)\] (\S+) (lane\d+) (rc=\d+ )?(.*)", l.strip())
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
        kk = (p, m.group(3), m.group(5))
        if m.group(4):
            if kk in st:
                durs.append(((t - st.pop(kk)).total_seconds(), m.group(5)))
        else:
            st[kk] = t
print("finished runs %d, running %d, mean %.0f s" % (len(durs), len(st), sum(d for d, _ in durs) / max(1, len(durs))))
print("dev done %d/1152, finals done %d/324" % (tot_t, tot_f))
