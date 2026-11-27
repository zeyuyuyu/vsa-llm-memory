"""B47 compute ledger: GPU-hours and CPU core-hours of every B47 PACE job from sacct output (b47_out/pace_sacct.txt,
fetched by: sacct -u <user> -S 2026-11-19T05:00 -X -P -o JobID,JobName%60,Account,Partition,AllocTRES%80,Elapsed,
State,NodeList), written to b47_out/compute.json. Local CPU work (bootstrap, verdict) is added by hand in the notes.

  python b47_ledger.py
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))


def hours(el):
    d = 0
    if "-" in el:
        d, el = el.split("-"); d = int(d)
    h, m, s = (int(x) for x in el.split(":"))
    return d * 24 + h + m / 60 + s / 3600


def main():
    from datetime import datetime
    st, claims, a800 = {}, [], 0.0
    lp = os.path.join(HERE, "b47_out", "box_logs", "gpu_ledger.txt")
    for l in (open(lp).read().split("\n") if os.path.exists(lp) else []):
        p_ = l.split()
        if len(p_) != 4:
            continue
        t = datetime.strptime(p_[3], "%Y-%m-%dT%H:%M:%SZ")
        if p_[2] == "start":
            st[(p_[0], p_[1])] = t
        elif (p_[0], p_[1]) in st:
            h = (t - st.pop((p_[0], p_[1]))).total_seconds() / 3600
            claims.append({"worker": p_[0], "gpu": p_[1], "hours": round(h, 2)}); a800 += h
    a800 = round(a800, 2)
    rows = [l.split("|") for l in open(os.path.join(HERE, "b47_out", "pace_sacct.txt")).read().strip().split("\n")[1:]]
    gpu = {"gts-nimam6": 0.0, "gts-nimam6-paid": 0.0}; cpu = {"gts-nimam6": 0.0, "gts-nimam6-paid": 0.0}
    per = []
    for r in rows:
        if len(r) < 7 or not r[1].startswith("b47_"):
            continue
        jid, name, acc, part, tres, el, state = r[:7]
        h = hours(el)
        g = re.search(r"gres/gpu=(\d+)", tres); c = re.search(r"cpu=(\d+)", tres)
        ng = int(g.group(1)) if g else 0; nc = int(c.group(1)) if c else 0
        if ng:
            gpu[acc] = gpu.get(acc, 0) + ng * h
        else:
            cpu[acc] = cpu.get(acc, 0) + nc * h
        per.append({"job": jid, "name": name, "account": acc, "partition": part, "gpus": ng, "cpus": nc,
                    "hours": round(h, 3), "state": state})
    out = {"gpu_hours": round(sum(gpu.values()), 2), "note": "gpu_hours = PACE GPU job time; A800 = worker claim time (GPU 0 was claimed 17:58-02:40 UTC but computed only until 19:28 UTC; the rest is idle claim while the session was paused)", "gpu_hours_by_account": {k: round(v, 2) for k, v in gpu.items()},
           "cpu_core_hours_pace": round(sum(cpu.values()), 2), "cpu_core_hours_by_account": {k: round(v, 2) for k, v in cpu.items()},
           "a800_gpu_hours": a800, "a800_claims": claims, "jobs": per}
    out["total_gpu_hours"] = round(out["gpu_hours"] + a800, 2)
    json.dump(out, open(os.path.join(HERE, "b47_out", "compute.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "jobs"}, indent=1))


if __name__ == "__main__":
    main()
