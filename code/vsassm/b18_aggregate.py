"""B18: combine the test-array summaries -> per length x task accuracy (official metric), QA1-5 mean, in-scope mean.
python3 b18_aggregate.py b18_test/student_s0"""
import glob
import json
import sys

LENS = ["0k", "4k", "16k", "64k", "128k", "512k", "1M"]
acc = {}
for f in glob.glob(f"{sys.argv[1]}/*/summary.json"):
    for k, v in json.load(open(f)).items():
        acc[k] = v
tasks = sorted({k.split("/")[1] for k in acc}, key=lambda t: int(t[2:]))
print("task  " + "  ".join(f"{l:>6s}" for l in LENS))
for t in tasks:
    print(f"{t:5s} " + "  ".join(f"{acc[f'{l}/{t}']['acc']:6.3f}" if f"{l}/{t}" in acc else "     -" for l in LENS))
for name, ts in (("QA1-5", ["qa1", "qa2", "qa3", "qa4", "qa5"]), ("qa1-10", [f"qa{i}" for i in range(1, 11)])):
    row = []
    for l in LENS:
        vals = [acc[f"{l}/{t}"]["acc"] for t in ts if f"{l}/{t}" in acc]
        row.append(f"{100 * sum(vals) / len(vals):6.1f}" if len(vals) == len(ts) else "     -")
    print(f"{name:6s}" + "  ".join(row))
n = {l: sorted({acc[k]['n'] for k in acc if k.startswith(l + '/')}) for l in LENS}
print("samples per task:", {l: v for l, v in n.items() if v})
