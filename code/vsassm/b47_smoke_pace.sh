#!/bin/bash
# B47 smoke on PACE (training split only: --smoke 64 trains on the first 64 training instances for one epoch and uses
# the same 64 as "dev"; no dev or test split is read). Times and memory of every task; the full fine-tune is first
# stopped by a tiny --max_hours (exit 3, checkpoint) and resumed.
cd /path/to/pace-scratch/b47
set -x
/usr/bin/python3 -u b47_ft.py --task gen --variant k32 --n 800 --rank 64 --lr 1e-4 --smoke 64 --out out/smoke/gen.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullgen --variant k32 --n 800 --lr 1e-5 --smoke 64 --max_hours 0.001 --out out/smoke/fullgen.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullgen --variant k32 --n 800 --lr 1e-5 --smoke 64 --out out/smoke/fullgen.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullcls --variant k32 --n 800 --lr 1e-5 --smoke 64 --out out/smoke/fullcls.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task scratch --variant k32 --n 800 --lr 1e-4 --smoke 64 --out out/smoke/scratch.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullgen --variant std --n 8000 --lr 1e-5 --smoke 64 --out out/smoke/fullgen_std.json; echo "rc=$?"
