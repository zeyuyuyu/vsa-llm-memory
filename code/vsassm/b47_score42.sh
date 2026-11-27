#!/bin/bash
# B47: score the test split ONCE from the saved dev-best state of the seed-42 tuning run of each selected configuration
# (out/select.json, written from the tuning runs' train/dev numbers only).
cd /path/to/pace-scratch/b47
set -x
/usr/bin/python3 -u b47_ft.py --task gen --variant k32 --n 800 --rank 64 --lr 0.0003 --seed 42 --score_test --out out/ft/gen_k32_n800_r64_lr0.0003_s42.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullgen --variant k32 --n 800 --rank 8 --lr 3e-06 --seed 42 --score_test --out out/ft/fullgen_k32_n800_r8_lr3e-06_s42.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullcls --variant k32 --n 800 --rank 8 --lr 3e-05 --seed 42 --score_test --out out/ft/fullcls_k32_n800_r8_lr3e-05_s42.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task scratch --variant k32 --n 800 --rank 8 --lr 0.0003 --seed 42 --score_test --out out/ft/scratch_k32_n800_r8_lr0.0003_s42.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task gen --variant std --n 8000 --rank 64 --lr 3e-05 --seed 42 --score_test --out out/ft/gen_std_n8000_r64_lr3e-05_s42.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullgen --variant std --n 8000 --rank 8 --lr 3e-06 --seed 42 --score_test --out out/ft/fullgen_std_n8000_r8_lr3e-06_s42.json; echo "rc=$?"
