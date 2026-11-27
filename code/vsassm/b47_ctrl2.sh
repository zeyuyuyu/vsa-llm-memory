#!/bin/bash
# B47 exploratory 2 (prereg/B47_exploratory2.md): lastpar positive control, train + dev only.
cd /path/to/pace-scratch/b47
set -x
/usr/bin/python3 -u b47_ft.py --task gen --variant k32 --n 800 --rank 8 --lr 0.0001 --seed 42 --label lastpar --out out/ctrl/gen_lastpar_k32_n800_r8_lr0.0001_s42.json; echo "rc=$?"
/usr/bin/python3 -u b47_ft.py --task fullgen --variant k32 --n 800 --rank 8 --lr 1e-05 --seed 42 --label lastpar --out out/ctrl/fullgen_lastpar_k32_n800_r8_lr1e-05_s42.json; echo "rc=$?"
