#!/bin/bash
# B44 smoke on PACE (train and dev only; no test split is read): LoRA cls and scratch on 64 training instances
# (one epoch, dev = the same 64), the cls run first stopped by a tiny --max_hours (exit 3) and resumed, CoT on 16 dev
# instances (zs-think, fs-nothink), and the features of the std dev split (timing).
cd /path/to/pace-scratch/b44
set -x
/usr/bin/python3 -u b44_lora.py --task cls --variant std --n 800 --lr 3e-4 --smoke 64 --max_hours 0.001 --out out/smoke/cls.json; echo "rc=$?"
/usr/bin/python3 -u b44_lora.py --task cls --variant std --n 800 --lr 3e-4 --smoke 64 --out out/smoke/cls.json; echo "rc=$?"
/usr/bin/python3 -u b44_lora.py --task scratch --variant std --n 800 --lr 3e-4 --smoke 64 --out out/smoke/scratch.json; echo "rc=$?"
/usr/bin/python3 -u b44_cot.py --arm fs-nothink --variant std --split dev --n 16 --bs 16 --out out/smoke/cot_fs-nothink_dev16.json; echo "rc=$?"
/usr/bin/python3 -u b44_cot.py --arm zs-think --variant std --split dev --n 16 --bs 16 --out out/smoke/cot_zs-think_dev16.json; echo "rc=$?"
/usr/bin/python3 -u b44_feats.py --enc qwen8 --items std:dev --feats feats_smoke; echo "rc=$?"
