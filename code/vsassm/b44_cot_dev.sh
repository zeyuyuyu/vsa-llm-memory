#!/bin/bash
# B44 CoT selection: the four prompting arms on the first 100 std dev instances (dev only).
for arm in zs-nothink fs-nothink zs-think fs-think; do
  /usr/bin/python3 -u b44_cot.py --arm $arm --variant std --split dev --n 100 --bs 12 --out out/cot/${arm}_std_dev100.json || echo "FAILED $arm"
done
