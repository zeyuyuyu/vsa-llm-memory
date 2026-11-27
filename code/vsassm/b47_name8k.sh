#!/bin/bash
# B47 secondary: the Phi-2 naming head (B31 arm, b44_name.py unchanged) trained on B44's first 8,000 std instances;
# Phi-2 fp16 frozen features by b44_feats.py (unchanged, B31's recipe). Test scored at each seed's dev-selected epoch.
cd /path/to/pace-scratch/b47
set -x
/usr/bin/python3 -u b44_feats.py --enc phi2 --items std:train std:dev std:test --data data --feats feats; echo "rc=$?"
/usr/bin/python3 -u b44_name.py --cell std:8000 --enc phi2 --arm name --feats feats --out out/name/phi2_name_std_n8000.json; echo "rc=$?"
