#!/bin/bash
# B48 features in one GPU job (prereg/B48_one_form_and_alias_naming_prereg.md): self-tests (CPU, both tokenizers),
# C1 features (Phi-2, Qwen3-8B; train, dev, test), story features (Qwen3-8B layers 9/18/27/36; W, P; train, dev, test).
# usage (inside a job): bash b48_feats.sh
set -x
/usr/bin/python3 -u b48_prep.py selftest --enc phi2 --out out
/usr/bin/python3 -u b48_prep.py selftest --enc qwen8 --out out
/usr/bin/python3 -u b48_prep.py featsC1 --enc phi2 --feats feats
/usr/bin/python3 -u b48_prep.py featsC1 --enc qwen8 --feats feats
/usr/bin/python3 -u b48_prep.py featsB --items W:train W:dev W:test P:train P:dev P:test --feats feats
