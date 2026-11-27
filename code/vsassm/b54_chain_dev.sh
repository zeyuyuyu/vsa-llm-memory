#!/bin/bash
# after dev1: T-rep dev1 again (design sentinel fixed), merge, dev2, choose, plan
cd /path/to/work/b54
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
until grep -q "^exit" logs/dev1.log; do sleep 20; done
B54_ARMS=T-rep bash b54_box.sh x 96 dev1 b54_out/dev_stores.json.gz b54_out/b54_dev1_trep.json --workers 96 > logs/dev1_trep.log 2>&1
$PY - <<'PYEOF'
import json
a = [r for r in json.load(open("b54_out/b54_dev1.json")) if r["arm"] != "T-rep"]
a += json.load(open("b54_out/b54_dev1_trep.json"))
json.dump(a, open("b54_out/b54_dev1m.json", "w"))
print("merged", len(a))
PYEOF
nice -n 10 env B54_CAL=b54_out/b54_channel.npz OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY b54_mlc.py dev2 b54_out/dev_stores.json.gz b54_out/b54_dev1m.json b54_out/b54_dev2.json --workers 100 > logs/dev2.log 2>&1
bash b54_box.sh x 1 choose b54_out/b54_dev1m.json,b54_out/b54_dev2.json b54_out/b54_choice.json > logs/choose.log 2>&1
bash b54_box.sh x 1 plan b54_out/test_stores.json.gz b54_out/b54_choice.json b54_out/b54_plan_test.json >> logs/choose.log 2>&1
echo chain-done >> logs/choose.log
