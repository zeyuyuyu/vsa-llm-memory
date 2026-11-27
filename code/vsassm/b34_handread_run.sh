#!/bin/bash
# B34: run every hand-reading case of b34_out/handread_cases.txt (ARM|K|STORE), 30 days, 3 failures each, in parallel
cd /path/to/work/b34
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p b34_out/handread
i=0
while IFS='|' read -r arm k sid; do
  i=$((i+1))
  $PY b34_check.py handread b34_out/test_stores.json.gz b34_out/b34_choice.json "$sid" "$arm" "$k" "30 d" 3 > b34_out/handread/$(printf %02d $i).txt 2>&1 &
done < b34_out/handread_cases.txt
wait
cat b34_out/handread/*.txt > b34_out/handread_all.txt
echo "exit $(date)"
