#!/bin/bash
# B54 test on the box: the registered plan, then 4-step extensions until every (arm, b, condition) is bracketed.
cd /path/to/work/b54
W=${1:-100}
bash b54_box.sh test $W > logs/test.log 2>&1
F=b54_out/b54_test.json; n=1
while [ $n -le 6 ]; do
  bash b54_box.sh x 1 extend b54_out/test_stores.json.gz b54_out/b54_choice.json $F b54_out/b54_plan_test_x$n.json > logs/extend_$n.log 2>&1
  if grep -q "extension: 0 " logs/extend_$n.log; then break; fi
  bash b54_box.sh x $W run b54_out/test_stores.json.gz b54_out/b54_plan_test_x$n.json b54_out/b54_test_x$n.json --workers $W > logs/test_x$n.log 2>&1
  F=$F,b54_out/b54_test_x$n.json; n=$((n+1))
done
echo "result files: $F" > logs/test_files.txt
echo chain-test-done >> logs/test_files.txt
