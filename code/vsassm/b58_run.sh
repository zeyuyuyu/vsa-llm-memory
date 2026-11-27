#!/bin/bash
# B58 test phase driver (local CPU).
#   bash b58_run.sh test iid W   |   bash b58_run.sh test dmg W   |   bash b58_run.sh rescore - W
# Each test run: plan (window = dev N* +- 4) -> run -> B38's extension rule repeated until nothing is added (<= 6 passes).
cd /path/to/vsassm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${3:-6}
ST=b34_out/test_stores.json.gz
case "$1" in
  test)
    S=$2; FLAG=$([ "$S" = "iid" ] && echo --iid-only || echo --dmg-only)
    python3 b58_store.py plan results/b58_choice.json results/b58_plan_test_$S.json $FLAG || exit 1
    python3 b58_store.py run $ST results/b58_plan_test_$S.json results/b58_test_${S}1.json --workers $W || exit 1
    files=results/b58_test_${S}1.json
    for p in 2 3 4 5 6 7; do
      python3 b58_store.py extend results/b58_choice.json $files results/b58_plan_test_${S}_x$p.json $FLAG > logs/b58/extend_${S}_$p.txt
      n=$(tail -n 1 logs/b58/extend_${S}_$p.txt | awk '{print $1}')
      echo "extension pass $p: $n additions"
      [ "$n" = "0" ] && break
      python3 b58_store.py run $ST results/b58_plan_test_${S}_x$p.json results/b58_test_${S}$p.json --workers $W || exit 1
      files=$files,results/b58_test_${S}$p.json
    done
    echo "$files" > results/b58_test_files_$S.txt ;;
  rescore)
    python3 b58_store.py rescore $ST results/b58_plan_rescore.json results/b58_ldpc_rescore1.json --workers $W || exit 1
    files=results/b58_ldpc_rescore1.json
    for p in 2 3 4 5 6 7; do
      python3 b58_store.py rescore_extend $files results/b58_plan_rescore_x$p.json > logs/b58/rescore_extend_$p.txt
      n=$(tail -n 1 logs/b58/rescore_extend_$p.txt | awk '{print $1}')
      echo "rescore extension pass $p: $n additions"
      [ "$n" = "0" ] && break
      python3 b58_store.py rescore $ST results/b58_plan_rescore_x$p.json results/b58_ldpc_rescore$p.json --workers $W || exit 1
      files=$files,results/b58_ldpc_rescore$p.json
    done
    echo "$files" > results/b58_rescore_files.txt ;;
esac
echo "exit $? $(date)"
