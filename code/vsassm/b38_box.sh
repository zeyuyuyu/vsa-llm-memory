#!/bin/bash
# B38 stages on the A800 box's CPUs (no GPU). On the box, in /path/to/work/b38:
#   nohup bash b38_box.sh STAGE [WORKERS] > logs/STAGE.log 2>&1 < /dev/null &
cd /path/to/work/b38
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${2:-120}
O=b38_out
case "$1" in
  dev)        $PY b38_eq.py dev $O/dev_stores.json.gz $O/b38_dev.json --workers $W ;;
  dev-cbx)    $PY b38_eq.py dev $O/dev_stores.json.gz $O/b38_dev_cbx.json V-cb --designs 1,2 --workers $W ;;
  dev-ecc)    $PY b38_eq.py dev $O/dev_stores.json.gz $O/b38_dev_ecc.json E-BCH,E-CAM --workers $W ;;
  choose)     $PY b38_design.py merge $O/b38_dev.json,$O/b38_dev_cbx.json,$O/b38_dev_ecc.json $O/b38_dev_merged.json && \
              $PY b38_design.py choose $O/b38_dev_merged.json $O/b38_choice.json ;;
  devfull)    $PY b38_design.py plan $O/dev_stores.json.gz $O/b38_choice.json $O/b38_plan_devfull.json && \
              $PY b38_eq.py test $O/dev_stores.json.gz $O/b38_plan_devfull.json $O/b38_devfull.json --workers $W ;;
  devfull-x)  $PY b38_design.py extend $O/dev_stores.json.gz $O/b38_choice.json $O/b38_devfull.json $O/b38_plan_devfull_x1.json && \
              $PY b38_eq.py test $O/dev_stores.json.gz $O/b38_plan_devfull_x1.json $O/b38_devfull_x1.json --workers $W ;;
  test)       $PY b38_design.py plan $O/test_stores.json.gz $O/b38_choice.json $O/b38_plan_test.json && \
              $PY b38_eq.py test $O/test_stores.json.gz $O/b38_plan_test.json $O/b38_test.json --workers $W ;;
  test-x)     # extension pass N (3rd argument): TESTS = the comma list of finished test files
              $PY b38_design.py extend $O/test_stores.json.gz $O/b38_choice.json $4 $O/b38_plan_test_x$3.json && \
              $PY b38_eq.py test $O/test_stores.json.gz $O/b38_plan_test_x$3.json $O/b38_test_x$3.json --workers $W ;;
  extend-loop)  # STORES (dev|test) FILES (comma list of finished result files) TAG: repeated 4-step extensions
              S=$O/$3_stores.json.gz; F=$4; n=1
              while true; do
                $PY b38_design.py extend $S $O/b38_choice.json $F $O/b38_plan_$5_x$n.json | tee -a logs/extend_$5.log
                if grep -q '"jobs": \[\]' $O/b38_plan_$5_x$n.json; then break; fi
                $PY b38_eq.py test $S $O/b38_plan_$5_x$n.json $O/b38_$5_x$n.json --workers $W
                F=$F,$O/b38_$5_x$n.json; n=$((n+1))
                [ $n -gt 6 ] && break
              done
              echo "result files: $F" ;;
  check-closed)   $PY b38_check.py closed $O/test_stores.json.gz $3 $O/b38_check_closed.json --workers $W ;;
  check-decoders) $PY b38_check.py decoders $O/test_stores.json.gz $3 $O/b38_check_decoders.txt --workers $W ;;
  check-device)   $PY b38_check.py device $O/test_stores.json.gz $3 $O/b38_check_device.json --workers $W ;;
esac
echo "exit $? $(date)"
