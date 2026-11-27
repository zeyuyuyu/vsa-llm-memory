#!/bin/bash
# B66 on the A800 box (prereg/B66_reconciliation_prereg.md, 'Compute'): fresh directory ~/b66_run, code copied from the
# repository (md5 in ~/b66_run/code_md5.txt), panel codes copied from ~/b62_run's B49 copies after an md5 check (the
# md5 table of b60_box.sh / b62_box.sh). One process per job, OMP / BLAS threads 1, nice 5, at most 48 workers.
# usage (on the box): bash b66_box.sh setup
#                     bash b66_box.sh sdmtdev WORKERS          # SDMTm dev, maps 1-3, all 7 encoders
#                     bash b66_box.sh vsardev WORKERS          # VSARm dev, maps 0-3, all 7 encoders, 6 parts each
#                     bash b66_box.sh sdmttest WORKERS         # after the committed choices are copied to results/
#                     bash b66_box.sh vsartest WORKERS
set -u
W=$HOME/b66_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export B64_DATA=$W/b49_out B66_OUT=$W/b66_out B66_RES=$W/results OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENCS="qwen8 b46 llama mistral e5 bge minilm"
case $1 in
setup)
  mkdir -p $W/b49_out/b32_data $W/b66_out/logs $W/results
  S=$HOME/b62_run/b49_out/b32_data
  declare -A MD5=([emb_qwen8_mean_L9_f16.npy]=1e4f5a91b57bd8987c899b770ff0c915 [emb_e5_base.npy]=836cff40465c5aa326052b7c36ef344d
    [emb_bge_small.npy]=5cda7b452f00669817796baa13d4cebe [emb_minilm.npy]=9d2d9bcccee31d61bc05ea306d255c54
    [emb_b46_granite33i_mean_L30_f16.npy]=c474cf933dc90f0d0f3eaff1b95e26a3
    [emb_llama31i_mean_L24_f16.npy]=a55892a0ae069b160cb2dfaedce914a4 [emb_mistral7i_eol_L24_f16.npy]=52207e3edaddb67145252805c30954be
    [entities.json]=f369b096f9edd953214193bc97c4ee32 [emb_strings.json]=1db6556d2d3a0b1d8d904d7cf7961b1e)
  for f in "${!MD5[@]}"; do
    dst=$W/b49_out/b32_data/$f
    [ -f $dst ] || cp $S/$f $dst
    if [ "$(md5sum < $dst | cut -c1-32)" = "${MD5[$f]}" ]; then echo "ok $f"; else echo "MISSING $f"; fi
  done
  (cd $W && md5sum b32_common.py b32_run.py b36_common.py b49_common.py b64_run.py b66_run.py > code_md5.txt; cat code_md5.txt) ;;
sdmtdev)
  cd $W || exit 1
  for m in 1 2 3; do
    $PY b66_run.py plan sdmtdev $m
    nohup nice -n 5 $PY b66_run.py runb64 $W/b66_out/sdmtm$m/plan_dev.jsonl --out $W/b66_out/sdmtm$m --map $m --workers $2 \
      > $W/b66_out/logs/sdmtdev_m$m.log 2>&1 &
  done ;;
sdmttest)
  cd $W || exit 1
  for m in 1 2 3; do
    $PY b66_run.py plan sdmttest $m
    nohup nice -n 5 $PY b66_run.py runb64 $W/b66_out/sdmtm$m/plan_test.jsonl --out $W/b66_out/sdmtm$m --map $m --workers $2 \
      > $W/b66_out/logs/sdmttest_m$m.log 2>&1 &
  done ;;
vsardev)
  cd $W || exit 1
  for e in $ENCS; do for m in 0 1 2 3; do for p in center:lin center:sgn center:sgn3 whiten:lin whiten:sgn whiten:sgn3; do
    echo "$e $m $p"; done; done; done > $W/b66_out/vsardev_jobs.txt
  nohup bash -c "cat $W/b66_out/vsardev_jobs.txt | xargs -P $2 -L 1 bash -c 'nice -n 5 $PY b66_run.py vsardev \$0 \$1 \$2 > $W/b66_out/logs/vsardev_\$0_m\$1_\${2/:/_}.log 2>&1'" \
    > $W/b66_out/logs/vsardev_all.log 2>&1 & ;;
vsartest)
  cd $W || exit 1
  for e in $ENCS; do for m in 1 2 3; do echo "$e $m"; done; done > $W/b66_out/vsartest_jobs.txt
  nohup bash -c "cat $W/b66_out/vsartest_jobs.txt | xargs -P $2 -L 1 bash -c 'nice -n 5 $PY b66_run.py vsartest \$0 \$1 > $W/b66_out/logs/vsartest_\$0_m\$1.log 2>&1'" \
    > $W/b66_out/logs/vsartest_all.log 2>&1 & ;;
*) echo "unknown $1" ;;
esac
