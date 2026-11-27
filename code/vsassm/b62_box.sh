#!/bin/bash
# B62 on the A800 box (prereg/B62_online_ls_write_prereg.md, 'Compute'): fresh directory ~/b62_run, code copied from the
# repository (md5 listed in b62_run/code_md5.txt), panel codes copied from ~/b60_run's B49 copies after an md5 check
# (the same md5 table as b60_box.sh). One process per job, OMP / BLAS threads 1, nice 5, at most 48 workers.
# usage (on the box):  bash b62_box.sh setup
#                      bash b62_box.sh trainnorm
#                      bash b62_box.sh run PLAN.jsonl WORKERS
set -u
W=$HOME/b62_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
ENVS="B60_DATA=$W/b49_out B62_OUT=$W/b62_out OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
case $1 in
setup)
  mkdir -p $W/b49_out/b32_data $W/b62_out/logs $W/results
  S=$HOME/b60_run/b49_out/b32_data
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
  (cd $W && md5sum b32_common.py b32_run.py b49_common.py b60_run.py b62_run.py > code_md5.txt; cat code_md5.txt) ;;
trainnorm)
  cd $W || exit 1
  for e in qwen8 b46 llama mistral e5 bge minilm; do
    env $ENVS nohup nice -n 5 $PY b62_run.py trainnorm $e > b62_out/logs/trainnorm_$e.log 2>&1 &
  done
  echo "trainnorm started" ;;
timing)
  cd $W || exit 1
  env $ENVS nohup nice -n 5 $PY b62_run.py timing $2 $3 $4 > b62_out/logs/timing_$2_$3_$4.log 2>&1 &
  echo "timing started pid $!" ;;
run)
  cd $W || exit 1
  env $ENVS nohup nice -n 5 $PY b62_run.py runplan b62_out/$2 --workers $3 > b62_out/logs/run_${2%.jsonl}.log 2>&1 &
  echo "started $2 pid $!" ;;
esac
