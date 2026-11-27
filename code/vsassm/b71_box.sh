#!/bin/bash
# B71 on the A800 box (prereg/B71_hebbian_remedies_prereg.md, 'Compute'): fresh directory ~/b71_run, code copied from the
# repository (md5 in ~/b71_run/code_md5.txt), panel codes copied from ~/b66_run's B49 copies after an md5 check (the md5
# table of b60_box.sh / b62_box.sh / b66_box.sh / b70_box.sh). One process per job, OMP / BLAS threads 1, nice 5, at most
# 48 workers. Plans are written locally (b71_run.py plan) and copied in; the committed choice files go to results/.
# usage (on the box): bash b71_box.sh setup
#                     bash b71_box.sh run PLAN.jsonl WORKERS
set -u
W=$HOME/b71_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export B60_DATA=$W/b49_out B71_OUT=$W/b71_out B71_RES=$W/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case $1 in
setup)
  mkdir -p $W/b49_out/b32_data $W/b71_out/logs $W/results
  S=$HOME/b66_run/b49_out/b32_data
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
  (cd $W && md5sum b32_common.py b32_run.py b36_common.py b49_common.py b53_run.py b56_run.py b60_run.py b62_run.py \
     b66_lsfc.py b71_run.py > code_md5.txt; cat code_md5.txt) ;;
run)
  cd $W || exit 1
  nohup nice -n 5 $PY b71_run.py runplan $W/b71_out/$2 --workers $3 > $W/b71_out/logs/run_${2%.jsonl}.log 2>&1 &
  echo "started $2 pid $!" ;;
*) echo "unknown $1" ;;
esac
