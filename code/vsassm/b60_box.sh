#!/bin/bash
# B60 on the A800 box (prereg/B60_corrected_lam_prereg.md, 'Compute'): fresh directory ~/b60_run, code copied from the
# repository (md5 listed in b60_run/code_md5.txt), panel codes copied from ~/b59_run's B49 copies after an md5 check
# against the repository's. One process per job, OMP / BLAS threads 1, nice 5, at most 48 workers.
# usage (on the box):  bash b60_box.sh setup
#                      bash b60_box.sh run PLAN.jsonl WORKERS
set -u
W=$HOME/b60_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
case $1 in
setup)
  mkdir -p $W/b49_out/b32_data $W/b60_out/logs $W/results
  S=$HOME/b59_run
  declare -A MD5=([emb_qwen8_mean_L9_f16.npy]=1e4f5a91b57bd8987c899b770ff0c915 [emb_e5_base.npy]=836cff40465c5aa326052b7c36ef344d
    [emb_bge_small.npy]=5cda7b452f00669817796baa13d4cebe [emb_minilm.npy]=9d2d9bcccee31d61bc05ea306d255c54
    [emb_b46_granite33i_mean_L30_f16.npy]=c474cf933dc90f0d0f3eaff1b95e26a3
    [emb_llama31i_mean_L24_f16.npy]=a55892a0ae069b160cb2dfaedce914a4 [emb_mistral7i_eol_L24_f16.npy]=52207e3edaddb67145252805c30954be
    [entities.json]=f369b096f9edd953214193bc97c4ee32 [emb_strings.json]=1db6556d2d3a0b1d8d904d7cf7961b1e)
  for f in "${!MD5[@]}"; do
    dst=$W/b49_out/b32_data/$f
    if [ ! -f $dst ]; then
      src=$(find $S -name "$f" 2>/dev/null | while read c; do [ "$(md5sum < "$c" | cut -c1-32)" = "${MD5[$f]}" ] && echo "$c" && break; done)
      [ -n "$src" ] && cp "$src" $dst
    fi
    if [ -f $dst ] && [ "$(md5sum < $dst | cut -c1-32)" = "${MD5[$f]}" ]; then echo "ok $f"; else echo "MISSING $f"; fi
  done
  (cd $W && md5sum b32_common.py b49_common.py b60_run.py > code_md5.txt; cat code_md5.txt) ;;
run)
  cd $W || exit 1
  B60_DATA=$W/b49_out B60_OUT=$W/b60_out OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup nice -n 5 $PY b60_run.py runplan b60_out/$2 --workers $3 > b60_out/logs/run_${2%.jsonl}.log 2>&1 &
  echo "started $2 pid $!" ;;
esac
